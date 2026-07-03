"""
SPICE co-simulation coupler — drives an ngspice circuit in lockstep with the
FDTD time loop through a single lumped port.

Physics / contract
-------------------
An FDTD lumped port (see :class:`~wavesim.sources.SpicePort`) is, seen from the
circuit, exactly a **Thévenin branch**: a voltage source ``v_mid`` (the
time-centred port voltage read from the fields) behind a series resistance
``κ/2`` (the port's discrete self-coupling; :meth:`LineSource.self_coupling`).
Each FDTD timestep we

1. set that Thévenin source to ``v_mid`` (via ngspice's *external* source
   callback, held constant across the step — a zero-order hold),
2. advance the ngspice transient by exactly one FDTD ``dt``,
3. read the branch current ``i_port`` back and hand it to the FDTD element,
   which injects it into Ampère's law.

If the circuit reduces to a Thévenin ``(Vs, Z)`` this returns
``i = (Vs − v_mid)/(Z + κ/2)`` — identical to the analytic law in
:class:`~wavesim.sources.LineSource`, which is the golden test for this module.

Engine
------
Runtime is **ngspice** via PySpice's ``NgSpiceShared`` (the schematic is authored
elsewhere — e.g. LTspice — and exported as a netlist; LTspice itself cannot
lockstep). We use ngspice's foreground ``step`` command (one transient point at a
time), so there is no background thread to synchronise. The port is spliced into
the user's netlist as an ``external`` voltage source in series with ``Rport=κ/2``
between the two named port nodes; its ``#branch`` current is the port current.

This module imports PySpice at import time, so it is optional: ``import wavesim``
does not require PySpice/ngspice. It is only imported when a :class:`SpicePort`
is first stepped.
"""

from __future__ import annotations

import os
import re
from typing import Sequence

from PySpice.Spice.NgSpice.Shared import NgSpiceShared, NgSpiceCommandError


# Control/analysis cards we strip from the user's netlist — wavesim owns the
# transient. ``.ends`` is deliberately NOT matched (only exact ``.end``).
_STRIP_RE = re.compile(
    r'^\s*\.(tran|ac|dc|op|noise|tf|pz|disto|sens|end|probe|backanno|save|plot|print|step|meas|measure|four)\b',
    re.IGNORECASE,
)
# ``.end`` must not swallow ``.ends``; the alternation above lists ``end`` and
# relies on the trailing \b, so ``.ends`` (word char 's') is left intact.


class _PortNgSpice(NgSpiceShared):
    """NgSpiceShared subclass wired to one :class:`SpiceCoupler`.

    ngspice calls :meth:`get_vsrc_data` whenever it needs the external port
    source's value (possibly several times per point during Newton iterations —
    we always return the held ``v_ext``), and :meth:`send_data` at each accepted
    transient point (we latch the sim time and the port branch current there).
    """

    def __init__(self, coupler: "SpiceCoupler", **kwargs) -> None:
        self._coupler = coupler
        super().__init__(send_data=True, **kwargs)

    def get_vsrc_data(self, voltage, time, node, ngspice_id):   # noqa: D102
        # ``voltage`` is a raw CFFI ``double *``; set element 0, return 0.
        voltage[0] = self._coupler._v_ext
        return 0

    def send_data(self, values, number_of_vectors, ngspice_id):  # noqa: D102
        c = self._coupler
        t = values.get('time')
        if t is not None:
            c._sim_time = t.real
        cur = values.get(c._branch_key)
        if cur is None:
            # Fall back: any single voltage-source branch current.
            for k, v in values.items():
                if k.lower().endswith('#branch'):
                    cur = v
                    if k.lower() == c._branch_key:
                        break
        if cur is not None:
            c._i_branch = cur.real
        return 0


class SpiceCoupler:
    """Lockstep bridge between an FDTD port and an ngspice circuit.

    Parameters
    ----------
    netlist : str
        Path to a SPICE netlist file (authored anywhere; standard primitives).
        Its two ``nodes`` must already exist — the user's circuit connects to
        them; wavesim splices the FDTD Thévenin companion across them.
    nodes : (str, str)
        The port node names ``(plus, minus)``; ``plus`` maps to the FDTD ``+``
        terminal (``p0``).
    kappa : float
        The FDTD port self-coupling κ (ohms); the series companion resistance is
        ``κ/2``.
    dt : float
        FDTD timestep (s). SPICE is locked to this; ngspice's max internal step
        is capped to ``dt``.
    library_path : str, optional
        Full path to the ngspice shared library (``ngspice.dll`` /
        ``libngspice.so``). Sets ``NGSPICE_LIBRARY_PATH`` for PySpice. If omitted,
        PySpice's own search is used (env var or bundled DLL).
    sign : float
        ±1 orientation of the reported branch current relative to the FDTD
        "positive out of p0" convention. Fixed by the golden test.
    uic : bool
        Pass ``uic`` to ``.tran`` (skip the DC operating point). Default False —
        start from the DC solution, which is right when the fields (hence the
        port source) start at zero.
    source_name : str
        Name of the spliced external voltage source (its ``#branch`` current is
        the port current).
    max_substeps : int
        Safety cap on ngspice ``step`` calls per FDTD ``dt`` (guards against a
        mis-wired transient that never advances time).
    """

    def __init__(self, *, netlist: str, nodes: Sequence[str], kappa: float,
                 dt: float, library_path: str | None = None,
                 sign: float = 1.0, uic: bool = False,
                 source_name: str = "vwsport", max_substeps: int = 64) -> None:
        if len(tuple(nodes)) != 2:
            raise ValueError(f"nodes must be (plus, minus); got {nodes!r}")
        self.netlist_path = netlist
        self.plus, self.minus = (str(n) for n in nodes)
        self.kappa = float(kappa)
        self.dt = float(dt)
        self.library_path = library_path
        self.sign = float(sign)
        self.uic = bool(uic)
        self.source_name = source_name
        self.max_substeps = int(max_substeps)

        self._branch_key = f"{source_name.lower()}#branch"
        # State latched by the callbacks.
        self._v_ext = 0.0        # held external-source value for this dt
        self._sim_time = 0.0     # ngspice transient time (from send_data)
        self._i_branch = 0.0     # latest port branch current (from send_data)
        self._ng: _PortNgSpice | None = None
        self._steps = 0          # number of FDTD dt's advanced
        self._started = False    # first advance issues `run`, then `resume`

    # ------------------------------------------------------------------ #
    # Netlist assembly
    # ------------------------------------------------------------------ #
    def build_circuit(self) -> str:
        """Read the user's netlist, strip analysis cards, splice the FDTD port
        Thévenin companion, and append our own ``.tran`` + ``.end``."""
        with open(self.netlist_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read().splitlines()

        body = [ln for ln in raw if not _STRIP_RE.match(ln)]
        # The first non-blank line of a SPICE deck is the title (ignored by the
        # parser). LTspice puts a ``*`` comment there; keep whatever is first,
        # but guarantee a title line exists.
        if not body or not body[0].strip():
            body = ["* wavesim SPICE co-simulation port"] + body

        mid = f"{self.source_name}_mid"
        tstop = self.dt * 1e12   # effectively unbounded; we drive via `step`
        companion = [
            f"* --- wavesim FDTD port (Thevenin companion: v_mid behind kappa/2) ---",
            f"{self.source_name} {self.plus} {mid} dc 0 external",
            f"rwsport {mid} {self.minus} {self.kappa / 2.0:.12g}",
            f".tran {self.dt:.12g} {tstop:.12g} 0 {self.dt:.12g}"
            + (" uic" if self.uic else ""),
            ".end",
        ]
        return "\n".join(body + companion) + "\n"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Load the ngspice shared library and the spliced circuit."""
        if self.library_path:
            # Point PySpice at the given shared library. Set both the env var and
            # the class attribute directly — PySpice caches LIBRARY_PATH lazily,
            # so a runtime env change alone is not reliably picked up.
            os.environ["NGSPICE_LIBRARY_PATH"] = self.library_path
            NgSpiceShared.LIBRARY_PATH = self.library_path
            # PySpice derives SPICE_LIB_DIR (model libs, spinit) from a bundled
            # path that is unset when pointing at an external DLL; supply it from
            # the standard package layout (…/Spice64_dll/dll-vs/ngspice.dll →
            # …/Spice64_dll/share/ngspice) so the load doesn't crash.
            from pathlib import Path
            share = Path(self.library_path).parent.parent / "share" / "ngspice"
            if share.is_dir():
                os.environ.setdefault("SPICE_LIB_DIR", str(share))
        self._ng = _PortNgSpice(self)
        self._ng.load_circuit(self.build_circuit())
        self._sim_time = 0.0
        self._i_branch = 0.0
        self._steps = 0
        self._started = False

    def _cmd(self, command: str) -> None:
        """Issue an ngspice command, swallowing the benign ``NgSpiceCommandError``
        PySpice raises when ngspice prints its pause/breakpoint notice on stderr
        (a genuine stall is caught by the time-advance check in :meth:`advance`)."""
        try:
            self._ng.exec_command(command)
        except NgSpiceCommandError:
            pass

    def advance(self, v_mid: float, dt: float) -> float:
        """Hold the port source at ``v_mid``, advance ngspice one FDTD ``dt``,
        and return the port current (FDTD sign convention).

        Uses a *moving breakpoint*: ``stop when time > k·dt`` then ``run`` (first
        step) / ``resume`` (thereafter), which advances the foreground transient
        to just past ``k·dt`` and pauses — clean lockstep with no background
        thread. (Plain ``step`` restarts the run from t=0 each call.)
        """
        if self._ng is None:
            self.start()
        self._v_ext = float(v_mid)
        self._steps += 1
        target = self._steps * dt
        prev = self._sim_time
        self._cmd(f"stop when time > {target:.12e}")
        self._cmd("resume" if self._started else "run")
        self._started = True
        self._cmd("delete all")            # clear the breakpoint for the next dt
        if self._sim_time <= prev + 0.5 * dt:
            raise RuntimeError(
                f"ngspice transient did not advance to t≈{target:.3e}s "
                f"(sim_time={self._sim_time:.3e}, was {prev:.3e}); the .tran may "
                "have ended (tstop) or failed to converge.")
        return self.sign * self._i_branch

    def close(self) -> None:
        """Tear down the ngspice instance."""
        if self._ng is not None:
            try:
                self._ng.exec_command("bg_halt")
            except Exception:
                pass
            try:
                self._ng.remove_circuit()
            except Exception:
                pass
            self._ng = None
