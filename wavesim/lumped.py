"""
Discrete companion models for lumped R / L / C elements.

This module holds the *circuit* half of :class:`~wavesim.sources.LineSource`:
given a network of R, L and C branches it produces, once per timestep, the
Thevenin pair the FDTD element needs

    V = −Z_eq · I + V_hist

where ``I`` is the port current (positive **out of** the ``+`` terminal, the
sign convention of the surrounding FDTD port) and ``V`` the port voltage. There
is no field, no grid and no geometry here, so the algebra can be exercised on
its own.

Why a companion model
---------------------
The FDTD element solves its circuit law semi-implicitly (Piket-May): the
impressed current ``I`` is a sample at step ``n+½`` and the port voltage it is
balanced against, ``V = (Vⁿ + Vⁿ⁺¹)/2``, is second-order accurate at ``n+½``
too. Both quantities therefore live at the *same* instant, which is exactly the
footing a trapezoidal integrator needs: each reactive branch is turned into a
resistance plus a history source built from its own previous ``(I, V)`` sample,

    L:  V = L·dI/dt   →   Z = 2L/dt,    V_hist = Z·I_prev − V_prev
    C:  I = C·dV/dt   →   Z = dt/(2C),  V_hist = V_prev − Z·I_prev
    R:                    Z = R,        V_hist = 0

(the L and C forms carry the port sign convention: the current *into* the
branch's ``+`` terminal is ``−I``). The trapezoidal rule is A-stable and every
``Z`` above is positive and finite, so no choice of L or C imposes a timestep
limit of its own — the element cannot destabilise a run that was stable
without it.

Combining branches is then ordinary circuit algebra, done on the pair rather
than on a single impedance: in series the pairs add, in parallel the
admittances add and the history sources combine as a current division.
"""

from __future__ import annotations

import math
from typing import List, Tuple

__all__ = ["LumpedNetwork"]


_TOPOLOGIES = ("series", "parallel")


class LumpedNetwork:
    """One- to three-branch R/L/C network with per-step trapezoidal state.

    Parameters
    ----------
    resistance, inductance, capacitance : float, optional
        Branch values in ohms / henries / farads (each > 0). Omitted branches
        are absent, not zeroed — a missing R is no resistor at all, which in
        series means a short and in parallel an open. At least one must be
        given.
    topology : {'series', 'parallel'}
        How the given branches are wired *between the element's two
        terminals*. With a single branch the two are identical.

    Notes
    -----
    The instance carries the integrator state (each branch's previous current
    and voltage sample), so one network belongs to one element and one run;
    :meth:`reset` returns it to the quiescent state.
    """

    __slots__ = ("topology", "_kinds", "_values", "_i_prev", "_v_prev")

    def __init__(self, *,
                 resistance: float | None = None,
                 inductance: float | None = None,
                 capacitance: float | None = None,
                 topology: str = "series") -> None:
        if topology not in _TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {_TOPOLOGIES}, got {topology!r}.")
        kinds: List[str] = []
        values: List[float] = []
        for kind, value, unit in (("R", resistance, "ohms"),
                                  ("L", inductance, "henries"),
                                  ("C", capacitance, "farads")):
            if value is None:
                continue
            value = float(value)
            if not value > 0.0:
                raise ValueError(
                    f"{kind} must be a positive value in {unit}, got {value!r}.")
            kinds.append(kind)
            values.append(value)
        if not kinds:
            raise ValueError(
                "LumpedNetwork needs at least one of resistance=, "
                "inductance= or capacitance=.")
        self.topology = topology
        self._kinds: Tuple[str, ...] = tuple(kinds)
        self._values: Tuple[float, ...] = tuple(values)
        self._i_prev: List[float] = [0.0] * len(kinds)
        self._v_prev: List[float] = [0.0] * len(kinds)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def branches(self) -> Tuple[Tuple[str, float], ...]:
        """The branches as ``(('R', 50.0), ('C', 1e-13), ...)``."""
        return tuple(zip(self._kinds, self._values))

    def reset(self) -> None:
        """Return every branch to the quiescent (zero current, zero voltage)
        state, as before a run."""
        self._i_prev = [0.0] * len(self._kinds)
        self._v_prev = [0.0] * len(self._kinds)

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v:g}" for k, v in self.branches)
        return f"LumpedNetwork({parts}, topology={self.topology!r})"

    # ------------------------------------------------------------------ #
    # Companion model
    # ------------------------------------------------------------------ #
    def _branch_companions(self, dt: float) -> List[Tuple[float, float]]:
        """Each branch's ``(Z, V_hist)`` for a step of length ``dt``.

        Depends only on the stored state, so calling this before and after the
        solve (as :meth:`companion` and :meth:`update` do) returns the same
        numbers — the state is advanced only at the end of :meth:`update`.
        """
        out: List[Tuple[float, float]] = []
        for kind, value, i_prev, v_prev in zip(
                self._kinds, self._values, self._i_prev, self._v_prev):
            if kind == "R":
                out.append((value, 0.0))
            elif kind == "L":
                z = 2.0 * value / dt
                out.append((z, z * i_prev - v_prev))
            else:                                   # "C"
                z = dt / (2.0 * value)
                out.append((z, v_prev - z * i_prev))
        return out

    def companion(self, dt: float) -> Tuple[float, float]:
        """The whole network's ``(Z_eq, V_hist)`` for this step.

        The element's terminal law is then ``V = −Z_eq·I + V_hist``, which the
        FDTD port solves together with its own ``V = v_mid + (κ/2)·I``.
        """
        parts = self._branch_companions(dt)
        if self.topology == "series" or len(parts) == 1:
            # Same current through every branch: the Thevenin pairs add.
            return (sum(z for z, _ in parts),
                    sum(vh for _, vh in parts))
        # Parallel: same voltage across every branch, currents add. Summing the
        # branch Nortons (I_b = (V_hist_b − V)/Z_b) gives the equivalent below.
        y = sum(1.0 / z for z, _ in parts)
        z_eq = 1.0 / y
        return z_eq, z_eq * sum(vh / z for z, vh in parts)

    def update(self, dt: float, i_elem: float, v_elem: float) -> None:
        """Advance the integrator state with this step's solved sample.

        Parameters
        ----------
        dt : float
            The step just taken — the same ``dt`` handed to :meth:`companion`.
        i_elem : float
            Current through the *element* at ``n+½``, port sign convention.
            This is not always the port current: with a parallel Norton drive
            the source current has to be taken out first (the FDTD element
            does that before calling).
        v_elem : float
            Voltage across the element at ``n+½``, i.e. the solved
            ``(Vⁿ + Vⁿ⁺¹)/2`` minus whatever ideal source sits in series
            with it.

        The pair is distributed over the branches by the topology — one shared
        current in series, one shared voltage in parallel — and each branch
        stores its own ``(I, V)`` for the next step's history term.
        """
        parts = self._branch_companions(dt)
        if self.topology == "series" or len(parts) == 1:
            for n, (z, vh) in enumerate(parts):
                self._i_prev[n] = i_elem
                self._v_prev[n] = -z * i_elem + vh
        else:
            for n, (z, vh) in enumerate(parts):
                self._i_prev[n] = (vh - v_elem) / z
                self._v_prev[n] = v_elem

    # ------------------------------------------------------------------ #
    # Continuum reference (documentation / tests)
    # ------------------------------------------------------------------ #
    def impedance_at(self, frequency: float, dt: float | None = None) -> complex:
        """The network's impedance at ``frequency`` (Hz), in ohms.

        With ``dt=None`` this is the ideal continuum value (``jωL``, ``1/jωC``).
        Given a ``dt`` it is instead the impedance the *discrete* companion
        actually presents: trapezoidal integration maps ``s → j(2/dt)·tan(ωdt/2)``,
        so a reactance is stretched slightly as ω approaches the Nyquist rate.
        Tests compare against this form; it is the exact behaviour of the
        scheme, not an approximation of it.
        """
        w = 2.0 * math.pi * frequency
        s = 1j * w if dt is None else (2.0 / dt) * 1j * math.tan(0.5 * w * dt)
        if s == 0:
            raise ValueError("impedance_at needs a nonzero frequency (a "
                             "capacitor branch is a DC open circuit).")
        zs = []
        for kind, value in self.branches:
            if kind == "R":
                zs.append(complex(value))
            elif kind == "L":
                zs.append(s * value)
            else:
                zs.append(1.0 / (s * value))
        if self.topology == "series" or len(zs) == 1:
            return sum(zs)
        return 1.0 / sum(1.0 / z for z in zs)
