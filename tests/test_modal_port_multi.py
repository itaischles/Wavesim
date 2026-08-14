"""Regression gate for **co-planar** :class:`~wavesim.sources.ModalPort`\\ s — two
or more modal ports terminating the same domain face, which is what a
multi-conductor S-parameter run needs.

Each port *assigns* its impedance sheet onto the ghost tangential H plane (it has
to: the sheet replaces what ``update_H`` left there rather than adding to it), so
before the fix the last port in ``Simulation.boundaries`` erased every earlier
one. Co-planar modes generally span the same cells, so this was not a partial
degradation but a total suppression — and a silent one, because the suppressed
port kept recording a plausible ``V(t)``/``I(t)`` off planes nobody clobbers. It
also lost its *termination*, not just its drive, so its mode reflected off the
face instead of being absorbed.

What the tests lock in:

* **An inert peer cannot kill a live port.** A driven port beside an
  ``amplitude=0`` one on the same face launches what it launches alone. This is
  the minimal reproducer and it failed hard.
* **Order independence.** Reversing ``boundaries`` changes nothing but round-off.
* **Superposition.** Two ports driven ±1 V simultaneously both launch ≈1 V.
* **N = 1 is bit-identical to the pre-fix write.** The single-port path is the
  main regression risk, so it is gated against a replica of the old code with
  ``array_equal``, not ``allclose``.

The cross-coupling of the *modal basis* is deliberately not tested here: the
conductor-basis modes ``solve_tem_modes`` returns are not mutually orthogonal, so
each port's ``V̄`` picks up some of the other mode. That is an accuracy question
about the basis, separate from the sheet arithmetic these tests cover.
"""
import numpy as np
import pytest

import wavesim as ws
from wavesim.mode_solver import solve_tem_modes

BACKEND = 'numba'

NX, DX, NZ, DZ = 36, 0.5e-3, 40, 2e-3
K_DRIVE, K_ABSORB = 1, NZ - 1


# --------------------------------------------------------------------------- #
# Geometry: two PEC traces in a PEC-walled box → two conductor-basis TEM modes.
# --------------------------------------------------------------------------- #

def _two_traces():
    """Two parallel PEC strips running the length of the box (cf. ``_stripline``
    in ``test_modal_port.py``, doubled and moved off centre)."""
    g = ws.create_grid(Nx=NX, Ny=NX, Nz=NZ, dx=DX, dy=DX, dz=DZ)
    ws.set_vacuum(g)
    c = 0.5 * NX * DX
    w = 0.15 * NX * DX
    for cx in (c - 0.22 * NX * DX, c + 0.22 * NX * DX):
        ws.set_box(g, cx - 0.5 * w, cx + 0.5 * w, c - DX, c + DX,
                   0.0, NZ * DZ, eps_r=1.0, pec=True)
    return g


def _modes(g, k):
    return solve_tem_modes(g, normal='z', position=k * g.dz, compute_params=True)


def _wave(port):
    """``a = (V + Z_ref·I)/2`` — the wave the port launched into the domain."""
    v = np.array(port.voltages)
    i = np.array(port.currents)
    return 0.5 * (v + port.reference_impedance * i)


def _run(amps, *, n_steps=260, reverse=False, absorb_far=True, cls=ws.ModalPort):
    """Drive the z0 face with one port per conductor mode; return their ``a(t)``.

    ``amps`` is one launch amplitude per mode, in ``solve_tem_modes`` order. The
    far face carries a matched port per mode too (``absorb_far``), so the only
    thing on either plane is modal sheets.
    """
    g = _two_traces()
    wf = ws.GaussianPulse.for_fmax(6e9)
    sim = ws.Simulation(g, cpml=None, pec_faces=('x0', 'x1', 'y0', 'y1'),
                        backend=BACKEND)
    ports = [cls(m, amplitude=a, waveform=wf if a else None)
             for m, a in zip(_modes(g, K_DRIVE), amps)]
    far = [cls(m, amplitude=0.0) for m in _modes(g, K_ABSORB)] if absorb_far else []
    order = list(ports) + far
    if reverse:
        order.reverse()
    for p in order:
        sim.add_boundary(p)
    for _ in range(n_steps):
        sim.step()
    return g, ports, [_wave(p) for p in ports]


# --------------------------------------------------------------------------- #
# The bug: an inert co-planar port suppressed a live one entirely.
# --------------------------------------------------------------------------- #

def test_two_modes_with_distinct_conductor_ids():
    """The fixture really is multi-conductor — one mode per trace, distinct
    ``conductor_id``. Everything below is meaningless otherwise."""
    modes = _modes(_two_traces(), K_DRIVE)
    assert len(modes) == 2
    assert len({m.conductor_id for m in modes}) == 2


@pytest.mark.slow
def test_inert_coplanar_port_does_not_suppress_a_live_one():
    """Port A at 1 V beside an ``amplitude=0`` port B on the same face launches
    the same wave as A alone.

    Pre-fix, whichever of the two came last in ``boundaries`` was the only one on
    the plane: with B last, A's launched ``a`` collapsed from ~1 V to ~0.03 V.
    """
    _, _, (a_pair, _) = _run([1.0, 0.0])
    _, _, (a_solo,) = _run_single(0)
    assert np.max(np.abs(a_solo)) == pytest.approx(1.0, abs=0.06), (
        f"solo launch {np.max(np.abs(a_solo)):.3f} V is not the ~1 V baseline")
    assert np.max(np.abs(a_pair)) == pytest.approx(
        np.max(np.abs(a_solo)), rel=0.02), (
        f"inert peer changed the launch: {np.max(np.abs(a_pair)):.4f} V "
        f"vs {np.max(np.abs(a_solo)):.4f} V alone")


def _run_single(idx, *, n_steps=260, cls=ws.ModalPort):
    """Same run as :func:`_run` but with *only* mode ``idx``'s ports present."""
    g = _two_traces()
    wf = ws.GaussianPulse.for_fmax(6e9)
    sim = ws.Simulation(g, cpml=None, pec_faces=('x0', 'x1', 'y0', 'y1'),
                        backend=BACKEND)
    port = cls(_modes(g, K_DRIVE)[idx], amplitude=1.0, waveform=wf)
    sim.add_boundary(port)
    sim.add_boundary(cls(_modes(g, K_ABSORB)[idx], amplitude=0.0))
    for _ in range(n_steps):
        sim.step()
    return g, [port], [_wave(port)]


@pytest.mark.slow
def test_coplanar_ports_are_order_independent():
    """Reversing ``boundaries`` is a round-off-level change, not a physical one.

    Pre-fix it decided *which port existed*: reversing the list swapped which of
    the two launched.
    """
    _, _, fwd = _run([1.0, -1.0])
    _, _, rev = _run([1.0, -1.0], reverse=True)
    for k, (f, r) in enumerate(zip(fwd, rev)):
        scale = max(np.max(np.abs(f)), 1e-30)
        assert np.max(np.abs(f - r)) / scale < 1e-9, (
            f"port {k} depends on boundary order (max diff "
            f"{np.max(np.abs(f - r)):.3g} on a {scale:.3g} V wave)")


@pytest.mark.slow
def test_coplanar_ports_superpose():
    """Both ports driven at once (+1 V and −1 V) each launch their own ≈1 V —
    the plane carries ``Σ_m s_m·(V̄_m − 2a_m)·ĥ_m``, not the last term written.

    The tolerance is 0.15 V rather than the 0.06 V of a solo launch because of
    the **modal basis**, not the summation: an undriven port on this fixture
    already reads ``max|a| = 0.089 V`` of its neighbour's wave (the
    conductor-basis modes are not mutually orthogonal), and that term adds on the
    differential drive and subtracts on the common-mode one — measured 1.089 and
    1.086 V for ``[1, −1]``, 0.907 and 0.910 V for ``[1, 1]``, against 1.006 V
    solo. Suppression, the bug under test, is a factor of ~11, not 9%.
    """
    for amps in ([1.0, -1.0], [1.0, 1.0]):
        _, _, (a0, a1) = _run(amps)
        for k, a in enumerate((a0, a1)):
            assert np.max(np.abs(a)) == pytest.approx(1.0, abs=0.15), (
                f"port {k} launched {np.max(np.abs(a)):.3f} V on drive {amps}")


# --------------------------------------------------------------------------- #
# The regression risk: one port must still write exactly what it always wrote.
# --------------------------------------------------------------------------- #

class _LegacyModalPort(ws.ModalPort):
    """``ModalPort`` with the pre-fix ghost write: plain per-port assignment, no
    co-planar accumulator. Wrong for N > 1 — that is the bug — but it is the
    reference N = 1 has to reproduce bit for bit."""

    def _write_ghost_h(self, grid, t, amp):
        for comp, (ii, jj, kk, vals) in self._h.items():
            getattr(grid, comp)[ii, jj, kk] = amp * vals


@pytest.mark.slow
def test_single_port_is_bit_identical_to_the_legacy_write():
    """One port per plane must produce the *exact* same arrays as before the fix
    — same cells written, same values, not merely ``allclose``."""
    g_new, ports_new, (a_new,) = _run_single(0, n_steps=120)
    g_old, ports_old, (a_old,) = _run_single(0, n_steps=120,
                                             cls=_LegacyModalPort)
    for comp in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        assert np.array_equal(getattr(g_new, comp), getattr(g_old, comp)), (
            f"single-port run diverged from the legacy write in {comp}")
    assert np.array_equal(a_new, a_old)
    assert np.array_equal(np.array(ports_new[0].voltages),
                          np.array(ports_old[0].voltages))
