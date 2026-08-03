"""The ModalPort ghost-H plane must not accumulate a spurious normal E.

A :class:`~wavesim.sources.ModalPort` overwrites the tangential H on its ghost
plane every step, so that one plane runs **open loop**: whatever the Ampère
update deposits on it can never act back on the H that produced it. What Ampère
deposits there is the discrete transverse divergence of the injected ``ê``, which
the mode solver's Laplacian drives to round-off at every node where it *solved*
for ``φ`` — and leaves alone at the nodes where it *pinned* ``φ``, i.e. the nodes
on the conductor, where the residual is the mode's real induced surface charge.

That residual is not conformal-specific: it measures the same either way. The
staircase run never showed it because ``build_pec_edge_masks`` dilates and had
already zeroed every conductor-node edge for an unrelated reason. The conformal
rule — zero an edge iff its own open length is zero — correctly keeps alive an
edge running *along* a grid-aligned conductor surface, and so removed that
accidental guard. Measured on the reference coax it reached 6.9e3 V/m on the
ghost plane against 0.35 V/m one cell in.

Everything here drives :meth:`wavesim.simulation.Simulation.run`. The guard lives
in a hook that only the real time loop calls, and plan R7 is the precedent for
why testing a guard at the wrong entry point proves nothing: check it from
outside the solver, on the path a user actually takes.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim.mode_solver import solve_tem_modes

from conformal_shapes import coax_fractions

A_IN, B_OUT = 3.0e-3, 9.0e-3
CELL = 0.5e-3
N_TR = 36                      # transverse cells: b=9mm lands inside the box
N_AX = 40


def _modulated_gaussian(f0=1e9, n_sigma=3.0):
    """Sine-modulated Gaussian — negligible DC, so a persistent ghost-plane ramp
    cannot be blamed on the drive's time integral (plan R8)."""
    width = n_sigma / (2.0 * np.pi * f0)
    t0 = 4.0 * width

    def wave(t):
        return float(np.exp(-0.5 * ((t - t0) / width) ** 2)
                     * np.sin(2.0 * np.pi * f0 * (t - t0)))
    return wave


def _coax(conformal=True, offset=0.0, threshold=0.4):
    """Air coax on a uniform 0.5 mm mesh.

    ``offset`` shifts the axis off the node ruler. At ``0.0`` the conductor
    surfaces ``r = a`` and ``r = b`` land *exactly* on grid nodes, which is the
    alignment that triggers this bug and is entirely ordinary — round radii, round
    cell sizes. A quarter-cell offset removes it.
    """
    grid = ws.create_grid(Nx=N_TR, Ny=N_TR, Nz=N_AX, dx=CELL, dy=CELL, dz=CELL)
    ws.set_vacuum(grid)
    cx = cy = 0.5 * N_TR * CELL + offset
    r2 = ((grid.xc[:, None, None] - cx) ** 2
          + (grid.yc[None, :, None] - cy) ** 2)
    grid.pec_mask = np.broadcast_to((r2 < A_IN ** 2) | (r2 > B_OUT ** 2),
                                    (grid.Nx, grid.Ny, grid.Nz)).copy()
    if conformal:
        ws.set_material_arrays(
            grid, grid.eps_x, grid.eps_y, grid.eps_z,
            grid.mu_x, grid.mu_y, grid.mu_z,
            conformal_area_threshold=threshold,
            **coax_fractions(grid, cx, cy, A_IN, B_OUT))
    return grid


def _run(grid, steps=400, backend='numba', amplitude=1.0, disarm=False):
    """Drive the real time loop; return ``(launch, absorb)``.

    ``disarm`` clears the guard after setup, which is how the tests below show
    they are actually sensitive to it rather than passing for free.
    """
    k_hi = grid.Nz - 1
    lo = solve_tem_modes(grid, normal='z', position=CELL, compute_params=True)[0]
    hi = solve_tem_modes(grid, normal='z', position=k_hi * CELL,
                         compute_params=True)[0]
    launch = ws.ModalPort(lo, amplitude=amplitude,
                          waveform=_modulated_gaussian() if amplitude else None)
    absorb = ws.ModalPort(hi, amplitude=0.0)
    sim = ws.Simulation(grid, backend=backend,
                        pec_faces=('x0', 'x1', 'y0', 'y1'))
    sim.add_boundary(launch)
    sim.add_boundary(absorb)
    if disarm:
        launch._setup(grid)
        absorb._setup(grid)
        launch._pin = absorb._pin = None
    sim.run(steps)
    return launch, absorb


def _ghost_vs_line(grid, launch, absorb):
    """``(max|Ez| on either ghost plane, max|Ez| on the line between them)``."""
    planes = (launch._h_k, absorb._h_k)
    ghost = max(float(np.abs(grid.Ez[:, :, k]).max()) for k in planes)
    interior = np.ones(grid.Nz, dtype=bool)
    for k in planes:
        interior[k] = False
    line = float(np.abs(grid.Ez[:, :, interior]).max())
    return ghost, line


# ---------------------------------------------------------------------- #
# The bug itself
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_ghost_plane_carries_no_spurious_normal_E(backend):
    """The port planes must not be the hottest ``Ez`` in a TEM run.

    For a TEM mode ``Ez`` is zero in the continuum, so *every* ``Ez`` in this run
    is numerical residue and the only meaningful statement is a comparison. The
    ghost planes used to carry ~2e4x the rest of the domain; they must now carry
    less than it. Asserted for both backends because the guard sits in the
    boundary hook, not in a kernel, and must not depend on which one runs.
    """
    grid = _coax(conformal=True)
    launch, absorb = _run(grid, backend=backend)
    ghost, line = _ghost_vs_line(grid, launch, absorb)
    assert ghost < line, (
        f"ghost planes carry max|Ez| = {ghost:.4g} V/m against {line:.4g} V/m "
        f"on the line between them")
    # Not merely "smaller" — the residual is killed to round-off.
    assert ghost < 1e-6 * line


def test_the_guard_is_what_keeps_it_quiet():
    """Sensitivity check: disarm the guard and the same run blows up.

    Without this the test above could pass because the case is too gentle to
    excite the defect, which is the R7 failure mode in a different costume.
    """
    grid = _coax(conformal=True)
    launch, absorb = _run(grid, disarm=True)
    ghost, line = _ghost_vs_line(grid, launch, absorb)
    assert ghost > 1e3 * line, (
        f"the reference case no longer excites the defect (ghost {ghost:.4g} "
        f"vs line {line:.4g}); it is not testing the guard any more")


def test_it_is_the_sheet_not_the_drive():
    """It survives ``amplitude = 0`` on both ports, so no launch-local fix works.

    The sheet amplitude is ``s·(V̄ − 2a)``; with ``a = 0`` it is still nonzero
    whenever the port sees any voltage, so a pure *absorber* accumulates the same
    residual. Seeded rather than driven, to have a field for the ports to see.

    The comparison is armed-against-disarmed on the *same* seeded run, rather
    than ghost-against-line: seeding a modal profile over a finite span of ``z``
    puts a genuine ``Ez`` transient at the seam, so the line is not a quiet
    reference here and only the guard's own effect is a clean signal.
    """
    def seeded_ghost(disarm):
        grid = _coax(conformal=True)
        mode = solve_tem_modes(grid, normal='z', position=CELL,
                               compute_params=True)[0]
        E, _H = mode._staggered_port_fields(grid)
        mid = grid.Nz // 2
        for comp, prof in E.items():
            getattr(grid, comp)[:, :, mid - 4:mid + 4] += prof[:, :, None]
        ws.apply_pec_mask(grid)
        launch, absorb = _run(grid, amplitude=0.0, disarm=disarm)
        return _ghost_vs_line(grid, launch, absorb)

    ghost_bad, _ = seeded_ghost(disarm=True)
    ghost_ok, line = seeded_ghost(disarm=False)
    assert ghost_bad > 1e3 * ghost_ok, (
        f"with no drive at all the guard should still be doing the work: "
        f"disarmed {ghost_bad:.4g} V/m vs armed {ghost_ok:.4g} V/m")
    assert ghost_ok < line


# ---------------------------------------------------------------------- #
# What the guard must NOT do
# ---------------------------------------------------------------------- #

def test_guard_is_absent_on_a_staircase_grid():
    """V2: nothing on the non-conformal path may change, so the guard is not
    built at all there — the hook returns before touching a field."""
    grid = _coax(conformal=False)
    launch, absorb = _run(grid, steps=60)
    assert launch._pin is None and absorb._pin is None
    before = grid.Ez.copy()
    launch.apply_post_E(grid, 0.0)
    absorb.apply_post_E(grid, 0.0)
    assert np.array_equal(grid.Ez, before)


def test_guard_zeroes_nothing_live_when_the_geometry_is_off_lattice():
    """The guard must be a no-op unless a conductor surface lands on the nodes.

    Shifting the axis a quarter cell leaves the cut cells every bit as small —
    this is not a small-cut phenomenon — but no node lies on the metal, so no
    *live* edge is pinned. If this ever starts zeroing live edges, the guard has
    become an over-zeroing dilation and is eating real conformal field.
    """
    grid = _coax(conformal=True, offset=0.25 * CELL)
    launch, _absorb = _run(grid, steps=60)
    comp, ii, jj, kk = launch._pin
    assert comp == 'Ez'
    live = grid.pec_edge_open_z[ii, jj, kk] > 0.0
    assert not np.any(live), (
        f"{int(np.count_nonzero(live))} live Ez edges pinned on an off-lattice "
        f"geometry — the guard is over-zeroing")


def test_the_line_is_untouched_on_the_reference_coax():
    """The guard changes the two ghost planes and nothing else.

    On this cross-section the spurious ``Ez`` cannot escape its plane — the Hx/Hy
    faces it feeds are fully covered, so their conformal ``1/A_open`` guard is
    zero and freezes them. The transverse field, which is the whole of a TEM
    wave, must therefore come out *bit-identical* with and without the guard.
    (That containment is a property of this geometry, not of the defect: on a
    grid-aligned rectangular conductor the same residue reaches the line.)
    """
    fields = {}
    for tag, disarm in (('armed', False), ('disarmed', True)):
        grid = _coax(conformal=True)
        launch, absorb = _run(grid, disarm=disarm)
        fields[tag] = (grid.Ex.copy(), grid.Ey.copy(), launch._h_k, absorb._h_k)
    for a, b in zip(fields['armed'][:2], fields['disarmed'][:2]):
        assert np.array_equal(a, b)
