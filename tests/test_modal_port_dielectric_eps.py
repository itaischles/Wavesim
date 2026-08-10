"""A ModalPort's sheet must be divergence-free on a conformal grid *with a
dielectric fill* — which is the case the rest of the suite cannot fail.

``ê`` has to be a null vector of the FDTD's own discrete transverse curl at every
FREE node of the port plane. That is not a nicety: :meth:`ModalPort.apply` writes
``ĥ`` as ghost H every step and the ghost plane is open loop, so whatever the next
Ampère update deposits there can never act back on the H that produced it. It
integrates instead — a static pile on *both* port planes (including the one with
``amplitude=0``), decaying into the domain, plus a DC current down the line.

The mode solver drives ``∇_t·(ε ê)`` to round-off at exactly those free nodes. So
the sheet is divergence-free if and only if the ε in that statement is the ε the
leapfrog steps. It was not: :func:`wavesim.mode_solver._face_eps` weighted each
face by the permittivity of the face **outward** where it straddles the conductor
surface — a face's stored ε there is whatever the voxeliser left inside the metal,
not a material property — while ``update_E`` consumed the stored per-edge ε
verbatim. Where those differ, the mode is solved on one material map and stepped
on another. :func:`wavesim.pec.conformal_edge_eps` is now the single answer both
halves ask.

**Why the existing suite missed it.** The two maps can only differ where ε does,
and the conformal cases already in the suite are air-filled or uniform. On such a
grid the outward rule is a no-op and the defect is unreachable — a point this file
asserts outright (:func:`test_an_air_filled_coax_cannot_fail_this`) rather than
leaving as a remark, because it is the whole reason a new file exists.
``ε_eff`` is no help either: it reads exactly 2.300000 on the broken grid, the
static operator being blind to the disagreement.

Everything here drives :meth:`wavesim.simulation.Simulation.run`. Plan R7: a guard
checked at the wrong entry point proves nothing, so the ports are set up by the
real time loop and the residual is read off the object the loop used.
"""

import contextlib
import warnings

import numpy as np
import pytest

import wavesim as ws
import wavesim.pec as pec
from wavesim.mode_solver import (solve_tem_modes, port_sheet_divergence,
                                 port_plane_pinned_nodes)

from conformal_shapes import coax_fractions

A_IN, B_OUT = 3.0e-3, 9.0e-3
CELL = 0.5e-3
N_TR = 36
N_AX = 40
EPS_R = 2.3

ROUNDOFF = 1e-9         # round-off measures ~1e-14; a real failure ~1e-1


def _coax(eps_r=EPS_R):
    """Conformal coax with a dielectric fill, built the way a user would.

    ``ws.set_coax`` fills only *between* the conductors and leaves the background
    ε = 1 inside the metal — so every edge crossing a conductor surface carries a
    permittivity that is not a material property. That is not a contrived input:
    it is what the library's own constructor produces, what a voxeliser produces,
    and what the workbench produced until 2026-08-10. It is also invisible on a
    staircase grid, where the dilation of
    :func:`~wavesim.pec.build_pec_edge_masks` holds every one of those edges at
    zero and their ε is never read.
    """
    grid = ws.create_grid(Nx=N_TR, Ny=N_TR, Nz=N_AX, dx=CELL, dy=CELL, dz=CELL)
    ws.set_vacuum(grid)
    cx = cy = 0.5 * N_TR * CELL
    ws.set_coax(grid, cx=cx, cy=cy, r_inner=A_IN, r_outer=B_OUT,
                eps_r_fill=eps_r)
    ws.set_material_arrays(
        grid, grid.eps_x, grid.eps_y, grid.eps_z,
        grid.mu_x, grid.mu_y, grid.mu_z,
        **coax_fractions(grid, cx, cy, A_IN, B_OUT))
    return grid


def _modulated_gaussian(f0=1e9, n_sigma=3.0):
    """Sine-modulated Gaussian — negligible DC, so nothing here can be blamed on
    the drive's time integral (plan R8; zero-DC drive is an amplifier of this
    defect, not its source)."""
    width = n_sigma / (2.0 * np.pi * f0)
    t0 = 4.0 * width

    def wave(t):
        return float(np.exp(-0.5 * ((t - t0) / width) ** 2)
                     * np.sin(2.0 * np.pi * f0 * (t - t0)))
    return wave


@contextlib.contextmanager
def _two_material_maps():
    """Put the solver back to reading two different ε maps.

    Patches :func:`wavesim.pec.outward_edge_eps` to the identity, which is the
    one choke point: :func:`~wavesim.pec.conformal_edge_eps` resolves that name
    through :mod:`wavesim.pec`'s globals at call time, so every consumer of the
    repaired map — the E update, the PML's matching coefficient, the loss
    coefficients, the Numba kernels, and the ``η`` a port's ``ĥ`` is divided by —
    reverts together, from one line.

    :mod:`wavesim.mode_solver` bound the name at import, so ``_face_eps`` keeps
    the rule and goes on weighting its faces the outward way. That asymmetry is
    the point: it is precisely the state the solver was in.
    """
    original = pec.outward_edge_eps
    pec.outward_edge_eps = lambda eps, node_pec, axis: np.asarray(
        eps, dtype=np.float64).copy()
    try:
        yield
    finally:
        pec.outward_edge_eps = original


def _run(grid, steps=300, amplitude=1.0, backend='numba'):
    """Drive the real time loop; return ``(launch, absorb)`` after ``run``."""
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
    sim.run(steps)
    return launch, absorb


def _ghost_vs_line(grid, launch, absorb):
    """``(max|Ez| on either ghost plane, max|Ez| on the line between them)``."""
    planes = (launch._h_k, absorb._h_k)
    ghost = max(float(np.abs(grid.Ez[:, :, k]).max()) for k in planes)
    interior = np.ones(grid.Nz, dtype=bool)
    for k in planes:
        interior[k] = False
    return ghost, float(np.abs(grid.Ez[:, :, interior]).max())


# ---------------------------------------------------------------------- #
# The case has to be capable of failing
# ---------------------------------------------------------------------- #

def test_the_fill_really_does_straddle_the_conductor():
    """Guards the premise. If ``set_coax`` ever starts filling the metal too,
    or the conductor stops meeting the grid, this file goes quiet without
    anyone noticing — the R7 failure mode in its most ordinary costume."""
    grid = _coax()
    node = ws.pec_node_mask(grid)
    straddling = 0
    for axis, comp in enumerate('xyz'):
        raw = getattr(grid, 'eps_' + comp)
        repaired = pec.conformal_edge_eps(grid)[axis]
        straddling += int(np.count_nonzero(repaired != raw))
    assert straddling > 100, (
        f"only {straddling} edges have an ε the outward rule moves; this grid "
        f"cannot exercise the two-map disagreement")
    assert node.any()
    # And the stored map really is non-uniform over the edges the run keeps.
    live = ~pec.build_conformal_edge_masks(grid)[0]
    assert len(np.unique(grid.eps_x[live])) > 1


def test_the_port_plane_has_free_nodes():
    """``port_sheet_divergence`` returns 0.0 for a plane with no free nodes, so
    an assertion on it would pass vacuously on an all-conductor cross-section."""
    grid = _coax()
    free = ~port_plane_pinned_nodes(grid, 'z', 1)
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    assert int(free.sum()) > 100


# ---------------------------------------------------------------------- #
# The defect
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_dielectric_port_sheet_is_divergence_free(backend):
    """The assertion this file exists for: round-off at the free nodes.

    Read off the ports the time loop set up, after it ran, on both backends —
    the mismatch was in a material map, which is backend-independent, and a
    result that held on only one of them would mean something else was going on.
    """
    grid = _coax()
    launch, absorb = _run(grid, backend=backend)
    for port in (launch, absorb):
        assert port.sheet_divergence < ROUNDOFF, (
            f"{port.mode.normal}-port at index {port.mode.slice_index} injects "
            f"a sheet with transverse divergence {port.sheet_divergence:.4g} at "
            f"nodes the mode solver solved φ for")


def test_disarming_the_shared_eps_brings_the_defect_straight_back():
    """Sensitivity check: with two maps again, the residual returns by ~1e12.

    Without this the test above could be passing because the case is too gentle
    to excite the defect rather than because the fix works.
    """
    with _two_material_maps():
        grid = _coax()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            launch, absorb = _run(grid)
    bad = max(launch.sheet_divergence, absorb.sheet_divergence)
    assert bad > 1e-3, (
        f"the reference case no longer excites the two-map defect "
        f"(divergence {bad:.4g}); it is not testing the fix any more")
    # And the solver says so rather than running on in silence, which is the
    # other half of the ask: nothing used to notice when the map was bad.
    assert any('transverse divergence' in str(w.message) for w in caught)


def test_an_air_filled_coax_cannot_fail_this():
    """Why the suite missed it for so long, asserted rather than asserted-about.

    With ε ≡ 1 the two maps are the same map, so the disarmed run is *also*
    clean. Every conformal port case already in the suite is air-filled or
    uniform, and none of them could have caught this however hard they pushed.
    """
    with _two_material_maps():
        grid = _coax(eps_r=1.0)
        launch, absorb = _run(grid)
    assert max(launch.sheet_divergence, absorb.sheet_divergence) < ROUNDOFF


# ---------------------------------------------------------------------- #
# The symptom the divergence causes
# ---------------------------------------------------------------------- #

def test_no_static_pile_on_the_port_planes():
    """End to end: the port planes must not be the hottest ``Ez`` in a TEM run.

    ``Ez`` is zero in the continuum for a TEM mode, so every ``Ez`` here is
    numerical residue and only a comparison means anything. With two material
    maps the ghost planes carry a static field far above the travelling wave;
    with one they sit below the line between them.
    """
    grid = _coax()
    launch, absorb = _run(grid)
    ghost, line = _ghost_vs_line(grid, launch, absorb)
    assert ghost < line, (
        f"port planes carry max|Ez| = {ghost:.4g} V/m against {line:.4g} V/m "
        f"on the line between them")

    with _two_material_maps():
        bad_grid = _coax()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bad_launch, bad_absorb = _run(bad_grid)
    bad_ghost, bad_line = _ghost_vs_line(bad_grid, bad_launch, bad_absorb)
    assert bad_ghost > bad_line, (
        "the disarmed run no longer piles up on the port planes; the symptom "
        "and the residual have come apart and one of them is lying")


def test_the_absorbing_port_piles_up_too_with_no_drive_at_all():
    """``amplitude = 0`` on both ports still does it, so no launch-local fix
    works. The sheet is ``s·(V̄ − 2a)``: with ``a = 0`` it is still nonzero
    wherever the port sees a voltage, so a pure absorber accumulates the same
    residual from whatever is seeded into the domain."""
    def seeded_ghost(disarmed):
        with _two_material_maps() if disarmed else contextlib.nullcontext():
            grid = _coax()
            mode = solve_tem_modes(grid, normal='z', position=CELL,
                                   compute_params=True)[0]
            E, _H = mode._staggered_port_fields(grid)
            mid = grid.Nz // 2
            for comp, prof in E.items():
                getattr(grid, comp)[:, :, mid - 4:mid + 4] += prof[:, :, None]
            ws.apply_pec_mask(grid)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                launch, absorb = _run(grid, amplitude=0.0)
        return _ghost_vs_line(grid, launch, absorb)

    ghost_bad, _ = seeded_ghost(disarmed=True)
    ghost_ok, line = seeded_ghost(disarmed=False)
    assert ghost_bad > 1e2 * ghost_ok, (
        f"with no drive at all the shared ε should still be doing the work: "
        f"two maps {ghost_bad:.4g} V/m vs one {ghost_ok:.4g} V/m")
    assert ghost_ok < line


# ---------------------------------------------------------------------- #
# What the fix must not do
# ---------------------------------------------------------------------- #

def test_a_staircase_grid_is_untouched():
    """V2: the repair returns the stored arrays *by identity* off the conformal
    path, so nothing on it can change — the dilation already holds every
    conductor-straddling edge at zero and their ε is never read."""
    grid = ws.create_grid(Nx=N_TR, Ny=N_TR, Nz=N_AX, dx=CELL, dy=CELL, dz=CELL)
    ws.set_vacuum(grid)
    c = 0.5 * N_TR * CELL
    ws.set_coax(grid, cx=c, cy=c, r_inner=A_IN, r_outer=B_OUT, eps_r_fill=EPS_R)
    repaired = pec.conformal_edge_eps(grid)
    assert repaired[0] is grid.eps_x
    assert repaired[1] is grid.eps_y
    assert repaired[2] is grid.eps_z


def test_the_repair_leaves_a_uniform_map_alone():
    """A grid whose ε is already uniform has nothing to repair, so the arrays
    must come back equal — otherwise the rule is inventing material."""
    grid = ws.create_grid(Nx=N_TR, Ny=N_TR, Nz=N_AX, dx=CELL, dy=CELL, dz=CELL)
    ws.set_vacuum(grid)
    c = 0.5 * N_TR * CELL
    for a in 'xyz':
        getattr(grid, 'eps_' + a)[...] = EPS_R
    ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                           grid.mu_x, grid.mu_y, grid.mu_z,
                           **coax_fractions(grid, c, c, A_IN, B_OUT))
    for repaired, comp in zip(pec.conformal_edge_eps(grid), 'xyz'):
        assert np.array_equal(repaired, getattr(grid, 'eps_' + comp))
