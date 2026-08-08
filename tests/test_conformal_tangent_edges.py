"""An E edge lying *in* a grid-aligned PEC surface must be held at zero.

Such an edge is not covered by the conductor at all — the metal is on one side of
it, so its own open fraction is a full 1.0 — and it is nonetheless a tangential E
on a PEC boundary. :func:`wavesim.pec.build_conformal_edge_masks` reads it off the
H face on the metal side: a face with zero open area lies wholly inside the
conductor, so the four edges of its Faraday contour lie in the conductor's
closure, tangent ones included.

Leaving them alive is what wrecked a :class:`~wavesim.sources.ModalPort` on a
conformal grid. The sheet's transverse divergence deposits the mode's induced
surface charge onto the port plane every step; a staircase run's dilation had
been swallowing it into the metal all along. With nothing to swallow it, it
integrates: on the case below the port plane reached 20x the physical field,
static, and the launch collapsed to 1e-4 V.

The condition is **grid alignment**, not cut-cell size, so the cases here are
deliberately aligned. ``tests/test_conformal_geometry.py`` and
``test_conformal_update.py`` cover the other side of it — that a genuine cut cell
is left alone.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim.mode_solver import solve_tem_modes
from wavesim.pec import (build_pec_edge_masks, build_conformal_edge_masks,
                         _dilate)

from conformal_shapes import binary_fractions, coax_fractions

CELL = 1.0e-3
N_TR, N_AX = 20, 100
EPS_R = 2.3


# ---------------------------------------------------------------------- #
# The geometric rule
# ---------------------------------------------------------------------- #

def _grid(mask, **fractions):
    grid = ws.create_grid(Nx=mask.shape[0], Ny=mask.shape[1], Nz=mask.shape[2],
                          dx=CELL, dy=CELL, dz=CELL)
    ws.set_vacuum(grid)
    grid.pec_mask = mask
    if fractions:
        ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                               grid.mu_x, grid.mu_y, grid.mu_z, **fractions)
    return grid


def _strict_fractions(mask):
    """0/1 fractions read as 'covered iff **strictly interior** to the metal'.

    The opposite reading to :func:`~conformal_shapes.binary_fractions`' closure
    convention, and the hostile one for this rule: nothing on the conductor's
    surface is marked covered, so every surface element has to be found through
    a face. It is also the reading a sampler naturally produces, since a sample
    point on the boundary is as likely to test outside as in.
    """
    def erode(m, axes):
        return ~_dilate(~m, axes)

    return dict(
        pec_edge_open_x=(~erode(mask, (1, 2))).astype(float),
        pec_edge_open_y=(~erode(mask, (0, 2))).astype(float),
        pec_edge_open_z=(~erode(mask, (0, 1))).astype(float),
        pec_face_open_x=(~erode(mask, (0,))).astype(float),
        pec_face_open_y=(~erode(mask, (1,))).astype(float),
        pec_face_open_z=(~erode(mask, (2,))).astype(float))


def _shapes():
    n = 16
    i = np.arange(n)[:, None, None]
    j = np.arange(n)[None, :, None]
    k = np.arange(n)[None, None, :]
    return {
        'block': ((i >= 4) & (i < 11) & (j >= 4) & (j < 11)
                  & (k >= 4) & (k < 11)),
        'slab': np.broadcast_to((i < 5), (n, n, n)),
        'shield': np.broadcast_to(((i < 3) | (i >= 13) | (j < 3) | (j >= 13)),
                                  (n, n, n)),
    }


@pytest.mark.parametrize('name', sorted(_shapes()))
def test_the_face_clause_alone_reproduces_the_staircase_dilation(name):
    """The whole point of the tangency clause, stated as an identity.

    Every edge fraction is set fully open, so the own-length clause contributes
    nothing and the answer is the *faces'* doing alone. Given faces read the way
    a 0/1 geometry has to read them — covered iff either cell it separates is
    metal, because a face on a staircased surface lies in the closure of the
    metal — the clause comes out as :func:`~wavesim.pec.build_pec_edge_masks`
    exactly. That is the dilation whose accidental coverage of surface-lying
    edges the first conformal rule removed, recovered from geometry rather than
    reinstated by hand: ``Ez`` is an Hx face dilated along y OR an Hy face
    dilated along x, which is ``dilate(mask, (x, y))``.
    """
    mask = np.ascontiguousarray(_shapes()[name])
    fr = binary_fractions(mask)
    for key in ('pec_edge_open_x', 'pec_edge_open_y', 'pec_edge_open_z'):
        fr[key] = np.ones_like(fr[key])
    grid = _grid(mask, **fr)
    for conf, stair, lbl in zip(build_conformal_edge_masks(grid),
                                build_pec_edge_masks(mask), 'xyz'):
        assert np.array_equal(conf, stair), (
            f"E{lbl}: conformal masks {int(conf.sum())} edges, staircase "
            f"{int(stair.sum())} — missing {int((stair & ~conf).sum())}, "
            f"extra {int((conf & ~stair).sum())}")


def test_a_surface_lying_edge_is_zeroed_though_it_is_fully_open():
    """The defect itself, in the smallest geometry that has it.

    A slab of metal filling ``x < 5``. The edges running *along* its face, in the
    plane of nodes ``x = 5``, are not covered by the metal at all — read the
    fractions strictly and their open length is the full 1.0 — and they are
    tangential E on a PEC surface. The own-length rule leaves every one of them
    alive; the face on the metal side (open area 0) is what gives them away.
    """
    mask = np.ascontiguousarray(_shapes()['slab'])
    grid = _grid(mask, **_strict_fractions(mask))
    surface = (slice(5, 6), slice(1, -1), slice(1, -1))

    assert np.all(grid.pec_edge_open_y[surface] == 1.0)
    assert np.all(grid.pec_edge_open_z[surface] == 1.0)
    _ex, ey, ez = build_conformal_edge_masks(grid)
    assert np.all(ey[surface])
    assert np.all(ez[surface])


def test_a_cut_cell_is_left_alone():
    """No fully covered face, no clause — the sub-cell geometry is untouched.

    A conductor crossing its cells at an angle is the case conformal PEC exists
    for, and the rule must not reach back into it and become the dilation again.
    Checked on the reference coax at a quarter-cell offset, where no surface
    lands on the ruler at all.

    'Untouched' is not 'zeroes nothing partially open'. This fixture samples the
    Hz face on a 64x64 sub-block and the Ex/Ey edges in closed form, so a face
    can round to zero area while its own boundary edge is still 1.5% open. The
    rule then zeroes that sliver — which is right, since the zero-area face has
    already frozen the H it borders, and E and H agreeing is what conformal PEC
    is for. What must not happen is a *substantially* open edge going dark.
    """
    n = 24
    cell = 0.5e-3
    grid = ws.create_grid(Nx=n, Ny=n, Nz=8, dx=cell, dy=cell, dz=cell)
    ws.set_vacuum(grid)
    cx = cy = 0.5 * n * cell + 0.25 * cell
    r2 = ((grid.xc[:, None, None] - cx) ** 2 + (grid.yc[None, :, None] - cy) ** 2)
    grid.pec_mask = np.broadcast_to((r2 < 3.0e-3 ** 2) | (r2 > 9.0e-3 ** 2),
                                    (grid.Nx, grid.Ny, grid.Nz)).copy()
    ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                           grid.mu_x, grid.mu_y, grid.mu_z,
                           **coax_fractions(grid, cx, cy, 3.0e-3, 9.0e-3))
    conformal = build_conformal_edge_masks(grid)
    fractions = (grid.pec_edge_open_x, grid.pec_edge_open_y,
                 grid.pec_edge_open_z)
    for conf, stair, frac, lbl in zip(conformal,
                                      build_pec_edge_masks(grid.pec_mask),
                                      fractions, 'xyz'):
        assert not np.any(conf & ~stair), (
            f"E{lbl}: zeroes {int((conf & ~stair).sum())} edges the staircase "
            f"dilation keeps — the rule has overshot its own upper bound")
        eaten = conf & (frac >= 0.05)
        assert not np.any(eaten), (
            f"E{lbl}: {int(eaten.sum())} edges at least 5% open were zeroed on "
            f"an off-lattice geometry — the rule is eating conformal field")
    # Stated the other way round, which is what 'left alone' means: the cut
    # edges — the ones carrying the sub-cell information — survive essentially
    # intact, rather than being rounded back into the metal.
    cut = sum(int(((f > 0.0) & (f < 1.0)).sum()) for f in fractions)
    lost = sum(int((c & (f > 0.0) & (f < 1.0)).sum())
               for c, f in zip(conformal, fractions))
    assert cut > 100 and lost < 0.02 * cut, (
        f"{lost} of {cut} cut edges zeroed")


def test_a_one_cell_gap_is_not_shorted():
    """Two conductors one cell apart still have a live edge between them.

    The near-miss for any rule phrased on *nodes* rather than on faces: both end
    nodes of that edge are metal, and the edge is free space. No face bounded by
    it is covered, so the clause does not fire.
    """
    n = 12
    i = np.arange(n)[:, None, None]
    mask = np.ascontiguousarray(
        np.broadcast_to((i < 4) | (i >= 7), (n, n, n)))     # gap = cell i=4..6
    grid = _grid(mask, **binary_fractions(mask))
    ex = build_conformal_edge_masks(grid)[0]
    assert not np.all(ex[5, 2:-2, 2:-2]), (
        "the edge spanning the one-cell gap was zeroed — the conductors are "
        "shorted")


def test_the_face_rule_matches_the_update_stencil():
    """The edge -> bounded-face adjacency, checked against one covered face.

    A single covered ``Hz[i,j,k]`` must kill exactly the four edges of its
    Faraday contour, ``Ex[i,j..j+1,k]`` and ``Ey[i..i+1,j,k]``
    (:mod:`wavesim.update`) — no more, no fewer. Written out by hand because an
    off-by-one in the shift direction is invisible on a symmetric shape.
    """
    n = 8
    mask = np.zeros((n, n, n), bool)
    grid = _grid(mask, **binary_fractions(mask))
    grid.pec_face_open_z[3, 4, 5] = 0.0
    ex, ey, ez = build_conformal_edge_masks(grid)
    assert sorted(map(tuple, np.argwhere(ex))) == [(3, 4, 5), (3, 5, 5)]
    assert sorted(map(tuple, np.argwhere(ey))) == [(3, 4, 5), (4, 4, 5)]
    assert not ez.any()


# ---------------------------------------------------------------------- #
# What it was breaking: the modal port
# ---------------------------------------------------------------------- #

def _square_coax():
    """Grid-aligned square coax — every surface on the node ruler.

    The maximal-tangency geometry: a round coax has a handful of nodes on the
    ruler, this has nothing else. Fractions read 'covered iff strictly interior
    to the metal', the strictest available and the hardest case for a rule that
    has to find the tangency through a face.
    """
    grid = ws.create_grid(Nx=N_TR, Ny=N_TR, Nz=N_AX, dx=CELL, dy=CELL, dz=CELL)
    ws.set_vacuum(grid)
    for axis in 'xyz':
        getattr(grid, 'eps_' + axis)[...] = EPS_R
    i = np.arange(N_TR)[:, None]
    j = np.arange(N_TR)[None, :]
    core = (i >= 8) & (i < 12) & (j >= 8) & (j < 12)
    shield = (i < 2) | (i >= 18) | (j < 2) | (j >= 18)
    grid.pec_mask = np.broadcast_to((core | shield)[:, :, None],
                                    (N_TR, N_TR, N_AX)).copy()
    ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                           grid.mu_x, grid.mu_y, grid.mu_z,
                           **_strict_fractions(grid.pec_mask))
    return grid


def _port_run(grid, steps=2099, backend='numba'):
    """Launch into one ModalPort, absorb at the other; return the §7 numbers."""
    lo = solve_tem_modes(grid, normal='z', position=CELL,
                         compute_params=True)[0]
    hi = solve_tem_modes(grid, normal='z', position=(grid.Nz - 1) * CELL,
                         compute_params=True)[0]
    launch = ws.ModalPort(lo, amplitude=1.0,
                          waveform=ws.GaussianPulse.for_fmax(1e9))
    absorb = ws.ModalPort(hi, amplitude=0.0)
    c = 0.5 * N_TR * CELL
    z_mid = (grid.Nz // 2) * CELL
    vmon = ws.VoltageMonitor(path=((c + 2.5 * CELL, c, z_mid),
                                   (c + 9.5 * CELL, c, z_mid)))
    sim = ws.Simulation(grid, monitors=[vmon], backend=backend,
                        pec_faces=('x0', 'x1', 'y0', 'y1'))
    sim.add_boundary(launch)
    sim.add_boundary(absorb)
    sim.run(steps)

    kp = lo.slice_index
    port = max(float(np.abs(grid.Ex[:, :, kp]).max()),
               float(np.abs(grid.Ey[:, :, kp]).max()))
    interior = np.ones(grid.Nz, bool)
    for k in (launch._h_k, absorb._h_k, lo.slice_index, hi.slice_index):
        interior[k] = False
    line = float(max(np.abs(grid.Ex[:, :, interior]).max(),
                     np.abs(grid.Ey[:, :, interior]).max()))
    v = np.abs(np.asarray(vmon.values, float))
    return dict(port=port, line=line, v_peak=float(v.max()),
                v_tail=float(v[int(0.75 * v.size):].max() / v.max()))


@pytest.fixture(scope='module')
def square_coax_run():
    """One 2099-step run, shared: it is the expensive thing in this file."""
    return _port_run(_square_coax())


def test_a_modal_port_does_not_pump_dc_into_its_own_plane(square_coax_run):
    """Acceptance §7.1: the port plane decays with the pulse.

    The residual field left on the mode plane must not exceed what the line
    between the ports carries — the port plane used to be the *hottest* place in
    the domain, by 20x, and static.
    """
    r = square_coax_run
    assert r['port'] <= r['line'], (
        f"port plane holds {r['port']:.4g} V/m against {r['line']:.4g} V/m "
        f"mid-line")
    assert r['v_tail'] < 1e-3, f"V tail is {r['v_tail']:.4g} of peak"


def test_the_launch_survives_the_tangency(square_coax_run):
    """Acceptance §7.3: the sheet still launches what it is calibrated to.

    A ``ModalPort`` reads its own plane back to decide what to radiate, so a
    static pile on that plane does not merely add an offset — it takes over the
    feedback. On this geometry the launch collapsed to 1e-4 V of a ~0.8 V wave.
    """
    assert 0.5 < square_coax_run['v_peak'] < 1.2, (
        f"launched {square_coax_run['v_peak']:.4g} V for amplitude = 1")
