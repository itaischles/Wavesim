"""Drawing an electrostatic solution: one quantity, on one plane.

Plotting tests are usually thin — a picture is hard to assert about — but two
things here are not cosmetic.

**Registration.** φ lives on the Yee nodes and the arrays carry ``N`` of them,
while ``pcolormesh(shading='flat')`` wants ``N+1`` boundaries for ``N`` samples
and ``grid.x`` supplies the boundaries of ``N`` *cells*. Feeding one to the
other draws the potential half a cell from where it was solved — the same
off-by-half-a-cell trap ``wavesim.parts`` and ``wavesim.pec`` exist to document,
arriving in the plotting layer.

**Collocation.** The three E components live on three different edge families,
so a plane of ``Ex`` and a plane of ``Ey`` are not sampled at the same points
until something moves them. Everything drawn here is brought to the nodes
first, which is the property that makes two of these plots comparable to each
other — and it is φ's own grid, so the only exact quantity in the picture is
the one that does not move.
"""

import matplotlib
matplotlib.use('Agg')                       # no display in CI; must precede pyplot

import matplotlib.pyplot as plt
import numpy as np
import pytest

import wavesim as ws
from wavesim.electrostatics import Electrostatics, _node_dual, _to_nodes
from wavesim.viz import _node_edges, plot_electrostatic_slice

QUANTITIES = ('phi', '|E|', '|D|', 'Ex', 'Ey', 'Ez', 'Dx', 'Dy', 'Dz')


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _solution(n=8, graded=False):
    """Two plates across z, solved. ``graded`` puts a non-uniform mesh in x."""
    ds = 1e-3
    if graded:
        xs = np.concatenate([[0.0], np.cumsum(np.linspace(ds, 2 * ds, n))])
        grid = ws.create_grid_rectilinear(xs, np.arange(n + 1) * ds,
                                          np.arange(n + 1) * ds)
    else:
        grid = ws.create_grid(n, n, n, ds, ds, ds)
    ws.set_vacuum(grid)
    top = (n - 1) * ds
    ws.set_box(grid, 0, n * ds, 0, n * ds, 0, ds, 1.0, pec=True, name="bot")
    ws.set_box(grid, 0, n * ds, 0, n * ds, top, top + ds, 1.0, pec=True,
               name="top")
    es = Electrostatics(grid)
    es.set_potential("bot", 0.0).set_potential("top", 1.0)
    return es.solve(boundary='neumann', method='direct')


def _quadmesh(ax):
    """The (Nb+1, Na+1, 2) corner coordinates the axes actually drew."""
    return np.asarray(ax.collections[0].get_coordinates())


# ---------------------------------------------------------------------- #
# Registration
# ---------------------------------------------------------------------- #

def test_node_samples_are_drawn_on_their_dual_cells():
    """The drawn width of node ``i`` is the control volume the solve used.

    ``_node_dual`` is what the operator integrates over, so agreeing with it is
    the definition of drawing the potential where it was solved — including the
    half cells at both ends, the detail that puts a Neumann symmetry plane on
    the edge of the picture rather than half a cell inside it.
    """
    n = 7
    nodes = np.concatenate([[0.0], np.cumsum(np.linspace(1e-3, 2e-3, n))])
    grid = ws.set_vacuum(ws.create_grid_rectilinear(nodes, nodes[:3],
                                                    nodes[:3]))
    edges = _node_edges(grid.x, grid.xc, n)

    assert len(edges) == n + 1
    assert np.allclose(np.diff(edges), _node_dual(grid.dxp))
    assert edges[0] == grid.x[0] and edges[-1] == grid.x[n - 1]


def test_the_drawn_mesh_is_the_dual_one_not_the_primary_one():
    """End to end, on a graded mesh where the two visibly differ.

    Handing φ to the primary node ruler instead would make this read
    ``grid.dxp``, and every feature in the picture would sit half a cell off.
    """
    sol = _solution(graded=True)
    _, ax = plot_electrostatic_slice(sol, 'phi', normal='z')
    xs = _quadmesh(ax)[0, :, 0]
    assert np.allclose(np.diff(xs), _node_dual(sol.grid.dxp))


def test_a_single_cell_axis_still_has_two_boundaries():
    """The quasi-2D case (Nz=1) must not produce a degenerate mesh."""
    grid = ws.set_vacuum(ws.create_grid(4, 4, 1, 1e-3, 1e-3, 1e-3))
    edges = _node_edges(grid.z, grid.zc, 1)
    assert len(edges) == 2 and edges[1] > edges[0]


# ---------------------------------------------------------------------- #
# Collocation
# ---------------------------------------------------------------------- #

def test_every_quantity_shares_one_coordinate_grid():
    """Two of these plots are comparable only if they sample the same points.

    ``Ex`` and ``Ey`` are stored on different edge families; collocating both
    to the nodes is what lets them be read against each other, and against φ.
    """
    sol = _solution()
    meshes = []
    for quantity in QUANTITIES:
        fig, ax = plot_electrostatic_slice(sol, quantity)
        meshes.append(_quadmesh(ax))
        plt.close(fig)
    for mesh in meshes[1:]:
        assert np.array_equal(mesh, meshes[0])


def test_the_walls_are_one_sided_rather_than_averaged_against_nothing():
    """Both end nodes take the single edge that exists.

    The array carries ``N`` slots for ``N+1`` nodes, so the last node's upper
    edge is absent and reads zero. Averaging against it would report exactly
    half the field at the top wall — the missing-end trap ``_node_dual``
    documents, in the interpolation instead of the control volumes.
    """
    E = np.zeros((5, 1, 1))
    E[:4, 0, 0] = [1.0, 2.0, 3.0, 4.0]          # last edge not carried
    (centred,) = _to_nodes((E,))[:1]
    assert centred[0, 0, 0] == 1.0              # no edge below node 0
    assert centred[1, 0, 0] == 1.5              # interior: the two-edge mean
    assert centred[4, 0, 0] == 4.0              # not 2.0


def test_a_uniform_field_stays_uniform_through_the_walls():
    """The parallel plate's gap field is constant, so its collocation must be
    too — including at the boundary nodes, which is what the one-sided rule at
    the walls buys."""
    sol = _solution()
    Ez = sol.E_nodes[2][4, 4, 2:6]              # inside the gap, off the plates
    assert np.allclose(Ez, Ez[0], rtol=1e-12)


# ---------------------------------------------------------------------- #
# Picking the plane
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("normal, shape_axes", [('z', (0, 1)), ('y', (0, 2)),
                                                ('x', (1, 2))])
def test_each_normal_slices_the_axes_the_snapshot_monitor_would(normal,
                                                                shape_axes):
    """'z' -> XY, 'y' -> XZ, 'x' -> YZ, as SnapshotMonitor and the mode solver."""
    sol = _solution(n=6)
    grid = sol.grid
    # A distinguishable shape per axis, so a transposed plane cannot pass.
    sol.phi = np.arange(grid.Nx * grid.Ny * grid.Nz, dtype=float).reshape(
        (grid.Nx, grid.Ny, grid.Nz))
    _, ax = plot_electrostatic_slice(sol, 'phi', normal=normal,
                                     conductors=False)
    a, b = shape_axes
    drawn = _quadmesh(ax)
    assert drawn.shape[:2] == (sol.phi.shape[b] + 1, sol.phi.shape[a] + 1)
    assert ax.get_xlabel() == f'{"xyz"[a]} (m)'
    assert ax.get_ylabel() == f'{"xyz"[b]} (m)'


def test_the_cut_position_is_honoured_and_reported():
    sol = _solution()
    _, ax = plot_electrostatic_slice(sol, 'phi', normal='z', position=6e-3)
    assert f'z = {sol.grid.z[6]:.4g} m' in ax.get_title()


def test_the_default_cut_is_the_middle_of_the_domain():
    sol = _solution(n=8)
    _, ax = plot_electrostatic_slice(sol, 'phi', normal='z')
    assert f'z = {sol.grid.z[4]:.4g} m' in ax.get_title()


def test_a_position_past_the_last_carried_node_is_clamped():
    """``N`` cells have ``N+1`` nodes and the arrays hold ``N``; asking for the
    far wall must land on the last node rather than index out of the array."""
    sol = _solution(n=8)
    _, ax = plot_electrostatic_slice(sol, 'phi', normal='z', position=8e-3)
    assert f'z = {sol.grid.z[7]:.4g} m' in ax.get_title()


# ---------------------------------------------------------------------- #
# Content
# ---------------------------------------------------------------------- #

def test_the_conductor_outline_comes_from_the_solved_conductor():
    """Outlined from ``node_pec``, the metal the solve pinned, not ``pec_mask``.

    The two differ by half a cell on every high-side surface — the distinction
    ``docs/mode_solver_staircase_node_mask.md`` was written about — so an
    outline from the cell mask would not sit on the equipotential the colours
    show.
    """
    sol = _solution()
    _, cut = plot_electrostatic_slice(sol, 'phi', normal='y')   # cuts the plates
    assert len(cut.collections) > 1

    _, bare = plot_electrostatic_slice(sol, 'phi', normal='y', conductors=False)
    assert len(bare.collections) == 1


def test_a_plane_with_no_conductor_boundary_draws_no_outline():
    """The plates span the full transverse extent, so an XY cut through the gap
    has no metal in it — and a contour of a uniformly-false plane would be a
    line drawn at no boundary at all."""
    sol = _solution()
    _, ax = plot_electrostatic_slice(sol, 'phi', normal='z')
    assert len(ax.collections) == 1


def test_the_drive_conditions_are_in_the_title():
    """A field picture without its boundary conditions is not interpretable."""
    sol = _solution()
    _, ax = plot_electrostatic_slice(sol, 'phi')
    assert "bot = 0 V" in ax.get_title() and "top = 1 V" in ax.get_title()


def test_a_signed_quantity_gets_a_diverging_scale_and_a_magnitude_does_not():
    sol = _solution()
    _, signed = plot_electrostatic_slice(sol, 'Ez', normal='y')
    _, unsigned = plot_electrostatic_slice(sol, '|E|', normal='y')
    assert signed.collections[0].get_clim()[0] < 0
    assert unsigned.collections[0].get_clim()[0] == 0.0


def test_the_plane_is_drawn_in_true_proportion_unless_asked_otherwise():
    """A field picture at a distorted aspect misleads about the geometry of the
    field, so equal is the default; a very long thin domain can opt out."""
    sol = _solution()
    _, true_shape = plot_electrostatic_slice(sol, 'phi')
    assert true_shape.get_aspect() == 1.0
    _, stretched = plot_electrostatic_slice(sol, 'phi', aspect='auto')
    assert stretched.get_aspect() == 'auto'


def test_an_elongated_plane_gets_a_figure_shaped_like_it():
    """Otherwise an equal-aspect cut through a long thin domain is a sliver
    adrift in a square figure."""
    ds = 1e-3
    grid = ws.set_vacuum(ws.create_grid(40, 6, 6, ds, ds, ds))
    ws.set_box(grid, 0, 40 * ds, 0, 6 * ds, 0, ds, 1.0, pec=True, name="bot")
    ws.set_box(grid, 0, 40 * ds, 0, 6 * ds, 5 * ds, 6 * ds, 1.0, pec=True,
               name="top")
    es = Electrostatics(grid)
    es.set_potential("bot", 0.0).set_potential("top", 1.0)
    sol = es.solve(boundary='neumann', method='direct')

    wide, _ = plot_electrostatic_slice(sol, 'phi', normal='y')   # 40 x 6 cells
    square, _ = plot_electrostatic_slice(sol, 'phi', normal='x')  # 6 x 6 cells
    w, h = wide.get_size_inches()
    assert w / h > square.get_size_inches()[0] / square.get_size_inches()[1]


def test_every_quantity_draws():
    sol = _solution()
    for quantity in QUANTITIES:
        fig, ax = plot_electrostatic_slice(sol, quantity, normal='y')
        assert ax.collections
        plt.close(fig)


def test_unknown_quantities_and_normals_are_refused_with_the_list():
    sol = _solution()
    with pytest.raises(ValueError, match="quantity must be one of"):
        plot_electrostatic_slice(sol, 'Hx')
    with pytest.raises(ValueError, match="normal must be"):
        plot_electrostatic_slice(sol, 'phi', normal='w')


def test_a_borrowed_axes_keeps_the_caller_s_title():
    """Embedding in a larger figure must not overwrite what the caller set."""
    sol = _solution()
    fig, ax = plt.subplots()
    ax.set_title('my panel')
    out_fig, out_ax = plot_electrostatic_slice(sol, 'phi', ax=ax)
    assert out_fig is fig and out_ax is ax
    assert ax.get_title() == 'my panel'


def test_drawing_does_not_disturb_the_solution():
    """Plotting is a read: φ and the fields come back identical."""
    sol = _solution()
    phi, Ex = sol.phi.copy(), sol.E[0].copy()
    for quantity in QUANTITIES:
        plt.close(plot_electrostatic_slice(sol, quantity)[0])
    assert np.array_equal(sol.phi, phi)
    assert np.array_equal(sol.E[0], Ex)
