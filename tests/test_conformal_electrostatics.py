"""The 3D electrostatic solver on conformal (Dey–Mittra) cut cells.

The FDTD steps the cut geometry, and :mod:`wavesim.mode_solver` solves the
2D electrostatic problem on it. This is the same substitution carried into the
3D solve, so a model built once describes one conductor to all three.

The substitution is a single one: the open fraction lands on the node-to-node
**distance**, never on the face area. That is the mode solver's derivation from
the conformal Faraday contour (``E·L = Δφ``), and it is also the only form the
stored geometry supports — ``pec_face_open_*`` measures the primary H faces,
which are not the dual-cell faces this operator's control volumes are bounded
by. Every weight is therefore the staircase weight times an open fraction, which
gives the reduction property the first three tests below pin.

Measured on the reference coax (C′ against the analytic
``2πε/ln(b/a)``, air, a = 3 mm, b = 9 mm, all-Neumann box):

    cells across    staircase    conformal
        24           +17.69%       −0.29%
        48            +7.44%       −0.07%
        96            +2.26%       −0.02%

which is the whole point of the commit: the staircase error is first order in h
and dominates everything else in the discretisation, and the cut cells remove
it outright rather than by refinement.
"""

import warnings

import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import EPS0
from wavesim.electrostatics import (Electrostatics, MIN_OPEN_FRACTION,
                                    _face_coefs, _open_lengths)
from wavesim.parts import pec_node_mask

from conformal_shapes import binary_fractions, coax_fractions

# The reference coax: air-filled, a = 3 mm, b = 9 mm, in a 24 mm box.
A_IN, B_OUT, SPAN, NZ = 3.0e-3, 9.0e-3, 24.0e-3, 4


def _coax(n, eps_r=1.0, geometry='conformal'):
    """Named coax at ``n`` cells across the box, cut / staircased / binary.

    ``'staircase'`` carries no fractions at all (the legacy path), ``'binary'``
    expresses that *same* staircased conductor as 0/1 fractions so it runs the
    conformal code over uncut geometry, and ``'conformal'`` is the true analytic
    cut. Returns ``(grid, dz)``.
    """
    ds = SPAN / n
    grid = ws.set_vacuum(ws.create_grid(n, n, NZ, ds, ds, ds))
    c = 0.5 * n * ds
    ws.set_coax(grid, c, c, A_IN, B_OUT, eps_r_fill=eps_r,
                name_inner="core", name_outer="shield")
    if geometry != 'staircase':
        fr = (coax_fractions(grid, c, c, A_IN, B_OUT) if geometry == 'conformal'
              else binary_fractions(grid.pec_mask))
        ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                               grid.mu_x, grid.mu_y, grid.mu_z, **fr)
    return grid, ds


def _solve(grid, boundary='neumann', **kw):
    es = Electrostatics(grid)
    es.set_potential("core", 1.0).set_potential("shield", 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return es.solve(boundary=boundary, method='direct', **kw)


def _capacitance(grid, dz):
    """C per unit length, from the charge on the inner conductor.

    The dual cells of the two end nodes are half cells (see ``_node_dual``), so
    the solved region is ``(Nz-1)·dz`` deep, not ``Nz·dz``.
    """
    return _solve(grid).charge("core") / ((NZ - 1) * dz)


def _analytic(eps_r=1.0):
    return 2 * np.pi * EPS0 * eps_r / np.log(B_OUT / A_IN)


# ---------------------------------------------------------------------- #
# Reduction: the conformal branch has to contain the staircase one
# ---------------------------------------------------------------------- #

def test_open_fractions_of_one_reproduce_the_staircase_coefficients():
    """With nothing cut, every stencil coefficient is the staircase one, bit
    for bit — on a **graded** mesh too, because the open fraction scales the
    existing centre distance rather than replacing it.

    The same ``node_pec`` goes into both so the ε rule is held fixed and the
    geometry weighting is the only thing under test.
    """
    rng = np.random.default_rng(0)
    shape = (7, 5, 4)
    nodes = [np.concatenate([[0.0], np.cumsum(rng.uniform(1e-4, 3e-4, n))])
             for n in shape]
    grid = ws.set_vacuum(ws.create_grid_rectilinear(*nodes))
    for axis in 'xyz':
        getattr(grid, 'eps_' + axis)[...] = rng.uniform(1.0, 4.0, shape)
    node_pec = rng.random(shape) < 0.2

    staircase = _face_coefs(grid, node_pec, _open_lengths(grid)[0])
    ones = {f'pec_{k}_open_{a}': np.ones(shape)
            for k in ('edge', 'face') for a in 'xyz'}
    ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                           grid.mu_x, grid.mu_y, grid.mu_z, **ones)
    opened = _face_coefs(grid, node_pec, _open_lengths(grid)[0])

    for legacy, cut in zip(staircase, opened):
        assert np.array_equal(legacy, cut)


def test_all_open_fractions_describe_empty_space():
    """Fractions of 1.0 everywhere mean no conductor at all.

    The conformal rule reads the conductor off the covered edges and ignores
    ``pec_mask`` entirely, so a coax whose fractions are all open has nothing to
    hold at a potential: every part grounds by default and the grounded box
    leaves φ ≡ 0.
    """
    grid, _ = _coax(24, geometry='staircase')
    ones = {f'pec_{k}_open_{a}': np.ones((grid.Nx, grid.Ny, grid.Nz))
            for k in ('edge', 'face') for a in 'xyz'}
    ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                           grid.mu_x, grid.mu_y, grid.mu_z, **ones)

    assert not pec_node_mask(grid).any()
    sol = _solve(grid, boundary='ground')
    assert np.abs(sol.phi).max() == 0.0


def test_binary_fractions_reproduce_the_staircase_solve():
    """The same conductor by two routes into the solver ⇒ identical everything.

    This is what separates the conformal *code path* from the cut cells: run the
    staircased coax through the open-fraction machinery as 0/1 and φ, the
    charge and the energy have to come back unchanged to the last bit.
    """
    staircase, dz = _coax(24, geometry='staircase')
    binary, _ = _coax(24, geometry='binary')

    assert np.array_equal(pec_node_mask(staircase), pec_node_mask(binary))
    a, b = _solve(staircase), _solve(binary)
    assert np.abs(a.phi - b.phi).max() == 0.0
    assert a.charge("core") == b.charge("core")
    assert a.energy == b.energy


# ---------------------------------------------------------------------- #
# The invariant, and the payoff
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [24, 48])
def test_homogeneous_fill_stays_exact_through_the_cut_cells(n):
    """C scales exactly with a uniform ε_r, cut cells and all.

    The zero-tolerance probe of [[homogeneous-fill-invariant]] in 3D: filling
    the line with ε_r must multiply its capacitance by exactly ε_r whatever the
    conductor geometry is. It survives because the filled and air operators stay
    exact scalar multiples of each other — the open fraction multiplies both by
    the same number, and the one-sided ε rule keeps conductor-adjacent faces
    from averaging in the meaningless permittivity inside the metal. Any leak
    between the two rules shows up here at the 15th digit rather than hiding
    inside a percent of staircase error.
    """
    air = _capacitance(*_coax(n, eps_r=1.0))
    filled = _capacitance(*_coax(n, eps_r=2.3))
    assert filled / air == pytest.approx(2.3, rel=1e-12)


def test_cut_cells_beat_the_staircase_on_the_analytic_coax():
    """The reference case, and the reason the commit exists: −0.29% against
    +17.69% at 24 cells across, where the staircased circle is at its worst."""
    exact = _analytic()
    staircase = abs(_capacitance(*_coax(24, geometry='staircase')) / exact - 1)
    conformal = abs(_capacitance(*_coax(24, geometry='conformal')) / exact - 1)
    assert conformal < 0.01
    assert staircase > 0.15


def test_the_cut_cell_error_converges_faster_than_first_order():
    """Staircase error is O(h); this is not.

    Halving h has to cut the error by more than two — measured 0.29% → 0.07% →
    0.02%, i.e. about h². A first-order result here would mean the fractions
    were being applied somewhere that only shifts the surface rather than
    resolving it.
    """
    exact = _analytic()
    errs = [abs(_capacitance(*_coax(n)) / exact - 1) for n in (24, 48, 96)]
    assert errs[0] / errs[1] > 2.5
    assert errs[1] / errs[2] > 2.5
    assert errs[-1] < 1e-3


def test_both_solvers_agree_on_the_cut_coax_to_round_off():
    """The 2D mode solver and this 3D solve return the *same* conformal C′.

    They share no code: this module assembles a 3D Poisson operator and reads C
    from ``∮ε∇φ·dA`` on the conductor, while :func:`~wavesim.mode_solver.solve_tem_modes`
    assembles a 2D one and reads C from its own quadratic form. Agreement to
    ~1e-15 on cut cells is therefore a real cross-check of both conformal
    weightings — and it is the free test the design called for, since a
    z-invariant structure extruded with Neumann end caps *is* the mode solver's
    cross-section.

    The staircase twin of this is
    ``test_electrostatics.test_both_solvers_agree_on_the_coax_to_round_off``;
    together they pin that the two solvers describe one conductor on both paths.
    """
    grid, dz = _coax(48)
    C_3d = _capacitance(grid, dz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C_2d = ws.solve_tem_modes(grid, normal='z', position=dz)[0].capacitance
    assert C_3d == pytest.approx(C_2d, rel=1e-12)


# ---------------------------------------------------------------------- #
# The field on a cut edge
# ---------------------------------------------------------------------- #

def _half_cut_plate():
    """Parallel plate whose upper electrode is pulled down half a cell.

    Uniform 1 V across 4 z-cells of PEC-to-PEC gap, then the top plate's face is
    moved to the middle of the cell below it by halving the open fraction of the
    Ez edges there. Analytic answer: the drop is spread over 3.5 cells.
    """
    n, ds = 6, 1e-3
    grid = ws.set_vacuum(ws.create_grid(3, 3, n, ds, ds, ds))
    ws.set_box(grid, 0, 3 * ds, 0, 3 * ds, 0, ds, 1.0, pec=True, name="bot")
    ws.set_box(grid, 0, 3 * ds, 0, 3 * ds, 5 * ds, 6 * ds, 1.0, pec=True,
               name="top")
    fr = binary_fractions(grid.pec_mask)
    fr['pec_edge_open_z'][:, :, 4] = 0.5          # top face lowered half a cell
    ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                           grid.mu_x, grid.mu_y, grid.mu_z, **fr)
    return grid, ds


def test_a_cut_edge_moves_the_conductor_surface_by_its_open_fraction():
    """Half-covering the last gap edge puts the drop over 3.5 cells, not 4.

    The sub-cell content of the whole commit, in the one geometry where it is
    exactly solvable: a cut edge is not a smaller coefficient, it is a
    conductor surface at a different place.
    """
    grid, ds = _half_cut_plate()
    es = Electrostatics(grid)
    es.set_potential("bot", 0.0).set_potential("top", 1.0)
    sol = es.solve(boundary='neumann', method='direct')

    phi = sol.phi[1, 1, :]
    Ez = -1.0 / (3.5 * ds)                       # uniform, over 3.5 cells
    for k in range(1, 5):                        # nodes 1..4 span the gap
        assert phi[k] == pytest.approx((k - 1) / 3.5, rel=1e-12)
    # The half-covered edge carries that same uniform field on its open part.
    assert sol.E[2][1, 1, 4] == pytest.approx(Ez, rel=1e-12)
    assert sol.E[2][1, 1, 2] == pytest.approx(Ez, rel=1e-12)


def test_a_fully_covered_edge_carries_no_field():
    """E is identically zero on every edge the FDTD would zero — not small, zero.

    It falls out of the open length being zero there, so nothing has to be
    masked afterwards.
    """
    grid, _ = _coax(24)
    sol = _solve(grid)
    for E, frac in zip(sol.E, (grid.pec_edge_open_x, grid.pec_edge_open_y,
                               grid.pec_edge_open_z)):
        assert np.abs(E[frac == 0.0]).max() == 0.0


def test_the_field_is_the_gradient_over_the_open_length():
    """``E = −Δφ/L_open`` on every live edge — the same divisor the operator's
    coefficients used, so φ, E and the charge are one discretisation rather than
    three."""
    grid, _ = _coax(24)
    sol = _solve(grid)
    live = grid.pec_edge_open_x > 0.0
    live[-1, :, :] = False                       # no node beyond the last
    dphi = np.zeros_like(sol.phi)
    dphi[:-1] = sol.phi[1:] - sol.phi[:-1]
    L = grid.pec_edge_open_x * grid.dxp[:, None, None]
    assert np.allclose(sol.E[0][live], -dphi[live] / L[live], rtol=1e-13)


# ---------------------------------------------------------------------- #
# Slivers
# ---------------------------------------------------------------------- #

def test_a_round_off_open_fraction_is_clamped_rather_than_divided_by():
    """A conductor face on the node ruler voxelises to ~1e-15, not to 0.

    Such an edge is *not* covered by the FDTD's own test, so it stays in the
    operator with a ``1/f`` weight of 1e15 — an infinity in all but name, on a
    matrix row. Clamping the length keeps it finite and still overwhelmingly
    strongly coupled, and says so out loud.
    """
    grid, _ = _half_cut_plate()
    grid.pec_edge_open_z[:, :, 4] = 1e-15

    lengths, n_clamped = _open_lengths(grid)
    assert n_clamped == 9
    assert np.isfinite(lengths[2]).all()
    assert lengths[2][1, 1, 4] == pytest.approx(MIN_OPEN_FRACTION * grid.dzp[4])

    es = Electrostatics(grid)
    es.set_potential("bot", 0.0).set_potential("top", 1.0)
    with pytest.warns(UserWarning, match="clamped"):
        sol = es.solve(boundary='neumann', method='direct')
    assert np.isfinite(sol.phi).all()
    # The clamped edge is a Dirichlet pin in all but name: its two ends sit at
    # the same potential to nine digits, so the plate has effectively grown.
    assert sol.phi[1, 1, 4] == pytest.approx(1.0, abs=1e-8)


def test_an_uncut_grid_reports_no_slivers_and_allocates_no_grids():
    """The staircase path keeps its broadcast widths — three full float grids of
    a constant would cost hundreds of megabytes on a large model to say nothing.
    """
    grid, _ = _coax(24, geometry='staircase')
    lengths, n_clamped = _open_lengths(grid)
    assert n_clamped == 0
    assert [L.shape for L in lengths] == [(grid.Nx, 1, 1), (1, grid.Ny, 1),
                                          (1, 1, grid.Nz)]
