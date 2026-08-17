"""A lumped element stretched across uneven edges must still drive a circuit.

An element spreads its impressed current over every Yee edge its line crosses.
For that to *be* a current rather than a pile of local charge, each edge has to
carry the same current through its own Ampere face — the injection must be
divergence-free along the line, depositing charge only at the two terminals.
Three things have to line up for that on a graded mesh:

1. the quadrature bins each edge by the cell containing the sample, using the
   staggering ``update.py`` actually implements;
2. each edge's weight is its *exact* overlap with the path, not a sub-step
   sampling of it;
3. the injection divides by the **dual-cell** volume the E update integrates
   Ampere's law over, not the primary cell half a cell away.

Get any of them wrong and a uniform grid still looks fine — the three errors are
all identically zero when every cell is the same size — while a graded mesh
leaks a large fraction of the port current onto the element's own interior nodes.

The measurement here is the grid's own discrete Gauss law, and
:func:`test_the_charge_functional_is_the_solvers_own_invariant` establishes that
before anything is concluded from it: the flux functional below is conserved to
round-off by a source-free ``update_E``, which is what pins the dual face areas
it is built from. Nothing about the port's conventions enters.
"""
import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import EPS0
from wavesim.grid import create_grid_rectilinear
from wavesim.update import update_E


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _graded(n, d0, ratio):
    """``n+1`` node coordinates whose cell widths grow geometrically."""
    return np.concatenate([[0.0], np.cumsum(d0 * ratio ** np.arange(n))])


def _graded_grid():
    """A mesh graded on all three axes, by a different ratio on each."""
    return create_grid_rectilinear(_graded(10, 1.0e-3, 1.30),
                                   _graded(10, 1.1e-3, 1.25),
                                   _graded(10, 0.9e-3, 1.35))


def _dual_charge(grid, absolute=False):
    """Gauss flux ``∮ εE·dA`` out of the dual cell around each node.

    The dual cell around node ``(i,j,k)`` spans centre-to-centre on every axis,
    so its ±x faces sit exactly where ``Ex[i-1,j,k]`` and ``Ex[i,j,k]`` live and
    both have area ``ndy[j]·ndz[k]`` — the node-centred dual widths. Nodes on the
    low boundary have no such cell and are left at zero.

    With ``absolute=True`` the six face terms are summed by magnitude instead of
    signed. That is not a charge; it is the size of the quantities the signed sum
    cancels, and so the scale a "this cancels to round-off" claim has to be
    measured against.
    """
    nd = {ax: grid.node_dual_widths(ax) for ax in 'xyz'}
    Dx = EPS0 * grid.eps_x * grid.Ex
    Dy = EPS0 * grid.eps_y * grid.Ey
    Dz = EPS0 * grid.eps_z * grid.Ez
    ax_area = nd['y'][None, :, None] * nd['z'][None, None, :]
    ay_area = nd['x'][:, None, None] * nd['z'][None, None, :]
    az_area = nd['x'][:, None, None] * nd['y'][None, :, None]
    if absolute:
        pair = lambda hi, lo: np.abs(hi) + np.abs(lo)
    else:
        pair = lambda hi, lo: hi - lo
    q = np.zeros_like(Dx)
    q[1:, 1:, 1:] = (ax_area[:, 1:, 1:] * pair(Dx[1:, 1:, 1:], Dx[:-1, 1:, 1:])
                     + ay_area[1:, :, 1:] * pair(Dy[1:, 1:, 1:], Dy[1:, :-1, 1:])
                     + az_area[1:, 1:, :] * pair(Dz[1:, 1:, 1:], Dz[1:, 1:, :-1]))
    return q


# --------------------------------------------------------------------------- #
# The measurement, before anything is measured with it
# --------------------------------------------------------------------------- #

def test_the_charge_functional_is_the_solvers_own_invariant():
    """``update_E`` conserves the flux above exactly, from an arbitrary H field.

    This is the guard on every other test in the module. ``div curl = 0`` holds
    discretely for *one* pairing of face areas with the update's divisors, so the
    functional being invariant is what identifies those areas as the grid's own
    — not a convention chosen to make the port look right. It is also why a
    charge appearing on an interior node is unambiguously the source's doing.
    """
    grid = _graded_grid()
    rng = np.random.default_rng(20260817)
    for comp in ('Hx', 'Hy', 'Hz'):
        setattr(grid, comp, rng.standard_normal(grid.Ex.shape))

    q_before = _dual_charge(grid)
    update_E(grid)
    q_after = _dual_charge(grid)

    # The field starts at rest, so the update is the *only* thing that could put
    # charge anywhere. Measure the residue against the size of the six face terms
    # it has to cancel, not against the (near-zero) result.
    interior = np.s_[1:-1, 1:-1, 1:-1]
    scale = _dual_charge(grid, absolute=True)[interior].max()
    assert scale > 0.0, "the probe field produced no flux to cancel"
    assert np.abs(q_after - q_before)[interior].max() < 1e-12 * scale


# --------------------------------------------------------------------------- #
# What the element does with its current
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("axis, i0, i1", [('x', 2, 6), ('y', 3, 7), ('z', 2, 6)])
def test_element_deposits_charge_only_at_its_terminals(axis, i0, i1):
    """One step of impressed current charges the two terminals and nothing else.

    The line runs node to node along ``axis``, transversely off-centre so the
    grading is felt on all three axes at once. Each interior node it passes
    through must come out neutral: whatever the element pushes into an edge it
    has to take straight out again through the next one.
    """
    grid = _graded_grid()
    nodes = {'x': grid.x, 'y': grid.y, 'z': grid.z}
    other = [a for a in 'xyz' if a != axis]
    fixed = {other[0]: nodes[other[0]][4], other[1]: nodes[other[1]][5]}

    def point(n):
        p = dict(fixed)
        p[axis] = nodes[axis][n]
        return (p['x'], p['y'], p['z'])

    current = 3.0
    src = ws.LineSource(p0=point(i0), p1=point(i1), current=lambda t: current)
    q_before = _dual_charge(grid)
    src.inject(grid, 0.0)
    dq = (_dual_charge(grid) - q_before) / (grid.dt * current)

    term0 = tuple(grid.position_to_index(*point(i0)))
    term1 = tuple(grid.position_to_index(*point(i1)))
    charged = {tuple(n) for n in np.argwhere(np.abs(dq) > 1e-9)}
    assert charged == {term0, term1}, (
        f"charge landed on {sorted(charged)}, expected only the terminals "
        f"{sorted((term0, term1))} — the element is charging its own interior "
        f"nodes instead of driving the current out of its ends")

    # And the terminals hold exactly the charge the port says it delivered —
    # ±I·dt, equal and opposite. Any shortfall is current the circuit never got.
    # ``p0`` is the "+" terminal and positive I leaves it, so it gains the charge.
    assert dq[term0] == pytest.approx(+1.0, abs=1e-9)
    assert dq[term1] == pytest.approx(-1.0, abs=1e-9)


def test_kappa_is_the_series_gap_capacitance_of_the_edges_it_spans():
    """κ = dt/C with C the series combination of each edge's own gap capacitance.

    κ is the port's model of the cell capacitance it bridges, and each edge it
    crosses contributes ``ε·A_dual/l`` in series. On a graded mesh ``A_dual`` is
    built from the node-centred dual widths; using the primary widths instead is
    off by their ratio, which is where a mis-stated κ comes from.
    """
    grid = _graded_grid()
    i, j, k0, k1 = 4, 5, 2, 6
    src = ws.LineSource(p0=(grid.x[i], grid.y[j], grid.z[k0]),
                        p1=(grid.x[i], grid.y[j], grid.z[k1]),
                        resistance=50.0)

    area = grid.node_dual_widths('x')[i] * grid.node_dual_widths('y')[j]
    inv_c = sum(grid.dzp[k] / (EPS0 * area) for k in range(k0, k1))
    assert src.self_coupling(grid) == pytest.approx(grid.dt * inv_c, rel=1e-12)


def test_the_quadrature_lands_one_edge_length_per_edge():
    """A node-to-node path weights each edge it crosses by that edge's length.

    The property everything above rests on, checked directly. A sub-step
    sampling of the path quantises these weights to the sub-step length, which a
    uniform grid absorbs (every edge is a whole number of sub-steps) and a graded
    one does not.
    """
    grid = _graded_grid()
    i, j, k0, k1 = 4, 5, 2, 6
    src = ws.LineSource(p0=(grid.x[i], grid.y[j], grid.z[k0]),
                        p1=(grid.x[i], grid.y[j], grid.z[k1]),
                        resistance=50.0)
    src.self_coupling(grid)                       # compiles the port
    edges = src._port['edges']

    assert set(edges) == {'Ez'}, f"a z-line picked up {sorted(edges)}"
    ii, jj, kk, w, _coef = edges['Ez']
    assert np.array_equal(np.sort(kk), np.arange(k0, k1))
    assert np.all(ii == i) and np.all(jj == j)
    assert w == pytest.approx(grid.dzp[kk], rel=1e-12)


def test_a_uniform_grid_is_the_same_element_it_always_was():
    """The graded-mesh corrections all collapse on a uniform grid.

    Each of the three is a difference between a primary and a dual quantity, so
    the whole fix has to be invisible when the two coincide — the port must land
    on the same edges, with the same weights, as the hand-computed answer.
    """
    ds = 1e-3
    grid = ws.create_grid(12, 12, 12, ds)
    i, j, k0, k1 = 4, 5, 2, 6
    src = ws.LineSource(p0=(i * ds, j * ds, k0 * ds), p1=(i * ds, j * ds, k1 * ds),
                        resistance=50.0)
    kappa = src.self_coupling(grid)
    ii, jj, kk, w, _coef = src._port['edges']['Ez']

    assert np.array_equal(np.sort(kk), np.arange(k0, k1))
    assert w == pytest.approx(np.full(k1 - k0, ds), rel=1e-12)
    # (k1-k0) gap capacitances ε·ds²/ds = ε·ds in series.
    assert kappa == pytest.approx(grid.dt * (k1 - k0) / (EPS0 * ds), rel=1e-12)
