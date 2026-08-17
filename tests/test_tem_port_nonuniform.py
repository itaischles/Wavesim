"""A modal port stretched over graded cells must deliver the current it reports.

``TEMPort`` hands a circuit a single ``(V*, I)`` pair and turns ``I`` into a
sheet of impressed transverse current, ``E += κ·Ê·I``. For that to *be* a
current, the charge it lands on the signal conductor each step has to be
``I·dt`` — no more, no less. What decides it is the volume ``κ = dt/(ε₀·Sv)``
is built from.

``Sv = Σ dV·ε_r·Ê²`` looks like an energy and is really a charge, but only
through a summation by parts::

    Σ dV·ε_r·Ê²  =  Σ (ε_r Ê·A_dual)·(Ê·L_prim)  =  Σ_nodes φ·(net flux)  =  Q

which holds on three conditions: ``dV`` factorises into the Ampere face area
``A_dual`` times the edge length ``L_prim``; ``Ê·L_prim = −Δφ``; and the
transverse divergence taken with those same ``A_dual`` vanishes at every free
node. The middle one is :meth:`TEMMode._staggered_port_fields`, the last is what
:func:`~wavesim.mode_solver._face_coefs` solves φ for. The first is this module:
writing ``dV`` as three primary widths (which it was) leaves the identity broken
by ``A_primary/A_dual`` per component, and the port then deposits
``I·(Sv_dual/Sv_primary)`` while telling the circuit ``I``.

A uniform grid hides all of it — ``dxp == dxd`` to the last ULP — which is why
the port's own suite (``test_modal_port*``, ``test_tem_port_impedance``,
``test_directional_launch``) never saw it.

As in ``tests/test_lumped_nonuniform.py``, the measurement is the grid's own
discrete Gauss law and :func:`test_the_charge_functional_is_the_solvers_own_invariant`
establishes it before anything is concluded from it.
"""
import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import EPS0
from wavesim.grid import create_grid_rectilinear
from wavesim.mode_solver import solve_tem_modes
from wavesim.parts import conductor_bodies
from wavesim.pec import conformal_edge_eps
from wavesim.update import update_E

from conformal_shapes import coax_fractions

A_IN, B_OUT, HALF = 3.0e-3, 7.0e-3, 9.0e-3
N_XY, N_Z = 24, 6


# --------------------------------------------------------------------------- #
# Geometry: one coax, reachable as a staircase or as true cut cells
# --------------------------------------------------------------------------- #

def _graded_axis(n, span, ratio):
    """``n+1`` node coordinates spanning ``span``, widths growing by ``ratio``."""
    w = ratio ** np.arange(n)
    w *= span / w.sum()
    return np.concatenate([[0.0], np.cumsum(w)])


def _coax(ratio, conformal):
    """Coax on a mesh graded by ``ratio`` per cell on all three axes.

    A constant ratio is deliberate: ``dp[i]/nd[i] = 2r/(1+r)`` is then the same
    at every index, so the whole error collapses to one number the algebra can
    be checked against, independent of where the conductor happens to fall.
    """
    x = _graded_axis(N_XY, 2 * HALF, ratio)
    z = _graded_axis(N_Z, N_Z * 1.0e-3, ratio)
    grid = create_grid_rectilinear(x, x.copy(), z)
    ws.set_vacuum(grid)
    c = HALF
    r2 = ((grid.xc[:, None, None] - c) ** 2 + (grid.yc[None, :, None] - c) ** 2)
    grid.pec_mask = np.broadcast_to((r2 < A_IN ** 2) | (r2 > B_OUT ** 2),
                                    (N_XY, N_XY, N_Z)).copy()
    if conformal:
        ws.set_material_arrays(grid, grid.eps_x, grid.eps_y, grid.eps_z,
                               grid.mu_x, grid.mu_y, grid.mu_z,
                               **coax_fractions(grid, c, c, A_IN, B_OUT))
    return grid


# --------------------------------------------------------------------------- #
# The measurement, before anything is measured with it
# --------------------------------------------------------------------------- #

def _dual_charge(grid, absolute=False):
    """Gauss flux ``∮ εE·dA`` out of the dual cell around each node.

    The ±x faces of node ``(i,j,k)``'s dual cell sit exactly where ``Ex[i-1]``
    and ``Ex[i]`` live, both of area ``ndy[j]·ndz[k]``. ``ε`` is the one
    ``update_E`` actually divides by, which on a cut-cell grid is not the stored
    array (:func:`wavesim.pec.conformal_edge_eps`).

    With ``absolute=True`` the six face terms are summed by magnitude — not a
    charge, but the scale a "this cancels to round-off" claim is measured
    against.
    """
    ex, ey, ez = conformal_edge_eps(grid)
    nd = {ax: grid.node_dual_widths(ax) for ax in 'xyz'}
    D = (EPS0 * ex * grid.Ex, EPS0 * ey * grid.Ey, EPS0 * ez * grid.Ez)
    area = (nd['y'][None, :, None] * nd['z'][None, None, :],
            nd['x'][:, None, None] * nd['z'][None, None, :],
            nd['x'][:, None, None] * nd['y'][None, :, None])
    pair = ((lambda hi, lo: np.abs(hi) + np.abs(lo)) if absolute
            else (lambda hi, lo: hi - lo))
    q = np.zeros_like(D[0])
    q[1:, 1:, 1:] = (
        area[0][:, 1:, 1:] * pair(D[0][1:, 1:, 1:], D[0][:-1, 1:, 1:])
        + area[1][1:, :, 1:] * pair(D[1][1:, 1:, 1:], D[1][1:, :-1, 1:])
        + area[2][1:, 1:, :] * pair(D[2][1:, 1:, 1:], D[2][1:, 1:, :-1]))
    return q


@pytest.mark.parametrize("conformal", [False, True])
def test_the_charge_functional_is_the_solvers_own_invariant(conformal):
    """``update_E`` conserves the flux above exactly, from an arbitrary H field.

    The guard on every other test here. ``div curl = 0`` holds discretely for
    *one* pairing of face areas with the update's divisors, so the functional
    being invariant is what identifies those areas as the grid's own rather than
    a convention chosen to make the port look right — and it is why charge
    appearing anywhere afterwards is unambiguously the port's doing. Run on the
    cut-cell path too, since that is where the port's ``f_open`` weighting has
    to keep meaning the same thing.
    """
    grid = _coax(1.05, conformal)
    rng = np.random.default_rng(20260817)
    for comp in ('Hx', 'Hy', 'Hz'):
        setattr(grid, comp, rng.standard_normal(grid.Ex.shape))

    q_before = _dual_charge(grid)
    update_E(grid)
    q_after = _dual_charge(grid)

    interior = np.s_[1:-1, 1:-1, 1:-1]
    scale = _dual_charge(grid, absolute=True)[interior].max()
    assert scale > 0.0, "the probe field produced no flux to cancel"
    assert np.abs(q_after - q_before)[interior].max() < 1e-12 * scale


# --------------------------------------------------------------------------- #
# What the port does with its current
# --------------------------------------------------------------------------- #

def _inject(grid, ratio, conformal, current=1.0):
    """Drive the port with ``current`` for one step; return ``(dq/(I·dt), …)``.

    ``E += coef·I`` is the whole of what the port's E sheet does to the grid;
    the time centring and the circuit solve above it only decide what ``I`` is.
    """
    modes = solve_tem_modes(grid, normal='z', position=grid.z[3],
                            compute_params=True)
    assert modes, "no TEM mode on the plane"
    mode = modes[0]
    kernel = mode.build_port_kernel(grid, directional=False)
    for comp, (ii, jj, kk, _w, coef) in kernel['edges'].items():
        getattr(grid, comp)[ii, jj, kk] += coef * current
    return _dual_charge(grid) / (grid.dt * current), mode


def _bodies(grid, k):
    """``(signal, ground, free)`` node masks.

    The conductor the FDTD really has is one node wider than ``pec_mask`` — the
    edge masks zero a node's tangential edges a cell before its own cell is
    metal — so the split is taken from :func:`wavesim.parts.conductor_bodies`,
    the connected components of the zeroed-edge graph. Getting this wrong reads
    surface charge as a divergence error.
    """
    labels, n = conductor_bodies(grid)
    on_edge = set(np.unique(np.concatenate([
        labels[0, :, k], labels[-1, :, k], labels[:, 0, k], labels[:, -1, k]])))
    inner = [lbl for lbl in range(1, n + 1) if lbl not in on_edge]
    assert len(inner) == 1, f"expected one inner conductor, got {inner}"
    free = labels == 0
    free[0] = free[:, 0] = free[:, :, 0] = False        # no dual cell there
    return labels == inner[0], (labels > 0) & (labels != inner[0]), free


@pytest.mark.parametrize("conformal", [False, True])
@pytest.mark.parametrize("ratio", [1.0, 1.05])
def test_the_port_deposits_the_current_it_reports(ratio, conformal):
    """One step of modal drive charges the two conductors by ``±I·dt``, exactly.

    The signal conductor gains it, the shield loses it, and no free node in the
    dielectric holds anything — the injected sheet is ``(ε₀κ/dt)·ε_r Ê``, a
    scalar multiple of the mode's own ``D̂``, so its divergence is the mode's and
    lands only where the mode terminates.

    Against the parent commit the graded cases read 0.9529 instead of 1: 4.7% of
    the port current simply not delivered, at a grading of 1.05 per cell.
    """
    grid = _coax(ratio, conformal)
    dq, _mode = _inject(grid, ratio, conformal)
    sig, gnd, free = _bodies(grid, 3)

    scale = np.abs(dq).max()
    assert np.abs(dq[free]).max() < 1e-12 * scale, (
        "the modal sheet charged the dielectric — its Ê is not a null vector "
        "of the transverse divergence the FDTD takes")
    assert dq[sig].sum() == pytest.approx(+1.0, abs=1e-9), (
        f"the port deposited {dq[sig].sum():.6f}·I·dt on the signal conductor "
        f"while reporting I to the circuit")
    assert dq[gnd].sum() == pytest.approx(-1.0, abs=1e-9)


def test_kappa_is_off_by_the_face_ratio_when_the_volume_is_all_primary():
    """The error has the size the face areas say it does, not an incidental one.

    With a constant grading ratio ``r`` every edge has
    ``dp/nd = 2r/(1+r)``, and each ``dV`` carries two such factors (the two
    transverse widths), so the all-primary ``Sv`` overstates the dual one by
    ``(2r/(1+r))²`` whatever the cross-section — the same number for the
    staircase and the cut-cell coax. This pins the correction as geometric
    rather than fitted.
    """
    r = 1.05
    expect = (2 * r / (1 + r)) ** 2
    for conformal in (False, True):
        grid = _coax(r, conformal)
        mode = solve_tem_modes(grid, normal='z', position=grid.z[3],
                               compute_params=True)[0]
        kappa = mode.build_port_kernel(grid, directional=False)['kappa']
        # κ = dt/(ε₀·Sv); an Sv too large by `expect` makes κ too small by it.
        assert _kappa_all_primary(mode, grid) == pytest.approx(
            kappa / expect, rel=1e-12)


def _kappa_all_primary(mode, grid):
    """κ as the parent commit built it — three primary widths per volume."""
    from wavesim.mode_solver import (_NORMAL_CFG, _eps_by_component,
                                     _plane_open_fractions, _plane_to_grid)
    cfg = _NORMAL_CFG[mode.normal]
    eps_of = _eps_by_component(grid)
    k = mode.slice_index
    E_stag, _H = mode._staggered_port_fields(grid)
    f_open = dict(zip(cfg['E'],
                      _plane_open_fractions(grid, cfg, mode.normal, k)))
    sv = 0.0
    for comp in cfg['E']:
        a, b = np.nonzero(E_stag[comp])
        if a.size == 0:
            continue
        ii, jj, kk = _plane_to_grid(mode.normal, k, a, b)
        dV = grid.dxp[ii] * grid.dyp[jj] * grid.dzp[kk]
        if f_open[comp] is not None:
            dV = dV * f_open[comp][a, b]
        sv += float(np.sum(dV * eps_of[comp][ii, jj, kk]
                           * E_stag[comp][a, b] ** 2))
    return grid.dt / (EPS0 * sv)


@pytest.mark.parametrize("conformal", [False, True])
@pytest.mark.parametrize("ratio", [1.0, 1.05])
def test_kappa_is_dt_over_the_capacitance_the_mode_solver_reports(ratio,
                                                                 conformal):
    """``ε₀·Sv`` is the mode's own ``C`` over the slab the sheet sits in.

    The port and the mode solver arrive at that capacitance by different routes
    — one sums ``dV·ε_r·Ê²`` over edges, the other takes the quadratic form
    ``−φᵀAφ`` of the operator (:func:`~wavesim.mode_solver._fv_energy`) — and
    they agree only if both weight an edge by ``A_dual·L_prim``. ``_fv_energy``
    always did, being the operator itself; the port did not, so the two
    disagreed by the face ratio on any graded mesh while looking identical on a
    uniform one. This is the closing of that gap, and the reason κ can be quoted
    as ``dt/C``.
    """
    grid = _coax(ratio, conformal)
    mode = solve_tem_modes(grid, normal='z', position=grid.z[3],
                           compute_params=True)[0]
    kappa = mode.build_port_kernel(grid, directional=False)['kappa']
    slab = mode.capacitance * grid.node_dual_widths('z')[mode.slice_index]
    assert kappa == pytest.approx(grid.dt / slab, rel=1e-12)


def test_a_uniform_grid_is_the_port_it_always_was():
    """The correction is a primary-vs-dual difference, so it must vanish exactly
    where the two coincide — bit for bit, not to a tolerance."""
    grid = _coax(1.0, conformal=False)
    mode = solve_tem_modes(grid, normal='z', position=grid.z[3],
                           compute_params=True)[0]
    kernel = mode.build_port_kernel(grid, directional=False)
    assert kernel['kappa'] == _kappa_all_primary(mode, grid)
