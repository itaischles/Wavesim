"""S5b — primary/dual pairing in the mode solver's Laplacian.

S0 found the solver's H update dividing by dual widths where the Faraday
contour says primary. The same inversion was living in ``mode_solver.py``: φ
sits on the Yee **nodes**, so the equation at a node is a flux balance over the
*dual* cell — dual face lengths, primary (node-to-node) coupling distances —
and ``_face_coefs`` had those the other way round.

A uniform mesh hides it exactly (``dxp == dxd`` to the last ULP), which is why
nothing caught it for so long; it only surfaced when S5's conformal derivation
made the node picture explicit. **Every existing test in the suite runs on a
per-axis-uniform mesh, so none of them can see this.** Hence this module.

The gate is sharper than S0's convergence-order test, because a parallel-plate
line has an exact answer on *any* mesh: with a homogeneous fill the field
between the plates is uniform, so φ must be piecewise linear in the plate-normal
coordinate whatever the grading. Getting that wrong is not a truncation error
that shrinks under refinement — it is wrong at every resolution, which makes it
a zero-tolerance assertion rather than an extrapolated slope.
"""

import numpy as np
import pytest

import wavesim as ws
from wavesim.constants import EPS0
from wavesim.grid import create_grid, create_grid_rectilinear
from wavesim.mode_solver import (_fv_energy, _node_dual, _slice,
                                 solve_tem_modes)


def _graded(n, width, amp=0.3):
    """``n`` cells spanning ``width`` with smoothly varying widths (±``amp``).

    ``x(s) = W·(s + amp·sin(2πs)/2π)`` on ``s = i/n``: monotone for amp < 1, and
    every fractional position ``i/n`` is a node at every ``n``, so a conductor
    placed at a fixed fraction is exactly node-aligned under refinement.
    """
    s = np.arange(n + 1) / n
    return width * (s + amp * np.sin(2 * np.pi * s) / (2 * np.pi))


def _parallel_plate(n=40, eps_r=1.0, graded=True, width=8e-3):
    """Two plates normal to y, spanning the full x extent — no corners, no
    fringing, and an exactly uniform field between them.

    The plates are placed at fixed index fractions (0.2 and 0.8 of the node
    range), so the geometry does not drift with ``n``. Solved with
    ``boundary='neumann'``: the plates span the full x extent and so touch the
    domain edge, which a grounded ring would swallow into the reference node,
    leaving no signal conductor. Zero-flux side walls are also what makes the
    problem exactly one-dimensional — no fringing to argue about.
    """
    if graded:
        ax = _graded(n, width)
        grid = create_grid_rectilinear(ax, _graded(n, width), np.arange(4) * 1e-3)
    else:
        grid = create_grid(n, n, 3, width / n, width / n, 1e-3)
    ws.set_vacuum(grid)
    for axis in 'xyz':
        getattr(grid, 'eps_' + axis)[...] = eps_r

    j = np.arange(grid.Ny)[None, :, None]
    lo, hi = int(0.2 * n), int(0.8 * n)
    grid.pec_mask = np.broadcast_to((j <= lo) | (j >= hi),
                                    (grid.Nx, grid.Ny, grid.Nz)).copy()
    return grid, lo, hi


def _solve(grid):
    return solve_tem_modes(grid, normal='z', position=1e-3,
                           boundary='neumann', compute_params=True)[0]


def _legacy_face_coefs(eps_a, eps_b, da_w, db_w, pec=None, f_a=None, f_b=None):
    """The pre-S5b coefficients: dual coupling distance, primary face length.

    Kept here rather than in the solver so the regression is pinned by an
    explicit statement of the thing being ruled out, the way
    ``tests/test_grid_spacing_order.py`` pins the old H-update pairing.
    """
    Na, Nb = eps_a.shape
    dac = 0.5 * (da_w[:-1] + da_w[1:])
    dbc = 0.5 * (db_w[:-1] + db_w[1:])
    la = np.broadcast_to(dac[:, None], (Na - 1, Nb)).astype(float)
    lb = np.broadcast_to(dbc[None, :], (Na, Nb - 1)).astype(float)
    if f_a is not None:
        la, lb = la * f_a[:Na - 1, :], lb * f_b[:, :Nb - 1]

    def eps_face(eps, src, nbr):
        ef = 0.5 * (eps[src] + eps[nbr])
        if pec is not None:
            ef = np.where(pec[nbr] & ~pec[src], eps[src], ef)
            ef = np.where(pec[src] & ~pec[nbr], eps[nbr], ef)
        return ef

    sa, na = np.s_[0:Na - 1, :], np.s_[1:Na, :]
    sb, nb = np.s_[:, 0:Nb - 1], np.s_[:, 1:Nb]
    ca = np.divide(np.broadcast_to(db_w[None, :], la.shape), la,
                   out=np.zeros_like(la), where=la > 0) * eps_face(eps_a, sa, na)
    cb = np.divide(np.broadcast_to(da_w[:, None], lb.shape), lb,
                   out=np.zeros_like(lb), where=lb > 0) * eps_face(eps_b, sb, nb)
    return ca, cb


def _with_legacy_pairing(fn):
    """Run ``fn()`` with the solver assembling the pre-S5b operator."""
    from wavesim import mode_solver as ms
    orig = ms._face_coefs
    try:
        ms._face_coefs = _legacy_face_coefs
        return fn()
    finally:
        ms._face_coefs = orig


def _plate_field(mode, grid, lo, hi):
    """``Δφ / dyp`` across each gap edge — the discrete E, which must be uniform."""
    phi = mode.phi[grid.Nx // 2, :]
    j = np.arange(lo, hi)
    return -(phi[j + 1] - phi[j]) / grid.dyp[j]


# ---------------------------------------------------------------------- #
# The exact case
# ---------------------------------------------------------------------- #

def test_graded_mesh_parallel_plate_field_is_exactly_uniform():
    """The whole gate, in one assertion.

    ``∇·(ε∇φ) = 0`` between parallel plates with a homogeneous fill has a
    uniform field, and a node-centred flux balance reproduces it on any mesh:
    the flux through every face is equal, so ``Δφ_j ∝ dyp[j]`` and ``E`` comes
    out constant to round-off. With the pre-S5b pairing ``Δφ_j ∝ dyd[j]``
    instead, so E wobbles with the grading — a first-order error present at
    every resolution, not a truncation term that shrinks under refinement.
    """
    grid, lo, hi = _parallel_plate()
    E = _plate_field(_solve(grid), grid, lo, hi)
    assert np.ptp(E) / np.abs(E).mean() < 1e-12, (
        f"E varies by {100 * np.ptp(E) / np.abs(E).mean():.3f}% across a graded "
        f"mesh; the field between parallel plates is uniform")


def test_the_old_pairing_fails_that_test():
    """Pins the regression so it cannot come back silently, the way
    ``test_grid_spacing_order.py`` pins the old H-update pairing."""
    grid, lo, hi = _parallel_plate()
    E = _plate_field(_with_legacy_pairing(lambda: _solve(grid)), grid, lo, hi)
    assert np.ptp(E) / np.abs(E).mean() > 0.02, (
        "the legacy pairing no longer misbehaves — has the test geometry "
        "stopped being graded?")


@pytest.mark.parametrize("eps_r", [1.0, 4.0])
def test_graded_mesh_capacitance_is_the_analytic_value(eps_r):
    """C = ε₀ε_r·W/d exactly, with ``W`` the dual measure of the transverse
    extent (``_node_dual(dxp).sum()``, i.e. x[0] → xc[-1]) and ``d`` the plate
    separation. There is no fringing to argue about: the plates span the full
    extent and the side walls carry zero flux.

    Asserted on :func:`_fv_energy`, the operator's own quadratic form, because
    that is what S5b changed. ``mode.capacitance`` on the *staircase* path still
    comes from the older collocated ``np.gradient`` integral, which reads 3.9%
    low on this case — a known and deliberate hold-over (S5 left it alone so
    every recorded staircase number stays bit-identical), and the same effect
    the conformal work's "binary" column quantified when switching to this
    integral took the reference coax's Z₀ error from +14.4% to +6.9%.
    """
    grid, lo, hi = _parallel_plate(eps_r=eps_r)
    mode = _solve(grid)
    C = _fv_energy(mode.phi, _slice(grid.eps_x, 'z', 1), _slice(grid.eps_y, 'z', 1),
                   grid.dxp, grid.dyp, mode.pec, None, None)
    d = grid.y[hi] - grid.y[lo]
    W = _node_dual(grid.dxp).sum()
    assert C == pytest.approx(EPS0 * eps_r * W / d, rel=1e-12)
    assert mode.eps_eff == pytest.approx(eps_r, rel=1e-12)


# ---------------------------------------------------------------------- #
# Nothing already recorded may move
# ---------------------------------------------------------------------- #

def _shielded_line(n=24, ds=2.5e-4):
    """Square inner conductor inside the grounded ring — a ``boundary='ground'``
    structure on a uniform mesh, i.e. the shape of every recorded result."""
    grid = create_grid(n, n, 3, ds)
    ws.set_vacuum(grid)
    idx = np.arange(n)
    inner = (np.abs(idx - n // 2) <= 3)
    grid.pec_mask = np.broadcast_to((inner[:, None] & inner[None, :])[:, :, None],
                                    (n, n, 3)).copy()
    return grid


def test_uniform_mesh_reproduces_the_old_pairing_bit_for_bit():
    """``dxp == dxd`` exactly on a uniform grid, so the re-pairing is a no-op
    there. That is both why the bug survived undetected and why no recorded
    result moves — asserted against the legacy coefficients themselves rather
    than argued from the spacing arrays.

    ``boundary='ground'`` (the default, and what every recorded case uses) also
    pins the whole edge ring, so the one place the two genuinely differ on a
    uniform mesh — the half dual cell a *free* boundary node owns — is never
    assembled.
    """
    grid = _shielded_line()
    assert np.array_equal(grid.dxp, grid.dxd)
    def solve(g):
        return solve_tem_modes(g, normal='z', position=g.zc[1],
                               compute_params=True)[0]

    new = solve(grid)
    old = _with_legacy_pairing(lambda: solve(_shielded_line()))
    assert np.array_equal(new.phi, old.phi)
    assert new.capacitance == old.capacitance
    assert new.impedance == old.impedance


def test_node_dual_matches_the_grids_own_dual_widths():
    """``_node_dual`` recomputes ``dxd`` shifted by one, because the solver only
    receives the primary widths of its (possibly sub-rectangular) solve region.
    Interior entries must agree with the grid exactly."""
    ax = _graded(24, 6e-3)
    grid = create_grid_rectilinear(ax, ax, np.arange(4) * 1e-3)
    wd = _node_dual(grid.dxp)
    assert np.array_equal(wd[1:], grid.dxd[:-1])
    assert wd[0] == 0.5 * grid.dxp[0]
