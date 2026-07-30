"""S0 — the primary/dual pairing of the update denominators.

``update_H`` differences E fields that sit on integer *nodes*, so it divides by
the PRIMARY widths; ``update_E`` differences H fields at cell *centres*, so it
divides by the DUAL widths. Getting this backwards costs an order of accuracy on
a graded mesh and is invisible on a uniform one.

The gate is a convergence-order measurement, not a stability check: the leapfrog
amplification matrix sits on the unit circle for *either* pairing (both conserve
energy under some diagonal inner product), so stability cannot discriminate.
"""

import numpy as np
import pytest

from wavesim.constants import C0, MU0, EPS0
from wavesim.grid import create_grid, create_grid_rectilinear
from wavesim.update import update_H, update_E


L = 0.1


def _graded_nodes(N, length=L):
    """Nodes of a smooth grading map, refined at FIXED profile.

    The local ratio → 1 as h → 0, so a consistent scheme must converge at 2nd
    order; a mesh with a *fixed* ratio would let a 1st-order scheme masquerade.
    """
    xi = np.linspace(0.0, 1.0, N + 1)
    return length * (xi + 0.5 * xi ** 2) / 1.5


# ---------------------------------------------------------------------- #
# Operator matrices, extracted from the real update functions
# ---------------------------------------------------------------------- #

def _operators(grid):
    """(M1, M2) = (-ΔH per unit E, +ΔE per unit H) as dense matrices."""
    n = grid.Nx * grid.Ny * grid.Nz
    E_names, H_names = ("Ex", "Ey", "Ez"), ("Hx", "Hy", "Hz")

    def zero():
        for nm in E_names + H_names:
            getattr(grid, nm)[...] = 0.0

    def columns(src_names, step, dst_names, sign):
        M = np.zeros((3 * n, 3 * n))
        for c, nm in enumerate(src_names):
            for idx in range(n):
                zero()
                getattr(grid, nm).reshape(-1)[idx] = 1.0
                step(grid)
                M[:, c * n + idx] = sign * np.concatenate(
                    [getattr(grid, d).reshape(-1) for d in dst_names])
        return M

    M1 = columns(E_names, update_H, H_names, -1.0)
    M2 = columns(H_names, update_E, E_names, +1.0)
    zero()
    return M1, M2


def _amplification_radius(grid):
    M1, M2 = _operators(grid)
    m = M1.shape[0]
    I = np.eye(m)
    A = np.block([[I - M2 @ M1, M2],
                  [-M1,         I]])
    return np.abs(np.linalg.eigvals(A)).max()


# ---------------------------------------------------------------------- #
# The gate: 2nd-order convergence on a graded mesh
# ---------------------------------------------------------------------- #

def _cavity_omega(N, variant):
    """Lowest resonance of a 1D PEC cavity, E on nodes / H on centres.

    Mirrors the z-staggering of the 3D code (Ex at an integer z-node, Hy at a z
    cell centre) as a 1D operator, so the two pairings can be compared directly
    without needing a 3D eigen-decomposition.
    """
    z = _graded_nodes(N)
    dp = np.diff(z)
    dd = 0.5 * (dp[:-1] + dp[1:])

    n = N - 1
    S = np.zeros((n, n))
    for r in range(n):
        k = r + 1
        if variant == 'primary_for_H':
            hi, lo, w = dp[k], dp[k - 1], dd[k - 1]
        else:                                     # dual_for_H — the old pairing
            hi, lo, w = (dd[k] if k < N - 1 else dp[k]), dd[k - 1], dp[k - 1]
        if r + 1 < n:
            S[r, r + 1] = (1.0 / hi) / w
        S[r, r] = -(1.0 / hi + 1.0 / lo) / w
        if r - 1 >= 0:
            S[r, r - 1] = (1.0 / lo) / w
    return C0 * np.sqrt(-np.linalg.eigvals(S).real.max())


def _orders(variant, sizes=(20, 40, 80, 160)):
    exact = np.pi * C0 / L
    errs = [abs(_cavity_omega(N, variant) - exact) / exact for N in sizes]
    return [np.log2(a / b) for a, b in zip(errs[:-1], errs[1:])]


def test_primary_for_H_is_second_order_on_a_graded_mesh():
    orders = _orders('primary_for_H')
    assert min(orders) > 1.9, orders


def test_dual_for_H_is_only_first_order():
    """Pins the regression this fix addresses: the old pairing loses an order."""
    orders = _orders('dual_for_H')
    assert max(orders) < 1.5, orders


def test_update_denominators_use_the_second_order_pairing():
    """The update functions read the arrays the order test says they should."""
    ax = _graded_nodes(6)
    grid = create_grid_rectilinear(ax, ax, ax)
    assert not np.allclose(grid.dxp, grid.dxd)      # the test can actually fail

    # A single Faraday step from one Ez edge: Hx[i,j,k] -= (dt/mu)*ΔEz/d, and the
    # Ez pair straddles the primary cell j, so d must be dyp[j].
    grid.Ez[2, 3, 1] = 1.0
    update_H(grid)
    assert grid.Hx[2, 2, 1] == pytest.approx(-grid.dt / MU0 / grid.dyp[2])

    # A single Ampere step from one Hz face: Hz[i,j,k] sits at yc[j], so the pair
    # Hz[j] - Hz[j-1] spans yc[j] - yc[j-1] = dyd[j-1].
    grid2 = create_grid_rectilinear(ax, ax, ax)
    grid2.Hz[2, 3, 1] = 1.0
    update_E(grid2)
    assert grid2.Ex[2, 3, 1] == pytest.approx(grid2.dt / EPS0 / grid2.dyd[2])


# ---------------------------------------------------------------------- #
# Uniform grids are untouched, and stability is not the discriminator
# ---------------------------------------------------------------------- #

def test_uniform_grid_is_bit_identical_across_the_pairing():
    """dxp == dxd to the last ULP on a uniform grid, so S0 cannot move a result."""
    grid = create_grid(5, 6, 4, 1e-3, 2e-3, 5e-4)
    for p, d in ((grid.dxp, grid.dxd), (grid.dyp, grid.dyd), (grid.dzp, grid.dzd)):
        assert np.array_equal(p, d)


def test_numba_matches_numpy_on_a_graded_mesh():
    """The pairing lives in four places (update, PML, and both of their Numba
    kernels); this is the only test that exercises them together off a uniform
    mesh, where a divergence between them could hide."""
    from wavesim.pml import init_cpml, update_H_pml, update_E_pml
    from wavesim import backend_numba as nb

    def axis(d0, npml=6, ngrade=10):
        # Uniform PML shell | graded interior | uniform PML shell — init_cpml
        # requires constant spacing across the absorbing cells.
        w = np.concatenate([np.full(npml, d0),
                            d0 * 1.15 ** np.arange(1, ngrade + 1),
                            d0 * 1.15 ** np.arange(ngrade, 0, -1),
                            np.full(npml, d0)])
        return np.concatenate([[0.0], np.cumsum(w)])

    def run(numba):
        grid = create_grid_rectilinear(axis(1e-3), axis(1.2e-3), axis(0.8e-3))
        cpml = init_cpml(grid, d_pml=6)
        i, j, k = grid.Nx // 2, grid.Ny // 2, grid.Nz // 2
        for n in range(30):
            (nb.update_H if numba else update_H)(grid)
            (nb.update_H_pml if numba else update_H_pml)(grid, cpml)
            (nb.update_E if numba else update_E)(grid)
            (nb.update_E_pml if numba else update_E_pml)(grid, cpml)
            grid.Ez[i, j, k] += float(np.exp(-((n - 12) / 5.0) ** 2))
        return grid

    ref, got = run(False), run(True)
    assert np.abs(ref.Ez).max() > 1.0                  # the run actually did something
    for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        assert np.array_equal(getattr(ref, name), getattr(got, name)), name


@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "graded"])
def test_scheme_is_stable(uniform):
    ax = _graded_nodes(4)
    grid = create_grid(4, 4, 4, 1e-3) if uniform else \
        create_grid_rectilinear(ax, ax, ax)
    assert _amplification_radius(grid) == pytest.approx(1.0, abs=1e-9)
