"""
backend_numba.py — Numba-accelerated, multithreaded drop-in for the hot solver
functions (ROADMAP §3).

Why this exists
---------------
The pure-NumPy curl/CPML updates in update.py / pml.py are vectorised but
*single-threaded*: at representative 3D sizes the solver pegs ~1 of the machine's
cores (<10% utilisation). This module reimplements the four hot functions
(`update_H`, `update_E`, `update_H_pml`, `update_E_pml`) as explicit-loop
`@njit(parallel=True)` kernels that:

  * keep the exact Yee staggering, signs, and physical coefficients of the NumPy
    reference (it remains the validation oracle),
  * mutate the SAME NumPy arrays in place, so the wrappers are signature-compatible
    with their numpy.py / pml.py counterparts and `Simulation(backend='numba')`
    swaps them in transparently,
  * parallelise the O(N³) volume work across all cores with `prange`.

Faithfulness to the NumPy reference
-----------------------------------
Every loop nest below mirrors one NumPy slice assignment from update.py / pml.py,
with the slice bounds turned into explicit `range(...)` limits:

    grid.Hx[:, :-1, :-1] -= coef * (dEz_dy[:, :, :-1] - dEy_dz[:, :-1, :])

becomes a `prange(Nx) × range(Ny-1) × range(Nz-1)` loop computing the same
per-cell expression. A single 3D kernel subsumes the `Nz=1` fast path via an
`if Nz > 1` branch, exactly reproducing which derivative terms the slice form
drops on a 2D slice — so results are bit-identical (no parallel reductions, just
independent per-cell writes).

CPML psi state
--------------
The psi arrays use the SAME boundary-slab layout as pml.py (compressed along their
derivative axis to the active PML indices `sel_*`). The kernels advance them in
place — `psi = b*psi + c*dF` — and add the `dt/(MU0·mu)` / `dt/(EPS0·eps)`-scaled
correction onto the field over the same sub-range the NumPy code touches. The psi
recursion runs over the full orthogonal extent (matching NumPy) while the field
correction is restricted to the interior slice, so the carried state matches step
for step.

Parallel-write safety: each `prange` loop is over a full axis that indexes a
DISTINCT field/psi plane per iteration (no two threads write the same cell), so
`parallel=True` is race-free without atomics.
"""

import numpy as np
from numba import njit, prange

from wavesim.grid import FDTDGrid
from wavesim.pml import CPMLArrays
from wavesim.constants import MU0, EPS0

# Numba folds module-level globals into the compiled kernels as constants.
_MU0 = MU0
_EPS0 = EPS0

_NJIT = dict(parallel=True, fastmath=False, cache=True)


# ====================================================================== #
# Field updates (curl)
# ====================================================================== #
@njit(**_NJIT)
def _update_H(Ex, Ey, Ez, Hx, Hy, Hz,
              mu_x, mu_y, mu_z, dt, dxd, dyd, dzd, Nx, Ny, Nz):
    # Every H derivative differences a cell-center E field, so it is divided by the
    # DUAL width along the differenced axis, indexed by the loop counter of that
    # axis (dxd[i]/dyd[j]/dzd[k]). Mirrors update.py's dyd[:-1] broadcast: the
    # output cell index equals the diff's lower index. On a uniform grid the dual
    # arrays are the exact constant spacing, so this is bit-identical to the old
    # scalar divisor.
    if Nz > 1:
        # Hx[:, :-1, :-1] -= coef * (dEz/dy - dEy/dz)
        for i in prange(Nx):
            for j in range(Ny - 1):
                for k in range(Nz - 1):
                    dEz_dy = (Ez[i, j + 1, k] - Ez[i, j, k]) / dyd[j]
                    dEy_dz = (Ey[i, j, k + 1] - Ey[i, j, k]) / dzd[k]
                    Hx[i, j, k] -= (dt / (_MU0 * mu_x[i, j, k])) * (dEz_dy - dEy_dz)

        # Hy[:-1, :, :-1] -= coef * (dEx/dz - dEz/dx)
        for i in prange(Nx - 1):
            for j in range(Ny):
                for k in range(Nz - 1):
                    dEx_dz = (Ex[i, j, k + 1] - Ex[i, j, k]) / dzd[k]
                    dEz_dx = (Ez[i + 1, j, k] - Ez[i, j, k]) / dxd[i]
                    Hy[i, j, k] -= (dt / (_MU0 * mu_y[i, j, k])) * (dEx_dz - dEz_dx)
    else:
        # Nz=1 slice fast path (z-derivatives dropped)
        for i in prange(Nx):
            for j in range(Ny - 1):
                dEz_dy = (Ez[i, j + 1, 0] - Ez[i, j, 0]) / dyd[j]
                Hx[i, j, 0] -= (dt / (_MU0 * mu_x[i, j, 0])) * dEz_dy
        for i in prange(Nx - 1):
            for j in range(Ny):
                dEz_dx = (Ez[i + 1, j, 0] - Ez[i, j, 0]) / dxd[i]
                Hy[i, j, 0] -= (dt / (_MU0 * mu_y[i, j, 0])) * (-dEz_dx)

    # Hz[:-1, :-1, :] -= coef * (dEy/dx - dEx/dy)   (identical for all Nz)
    for i in prange(Nx - 1):
        for j in range(Ny - 1):
            for k in range(Nz):
                dEy_dx = (Ey[i + 1, j, k] - Ey[i, j, k]) / dxd[i]
                dEx_dy = (Ex[i, j + 1, k] - Ex[i, j, k]) / dyd[j]
                Hz[i, j, k] -= (dt / (_MU0 * mu_z[i, j, k])) * (dEy_dx - dEx_dy)


@njit(**_NJIT)
def _update_E(Ex, Ey, Ez, Hx, Hy, Hz,
              eps_x, eps_y, eps_z, dt, dxp, dyp, dzp, Nx, Ny, Nz):
    # Every E derivative differences an integer-node H field, so it is divided by
    # the PRIMARY width along the differenced axis. The updated cell index is the
    # diff's UPPER index (Hz[j]-Hz[j-1] at cell j), matching update.py's dyp[:-1]
    # broadcast where output index n uses dp[n]: here cell j uses dyp[j-1]. Uniform
    # grids keep the exact constant spacing → bit-identical to the scalar divisor.
    if Nz > 1:
        # Ex[:, 1:, 1:] += coef * (dHz/dy - dHy/dz)
        for i in prange(Nx):
            for j in range(1, Ny):
                for k in range(1, Nz):
                    dHz_dy = (Hz[i, j, k] - Hz[i, j - 1, k]) / dyp[j - 1]
                    dHy_dz = (Hy[i, j, k] - Hy[i, j, k - 1]) / dzp[k - 1]
                    Ex[i, j, k] += (dt / (_EPS0 * eps_x[i, j, k])) * (dHz_dy - dHy_dz)

        # Ey[1:, :, 1:] += coef * (dHx/dz - dHz/dx)
        for i in prange(1, Nx):
            for j in range(Ny):
                for k in range(1, Nz):
                    dHx_dz = (Hx[i, j, k] - Hx[i, j, k - 1]) / dzp[k - 1]
                    dHz_dx = (Hz[i, j, k] - Hz[i - 1, j, k]) / dxp[i - 1]
                    Ey[i, j, k] += (dt / (_EPS0 * eps_y[i, j, k])) * (dHx_dz - dHz_dx)
    else:
        # Nz=1 slice fast path
        for i in prange(Nx):
            for j in range(1, Ny):
                dHz_dy = (Hz[i, j, 0] - Hz[i, j - 1, 0]) / dyp[j - 1]
                Ex[i, j, 0] += (dt / (_EPS0 * eps_x[i, j, 0])) * dHz_dy
        for i in prange(1, Nx):
            for j in range(Ny):
                dHz_dx = (Hz[i, j, 0] - Hz[i - 1, j, 0]) / dxp[i - 1]
                Ey[i, j, 0] += (dt / (_EPS0 * eps_y[i, j, 0])) * (-dHz_dx)

    # Ez[1:, 1:, :] += coef * (dHy/dx - dHx/dy)   (identical for all Nz)
    for i in prange(1, Nx):
        for j in range(1, Ny):
            for k in range(Nz):
                dHy_dx = (Hy[i, j, k] - Hy[i - 1, j, k]) / dxp[i - 1]
                dHx_dy = (Hx[i, j, k] - Hx[i, j - 1, k]) / dyp[j - 1]
                Ez[i, j, k] += (dt / (_EPS0 * eps_z[i, j, k])) * (dHy_dx - dHx_dy)


# ====================================================================== #
# CPML corrections
# ====================================================================== #
@njit(**_NJIT)
def _update_H_pml(Ex, Ey, Ez, Hx, Hy, Hz, mu_x, mu_y, mu_z,
                  dt, dxd, dyd, dzd, Nx, Ny, Nz,
                  sx, sy, sz, bxH, cxH, byH, cyH, bzH, czH,
                  psi_Ez_y, psi_Ey_z, psi_Ex_z, psi_Ez_x, psi_Ey_x, psi_Ex_y):
    # dxd/dyd/dzd are the DUAL widths sampled at the slab indices (per-slab, in
    # sel_*H order), so they index by the slab counter p — matching update_H's
    # per-cell dual divisor. On a uniform grid they are the constant PML spacing.
    n_xH = sx.shape[0]
    n_yH = sy.shape[0]
    n_zH = sz.shape[0]

    # ---------- Hx: -psi_Ez_y (y axis), +psi_Ey_z (z axis) ----------
    # psi_Ez_y[i, p, k], j = sy[p]; recursion over full (i, k); correct Hx[:, sy, :-1]
    for i in prange(Nx):
        for p in range(n_yH):
            j = sy[p]
            for k in range(Nz):
                dEz_dy = (Ez[i, j + 1, k] - Ez[i, j, k]) / dyd[p]
                psi_Ez_y[i, p, k] = byH[p] * psi_Ez_y[i, p, k] + cyH[p] * dEz_dy
    if Nz > 1:
        for i in prange(Nx):
            for p in range(n_yH):
                j = sy[p]
                for k in range(Nz - 1):
                    Hx[i, j, k] -= (dt / (_MU0 * mu_x[i, j, k])) * psi_Ez_y[i, p, k]
        # psi_Ey_z[i, j, p], k = sz[p]; correct Hx[:, :-1, sz]
        for i in prange(Nx):
            for j in range(Ny):
                for p in range(n_zH):
                    k = sz[p]
                    dEy_dz = (Ey[i, j, k + 1] - Ey[i, j, k]) / dzd[p]
                    psi_Ey_z[i, j, p] = bzH[p] * psi_Ey_z[i, j, p] + czH[p] * dEy_dz
        for i in prange(Nx):
            for j in range(Ny - 1):
                for p in range(n_zH):
                    k = sz[p]
                    Hx[i, j, k] += (dt / (_MU0 * mu_x[i, j, k])) * psi_Ey_z[i, j, p]
    else:
        for i in prange(Nx):
            for p in range(n_yH):
                j = sy[p]
                Hx[i, j, 0] -= (dt / (_MU0 * mu_x[i, j, 0])) * psi_Ez_y[i, p, 0]

    # ---------- Hy: +psi_Ez_x (x axis), -psi_Ex_z (z axis) ----------
    # psi_Ez_x[p, j, k], i = sx[p]; correct Hy[sx, :, :-1]
    for j in prange(Ny):
        for p in range(n_xH):
            i = sx[p]
            for k in range(Nz):
                dEz_dx = (Ez[i + 1, j, k] - Ez[i, j, k]) / dxd[p]
                psi_Ez_x[p, j, k] = bxH[p] * psi_Ez_x[p, j, k] + cxH[p] * dEz_dx
    if Nz > 1:
        for j in prange(Ny):
            for p in range(n_xH):
                i = sx[p]
                for k in range(Nz - 1):
                    Hy[i, j, k] += (dt / (_MU0 * mu_y[i, j, k])) * psi_Ez_x[p, j, k]
        # psi_Ex_z[i, j, p], k = sz[p]; correct Hy[:-1, :, sz]
        for i in prange(Nx):
            for j in range(Ny):
                for p in range(n_zH):
                    k = sz[p]
                    dEx_dz = (Ex[i, j, k + 1] - Ex[i, j, k]) / dzd[p]
                    psi_Ex_z[i, j, p] = bzH[p] * psi_Ex_z[i, j, p] + czH[p] * dEx_dz
        for i in prange(Nx - 1):
            for j in range(Ny):
                for p in range(n_zH):
                    k = sz[p]
                    Hy[i, j, k] -= (dt / (_MU0 * mu_y[i, j, k])) * psi_Ex_z[i, j, p]
    else:
        for j in prange(Ny):
            for p in range(n_xH):
                i = sx[p]
                Hy[i, j, 0] += (dt / (_MU0 * mu_y[i, j, 0])) * psi_Ez_x[p, j, 0]

    # ---------- Hz: -psi_Ey_x (x axis), +psi_Ex_y (y axis) ----------
    # psi_Ey_x[p, j, k], i = sx[p]; correct Hz[sx, :-1, :]
    for j in prange(Ny):
        for p in range(n_xH):
            i = sx[p]
            for k in range(Nz):
                dEy_dx = (Ey[i + 1, j, k] - Ey[i, j, k]) / dxd[p]
                psi_Ey_x[p, j, k] = bxH[p] * psi_Ey_x[p, j, k] + cxH[p] * dEy_dx
    for j in prange(Ny - 1):
        for p in range(n_xH):
            i = sx[p]
            for k in range(Nz):
                Hz[i, j, k] -= (dt / (_MU0 * mu_z[i, j, k])) * psi_Ey_x[p, j, k]
    # psi_Ex_y[i, p, k], j = sy[p]; correct Hz[:-1, sy, :]
    for i in prange(Nx):
        for p in range(n_yH):
            j = sy[p]
            for k in range(Nz):
                dEx_dy = (Ex[i, j + 1, k] - Ex[i, j, k]) / dyd[p]
                psi_Ex_y[i, p, k] = byH[p] * psi_Ex_y[i, p, k] + cyH[p] * dEx_dy
    for i in prange(Nx - 1):
        for p in range(n_yH):
            j = sy[p]
            for k in range(Nz):
                Hz[i, j, k] += (dt / (_MU0 * mu_z[i, j, k])) * psi_Ex_y[i, p, k]


@njit(**_NJIT)
def _update_E_pml(Ex, Ey, Ez, Hx, Hy, Hz, eps_x, eps_y, eps_z,
                  dt, dxp, dyp, dzp, Nx, Ny, Nz,
                  sx, sy, sz, bxE, cxE, byE, cyE, bzE, czE,
                  psi_Hz_y, psi_Hy_z, psi_Hx_z, psi_Hz_x, psi_Hy_x, psi_Hx_y):
    # dxp/dyp/dzp are the PRIMARY widths sampled at sel_* - 1 (per-slab, in sel_*E
    # order), so they index by the slab counter p — matching update_E's per-cell
    # primary divisor. On a uniform grid they are the constant PML spacing.
    n_xE = sx.shape[0]
    n_yE = sy.shape[0]
    n_zE = sz.shape[0]

    # ---------- Ex: +psi_Hz_y (y axis), -psi_Hy_z (z axis) ----------
    # psi_Hz_y[i, p, k], j = sy[p]; correct Ex[:, sy, 1:]
    for i in prange(Nx):
        for p in range(n_yE):
            j = sy[p]
            for k in range(Nz):
                dHz_dy = (Hz[i, j, k] - Hz[i, j - 1, k]) / dyp[p]
                psi_Hz_y[i, p, k] = byE[p] * psi_Hz_y[i, p, k] + cyE[p] * dHz_dy
    if Nz > 1:
        for i in prange(Nx):
            for p in range(n_yE):
                j = sy[p]
                for k in range(1, Nz):
                    Ex[i, j, k] += (dt / (_EPS0 * eps_x[i, j, k])) * psi_Hz_y[i, p, k]
        # psi_Hy_z[i, j, p], k = sz[p]; correct Ex[:, 1:, sz]
        for i in prange(Nx):
            for j in range(Ny):
                for p in range(n_zE):
                    k = sz[p]
                    dHy_dz = (Hy[i, j, k] - Hy[i, j, k - 1]) / dzp[p]
                    psi_Hy_z[i, j, p] = bzE[p] * psi_Hy_z[i, j, p] + czE[p] * dHy_dz
        for i in prange(Nx):
            for j in range(1, Ny):
                for p in range(n_zE):
                    k = sz[p]
                    Ex[i, j, k] -= (dt / (_EPS0 * eps_x[i, j, k])) * psi_Hy_z[i, j, p]
    else:
        for i in prange(Nx):
            for p in range(n_yE):
                j = sy[p]
                Ex[i, j, 0] += (dt / (_EPS0 * eps_x[i, j, 0])) * psi_Hz_y[i, p, 0]

    # ---------- Ey: -psi_Hz_x (x axis), +psi_Hx_z (z axis) ----------
    # psi_Hz_x[p, j, k], i = sx[p]; correct Ey[sx, :, 1:]
    for j in prange(Ny):
        for p in range(n_xE):
            i = sx[p]
            for k in range(Nz):
                dHz_dx = (Hz[i, j, k] - Hz[i - 1, j, k]) / dxp[p]
                psi_Hz_x[p, j, k] = bxE[p] * psi_Hz_x[p, j, k] + cxE[p] * dHz_dx
    if Nz > 1:
        for j in prange(Ny):
            for p in range(n_xE):
                i = sx[p]
                for k in range(1, Nz):
                    Ey[i, j, k] -= (dt / (_EPS0 * eps_y[i, j, k])) * psi_Hz_x[p, j, k]
        # psi_Hx_z[i, j, p], k = sz[p]; correct Ey[1:, :, sz]
        for i in prange(Nx):
            for j in range(Ny):
                for p in range(n_zE):
                    k = sz[p]
                    dHx_dz = (Hx[i, j, k] - Hx[i, j, k - 1]) / dzp[p]
                    psi_Hx_z[i, j, p] = bzE[p] * psi_Hx_z[i, j, p] + czE[p] * dHx_dz
        for i in prange(1, Nx):
            for j in range(Ny):
                for p in range(n_zE):
                    k = sz[p]
                    Ey[i, j, k] += (dt / (_EPS0 * eps_y[i, j, k])) * psi_Hx_z[i, j, p]
    else:
        for j in prange(Ny):
            for p in range(n_xE):
                i = sx[p]
                Ey[i, j, 0] -= (dt / (_EPS0 * eps_y[i, j, 0])) * psi_Hz_x[p, j, 0]

    # ---------- Ez: +psi_Hy_x (x axis), -psi_Hx_y (y axis) ----------
    # psi_Hy_x[p, j, k], i = sx[p]; correct Ez[sx, 1:, :]
    for j in prange(Ny):
        for p in range(n_xE):
            i = sx[p]
            for k in range(Nz):
                dHy_dx = (Hy[i, j, k] - Hy[i - 1, j, k]) / dxp[p]
                psi_Hy_x[p, j, k] = bxE[p] * psi_Hy_x[p, j, k] + cxE[p] * dHy_dx
    for j in prange(1, Ny):
        for p in range(n_xE):
            i = sx[p]
            for k in range(Nz):
                Ez[i, j, k] += (dt / (_EPS0 * eps_z[i, j, k])) * psi_Hy_x[p, j, k]
    # psi_Hx_y[i, p, k], j = sy[p]; correct Ez[1:, sy, :]
    for i in prange(Nx):
        for p in range(n_yE):
            j = sy[p]
            for k in range(Nz):
                dHx_dy = (Hx[i, j, k] - Hx[i, j - 1, k]) / dyp[p]
                psi_Hx_y[i, p, k] = byE[p] * psi_Hx_y[i, p, k] + cyE[p] * dHx_dy
    for i in prange(1, Nx):
        for p in range(n_yE):
            j = sy[p]
            for k in range(Nz):
                Ez[i, j, k] -= (dt / (_EPS0 * eps_z[i, j, k])) * psi_Hx_y[i, p, k]


# ====================================================================== #
# Thin wrappers — signature-compatible with update.py / pml.py
# ====================================================================== #
def update_H(grid: FDTDGrid) -> FDTDGrid:
    """Numba-accelerated drop-in for :func:`wavesim.update.update_H`."""
    _update_H(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
              grid.mu_x, grid.mu_y, grid.mu_z,
              grid.dt, grid.dxd, grid.dyd, grid.dzd, grid.Nx, grid.Ny, grid.Nz)
    return grid


def update_E(grid: FDTDGrid) -> FDTDGrid:
    """Numba-accelerated drop-in for :func:`wavesim.update.update_E`."""
    _update_E(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
              grid.eps_x, grid.eps_y, grid.eps_z,
              grid.dt, grid.dxp, grid.dyp, grid.dzp, grid.Nx, grid.Ny, grid.Nz)
    return grid


def _ravel(a):
    """Slab (b, c) coefficient arrays are stored reshaped to broadcast; the kernels
    want them as 1D in slab order, which ravel() recovers (C-order over a singleton
    -> n -> singleton reshape)."""
    return np.ascontiguousarray(a).ravel()


def update_H_pml(grid: FDTDGrid, cpml: CPMLArrays) -> tuple[FDTDGrid, CPMLArrays]:
    """Numba-accelerated drop-in for :func:`wavesim.pml.update_H_pml`."""
    _update_H_pml(
        grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
        grid.mu_x, grid.mu_y, grid.mu_z,
        grid.dt, _ravel(cpml.dxd_sH), _ravel(cpml.dyd_sH), _ravel(cpml.dzd_sH),
        grid.Nx, grid.Ny, grid.Nz,
        cpml.sel_xH, cpml.sel_yH, cpml.sel_zH,
        _ravel(cpml.bxH_s), _ravel(cpml.cxH_s),
        _ravel(cpml.byH_s), _ravel(cpml.cyH_s),
        _ravel(cpml.bzH_s), _ravel(cpml.czH_s),
        cpml.psi_Ez_y, cpml.psi_Ey_z, cpml.psi_Ex_z,
        cpml.psi_Ez_x, cpml.psi_Ey_x, cpml.psi_Ex_y)
    return grid, cpml


def update_E_pml(grid: FDTDGrid, cpml: CPMLArrays) -> tuple[FDTDGrid, CPMLArrays]:
    """Numba-accelerated drop-in for :func:`wavesim.pml.update_E_pml`."""
    _update_E_pml(
        grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
        grid.eps_x, grid.eps_y, grid.eps_z,
        grid.dt, _ravel(cpml.dxp_sE), _ravel(cpml.dyp_sE), _ravel(cpml.dzp_sE),
        grid.Nx, grid.Ny, grid.Nz,
        cpml.sel_xE, cpml.sel_yE, cpml.sel_zE,
        _ravel(cpml.bxE_s), _ravel(cpml.cxE_s),
        _ravel(cpml.byE_s), _ravel(cpml.cyE_s),
        _ravel(cpml.bzE_s), _ravel(cpml.czE_s),
        cpml.psi_Hz_y, cpml.psi_Hy_z, cpml.psi_Hx_z,
        cpml.psi_Hz_x, cpml.psi_Hy_x, cpml.psi_Hx_y)
    return grid, cpml
