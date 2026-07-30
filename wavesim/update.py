"""
update.py — E and H field update functions (pure).

Full 3D curl operators, vectorised over the entire grid using NumPy
slicing. No loops over cells.

Yee grid staggering (standard 3D convention, Taflove):
    Ex[i,j,k]  at  (i+½,  j,    k  ) · (dx, dy, dz)
    Ey[i,j,k]  at  (i,    j+½,  k  ) · (dx, dy, dz)
    Ez[i,j,k]  at  (i,    j,    k+½) · (dx, dy, dz)

    Hx[i,j,k]  at  (i,    j+½,  k+½) · (dx, dy, dz)
    Hy[i,j,k]  at  (i+½,  j,    k+½) · (dx, dy, dz)
    Hz[i,j,k]  at  (i+½,  j+½,  k  ) · (dx, dy, dz)

E lives on cell *edges* (Ex spans node (i,j,k) → (i+1,j,k)), H on face centres.
So cell (i,j,k) owns twelve E-edges but only three of them — Ex[i,j,k],
Ey[i,j,k], Ez[i,j,k] — carry its own index; the other nine are indexed by its
neighbours. See wavesim.pec.build_pec_edge_masks, which depends on this.

Update equations (Faraday — H step):
    Hx[i,j,k] -= (dt/mu_x) * ( (Ez[i,j+1,k]-Ez[i,j,k])/dy - (Ey[i,j,k+1]-Ey[i,j,k])/dz )
    Hy[i,j,k] -= (dt/mu_y) * ( (Ex[i,j,k+1]-Ex[i,j,k])/dz - (Ez[i+1,j,k]-Ez[i,j,k])/dx )
    Hz[i,j,k] -= (dt/mu_z) * ( (Ey[i+1,j,k]-Ey[i,j,k])/dx - (Ex[i,j+1,k]-Ex[i,j,k])/dy )

Update equations (Ampere — E step):
    Ex[i,j,k] += (dt/eps_x) * ( (Hz[i,j,k]-Hz[i,j-1,k])/dy - (Hy[i,j,k]-Hy[i,j,k-1])/dz )
    Ey[i,j,k] += (dt/eps_y) * ( (Hx[i,j,k]-Hx[i,j,k-1])/dz - (Hz[i,j,k]-Hz[i-1,j,k])/dx )
    Ez[i,j,k] += (dt/eps_z) * ( (Hy[i,j,k]-Hy[i-1,j,k])/dx - (Hx[i,j,k]-Hx[i,j-1,k])/dy )

The `if grid.Nz > 1` branches below are the full-3D path; the `else` branches are
a deliberate `Nz=1` fast path that drops the z-derivative entirely (used by the
2D-slice Tests 00–04 and quick iteration). Both paths are validated — full 3D by
Tests 05 (coax) and 06 (cavity). They are kept rather than collapsed because the
fast path is genuinely cheaper for slice runs. The optional Numba backend
(`wavesim/backend_numba.py`, ROADMAP §3) collapses both into a single 3D kernel
guarded by a plain `if Nz > 1`, and is bit-identical to this reference.
"""

import numpy as np
from wavesim.grid import FDTDGrid
from wavesim.constants import MU0, EPS0
from wavesim.pec import conformal_geometry


def update_H(grid: FDTDGrid) -> FDTDGrid:
    """
    Advance H fields by half a timestep using the full 3D curl of E.

    Dispatches to the conformal (Dey–Mittra) cut-cell form when the grid carries
    open-fraction geometry. The staircase branch below is left exactly as it was
    — a model with no conductors must not pay for the six extra array reads per
    face that the conformal form costs.
    """
    if grid.is_conformal:
        return _update_H_conformal(grid)

    dt = grid.dt
    # Every H derivative differences an E field that sits on an integer NODE along
    # the differenced axis (Ez[i,j,k] is at y[j]; Ey[i,j,k] is at x[i]), so the
    # denominator is the PRIMARY width ``dp``. Equivalently: the Faraday contour of
    # an H face is bounded by nodes, so its sides are primary widths.
    # Sliced to the diff-output length and broadcast onto that axis (Yee ``[:-1]``
    # alignment: output index n uses ``dp[n] = s[n+1]-s[n]``). On a uniform grid
    # ``dp`` is the exact constant spacing, so this is bit-identical to the old
    # scalar ``/dy`` divisor.
    dxp = grid.dxp[:-1][:, None, None]
    dyp = grid.dyp[:-1][None, :, None]
    dzp = grid.dzp[:-1][None, None, :]

    # ------------------------------------------------------------------
    # dEz/dy and dEy/dz terms for Hx
    # ------------------------------------------------------------------
    dEz_dy = (grid.Ez[:, 1:, :] - grid.Ez[:, :-1, :]) / dyp

    if grid.Nz > 1:
        dEy_dz = (grid.Ey[:, :, 1:] - grid.Ey[:, :, :-1]) / dzp
        grid.Hx[:, :-1, :-1] -= (dt / (MU0 * grid.mu_x[:, :-1, :-1])) * (
            dEz_dy[:, :, :-1] - dEy_dz[:, :-1, :]
        )
    else:
        grid.Hx[:, :-1, :] -= (dt / (MU0 * grid.mu_x[:, :-1, :])) * dEz_dy

    # ------------------------------------------------------------------
    # dEx/dz and dEz/dx terms for Hy
    # ------------------------------------------------------------------
    dEz_dx = (grid.Ez[1:, :, :] - grid.Ez[:-1, :, :]) / dxp

    if grid.Nz > 1:
        dEx_dz = (grid.Ex[:, :, 1:] - grid.Ex[:, :, :-1]) / dzp
        grid.Hy[:-1, :, :-1] -= (dt / (MU0 * grid.mu_y[:-1, :, :-1])) * (
            dEx_dz[:-1, :, :] - dEz_dx[:, :, :-1]
        )
    else:
        grid.Hy[:-1, :, :] -= (dt / (MU0 * grid.mu_y[:-1, :, :])) * (-dEz_dx)

    # ------------------------------------------------------------------
    # dEy/dx and dEx/dy terms for Hz
    # ------------------------------------------------------------------
    dEy_dx = (grid.Ey[1:, :, :] - grid.Ey[:-1, :, :]) / dxp
    dEx_dy = (grid.Ex[:, 1:, :] - grid.Ex[:, :-1, :]) / dyp

    grid.Hz[:-1, :-1, :] -= (dt / (MU0 * grid.mu_z[:-1, :-1, :])) * (
        dEy_dx[:, :-1, :] - dEx_dy[:-1, :, :]
    )

    return grid


def _update_H_conformal(grid: FDTDGrid) -> FDTDGrid:
    """Faraday on cut cells: the same contour integral, with the covered parts of
    the contour removed and the *open* face area as the denominator.

        Hz[i,j,k] -= (dt/(μ·A_open)) · [ Ey[i+1,j,k]·Ly[i+1,j,k] − Ey[i,j,k]·Ly[i,j,k]
                                       − Ex[i,j+1,k]·Lx[i,j+1,k] + Ex[i,j,k]·Lx[i,j,k] ]

    with Hx and Hy the cyclic permutations. ``L`` is the open length of the E edge
    (metres) and ``A_open`` the open area of the H face.

    Where nothing is cut, ``L`` is the full primary width and ``A_open`` the full
    face area, and every term reduces **algebraically** to the staircase branch.
    Not bit-identically, though: this form evaluates ``(E₁·d − E₀·d)/(d·d')``
    where the staircase branch evaluates ``(E₁ − E₀)/d'``, and floating-point
    multiplication does not distribute exactly. Measured agreement is 1–2 ULP
    (``tests/test_conformal_update.py``). Exact bit-identity is the property of
    the *absent* case — no fraction arrays at all, which dispatches to the
    staircase branch above and cannot differ by construction.

    E on a partially covered edge is the unknown representing the field on the
    *open* part of that edge, so the E update needs no conformal form at all.
    """
    dt = grid.dt
    g = conformal_geometry(grid)

    # Flux-weighted edge fields: the E·dl contributions, ready to difference.
    ExL = grid.Ex * g.Lx
    EyL = grid.Ey * g.Ly
    EzL = grid.Ez * g.Lz

    # ------------------------------------------------------------------
    # Hx — contour (Ez[i,j,k], Ez[i,j+1,k], Ey[i,j,k], Ey[i,j,k+1])
    # ------------------------------------------------------------------
    dEzL_y = EzL[:, 1:, :] - EzL[:, :-1, :]

    if grid.Nz > 1:
        dEyL_z = EyL[:, :, 1:] - EyL[:, :, :-1]
        grid.Hx[:, :-1, :-1] -= (dt / (MU0 * grid.mu_x[:, :-1, :-1])
                                 * g.inv_Ax[:, :-1, :-1]) * (
            dEzL_y[:, :, :-1] - dEyL_z[:, :-1, :]
        )
    else:
        grid.Hx[:, :-1, :] -= (dt / (MU0 * grid.mu_x[:, :-1, :])
                               * g.inv_Ax[:, :-1, :]) * dEzL_y

    # ------------------------------------------------------------------
    # Hy — contour (Ex[i,j,k], Ex[i,j,k+1], Ez[i,j,k], Ez[i+1,j,k])
    # ------------------------------------------------------------------
    dEzL_x = EzL[1:, :, :] - EzL[:-1, :, :]

    if grid.Nz > 1:
        dExL_z = ExL[:, :, 1:] - ExL[:, :, :-1]
        grid.Hy[:-1, :, :-1] -= (dt / (MU0 * grid.mu_y[:-1, :, :-1])
                                 * g.inv_Ay[:-1, :, :-1]) * (
            dExL_z[:-1, :, :] - dEzL_x[:, :, :-1]
        )
    else:
        grid.Hy[:-1, :, :] -= (dt / (MU0 * grid.mu_y[:-1, :, :])
                               * g.inv_Ay[:-1, :, :]) * (-dEzL_x)

    # ------------------------------------------------------------------
    # Hz — contour (Ey[i,j,k], Ey[i+1,j,k], Ex[i,j,k], Ex[i,j+1,k])
    # ------------------------------------------------------------------
    dEyL_x = EyL[1:, :, :] - EyL[:-1, :, :]
    dExL_y = ExL[:, 1:, :] - ExL[:, :-1, :]

    grid.Hz[:-1, :-1, :] -= (dt / (MU0 * grid.mu_z[:-1, :-1, :])
                             * g.inv_Az[:-1, :-1, :]) * (
        dEyL_x[:, :-1, :] - dExL_y[:-1, :, :]
    )

    return grid


def update_E(grid: FDTDGrid) -> FDTDGrid:
    dt = grid.dt
    # Every E derivative differences an H field that sits at a cell CENTRE along
    # the differenced axis (Hz[i,j,k] is at yc[j]), so the denominator is the DUAL
    # width ``dd``. Same Yee ``[:-1]`` alignment and broadcast as ``update_H``, but
    # the difference is written with the upper index, so output index n uses
    # ``dd[n] = (dp[n]+dp[n+1])/2`` and lands on cell n+1. Uniform grids stay
    # bit-identical (``dd == dp == ds``).
    dxd = grid.dxd[:-1][:, None, None]
    dyd = grid.dyd[:-1][None, :, None]
    dzd = grid.dzd[:-1][None, None, :]

    # Ex: dHz/dy - dHy/dz
    dHz_dy = (grid.Hz[:, 1:, :] - grid.Hz[:, :-1, :]) / dyd
    if grid.Nz > 1:
        dHy_dz = (grid.Hy[:, :, 1:] - grid.Hy[:, :, :-1]) / dzd
        grid.Ex[:, 1:, 1:] += (dt / (EPS0 * grid.eps_x[:, 1:, 1:])) * (
            dHz_dy[:, :, 1:] - dHy_dz[:, 1:, :]
        )
    else:
        grid.Ex[:, 1:, :] += (dt / (EPS0 * grid.eps_x[:, 1:, :])) * dHz_dy

    # Ey: dHx/dz - dHz/dx
    dHz_dx = (grid.Hz[1:, :, :] - grid.Hz[:-1, :, :]) / dxd
    if grid.Nz > 1:
        dHx_dz = (grid.Hx[:, :, 1:] - grid.Hx[:, :, :-1]) / dzd
        grid.Ey[1:, :, 1:] += (dt / (EPS0 * grid.eps_y[1:, :, 1:])) * (
            dHx_dz[1:, :, :] - dHz_dx[:, :, 1:]
        )
    else:
        grid.Ey[1:, :, :] += (dt / (EPS0 * grid.eps_y[1:, :, :])) * (-dHz_dx)

    # Ez: dHy/dx - dHx/dy
    dHy_dx = (grid.Hy[1:, :, :] - grid.Hy[:-1, :, :]) / dxd
    dHx_dy = (grid.Hx[:, 1:, :] - grid.Hx[:, :-1, :]) / dyd
    grid.Ez[1:, 1:, :] += (dt / (EPS0 * grid.eps_z[1:, 1:, :])) * (
        dHy_dx[:, 1:, :] - dHx_dy[1:, :, :]
    )

    return grid