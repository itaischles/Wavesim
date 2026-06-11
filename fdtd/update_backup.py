"""
update.py — E and H field update functions (pure).

Full 3D curl operators, vectorised over the entire grid using NumPy
slicing. No loops over cells.

Yee grid staggering (standard 3D convention, Taflove):
    Ex[i,j,k]  at  (i,    j+½,  k+½) · (dx, dy, dz)
    Ey[i,j,k]  at  (i+½,  j,    k+½) · (dx, dy, dz)
    Ez[i,j,k]  at  (i+½,  j+½,  k  ) · (dx, dy, dz)

    Hx[i,j,k]  at  (i+½,  j,    k  ) · (dx, dy, dz)
    Hy[i,j,k]  at  (i,    j+½,  k  ) · (dx, dy, dz)
    Hz[i,j,k]  at  (i,    j,    k+½) · (dx, dy, dz)

Update equations (Faraday — H step):
    Hx[i,j,k] -= (dt/mu_x) * ( (Ez[i,j+1,k]-Ez[i,j,k])/dy - (Ey[i,j,k+1]-Ey[i,j,k])/dz )
    Hy[i,j,k] -= (dt/mu_y) * ( (Ex[i,j,k+1]-Ex[i,j,k])/dz - (Ez[i+1,j,k]-Ez[i,j,k])/dx )
    Hz[i,j,k] -= (dt/mu_z) * ( (Ey[i+1,j,k]-Ey[i,j,k])/dx - (Ex[i,j+1,k]-Ex[i,j,k])/dy )

Update equations (Ampere — E step):
    Ex[i,j,k] += (dt/eps_x) * ( (Hz[i,j,k]-Hz[i,j-1,k])/dy - (Hy[i,j,k]-Hy[i,j,k-1])/dz )
    Ey[i,j,k] += (dt/eps_y) * ( (Hx[i,j,k]-Hx[i,j,k-1])/dz - (Hz[i,j,k]-Hz[i-1,j,k])/dx )
    Ez[i,j,k] += (dt/eps_z) * ( (Hy[i,j,k]-Hy[i-1,j,k])/dx - (Hx[i,j,k]-Hx[i,j-1,k])/dy )
"""

import numpy as np
from fdtd.grid import FDTDGrid
from fdtd.constants import MU0, EPS0


def update_H(grid: FDTDGrid) -> FDTDGrid:
    """
    Advance H fields by half a timestep using the full 3D curl of E.

    All three curl terms are present. For Nz=1, z-derivatives evaluate to
    zero automatically — see guard comment below.

    # 3D-UPGRADE: remove the Nz==1 guards when Nz > 1.
    """
    dt = grid.dt
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    # ------------------------------------------------------------------
    # dEz/dy and dEy/dz terms for Hx
    # ------------------------------------------------------------------
    dEz_dy = (grid.Ez[:, 1:, :] - grid.Ez[:, :-1, :]) / dy  # shape (Nx, Ny-1, Nz)

    # 3D-UPGRADE: remove this guard when Nz > 1
    if grid.Nz > 1:
        dEy_dz = np.zeros_like(grid.Ey)
        dEy_dz[:, :, :-1] = (grid.Ey[:, :, 1:] - grid.Ey[:, :, :-1]) / dz
    else:
        dEy_dz = np.zeros_like(grid.Ey)

    # Hx update — interior cells [:, :-1, :-1] but guarded for Nz=1
    if grid.Nz > 1:
        grid.Hx[:, :-1, :-1] -= (dt / (MU0 * grid.mu_x[:, :-1, :-1])) * (
            dEz_dy[:, :, :-1] - dEy_dz[:, :-1, :-1]
        )
    else:
        # Nz=1: k-axis has size 1, slice [:,:,:-1] is empty → operate on full k
        grid.Hx[:, :-1, :] -= (dt / (MU0 * grid.mu_x[:, :-1, :])) * (
            dEz_dy[:, :, :]
            # dEy_dz term is zero for Nz=1
        )

    # ------------------------------------------------------------------
    # dEx/dz and dEz/dx terms for Hy
    # ------------------------------------------------------------------
    dEz_dx = (grid.Ez[1:, :, :] - grid.Ez[:-1, :, :]) / dx  # shape (Nx-1, Ny, Nz)

    # 3D-UPGRADE: remove this guard when Nz > 1
    if grid.Nz > 1:
        dEx_dz = np.zeros_like(grid.Ex)
        dEx_dz[:, :, :-1] = (grid.Ex[:, :, 1:] - grid.Ex[:, :, :-1]) / dz
    else:
        dEx_dz = np.zeros_like(grid.Ex)

    if grid.Nz > 1:
        grid.Hy[:-1, :, :-1] -= (dt / (MU0 * grid.mu_y[:-1, :, :-1])) * (
            dEx_dz[:-1, :, :-1] - dEz_dx[:, :, :-1]
        )
    else:
        grid.Hy[:-1, :, :] -= (dt / (MU0 * grid.mu_y[:-1, :, :])) * (
            # dEx_dz term is zero for Nz=1
            - dEz_dx[:, :, :]
        )

    # ------------------------------------------------------------------
    # dEy/dx and dEx/dy terms for Hz
    # ------------------------------------------------------------------
    dEy_dx = (grid.Ey[1:, :, :] - grid.Ey[:-1, :, :]) / dx  # shape (Nx-1, Ny, Nz)
    dEx_dy = (grid.Ex[:, 1:, :] - grid.Ex[:, :-1, :]) / dy  # shape (Nx, Ny-1, Nz)

    if grid.Nz > 1:
        grid.Hz[:-1, :-1, :] -= (dt / (MU0 * grid.mu_z[:-1, :-1, :])) * (
            dEy_dx[:, :-1, :] - dEx_dy[:-1, :, :]
        )
    else:
        grid.Hz[:-1, :-1, :] -= (dt / (MU0 * grid.mu_z[:-1, :-1, :])) * (
            dEy_dx[:, :-1, :] - dEx_dy[:-1, :, :]
        )

    return grid


def update_E(grid: FDTDGrid) -> FDTDGrid:
    dt = grid.dt
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    # Ex: dHz/dy - dHy/dz
    dHz_dy = (grid.Hz[:, 1:, :] - grid.Hz[:, :-1, :]) / dy
    if grid.Nz > 1:
        dHy_dz = (grid.Hy[:, :, 1:] - grid.Hy[:, :, :-1]) / dz
        grid.Ex[:, 1:, 1:] += (dt / (EPS0 * grid.eps_x[:, 1:, 1:])) * (
            dHz_dy[:, :, 1:] - dHy_dz[:, 1:, :]
        )
    else:
        grid.Ex[:, 1:, :] += (dt / (EPS0 * grid.eps_x[:, 1:, :])) * dHz_dy

    # Ey: dHx/dz - dHz/dx
    dHz_dx = (grid.Hz[1:, :, :] - grid.Hz[:-1, :, :]) / dx
    if grid.Nz > 1:
        dHx_dz = (grid.Hx[:, :, 1:] - grid.Hx[:, :, :-1]) / dz
        grid.Ey[1:, :, 1:] += (dt / (EPS0 * grid.eps_y[1:, :, 1:])) * (
            dHx_dz[1:, :, :] - dHz_dx[:, :, 1:]
        )
    else:
        grid.Ey[1:, :, :] += (dt / (EPS0 * grid.eps_y[1:, :, :])) * (-dHz_dx)

    # Ez: dHy/dx - dHx/dy
    dHy_dx = (grid.Hy[1:, :, :] - grid.Hy[:-1, :, :]) / dx
    dHx_dy = (grid.Hx[:, 1:, :] - grid.Hx[:, :-1, :]) / dy
    grid.Ez[1:, 1:, :] += (dt / (EPS0 * grid.eps_z[1:, 1:, :])) * (
        dHy_dx[:, 1:, :] - dHx_dy[1:, :, :]
    )

    return grid
