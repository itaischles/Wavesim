from dataclasses import dataclass
import numpy as np
from fdtd.grid import FDTDGrid
from fdtd.constants import EPS0, ETA0

@dataclass
class CPMLArrays:
    # Auxiliary E-curl correction variables (one per PML face per axis)
    psi_Ex_y: np.ndarray   # correction to dEx/dy in Hz update
    psi_Ex_z: np.ndarray   # correction to dEx/dz in Hy update
    psi_Ey_x: np.ndarray
    psi_Ey_z: np.ndarray
    psi_Ez_x: np.ndarray
    psi_Ez_y: np.ndarray

    # Auxiliary H-curl correction variables
    psi_Hx_y: np.ndarray   # correction to dHx/dy in Ez update
    psi_Hx_z: np.ndarray   # correction to dHx/dz in Ey update
    psi_Hy_x: np.ndarray
    psi_Hy_z: np.ndarray
    psi_Hz_x: np.ndarray
    psi_Hz_y: np.ndarray

    # Precomputed b, c profile arrays for each axis and grid type
    bx_E: np.ndarray   # shape (Nx,) — one value per x-cell
    cx_E: np.ndarray
    bx_H: np.ndarray
    cx_H: np.ndarray

    by_E: np.ndarray   # shape (Ny,)
    cy_E: np.ndarray
    by_H: np.ndarray
    cy_H: np.ndarray

    bz_E: np.ndarray   # shape (Nz,)
    cz_E: np.ndarray
    bz_H: np.ndarray
    cz_H: np.ndarray

    d_pml: int         # PML thickness in cells


def _calc_profile_1d(N: int, ds: float, dt: float, d_pml: int, is_staggered: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to compute 1D b and c parameter profiles for an axis.
    Handles uniform and staggered Yee-grid coordinates cleanly to eliminate index shifting errors.
    """
    # # 3D-UPGRADE: For Nz=1 slices, this guard prevents negative index errors
    if N <= 2 * d_pml:
        return np.ones(N), np.zeros(N)

    sigma = np.zeros(N)
    alpha = np.zeros(N)
    kappa = np.ones(N)  # kappa_max = 1 for v1 propagating fields

    m = 3
    sigma_max = 0.8 * (m + 1) / (ETA0 * ds)
    alpha_max = 0.05
    d_pml_m = d_pml * ds

    left_interface = d_pml * ds
    right_interface = (N - d_pml) * ds

    for i in range(N):
        coord = (i + 0.5) * ds if is_staggered else i * ds

        if coord < left_interface:
            depth = left_interface - coord
            sigma[i] = sigma_max * (depth / d_pml_m) ** m
            alpha[i] = alpha_max * (1.0 - depth / d_pml_m)
        elif coord > right_interface:
            depth = coord - right_interface
            sigma[i] = sigma_max * (depth / d_pml_m) ** m
            alpha[i] = alpha_max * (1.0 - depth / d_pml_m)

    b = np.exp(-(sigma / kappa + alpha) * dt / EPS0)

    denom = sigma * kappa + (kappa ** 2) * alpha
    c = np.zeros(N)
    mask = denom > 0
    c[mask] = (sigma[mask] / denom[mask]) * (b[mask] - 1.0)

    return b, c


def init_cpml(grid: FDTDGrid, d_pml: int = 10) -> CPMLArrays:
    """
    Allocate auxiliary arrays and precompute b, c profiles for all 6 faces.
    # 3D-UPGRADE: z-face PML arrays are allocated but zeroed when Nz=1.
    """
    shape = (grid.Nx, grid.Ny, grid.Nz)

    # Compute 1D profile coefficients per grid type and axis
    bx_E, cx_E = _calc_profile_1d(grid.Nx, grid.dx, grid.dt, d_pml, is_staggered=True)
    bx_H, cx_H = _calc_profile_1d(grid.Nx, grid.dx, grid.dt, d_pml, is_staggered=False)

    by_E, cy_E = _calc_profile_1d(grid.Ny, grid.dy, grid.dt, d_pml, is_staggered=True)
    by_H, cy_H = _calc_profile_1d(grid.Ny, grid.dy, grid.dt, d_pml, is_staggered=False)

    # # 3D-UPGRADE: Evaluates to bz=1, cz=0 automatically when Nz=1 due to length guard
    bz_E, cz_E = _calc_profile_1d(grid.Nz, grid.dz, grid.dt, d_pml, is_staggered=True)
    bz_H, cz_H = _calc_profile_1d(grid.Nz, grid.dz, grid.dt, d_pml, is_staggered=False)

    return CPMLArrays(
        psi_Ex_y=np.zeros(shape), psi_Ex_z=np.zeros(shape),
        psi_Ey_x=np.zeros(shape), psi_Ey_z=np.zeros(shape),
        psi_Ez_x=np.zeros(shape), psi_Ez_y=np.zeros(shape),
        psi_Hx_y=np.zeros(shape), psi_Hx_z=np.zeros(shape),
        psi_Hy_x=np.zeros(shape), psi_Hy_z=np.zeros(shape),
        psi_Hz_x=np.zeros(shape), psi_Hz_y=np.zeros(shape),
        bx_E=bx_E, cx_E=cx_E, bx_H=bx_H, cx_H=cx_H,
        by_E=by_E, cy_E=cy_E, by_H=by_H, cy_H=cy_H,
        bz_E=bz_E, cz_E=cz_E, bz_H=bz_H, cz_H=cz_H,
        d_pml=d_pml
    )


def update_H_pml(grid: FDTDGrid, cpml: CPMLArrays) -> tuple[FDTDGrid, CPMLArrays]:
    """Update ψ arrays for H-field and apply CPML correction to H update."""
    dt = grid.dt

    # --- Hx Corrections (uses dEz/dy and dEy/dz) ---
    dEz_dy = (grid.Ez[:, 1:, :] - grid.Ez[:, :-1, :]) / grid.dy
    cpml.psi_Ez_y[:, :-1, :] = (
        cpml.by_H[:-1].reshape(1, -1, 1) * cpml.psi_Ez_y[:, :-1, :] +
        cpml.cy_H[:-1].reshape(1, -1, 1) * dEz_dy
    )
    grid.Hx[:, :-1, :] -= (dt / grid.mu_x[:, :-1, :]) * cpml.psi_Ez_y[:, :-1, :]

    # # 3D-UPGRADE: Activates automatically when Nz > 1
    if grid.Nz > 1:
        dEy_dz = (grid.Ey[:, :, 1:] - grid.Ey[:, :, :-1]) / grid.dz
        cpml.psi_Ey_z[:, :, :-1] = (
            cpml.bz_H[:-1].reshape(1, 1, -1) * cpml.psi_Ey_z[:, :, :-1] +
            cpml.cz_H[:-1].reshape(1, 1, -1) * dEy_dz
        )
        grid.Hx[:, :, :-1] += (dt / grid.mu_x[:, :, :-1]) * cpml.psi_Ey_z[:, :, :-1]

    # --- Hy Corrections (uses dEx/dz and dEz/dx) ---
    # # 3D-UPGRADE: Activates automatically when Nz > 1
    if grid.Nz > 1:
        dEx_dz = (grid.Ex[:, :, 1:] - grid.Ex[:, :, :-1]) / grid.dz
        cpml.psi_Ex_z[:, :, :-1] = (
            cpml.bz_H[:-1].reshape(1, 1, -1) * cpml.psi_Ex_z[:, :, :-1] +
            cpml.cz_H[:-1].reshape(1, 1, -1) * dEx_dz
        )
        grid.Hy[:, :, :-1] -= (dt / grid.mu_y[:, :, :-1]) * cpml.psi_Ex_z[:, :, :-1]

    dEz_dx = (grid.Ez[1:, :, :] - grid.Ez[:-1, :, :]) / grid.dx
    cpml.psi_Ez_x[:-1, :, :] = (
        cpml.bx_H[:-1].reshape(-1, 1, 1) * cpml.psi_Ez_x[:-1, :, :] +
        cpml.cx_H[:-1].reshape(-1, 1, 1) * dEz_dx
    )
    grid.Hy[:-1, :, :] += (dt / grid.mu_y[:-1, :, :]) * cpml.psi_Ez_x[:-1, :, :]

    # --- Hz Corrections (uses dEy/dx and dEx/dy) ---
    dEy_dx = (grid.Ey[1:, :, :] - grid.Ey[:-1, :, :]) / grid.dx
    cpml.psi_Ey_x[:-1, :, :] = (
        cpml.bx_H[:-1].reshape(-1, 1, 1) * cpml.psi_Ey_x[:-1, :, :] +
        cpml.cx_H[:-1].reshape(-1, 1, 1) * dEy_dx
    )
    grid.Hz[:-1, :, :] -= (dt / grid.mu_z[:-1, :, :]) * cpml.psi_Ey_x[:-1, :, :]

    dEx_dy = (grid.Ex[:, 1:, :] - grid.Ex[:, :-1, :]) / grid.dy
    cpml.psi_Ex_y[:, :-1, :] = (
        cpml.by_H[:-1].reshape(1, -1, 1) * cpml.psi_Ex_y[:, :-1, :] +
        cpml.cy_H[:-1].reshape(1, -1, 1) * dEx_dy
    )
    grid.Hz[:, :-1, :] += (dt / grid.mu_z[:, :-1, :]) * cpml.psi_Ex_y[:, :-1, :]

    return grid, cpml


def update_E_pml(grid: FDTDGrid, cpml: CPMLArrays) -> tuple[FDTDGrid, CPMLArrays]:
    """Update ψ arrays for E-field and apply CPML correction to E update."""
    dt = grid.dt

    # --- Ex Corrections (uses dHz/dy and dHy/dz) ---
    dHz_dy = (grid.Hz[:, 1:, :] - grid.Hz[:, :-1, :]) / grid.dy
    cpml.psi_Hz_y[:, 1:, :] = (
        cpml.by_E[1:].reshape(1, -1, 1) * cpml.psi_Hz_y[:, 1:, :] +
        cpml.cy_E[1:].reshape(1, -1, 1) * dHz_dy
    )
    grid.Ex[:, 1:, :] += (dt / grid.eps_x[:, 1:, :]) * cpml.psi_Hz_y[:, 1:, :]

    # # 3D-UPGRADE: Activates automatically when Nz > 1
    if grid.Nz > 1:
        dHy_dz = (grid.Hy[:, :, 1:] - grid.Hy[:, :, :-1]) / grid.dz
        cpml.psi_Hy_z[:, :, 1:] = (
            cpml.bz_E[1:].reshape(1, 1, -1) * cpml.psi_Hy_z[:, :, 1:] +
            cpml.cz_E[1:].reshape(1, 1, -1) * dHy_dz
        )
        grid.Ex[:, :, 1:] -= (dt / grid.eps_x[:, :, 1:]) * cpml.psi_Hy_z[:, :, 1:]

    # --- Ey Corrections (uses dHx/dz and dHz/dx) ---
    # # 3D-UPGRADE: Activates automatically when Nz > 1
    if grid.Nz > 1:
        dHx_dz = (grid.Hx[:, :, 1:] - grid.Hx[:, :, :-1]) / grid.dz
        cpml.psi_Hx_z[:, :, 1:] = (
            cpml.bz_E[1:].reshape(1, 1, -1) * cpml.psi_Hx_z[:, :, 1:] +
            cpml.cz_E[1:].reshape(1, 1, -1) * dHx_dz
        )
        grid.Ey[:, :, 1:] += (dt / grid.eps_y[:, :, 1:]) * cpml.psi_Hx_z[:, :, 1:]

    dHz_dx = (grid.Hz[1:, :, :] - grid.Hz[:-1, :, :]) / grid.dx
    cpml.psi_Hz_x[1:, :, :] = (
        cpml.bx_E[1:].reshape(-1, 1, 1) * cpml.psi_Hz_x[1:, :, :] +
        cpml.cx_E[1:].reshape(-1, 1, 1) * dHz_dx
    )
    grid.Ey[1:, :, :] -= (dt / grid.eps_y[1:, :, :]) * cpml.psi_Hz_x[1:, :, :]

    # --- Ez Corrections (uses dHy/dx and dHx/dy) ---
    dHy_dx = (grid.Hy[1:, :, :] - grid.Hy[:-1, :, :]) / grid.dx
    cpml.psi_Hy_x[1:, :, :] = (
        cpml.bx_E[1:].reshape(-1, 1, 1) * cpml.psi_Hy_x[1:, :, :] +
        cpml.cx_E[1:].reshape(-1, 1, 1) * dHy_dx
    )
    grid.Ez[1:, :, :] += (dt / grid.eps_z[1:, :, :]) * cpml.psi_Hy_x[1:, :, :]

    dHx_dy = (grid.Hx[:, 1:, :] - grid.Hx[:, :-1, :]) / grid.dy
    cpml.psi_Hx_y[:, 1:, :] = (
        cpml.by_E[1:].reshape(1, -1, 1) * cpml.psi_Hx_y[:, 1:, :] +
        cpml.cy_E[1:].reshape(1, -1, 1) * dHx_dy
    )
    grid.Ez[:, 1:, :] -= (dt / grid.eps_z[:, 1:, :]) * cpml.psi_Hx_y[:, 1:, :]

    return grid, cpml