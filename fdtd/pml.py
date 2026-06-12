"""
pml.py — CPML (Convolutional PML) boundary absorption.

CPML replaces each spatial derivative dF/ds inside the PML region with

    dF/ds  ->  (1/kappa_s) * dF/ds  +  psi_s

where psi_s is a recursive convolution variable updated every timestep:

    psi_s^{n+1} = b_s * psi_s^n + c_s * dF/ds

For v1 we use kappa = 1 everywhere (propagating fields only), so the main
field update in update.py already supplies the (1/kappa)*dF/ds part. This
module's only job is to advance the psi arrays and ADD the psi correction
on top of the field that update.py has already advanced.

CRITICAL — coefficient consistency with update.py
--------------------------------------------------
update.py advances the fields with PHYSICAL coefficients:

    Hx -= (dt / (MU0 * mu_x))  * (curl E)
    Ex += (dt / (EPS0 * eps_x)) * (curl H)

The CPML correction is part of that same curl, so it MUST use the identical
coefficient. The psi term is therefore scaled by dt/(MU0*mu) for H updates
and dt/(EPS0*eps) for E updates — NOT by dt/mu or dt/eps. Dropping MU0/EPS0
makes the correction ~1e-6 (or ~1e-11) times too small and the PML stops
absorbing.

Staggering convention (matches the Yee layout in update.py)
-----------------------------------------------------------
Along any axis a, an E-field component that carries a derivative d/da sits at
a half-integer (staggered) coordinate; the corresponding H-field component
sits at an integer (non-staggered) coordinate. Hence:

    *_E profiles : sampled at (i + 0.5) * ds   (used in the E-field updates)
    *_H profiles : sampled at  i        * ds   (used in the H-field updates)

Sign convention (matches update.py exactly)
--------------------------------------------
    Hx -= coef * (dEz/dy - dEy/dz)   -> psi correction:  -psi_Ez_y, +psi_Ey_z
    Hy -= coef * (dEx/dz - dEz/dx)   -> psi correction:  -psi_Ex_z, +psi_Ez_x
    Hz -= coef * (dEy/dx - dEx/dy)   -> psi correction:  -psi_Ey_x, +psi_Ex_y
    Ex += coef * (dHz/dy - dHy/dz)   -> psi correction:  +psi_Hz_y, -psi_Hy_z
    Ey += coef * (dHx/dz - dHz/dx)   -> psi correction:  +psi_Hx_z, -psi_Hz_x
    Ez += coef * (dHy/dx - dHx/dy)   -> psi correction:  +psi_Hy_x, -psi_Hx_y
"""

from dataclasses import dataclass
import numpy as np

from fdtd.grid import FDTDGrid
from fdtd.constants import EPS0, MU0, ETA0


@dataclass
class CPMLArrays:
    # --- Auxiliary convolution variables for the H-field updates -------- #
    # (these correct E-derivatives that appear in the curl of E)
    psi_Ez_y: np.ndarray   # correction to dEz/dy in the Hx update
    psi_Ey_z: np.ndarray   # correction to dEy/dz in the Hx update
    psi_Ex_z: np.ndarray   # correction to dEx/dz in the Hy update
    psi_Ez_x: np.ndarray   # correction to dEz/dx in the Hy update
    psi_Ey_x: np.ndarray   # correction to dEy/dx in the Hz update
    psi_Ex_y: np.ndarray   # correction to dEx/dy in the Hz update

    # --- Auxiliary convolution variables for the E-field updates -------- #
    # (these correct H-derivatives that appear in the curl of H)
    psi_Hz_y: np.ndarray   # correction to dHz/dy in the Ex update
    psi_Hy_z: np.ndarray   # correction to dHy/dz in the Ex update
    psi_Hx_z: np.ndarray   # correction to dHx/dz in the Ey update
    psi_Hz_x: np.ndarray   # correction to dHz/dx in the Ey update
    psi_Hy_x: np.ndarray   # correction to dHy/dx in the Ez update
    psi_Hx_y: np.ndarray   # correction to dHx/dy in the Ez update

    # --- Precomputed (b, c) profiles, one 1D array per axis & grid ------ #
    bx_E: np.ndarray; cx_E: np.ndarray   # shape (Nx,)
    bx_H: np.ndarray; cx_H: np.ndarray
    by_E: np.ndarray; cy_E: np.ndarray   # shape (Ny,)
    by_H: np.ndarray; cy_H: np.ndarray
    bz_E: np.ndarray; cz_E: np.ndarray   # shape (Nz,)
    bz_H: np.ndarray; cz_H: np.ndarray

    d_pml: int             # PML thickness in cells


# ---------------------------------------------------------------------- #
# Profile construction
# ---------------------------------------------------------------------- #
def _calc_profile_1d(N, ds, dt, d_pml, staggered, low=True, high=True):
    """
    Build the 1D (b, c) CPML coefficient arrays along one axis.

    Roden-Gedney profiles (kappa_max = 1 for v1):
        sigma(d) = sigma_max * (d / d_pml)^m
        alpha(d) = alpha_max * (1 - d / d_pml)
        sigma_max = 0.8 * (m + 1) / (ETA0 * ds),   m = 3
        alpha_max = 0.05

    where d is the depth into the PML (0 at the inner edge, d_pml*ds at the
    domain boundary). Returns (b, c) of length N; both reduce to (1, 0)
    outside the PML so the correction vanishes in the interior.

    Parameters
    ----------
    low, high : bool
        Whether to build the absorbing slab at the low-index face (depth grows
        towards index 0) and the high-index face (towards index N-1). Set a
        face False to leave it transparent (e.g. when that face is a PEC wall
        or a symmetry plane). With b=1, c=0 there the psi recursion stays at 0,
        so no correction is applied on that side.
    """
    b = np.ones(N)
    c = np.zeros(N)

    # Too thin to host two PML slabs (e.g. Nz=1 slice), or both faces disabled:
    # no PML on this axis.
    if N <= 2 * d_pml or not (low or high):
        return b, c

    m = 3
    sigma_max = 0.8 * (m + 1) / (ETA0 * ds)
    alpha_max = 0.05
    d_pml_len = d_pml * ds

    # Inner edges of the two PML slabs, in metres.
    left_edge = d_pml * ds
    right_edge = (N - d_pml) * ds

    sigma = np.zeros(N)
    alpha = np.zeros(N)

    # Coordinate of each cell along this axis. The E-field derivatives that
    # this profile corrects are centred half a cell to the LEFT of the index
    # (e.g. Ez[i] is driven by Hy[i]-Hy[i-1], centred at x=(i-0.5)*ds), so the
    # staggered grid uses a -0.5 offset — matching the differencing in
    # update.py. Using +0.5 here mis-aligns the two PML slabs in opposite
    # directions and produces phase-reversed residual reflections.
    idx = np.arange(N)
    coord = (idx - 0.5) * ds if staggered else idx * ds

    # Left slab (low-index face): depth grows towards the boundary (coord -> 0).
    # clip handles the unused over-range staggered node at idx=0 (coord<0).
    if low:
        left = coord < left_edge
        depth_l = np.clip((left_edge - coord[left]) / d_pml_len, 0.0, 1.0)
        sigma[left] = sigma_max * depth_l ** m
        alpha[left] = alpha_max * (1.0 - depth_l)

    # Right slab (high-index face): depth grows towards the boundary (coord -> N*ds).
    if high:
        right = coord > right_edge
        depth_r = np.clip((coord[right] - right_edge) / d_pml_len, 0.0, 1.0)
        sigma[right] = sigma_max * depth_r ** m
        alpha[right] = alpha_max * (1.0 - depth_r)

    # kappa = 1, so the standard CPML coefficients simplify to:
    #   b = exp(-(sigma + alpha) * dt / EPS0)
    #   c = sigma / (sigma + alpha) * (b - 1)
    b = np.exp(-(sigma + alpha) * dt / EPS0)

    denom = sigma + alpha
    mask = denom > 0.0
    c[mask] = (sigma[mask] / denom[mask]) * (b[mask] - 1.0)

    return b, c


ALL_FACES = ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')


def init_cpml(grid: FDTDGrid, d_pml: int = 10,
              faces: tuple = ALL_FACES) -> CPMLArrays:
    """
    Allocate the psi convolution arrays and precompute the (b, c) profiles
    for every axis on both the E (staggered) and H (non-staggered) grids.

    Parameters
    ----------
    faces : tuple of str
        Which domain faces are absorbing CPML. Any subset of
        ('x0','x1','y0','y1','z0','z1'). Faces left out are transparent — use
        this when a face is a PEC wall or symmetry plane (e.g. a waveguide with
        PEC side walls absorbs only on the propagation-axis faces:
        faces=('x0','x1')). Defaults to all six faces.
        'x0' = the face at i=0, 'x1' = the face at i=Nx-1, etc.

    # 3D-UPGRADE: the z-axis profiles return (1, 0) when Nz <= 2*d_pml, so the
    #             z-face PML is inert for Nz=1 slices and activates on its own
    #             once Nz is large enough (subject to z0/z1 being in `faces`).
    """
    bad = set(faces) - set(ALL_FACES)
    if bad:
        raise ValueError(f"Unknown face(s) {sorted(bad)}. "
                         f"Must be a subset of {ALL_FACES}.")

    # Map each axis to (low-index face enabled, high-index face enabled).
    x_lo, x_hi = 'x0' in faces, 'x1' in faces
    y_lo, y_hi = 'y0' in faces, 'y1' in faces
    z_lo, z_hi = 'z0' in faces, 'z1' in faces

    shape = (grid.Nx, grid.Ny, grid.Nz)
    z = lambda: np.zeros(shape, dtype=np.float64)

    bx_E, cx_E = _calc_profile_1d(grid.Nx, grid.dx, grid.dt, d_pml, True,  x_lo, x_hi)
    bx_H, cx_H = _calc_profile_1d(grid.Nx, grid.dx, grid.dt, d_pml, False, x_lo, x_hi)
    by_E, cy_E = _calc_profile_1d(grid.Ny, grid.dy, grid.dt, d_pml, True,  y_lo, y_hi)
    by_H, cy_H = _calc_profile_1d(grid.Ny, grid.dy, grid.dt, d_pml, False, y_lo, y_hi)
    bz_E, cz_E = _calc_profile_1d(grid.Nz, grid.dz, grid.dt, d_pml, True,  z_lo, z_hi)
    bz_H, cz_H = _calc_profile_1d(grid.Nz, grid.dz, grid.dt, d_pml, False, z_lo, z_hi)

    return CPMLArrays(
        psi_Ez_y=z(), psi_Ey_z=z(), psi_Ex_z=z(),
        psi_Ez_x=z(), psi_Ey_x=z(), psi_Ex_y=z(),
        psi_Hz_y=z(), psi_Hy_z=z(), psi_Hx_z=z(),
        psi_Hz_x=z(), psi_Hy_x=z(), psi_Hx_y=z(),
        bx_E=bx_E, cx_E=cx_E, bx_H=bx_H, cx_H=cx_H,
        by_E=by_E, cy_E=cy_E, by_H=by_H, cy_H=cy_H,
        bz_E=bz_E, cz_E=cz_E, bz_H=bz_H, cz_H=cz_H,
        d_pml=d_pml,
    )


# ---------------------------------------------------------------------- #
# H-field CPML correction
# ---------------------------------------------------------------------- #
def update_H_pml(grid: FDTDGrid, cpml: CPMLArrays) -> tuple[FDTDGrid, CPMLArrays]:
    """
    Advance the E-derivative psi arrays and add their correction onto the H
    fields that update_H has already advanced. Coefficient is dt/(MU0*mu) to
    match update.py exactly.
    """
    dt = grid.dt
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    Nz = grid.Nz

    # ---------- Hx  (curl term: dEz/dy - dEy/dz), region Hx[:, :-1, :-1 or :] ----------
    # dEz/dy lives at the Hx location -> non-staggered (H) profile along y.
    dEz_dy = (grid.Ez[:, 1:, :] - grid.Ez[:, :-1, :]) / dy            # (Nx, Ny-1, Nz)
    cpml.psi_Ez_y[:, :-1, :] = (
        cpml.by_H[:-1].reshape(1, -1, 1) * cpml.psi_Ez_y[:, :-1, :]
        + cpml.cy_H[:-1].reshape(1, -1, 1) * dEz_dy
    )

    if Nz > 1:
        coef = dt / (MU0 * grid.mu_x[:, :-1, :-1])
        grid.Hx[:, :-1, :-1] -= coef * cpml.psi_Ez_y[:, :-1, :-1]

        dEy_dz = (grid.Ey[:, :, 1:] - grid.Ey[:, :, :-1]) / dz        # (Nx, Ny, Nz-1)
        cpml.psi_Ey_z[:, :, :-1] = (
            cpml.bz_H[:-1].reshape(1, 1, -1) * cpml.psi_Ey_z[:, :, :-1]
            + cpml.cz_H[:-1].reshape(1, 1, -1) * dEy_dz
        )
        grid.Hx[:, :-1, :-1] += coef * cpml.psi_Ey_z[:, :-1, :-1]
    else:
        coef = dt / (MU0 * grid.mu_x[:, :-1, :])
        grid.Hx[:, :-1, :] -= coef * cpml.psi_Ez_y[:, :-1, :]

    # ---------- Hy  (curl term: dEx/dz - dEz/dx), region Hy[:-1, :, :-1 or :] ----------
    # dEz/dx lives at the Hy location -> non-staggered (H) profile along x.
    dEz_dx = (grid.Ez[1:, :, :] - grid.Ez[:-1, :, :]) / dx            # (Nx-1, Ny, Nz)
    cpml.psi_Ez_x[:-1, :, :] = (
        cpml.bx_H[:-1].reshape(-1, 1, 1) * cpml.psi_Ez_x[:-1, :, :]
        + cpml.cx_H[:-1].reshape(-1, 1, 1) * dEz_dx
    )

    if Nz > 1:
        coef = dt / (MU0 * grid.mu_y[:-1, :, :-1])
        grid.Hy[:-1, :, :-1] += coef * cpml.psi_Ez_x[:-1, :, :-1]

        dEx_dz = (grid.Ex[:, :, 1:] - grid.Ex[:, :, :-1]) / dz        # (Nx, Ny, Nz-1)
        cpml.psi_Ex_z[:, :, :-1] = (
            cpml.bz_H[:-1].reshape(1, 1, -1) * cpml.psi_Ex_z[:, :, :-1]
            + cpml.cz_H[:-1].reshape(1, 1, -1) * dEx_dz
        )
        grid.Hy[:-1, :, :-1] -= coef * cpml.psi_Ex_z[:-1, :, :-1]
    else:
        coef = dt / (MU0 * grid.mu_y[:-1, :, :])
        grid.Hy[:-1, :, :] += coef * cpml.psi_Ez_x[:-1, :, :]

    # ---------- Hz  (curl term: dEy/dx - dEx/dy), region Hz[:-1, :-1, :] ----------
    # dEy/dx lives at the Hz location -> non-staggered (H) profile along x.
    dEy_dx = (grid.Ey[1:, :, :] - grid.Ey[:-1, :, :]) / dx            # (Nx-1, Ny, Nz)
    cpml.psi_Ey_x[:-1, :, :] = (
        cpml.bx_H[:-1].reshape(-1, 1, 1) * cpml.psi_Ey_x[:-1, :, :]
        + cpml.cx_H[:-1].reshape(-1, 1, 1) * dEy_dx
    )
    coef_z = dt / (MU0 * grid.mu_z[:-1, :-1, :])
    grid.Hz[:-1, :-1, :] -= coef_z * cpml.psi_Ey_x[:-1, :-1, :]

    # dEx/dy lives at the Hz location -> non-staggered (H) profile along y.
    dEx_dy = (grid.Ex[:, 1:, :] - grid.Ex[:, :-1, :]) / dy            # (Nx, Ny-1, Nz)
    cpml.psi_Ex_y[:, :-1, :] = (
        cpml.by_H[:-1].reshape(1, -1, 1) * cpml.psi_Ex_y[:, :-1, :]
        + cpml.cy_H[:-1].reshape(1, -1, 1) * dEx_dy
    )
    grid.Hz[:-1, :-1, :] += coef_z * cpml.psi_Ex_y[:-1, :-1, :]

    return grid, cpml


# ---------------------------------------------------------------------- #
# E-field CPML correction
# ---------------------------------------------------------------------- #
def update_E_pml(grid: FDTDGrid, cpml: CPMLArrays) -> tuple[FDTDGrid, CPMLArrays]:
    """
    Advance the H-derivative psi arrays and add their correction onto the E
    fields that update_E has already advanced. Coefficient is dt/(EPS0*eps) to
    match update.py exactly.
    """
    dt = grid.dt
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    Nz = grid.Nz

    # ---------- Ex  (curl term: dHz/dy - dHy/dz), region Ex[:, 1:, 1: or :] ----------
    # dHz/dy lives at the Ex location -> staggered (E) profile along y.
    dHz_dy = (grid.Hz[:, 1:, :] - grid.Hz[:, :-1, :]) / dy            # (Nx, Ny-1, Nz)
    cpml.psi_Hz_y[:, 1:, :] = (
        cpml.by_E[1:].reshape(1, -1, 1) * cpml.psi_Hz_y[:, 1:, :]
        + cpml.cy_E[1:].reshape(1, -1, 1) * dHz_dy
    )

    if Nz > 1:
        coef = dt / (EPS0 * grid.eps_x[:, 1:, 1:])
        grid.Ex[:, 1:, 1:] += coef * cpml.psi_Hz_y[:, 1:, 1:]

        dHy_dz = (grid.Hy[:, :, 1:] - grid.Hy[:, :, :-1]) / dz        # (Nx, Ny, Nz-1)
        cpml.psi_Hy_z[:, :, 1:] = (
            cpml.bz_E[1:].reshape(1, 1, -1) * cpml.psi_Hy_z[:, :, 1:]
            + cpml.cz_E[1:].reshape(1, 1, -1) * dHy_dz
        )
        grid.Ex[:, 1:, 1:] -= coef * cpml.psi_Hy_z[:, 1:, 1:]
    else:
        coef = dt / (EPS0 * grid.eps_x[:, 1:, :])
        grid.Ex[:, 1:, :] += coef * cpml.psi_Hz_y[:, 1:, :]

    # ---------- Ey  (curl term: dHx/dz - dHz/dx), region Ey[1:, :, 1: or :] ----------
    # dHz/dx lives at the Ey location -> staggered (E) profile along x.
    dHz_dx = (grid.Hz[1:, :, :] - grid.Hz[:-1, :, :]) / dx            # (Nx-1, Ny, Nz)
    cpml.psi_Hz_x[1:, :, :] = (
        cpml.bx_E[1:].reshape(-1, 1, 1) * cpml.psi_Hz_x[1:, :, :]
        + cpml.cx_E[1:].reshape(-1, 1, 1) * dHz_dx
    )

    if Nz > 1:
        coef = dt / (EPS0 * grid.eps_y[1:, :, 1:])
        grid.Ey[1:, :, 1:] -= coef * cpml.psi_Hz_x[1:, :, 1:]

        dHx_dz = (grid.Hx[:, :, 1:] - grid.Hx[:, :, :-1]) / dz        # (Nx, Ny, Nz-1)
        cpml.psi_Hx_z[:, :, 1:] = (
            cpml.bz_E[1:].reshape(1, 1, -1) * cpml.psi_Hx_z[:, :, 1:]
            + cpml.cz_E[1:].reshape(1, 1, -1) * dHx_dz
        )
        grid.Ey[1:, :, 1:] += coef * cpml.psi_Hx_z[1:, :, 1:]
    else:
        coef = dt / (EPS0 * grid.eps_y[1:, :, :])
        grid.Ey[1:, :, :] -= coef * cpml.psi_Hz_x[1:, :, :]

    # ---------- Ez  (curl term: dHy/dx - dHx/dy), region Ez[1:, 1:, :] ----------
    # dHy/dx lives at the Ez location -> staggered (E) profile along x.
    dHy_dx = (grid.Hy[1:, :, :] - grid.Hy[:-1, :, :]) / dx            # (Nx-1, Ny, Nz)
    cpml.psi_Hy_x[1:, :, :] = (
        cpml.bx_E[1:].reshape(-1, 1, 1) * cpml.psi_Hy_x[1:, :, :]
        + cpml.cx_E[1:].reshape(-1, 1, 1) * dHy_dx
    )
    coef_z = dt / (EPS0 * grid.eps_z[1:, 1:, :])
    grid.Ez[1:, 1:, :] += coef_z * cpml.psi_Hy_x[1:, 1:, :]

    # dHx/dy lives at the Ez location -> staggered (E) profile along y.
    dHx_dy = (grid.Hx[:, 1:, :] - grid.Hx[:, :-1, :]) / dy            # (Nx, Ny-1, Nz)
    cpml.psi_Hx_y[:, 1:, :] = (
        cpml.by_E[1:].reshape(1, -1, 1) * cpml.psi_Hx_y[:, 1:, :]
        + cpml.cy_E[1:].reshape(1, -1, 1) * dHx_dy
    )
    grid.Ez[1:, 1:, :] -= coef_z * cpml.psi_Hx_y[1:, 1:, :]

    return grid, cpml
