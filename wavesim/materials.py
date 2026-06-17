"""
materials.py — Material array builders.

Two clearly separated roles:

PRODUCTION PATH (called by the future FreeCAD CAD importer and by any test):
    set_vacuum()             — reset entire domain to eps_r=mu_r=1
    set_material_arrays()    — directly assign pre-computed arrays

TEST SCAFFOLDING (used only in tests and examples):
    set_box()                — axis-aligned dielectric or PEC box
    set_cylinder()           — cylindrical rod aligned with Z
    set_coax()               — coaxial cross-section in XY plane

All geometry functions ultimately write to eps/mu arrays or grid.pec_mask.
The future CAD importer bypasses scaffolding and calls set_material_arrays()
directly.
"""

import numpy as np
from wavesim.grid import FDTDGrid


# ======================================================================= #
# PRODUCTION PATH
# ======================================================================= #

def set_vacuum(grid: FDTDGrid) -> FDTDGrid:
    """
    Set entire domain to eps_r=1, mu_r=1 (vacuum).
    Always call this first before placing any material regions.
    """
    grid.eps_x[:] = 1.0
    grid.eps_y[:] = 1.0
    grid.eps_z[:] = 1.0
    grid.mu_x[:]  = 1.0
    grid.mu_y[:]  = 1.0
    grid.mu_z[:]  = 1.0
    return grid

def set_dielectric(grid: FDTDGrid,
                   EPS_X: float, EPS_Y: float = None, EPS_Z: float = None) -> FDTDGrid:
    """
    Set entire domain to uniform dielectric with specified epsilon values.
    If EPS_Y or EPS_Z are not provided, they default to EPS_X.
    """
    if EPS_Y is None:
        EPS_Y = EPS_X
    if EPS_Z is None:
        EPS_Z = EPS_X

    grid.eps_x[:] = EPS_X
    grid.eps_y[:] = EPS_Y
    grid.eps_z[:] = EPS_Z
    grid.mu_x[:]  = 1.0
    grid.mu_y[:]  = 1.0
    grid.mu_z[:]  = 1.0
    return grid

def set_material_arrays(grid: FDTDGrid,
                        eps_x: np.ndarray, eps_y: np.ndarray, eps_z: np.ndarray,
                        mu_x:  np.ndarray, mu_y:  np.ndarray, mu_z:  np.ndarray,
                        pec_mask: np.ndarray = None) -> FDTDGrid:
    """
    Directly assign pre-computed material arrays to the grid.

    This is the function the future CAD importer will call after voxelising
    a FreeCAD geometry into NumPy arrays.

    All arrays must have shape (Nx, Ny, Nz).
    If pec_mask is provided it is written into grid.pec_mask.
    """
    shape = (grid.Nx, grid.Ny, grid.Nz)
    for name, arr in [('eps_x', eps_x), ('eps_y', eps_y), ('eps_z', eps_z),
                      ('mu_x',  mu_x),  ('mu_y',  mu_y),  ('mu_z',  mu_z)]:
        if arr.shape != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")

    grid.eps_x = eps_x.copy()
    grid.eps_y = eps_y.copy()
    grid.eps_z = eps_z.copy()
    grid.mu_x  = mu_x.copy()
    grid.mu_y  = mu_y.copy()
    grid.mu_z  = mu_z.copy()

    if pec_mask is not None:
        if pec_mask.shape != shape:
            raise ValueError(f"pec_mask: expected shape {shape}, got {pec_mask.shape}")
        grid.pec_mask = pec_mask.astype(bool)

    return grid


# ======================================================================= #
# TEST SCAFFOLDING
# ======================================================================= #

def _metre_to_cell(x: float, cell_size: float) -> int:
    """Convert a physical coordinate (metres) to the nearest cell index."""
    return int(round(x / cell_size))


def set_box(grid: FDTDGrid,
            x0: float, x1: float,
            y0: float, y1: float,
            z0: float, z1: float,
            eps_r: float, mu_r: float = 1.0,
            pec: bool = False) -> FDTDGrid:
    """
    Fill an axis-aligned box with a uniform material, or mark as PEC.

    Parameters
    ----------
    x0, x1, y0, y1, z0, z1 : float
        Box corners in metres. Snapped to nearest cell.
    eps_r, mu_r : float
        Relative permittivity / permeability of the fill material.
    pec : bool
        If True, mark the region as PEC in grid.pec_mask instead of
        writing eps/mu values.
    """
    i0 = _metre_to_cell(x0, grid.dx)
    i1 = _metre_to_cell(x1, grid.dx)
    j0 = _metre_to_cell(y0, grid.dy)
    j1 = _metre_to_cell(y1, grid.dy)
    k0 = _metre_to_cell(z0, grid.dz)
    k1 = _metre_to_cell(z1, grid.dz)

    # Clamp to domain
    i0 = max(0, i0); i1 = min(grid.Nx, i1)
    j0 = max(0, j0); j1 = min(grid.Ny, j1)
    k0 = max(0, k0); k1 = min(grid.Nz, k1)

    sl = np.s_[i0:i1, j0:j1, k0:k1]

    if pec:
        if grid.pec_mask is None:
            grid.pec_mask = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool)
        grid.pec_mask[sl] = True
    else:
        grid.eps_x[sl] = eps_r
        grid.eps_y[sl] = eps_r
        grid.eps_z[sl] = eps_r
        grid.mu_x[sl]  = mu_r
        grid.mu_y[sl]  = mu_r
        grid.mu_z[sl]  = mu_r

    return grid


def set_cylinder(grid: FDTDGrid,
                 cx: float, cy: float,
                 radius: float,
                 z0: float, z1: float,
                 eps_r: float, mu_r: float = 1.0,
                 pec: bool = False) -> FDTDGrid:
    """
    Fill a cylindrical rod aligned with Z, or mark as PEC.

    Parameters
    ----------
    cx, cy : float
        Centre of the cylinder in the XY plane (metres).
    radius : float
        Cylinder radius (metres).
    z0, z1 : float
        Axial extent (metres).
    eps_r, mu_r : float
        Material properties (ignored when pec=True).
    pec : bool
        If True, mark the cylinder as PEC.
    """
    k0 = max(0, _metre_to_cell(z0, grid.dz))
    k1 = min(grid.Nz, _metre_to_cell(z1, grid.dz))

    # Build a 2D mask in XY for the circular cross-section
    ix = np.arange(grid.Nx)
    iy = np.arange(grid.Ny)
    # Physical centre coordinates
    cx_cell = cx / grid.dx
    cy_cell = cy / grid.dy
    # Distance from centre in cell units (scaled by physical cell size)
    IX, IY = np.meshgrid(ix, iy, indexing='ij')  # (Nx, Ny)
    dist = np.sqrt(((IX - cx_cell) * grid.dx)**2 +
                   ((IY - cy_cell) * grid.dy)**2)  # metres
    mask_2d = dist <= radius  # shape (Nx, Ny)

    if pec:
        if grid.pec_mask is None:
            grid.pec_mask = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool)
        grid.pec_mask[:, :, k0:k1] |= mask_2d[:, :, np.newaxis]
    else:
        # Apply material to all z slices in range
        for k in range(k0, k1):
            grid.eps_x[:, :, k] = np.where(mask_2d, eps_r, grid.eps_x[:, :, k])
            grid.eps_y[:, :, k] = np.where(mask_2d, eps_r, grid.eps_y[:, :, k])
            grid.eps_z[:, :, k] = np.where(mask_2d, eps_r, grid.eps_z[:, :, k])
            grid.mu_x[:, :, k]  = np.where(mask_2d, mu_r,  grid.mu_x[:, :, k])
            grid.mu_y[:, :, k]  = np.where(mask_2d, mu_r,  grid.mu_y[:, :, k])
            grid.mu_z[:, :, k]  = np.where(mask_2d, mu_r,  grid.mu_z[:, :, k])

    return grid


def set_coax(grid: FDTDGrid,
             cx: float, cy: float,
             r_inner: float, r_outer: float,
             eps_r_fill: float = 1.0) -> FDTDGrid:
    """
    Build a coaxial cross-section in the XY plane.

    Inner conductor  → marked PEC in grid.pec_mask.
    Outer conductor  → marked PEC in grid.pec_mask.
    Dielectric fill  → eps_r_fill written to eps arrays between conductors.

    Parameters
    ----------
    cx, cy : float
        Centre of the coaxial structure (metres).
    r_inner, r_outer : float
        Inner and outer conductor radii (metres).
    eps_r_fill : float
        Relative permittivity of the dielectric between conductors.

    Notes
    -----
    The outer conductor is modelled as a single-cell-thick ring at r_outer.
    For the test, outer conductor cells (r >= r_outer) are marked PEC so the
    simulation domain edge is the conductor wall.
    """
    # Full z extent (entire domain depth)
    z0 = 0.0
    z1 = grid.Nz * grid.dz

    # 1. Dielectric fill between inner and outer conductor
    #    We set the fill material for the annular region first via cylinder
    #    (inner conductor will overwrite with PEC afterwards)
    set_cylinder(grid, cx, cy, r_outer, z0, z1, eps_r_fill, pec=False)

    # 2. Mark inner conductor as PEC
    set_cylinder(grid, cx, cy, r_inner, z0, z1, eps_r=1.0, pec=True)

    # 3. Mark outer conductor as PEC (everything at or beyond r_outer)
    #    Build the outer ring mask directly
    ix = np.arange(grid.Nx)
    iy = np.arange(grid.Ny)
    cx_cell = cx / grid.dx
    cy_cell = cy / grid.dy
    IX, IY = np.meshgrid(ix, iy, indexing='ij')
    dist = np.sqrt(((IX - cx_cell) * grid.dx)**2 +
                   ((IY - cy_cell) * grid.dy)**2)
    outer_mask = dist >= r_outer  # shape (Nx, Ny)

    if grid.pec_mask is None:
        grid.pec_mask = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool)
    grid.pec_mask[:, :, :] |= outer_mask[:, :, np.newaxis]

    return grid
