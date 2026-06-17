"""
pec.py — PEC enforcement.

Two distinct operations:

1. apply_pec_faces() — domain boundary condition (walls of the simulation box)
   Zeros tangential E-field components on specified domain faces.

2. apply_pec_mask()  — interior material (solid conductors inside the domain)
   Zeros all E components where grid.pec_mask is True.

Correct timestep order (from the main loop):
    E update → CPML E correction → apply_pec_faces → apply_pec_mask → monitors

PEC enforcement must come after every E update, including after CPML corrections.
"""

import numpy as np
from wavesim.grid import FDTDGrid


def apply_pec_faces(grid: FDTDGrid,
                    faces: tuple = ('x0', 'x1', 'y0', 'y1')) -> FDTDGrid:
    """
    Zero tangential E-field components on specified domain faces.

    Parameters
    ----------
    faces : tuple of str
        Any subset of ('x0', 'x1', 'y0', 'y1', 'z0', 'z1').
        'x0' = the face at i=0, 'x1' = face at i=Nx-1, etc.

    Notes
    -----
    Tangential components on a face are those *not* normal to the face.
    - x-face: tangential = Ey, Ez
    - y-face: tangential = Ex, Ez
    - z-face: tangential = Ex, Ey
    """
    for face in faces:
        if face == 'x0':
            grid.Ey[0, :, :] = 0.0
            grid.Ez[0, :, :] = 0.0
        elif face == 'x1':
            grid.Ey[-1, :, :] = 0.0
            grid.Ez[-1, :, :] = 0.0
        elif face == 'y0':
            grid.Ex[:, 0, :] = 0.0
            grid.Ez[:, 0, :] = 0.0
        elif face == 'y1':
            grid.Ex[:, -1, :] = 0.0
            grid.Ez[:, -1, :] = 0.0
        elif face == 'z0':
            grid.Ex[:, :, 0] = 0.0
            grid.Ey[:, :, 0] = 0.0
        elif face == 'z1':
            grid.Ex[:, :, -1] = 0.0
            grid.Ey[:, :, -1] = 0.0
        else:
            raise ValueError(f"Unknown face '{face}'. "
                             f"Must be one of: x0, x1, y0, y1, z0, z1")
    return grid


def apply_pec_mask(grid: FDTDGrid) -> FDTDGrid:
    """
    Zero all E-field components inside cells where grid.pec_mask is True.

    If grid.pec_mask is None or all-False, this is a no-op.

    Called every timestep after apply_pec_faces.

    Implementation note:
    For v1, all E components where pec_mask[i,j,k] == True are zeroed.
    This is sufficient for most geometries. A more accurate surface treatment
    (zeroing only tangential components at the exact Yee face) can be added in v2.
    """
    if grid.pec_mask is None:
        return grid

    mask = grid.pec_mask  # shape (Nx, Ny, Nz), dtype bool

    grid.Ex[mask] = 0.0
    grid.Ey[mask] = 0.0
    grid.Ez[mask] = 0.0

    return grid
