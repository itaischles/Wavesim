"""
pec.py — PEC enforcement.

Two distinct operations:

1. apply_pec_faces() — domain boundary condition (walls of the simulation box)
   Zeros tangential E-field components on specified domain faces.

2. apply_pec_mask()  — interior material (solid conductors inside the domain)
   Zeros E on every Yee edge adjoining a cell where grid.pec_mask is True.

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


def _dilate(mask: np.ndarray, axes: tuple) -> np.ndarray:
    """``mask`` OR-ed with itself shifted one cell in the ``+`` direction of each
    axis in ``axes`` — i.e. the union over the 2×2 block of cells that share an
    edge running along the *remaining* axis."""
    out = mask.copy()
    for ax in axes:
        src = out.copy()
        dst_sl = [slice(None)] * 3
        src_sl = [slice(None)] * 3
        dst_sl[ax] = slice(1, None)
        src_sl[ax] = slice(0, -1)
        out[tuple(dst_sl)] |= src[tuple(src_sl)]
    return out


def build_pec_edge_masks(pec_mask: np.ndarray) -> tuple:
    """Per-component E masks for a cell-wise PEC mask: ``(ex, ey, ez)``.

    An E-edge is inside the conductor if **any** of the (up to four) cells
    sharing it is PEC. With the Yee staggering of :mod:`wavesim.update`
    (``Ex[i,j,k]`` spans node ``(i,j,k)`` → ``(i+1,j,k)``), the cells sharing an
    x-edge are ``(i, j-1..j, k-1..k)``, so the x-edge mask is the cell mask
    dilated by one in ``+y`` and ``+z``; likewise for the other two components.
    """
    return (_dilate(pec_mask, (1, 2)),      # Ex — perpendicular axes y, z
            _dilate(pec_mask, (0, 2)),      # Ey — perpendicular axes x, z
            _dilate(pec_mask, (0, 1)))      # Ez — perpendicular axes x, y


def apply_pec_mask(grid: FDTDGrid) -> FDTDGrid:
    """
    Zero the E-field on every Yee edge belonging to a PEC cell.

    If grid.pec_mask is None or all-False, this is a no-op.

    Called every timestep after apply_pec_faces.

    Implementation note
    -------------------
    A cell owns twelve edges, but only three of them (``Ex[i,j,k]``,
    ``Ey[i,j,k]``, ``Ez[i,j,k]``) carry its own index — the other nine are
    indexed by its neighbours. Zeroing only ``E*[mask]``, as this function did
    originally, therefore left E alive half a cell *inside* the metal on each
    conductor's high-x/high-y/high-z faces. E and H then saw different effective
    geometry, breaking the LC = με identity: on an RG58-like coax the wave ran
    6.8% slow (ε_eff 2.648 against a true 2.300). Zeroing an edge when *any*
    adjoining cell is PEC brings that to 2.306 — 0.12%, which is peak-timing
    resolution. See ``tests/test_homogeneous_fill.py``.

    The three per-component masks are derived from ``grid.pec_mask`` and cached
    on the grid, keyed on the mask object's identity. Replacing
    ``grid.pec_mask`` with a new array invalidates the cache automatically;
    mutating it **in place** after the first step does not, so call
    :func:`build_pec_edge_masks` yourself (or clear ``grid._pec_edge_cache``) if
    you need to do that mid-run.
    """
    if grid.pec_mask is None:
        return grid

    mask = grid.pec_mask  # shape (Nx, Ny, Nz), dtype bool

    cache = getattr(grid, '_pec_edge_cache', None)
    if cache is None or cache[0] is not mask:
        cache = (mask,) + build_pec_edge_masks(mask)
        grid._pec_edge_cache = cache
    _, ex, ey, ez = cache

    grid.Ex[ex] = 0.0
    grid.Ey[ey] = 0.0
    grid.Ez[ez] = 0.0

    return grid
