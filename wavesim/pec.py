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

Conformal (Dey–Mittra) PEC adds a third thing: conformal_geometry(), which turns
the dimensionless open-fraction arrays carried on the grid into the metre-valued
open edge lengths and open face areas the conformal H update integrates over.
"""

from dataclasses import dataclass
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


# ======================================================================= #
# Conformal (Dey–Mittra) PEC geometry
# ======================================================================= #

@dataclass(frozen=True)
class ConformalGeometry:
    """Metre-valued cut-cell geometry derived from the grid's open fractions.

    All arrays have shape ``(Nx, Ny, Nz)``.

    ``Lx``/``Ly``/``Lz`` — open length (m) of the E edges ``Ex``/``Ey``/``Ez``.
    ``Ax``/``Ay``/``Az`` — open area (m²) of the H faces ``Hx``/``Hy``/``Hz``.
    ``inv_Ax``/``inv_Ay``/``inv_Az`` — ``1/A_open`` (m⁻²) for the update kernels.

    An entry of zero in ``L`` or ``A`` is a fully covered edge or face. The
    reciprocal is **guarded**: a fully covered face gets ``inv_A = 0`` rather
    than an infinity, which freezes H there — correct, since the face lies
    wholly inside the conductor and all four of its edges are zeroed too, so the
    contour integral is 0 and the update would otherwise be 0/0.

    The guard also carries the small-cut area threshold: a face whose open
    fraction is below ``grid.conformal_area_threshold`` is treated as fully PEC,
    which is exactly ``inv_A = 0`` for that face. ``n_suppressed`` counts how
    many faces that hit — a number worth watching, since every one of them is a
    cut the run staircased rather than resolved.
    """
    Lx: np.ndarray
    Ly: np.ndarray
    Lz: np.ndarray
    Ax: np.ndarray
    Ay: np.ndarray
    Az: np.ndarray
    inv_Ax: np.ndarray
    inv_Ay: np.ndarray
    inv_Az: np.ndarray
    n_suppressed: int = 0


def build_conformal_geometry(grid: FDTDGrid) -> ConformalGeometry:
    """Open fractions × grid spacing → open lengths and areas (uncached).

    Primary widths throughout, because the Faraday contour of an H face is
    bounded by *nodes*: the ``Hz[i,j,k]`` face spans nodes ``(i..i+1, j..j+1)``,
    so its sides are the E edges ``Ex`` (length ``dxp[i]``) and ``Ey`` (length
    ``dyp[j]``) and its area is ``dxp[i]·dyp[j]``. This is the same convention
    the workbench voxeliser uses to define an edge (node ``(i,j,k)`` → node
    ``(i+1,j,k)``), so there is exactly one geometric definition across the
    process boundary.
    """
    dxp = grid.dxp[:, None, None]
    dyp = grid.dyp[None, :, None]
    dzp = grid.dzp[None, None, :]

    Ax = grid.pec_face_open_x * (dyp * dzp)
    Ay = grid.pec_face_open_y * (dzp * dxp)
    Az = grid.pec_face_open_z * (dxp * dyp)

    thr = float(grid.conformal_area_threshold)
    inv_Ax, nx = _guarded_inverse(Ax, grid.pec_face_open_x, thr)
    inv_Ay, ny = _guarded_inverse(Ay, grid.pec_face_open_y, thr)
    inv_Az, nz = _guarded_inverse(Az, grid.pec_face_open_z, thr)

    return ConformalGeometry(
        Lx=grid.pec_edge_open_x * dxp,
        Ly=grid.pec_edge_open_y * dyp,
        Lz=grid.pec_edge_open_z * dzp,
        Ax=Ax, Ay=Ay, Az=Az,
        inv_Ax=inv_Ax, inv_Ay=inv_Ay, inv_Az=inv_Az,
        n_suppressed=nx + ny + nz,
    )


def _guarded_inverse(area: np.ndarray, fraction: np.ndarray,
                     threshold: float) -> tuple:
    """``(1/area, n_suppressed)``, with sliver and fully covered faces zeroed.

    Thresholding the *fraction* rather than the area keeps the rule
    resolution-independent on a graded mesh, where ``A_full`` varies per cell.
    """
    live = (fraction >= threshold) & (area > 0.0)
    inv = np.divide(1.0, area, out=np.zeros_like(area), where=live)
    n_suppressed = int(np.count_nonzero((fraction > 0.0) & ~live))
    return inv, n_suppressed


def conformal_geometry(grid: FDTDGrid):
    """Cached :class:`ConformalGeometry` for ``grid``, or ``None`` if staircase.

    Cached on the grid and keyed on the *identity* of the six fraction arrays
    plus the value of ``conformal_area_threshold``, the same scheme (and the same
    caveat) as the PEC edge-mask cache in :func:`apply_pec_mask`: replacing an
    array invalidates the cache automatically, mutating one **in place** does not
    — clear ``grid._conformal_cache`` yourself if you need to do that mid-run.
    """
    if not grid.is_conformal:
        return None

    arrays = (grid.pec_edge_open_x, grid.pec_edge_open_y, grid.pec_edge_open_z,
              grid.pec_face_open_x, grid.pec_face_open_y, grid.pec_face_open_z)
    threshold = float(grid.conformal_area_threshold)

    cache = getattr(grid, '_conformal_cache', None)
    if cache is None or cache[1] != threshold or len(cache[0]) != len(arrays) \
            or any(a is not b for a, b in zip(cache[0], arrays)):
        cache = (arrays, threshold, build_conformal_geometry(grid))
        grid._conformal_cache = cache
    return cache[2]


def count_cut_cells(grid: FDTDGrid) -> int:
    """Number of H faces that are partially — not fully — covered by conductor.

    The sanity number echoed as ``summary["cut_cells"]``; zero means the
    conformal path has nothing to do that the staircase path would not.
    """
    if not grid.is_conformal:
        return 0
    total = 0
    for name in ('pec_face_open_x', 'pec_face_open_y', 'pec_face_open_z'):
        f = getattr(grid, name)
        total += int(np.count_nonzero((f > 0.0) & (f < 1.0)))
    return total
