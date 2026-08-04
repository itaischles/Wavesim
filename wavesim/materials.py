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

All geometry functions ultimately write to eps/mu/sigma arrays or grid.pec_mask.
The future CAD importer bypasses scaffolding and calls set_material_arrays()
directly.

Conductivity is opt-in: every placement function takes ``sigma`` (S/m,
default 0), and the σ arrays are allocated only once a nonzero one is placed, so
a lossless model never carries them. Lossy dielectrics only — see
:mod:`wavesim.loss` for why a metal-scale σ is not a model of a metal.
"""

import numpy as np
from wavesim.grid import FDTDGrid
from wavesim.subpixel import smooth_shape_region


# ======================================================================= #
# PRODUCTION PATH
# ======================================================================= #

def set_vacuum(grid: FDTDGrid) -> FDTDGrid:
    """
    Set entire domain to eps_r=1, mu_r=1, sigma=0 (vacuum).
    Always call this first before placing any material regions.
    """
    grid.eps_x[:] = 1.0
    grid.eps_y[:] = 1.0
    grid.eps_z[:] = 1.0
    grid.mu_x[:]  = 1.0
    grid.mu_y[:]  = 1.0
    grid.mu_z[:]  = 1.0
    _clear_sigma(grid)
    return grid

def set_dielectric(grid: FDTDGrid,
                   EPS_X: float, EPS_Y: float = None, EPS_Z: float = None,
                   sigma: float = 0.0) -> FDTDGrid:
    """
    Set entire domain to uniform dielectric with specified epsilon values.
    If EPS_Y or EPS_Z are not provided, they default to EPS_X.

    ``sigma`` is the (isotropic) electric conductivity in S/m — lossy
    dielectrics only; see :mod:`wavesim.loss`.
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
    if sigma:
        for arr in _sigma_arrays(grid):
            arr[:] = sigma
    else:
        _clear_sigma(grid)
    return grid


# ----------------------------------------------------------------------- #
# Conductivity plumbing
#
# The σ arrays are allocated on first use rather than with the grid, so a
# lossless model neither carries them nor takes the lossy update path (see
# wavesim.loss). ``sigma=0.0`` therefore does not allocate: passing it means
# "no loss here", not "make this grid lossy with zero loss".
# ----------------------------------------------------------------------- #

def _sigma_arrays(grid: FDTDGrid) -> tuple:
    """The three σ arrays, allocating zeros (and so making the grid lossy) once."""
    if grid.sigma_x is None:
        shape = (grid.Nx, grid.Ny, grid.Nz)
        dtype = grid.eps_x.dtype
        grid.sigma_x = np.zeros(shape, dtype=dtype)
        grid.sigma_y = np.zeros(shape, dtype=dtype)
        grid.sigma_z = np.zeros(shape, dtype=dtype)
    return grid.sigma_x, grid.sigma_y, grid.sigma_z


def _clear_sigma(grid: FDTDGrid) -> None:
    """Drop the σ arrays entirely, returning the grid to the lossless path."""
    grid.sigma_x = grid.sigma_y = grid.sigma_z = None
    grid._loss_cache = None


def _reject_conductive_pec(sigma: float, what: str) -> None:
    """A region cannot be both PEC and a lossy dielectric.

    Silently ignoring the σ would be defensible — ``apply_pec_mask`` zeroes E in
    the cell afterwards either way, so PEC wins on overlap — but a caller who
    typed both wants one of them and we cannot tell which.
    """
    if sigma:
        raise ValueError(
            f"{what}: pec=True and sigma={sigma} are mutually exclusive. A PEC "
            f"cell has its E zeroed after every update, so the conductivity "
            f"would have no effect. Drop one: PEC for a good conductor, sigma "
            f"for a lossy dielectric.")


def _reject_subpixel_sigma(sigma: float, what: str) -> None:
    """Refuse subpixel smoothing of a lossy shape.

    Kottke's normal/tangential reduction is derived for a real ε. Conductivity
    is the imaginary part of ``eps~ = eps - j*sigma/(w*eps0)``, so the correct
    smoothing of a lossy interface is frequency-dependent and a real (ε, σ) pair
    cannot carry it. Applying the ε rule to σ regardless is a common
    approximation and may be added later; until it is measured, a staircased
    boundary whose error is understood beats a smoothed one whose error is not.
    """
    if sigma:
        raise NotImplementedError(
            f"{what}: subpixel smoothing of a lossy material is not supported. "
            f"Kottke's tensor reduction is derived for a real permittivity, and "
            f"the correct smoothing of a conductive interface is "
            f"frequency-dependent (sigma is the imaginary part of eps). Place "
            f"the lossy region with subpixel=False, or place it lossless and "
            f"write grid.sigma_* yourself.")

_CONFORMAL_KEYS = ('pec_edge_open_x', 'pec_edge_open_y', 'pec_edge_open_z',
                   'pec_face_open_x', 'pec_face_open_y', 'pec_face_open_z')


def set_material_arrays(grid: FDTDGrid,
                        eps_x: np.ndarray, eps_y: np.ndarray, eps_z: np.ndarray,
                        mu_x:  np.ndarray, mu_y:  np.ndarray, mu_z:  np.ndarray,
                        sigma_x: np.ndarray = None,
                        sigma_y: np.ndarray = None,
                        sigma_z: np.ndarray = None,
                        pec_mask: np.ndarray = None,
                        pec_edge_open_x: np.ndarray = None,
                        pec_edge_open_y: np.ndarray = None,
                        pec_edge_open_z: np.ndarray = None,
                        pec_face_open_x: np.ndarray = None,
                        pec_face_open_y: np.ndarray = None,
                        pec_face_open_z: np.ndarray = None,
                        conformal_area_threshold: float = None) -> FDTDGrid:
    """
    Directly assign pre-computed material arrays to the grid.

    This is the function the FreeCAD CAD importer calls after voxelising a
    geometry into NumPy arrays.

    All arrays must have shape (Nx, Ny, Nz).
    If pec_mask is provided it is written into grid.pec_mask.

    Conductivity
    ------------
    The three ``sigma_*`` arrays are the electric conductivity in S/m seen by
    Ex/Ey/Ez, all three together or none at all. Passing them switches the
    solver onto the two-coefficient E update (:mod:`wavesim.loss`); omitting
    them (the default) **clears** any conductivity the grid was carrying and
    leaves every existing code path untouched and bit-identical.

    Lossy dielectrics only. Good conductors belong in ``pec_mask``; where a σ
    region and ``pec_mask`` overlap, PEC wins, because ``apply_pec_mask`` zeroes
    E after the update regardless.

    Conformal PEC
    -------------
    The six ``pec_*_open_*`` arrays are the open-fraction geometry described in
    :class:`~wavesim.grid.FDTDGrid` — dimensionless, in [0, 1], all six together
    or none at all. Passing them switches the solver onto the conformal
    (Dey–Mittra) path; omitting them (the default) leaves every existing code
    path untouched and bit-identical.

    ``conformal_area_threshold`` overrides the small-cut stability threshold
    (default 0.4 — see :class:`~wavesim.grid.FDTDGrid`). Real geometry needs it:
    an analytic coax on a 32-cell transverse mesh produces open-area fractions
    down to 0.011, and the run diverges without it.
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

    sigmas = (sigma_x, sigma_y, sigma_z)
    if any(s is not None for s in sigmas):
        # All three or none, for the same reason the conformal set is all six:
        # a partial set would leave one field component lossless and the other
        # two damped, which is not a material anyone asked for.
        names = ('sigma_x', 'sigma_y', 'sigma_z')
        missing = [n for n, s in zip(names, sigmas) if s is None]
        if missing:
            raise ValueError(
                "conductivity must be supplied for all three components; "
                f"missing {missing}")
        for name, arr in zip(names, sigmas):
            if arr.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
            if np.any(arr < 0.0):
                raise ValueError(f"{name}: conductivity must be non-negative")
            setattr(grid, name, arr.copy())
        grid._loss_cache = None
    else:
        _clear_sigma(grid)

    if pec_mask is not None:
        if pec_mask.shape != shape:
            raise ValueError(f"pec_mask: expected shape {shape}, got {pec_mask.shape}")
        grid.pec_mask = pec_mask.astype(bool)

    if conformal_area_threshold is not None:
        if not 0.0 <= conformal_area_threshold < 1.0:
            raise ValueError("conformal_area_threshold must lie in [0, 1), got "
                             f"{conformal_area_threshold}")
        grid.conformal_area_threshold = float(conformal_area_threshold)

    conformal = dict(zip(_CONFORMAL_KEYS,
                         (pec_edge_open_x, pec_edge_open_y, pec_edge_open_z,
                          pec_face_open_x, pec_face_open_y, pec_face_open_z)))
    given = [k for k, v in conformal.items() if v is not None]
    if given:
        # All six or none: a partial set would silently mix conformal edges with
        # staircase faces (or the reverse), and E and H would see different
        # geometry — the exact failure mode the edge dilation was added to fix.
        missing = [k for k in _CONFORMAL_KEYS if conformal[k] is None]
        if missing:
            raise ValueError(
                "conformal PEC arrays must be supplied as a complete set of six; "
                f"missing {missing}")
        for name in _CONFORMAL_KEYS:
            arr = np.ascontiguousarray(conformal[name], dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
            if not np.all((arr >= 0.0) & (arr <= 1.0)):
                raise ValueError(f"{name}: open fractions must lie in [0, 1]")
            setattr(grid, name, arr)

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
            pec: bool = False, sigma: float = 0.0,
            subpixel: bool = False, oversample: int = 4) -> FDTDGrid:
    """
    Fill an axis-aligned box with a uniform material, or mark as PEC.

    Parameters
    ----------
    x0, x1, y0, y1, z0, z1 : float
        Box corners in metres. Snapped to nearest cell (unless ``subpixel``).
    eps_r, mu_r : float
        Relative permittivity / permeability of the fill material.
    pec : bool
        If True, mark the region as PEC in grid.pec_mask instead of
        writing eps/mu values.
    sigma : float
        Electric conductivity of the fill in S/m (default 0 — lossless).
        **Lossy dielectrics only**: a metal-scale sigma is not a model of a
        metal, see :mod:`wavesim.loss`. Mutually exclusive with ``pec``, and not
        supported together with ``subpixel``.
    subpixel : bool
        If True (dielectric only), place the box with **subpixel smoothing**:
        the true physical box edges are honoured and boundary cells receive the
        anisotropic effective permittivity (see :mod:`wavesim.subpixel`) instead
        of being snapped to whole cells. Restores ~2nd-order accuracy and makes
        results vary smoothly with the box size. Not supported with ``pec=True``.
    oversample : int
        Sub-samples per cell per axis used when ``subpixel=True`` (default 4).
    """
    _reject_conductive_pec(sigma if pec else 0.0, "set_box")

    if subpixel:
        if pec:
            raise NotImplementedError(
                "subpixel smoothing is for dielectrics only; PEC is a hard "
                "field constraint, not a material average (see wavesim.pec). "
                "Use pec=True without subpixel.")
        _reject_subpixel_sigma(sigma, "set_box")
        return smooth_shape_region(
            grid,
            lambda X, Y, Z: ((X >= x0) & (X <= x1) &
                             (Y >= y0) & (Y <= y1) &
                             (Z >= z0) & (Z <= z1)),
            eps_r, (x0, x1), (y0, y1), (z0, z1),
            oversample=oversample, mu_r=mu_r)

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
        if sigma or grid.is_lossy:
            # Write zeros too once the grid is lossy, so re-placing a lossless
            # material over a lossy region actually clears it.
            for arr in _sigma_arrays(grid):
                arr[sl] = sigma

    return grid


def set_cylinder(grid: FDTDGrid,
                 cx: float, cy: float,
                 radius: float,
                 z0: float, z1: float,
                 eps_r: float, mu_r: float = 1.0,
                 pec: bool = False, sigma: float = 0.0,
                 subpixel: bool = False, oversample: int = 4) -> FDTDGrid:
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
    sigma : float
        Electric conductivity of the fill in S/m (default 0 — lossless). Lossy
        dielectrics only; see :func:`set_box` and :mod:`wavesim.loss`.
    subpixel : bool
        If True (dielectric only), place the rod with **subpixel smoothing** so
        the curved boundary is anti-staircased with an anisotropic effective
        permittivity (see :mod:`wavesim.subpixel`). Not supported with ``pec=True``.
    oversample : int
        Sub-samples per cell per axis used when ``subpixel=True`` (default 4).
    """
    _reject_conductive_pec(sigma if pec else 0.0, "set_cylinder")

    if subpixel:
        if pec:
            raise NotImplementedError(
                "subpixel smoothing is for dielectrics only; PEC is a hard "
                "field constraint, not a material average (see wavesim.pec). "
                "Use pec=True without subpixel.")
        _reject_subpixel_sigma(sigma, "set_cylinder")
        return smooth_shape_region(
            grid,
            lambda X, Y, Z: (((X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2) &
                             (Z >= z0) & (Z <= z1)),
            eps_r, (cx - radius, cx + radius), (cy - radius, cy + radius),
            (z0, z1), oversample=oversample, mu_r=mu_r)

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
        sigmas = _sigma_arrays(grid) if (sigma or grid.is_lossy) else ()
        for k in range(k0, k1):
            grid.eps_x[:, :, k] = np.where(mask_2d, eps_r, grid.eps_x[:, :, k])
            grid.eps_y[:, :, k] = np.where(mask_2d, eps_r, grid.eps_y[:, :, k])
            grid.eps_z[:, :, k] = np.where(mask_2d, eps_r, grid.eps_z[:, :, k])
            grid.mu_x[:, :, k]  = np.where(mask_2d, mu_r,  grid.mu_x[:, :, k])
            grid.mu_y[:, :, k]  = np.where(mask_2d, mu_r,  grid.mu_y[:, :, k])
            grid.mu_z[:, :, k]  = np.where(mask_2d, mu_r,  grid.mu_z[:, :, k])
            for arr in sigmas:
                arr[:, :, k] = np.where(mask_2d, sigma, arr[:, :, k])

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
