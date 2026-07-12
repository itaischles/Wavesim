"""
subpixel.py — Subpixel smoothing of the permittivity at material interfaces.

Motivation
----------
A staircased (piecewise-constant, per-cell) permittivity makes the FDTD spatial
error only *first order* (O(Δx)) whenever a dielectric boundary does not fall on
a cell edge, and it makes derived quantities (resonant frequencies, S-params)
jump discontinuously as the geometry is nudged by less than a cell — fatal for
shape optimisation. Replacing the boundary cells with an *anisotropic effective
permittivity* restores (close to) second-order accuracy and makes results vary
smoothly with geometry. This is the technique used by Meep:

    https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/

The effective inverse-permittivity tensor for a locally planar interface with
unit normal ``n`` (Kottke, Farjadpour et al. 2006) is

    ε_eff⁻¹ = ⟨ε⁻¹⟩ · (n nᵀ)  +  ⟨ε⟩⁻¹ · (I − n nᵀ)

where ``⟨·⟩`` is the volume average over the cell. In words: the field component
*perpendicular* to the interface (along ``n``, where D_⊥ is continuous) sees the
**harmonic** mean 1/⟨ε⁻¹⟩; the components *parallel* to the interface (where E_∥
is continuous) see the **arithmetic** mean ⟨ε⟩. Both limits are recovered exactly
by the formula (n·n=1 → harmonic, n·n=0 → arithmetic).

This solver stores only a **diagonal** (per-axis) permittivity — ``eps_x`` seen by
``Ex``, etc. — so we cannot carry the full off-diagonal tensor. We therefore use
the diagonal of the inverse-tensor, which is the consistent reduction for a
component-wise ``E = ε⁻¹ D`` update:

    1/eps_d = ⟨ε⁻¹⟩ · n_d²  +  ⟨ε⟩⁻¹ · (1 − n_d²)          (d = x, y, z)

The two physically exact limits (field purely tangential / purely normal to the
interface) are reproduced regardless; only obliquely-cut cells — where the exact
answer is a genuinely non-diagonal tensor no diagonal solver can represent — are
approximated. The three components are co-located at the cell centre rather than
shifted to each Yee point; that sub-cell shift is a secondary refinement omitted
here (documented approximation appropriate to a diagonal engine).

The interface normal is estimated from the gradient of the finely-sampled
permittivity, averaged over the cell (∇ε points across the interface). Where the
cell is homogeneous the gradient is zero, ``n = 0`` and the formula returns the
(equal) arithmetic/harmonic value — i.e. smoothing is a no-op away from
boundaries, as it must be.

Public API
----------
``reduce_fine_eps``     — core reducer: fine scalar ε field → diagonal (εx,εy,εz).
``smooth_from_function``— sample a material function ε(x,y,z) over the whole grid
                          at ``oversample`` sub-samples/cell and reduce (writes
                          the grid in place unless ``write=False``).
``smooth_shape_region`` — smooth one analytic shape into a sub-block of the grid;
                          used by the ``subpixel=True`` path of the
                          :mod:`wavesim.materials` geometry helpers.

PEC / metals
------------
This module smooths **dielectrics** (finite, real ε). Perfect conductors are a
hard field constraint (tangential E ≡ 0), not a material average, so they are not
handled here — see :func:`wavesim.pec.apply_pec_mask` and the notes in the module
docstring of :mod:`wavesim.pec`.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_triplet(oversample):
    """Normalise ``oversample`` to an ``(ox, oy, oz)`` int tuple (>=1)."""
    if np.isscalar(oversample):
        o = int(oversample)
        trip = (o, o, o)
    else:
        trip = tuple(int(v) for v in oversample)
        if len(trip) != 3:
            raise ValueError("oversample must be an int or a length-3 sequence")
    if any(o < 1 for o in trip):
        raise ValueError("oversample factors must be >= 1")
    return trip


def _block_mean(a, os):
    """Mean of ``a`` over non-overlapping ``os = (ox,oy,oz)`` sub-blocks.

    ``a`` has shape ``(Nx*ox, Ny*oy, Nz*oz)``; the result is ``(Nx, Ny, Nz)``.
    """
    ox, oy, oz = os
    nx, ny, nz = a.shape[0] // ox, a.shape[1] // oy, a.shape[2] // oz
    return a.reshape(nx, ox, ny, oy, nz, oz).mean(axis=(1, 3, 5))


def _fine_gradient(a):
    """``∇a`` on the fine grid, one component per axis, robust to singleton axes.

    Returns ``(gx, gy, gz)`` each the shape of ``a``. Only direction matters
    (the normal is normalised later), so the index-space gradient is used; axes
    of length 1 (e.g. a 2D-in-3D ``Nz=1`` slice) contribute a zero component.
    """
    grads = []
    for ax in range(3):
        if a.shape[ax] > 1:
            grads.append(np.gradient(a, axis=ax))
        else:
            grads.append(np.zeros_like(a))
    return grads


# --------------------------------------------------------------------------- #
# Core reducer
# --------------------------------------------------------------------------- #
def reduce_fine_eps(eps_fine, oversample):
    """Reduce a finely-sampled scalar permittivity to a smoothed diagonal tensor.

    Parameters
    ----------
    eps_fine : ndarray, shape ``(Nx*ox, Ny*oy, Nz*oz)``
        Relative permittivity sampled on a uniform sub-grid, ``ox``/``oy``/``oz``
        samples per coarse Yee cell along each axis. Must be strictly positive.
    oversample : int or (int, int, int)
        Sub-samples per cell per axis (the ``ox, oy, oz`` above).

    Returns
    -------
    (eps_x, eps_y, eps_z) : tuple of ndarray, each shape ``(Nx, Ny, Nz)``
        The smoothed per-axis relative permittivity. In a homogeneous cell all
        three equal the (common) cell value; only interface cells differ.

    Notes
    -----
    Implements the diagonal of the Kottke effective-inverse-permittivity tensor
    (see the module docstring):

        1/eps_d = ⟨ε⁻¹⟩ n_d² + ⟨ε⟩⁻¹ (1 − n_d²)

    with ``n`` the unit interface normal from the sub-grid gradient of ε.
    """
    os = _as_triplet(oversample)
    eps_fine = np.asarray(eps_fine, dtype=np.float64)
    if np.any(eps_fine <= 0):
        raise ValueError("eps_fine must be strictly positive")

    mean_eps = _block_mean(eps_fine, os)            # ⟨ε⟩   (arithmetic)
    mean_inv = _block_mean(1.0 / eps_fine, os)      # ⟨ε⁻¹⟩ (→ harmonic mean)

    # Interface normal from the sub-grid gradient of ε, averaged over the cell.
    gx, gy, gz = _fine_gradient(eps_fine)
    nx = _block_mean(gx, os)
    ny = _block_mean(gy, os)
    nz = _block_mean(gz, os)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    scale = np.where(norm > 0.0, 1.0 / np.where(norm > 0.0, norm, 1.0), 0.0)
    nx *= scale
    ny *= scale
    nz *= scale

    inv_mean_eps = 1.0 / mean_eps                   # ⟨ε⟩⁻¹  (tangential inverse)

    def _component(n2):
        inv_d = mean_inv * n2 + inv_mean_eps * (1.0 - n2)
        return 1.0 / inv_d

    eps_x = _component(nx * nx)
    eps_y = _component(ny * ny)
    eps_z = _component(nz * nz)
    return eps_x, eps_y, eps_z


# --------------------------------------------------------------------------- #
# Fine-sampling helpers (support the non-uniform / rectilinear grid)
# --------------------------------------------------------------------------- #
def _fine_axis(nodes, widths, os, i0=0, i1=None):
    """Fine sample coordinates (cell-sub-centres) for cells ``[i0, i1)``.

    ``os`` sample points per cell are placed at the centres of ``os`` equal
    sub-intervals of each cell, so cell ``i`` (spanning ``nodes[i]..nodes[i+1]``,
    physical width ``widths[i]``) contributes ``nodes[i] + (m+0.5)/os * widths[i]``
    for ``m = 0..os-1``. Works for non-uniform cell widths.
    """
    if i1 is None:
        i1 = len(widths)
    frac = (np.arange(os) + 0.5) / os                       # (os,)
    left = nodes[i0:i1]                                      # (n,)
    w = widths[i0:i1]                                        # (n,)
    return (left[:, None] + frac[None, :] * w[:, None]).ravel()


def smooth_from_function(grid, eps_func, oversample=4, write=True):
    """Subpixel-smooth an arbitrary material function ε(x, y, z) onto ``grid``.

    This is the general entry point (and the one the future CAD importer can
    call with a nearest-neighbour sampler wrapping a high-resolution voxel
    array). The whole domain is sampled at ``oversample`` points/cell/axis, then
    reduced with :func:`reduce_fine_eps`.

    Parameters
    ----------
    grid : FDTDGrid
    eps_func : callable
        ``eps_func(X, Y, Z) -> eps`` — vectorised over broadcastable coordinate
        arrays in **metres**, returning the relative permittivity (> 0). It is
        called once with three ``(Nx*ox, Ny*oy, Nz*oz)`` meshgrids.
    oversample : int or (int, int, int), optional
        Sub-samples per cell per axis (default 4). Higher is more accurate but
        costs ``O(oversample³)`` memory/time at setup.
    write : bool, optional
        If True (default) assign the result to ``grid.eps_x/eps_y/eps_z`` and
        return the grid; otherwise leave the grid untouched and return the
        ``(eps_x, eps_y, eps_z)`` tuple.

    Returns
    -------
    FDTDGrid or (eps_x, eps_y, eps_z)
        The grid (``write=True``) or the smoothed arrays (``write=False``).
    """
    ox, oy, oz = _as_triplet(oversample)
    xf = _fine_axis(grid.x, grid.dxp, ox)
    yf = _fine_axis(grid.y, grid.dyp, oy)
    zf = _fine_axis(grid.z, grid.dzp, oz)
    X, Y, Z = np.meshgrid(xf, yf, zf, indexing='ij')
    eps_fine = np.asarray(eps_func(X, Y, Z), dtype=np.float64)
    if eps_fine.shape != X.shape:
        eps_fine = np.broadcast_to(eps_fine, X.shape)
    eps_x, eps_y, eps_z = reduce_fine_eps(eps_fine, (ox, oy, oz))

    if not write:
        return eps_x, eps_y, eps_z
    grid.eps_x[:] = eps_x
    grid.eps_y[:] = eps_y
    grid.eps_z[:] = eps_z
    return grid


# --------------------------------------------------------------------------- #
# Shape-local smoothing (used by the materials.py geometry helpers)
# --------------------------------------------------------------------------- #
def _cell_span(nodes, lo, hi, ncell, margin=1):
    """Half-open cell index range ``[a, b)`` covering physical ``[lo, hi]``.

    Expanded by ``margin`` cells on each side (so boundary cells keep valid
    fine-gradient neighbours) and clamped to ``[0, ncell]``.
    """
    centres = 0.5 * (nodes[:-1] + nodes[1:])
    inside = np.where((centres >= lo - (nodes[1:] - nodes[:-1])) &
                      (centres <= hi + (nodes[1:] - nodes[:-1])))[0]
    if inside.size == 0:
        # Shape falls between cell centres — still touch the nearest cell.
        a = int(np.clip(np.searchsorted(centres, 0.5 * (lo + hi)), 0, ncell - 1))
        return max(0, a - margin), min(ncell, a + 1 + margin)
    a = max(0, int(inside[0]) - margin)
    b = min(ncell, int(inside[-1]) + 1 + margin)
    return a, b


def smooth_shape_region(grid, inside_fn, eps_r,
                        x_range, y_range, z_range,
                        oversample=4, mu_r=1.0):
    """Smooth one analytic shape (uniform ``eps_r`` fill) into ``grid`` in place.

    The shape is defined by ``inside_fn(X, Y, Z) -> bool`` (vectorised, metres).
    Only the sub-block of cells overlapping the shape's bounding box (plus a
    1-cell margin) is touched: cells fully inside the shape become exactly
    ``eps_r``, cells fully outside keep the existing (background) permittivity,
    and boundary cells receive the Kottke-smoothed anisotropic value. The
    background inside the block is taken from the current ``grid.eps_x`` (assumed
    locally uniform per cell — true for the scaffolding geometry helpers).

    ``mu_r`` (if != 1) is applied by simple volume-fraction (arithmetic) averaging
    of the fill; permeability interfaces are rare and not tensor-smoothed here.
    """
    ox, oy, oz = _as_triplet(oversample)

    ia, ib = _cell_span(grid.x, *x_range, grid.Nx)
    ja, jb = _cell_span(grid.y, *y_range, grid.Ny)
    ka, kb = _cell_span(grid.z, *z_range, grid.Nz)

    xf = _fine_axis(grid.x, grid.dxp, ox, ia, ib)
    yf = _fine_axis(grid.y, grid.dyp, oy, ja, jb)
    zf = _fine_axis(grid.z, grid.dzp, oz, ka, kb)
    X, Y, Z = np.meshgrid(xf, yf, zf, indexing='ij')

    inside = np.asarray(inside_fn(X, Y, Z), dtype=bool)
    inside = np.broadcast_to(inside, X.shape)

    # Background sampled per coarse cell from the existing eps_x, tiled to fine.
    bg = grid.eps_x[ia:ib, ja:jb, ka:kb]
    bg_fine = np.repeat(np.repeat(np.repeat(bg, ox, axis=0), oy, axis=1), oz, axis=2)

    eps_fine = np.where(inside, float(eps_r), bg_fine)
    ex, ey, ez = reduce_fine_eps(eps_fine, (ox, oy, oz))

    grid.eps_x[ia:ib, ja:jb, ka:kb] = ex
    grid.eps_y[ia:ib, ja:jb, ka:kb] = ey
    grid.eps_z[ia:ib, ja:jb, ka:kb] = ez

    if mu_r != 1.0:
        frac = _block_mean(inside.astype(np.float64), (ox, oy, oz))
        mu_bg = grid.mu_x[ia:ib, ja:jb, ka:kb]
        mu_eff = frac * float(mu_r) + (1.0 - frac) * mu_bg
        grid.mu_x[ia:ib, ja:jb, ka:kb] = mu_eff
        grid.mu_y[ia:ib, ja:jb, ka:kb] = mu_eff
        grid.mu_z[ia:ib, ja:jb, ka:kb] = mu_eff

    return grid
