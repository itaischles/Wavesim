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
    axis in ``axes`` — entry ``i`` along such an axis picks up entry ``i-1``.

    With two axes this is the union over the 2×2 block of cells that share an
    edge running along the *remaining* axis (:func:`build_pec_edge_masks`); with
    one it is the union over the pair of H faces an edge separates
    (:func:`_edges_bounding_covered_faces`)."""
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

    Conformal PEC
    -------------
    With cut-cell geometry present the dilation is **replaced** by the
    geometrically exact rule of :func:`build_conformal_edge_masks`: zero an edge
    iff none of it is open, or it bounds an H face that is itself fully covered.
    The dilation was a conservative over-zeroing that bought E/H consistency at
    the price of losing the sub-cell geometry; under conformal PEC that
    consistency comes from E and H being derived from the *same* cut geometry
    instead, which is the proper fix rather than the safe one. The second clause
    is part of being exact, not a retreat toward the dilation — see there.
    """
    if grid.is_conformal:
        return _apply_conformal_edge_mask(grid)

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


# An open fraction at or below this counts as *covered*. An exact ``== 0.0`` test
# is not enough: a conductor face lying **on** the grid ruler is a tangency, and a
# fraction generator — analytic or voxelised — resolves it to round-off, not to
# zero. Such an edge or face produces fractions like 1.7e-15, which is covered in
# every sense that matters but escapes the exact test. The threshold sits far
# below any fraction a real cut produces (the smallest on the reference coax is
# 0.062) and far above the ~1e-16 round-off floor, so it separates the two
# cleanly. :mod:`wavesim.mode_solver` re-exports it for the port machinery.
COVERED_FRACTION_TOL = 1e-9


def _edges_bounding_covered_faces(grid: FDTDGrid, tol: float) -> tuple:
    """Per-component E masks for edges that **bound a fully covered H face**.

    A face with zero open area lies wholly inside the conductor, so the four
    edges of its Faraday contour lie in the conductor's *closure* — either
    strictly inside it, or tangent to its surface. Tangential E vanishes on a PEC
    surface, so either way the edge carries no field.

    That second case is the one this rule exists for, because it is the one an
    edge's own open fraction cannot see. An edge running **along** a grid-aligned
    conductor face is not covered by the conductor at all — its open fraction is
    a full 1.0 — yet it lies in the surface and must be zero. The face it bounds
    on the metal side is what gives it away.

    The adjacency is :mod:`wavesim.update`'s: ``Hz[i,j,k]`` spans nodes
    ``(i..i+1, j..j+1)``, so its contour is ``Ex[i,j,k]``, ``Ex[i,j+1,k]``,
    ``Ey[i,j,k]``, ``Ey[i+1,j,k]``. Inverting that, ``Ex[i,j,k]`` bounds
    ``Hy[i,j,k-1..k]`` and ``Hz[i,j-1..j,k]`` — the two face families whose
    normals are perpendicular to x, each OR-ed one step back along the third
    axis. Hence one :func:`_dilate` per family.
    """
    cov_x = grid.pec_face_open_x <= tol
    cov_y = grid.pec_face_open_y <= tol
    cov_z = grid.pec_face_open_z <= tol
    return (_dilate(cov_y, (2,)) | _dilate(cov_z, (1,)),    # Ex — Hy, Hz
            _dilate(cov_z, (0,)) | _dilate(cov_x, (2,)),    # Ey — Hz, Hx
            _dilate(cov_x, (1,)) | _dilate(cov_y, (0,)))    # Ez — Hx, Hy


def build_conformal_edge_masks(grid: FDTDGrid,
                               tol: float = COVERED_FRACTION_TOL) -> tuple:
    """Per-component E masks from cut geometry: ``(ex, ey, ez)``.

    An edge is dead iff **none of it is open**, *or* it bounds an H face that is
    itself fully covered (:func:`_edges_bounding_covered_faces`). A partially
    covered edge with a live neighbouring face stays alive and carries the field
    on its open part — that is what the unknown means in the conformal
    formulation, and it is exactly the sub-cell information the staircase
    dilation threw away.

    The second clause is not a re-run of that dilation. It fires only where the
    conductor is **tangent to the grid**, which is where the first clause has a
    blind spot: an edge lying in a grid-aligned conductor surface is fully open
    by its own measure, and is nonetheless a tangential E on a PEC boundary. On a
    cut cell — a surface crossing a cell at an angle — no face is fully covered
    and the clause does nothing, so the sub-cell geometry survives untouched. At
    the other extreme, on an all-or-nothing (0/1) geometry whose faces are read
    the only way that geometry can read them — covered iff either cell the face
    separates is metal, since such a face lies in the closure of the metal — the
    clause reproduces :func:`build_pec_edge_masks` **exactly**, which is the right
    answer there. It has to: if a cell touching an edge is solid metal, one of
    the faces that edge bounds is a face of that cell.

    Leaving those edges alive is what broke a :class:`~wavesim.sources.ModalPort`
    on a conformal grid. The sheet's transverse divergence deposits the mode's
    induced surface charge onto its own plane every step; on a staircase grid the
    conductor swallows it, because the dilation held exactly those edges at zero.
    Without the guard it integrates with nothing to restore it: on the plan's
    reference coax the port-plane field reached 20× the physical field, static,
    and the whole line went with it. See ``ModalPort.apply_post_E`` for the same
    integral one plane over, and ``tests/test_conformal_tangent_edges.py``.

    It also removes a mismatch the conformal H update already had on its own. A
    fully covered face gets ``inv_A = 0`` from :func:`_guarded_inverse` and so
    freezes — while, before this, the edges of its contour kept carrying E. E and
    H now agree there, which is the same consistency argument that made clamping
    the right treatment for sliver faces.
    """
    covered = _edges_bounding_covered_faces(grid, tol)
    return (covered[0] | (grid.pec_edge_open_x <= tol),
            covered[1] | (grid.pec_edge_open_y <= tol),
            covered[2] | (grid.pec_edge_open_z <= tol))


def _apply_conformal_edge_mask(grid: FDTDGrid) -> FDTDGrid:
    """Zero E on fully covered edges; cached like the staircase edge masks.

    Keyed on all six fraction arrays, not just the three edge ones: the tangency
    clause of :func:`build_conformal_edge_masks` reads the face fractions too.
    """
    key = (grid.pec_edge_open_x, grid.pec_edge_open_y, grid.pec_edge_open_z,
           grid.pec_face_open_x, grid.pec_face_open_y, grid.pec_face_open_z)

    cache = getattr(grid, '_conformal_edge_cache', None)
    if cache is None or any(a is not b for a, b in zip(cache[0], key)):
        cache = (key,) + build_conformal_edge_masks(grid)
        grid._conformal_edge_cache = cache
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
    fraction is below ``grid.conformal_area_threshold`` has its area **clamped**
    to ``threshold·A_full``, bounding the coefficient without disturbing the
    contour. ``n_clamped`` counts how many faces that hit — worth watching, since
    every one is a cut whose geometry the run only partly resolved.
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
    n_clamped: int = 0


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
    inv_Ax, nx = _guarded_inverse(Ax, grid.pec_face_open_x, thr, dyp * dzp)
    inv_Ay, ny = _guarded_inverse(Ay, grid.pec_face_open_y, thr, dzp * dxp)
    inv_Az, nz = _guarded_inverse(Az, grid.pec_face_open_z, thr, dxp * dyp)

    return ConformalGeometry(
        Lx=grid.pec_edge_open_x * dxp,
        Ly=grid.pec_edge_open_y * dyp,
        Lz=grid.pec_edge_open_z * dzp,
        Ax=Ax, Ay=Ay, Az=Az,
        inv_Ax=inv_Ax, inv_Ay=inv_Ay, inv_Az=inv_Az,
        n_clamped=nx + ny + nz,
    )


def _guarded_inverse(area: np.ndarray, fraction: np.ndarray,
                     threshold: float, full_area: np.ndarray) -> tuple:
    """``(1/A_eff, n_clamped)`` — the bounded reciprocal open area.

    Sliver faces are handled by **clamping** the area to the threshold,
    ``A_eff = max(A_open, threshold·A_full)``, rather than by killing the face
    outright. Both bound the ``1/A_open`` coefficient and so both cure the
    small-cut instability, but only clamping keeps E and H looking at the same
    geometry.

    Killing the face (the plan's original wording, "treated as fully PEC")
    freezes H there while the four edges of its contour keep carrying E, because
    those edges are shared with neighbouring faces that are *not* suppressed and
    cannot simply be zeroed. That mismatch is exactly what the homogeneous-fill
    invariant detects: with it, ε_eff on the reference coax read +5.8% at
    threshold 0.4 and +24.0% at 0.5, tracking the suppressed-face count. Clamping
    brings both back to the staircase level, because the face still integrates
    its true contour — only the coefficient is limited.

    Thresholding the *fraction* rather than the area keeps the rule
    resolution-independent on a graded mesh, where ``A_full`` varies per cell.
    """
    open_face = fraction > 0.0
    a_eff = np.maximum(area, threshold * full_area)
    inv = np.divide(1.0, a_eff, out=np.zeros_like(area), where=open_face)
    n_clamped = int(np.count_nonzero(open_face & (fraction < threshold)))
    return inv, n_clamped


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


# ======================================================================= #
# Per-edge permittivity across a conductor surface
# ======================================================================= #

def _axis_slice(ndim: int, axis: int, lo, hi) -> tuple:
    """``a[..., lo:hi, ...]`` along ``axis`` of an ``ndim``-dimensional array."""
    sl = [slice(None)] * ndim
    sl[axis] = slice(lo, hi)
    return tuple(sl)


def outward_edge_eps(eps: np.ndarray, node_pec: np.ndarray,
                     axis: int) -> np.ndarray:
    """ε on each edge along ``axis``, with edges *straddling* metal taking the
    permittivity of the next edge **outward**.

    ``eps[i]`` is the value stored on the edge joining node ``i`` to node
    ``i+1`` along ``axis``; ``node_pec`` is
    :func:`~wavesim.parts.pec_node_mask`. Where exactly one of the two end nodes
    is on a conductor, the edge crosses the conductor surface and the ε the
    voxeliser left on it is not a material property at all — it is whatever
    filled the metal (typically 1.0). Such an edge borrows the ε of its
    neighbour on the *free* side, which lies wholly in the dielectric. Every
    other edge keeps its stored value: it already sits exactly where it is
    wanted, and averaging it with a neighbour smears an interface that is
    already in the right place (0.82% of C' on a two-layer parallel plate,
    against exact).

    Returns a new array of the same shape. The last edge along ``axis`` reaches
    a node the arrays do not carry, so it is left alone — the same ``N`` slots
    for ``N+1`` nodes convention :func:`~wavesim.parts.pec_node_mask` documents.

    This is the *whole* rule: :func:`conformal_edge_eps` applies it to the FDTD's
    material map and :func:`wavesim.mode_solver._face_eps` to the mode solver's
    face weights, and they call this one function so they cannot disagree. They
    used to disagree, and that is a class of bug with no symptom until a port is
    involved: the mode comes out of one material map, the leapfrog steps it on
    another, and ``ê`` stops being a null vector of the transverse curl that
    carries it. On a ModalPort's ghost plane — which runs open loop — the
    leftover then integrates into a static pile.
    """
    eps = np.asarray(eps, dtype=np.float64)
    n = eps.shape[axis]
    out = eps.copy()
    if n < 2:
        return out

    sl = lambda lo, hi: _axis_slice(eps.ndim, axis, lo, hi)       # noqa: E731
    lo_node = node_pec[sl(0, n - 1)]
    hi_node = node_pec[sl(1, n)]
    face = eps[sl(0, n - 1)]
    outward_hi = eps[sl(1, n)]                                    # edge i+1
    outward_lo = np.concatenate(                                  # edge i-1
        [eps[sl(0, 1)], eps[sl(0, n - 2)]], axis=axis)            # clamped at 0
    fixed = np.where(lo_node & ~hi_node, outward_hi, face)
    fixed = np.where(hi_node & ~lo_node, outward_lo, fixed)
    out[sl(0, n - 1)] = fixed
    return out


def conformal_edge_eps(grid: FDTDGrid) -> tuple:
    """``(eps_x, eps_y, eps_z)`` as the solver must read them on a cut-cell grid.

    On a conformal grid this is the stored map with :func:`outward_edge_eps`
    applied along each component's own axis; on a staircase grid it is the stored
    arrays themselves, unchanged and by identity — the dilation of
    :func:`build_pec_edge_masks` already holds every conductor-straddling edge at
    zero there, so their ε is never read and repairing it would only be a way to
    make the two paths differ.

    Cached on the grid, keyed on the *identity* of the three ε arrays and the six
    fraction arrays: the same scheme, and the same caveat, as
    :func:`conformal_geometry` — replacing an array invalidates the cache,
    mutating one in place does not.

    Everything that reads ε during a run goes through here: the E update
    (:func:`wavesim.update.update_E`, its lossy twin's coefficients, and the
    Numba kernels), and the wave impedance ``η`` that
    :mod:`wavesim.mode_solver` builds a port's ``ĥ`` sheet from. Both halves have
    to move together — repairing the material the leapfrog steps on while leaving
    the sheet built from the raw map trades one mismatch for another.
    """
    if not grid.is_conformal:
        return grid.eps_x, grid.eps_y, grid.eps_z

    arrays = (grid.eps_x, grid.eps_y, grid.eps_z,
              grid.pec_edge_open_x, grid.pec_edge_open_y, grid.pec_edge_open_z,
              grid.pec_face_open_x, grid.pec_face_open_y, grid.pec_face_open_z)

    cache = getattr(grid, '_conformal_eps_cache', None)
    if cache is None or any(a is not b for a, b in zip(cache[0], arrays)):
        from wavesim.parts import pec_node_mask       # circular at module level
        node = pec_node_mask(grid)
        eps = tuple(outward_edge_eps(getattr(grid, 'eps_' + c), node, ax)
                    for ax, c in enumerate('xyz'))
        cache = (arrays, eps)
        grid._conformal_eps_cache = cache
    return cache[1]


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
