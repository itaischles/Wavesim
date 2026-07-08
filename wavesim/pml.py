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

Boundary-slab psi storage
-------------------------
Each psi corrects a derivative along ONE axis, and its (b, c) profile is
nonzero only within `d_pml` cells of that axis' absorbing faces. The recursion
is purely local (psi[j] depends only on psi[j] and dF[j]), and where c = 0 the
profile also has b = 1, so an interior psi cell initialised to 0 stays exactly
0 forever and contributes 0.0 to the field. We therefore store each psi array
compressed along its derivative axis to just those active boundary indices
(`sel_*` below) — typically ~2*d_pml cells instead of the full axis length,
which roughly halves the total solver footprint at large 3D sizes (the 12
psi arrays were the dominant term). The result is bit-identical to a
full-volume allocation because every cell we drop held an identical 0.0.

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

from wavesim.grid import FDTDGrid
from wavesim.constants import EPS0, MU0, ETA0


@dataclass
class CPMLArrays:
    # --- Auxiliary convolution variables for the H-field updates -------- #
    # (these correct E-derivatives that appear in the curl of E)
    # Each array is compressed along its derivative axis to the active PML
    # indices (sel_*H below): psi_Ez_y has shape (Nx, n_yH, Nz), etc.
    psi_Ez_y: np.ndarray   # correction to dEz/dy in the Hx update  (y axis)
    psi_Ey_z: np.ndarray   # correction to dEy/dz in the Hx update  (z axis)
    psi_Ex_z: np.ndarray   # correction to dEx/dz in the Hy update  (z axis)
    psi_Ez_x: np.ndarray   # correction to dEz/dx in the Hy update  (x axis)
    psi_Ey_x: np.ndarray   # correction to dEy/dx in the Hz update  (x axis)
    psi_Ex_y: np.ndarray   # correction to dEx/dy in the Hz update  (y axis)

    # --- Auxiliary convolution variables for the E-field updates -------- #
    # (these correct H-derivatives that appear in the curl of H; compressed
    #  along their derivative axis to the active PML indices sel_*E)
    psi_Hz_y: np.ndarray   # correction to dHz/dy in the Ex update  (y axis)
    psi_Hy_z: np.ndarray   # correction to dHy/dz in the Ex update  (z axis)
    psi_Hx_z: np.ndarray   # correction to dHx/dz in the Ey update  (z axis)
    psi_Hz_x: np.ndarray   # correction to dHz/dx in the Ey update  (x axis)
    psi_Hy_x: np.ndarray   # correction to dHy/dx in the Ez update  (x axis)
    psi_Hx_y: np.ndarray   # correction to dHx/dy in the Ez update  (y axis)

    # --- Precomputed (b, c) profiles, one 1D array per axis & grid ------ #
    # Full-length profiles (cheap 1D arrays) kept for visualisation/inspection.
    bx_E: np.ndarray; cx_E: np.ndarray   # shape (Nx,)
    bx_H: np.ndarray; cx_H: np.ndarray
    by_E: np.ndarray; cy_E: np.ndarray   # shape (Ny,)
    by_H: np.ndarray; cy_H: np.ndarray
    bz_E: np.ndarray; cz_E: np.ndarray   # shape (Nz,)
    bz_H: np.ndarray; cz_H: np.ndarray

    # --- Active boundary-slab indices along each (axis, grid) ----------- #
    # sel_aG holds the indices along axis a (for the G in {E,H} grid) where the
    # profile is nonzero — i.e. the two PML slabs. The psi arrays are stored at
    # exactly these indices along their derivative axis.
    sel_xH: np.ndarray; sel_yH: np.ndarray; sel_zH: np.ndarray
    sel_xE: np.ndarray; sel_yE: np.ndarray; sel_zE: np.ndarray

    # --- (b, c) sampled at sel_*, reshaped to broadcast along that axis -- #
    bxH_s: np.ndarray; cxH_s: np.ndarray     # shape (n_xH, 1, 1)
    byH_s: np.ndarray; cyH_s: np.ndarray     # shape (1, n_yH, 1)
    bzH_s: np.ndarray; czH_s: np.ndarray     # shape (1, 1, n_zH)
    bxE_s: np.ndarray; cxE_s: np.ndarray     # shape (n_xE, 1, 1)
    byE_s: np.ndarray; cyE_s: np.ndarray     # shape (1, n_yE, 1)
    bzE_s: np.ndarray; czE_s: np.ndarray     # shape (1, 1, n_zE)

    # --- Per-cell spacing sampled at sel_*, reshaped to broadcast (Session 4) -- #
    # The CPML correction is part of the SAME curl as the base field update, so
    # its derivative divisor must match update.py cell-for-cell: DUAL widths on
    # the H-side psi (which correct E-derivatives, like update_H) and PRIMARY
    # widths on the E-side psi (which correct H-derivatives, like update_E). Each
    # is sampled at that face's slab indices and reshaped to broadcast along the
    # derivative axis. On a uniform grid they are the constant PML spacing, so
    # this is bit-identical to the pre-rehaul scalar ``/ds`` divisor.
    #   H-side sampled at sel_*H          -> dual width dd[sel]
    #   E-side sampled at sel_*E - 1      -> primary width dp[sel-1]  (Yee [1:] cell
    #                                        j is driven by the diff at output j-1)
    dxd_sH: np.ndarray; dyd_sH: np.ndarray; dzd_sH: np.ndarray   # H-side (dual)
    dxp_sE: np.ndarray; dyp_sE: np.ndarray; dzp_sE: np.ndarray   # E-side (primary)

    d_pml: int             # PML thickness in cells


# ---------------------------------------------------------------------- #
# Profile construction
# ---------------------------------------------------------------------- #
def _pml_face_widths(dp, d_pml, low, high, axis):
    """(ds_lo, ds_hi): the constant PML cell width at the low- and high-index
    absorbing faces of one axis, read off the primary spacing array ``dp``.

    Non-uniform CPML relies on the **uniform-PML constraint**: the outermost
    ``d_pml`` cells of each absorbing face have constant spacing (the future mesh
    generator guarantees this; it keeps the delicate grading math essentially
    unchanged). The solver still handles a different constant per face, but not a
    graded PML — so we assert the constraint holds and fail loudly on a mesh that
    violates it instead of silently mis-grading the absorber. A per-cell
    ``sigma_max`` + cumulative-sum profile would lift this restriction; it is a
    documented follow-on, not built here.

    Only enabled faces on a thick-enough axis are checked; the edge widths
    ``dp[0]`` / ``dp[-1]`` are returned regardless (unused where the profile is
    inert, e.g. an Nz=1 slice).
    """
    N = len(dp)
    ds_lo = float(dp[0])
    ds_hi = float(dp[-1])
    if N > 2 * d_pml:
        if low and not np.allclose(dp[:d_pml], ds_lo):
            raise ValueError(
                f"Non-uniform CPML requires a uniform PML shell: the {d_pml} "
                f"cells at the low {axis}-face are not constant width "
                f"({dp[:d_pml].min():.3e}..{dp[:d_pml].max():.3e} m). Keep the "
                f"outer d_pml cells at constant spacing per axis.")
        if high and not np.allclose(dp[N - d_pml:], ds_hi):
            raise ValueError(
                f"Non-uniform CPML requires a uniform PML shell: the {d_pml} "
                f"cells at the high {axis}-face are not constant width "
                f"({dp[N - d_pml:].min():.3e}..{dp[N - d_pml:].max():.3e} m). "
                f"Keep the outer d_pml cells at constant spacing per axis.")
    return ds_lo, ds_hi


def _calc_profile_1d(N, ds_lo, ds_hi, dt, d_pml, staggered, low=True, high=True):
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

    Non-uniform grid (Session 4)
    ----------------------------
    ``ds_lo`` / ``ds_hi`` are the (constant) PML cell widths at the low- and
    high-index faces (from :func:`_pml_face_widths`). Under the uniform-PML
    constraint the outermost ``d_pml`` cells of each face are constant, so the
    grading *depth* is purely index-based — the depth ratios below cancel ``ds``
    exactly — and the only place the physical width enters is
    ``sigma_max = 0.8*(m+1)/(ETA0*ds)``. Each slab therefore uses its own face
    width. On a uniform grid ``ds_lo == ds_hi`` equal today's scalar spacing, so
    this reproduces the pre-rehaul profile bit-for-bit.

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
    alpha_max = 0.05
    idx = np.arange(N)

    sigma = np.zeros(N)
    alpha = np.zeros(N)

    # Left slab (low-index face): depth grows towards the boundary (coord -> 0).
    # The E-field derivatives this profile corrects are centred half a cell to
    # the LEFT of the index (e.g. Ez[i] is driven by Hy[i]-Hy[i-1], centred at
    # x=(i-0.5)*ds), so the staggered grid uses a -0.5 offset — matching the
    # differencing in update.py. Using +0.5 here mis-aligns the two PML slabs in
    # opposite directions and produces phase-reversed residual reflections. clip
    # handles the unused over-range staggered node at idx=0 (coord < 0).
    if low:
        sigma_max = 0.8 * (m + 1) / (ETA0 * ds_lo)
        d_pml_len = d_pml * ds_lo
        left_edge = d_pml * ds_lo
        coord = (idx - 0.5) * ds_lo if staggered else idx * ds_lo
        left = coord < left_edge
        depth_l = np.clip((left_edge - coord[left]) / d_pml_len, 0.0, 1.0)
        sigma[left] = sigma_max * depth_l ** m
        alpha[left] = alpha_max * (1.0 - depth_l)

    # Right slab (high-index face): depth grows towards the boundary (coord -> N*ds).
    if high:
        sigma_max = 0.8 * (m + 1) / (ETA0 * ds_hi)
        d_pml_len = d_pml * ds_hi
        right_edge = (N - d_pml) * ds_hi
        coord = (idx - 0.5) * ds_hi if staggered else idx * ds_hi
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


def _slab_indices(c, grid_type):
    """
    Active boundary-slab indices along one axis: the positions where the
    profile c is nonzero, within the index range that axis' updates actually
    touch. H-grid updates use the region [0, N-1) (a trailing `[:-1]` slice);
    E-grid updates use [1, N) (a leading `[1:]` slice). Indices outside the
    nonzero-c set carry psi == 0 identically, so dropping them is exact.
    """
    N = len(c)
    if grid_type == 'H':
        return np.nonzero(c[:-1])[0].astype(np.intp)            # in [0, N-1)
    return (np.nonzero(c[1:])[0] + 1).astype(np.intp)           # in [1, N)


ALL_FACES = ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')


def init_cpml(grid: FDTDGrid, d_pml: int = 10,
              faces: tuple = ALL_FACES) -> CPMLArrays:
    """
    Allocate the psi convolution arrays and precompute the (b, c) profiles
    for every axis on both the E (staggered) and H (non-staggered) grids.

    The psi arrays are stored as boundary slabs: each is full-size on the two
    axes orthogonal to its derivative, and compressed along its derivative axis
    to just the active PML indices (sel_*). This is bit-identical to a
    full-volume allocation (the dropped cells hold 0 forever) but roughly halves
    the solver's memory footprint at large 3D sizes.

    Parameters
    ----------
    faces : tuple of str
        Which domain faces are absorbing CPML. Any subset of
        ('x0','x1','y0','y1','z0','z1'). Faces left out are transparent — use
        this when a face is a PEC wall or symmetry plane (e.g. a waveguide with
        PEC side walls absorbs only on the propagation-axis faces:
        faces=('x0','x1')). Defaults to all six faces.
        'x0' = the face at i=0, 'x1' = the face at i=Nx-1, etc.

    Notes
    -----
    The z-axis profiles return (1, 0) when Nz <= 2*d_pml, so the z-face PML is
    inert for Nz=1 slices (sel_zE/sel_zH are empty) and activates on its own
    once Nz is large enough (subject to z0/z1 being in `faces`).
    """
    bad = set(faces) - set(ALL_FACES)
    if bad:
        raise ValueError(f"Unknown face(s) {sorted(bad)}. "
                         f"Must be a subset of {ALL_FACES}.")

    # Map each axis to (low-index face enabled, high-index face enabled).
    x_lo, x_hi = 'x0' in faces, 'x1' in faces
    y_lo, y_hi = 'y0' in faces, 'y1' in faces
    z_lo, z_hi = 'z0' in faces, 'z1' in faces

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    # PML cell width per face, from the primary spacing arrays (the true cell
    # widths). On a uniform grid these equal grid.dx/dy/dz exactly; on a
    # non-uniform grid grid.dx is the *minimum* interior width and must NOT be
    # used here — the profile grades against the uniform PML shell, not the fine
    # interior. _pml_face_widths also asserts that shell is uniform.
    dsx_lo, dsx_hi = _pml_face_widths(grid.dxp, d_pml, x_lo, x_hi, 'x')
    dsy_lo, dsy_hi = _pml_face_widths(grid.dyp, d_pml, y_lo, y_hi, 'y')
    dsz_lo, dsz_hi = _pml_face_widths(grid.dzp, d_pml, z_lo, z_hi, 'z')

    bx_E, cx_E = _calc_profile_1d(Nx, dsx_lo, dsx_hi, grid.dt, d_pml, True,  x_lo, x_hi)
    bx_H, cx_H = _calc_profile_1d(Nx, dsx_lo, dsx_hi, grid.dt, d_pml, False, x_lo, x_hi)
    by_E, cy_E = _calc_profile_1d(Ny, dsy_lo, dsy_hi, grid.dt, d_pml, True,  y_lo, y_hi)
    by_H, cy_H = _calc_profile_1d(Ny, dsy_lo, dsy_hi, grid.dt, d_pml, False, y_lo, y_hi)
    bz_E, cz_E = _calc_profile_1d(Nz, dsz_lo, dsz_hi, grid.dt, d_pml, True,  z_lo, z_hi)
    bz_H, cz_H = _calc_profile_1d(Nz, dsz_lo, dsz_hi, grid.dt, d_pml, False, z_lo, z_hi)

    # Active boundary-slab indices along each (axis, grid).
    sel_xH = _slab_indices(cx_H, 'H'); sel_xE = _slab_indices(cx_E, 'E')
    sel_yH = _slab_indices(cy_H, 'H'); sel_yE = _slab_indices(cy_E, 'E')
    sel_zH = _slab_indices(cz_H, 'H'); sel_zE = _slab_indices(cz_E, 'E')

    # psi arrays and the sampled coefficients follow the grid's storage dtype
    # (float32 for the GPU path); the full 1D profiles above stay float64 for
    # inspection. The profiles are built in float64 and cast on sampling so the
    # exp()/division that shapes them keeps full precision.
    dtype = grid.Ex.dtype

    # (b, c) sampled at the slab indices, reshaped to broadcast along the axis.
    def _rs(arr, sel, axis):
        shape = [1, 1, 1]; shape[axis] = -1
        return arr[sel].reshape(shape).astype(dtype, copy=False)

    bxH_s, cxH_s = _rs(bx_H, sel_xH, 0), _rs(cx_H, sel_xH, 0)
    byH_s, cyH_s = _rs(by_H, sel_yH, 1), _rs(cy_H, sel_yH, 1)
    bzH_s, czH_s = _rs(bz_H, sel_zH, 2), _rs(cz_H, sel_zH, 2)
    bxE_s, cxE_s = _rs(bx_E, sel_xE, 0), _rs(cx_E, sel_xE, 0)
    byE_s, cyE_s = _rs(by_E, sel_yE, 1), _rs(cy_E, sel_yE, 1)
    bzE_s, czE_s = _rs(bz_E, sel_zE, 2), _rs(cz_E, sel_zE, 2)

    # Per-cell spacing divisors sampled at the slab indices (Session 4). The CPML
    # correction shares the base update's curl, so it uses the same divisor: dual
    # widths on the H-side (sampled at sel_*H, like update_H's dyd[sy]) and
    # primary widths on the E-side (sampled at sel_*E - 1, since the Yee [1:] cell
    # j is driven by the diff at output index j-1, like update_E's dyp[sy-1]).
    dxd_sH = _rs(grid.dxd, sel_xH, 0); dyd_sH = _rs(grid.dyd, sel_yH, 1)
    dzd_sH = _rs(grid.dzd, sel_zH, 2)
    dxp_sE = _rs(grid.dxp, sel_xE - 1, 0); dyp_sE = _rs(grid.dyp, sel_yE - 1, 1)
    dzp_sE = _rs(grid.dzp, sel_zE - 1, 2)

    # Allocate each psi slab: full on the two orthogonal axes, len(sel) on the
    # derivative axis.
    nxH, nyH, nzH = len(sel_xH), len(sel_yH), len(sel_zH)
    nxE, nyE, nzE = len(sel_xE), len(sel_yE), len(sel_zE)
    z = lambda shape: np.zeros(shape, dtype=dtype)

    return CPMLArrays(
        # H-update psi (E-derivative corrections), compressed on their axis
        psi_Ez_y=z((Nx, nyH, Nz)), psi_Ey_z=z((Nx, Ny, nzH)),
        psi_Ex_z=z((Nx, Ny, nzH)), psi_Ez_x=z((nxH, Ny, Nz)),
        psi_Ey_x=z((nxH, Ny, Nz)), psi_Ex_y=z((Nx, nyH, Nz)),
        # E-update psi (H-derivative corrections)
        psi_Hz_y=z((Nx, nyE, Nz)), psi_Hy_z=z((Nx, Ny, nzE)),
        psi_Hx_z=z((Nx, Ny, nzE)), psi_Hz_x=z((nxE, Ny, Nz)),
        psi_Hy_x=z((nxE, Ny, Nz)), psi_Hx_y=z((Nx, nyE, Nz)),
        # Full 1D profiles
        bx_E=bx_E, cx_E=cx_E, bx_H=bx_H, cx_H=cx_H,
        by_E=by_E, cy_E=cy_E, by_H=by_H, cy_H=cy_H,
        bz_E=bz_E, cz_E=cz_E, bz_H=bz_H, cz_H=cz_H,
        # Slab indices and sampled coefficients
        sel_xH=sel_xH, sel_yH=sel_yH, sel_zH=sel_zH,
        sel_xE=sel_xE, sel_yE=sel_yE, sel_zE=sel_zE,
        bxH_s=bxH_s, cxH_s=cxH_s, byH_s=byH_s, cyH_s=cyH_s,
        bzH_s=bzH_s, czH_s=czH_s,
        bxE_s=bxE_s, cxE_s=cxE_s, byE_s=byE_s, cyE_s=cyE_s,
        bzE_s=bzE_s, czE_s=czE_s,
        # Sampled per-cell spacing divisors
        dxd_sH=dxd_sH, dyd_sH=dyd_sH, dzd_sH=dzd_sH,
        dxp_sE=dxp_sE, dyp_sE=dyp_sE, dzp_sE=dzp_sE,
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

    psi arrays are boundary slabs (see module docstring): the derivative axis
    carries only the active PML indices sel_*H, so the recursion and correction
    are restricted to those indices via fancy indexing — bit-identical to the
    full-volume formulation since every dropped cell held 0.
    """
    dt = grid.dt
    Nz = grid.Nz
    sx, sy, sz = cpml.sel_xH, cpml.sel_yH, cpml.sel_zH
    # Non-uniform divisors: every E-derivative in the H-side curl is divided by
    # the DUAL width, sampled at the slab indices (Session 4). On a uniform grid
    # these are the constant PML spacing → bit-identical to the old scalar /ds.
    dxd, dyd, dzd = cpml.dxd_sH, cpml.dyd_sH, cpml.dzd_sH

    # ---------- Hx  (curl term: dEz/dy - dEy/dz) ----------
    # dEz/dy lives at the Hx location -> non-staggered (H) profile along y.
    dEz_dy = (grid.Ez[:, sy + 1, :] - grid.Ez[:, sy, :]) / dyd        # (Nx, n_yH, Nz)
    cpml.psi_Ez_y = cpml.byH_s * cpml.psi_Ez_y + cpml.cyH_s * dEz_dy

    if Nz > 1:
        grid.Hx[:, sy, :-1] -= (dt / (MU0 * grid.mu_x[:, sy, :-1])) \
            * cpml.psi_Ez_y[:, :, :-1]

        dEy_dz = (grid.Ey[:, :, sz + 1] - grid.Ey[:, :, sz]) / dzd    # (Nx, Ny, n_zH)
        cpml.psi_Ey_z = cpml.bzH_s * cpml.psi_Ey_z + cpml.czH_s * dEy_dz
        grid.Hx[:, :-1, sz] += (dt / (MU0 * grid.mu_x[:, :-1, sz])) \
            * cpml.psi_Ey_z[:, :-1, :]
    else:
        grid.Hx[:, sy, :] -= (dt / (MU0 * grid.mu_x[:, sy, :])) * cpml.psi_Ez_y

    # ---------- Hy  (curl term: dEx/dz - dEz/dx) ----------
    # dEz/dx lives at the Hy location -> non-staggered (H) profile along x.
    dEz_dx = (grid.Ez[sx + 1, :, :] - grid.Ez[sx, :, :]) / dxd        # (n_xH, Ny, Nz)
    cpml.psi_Ez_x = cpml.bxH_s * cpml.psi_Ez_x + cpml.cxH_s * dEz_dx

    if Nz > 1:
        grid.Hy[sx, :, :-1] += (dt / (MU0 * grid.mu_y[sx, :, :-1])) \
            * cpml.psi_Ez_x[:, :, :-1]

        dEx_dz = (grid.Ex[:, :, sz + 1] - grid.Ex[:, :, sz]) / dzd    # (Nx, Ny, n_zH)
        cpml.psi_Ex_z = cpml.bzH_s * cpml.psi_Ex_z + cpml.czH_s * dEx_dz
        grid.Hy[:-1, :, sz] -= (dt / (MU0 * grid.mu_y[:-1, :, sz])) \
            * cpml.psi_Ex_z[:-1, :, :]
    else:
        grid.Hy[sx, :, :] += (dt / (MU0 * grid.mu_y[sx, :, :])) * cpml.psi_Ez_x

    # ---------- Hz  (curl term: dEy/dx - dEx/dy) ----------
    # dEy/dx lives at the Hz location -> non-staggered (H) profile along x.
    dEy_dx = (grid.Ey[sx + 1, :, :] - grid.Ey[sx, :, :]) / dxd        # (n_xH, Ny, Nz)
    cpml.psi_Ey_x = cpml.bxH_s * cpml.psi_Ey_x + cpml.cxH_s * dEy_dx
    grid.Hz[sx, :-1, :] -= (dt / (MU0 * grid.mu_z[sx, :-1, :])) \
        * cpml.psi_Ey_x[:, :-1, :]

    # dEx/dy lives at the Hz location -> non-staggered (H) profile along y.
    dEx_dy = (grid.Ex[:, sy + 1, :] - grid.Ex[:, sy, :]) / dyd        # (Nx, n_yH, Nz)
    cpml.psi_Ex_y = cpml.byH_s * cpml.psi_Ex_y + cpml.cyH_s * dEx_dy
    grid.Hz[:-1, sy, :] += (dt / (MU0 * grid.mu_z[:-1, sy, :])) \
        * cpml.psi_Ex_y[:-1, :, :]

    return grid, cpml


# ---------------------------------------------------------------------- #
# E-field CPML correction
# ---------------------------------------------------------------------- #
def update_E_pml(grid: FDTDGrid, cpml: CPMLArrays) -> tuple[FDTDGrid, CPMLArrays]:
    """
    Advance the H-derivative psi arrays and add their correction onto the E
    fields that update_E has already advanced. Coefficient is dt/(EPS0*eps) to
    match update.py exactly.

    psi arrays are boundary slabs (see module docstring): the derivative axis
    carries only the active PML indices sel_*E. Each slab index j holds the
    derivative centred between samples j-1 and j (the staggered [1:] region),
    so the differences below read (F[sel] - F[sel-1]).
    """
    dt = grid.dt
    Nz = grid.Nz
    sx, sy, sz = cpml.sel_xE, cpml.sel_yE, cpml.sel_zE
    # Non-uniform divisors: every H-derivative in the E-side curl is divided by
    # the PRIMARY width, sampled at sel_* - 1 (Session 4). On a uniform grid these
    # are the constant PML spacing → bit-identical to the old scalar /ds.
    dxp, dyp, dzp = cpml.dxp_sE, cpml.dyp_sE, cpml.dzp_sE

    # ---------- Ex  (curl term: dHz/dy - dHy/dz) ----------
    # dHz/dy lives at the Ex location -> staggered (E) profile along y.
    dHz_dy = (grid.Hz[:, sy, :] - grid.Hz[:, sy - 1, :]) / dyp        # (Nx, n_yE, Nz)
    cpml.psi_Hz_y = cpml.byE_s * cpml.psi_Hz_y + cpml.cyE_s * dHz_dy

    if Nz > 1:
        grid.Ex[:, sy, 1:] += (dt / (EPS0 * grid.eps_x[:, sy, 1:])) \
            * cpml.psi_Hz_y[:, :, 1:]

        dHy_dz = (grid.Hy[:, :, sz] - grid.Hy[:, :, sz - 1]) / dzp    # (Nx, Ny, n_zE)
        cpml.psi_Hy_z = cpml.bzE_s * cpml.psi_Hy_z + cpml.czE_s * dHy_dz
        grid.Ex[:, 1:, sz] -= (dt / (EPS0 * grid.eps_x[:, 1:, sz])) \
            * cpml.psi_Hy_z[:, 1:, :]
    else:
        grid.Ex[:, sy, :] += (dt / (EPS0 * grid.eps_x[:, sy, :])) * cpml.psi_Hz_y

    # ---------- Ey  (curl term: dHx/dz - dHz/dx) ----------
    # dHz/dx lives at the Ey location -> staggered (E) profile along x.
    dHz_dx = (grid.Hz[sx, :, :] - grid.Hz[sx - 1, :, :]) / dxp        # (n_xE, Ny, Nz)
    cpml.psi_Hz_x = cpml.bxE_s * cpml.psi_Hz_x + cpml.cxE_s * dHz_dx

    if Nz > 1:
        grid.Ey[sx, :, 1:] -= (dt / (EPS0 * grid.eps_y[sx, :, 1:])) \
            * cpml.psi_Hz_x[:, :, 1:]

        dHx_dz = (grid.Hx[:, :, sz] - grid.Hx[:, :, sz - 1]) / dzp    # (Nx, Ny, n_zE)
        cpml.psi_Hx_z = cpml.bzE_s * cpml.psi_Hx_z + cpml.czE_s * dHx_dz
        grid.Ey[1:, :, sz] += (dt / (EPS0 * grid.eps_y[1:, :, sz])) \
            * cpml.psi_Hx_z[1:, :, :]
    else:
        grid.Ey[sx, :, :] -= (dt / (EPS0 * grid.eps_y[sx, :, :])) * cpml.psi_Hz_x

    # ---------- Ez  (curl term: dHy/dx - dHx/dy) ----------
    # dHy/dx lives at the Ez location -> staggered (E) profile along x.
    dHy_dx = (grid.Hy[sx, :, :] - grid.Hy[sx - 1, :, :]) / dxp        # (n_xE, Ny, Nz)
    cpml.psi_Hy_x = cpml.bxE_s * cpml.psi_Hy_x + cpml.cxE_s * dHy_dx
    grid.Ez[sx, 1:, :] += (dt / (EPS0 * grid.eps_z[sx, 1:, :])) \
        * cpml.psi_Hy_x[:, 1:, :]

    # dHx/dy lives at the Ez location -> staggered (E) profile along y.
    dHx_dy = (grid.Hx[:, sy, :] - grid.Hx[:, sy - 1, :]) / dyp        # (Nx, n_yE, Nz)
    cpml.psi_Hx_y = cpml.byE_s * cpml.psi_Hx_y + cpml.cyE_s * dHx_dy
    grid.Ez[1:, sy, :] -= (dt / (EPS0 * grid.eps_z[1:, sy, :])) \
        * cpml.psi_Hx_y[1:, :, :]

    return grid, cpml
