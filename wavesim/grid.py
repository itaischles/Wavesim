"""
grid.py — FDTDGrid dataclass and construction helpers.

The FDTDGrid is the central state object passed to every function in the
FDTD engine. All field and material data live here.

Design decisions:
- All arrays are full 3D: shape (Nx, Ny, Nz), even for Nz=1 slices.
- dt is always computed from the full 3D CFL condition (correct for Nz>1 too).
- Material arrays are split into x/y/z components to support future
  anisotropic materials.

Non-uniform (rectilinear) grid — Session 2 of ``docs/nonuniform_grid_plan.md``
------------------------------------------------------------------------------
The grid is now **rectilinear**: cell widths may vary per-axis, per-index. The
geometric source of truth is the node-coordinate array per axis (``x``/``y``/``z``,
strictly increasing, length ``N+1`` for ``N`` cells) — this is where the future
snapping / mesh generator will write.

From the coordinates we precompute two 1-D spacing arrays per axis, following the
Yee-layout result in the plan:

- **primary widths** ``dxp[i] = x[i+1] - x[i]``  (length ``N``) — the denominator
  for every ``update_E`` derivative (differences an integer-node H field).
- **dual widths** ``dxd[i] = (dxp[i] + dxp[i+1]) / 2``  (length ``N``, the last
  entry replicates ``dxp[-1]`` as boundary padding) — the denominator for every
  ``update_H`` derivative (differences a half-node E field).

Center (half-node) coordinates ``xc[i] = (x[i] + x[i+1]) / 2`` are stored for
index lookups and the PML.

**Uniform grids reproduce today's results bit-for-bit.** ``create_grid`` builds
the spacing arrays as exact constants (``np.full(N, ds)``) rather than by
subtracting node coordinates, so ``dxp == dxd == ds`` to the last ULP and dt /
the update coefficients are byte-identical to the pre-rehaul scalar path. The
scalar ``dx``/``dy``/``dz`` fields are retained for backward compatibility: on a
uniform grid they equal the spacing exactly; on a non-uniform grid they hold the
*minimum* width per axis and must **not** be used for per-cell math — use the
spacing arrays / :meth:`FDTDGrid.cell_volume` instead.
"""

from dataclasses import dataclass, field
import numpy as np
from wavesim.constants import C0

# Conservative 3D CFL stability factor (Taflove eq. 4.78).
_CFL = 0.99


@dataclass
class FDTDGrid:
    # ------------------------------------------------------------------ #
    # Field arrays — shape (Nx, Ny, Nz) always
    # ------------------------------------------------------------------ #
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Hx: np.ndarray
    Hy: np.ndarray
    Hz: np.ndarray

    # ------------------------------------------------------------------ #
    # Material arrays — shape (Nx, Ny, Nz)
    # Stored as separate x/y/z tensors for future anisotropic support.
    # eps_x is the permittivity seen by Ex, etc.
    # ------------------------------------------------------------------ #
    eps_x: np.ndarray   # relative permittivity seen by Ex
    eps_y: np.ndarray
    eps_z: np.ndarray
    mu_x: np.ndarray    # relative permeability seen by Hx
    mu_y: np.ndarray
    mu_z: np.ndarray

    # ------------------------------------------------------------------ #
    # Grid spacing (metres) — scalar, retained for backward compatibility.
    # Uniform grid: equals the (constant) spacing exactly.
    # Non-uniform grid: the MINIMUM width per axis — do not use for per-cell
    # math; use the spacing arrays (dxp/dxd …) or cell_volume() instead.
    # ------------------------------------------------------------------ #
    dx: float
    dy: float
    dz: float
    dt: float           # Computed from CFL condition — do not set manually

    # ------------------------------------------------------------------ #
    # Domain size (stored for convenience — derived from array shapes)
    # ------------------------------------------------------------------ #
    Nx: int
    Ny: int
    Nz: int

    # ------------------------------------------------------------------ #
    # Node coordinates (metres) — strictly increasing, length N+1 per axis.
    # Geometric source of truth; the future snapping/mesh generator writes here.
    # ------------------------------------------------------------------ #
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    # ------------------------------------------------------------------ #
    # Per-axis spacing arrays (metres), length N per axis. Broadcast-ready for
    # the hot loops (NumPy oracle indexes with [None,:,None]; the Numba kernel
    # indexes by loop counter).
    #   dxp/dyp/dzp : primary widths  — denominators for update_E
    #   dxd/dyd/dzd : dual widths      — denominators for update_H
    # ------------------------------------------------------------------ #
    dxp: np.ndarray
    dyp: np.ndarray
    dzp: np.ndarray
    dxd: np.ndarray
    dyd: np.ndarray
    dzd: np.ndarray

    # ------------------------------------------------------------------ #
    # Cell-center (half-node) coordinates (metres), length N per axis.
    # ------------------------------------------------------------------ #
    xc: np.ndarray
    yc: np.ndarray
    zc: np.ndarray

    # ------------------------------------------------------------------ #
    # PEC body mask — shape (Nx, Ny, Nz), dtype bool
    # True = cell is a perfect electric conductor.
    # E components inside PEC cells are zeroed after every E update.
    # Set by pec.py geometry helpers or directly by the future CAD importer.
    # ------------------------------------------------------------------ #
    pec_mask: np.ndarray = field(default=None)

    # ------------------------------------------------------------------ #
    # Simulation time
    # ------------------------------------------------------------------ #
    time_step: int = 0

    # ------------------------------------------------------------------ #
    # Coordinate conversion — metres -> cell index
    # ------------------------------------------------------------------ #
    # Every public API speaks metres; these helpers are the single place
    # physical positions are snapped to the Yee grid. Sources, monitors, and
    # viz all route through them so the rounding convention is consistent.
    #
    # On a uniform grid the searchsorted nearest-node lookup below reproduces
    # the old ``round(pos/ds)`` result; on a non-uniform grid it snaps to the
    # nearest actual node coordinate.
    # ------------------------------------------------------------------ #
    def _coords(self, axis: str) -> np.ndarray:
        coords = {'x': self.x, 'y': self.y, 'z': self.z}.get(axis)
        if coords is None:
            raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
        return coords

    def axis_index(self, axis: str, pos: float) -> int:
        """Nearest node index along ``axis`` ('x'/'y'/'z') for ``pos`` (metres)."""
        return _nearest_index(self._coords(axis), pos)

    def position_to_index(self, x: float, y: float, z: float) -> tuple:
        """Nearest node index ``(i, j, k)`` for a physical ``(x, y, z)`` in metres."""
        return (_nearest_index(self.x, x),
                _nearest_index(self.y, y),
                _nearest_index(self.z, z))

    def cell_volume(self) -> np.ndarray:
        """Per-cell volume (Nx, Ny, Nz) — outer product of the dual widths.

        Used by the energy monitor and port code (Session 5) so integrals weight
        each cell by its true local volume. Reduces to the constant
        ``dx*dy*dz`` on a uniform grid.
        """
        return (self.dxd[:, None, None]
                * self.dyd[None, :, None]
                * self.dzd[None, None, :])


def _nearest_index(coords: np.ndarray, pos: float) -> int:
    """Index of the node in ``coords`` (increasing) closest to ``pos``.

    On a uniform axis this equals ``round(pos/ds)`` for in-domain positions.
    Ties break toward the lower index; out-of-range positions clamp to the ends.
    """
    n = len(coords)
    idx = int(np.searchsorted(coords, pos))
    if idx <= 0:
        return 0
    if idx >= n:
        return n - 1
    # coords[idx-1] <= pos <= coords[idx]; pick the closer node.
    if (coords[idx] - pos) < (pos - coords[idx - 1]):
        return idx
    return idx - 1


# ---------------------------------------------------------------------------- #
# Axis builders
# ---------------------------------------------------------------------------- #
def _axis_uniform(N: int, ds: float):
    """(coords, primary, dual, centers) for a uniform axis of ``N`` cells.

    Spacing arrays are exact constants (not node-coordinate differences) so a
    uniform grid reproduces the pre-rehaul scalar results bit-for-bit.
    """
    coords = np.arange(N + 1, dtype=np.float64) * ds
    primary = np.full(N, ds, dtype=np.float64)
    dual = np.full(N, ds, dtype=np.float64)
    centers = (np.arange(N, dtype=np.float64) + 0.5) * ds
    return coords, primary, dual, centers


def _axis_nonuniform(coords):
    """(coords, primary, dual, centers) for arbitrary increasing node coords."""
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    if coords.ndim != 1 or coords.size < 2:
        raise ValueError("coordinate array must be 1-D with at least 2 nodes")
    primary = np.diff(coords)
    if not np.all(primary > 0):
        raise ValueError("node coordinates must be strictly increasing")
    dual = np.empty_like(primary)
    dual[:-1] = 0.5 * (primary[:-1] + primary[1:])
    dual[-1] = primary[-1]                    # replicate last cell width (padding)
    centers = 0.5 * (coords[:-1] + coords[1:])
    return coords, primary, dual, centers


def _cfl_dt(dxp, dyp, dzp) -> float:
    """3D CFL timestep from the *minimum* spacing on each axis (Taflove 4.78)."""
    return _CFL / (C0 * np.sqrt(1.0 / dxp.min()**2
                                + 1.0 / dyp.min()**2
                                + 1.0 / dzp.min()**2))


def _assemble_grid(ax, ay, az, dtype) -> FDTDGrid:
    """Build an FDTDGrid from three ``_axis_*`` tuples (coords, dp, dd, centers)."""
    (x, dxp, dxd, xc) = ax
    (y, dyp, dyd, yc) = ay
    (z, dzp, dzd, zc) = az
    Nx, Ny, Nz = dxp.size, dyp.size, dzp.size

    dt = _cfl_dt(dxp, dyp, dzp)

    shape = (Nx, Ny, Nz)
    zeros = lambda: np.zeros(shape, dtype=dtype)
    ones = lambda: np.ones(shape, dtype=dtype)

    return FDTDGrid(
        # Fields — initialised to zero
        Ex=zeros(), Ey=zeros(), Ez=zeros(),
        Hx=zeros(), Hy=zeros(), Hz=zeros(),

        # Materials — initialised to vacuum (eps_r = mu_r = 1)
        eps_x=ones(), eps_y=ones(), eps_z=ones(),
        mu_x=ones(),  mu_y=ones(),  mu_z=ones(),

        # Scalar spacing — min width per axis (== the constant on a uniform grid)
        dx=float(dxp.min()), dy=float(dyp.min()), dz=float(dzp.min()), dt=dt,

        # Domain size
        Nx=Nx, Ny=Ny, Nz=Nz,

        # Node coordinates (geometric source of truth)
        x=x, y=y, z=z,

        # Per-axis spacing arrays
        dxp=dxp, dyp=dyp, dzp=dzp,
        dxd=dxd, dyd=dyd, dzd=dzd,

        # Cell-center coordinates
        xc=xc, yc=yc, zc=zc,

        # PEC mask starts as None (no interior conductors)
        pec_mask=None,

        time_step=0,
    )


def create_grid(Nx: int, Ny: int, Nz: int,
                dx: float, dy: float = None, dz: float = None,
                dtype=np.float64) -> FDTDGrid:
    """
    Allocate a **uniform** grid with all fields and materials initialised to
    zero/vacuum.

    This is the backward-compatible entry point: pass scalar spacings and the
    grid builds constant coordinate + spacing arrays that reproduce the
    pre-rehaul results bit-for-bit. For a non-uniform (rectilinear) grid use
    :func:`create_grid_rectilinear`.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of cells along each axis. Use Nz=1 for 2D-in-3D operation.
    dx : float
        Cell size in x (metres). If dy/dz are omitted they default to dx
        (uniform cubic cells).
    dtype : numpy dtype, optional
        Storage dtype for all field and material arrays (default
        ``np.float64``). Pass ``np.float32`` for the GPU (``backend='cuda'``)
        path: it halves memory traffic and, on consumer NVIDIA cards, avoids the
        heavily throttled float64 arithmetic. The NumPy/Numba CPU backends run
        correctly in either precision but are validated in float64.
        :func:`wavesim.pml.init_cpml` follows this dtype automatically.

    Returns
    -------
    FDTDGrid
        Fully initialised grid. All fields are zero; all materials are vacuum
        (eps_r = mu_r = 1); dt is set from the 3D CFL condition.

    Notes
    -----
    dt is computed from the conservative 3D CFL condition:
        dt = CFL / (c * sqrt(1/dx² + 1/dy² + 1/dz²))
    with CFL = 0.99.
    """
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx

    return _assemble_grid(
        _axis_uniform(Nx, dx),
        _axis_uniform(Ny, dy),
        _axis_uniform(Nz, dz),
        dtype,
    )


def create_grid_rectilinear(x, y, z, dtype=np.float64) -> FDTDGrid:
    """
    Allocate a **non-uniform (rectilinear)** grid from per-axis node coordinates.

    Parameters
    ----------
    x, y, z : 1-D array-like
        Strictly increasing node coordinates in metres. An axis with ``N+1``
        nodes has ``N`` cells; for a 2D-in-3D slice pass e.g. ``z=[0, dz]``
        (two nodes → one cell).
    dtype : numpy dtype, optional
        Storage dtype for the field/material arrays (see :func:`create_grid`).
        Coordinate/spacing arrays are always float64 (geometry).

    Returns
    -------
    FDTDGrid
        Fully initialised grid. Primary/dual spacing arrays are derived from the
        coordinates (2nd-order for smooth grading, 1st-order at abrupt jumps).
        A recommended max grading ratio between adjacent cells is ~1.5–2×.

    Notes
    -----
    dt uses the CFL condition reduced over the *minimum* spacing per axis, so a
    locally refined region sets the stable timestep for the whole domain.
    """
    return _assemble_grid(
        _axis_nonuniform(x),
        _axis_nonuniform(y),
        _axis_nonuniform(z),
        dtype,
    )
