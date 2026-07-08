"""
monitors.py — Diagnostic monitors.

All monitors follow the same pattern:
    - A dataclass holding location/config and accumulated data lists
    - A record_*() function that appends current values to the monitor

Usage:
    mon = FieldProbe(component='Ez', x=50e-3, y=50e-3, z=0.0)
    # Or a field magnitude:
    mon = FieldProbe(component='|E|', x=50e-3, y=50e-3, z=0.0)
    # In time loop:
    mon = record_field(mon, grid)

All monitor locations are given in metres and snapped to the nearest cell
against the grid inside the record_* functions.

Line-integral monitors (voltage and current) take a *path*: a sequence of
(x, y, z) vertices in metres defining a polyline. The polyline is subdivided
into sub-cell steps and each step samples the field component nearest to its
midpoint on the staggered Yee grid, so arbitrary (staircased) curves work.
"""

from dataclasses import dataclass, field
import numpy as np
from wavesim.grid import FDTDGrid
from wavesim.constants import EPS0, MU0


# ======================================================================= #
# FieldProbe — single component at a fixed cell
# ======================================================================= #

@dataclass
class FieldProbe:
    """
    Record a single field value at a fixed location given in metres.

    ``component`` selects what is recorded:
        - A single component: 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'
        - A field magnitude:  '|E|' or '|H|', where
              |E| = sqrt(Ex² + Ey² + Ez²)
              |H| = sqrt(Hx² + Hy² + Hz²)
    """
    component: str      # 'Ex'..'Hz', or '|E|' / '|H|'
    x: float
    y: float
    z: float
    times:  list = field(default_factory=list)
    values: list = field(default_factory=list)


def record_field(monitor: FieldProbe, grid: FDTDGrid) -> FieldProbe:
    """Append current field value (component or magnitude) to the monitor."""
    i, j, k = grid.position_to_index(monitor.x, monitor.y, monitor.z)
    comp = monitor.component
    if comp in ('|E|', '|H|'):
        f = comp[1]  # 'E' or 'H'
        value = np.sqrt(
            getattr(grid, f + 'x')[i, j, k]**2 +
            getattr(grid, f + 'y')[i, j, k]**2 +
            getattr(grid, f + 'z')[i, j, k]**2
        )
    else:
        value = getattr(grid, comp)[i, j, k]
    monitor.times.append(grid.time_step * _get_dt(grid))
    monitor.values.append(float(value))
    return monitor


# ======================================================================= #
# SnapshotMonitor — 2D slice of a field component at regular intervals
# ======================================================================= #

@dataclass
class SnapshotMonitor:
    """
    Capture a 2D slice of a field component at regular intervals.

    The slice is the plane perpendicular to ``normal`` ('x', 'y' or 'z') at
    position ``at_z`` (metres) along that axis:
        - normal='z' -> an XY slice, ``grid[:, :, k]`` (the default)
        - normal='y' -> an XZ slice, ``grid[:, j, :]``
        - normal='x' -> a YZ slice, ``grid[i, :, :]``
    ``at_z`` keeps its name for backward compatibility; for a non-z normal it is
    simply the coordinate along ``normal``.

    ``component`` selects what is recorded:
        - A single component: 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'
        - A field magnitude:  '|E|' or '|H|', where
              |E| = sqrt(Ex² + Ey² + Ez²)
              |H| = sqrt(Hx² + Hy² + Hz²)
    """
    component: str      # 'Ex'..'Hz', or '|E|' / '|H|'
    at_z: float         # slice position (metres) along `normal` (use 0 for a 1-cell axis)
    every_N_steps: int  # record every N timesteps
    normal: str = 'z'   # axis the slice plane is perpendicular to: 'x'/'y'/'z'
    snapshots:   list = field(default_factory=list)
    snap_times:  list = field(default_factory=list)


def record_snapshot(monitor: SnapshotMonitor, grid: FDTDGrid) -> SnapshotMonitor:
    """Append a 2D slice (component or magnitude) if this is a recording timestep."""
    if grid.time_step % monitor.every_N_steps == 0:
        normal = getattr(monitor, 'normal', 'z')
        idx = grid.axis_index(normal, monitor.at_z)

        def _slice(arr):
            """The 2D plane of *arr* perpendicular to ``normal`` at ``idx``."""
            if normal == 'z':
                return arr[:, :, idx]
            if normal == 'y':
                return arr[:, idx, :]
            return arr[idx, :, :]  # normal == 'x'

        comp = monitor.component
        if comp in ('|E|', '|H|'):
            f = comp[1]  # 'E' or 'H'
            arr = np.sqrt(
                _slice(getattr(grid, f + 'x'))**2 +
                _slice(getattr(grid, f + 'y'))**2 +
                _slice(getattr(grid, f + 'z'))**2
            )
            monitor.snapshots.append(arr)
        else:
            monitor.snapshots.append(_slice(getattr(grid, comp)).copy())
        monitor.snap_times.append(grid.time_step * _get_dt(grid))
    return monitor


# ======================================================================= #
# EnergyMonitor — total EM energy in the domain
# ======================================================================= #

@dataclass
class EnergyMonitor:
    """
    Track total electromagnetic energy in the domain.
    U = 0.5 * sum( (eps*|E|² + mu*|H|²) * dV_cell )
    Must not grow over time in a stable simulation.

    Each cell is weighted by its own local volume (``grid.cell_volume()``), so
    the sum is correct on a non-uniform (rectilinear) grid. On a uniform grid
    ``dV_cell`` is the constant ``dx*dy*dz`` and this reduces to the old result.
    """
    times:  list = field(default_factory=list)
    values: list = field(default_factory=list)


def record_energy(monitor: EnergyMonitor, grid: FDTDGrid) -> EnergyMonitor:
    """Compute total field energy and append to time series."""
    dV = grid.cell_volume()                       # per-cell (Nx, Ny, Nz)

    E_energy = 0.5 * EPS0 * (
        np.sum(dV * grid.eps_x * grid.Ex**2) +
        np.sum(dV * grid.eps_y * grid.Ey**2) +
        np.sum(dV * grid.eps_z * grid.Ez**2)
    )
    H_energy = 0.5 * MU0 * (
        np.sum(dV * grid.mu_x * grid.Hx**2) +
        np.sum(dV * grid.mu_y * grid.Hy**2) +
        np.sum(dV * grid.mu_z * grid.Hz**2)
    )

    monitor.times.append(grid.time_step * _get_dt(grid))
    monitor.values.append(float(E_energy + H_energy))
    return monitor


# ======================================================================= #
# VoltageMonitor / CurrentMonitor — line integrals of E / H along a curve
# ======================================================================= #
#
# Both monitors integrate a field along a user-given polyline (vertices in
# metres). The integral is evaluated with the midpoint rule on sub-segments no
# longer than half the smallest cell size; each midpoint samples the nearest
# staggered Yee location of the component parallel to the step. Because the
# curve is fixed for a run, this sampling is compiled once into per-component
# index/weight arrays (a quadrature) and each timestep costs only a few small
# fancy-indexed dot products.

# Yee stagger offsets in cell units, matching update.py:
#   Ex[i,j,k] at (i, j+1/2, k+1/2) etc.
_YEE_OFFSETS = {
    'Ex': (0.0, 0.5, 0.5), 'Ey': (0.5, 0.0, 0.5), 'Ez': (0.5, 0.5, 0.0),
    'Hx': (0.5, 0.0, 0.0), 'Hy': (0.0, 0.5, 0.0), 'Hz': (0.0, 0.0, 0.5),
}


def _yee_index(grid: FDTDGrid, axis: str, off: float, coords) -> np.ndarray:
    """Nearest staggered-Yee index along ``axis`` for physical ``coords`` (metres).

    ``off`` is the component's half-cell offset on this axis (from
    :data:`_YEE_OFFSETS`): ``0.0`` places it on the integer node (coordinate
    ``grid.x[i]``), ``0.5`` on the cell centre (``grid.xc[i]``). We snap to the
    nearest actual Yee location, so a non-uniform (rectilinear) grid works — the
    old ``round(coord/ds - off)`` assumed uniform spacing.

    On a uniform grid this reproduces that rounding for every non-tie position
    (both reduce to round-half-up); exact half-way ties may pick the neighbour
    the old even-rounding would not, which is physically inconsequential.
    """
    N = {'x': grid.Nx, 'y': grid.Ny, 'z': grid.Nz}[axis]
    coords = np.asarray(coords, dtype=np.float64)
    if N == 1:
        return np.zeros(coords.shape, dtype=np.intp)
    if off == 0.0:
        locs = grid._coords(axis)[:N]                    # integer-node coords x[0..N-1]
    else:
        locs = {'x': grid.xc, 'y': grid.yc, 'z': grid.zc}[axis]   # cell centres
    bounds = 0.5 * (locs[:-1] + locs[1:])                # midpoints between Yee locs
    idx = np.searchsorted(bounds, coords, side='right')
    return np.clip(idx, 0, N - 1).astype(np.intp)


@dataclass
class VoltageMonitor:
    """
    Record the voltage V(t) = ∫ E·dl along an *open* curve.

    With E = -∇φ this is φ(start) - φ(end): the potential of the curve's
    first vertex relative to its last. Integrate from the "+" conductor to
    the reference conductor to read a positive voltage.

    ``path`` is a sequence of (x, y, z) vertices in metres; straight segments
    connect consecutive vertices. The path may pass through PEC cells (E is
    zero there), so spanning a gap conductor-surface-to-conductor-surface is
    easiest done by letting the endpoints sit slightly inside the metal.
    """
    path: tuple         # sequence of (x, y, z) vertices in metres
    times:  list = field(default_factory=list)
    values: list = field(default_factory=list)
    _quad: dict = field(default=None, repr=False)   # built lazily vs the grid


@dataclass
class CurrentMonitor:
    """
    Record the current I(t) = ∮ H·dl around a *closed* curve (Ampère's law).

    Positive current flows in the direction given by the right-hand rule
    applied to the traversal order of ``path`` (curl fingers along the path,
    thumb points along positive I).

    ``path`` is a sequence of (x, y, z) vertices in metres; if the last vertex
    does not coincide with the first, the loop is closed automatically. Use
    :func:`circular_path` to build a circular loop around a conductor.
    """
    path: tuple         # sequence of (x, y, z) vertices in metres
    times:  list = field(default_factory=list)
    values: list = field(default_factory=list)
    _quad: dict = field(default=None, repr=False)   # built lazily vs the grid


def circular_path(cx: float, cy: float, cz: float, radius: float,
                  normal: str = 'z', n_points: int = 64) -> np.ndarray:
    """
    Vertices (metres) of a closed circle — convenience for CurrentMonitor.

    The circle of ``radius`` lies in the plane perpendicular to ``normal``
    ('x'/'y'/'z') centred at (cx, cy, cz), traversed counterclockwise when
    viewed from the +normal side, so a CurrentMonitor on it reads current in
    the +normal direction as positive.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    ca, sa = radius * np.cos(theta), radius * np.sin(theta)
    zeros = np.zeros_like(theta)
    # Columns ordered so (first-axis, second-axis) is CCW about +normal.
    if normal == 'z':
        offsets = np.column_stack([ca, sa, zeros])
    elif normal == 'y':
        offsets = np.column_stack([sa, zeros, ca])   # CCW about +y: (z, x) plane
    elif normal == 'x':
        offsets = np.column_stack([zeros, ca, sa])
    else:
        raise ValueError(f"normal must be 'x', 'y' or 'z', got {normal!r}")
    pts = np.array([cx, cy, cz]) + offsets
    return np.vstack([pts, pts[0]])                  # explicitly closed


def _build_path_quadrature(path, grid: FDTDGrid, field_name: str,
                           close: bool) -> dict:
    """
    Compile a polyline into per-component Yee-grid quadrature weights.

    Returns {component: (ii, jj, kk, w)} such that
        ∫ F·dl  ≈  Σ_comp  Σ_m  F_comp[ii_m, jj_m, kk_m] * w_m
    with F the E or H field (``field_name`` 'E' or 'H'). Each straight segment
    is split into sub-steps no longer than half the smallest cell size; every
    sub-step contributes its vector length, split per axis, at the staggered
    Yee location of that axis' component nearest the sub-step midpoint.

    Index snapping goes through :func:`_yee_index` (a coordinate lookup), so the
    quadrature is correct on a non-uniform (rectilinear) grid; ``w`` is already a
    true physical length and needs no change. Ports (``sources.py``) reuse this
    so a LineSource and a monitor on the same path agree on ∫E·dl.
    """
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        raise ValueError(
            f"path must be a sequence of >= 2 (x, y, z) vertices, "
            f"got array of shape {pts.shape}")
    if close and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    shape = (grid.Nx, grid.Ny, grid.Nz)
    # grid.dx/dy/dz hold the MINIMUM width per axis on a non-uniform grid, so
    # this sub-step is <= half the smallest cell anywhere (fine everywhere).
    max_step = 0.5 * min(grid.dx, grid.dy, grid.dz)

    # Accumulate flat-index -> weight per component (duplicates summed).
    flat_idx = {c: [] for c in 'xyz'}
    weights = {c: [] for c in 'xyz'}

    for p0, p1 in zip(pts[:-1], pts[1:]):
        seg = p1 - p0
        length = np.linalg.norm(seg)
        if length == 0.0:
            continue
        # Guard the ceil against floating-point dust: when a segment length is an
        # exact multiple of max_step the count must not tip to one extra sub-step.
        # On a rectilinear grid ``grid.dx`` (hence max_step) carries the rounding
        # of ``np.diff``, so an unguarded ceil would add a spurious sub-step and
        # alias the per-edge weights (a line lying on cell edges bins unevenly).
        n_sub = max(1, int(np.ceil(length / max_step * (1.0 - 1e-9))))
        dl = seg / n_sub                              # vector step (m)
        t_mid = (np.arange(n_sub) + 0.5) / n_sub
        mids = p0 + t_mid[:, None] * seg              # (n_sub, 3) midpoints

        for a, axis in enumerate('xyz'):
            if dl[a] == 0.0:
                continue
            comp = field_name + axis
            off = _YEE_OFFSETS[comp]
            idx = [_yee_index(grid, b_axis, off[b], mids[:, b])
                   for b, b_axis in enumerate('xyz')]
            flat_idx[axis].append(np.ravel_multi_index(idx, shape))
            weights[axis].append(np.full(n_sub, dl[a]))

    quad = {}
    for axis in 'xyz':
        if not flat_idx[axis]:
            continue
        flat = np.concatenate(flat_idx[axis])
        w = np.concatenate(weights[axis])
        uniq, inv = np.unique(flat, return_inverse=True)
        w_sum = np.zeros(uniq.size)
        np.add.at(w_sum, inv, w)
        ii, jj, kk = np.unravel_index(uniq, shape)
        quad[field_name + axis] = (ii, jj, kk, w_sum)
    return quad


def _integrate_path(monitor, grid: FDTDGrid, field_name: str, close: bool):
    """Evaluate the monitor's line integral, building its quadrature once."""
    if monitor._quad is None:
        monitor._quad = _build_path_quadrature(
            monitor.path, grid, field_name, close)
    total = 0.0
    for comp, (ii, jj, kk, w) in monitor._quad.items():
        total += np.dot(getattr(grid, comp)[ii, jj, kk], w)
    monitor.times.append(grid.time_step * _get_dt(grid))
    monitor.values.append(float(total))
    return monitor


def record_voltage(monitor: VoltageMonitor, grid: FDTDGrid) -> VoltageMonitor:
    """Append V = ∫ E·dl (start→end of the open path) to the monitor."""
    return _integrate_path(monitor, grid, 'E', close=False)


def record_current(monitor: CurrentMonitor, grid: FDTDGrid) -> CurrentMonitor:
    """Append I = ∮ H·dl around the (auto-)closed path to the monitor."""
    return _integrate_path(monitor, grid, 'H', close=True)


# ======================================================================= #
# Internal helper
# ======================================================================= #

def _get_dt(grid: FDTDGrid) -> float:
    """Return grid.dt — centralised so time axis is always in seconds."""
    return grid.dt
