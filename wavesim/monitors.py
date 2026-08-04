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

    Output contract — where the recorded data lives
    -----------------------------------------------
    Snapshots are **collocated to cell centres**, not left on the staggered Yee
    locations they are stored at. Every component of every snapshot therefore
    shares one coordinate grid: frame element ``[a, b]`` is the field at the
    centre of cell ``(a, b)`` of the slice plane, i.e. at ``grid.xc[a]``,
    ``grid.yc[b]`` (and ``grid.zc[idx]`` on the normal axis) — the cell-centre
    coordinate arrays, valid on a non-uniform grid.

    Collocation consumes one neighbour per averaged axis, so each frame is
    **one cell shorter than the grid along each in-plane axis**: a z-normal
    slice of an ``(Nx, Ny, Nz)`` grid yields frames of shape ``(Nx-1, Ny-1)``,
    spanning cells ``0 .. N-2``. All components are cropped to this same
    interior even when they would not individually lose that axis, so that a
    single coordinate grid serves the whole snapshot. An axis of length 1 (a 2D
    run) is degenerate — it has no neighbour and the field is invariant along it
    — and is left at length 1.

    ``'|E|'`` / ``'|H|'`` collocate each component first and root-sum-square
    afterwards, so they are the true magnitude at the cell centre rather than a
    mix of three different sample points.

    Output contract — when the recorded data lives
    ----------------------------------------------
    H trails E by half a timestep in the leapfrog (``update_H`` then
    ``update_E`` then record), so **H snapshots are averaged onto the E
    timebase**: the frame stamped ``t`` is the mean of the collocated H at
    ``t - dt/2`` and ``t + dt/2``. E and H frames sharing a ``snap_times`` entry
    therefore represent the same instant, which is what a Poynting vector
    computed downstream requires. This costs one step of latency: a stashed H
    frame that the run ends before completing is dropped, so ``snapshots`` and
    ``snap_times`` always stay the same length. This assumes
    :func:`record_snapshot` is called on *every* timestep (as ``Simulation``
    does), not only on recording steps.

    ``snap_times[n]`` is the physical time of the E field in the frame,
    ``(time_step + 1) * dt`` — monitors run after ``update_E`` has already
    advanced E to the end of the step.
    """
    component: str      # 'Ex'..'Hz', or '|E|' / '|H|'
    at_z: float         # slice position (metres) along `normal` (use 0 for a 1-cell axis)
    every_N_steps: int  # record every N timesteps
    normal: str = 'z'   # axis the slice plane is perpendicular to: 'x'/'y'/'z'
    snapshots:   list = field(default_factory=list)
    snap_times:  list = field(default_factory=list)
    # Half-step carry for H components: (snap_time, collocated frame) awaiting
    # the next step's H so the two can be averaged onto the E timebase.
    _pending: tuple = field(default=None, repr=False)


# Yee half-cell offsets in cell units, per the update.py module docstring (the
# authority on the staggering):
#     Ex (i+½, j,   k  )      Hx (i,   j+½, k+½)
#     Ey (i,   j+½, k  )      Hy (i+½, j,   k+½)
#     Ez (i,   j,   k+½)      Hz (i+½, j+½, k  )
# The target is the cell centre (i+½, j+½, k+½). An axis carrying 0.5 is already
# centred; an axis carrying 0.0 sits on the cell's low node and must be averaged
# with index+1 to reach the centre. So E components need a 4-point average (their
# two transverse axes) and H components a 2-point one.
#
# Both weights are exactly 0.5 on a non-uniform rectilinear grid: grid.py defines
# xc[i] = (x[i] + x[i+1]) / 2, so the cell centre is the arithmetic midpoint of
# the two bounding nodes whatever the local spacing, and linear interpolation to
# a midpoint is ½/½ regardless of how far apart the samples are. No spacing
# arrays enter.
#
# NOTE: this deliberately does not reuse `_YEE_OFFSETS` further down, which
# encodes a *different* (transposed) staggering convention — see the comment
# there.
_CENTRE_OFFSETS = {
    'Ex': (0.5, 0.0, 0.0), 'Ey': (0.0, 0.5, 0.0), 'Ez': (0.0, 0.0, 0.5),
    'Hx': (0.0, 0.5, 0.5), 'Hy': (0.5, 0.0, 0.5), 'Hz': (0.5, 0.5, 0.0),
}

_AXES = ('x', 'y', 'z')


def _along(ndim: int, axis: int, s: slice) -> tuple:
    """An index tuple applying ``s`` to ``axis`` and taking all of the rest."""
    idx = [slice(None)] * ndim
    idx[axis] = s
    return tuple(idx)


def _collocate_slice(arr: np.ndarray, comp: str, normal: str,
                     idx: int, shape: tuple) -> np.ndarray:
    """
    Average *arr* (component ``comp``) onto cell centres and return the 2D plane
    perpendicular to ``normal`` at index ``idx``, cropped to the common interior.

    The normal axis is reduced first so at most two planes of the volume are ever
    touched. Averaging on that axis needs the plane at ``idx+1``; on the last
    cell of the axis there is none, so the neighbour index is clamped to ``idx``
    and the value keeps its native half-cell offset on that axis alone. That is a
    documented degradation on the extreme boundary plane, and it is also what
    makes a 1-cell axis work: a 2D run is invariant along its thin axis, so
    "average the plane with itself" is exact there.
    """
    offs = _CENTRE_OFFSETS[comp]
    n_ax = _AXES.index(normal)

    # --- normal axis: collapse to a single plane -----------------------------
    if offs[n_ax] == 0.0:
        hi = min(idx + 1, shape[n_ax] - 1)          # clamp, never wrap
        plane = 0.5 * (np.take(arr, idx, n_ax) + np.take(arr, hi, n_ax))
    else:
        plane = np.take(arr, idx, n_ax)             # np.take copies: never a
                                                    # view onto the live field
    # --- in-plane axes: average (offset 0) or crop (offset ½), both lose one --
    for a in (a for a in range(3) if a != n_ax):
        if shape[a] == 1:
            continue                                # degenerate axis, no neighbour
        pa = a if a < n_ax else a - 1                # its axis within `plane`
        lo = plane[_along(2, pa, slice(None, -1))]
        if offs[a] == 0.0:
            plane = 0.5 * (lo + plane[_along(2, pa, slice(1, None))])
        else:
            plane = lo
    return plane


def _collocated_frame(monitor: SnapshotMonitor, grid: FDTDGrid) -> np.ndarray:
    """The monitor's current 2D frame, collocated to cell centres."""
    normal = getattr(monitor, 'normal', 'z')
    idx = grid.axis_index(normal, monitor.at_z)
    shape = (grid.Nx, grid.Ny, grid.Nz)
    comp = monitor.component

    if comp in ('|E|', '|H|'):
        f = comp[1]                                  # 'E' or 'H'
        # Collocate each component *first*, then root-sum-square, so the result
        # is the magnitude at one real location.
        return np.sqrt(sum(
            _collocate_slice(getattr(grid, f + a), f + a, normal, idx, shape)**2
            for a in _AXES
        ))
    return _collocate_slice(getattr(grid, comp), comp, normal, idx, shape)


def record_snapshot(monitor: SnapshotMonitor, grid: FDTDGrid) -> SnapshotMonitor:
    """
    Append a cell-centre-collocated 2D slice if this is a recording timestep.

    Must be called on every timestep (not only recording ones): H snapshots are
    averaged across the half step, which needs the frame from the step after the
    recording step. See :class:`SnapshotMonitor` for the output contract.
    """
    recording = grid.time_step % monitor.every_N_steps == 0
    if not recording and monitor._pending is None:
        return monitor

    frame = _collocated_frame(monitor, grid)         # computed once, used twice
                                                     # when every_N_steps == 1

    # Complete a stashed H frame *before* stashing this one, so consecutive
    # recording steps each get their own carry.
    if monitor._pending is not None:
        t_stash, previous = monitor._pending
        monitor._pending = None
        monitor.snapshots.append(0.5 * (previous + frame))
        monitor.snap_times.append(t_stash)

    if recording:
        # E has just been advanced to the end of this step; time_step still holds
        # the step's start index, so the field in `frame` is at (n+1)*dt.
        t = (grid.time_step + 1) * _get_dt(grid)
        if monitor.component[0] == 'H' or monitor.component == '|H|':
            # H is at t - dt/2 here; carry it until the next step supplies
            # t + dt/2 and the mean lands on the E timebase.
            monitor._pending = (t, frame)
        else:
            monitor.snapshots.append(frame)
            monitor.snap_times.append(t)
    return monitor


# ======================================================================= #
# PoyntingMonitor — 2D slice of the instantaneous power flow S = E × H
# ======================================================================= #

@dataclass
class PoyntingMonitor:
    """
    Capture the instantaneous Poynting vector S = E × H on a 2D slice.

    This is the power-flow companion to :class:`SnapshotMonitor`: same slice
    geometry, same recording cadence, but it records the local energy flux
    density (W/m²) as a *vector* rather than a single field component, so a
    downstream plot can show which way — and how strongly — power is flowing
    through the plane.

    The slice is the plane perpendicular to ``normal`` ('x', 'y' or 'z') at
    position ``at_z`` (metres) along that axis — identical to
    :class:`SnapshotMonitor`:
        - normal='z' -> an XY slice (the default)
        - normal='y' -> an XZ slice
        - normal='x' -> a YZ slice
    ``at_z`` keeps its name for parity with SnapshotMonitor; for a non-z normal
    it is simply the coordinate along ``normal``.

    Output contract
    ---------------
    Each recorded frame is an array of shape ``(Na, Nb, 3)``: the two in-plane
    axes of the slice followed by the three Cartesian components of S in
    ``(x, y, z)`` order, i.e. ``frame[a, b] = (Sx, Sy, Sz)`` at the centre of
    cell ``(a, b)`` of the slice plane. For a z-normal slice ``(Sx, Sy)`` is the
    in-plane power flow and ``Sz`` the flow through the plane.

    S = E × H requires E and H at the **same location and the same instant**.
    Both are met by reusing SnapshotMonitor's machinery:

    - *Location.* Every one of the six field components is collocated to the cell
      centre first (see :func:`_collocate_slice`), then the cross product is
      formed there. Frames are therefore **one cell shorter than the grid along
      each in-plane axis** (a z-normal slice of an ``(Nx, Ny, Nz)`` grid yields
      ``(Nx-1, Ny-1, 3)``) and share the cell-centre coordinate arrays
      ``grid.xc`` / ``grid.yc`` / ``grid.zc``, exactly like SnapshotMonitor.

    - *Instant.* H trails E by half a timestep in the leapfrog, so the collocated
      H is averaged across the half step onto the E timebase: the frame stamped
      ``t`` crosses E at ``t`` with the mean of H at ``t - dt/2`` and
      ``t + dt/2``. This costs one step of latency — a stashed frame the run ends
      before completing is dropped, so ``snapshots`` and ``snap_times`` always
      stay the same length. As with SnapshotMonitor this assumes
      :func:`record_poynting` runs on *every* timestep (as ``Simulation`` does).

    ``snap_times[n]`` is the physical time of the frame, ``(time_step + 1) * dt``
    at the recording step.
    """
    at_z: float         # slice position (metres) along `normal` (use 0 for a 1-cell axis)
    every_N_steps: int  # record every N timesteps
    normal: str = 'z'   # axis the slice plane is perpendicular to: 'x'/'y'/'z'
    snapshots:   list = field(default_factory=list)   # each (Na, Nb, 3): Sx,Sy,Sz
    snap_times:  list = field(default_factory=list)
    # Half-step carry: (snap_time, E frame, H frame) — E is already on its
    # timebase; the stashed H (at t - dt/2) waits for the next step's H (t + dt/2)
    # so their mean lands H on the E timebase before the cross product is formed.
    _pending: tuple = field(default=None, repr=False)


def _collocated_vector(grid: FDTDGrid, f: str, normal: str,
                       idx: int, shape: tuple) -> np.ndarray:
    """Stack the three components of field ``f`` ('E' or 'H'), each collocated to
    cell centres on the slice plane, into one ``(Na, Nb, 3)`` frame in x,y,z order."""
    return np.stack(
        [_collocate_slice(getattr(grid, f + a), f + a, normal, idx, shape)
         for a in _AXES], axis=-1)


def record_poynting(monitor: PoyntingMonitor, grid: FDTDGrid) -> PoyntingMonitor:
    """
    Append a cell-centre-collocated 2D slice of S = E × H if this is a recording step.

    Must be called on every timestep (not only recording ones): H is averaged
    across the half step onto the E timebase, which needs the frame from the step
    after the recording step. See :class:`PoyntingMonitor` for the output contract.
    """
    recording = grid.time_step % monitor.every_N_steps == 0
    if not recording and monitor._pending is None:
        return monitor

    normal = getattr(monitor, 'normal', 'z')
    idx = grid.axis_index(normal, monitor.at_z)
    shape = (grid.Nx, grid.Ny, grid.Nz)

    # H at this step is at t - dt/2 relative to the E it accompanies; needed both
    # to complete a stashed frame (as its t + dt/2 partner) and to stash a new one.
    H_frame = _collocated_vector(grid, 'H', normal, idx, shape)

    # Complete a stashed frame *before* stashing this one, so consecutive
    # recording steps each get their own carry (matters when every_N_steps == 1).
    if monitor._pending is not None:
        t_stash, E_stash, H_stash = monitor._pending
        monitor._pending = None
        H_at_t = 0.5 * (H_stash + H_frame)           # onto the E timebase
        monitor.snapshots.append(np.cross(E_stash, H_at_t))
        monitor.snap_times.append(t_stash)

    if recording:
        E_frame = _collocated_vector(grid, 'E', normal, idx, shape)
        # E has just been advanced to the end of this step; time_step still holds
        # the step's start index, so the field is at (n+1)*dt.
        t = (grid.time_step + 1) * _get_dt(grid)
        monitor._pending = (t, E_frame, H_frame)
    return monitor


# ======================================================================= #
# EnergyMonitor — total EM energy in the domain
# ======================================================================= #

_ALL_FACES = ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')


@dataclass
class EnergyMonitor:
    """
    Track total electromagnetic energy in the domain.
    U = 0.5 * sum( (eps*|E|² + mu*|H|²) * dV_cell )
    Must not grow over time in a stable simulation.

    Each cell is weighted by its own local volume (``grid.cell_volume()``), so
    the sum is correct on a non-uniform (rectilinear) grid. On a uniform grid
    ``dV_cell`` is the constant ``dx*dy*dz`` and this reduces to the old result.

    ``region`` selects which cells are summed:
        - ``'full'``      (default) — the whole grid, including the PML shell.
        - ``'interior'``  — only the physical domain, dropping the outermost
          ``d_pml`` cells on every face that carries absorbing CPML. Energy
          leaking into the PML (where it is absorbed, not conserved) is then
          excluded, so a stable run's interior energy should decay to ~0.

    For ``region='interior'`` the PML geometry (``d_pml`` and ``faces``) is
    filled in automatically from the run's ``CPMLArrays`` when the monitor is
    registered on a :class:`~wavesim.simulation.Simulation`. Set them explicitly
    to override, or when driving the recorder outside a ``Simulation``. If a run
    has no CPML, ``'interior'`` trims nothing and equals ``'full'``.
    """
    region: str = 'full'            # 'full' (incl. PML) or 'interior' (excl. PML)
    d_pml:  int = None              # PML thickness in cells; None → auto-fill
    faces:  tuple = None            # faces carrying PML; None → auto-fill
    times:  list = field(default_factory=list)
    values: list = field(default_factory=list)

    def __post_init__(self):
        if self.region not in ('full', 'interior'):
            raise ValueError(
                f"EnergyMonitor.region must be 'full' or 'interior', "
                f"got {self.region!r}.")


def _interior_slices(grid: FDTDGrid, d_pml: int, faces: tuple) -> tuple:
    """Slices selecting the physical domain, trimming ``d_pml`` PML cells per face.

    A face's outer shell is dropped only when its face key is in ``faces`` *and*
    the axis is thick enough to actually host a PML slab (``N > 2*d_pml``); a
    thin axis (e.g. ``Nz=1`` in a 2D slice) has no PML and is never trimmed,
    matching :func:`wavesim.pml.init_cpml`, which makes such an axis' PML inert.
    """
    def axis(N, lo_face, hi_face):
        if d_pml <= 0 or N <= 2 * d_pml:
            return slice(0, N)
        lo = d_pml if lo_face in faces else 0
        hi = d_pml if hi_face in faces else 0
        return slice(lo, N - hi)

    return (axis(grid.Nx, 'x0', 'x1'),
            axis(grid.Ny, 'y0', 'y1'),
            axis(grid.Nz, 'z0', 'z1'))


def record_energy(monitor: EnergyMonitor, grid: FDTDGrid) -> EnergyMonitor:
    """Compute total field energy and append to time series."""
    dV = grid.cell_volume()                       # per-cell (Nx, Ny, Nz)

    if monitor.region == 'interior':
        sl = _interior_slices(grid, monitor.d_pml or 0,
                              monitor.faces if monitor.faces is not None else _ALL_FACES)
    else:
        sl = (slice(None), slice(None), slice(None))

    E_energy = 0.5 * EPS0 * (
        np.sum((dV * grid.eps_x * grid.Ex**2)[sl]) +
        np.sum((dV * grid.eps_y * grid.Ey**2)[sl]) +
        np.sum((dV * grid.eps_z * grid.Ez**2)[sl])
    )
    H_energy = 0.5 * MU0 * (
        np.sum((dV * grid.mu_x * grid.Hx**2)[sl]) +
        np.sum((dV * grid.mu_y * grid.Hy**2)[sl]) +
        np.sum((dV * grid.mu_z * grid.Hz**2)[sl])
    )

    monitor.times.append(grid.time_step * _get_dt(grid))
    monitor.values.append(float(E_energy + H_energy))
    return monitor


# ======================================================================= #
# DissipationMonitor — ohmic power lost to conductive material
# ======================================================================= #

@dataclass
class DissipationMonitor:
    """
    Track the instantaneous ohmic power dissipated in lossy dielectric.

    P = sum( sigma * |E|^2 * dV_cell ),  the volume integral of J·E.

    This is the term that makes :class:`EnergyMonitor` legitimately decay: for a
    closed PEC cavity, ``U(0) - U(t) = ∫P dt``. Without it a lossy run has no way
    to tell absorbed power from a solver leak.

    No factor of ½ — that belongs to time-*averaged* phasor amplitudes, and these
    are instantaneous fields. Anisotropic sigma is honoured per component.

    Resolve your fields
    -------------------
    This records ``sigma*|E^n|²``, the natural instantaneous J·E. The quantity
    the *scheme* actually dissipates is ``sigma*|Ebar|²`` with
    ``Ebar = (E^{n+1} + E^n)/2``, because that is the average the lossy update
    solves for (see :mod:`wavesim.loss`). The two agree wherever the field is
    resolved in time, and diverge where it is not: on a mode oscillating at the
    Nyquist rate ``E^{n+1} ≈ -E^n``, so ``Ebar ≈ 0`` while ``|E^n|²`` is at its
    largest, and this over-reports.

    Measured on a closed lossy cavity, the balance ``U(0) - U(t) = ∫P dt``
    closes to **0.10%** from a smooth initial field and is **22% high** from
    white noise. That is not a bug in either the monitor or the solver — it is
    what "unresolved" means. If the balance does not close, suspect the mesh or
    the timestep before the physics.

    Records ``0.0`` on a lossless grid rather than refusing, so a monitor can be
    attached to a sweep whose members may or may not carry loss.

    ``region`` / ``d_pml`` / ``faces`` select which cells are summed, exactly as
    on :class:`EnergyMonitor`, and are auto-filled from the run's CPML the same
    way. ``'interior'`` is usually what you want: the PML shell dissipates by
    design and would swamp the material's own loss.
    """
    region: str = 'full'            # 'full' (incl. PML) or 'interior' (excl. PML)
    d_pml:  int = None              # PML thickness in cells; None → auto-fill
    faces:  tuple = None            # faces carrying PML; None → auto-fill
    times:  list = field(default_factory=list)
    values: list = field(default_factory=list)

    def __post_init__(self):
        if self.region not in ('full', 'interior'):
            raise ValueError(
                f"DissipationMonitor.region must be 'full' or 'interior', "
                f"got {self.region!r}.")

    def energy(self) -> float:
        """Total energy dissipated so far (J), by trapezoid over the samples.

        The companion to ``U(0) - U(t)`` from :class:`EnergyMonitor`; the two
        agree to the scheme's order when both are recorded every step.
        """
        if len(self.times) < 2:
            return 0.0
        return float(np.trapezoid(self.values, self.times))


def record_dissipation(monitor: DissipationMonitor,
                       grid: FDTDGrid) -> DissipationMonitor:
    """Compute instantaneous ohmic power and append to time series."""
    if not grid.is_lossy:
        power = 0.0
    else:
        dV = grid.cell_volume()

        if monitor.region == 'interior':
            sl = _interior_slices(grid, monitor.d_pml or 0,
                                  monitor.faces if monitor.faces is not None
                                  else _ALL_FACES)
        else:
            sl = (slice(None), slice(None), slice(None))

        power = (
            np.sum((dV * grid.sigma_x * grid.Ex**2)[sl]) +
            np.sum((dV * grid.sigma_y * grid.Ey**2)[sl]) +
            np.sum((dV * grid.sigma_z * grid.Ez**2)[sl])
        )

    monitor.times.append(grid.time_step * _get_dt(grid))
    monitor.values.append(float(power))
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

# Yee stagger offsets in cell units, used by the path-integral monitors.
#   Ex[i,j,k] at (i, j+1/2, k+1/2) etc.
#
# WARNING: this table contradicts the staggering documented in update.py, which
# puts Ex at (i+1/2, j, k) — the transpose of the offsets below. It is left as-is
# here because the line-integral monitors and the ports in sources.py share it
# and were validated against each other with it; changing it shifts every port
# and probe by half a cell and must be done (and re-validated) as its own change.
# `_CENTRE_OFFSETS` above follows update.py and is the correct one.
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
