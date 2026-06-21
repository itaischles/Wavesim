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
    """Capture a 2D XY slice of a field component at regular intervals."""
    component: str      # 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'
    at_z: float         # z position (metres) of the XY slice (use 0 for Nz=1)
    every_N_steps: int  # record every N timesteps
    snapshots:   list = field(default_factory=list)
    snap_times:  list = field(default_factory=list)


def record_snapshot(monitor: SnapshotMonitor, grid: FDTDGrid) -> SnapshotMonitor:
    """Append a 2D slice to the snapshot list if this is a recording timestep."""
    if grid.time_step % monitor.every_N_steps == 0:
        k = grid.axis_index('z', monitor.at_z)
        arr = getattr(grid, monitor.component)
        monitor.snapshots.append(arr[:, :, k].copy())
        monitor.snap_times.append(grid.time_step * _get_dt(grid))
    return monitor


# ======================================================================= #
# EnergyMonitor — total EM energy in the domain
# ======================================================================= #

@dataclass
class EnergyMonitor:
    """
    Track total electromagnetic energy in the domain.
    U = 0.5 * sum(eps*|E|² + mu*|H|²) * dx*dy*dz
    Must not grow over time in a stable simulation.
    """
    times:  list = field(default_factory=list)
    values: list = field(default_factory=list)


def record_energy(monitor: EnergyMonitor, grid: FDTDGrid) -> EnergyMonitor:
    """Compute total field energy and append to time series."""
    dV = grid.dx * grid.dy * grid.dz

    E_energy = 0.5 * EPS0 * dV * (
        np.sum(grid.eps_x * grid.Ex**2) +
        np.sum(grid.eps_y * grid.Ey**2) +
        np.sum(grid.eps_z * grid.Ez**2)
    )
    H_energy = 0.5 * MU0 * dV * (
        np.sum(grid.mu_x * grid.Hx**2) +
        np.sum(grid.mu_y * grid.Hy**2) +
        np.sum(grid.mu_z * grid.Hz**2)
    )

    monitor.times.append(grid.time_step * _get_dt(grid))
    monitor.values.append(float(E_energy + H_energy))
    return monitor


# ======================================================================= #
# Internal helper
# ======================================================================= #

def _get_dt(grid: FDTDGrid) -> float:
    """Return grid.dt — centralised so time axis is always in seconds."""
    return grid.dt
