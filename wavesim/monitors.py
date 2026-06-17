"""
monitors.py — Diagnostic monitors.

All monitors follow the same pattern:
    - A dataclass holding location/config and accumulated data lists
    - A record_*() function that appends current values to the monitor

Usage:
    mon = FieldMonitor(component='Ez', i=100, j=100, k=0)
    # In time loop:
    mon = record_field(mon, grid)
"""

from dataclasses import dataclass, field
import numpy as np
from wavesim.grid import FDTDGrid
from wavesim.constants import EPS0, MU0


# ======================================================================= #
# FieldMonitor — single component at a fixed cell
# ======================================================================= #

@dataclass
class FieldMonitor:
    """Record a single field component at a fixed cell location."""
    component: str      # 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'
    i: int
    j: int
    k: int
    times:  list = field(default_factory=list)
    values: list = field(default_factory=list)


def record_field(monitor: FieldMonitor, grid: FDTDGrid) -> FieldMonitor:
    """Append current field value and timestep time to the monitor."""
    arr = getattr(grid, monitor.component)
    monitor.times.append(grid.time_step * _get_dt(grid))
    monitor.values.append(float(arr[monitor.i, monitor.j, monitor.k]))
    return monitor


# ======================================================================= #
# MagnitudeMonitor — |E| or |H| at a fixed cell
# ======================================================================= #

@dataclass
class MagnitudeMonitor:
    """
    Record |E| or |H| magnitude at a fixed cell location.
    |E| = sqrt(Ex² + Ey² + Ez²)
    |H| = sqrt(Hx² + Hy² + Hz²)
    """
    field: str          # 'E' or 'H'
    i: int
    j: int
    k: int
    times:  list = field(default_factory=list)
    values: list = field(default_factory=list)


def record_magnitude(monitor: MagnitudeMonitor, grid: FDTDGrid) -> MagnitudeMonitor:
    """Compute and append |E| or |H| at the monitor location."""
    i, j, k = monitor.i, monitor.j, monitor.k
    if monitor.field == 'E':
        mag = np.sqrt(grid.Ex[i,j,k]**2 + grid.Ey[i,j,k]**2 + grid.Ez[i,j,k]**2)
    elif monitor.field == 'H':
        mag = np.sqrt(grid.Hx[i,j,k]**2 + grid.Hy[i,j,k]**2 + grid.Hz[i,j,k]**2)
    else:
        raise ValueError(f"MagnitudeMonitor.field must be 'E' or 'H', got '{monitor.field}'")
    monitor.times.append(grid.time_step * _get_dt(grid))
    monitor.values.append(float(mag))
    return monitor


# ======================================================================= #
# SnapshotMonitor — 2D slice of a field component at regular intervals
# ======================================================================= #

@dataclass
class SnapshotMonitor:
    """Capture a 2D slice of a field component at regular intervals."""
    component: str      # 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'
    at_z_index: int     # which z-index to capture (use 0 for Nz=1)
    every_N_steps: int  # record every N timesteps
    snapshots:   list = field(default_factory=list)
    snap_times:  list = field(default_factory=list)


def record_snapshot(monitor: SnapshotMonitor, grid: FDTDGrid) -> SnapshotMonitor:
    """Append a 2D slice to the snapshot list if this is a recording timestep."""
    if grid.time_step % monitor.every_N_steps == 0:
        arr = getattr(grid, monitor.component)
        monitor.snapshots.append(arr[:, :, monitor.at_z_index].copy())
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
