"""
wavesim — a small, readable FDTD electromagnetics engine.

This top-level package re-exports the public API so a script needs only::

    import wavesim as ws

    grid = ws.create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3)
    grid = ws.set_vacuum(grid)
    cpml = ws.init_cpml(grid, d_pml=10)

    sim = ws.Simulation(grid, cpml=cpml)
    sim.add_source(ws.PointSource('Ez', 50e-3, 50e-3, 0.0, ws.GaussianPulse.for_fmax(10e9)))
    snap = sim.add_monitor(ws.SnapshotMonitor('Ez', at_z=0.0, every_N_steps=20))
    sim.run(2000)

instead of a dozen ``from wavesim.<module> import ...`` lines. The individual
modules remain importable directly (``from wavesim.grid import create_grid``);
this namespace is purely additive.

The eagerly imported names below are all numpy-only, so ``import wavesim`` stays
cheap. The plotting helpers in :mod:`wavesim.viz` pull in matplotlib, so they are
exposed lazily (PEP 562 ``__getattr__``) — ``ws.plot_field_snapshot`` works on
first use without forcing a matplotlib import on every ``import wavesim`` (e.g.
for headless solver/benchmark runs).
"""

from importlib import import_module

# --- numpy-only core: safe to import eagerly --------------------------------- #
from wavesim.constants import C0, EPS0, MU0, ETA0
from wavesim.grid import FDTDGrid, create_grid
from wavesim.materials import (
    set_vacuum, set_material_arrays, set_box, set_cylinder, set_coax,
)
from wavesim.pml import (
    CPMLArrays, init_cpml, update_H_pml, update_E_pml, ALL_FACES,
)
from wavesim.pec import apply_pec_faces, apply_pec_mask
from wavesim.sources import (
    Waveform, GaussianPulse,
    Source, PointSource, ArraySource, PlaneSource, LineSource, VolumeSource,
)
from wavesim.monitors import (
    FieldProbe, SnapshotMonitor, EnergyMonitor,
    record_field, record_snapshot, record_energy,
)
from wavesim.update import update_H, update_E
from wavesim.simulation import Simulation

__version__ = "0.2.0"

# --- lazy (matplotlib-backed) viz helpers ------------------------------------ #
# Mapped to their module so ``ws.<name>`` resolves on first access without
# importing matplotlib at package-import time.
_LAZY = {
    name: "wavesim.viz" for name in (
        "plot_grid_xy", "plot_materials_xy",
        "plot_field_snapshot", "animate_snapshots", "plot_monitor_time_series",
        "plot_field_slices_3d", "animate_field_slices_3d", "plot_energy",
    )
}


def __getattr__(name):
    """Lazily resolve viz helpers (and the ``viz`` submodule) on first use."""
    if name == "viz":
        return import_module("wavesim.viz")
    module = _LAZY.get(name)
    if module is not None:
        return getattr(import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY) + ["viz"])


__all__ = [
    # constants
    "C0", "EPS0", "MU0", "ETA0",
    # grid
    "FDTDGrid", "create_grid",
    # materials
    "set_vacuum", "set_material_arrays", "set_box", "set_cylinder", "set_coax",
    # pml
    "CPMLArrays", "init_cpml", "update_H_pml", "update_E_pml", "ALL_FACES",
    # pec
    "apply_pec_faces", "apply_pec_mask",
    # sources
    "Waveform", "GaussianPulse",
    "Source", "PointSource", "ArraySource", "PlaneSource", "LineSource", "VolumeSource",
    # monitors
    "FieldProbe", "SnapshotMonitor", "EnergyMonitor",
    "record_field", "record_snapshot", "record_energy",
    # update
    "update_H", "update_E",
    # simulation
    "Simulation",
    # viz (lazy)
    *_LAZY,
]
