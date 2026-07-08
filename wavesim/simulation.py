"""
simulation.py — Simulation orchestration class.

A thin wrapper that runs the canonical FDTD time loop so scripts don't have to
re-type it. It *orchestrates* the existing pure functions — it does not replace
them or hide the physics. Anything you can do by hand you can still do by hand;
``Simulation`` just bundles the grid, the optional CPML, the sources, the
monitors, and the PEC-face list, and steps them in the fixed order:

    update_H → update_H_pml → update_E → update_E_pml
    → apply_pec_faces → apply_pec_mask
    → sources.inject → monitors.record → time_step += 1

Example
-------
    import wavesim as ws

    grid = ws.create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3)
    grid = ws.set_vacuum(grid)
    cpml = ws.init_cpml(grid, d_pml=10)

    sim = ws.Simulation(grid, cpml=cpml)
    sim.add_source(ws.PointSource('Ez', 50e-3, 50e-3, 0.0, ws.GaussianPulse.for_fmax(10e9)))
    snap = sim.add_monitor(ws.SnapshotMonitor('Ez', at_z=0.0, every_N_steps=20))
    sim.run(2000)
    # snap.snapshots now holds the recorded frames; sim.grid is the final state.
"""

import sys
import time
from typing import Callable, Iterable

from wavesim.grid import FDTDGrid
from wavesim.pml import CPMLArrays
from wavesim.pec import apply_pec_faces, apply_pec_mask
from wavesim.sources import Source
from wavesim.monitors import (
    FieldProbe, SnapshotMonitor, EnergyMonitor,
    VoltageMonitor, CurrentMonitor,
    record_field, record_snapshot, record_energy,
    record_voltage, record_current,
)


# Map each monitor type to its recorder. Keeps the monitors as plain data
# while letting the loop dispatch uniformly.
_RECORDERS = {
    FieldProbe:     record_field,
    SnapshotMonitor:  record_snapshot,
    EnergyMonitor:    record_energy,
    VoltageMonitor:   record_voltage,
    CurrentMonitor:   record_current,
}


def _load_backend(backend: str):
    """Return ``(update_H, update_E, update_H_pml, update_E_pml)`` for a backend.

    Importing the numba backend is deferred to here so that ``numba`` is only a
    dependency when ``backend='numba'`` is actually requested — the default numpy
    path has no extra imports.
    """
    if backend == 'numpy':
        from wavesim.update import update_H, update_E
        from wavesim.pml import update_H_pml, update_E_pml
    elif backend == 'numba':
        from wavesim.backend_numba import (
            update_H, update_E, update_H_pml, update_E_pml)
    elif backend == 'cuda':
        from wavesim.backend_cuda import (
            update_H, update_E, update_H_pml, update_E_pml)
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Expected 'numpy', 'numba' or 'cuda'.")
    return update_H, update_E, update_H_pml, update_E_pml


class Simulation:
    """
    Orchestrates the canonical FDTD time loop over a grid and its components.

    Parameters
    ----------
    grid : FDTDGrid
        The state object (already given materials/geometry).
    cpml : CPMLArrays, optional
        From ``init_cpml``. If omitted, the CPML correction steps are skipped
        (use this for a closed, lossless PEC cavity).
    sources : iterable of Source, optional
        Excitations injected each step (see :mod:`wavesim.sources`).
    monitors : iterable, optional
        Any mix of FieldProbe / SnapshotMonitor / EnergyMonitor /
        VoltageMonitor / CurrentMonitor; recorded each step.
    pec_faces : tuple of str, optional
        Domain faces to hold as PEC walls each step, e.g. ('y0', 'y1').
        ``apply_pec_mask`` always runs as well (it is a no-op when the grid has
        no ``pec_mask``), so interior conductors placed by the material helpers
        are enforced automatically.
    backend : {'numpy', 'numba', 'cuda'}, optional
        Which implementation of the four hot update functions to call. ``'numpy'``
        (default) uses the validated reference in :mod:`wavesim.update` /
        :mod:`wavesim.pml`; ``'numba'`` uses the multithreaded JIT kernels in
        :mod:`wavesim.backend_numba`, which are bit-for-bit identical (no parallel
        reductions) but parallelised across cores for large 3D grids. ``'cuda'``
        runs the curl/CPML/PEC updates on an NVIDIA GPU
        (:mod:`wavesim.backend_cuda`): ``run()`` keeps the fields resident on the
        device for the whole run (no per-step transfer when there are no host
        hooks), and matches the reference to floating-point tolerance. Allocate
        the grid as ``float32`` (``create_grid(..., dtype=np.float32)``) for the
        best GPU throughput on consumer cards. The first step of ``'numba'`` /
        ``'cuda'`` pays a one-time JIT/kernel compile cost. PEC, sources, and
        monitors are backend-independent and run identically either way.

        .. note::
           ``'cuda'`` needs a CUDA-capable GPU and the toolkit; on machines where
           Windows Smart App Control blocks the default binding, set
           ``NUMBA_CUDA_USE_NVIDIA_BINDING=0`` (backend_cuda sets this on import).
           Per-step host hooks (sources / monitors) currently sync the E/H fields
           around them; footprint-only sync is a future optimisation.

    Notes
    -----
    The simulation time passed to sources is ``grid.time_step * grid.dt``,
    evaluated *before* the counter is incremented — identical to the ``t = n*dt``
    used by the hand-written loops, so results are bit-for-bit the same.
    """

    def __init__(self, grid: FDTDGrid,
                 cpml: CPMLArrays = None,
                 sources: Iterable[Source] = (),
                 monitors: Iterable = (),
                 pec_faces: tuple = (),
                 backend: str = 'numpy') -> None:
        self.grid = grid
        self.cpml = cpml
        self.sources = list(sources)
        self.monitors = list(monitors)
        self.pec_faces = tuple(pec_faces)
        self.backend = backend
        self._update_H, self._update_E, self._update_H_pml, self._update_E_pml = \
            _load_backend(backend)

    # ------------------------------------------------------------------ #
    # Building up the simulation
    # ------------------------------------------------------------------ #
    def add_source(self, source: Source) -> Source:
        """Register a source; returns it for convenience."""
        self.sources.append(source)
        return source

    def add_monitor(self, monitor):
        """Register a monitor; returns it so you can read its data later."""
        if type(monitor) not in _RECORDERS:
            raise TypeError(
                f"Unknown monitor type {type(monitor).__name__}. "
                f"Expected one of {[t.__name__ for t in _RECORDERS]}.")
        self.monitors.append(monitor)
        return monitor

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #
    def step(self) -> FDTDGrid:
        """Advance the simulation by one timestep (the canonical loop body)."""
        grid = self.grid
        t = grid.time_step * grid.dt

        # 1-2. H update (+ CPML correction)
        grid = self._update_H(grid)
        if self.cpml is not None:
            grid, self.cpml = self._update_H_pml(grid, self.cpml)

        # 3-4. E update (+ CPML correction)
        grid = self._update_E(grid)
        if self.cpml is not None:
            grid, self.cpml = self._update_E_pml(grid, self.cpml)

        # 5. PEC — always after the E update (+ CPML)
        if self.pec_faces:
            grid = apply_pec_faces(grid, faces=self.pec_faces)
        grid = apply_pec_mask(grid)              # no-op if no pec_mask

        # 6. Sources (soft, additive)
        for src in self.sources:
            src.inject(grid, t)

        # 7. Monitors
        for mon in self.monitors:
            _RECORDERS[type(mon)](mon, grid)

        # 8. Advance the step counter (monitors timestamp from it)
        grid.time_step += 1

        self.grid = grid
        return grid

    def run(self, n_steps: int,
            callback: Callable[["Simulation", int], None] = None,
            verbose: int = 0) -> FDTDGrid:
        """
        Run ``n_steps`` timesteps.

        Parameters
        ----------
        n_steps : int
            Number of steps to advance.
        callback : callable, optional
            Called as ``callback(sim, n)`` after each step — handy for custom
            per-step logic without unrolling the loop.
        verbose : int, optional
            Console verbosity (default ``0``):

            * ``0`` — silent (the original behaviour).
            * ``1`` — print a rolling one-line status to stderr,
              ``step n/N (pct) | steps/s | sim-time | ETA``, updated in place and
              throttled to ~10 Hz so it adds negligible overhead to the loop.

        Returns
        -------
        FDTDGrid
            The final grid state (also available as ``self.grid``).
        """
        if self.backend == 'cuda':
            return self._run_cuda_resident(n_steps, callback, verbose)

        report = self._make_progress_reporter(n_steps) if verbose >= 1 else None
        for n in range(n_steps):
            self.step()
            if callback is not None:
                callback(self, n)
            if report is not None:
                report(n)
        return self.grid

    def _run_cuda_resident(self, n_steps: int,
                           callback: Callable[["Simulation", int], None] = None,
                           verbose: int = 0) -> FDTDGrid:
        """GPU fast path: keep the fields resident on the device across the whole
        run (see :class:`wavesim.backend_cuda.CudaResident`).

        The curl/CPML/PEC updates run on the GPU with no per-step transfer. Host
        hooks (sources, monitors, callback) still run on the CPU; on the steps
        where they are present the E/H fields are synced device->host before them
        and host->device after, preserving the exact ``step()`` semantics and
        ordering. With no per-step hooks nothing is transferred until the end.
        """
        from wavesim.backend_cuda import CudaResident

        res = CudaResident(self.grid, self.cpml, self.pec_faces)
        # A callback may inspect the fields, so treat it as needing a host sync.
        has_hooks = bool(self.sources or self.monitors or callback is not None)
        report = self._make_progress_reporter(n_steps) if verbose >= 1 else None

        for n in range(n_steps):
            grid = self.grid
            t = grid.time_step * grid.dt

            # 1-4. H, CPML-H, E, CPML-E, PEC — all on the device.
            res.step_evolution()

            # 5-6. Host hooks: sync E/H down, inject/record, sync back.
            if has_hooks:
                res.download_EH(grid)
                for src in self.sources:
                    src.inject(grid, t)
                for mon in self.monitors:
                    _RECORDERS[type(mon)](mon, grid)
                if self.sources:
                    res.upload_EH(grid)   # push source writes back to the device

            grid.time_step += 1
            if callback is not None:
                callback(self, n)
            if report is not None:
                report(n)

        res.download_EH(self.grid)   # final host copy of the fields
        res.sync()
        return self.grid

    # ------------------------------------------------------------------ #
    # Progress reporting
    # ------------------------------------------------------------------ #
    def _make_progress_reporter(self, n_steps: int):
        """Build a throttled rolling-progress printer for a ``run`` of length
        ``n_steps``. Returns a ``report(n)`` closure to call after each step
        (``n`` is the 0-based step index), or ``None`` if there is nothing to
        report. The first/last steps are always drawn; in between it updates at
        most every ~0.1 s so the print cost stays off the hot path."""
        if n_steps <= 0:
            return None

        stream = sys.stderr
        t0 = time.perf_counter()
        last_drawn = [0.0]
        dt = self.grid.dt

        def report(n):
            now = time.perf_counter()
            done = n + 1
            is_last = done == n_steps
            # Throttle: skip unless ~0.1 s elapsed since the last redraw, but
            # always draw the final step so the line ends on 100%.
            if not is_last and (now - last_drawn[0]) < 0.1:
                return
            last_drawn[0] = now

            elapsed = now - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            pct = 100.0 * done / n_steps
            sim_t = self.grid.time_step * dt          # physical time reached
            eta = (n_steps - done) / rate if rate > 0 else 0.0

            line = (f"\r  {self.backend} | step {done}/{n_steps} ({pct:5.1f}%)"
                    f" | {rate:7.0f} steps/s | t={_fmt_time(sim_t)}"
                    f" | ETA {_fmt_dur(eta)}")
            stream.write(line)
            if is_last:
                stream.write(f"   done in {_fmt_dur(elapsed)}\n")
            stream.flush()

        return report


def _fmt_time(seconds: float) -> str:
    """Format a physical simulation time with an SI prefix (ns/µs/ms/s)."""
    for scale, unit in ((1e-12, "ps"), (1e-9, "ns"), (1e-6, "us"), (1e-3, "ms")):
        if abs(seconds) < scale * 1000:
            return f"{seconds / scale:6.2f} {unit}"
    return f"{seconds:6.2f} s"


def _fmt_dur(seconds: float) -> str:
    """Format a wall-clock duration compactly (e.g. ``3.4s`` or ``1m02s``)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"
