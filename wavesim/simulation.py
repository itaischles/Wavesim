"""
simulation.py — Simulation orchestration class.

A thin wrapper that runs the canonical FDTD time loop so scripts don't have to
re-type it. It *orchestrates* the existing pure functions — it does not replace
them or hide the physics. Anything you can do by hand you can still do by hand;
``Simulation`` just bundles the grid, the optional CPML, the sources, the
monitors, and the PEC-face list, and steps them in the fixed order:

    update_H → update_H_pml → boundaries.apply
    → update_E → update_E_pml
    → apply_pec_faces → apply_pec_mask → boundaries.apply_post_E
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
import warnings
from typing import Callable, Iterable

from wavesim.grid import FDTDGrid
from wavesim.pml import CPMLArrays
from wavesim.pec import apply_pec_faces, apply_pec_mask
from wavesim.sources import Source
from wavesim.monitors import (
    FieldProbe, SnapshotMonitor, PoyntingMonitor, EnergyMonitor,
    DissipationMonitor, VoltageMonitor, CurrentMonitor,
    record_field, record_snapshot, record_poynting, record_energy,
    record_dissipation, record_voltage, record_current,
)


# Map each monitor type to its recorder. Keeps the monitors as plain data
# while letting the loop dispatch uniformly.
_RECORDERS = {
    FieldProbe:     record_field,
    SnapshotMonitor:  record_snapshot,
    PoyntingMonitor:  record_poynting,
    EnergyMonitor:    record_energy,
    DissipationMonitor: record_dissipation,
    VoltageMonitor:   record_voltage,
    CurrentMonitor:   record_current,
}

# Monitors that carry the region/d_pml/faces trio and want it filled from the
# run's CPML (see Simulation._autofill_region).
_REGION_MONITORS = (EnergyMonitor, DissipationMonitor)


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
        Any mix of FieldProbe / SnapshotMonitor / PoyntingMonitor /
        EnergyMonitor / DissipationMonitor / VoltageMonitor / CurrentMonitor;
        recorded each step.
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

    conformal_stability : {'auto', 'warn', 'off'}, optional
        What to do about the conformal-PEC small-cut instability (plan S7).
        A cut-cell grid can diverge at the default clamp threshold while a
        *finer* mesh of the same model is fine, so the setting cannot be
        reasoned about and is measured instead:
        :func:`wavesim.stability.probe_growth` seeds noise, steps this exact
        scheme with no sources, and reports the growth rate.

        ``'auto'`` (default) raises ``grid.conformal_area_threshold`` in place
        until the probe is quiet, and warns when it had to.
        ``'warn'`` measures and warns but changes nothing.
        ``'off'`` skips the measurement.

        **Only a conformal grid is ever probed**, so nothing that ran before
        this existed pays for it, and a conformal grid whose threshold already
        works is left untouched — the check costs a few seconds and changes
        nothing. Read ``grid.conformal_area_threshold`` *after* construction
        when recording what a run did; under ``'auto'`` it is no longer
        necessarily the value that was asked for.

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
                 boundaries: Iterable = (),
                 backend: str = 'numpy',
                 conformal_stability: str = 'auto') -> None:
        self.grid = grid
        self.cpml = cpml
        self.sources = list(sources)
        self.monitors = list(monitors)
        self.pec_faces = tuple(pec_faces)
        self.boundaries = list(boundaries)
        self.backend = backend
        self._update_H, self._update_E, self._update_H_pml, self._update_E_pml = \
            _load_backend(backend)
        self.conformal_stability = conformal_stability
        self._check_conformal_stability()
        for mon in self.monitors:
            self._autofill_region(mon)

    def _check_conformal_stability(self) -> None:
        """Measure — and by default fix — the S7 small-cut instability.

        Runs here rather than in ``run()`` so the threshold is settled before
        anything reads it back, and because a caller that has built a Simulation
        has by then supplied the two things the probe needs that the grid does
        not carry: the PEC walls and the CPML.

        The clamp threshold reaches nothing but ``inv_A`` in the H update — the
        mode solver reads the open *edge* fractions and never the clamped face
        areas — so raising it here cannot invalidate a port solved earlier.
        """
        mode = self.conformal_stability
        if mode == 'off' or not self.grid.is_conformal:
            return
        if mode not in ('auto', 'warn'):
            raise ValueError(
                f"conformal_stability must be 'auto', 'warn' or 'off', got "
                f"{mode!r}")

        if self.backend == 'cuda':
            # The GPU has no conformal H kernel, and ``run()`` reaches it
            # through CudaResident, which does not check (plan R7). Probing
            # there would step the *staircase* scheme and report a reassuring
            # 1.000 for a grid whose conformal update was never executed — a
            # false pass is worse than no check. Defer to the backend's own
            # guard so there is one message, not two.
            from wavesim.backend_cuda import _refuse_conformal
            _refuse_conformal(self.grid)

        from wavesim.stability import ensure_stable_threshold, probe_growth

        probe_kw = dict(pec_faces=self.pec_faces, cpml=self.cpml,
                        backend=self.backend)
        if mode == 'auto':
            ensure_stable_threshold(self.grid, **probe_kw)
            return

        probe = probe_growth(self.grid, **probe_kw)
        if not probe.stable:
            warnings.warn(
                f"conformal PEC: this grid diverges at clamp threshold "
                f"{self.grid.conformal_area_threshold:.2f} — the seeded free "
                f"run grows {probe.growth:.4g} per step, implying "
                f"dt_max/dt = {probe.margin:.5f}. Raise "
                f"conformal_area_threshold, or pass "
                f"conformal_stability='auto' to have it raised for you.",
                RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------ #
    # Building up the simulation
    # ------------------------------------------------------------------ #
    def add_source(self, source: Source) -> Source:
        """Register a source; returns it for convenience."""
        self.sources.append(source)
        return source

    def add_boundary(self, boundary):
        """Register an absorbing/port boundary; returns it for convenience.

        Boundaries run **between the H and E updates** each step (unlike sources,
        which run after the E update), because a modal impedance-sheet port sets
        the ghost tangential H that the very next E update consumes — a source
        hook would be clobbered by the following step's H update before it is
        ever used. See :class:`~wavesim.sources.ModalPort`."""
        self.boundaries.append(boundary)
        return boundary

    def add_monitor(self, monitor):
        """Register a monitor; returns it so you can read its data later."""
        if type(monitor) not in _RECORDERS:
            raise TypeError(
                f"Unknown monitor type {type(monitor).__name__}. "
                f"Expected one of {[t.__name__ for t in _RECORDERS]}.")
        self._autofill_region(monitor)
        self.monitors.append(monitor)
        return monitor

    def _autofill_region(self, monitor) -> None:
        """Fill an ``'interior'`` monitor's PML geometry from this run's CPML.

        The monitor needs the PML thickness and absorbing faces to know which
        outer cells to drop, but they live on the ``CPMLArrays``, not the grid.
        Copy them from ``self.cpml`` unless the user set them explicitly. With no
        CPML, ``'interior'`` trims nothing (``d_pml=0``) and equals ``'full'``.
        """
        if not isinstance(monitor, _REGION_MONITORS) or monitor.region != 'interior':
            return
        if self.cpml is not None:
            if monitor.d_pml is None:
                monitor.d_pml = self.cpml.d_pml
            if monitor.faces is None:
                monitor.faces = self.cpml.faces
        else:
            if monitor.d_pml is None:
                monitor.d_pml = 0

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

        # 2b. Boundaries — modal impedance-sheet ports set the ghost tangential H
        # here, *between* the H and E updates, so the E update below consumes it.
        for bnd in self.boundaries:
            bnd.apply(grid, t)

        # 3-4. E update (+ CPML correction)
        grid = self._update_E(grid)
        if self.cpml is not None:
            grid, self.cpml = self._update_E_pml(grid, self.cpml)

        # 5. PEC — always after the E update (+ CPML)
        if self.pec_faces:
            grid = apply_pec_faces(grid, faces=self.pec_faces)
        grid = apply_pec_mask(grid)              # no-op if no pec_mask

        # 5b. Boundaries, second hook — a port that has to *constrain* E (rather
        # than write H) needs the slot after the E update and after the PEC
        # masking, which is where every other E constraint is applied. Optional:
        # a boundary without one is unaffected. See
        # :meth:`~wavesim.sources.ModalPort.apply_post_E`.
        for bnd in self.boundaries:
            post = getattr(bnd, 'apply_post_E', None)
            if post is not None:
                post(grid, t)

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
