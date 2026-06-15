"""
sources.py — Source waveforms and the v2 Source injection abstraction.

Two layers live here, and they compose:

1. Waveforms — the *time* part of an excitation. A waveform is any callable
   ``f(t) -> float``. ``GaussianSource`` is the built-in baseband pulse (and is
   itself callable); for a narrowband/CW drive pass your own ``lambda t: ...``.

2. Source objects — the v2 injection abstraction (ROADMAP §2). A ``Source``
   bundles *where* (``spatial_profile(grid)``), *when* (``time_function(t)``)
   and *which component* it drives, and exposes ``inject(grid, t)`` that does the
   soft, additive write. ``Simulation`` calls ``inject`` once per timestep; you
   can also call it yourself from a hand-written loop.

The functional v1 recipe still works unchanged — a ``Source`` is purely a
convenience around the same one-liner:

    source = GaussianSource(t0=30*grid.dt, width=10*grid.dt)
    # in a hand-written loop:
    grid.Ez[i, j, k] += gaussian_pulse(source, t)        # or: source(t)

Choosing parameters for a target maximum frequency f_max:
    width = 1.0 / (2 * np.pi * f_max)   # -3 dB bandwidth ≈ f_max
    t0    = 4 * width                    # pulse fully risen by t=0 within 1% of peak

Soft injection (+=) is transparent to passing waves (no impedance mismatch).
Hard injection (=) reflects waves — do not use; every Source here adds (+=).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import numpy as np

from wavesim.grid import FDTDGrid


# ====================================================================== #
# Waveforms (the time part)
# ====================================================================== #

@dataclass
class GaussianSource:
    """Gaussian pulse waveform.

    Callable: ``GaussianSource(...)(t)`` returns the pulse value at ``t``, so an
    instance can be passed directly as the ``waveform`` of a :class:`Source`.
    """
    t0: float           # pulse centre time (s)
    width: float        # pulse half-width / standard deviation (s)
                        # spectral bandwidth ≈ 1 / (2π · width)
    amplitude: float = 1.0

    def __call__(self, t: float) -> float:
        return gaussian_pulse(self, t)


def gaussian_pulse(source: GaussianSource, t: float) -> float:
    """
    Evaluate Gaussian pulse at time t.

    Returns:
        amplitude * exp(-0.5 * ((t - t0) / width)²)
    """
    return source.amplitude * np.exp(-0.5 * ((t - source.t0) / source.width) ** 2)


def make_source_for_fmax(f_max: float, amplitude: float = 1.0) -> GaussianSource:
    """
    Convenience constructor: create a GaussianSource targeting f_max Hz.

    Parameters
    ----------
    f_max : float
        Maximum frequency of interest (Hz).
        The pulse bandwidth (-3 dB) will be approximately f_max.
    amplitude : float
        Peak amplitude of the pulse.

    Returns
    -------
    GaussianSource
        With t0 and width chosen so the pulse is fully contained in the
        simulation window (< 1% amplitude at t=0).
    """
    width = 1.0 / (2.0 * np.pi * f_max)
    t0    = 4.0 * width   # pulse fully risen by t=0 within ~0% of peak
    return GaussianSource(t0=t0, width=width, amplitude=amplitude)


# ====================================================================== #
# Source objects (the v2 injection abstraction)
# ====================================================================== #

class Source(ABC):
    """
    Base class for excitations.

    A Source knows three things:
      * ``component`` — which field array it drives ('Ex'..'Hz').
      * ``time_function(t)`` — the scalar waveform value at time ``t``.
      * ``spatial_profile(grid)`` — additive weights with the shape of the
        component array (Nx, Ny, Nz); the field gets ``time_function(t)`` times
        this profile added to it each step.

    ``inject(grid, t)`` performs the soft (additive) write and is what the time
    loop calls. The spatial profile is built once and cached, since geometry is
    fixed for the run. Subclasses implement ``time_function`` and
    ``spatial_profile``; a point/cheap source may override ``inject`` to avoid
    materialising a full-grid profile (see :class:`PointSource`).
    """

    component: str = 'Ez'

    def __init__(self) -> None:
        self._profile_cache: np.ndarray | None = None

    @abstractmethod
    def time_function(self, t: float) -> float:
        """Scalar waveform value at time ``t`` (seconds)."""

    @abstractmethod
    def spatial_profile(self, grid: FDTDGrid) -> np.ndarray:
        """Additive weights for ``component``, broadcastable to (Nx, Ny, Nz)."""

    def inject(self, grid: FDTDGrid, t: float) -> None:
        """Soft-add ``time_function(t) * spatial_profile(grid)`` into the field."""
        if self._profile_cache is None:
            self._profile_cache = self.spatial_profile(grid)
        arr = getattr(grid, self.component)
        arr += self.time_function(t) * self._profile_cache


class PointSource(Source):
    """
    Soft point excitation: one cell of one component, driven by a waveform.

    Equivalent to the v1 one-liner ``grid.<component>[i, j, k] += waveform(t)``,
    but as a reusable object. Overrides ``inject`` so no full-grid profile is
    allocated (the profile is a single cell).

    Parameters
    ----------
    component : str
        Field component to drive ('Ex'..'Hz').
    i, j, k : int
        Cell indices of the injection point (use k=0 for an Nz=1 slice).
    waveform : Callable[[float], float]
        Time function, e.g. a ``GaussianSource`` instance or a custom lambda.
    """

    def __init__(self, component: str, i: int, j: int, k: int,
                 waveform: Callable[[float], float]) -> None:
        super().__init__()
        self.component = component
        self.i, self.j, self.k = i, j, k
        self.waveform = waveform

    def time_function(self, t: float) -> float:
        return self.waveform(t)

    def spatial_profile(self, grid: FDTDGrid) -> np.ndarray:
        """Full-grid profile with a single 1.0 at (i, j, k) — for inspection."""
        prof = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=np.float64)
        prof[self.i, self.j, self.k] = 1.0
        return prof

    def inject(self, grid: FDTDGrid, t: float) -> None:
        getattr(grid, self.component)[self.i, self.j, self.k] += self.waveform(t)


class ArraySource(Source):
    """
    Distributed soft excitation from a user-supplied spatial profile.

    Covers line sources, shaped/annular drives, modal profiles, etc. — anything
    where you can express the per-cell injection weights as an array. The field
    update each step is ``component += waveform(t) * profile``.

    Parameters
    ----------
    component : str
        Field component to drive ('Ex'..'Hz').
    profile : np.ndarray
        Weights of shape (Nx, Ny, Nz) matching the grid. Zero cells are not
        driven; a single nonzero z-plane gives a planar source, etc.
    waveform : Callable[[float], float]
        Time function (e.g. a ``GaussianSource`` or a modulated-carrier lambda).

    Notes
    -----
    The profile is validated against the grid shape on first injection.
    """

    def __init__(self, component: str, profile: np.ndarray,
                 waveform: Callable[[float], float]) -> None:
        super().__init__()
        self.component = component
        self.profile = np.asarray(profile, dtype=np.float64)
        self.waveform = waveform

    def time_function(self, t: float) -> float:
        return self.waveform(t)

    def spatial_profile(self, grid: FDTDGrid) -> np.ndarray:
        expected = (grid.Nx, grid.Ny, grid.Nz)
        if self.profile.shape != expected:
            raise ValueError(
                f"ArraySource profile shape {self.profile.shape} does not match "
                f"grid shape {expected}.")
        return self.profile
