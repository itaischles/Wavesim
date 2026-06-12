"""
sources.py — Source time function and injection helpers.

For v1, spatial injection is hard-coded in each test/example (one line
directly in the time loop). The only reusable abstraction is the time
function itself.

Usage:
    source = GaussianSource(t0=30*grid.dt, width=10*grid.dt)
    # In time loop:
    grid.Ez[i, j, k] += gaussian_pulse(source, t)

Choosing parameters for a target maximum frequency f_max:
    width = 1.0 / (2 * np.pi * f_max)   # -3 dB bandwidth ≈ f_max
    t0    = 4 * width                    # pulse fully risen by t=0 within 1% of peak

Soft injection (+=) is transparent to passing waves (no impedance mismatch).
Hard injection (=) reflects waves — do not use.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class GaussianSource:
    """Gaussian pulse source parameters."""
    t0: float           # pulse centre time (s)
    width: float        # pulse half-width / standard deviation (s)
                        # spectral bandwidth ≈ 1 / (2π · width)
    amplitude: float = 1.0


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
