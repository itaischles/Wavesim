"""Waveform temporal profiles.

The interesting assertions here are about :class:`~wavesim.sources.Sinusoid`'s
turn-on. A bare ``sin(ωt)`` switched on at t=0 is continuous in value but not in
slope, and that kink is a broadband impulse — the whole reason the class exists.
So the tests pin *smoothness*, not just amplitude: the ramped waveform's initial
slope must be far below the bare carrier's, and the steady state must still be a
clean sinusoid of the requested amplitude and frequency.
"""
import numpy as np
import pytest

import wavesim as ws


FREQ = 10e9
PERIOD = 1.0 / FREQ


def test_zero_before_turn_on():
    wf = ws.Sinusoid(frequency=FREQ)
    assert wf(-PERIOD) == 0.0
    assert wf(0.0) == 0.0


def test_reaches_full_amplitude_after_ramp():
    wf = ws.Sinusoid(frequency=FREQ, amplitude=3.0, ramp_cycles=3.0)
    # Sample a few periods past the ramp; the envelope is 1 there, so the
    # extremes must be the requested amplitude.
    t = np.linspace(3.0 * PERIOD, 8.0 * PERIOD, 2001)
    peak = max(abs(wf(ti)) for ti in t)
    assert peak == pytest.approx(3.0, rel=1e-4)


def test_ramp_is_monotonic_and_below_amplitude():
    wf = ws.Sinusoid(frequency=FREQ, amplitude=1.0, ramp_cycles=3.0)
    # Envelope recovered at carrier peaks (t where sin = 1): quarter period in,
    # then every period. Must rise monotonically and never overshoot.
    peaks = [wf(0.25 * PERIOD + n * PERIOD) for n in range(3)]
    assert all(0.0 < p < 1.0 for p in peaks)
    assert peaks == sorted(peaks)


def test_turn_on_is_smooth():
    """Ramped slope at t=0+ must be far below the bare carrier's."""
    dt = PERIOD / 1000.0
    ramped = ws.Sinusoid(frequency=FREQ, ramp_cycles=3.0)
    bare = ws.Sinusoid(frequency=FREQ, ramp_cycles=0.0)
    slope_ramped = (ramped(dt) - ramped(0.0)) / dt
    slope_bare = (bare(dt) - bare(0.0)) / dt
    assert abs(slope_ramped) < 0.01 * abs(slope_bare)


def test_ramp_cycles_zero_is_bare_carrier():
    wf = ws.Sinusoid(frequency=FREQ, amplitude=2.0, phase=0.5, ramp_cycles=0.0)
    t = 1.7 * PERIOD
    assert wf(t) == pytest.approx(
        2.0 * np.sin(2.0 * np.pi * FREQ * t + 0.5), rel=1e-12)


def test_phase_shifts_the_carrier():
    """phase=π/2 makes the carrier start at its peak (envelope still zero)."""
    quarter = ws.Sinusoid(frequency=FREQ, phase=np.pi / 2, ramp_cycles=0.0)
    assert quarter(0.0) == 0.0          # t <= 0 is identically zero
    tiny = 1e-15
    assert quarter(tiny) == pytest.approx(1.0, abs=1e-6)


def test_center_frequency_is_the_carrier():
    assert ws.Sinusoid(frequency=FREQ).center_frequency == FREQ


def test_usable_as_a_source_waveform():
    """A Sinusoid is a plain callable, so any Source accepts it."""
    grid = ws.create_grid(Nx=8, Ny=8, Nz=8, dx=1e-3, dy=1e-3, dz=1e-3)
    ws.set_vacuum(grid)
    src = ws.PointSource('Ez', 4e-3, 4e-3, 4e-3, ws.Sinusoid(frequency=FREQ))
    src.inject(grid, 2.0 * PERIOD)
    assert np.count_nonzero(grid.Ez) == 1
