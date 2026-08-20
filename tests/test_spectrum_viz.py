"""Drawing spectra, Z(f) and Y(f).

Plotting tests are usually thin — a picture is hard to assert about — but three
things here are not cosmetic, and each has a way to fail that produces a
plausible-looking plot rather than an error:

**The frequency window.** An rfft axis runs to Nyquist, which on an FDTD record
is hundreds of gigahertz, while a broadband pulse illuminates the first few
percent of that. A plot that autoscales to the data extent is a spike in the
leftmost pixel column with a vast empty plain to its right, and it is not
obviously *wrong* — you just cannot read anything off it. So the default x-limit
tracks the usable band, and that is asserted.

**The masked bins.** Out-of-band bins are NaN by construction, and NaN is
contagious: ``np.unwrap`` propagates one forward through every sample after it,
which would silently blank the rest of a phase curve. The phase is unwrapped per
contiguous run instead, and both halves of that — the segments survive, the gap
stays a gap — are pinned.

**Units on one axis.** Volts per hertz and amps per hertz differ by orders of
magnitude on any real port, so sharing a y-axis flattens one of them to the
baseline. They go on twin axes, as in the time-domain plot.
"""

import matplotlib
matplotlib.use('Agg')                       # no display in CI; must precede pyplot

import matplotlib.pyplot as plt
import numpy as np
import pytest

import wavesim as ws
from wavesim.spectrum import spectrum, impedance, admittance, usable_band
from wavesim.viz import (plot_spectrum, plot_bode, plot_impedance_parts,
                         _freq_axis, _phase_deg)


DT = 1e-12
N = 2048


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


class VoltageMonitor:
    """Duck-type of the real monitor; the class name selects the stagger."""
    def __init__(self, times, values):
        self.times, self.values = list(times), list(values)


class CurrentMonitor(VoltageMonitor):
    pass


def _port(R=25.0, C=1.3e-12):
    """A recorded (V, I) pair through a known R∥C — see test_spectrum.py."""
    t = np.arange(N) * DT
    v = np.exp(-0.5 * ((t - 400e-12) / 40e-12) ** 2)
    f = np.fft.rfftfreq(N, DT)
    Y = 1.0 / R + 2j * np.pi * f * C
    i = np.fft.irfft(np.fft.rfft(v) * Y * np.exp(-1j * np.pi * f * DT), n=N)
    return VoltageMonitor(t, v), CurrentMonitor(t, i)


def _z():
    vm, im = _port()
    return impedance(vm, im)


def _spans(ax):
    """The axvspan patches on ``ax`` (a Rectangle, in current matplotlib)."""
    return [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]


def _curves(ax):
    """Data lines, excluding the dotted/solid reference guides."""
    return [l for l in ax.lines if l.get_linestyle() not in (':', 'None')
            and l.get_linewidth() > 1.0]


# ======================================================================= #
# The frequency axis
# ======================================================================= #

@pytest.mark.parametrize("top,expected", [
    (5e12, 'THz'), (20e9, 'GHz'), (3e6, 'MHz'), (4e3, 'kHz'), (100.0, 'Hz'),
])
def test_frequency_prefix_follows_the_data(top, expected):
    label, scale = _freq_axis(np.linspace(0.0, top, 10))
    assert expected in label
    assert top / scale >= 1.0


def test_axis_defaults_to_the_usable_band_not_to_nyquist():
    vm, im = _port()
    S = spectrum(vm)
    fig, ax = plot_spectrum(S)
    _lo, hi = usable_band(S)
    _label, scale = _freq_axis(S.freqs)

    # Nyquist is 500 GHz here and the pulse dies by ~20 GHz. The view must
    # follow the signal, or the plot is one pixel of data.
    assert ax.get_xlim()[1] == pytest.approx(1.05 * hi / scale, rel=1e-9)
    assert ax.get_xlim()[1] < 0.1 * S.freqs.max() / scale


def test_fmax_is_honoured_in_hertz():
    fig, ax = plot_spectrum(spectrum(_port()[0]), fmax=8e9)
    _label, scale = _freq_axis(np.array([8e9]))
    assert ax.get_xlim()[1] == pytest.approx(8e9 / scale)


# ======================================================================= #
# plot_spectrum
# ======================================================================= #

def test_volts_and_amps_land_on_separate_axes():
    vm, im = _port()
    fig, ax = plot_spectrum([spectrum(vm), spectrum(im)])
    assert hasattr(ax, 'right_ax')
    assert ax.get_ylabel().startswith('Magnitude (V')
    assert ax.right_ax.get_ylabel().startswith('Magnitude (A')
    assert len(ax.lines) == 1 and len(ax.right_ax.lines) == 1


def test_a_lone_current_keeps_the_primary_axis():
    fig, ax = plot_spectrum(spectrum(_port()[1]))
    assert not hasattr(ax, 'right_ax')
    assert ax.get_ylabel().startswith('Magnitude (A')


def test_the_shaded_span_is_the_usable_band():
    vm, _im = _port()
    S = spectrum(vm)
    fig, ax = plot_spectrum(S)
    lo, hi = usable_band(S)
    _label, scale = _freq_axis(S.freqs)

    spans = _spans(ax)
    assert len(spans) == 1
    assert spans[0].get_x() == pytest.approx(lo / scale, rel=1e-9)
    assert spans[0].get_x() + spans[0].get_width() == pytest.approx(
        hi / scale, rel=1e-9)


def test_shading_can_be_switched_off():
    fig, ax = plot_spectrum(spectrum(_port()[0]), shade=False)
    assert not _spans(ax)


def test_db_switches_the_scale_rather_than_stacking_two_logs():
    fig, ax = plot_spectrum(spectrum(_port()[0]), db=True)
    assert ax.get_yscale() == 'linear'        # dB is already logarithmic
    assert 'dB' in ax.get_ylabel()


def test_monitors_are_transformed_on_the_way_in():
    # The whole point of the adapter: no explicit spectrum() call needed.
    vm, im = _port()
    fig, ax = plot_spectrum([vm, im])
    assert len(ax.lines) == 1 and len(ax.right_ax.lines) == 1


def test_a_port_pair_names_its_quantity():
    class Port:
        def __init__(self, vm, im):
            self.times = vm.times
            self.voltages = vm.values
            self.currents = im.values

    port = Port(*_port())
    fig, ax = plot_spectrum([(port, 'V'), (port, 'I')])
    assert len(ax.lines) == 1 and len(ax.right_ax.lines) == 1


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        plot_spectrum([])


# ======================================================================= #
# plot_bode
# ======================================================================= #

def test_bode_draws_magnitude_and_phase():
    Z = _z()
    fig, (ax_mag, ax_ph) = plot_bode(Z)
    assert ax_mag.get_yscale() == 'log'
    assert 'Ω' in ax_mag.get_ylabel()
    assert ax_ph.get_ylabel() == 'Phase (deg)'
    assert len(ax_mag.lines) == 1

    # The drawn magnitude is the data, not a re-derivation of it.
    y = ax_mag.lines[0].get_ydata()
    assert np.allclose(y[np.isfinite(y)], Z.magnitude[np.isfinite(Z.values)])


def test_bode_phase_matches_the_impedance():
    # R∥C: the phase runs from 0 at DC toward -90° well past the corner.
    Z = _z()
    fig, (_ax_mag, ax_ph) = plot_bode(Z)
    y = _curves(ax_ph)[0].get_ydata()
    y = y[np.isfinite(y)]
    assert y.max() == pytest.approx(0.0, abs=1e-6)
    assert -90.0 < y.min() < -45.0


def test_bode_overlays_several_spectra():
    vm, im = _port()
    fig, (ax_mag, ax_ph) = plot_bode([impedance(vm, im), impedance(vm, im)])
    # The phase axis also carries the 0/±90° guides, which are not curves.
    assert len(_curves(ax_mag)) == 2 and len(_curves(ax_ph)) == 2


def test_bode_can_go_log_in_frequency():
    fig, (ax_mag, _ax_ph) = plot_bode(_z(), logf=True)
    assert ax_mag.get_xscale() == 'log'


# ======================================================================= #
# Phase across a masked gap
# ======================================================================= #

def _gapped(freqs, phases, blank):
    """A Spectrum whose ``blank`` bins are NaN, for the gap tests."""
    vals = np.exp(1j * np.asarray(phases, dtype=float))
    vals[blank] = np.nan + 0j
    return ws.Spectrum(freqs=np.asarray(freqs, dtype=float), values=vals, dt=DT)


def test_a_masked_gap_does_not_erase_the_rest_of_the_phase():
    # The failure this guards: np.unwrap over an array containing one NaN
    # returns NaN for every sample after it, so a single out-of-band bin would
    # blank the whole high-frequency half of the curve.
    f = np.linspace(0.0, 10e9, 40)
    ph = np.linspace(0.0, 6.0, 40)
    blank = slice(10, 14)
    S = _gapped(f, ph, blank)

    out = _phase_deg(S, unwrap=True)
    assert np.all(np.isnan(out[blank]))       # the gap stays a gap
    after = np.ones(40, dtype=bool)
    after[blank] = False
    assert np.all(np.isfinite(out[after]))    # and everything else survives


def test_each_run_is_unwrapped_on_its_own():
    # Continuous within a segment: a phase sweeping past ±180° must not fold.
    f = np.linspace(0.0, 10e9, 40)
    ph = np.linspace(0.0, 12.0, 40)           # ~2 full turns
    S = _gapped(f, ph, slice(20, 22))
    out = _phase_deg(S, unwrap=True)
    for seg in (slice(0, 20), slice(22, 40)):
        assert np.max(np.abs(np.diff(out[seg]))) < 180.0


def test_unwrap_can_be_switched_off():
    f = np.linspace(0.0, 10e9, 40)
    S = _gapped(f, np.linspace(0.0, 12.0, 40), slice(0, 0))
    wrapped = _phase_deg(S, unwrap=False)
    assert wrapped.min() >= -180.0 and wrapped.max() <= 180.0


def test_an_entirely_masked_spectrum_draws_nothing_rather_than_raising():
    f = np.linspace(0.0, 10e9, 20)
    S = ws.Spectrum(freqs=f, values=np.full(20, np.nan + 0j), dt=DT)
    assert np.all(np.isnan(_phase_deg(S, unwrap=True)))
    fig, (ax_mag, _ax_ph) = plot_bode(S)      # and the plot still builds
    assert len(ax_mag.lines) == 1


# ======================================================================= #
# plot_impedance_parts
# ======================================================================= #

def test_parts_plot_draws_r_and_x():
    Z = _z()
    fig, ax = plot_impedance_parts(Z)
    assert len(ax.lines) == 3                 # R, X, and the zero line
    assert np.allclose(_curves(ax)[0].get_ydata(), Z.real, equal_nan=True)
    assert np.allclose(_curves(ax)[1].get_ydata(), Z.imag, equal_nan=True)
    assert 'R, X (Ω)' in ax.get_ylabel()


def test_parts_plot_renames_itself_for_an_admittance():
    # Same picture, different names: G and B, in siemens.
    vm, im = _port()
    fig, ax = plot_impedance_parts(admittance(vm, im))
    assert 'G, B (S)' in ax.get_ylabel()
    assert [l.get_label() for l in _curves(ax)[:2]] == ['G', 'B']


def test_parts_plot_is_linear_in_both_axes():
    # The whole reason this plot exists next to the Bode one: a straight-line
    # X = 2πfL is only straight on linear axes.
    fig, ax = plot_impedance_parts(_z())
    assert ax.get_xscale() == 'linear' and ax.get_yscale() == 'linear'


# ======================================================================= #
# usable_band
# ======================================================================= #

def test_usable_band_narrows_as_the_floor_rises():
    S = spectrum(_port()[0])
    wide, narrow = usable_band(S, floor=1e-6), usable_band(S, floor=1e-1)
    assert wide[1] > narrow[1]


def test_usable_band_intersects_over_several_spectra():
    vm, im = _port()
    v, i = spectrum(vm), spectrum(im)
    both = usable_band(v, i)
    assert both[1] <= min(usable_band(v)[1], usable_band(i)[1]) + 1e-9


def test_usable_band_of_silence_is_nan():
    f = np.linspace(0.0, 10e9, 20)
    S = ws.Spectrum(freqs=f, values=np.zeros(20, dtype=complex), dt=DT)
    assert all(np.isnan(x) for x in usable_band(S))


def test_usable_band_rejects_mismatched_axes():
    S = spectrum(_port()[0])
    other = ws.Spectrum(freqs=S.freqs[:10], values=S.values[:10], dt=DT)
    with pytest.raises(ValueError, match="one frequency axis"):
        usable_band(S, other)


# ======================================================================= #
# Lazy export
# ======================================================================= #

def test_the_plots_are_reachable_from_the_package_namespace():
    for name in ('plot_spectrum', 'plot_bode', 'plot_impedance_parts'):
        assert callable(getattr(ws, name))
        assert name in ws.__all__
