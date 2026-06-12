"""
test_01_source_viz.py — Source waveform visualisation.

NO PHYSICS. NO TIME LOOP.

Validates:
    - GaussianSource construction via make_source_for_fmax()
    - Pulse is fully contained in the simulation window (< 1% amplitude at both ends)
    - Printed bandwidth matches f_max target
    - plot_source_waveform() renders without error

Pass criteria:
    1. Amplitude at t=0      < 1% of peak
    2. Amplitude at t=t_end  < 1% of peak
    3. Bandwidth printed to stdout ≈ f_max (within 5%)

Run:
    cd C:\\Users\\itais\\Desktop\\Wavesim
    python tests\\test_01_source_viz.py

Output: saves test_01_output.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavesim.grid import create_grid
from wavesim.sources import GaussianSource, gaussian_pulse, make_source_for_fmax
from wavesim.viz import plot_source_waveform


def test_01_source_viz():
    print("=" * 60)
    print("TEST 01 — Source Waveform Visualisation")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Create a minimal grid — only need dt
    # ------------------------------------------------------------------ #
    grid = create_grid(Nx=200, Ny=200, Nz=1, dx=0.5e-3)
    n_steps = 2000

    print(f"\nGrid dt = {grid.dt * 1e12:.4f} ps")
    print(f"Simulation window = {n_steps * grid.dt * 1e9:.3f} ns")

    # ------------------------------------------------------------------ #
    # Step 2: Build source targeting f_max = 10 GHz
    # ------------------------------------------------------------------ #
    f_max = 10e9   # 10 GHz
    source = make_source_for_fmax(f_max)

    print(f"\nGaussianSource:")
    print(f"  f_max  = {f_max/1e9:.1f} GHz")
    print(f"  t0     = {source.t0 * 1e12:.2f} ps")
    print(f"  width  = {source.width * 1e12:.2f} ps")

    # ------------------------------------------------------------------ #
    # Step 3: Check pulse is contained in the window
    # ------------------------------------------------------------------ #
    t_start = 0.0
    t_end   = n_steps * grid.dt

    amp_at_start = abs(gaussian_pulse(source, t_start)) / source.amplitude
    amp_at_end   = abs(gaussian_pulse(source, t_end))   / source.amplitude

    print(f"\nWindow containment check:")
    print(f"  Amplitude at t=0:     {amp_at_start:.2e}  (must be < 0.01)")
    print(f"  Amplitude at t=t_end: {amp_at_end:.2e}  (must be < 0.01)")

    assert amp_at_start < 0.01, \
        f"Pulse not contained at t=0: amplitude = {amp_at_start:.4f} (> 1%)"
    assert amp_at_end < 0.01, \
        f"Pulse not contained at t_end: amplitude = {amp_at_end:.4f} (> 1%)"

    print("  [PASS] Pulse fully contained in simulation window")

    # ------------------------------------------------------------------ #
    # Step 4: Check bandwidth
    # ------------------------------------------------------------------ #
    bw_hz = 1.0 / (2.0 * np.pi * source.width)
    bw_error = abs(bw_hz - f_max) / f_max

    print(f"\nBandwidth check:")
    print(f"  Target f_max  = {f_max/1e9:.2f} GHz")
    print(f"  Pulse BW (-3dB) = {bw_hz/1e9:.3f} GHz")
    print(f"  Error         = {bw_error*100:.2f}%  (must be < 5%)")

    assert bw_error < 0.05, \
        f"Bandwidth error too large: {bw_error*100:.2f}% (expected < 5%)"

    print("  [PASS] Bandwidth matches target")

    # ------------------------------------------------------------------ #
    # Step 5: Check peak is at t0
    # ------------------------------------------------------------------ #
    t_arr = np.arange(n_steps) * grid.dt
    values = np.array([gaussian_pulse(source, t) for t in t_arr])
    t_peak = t_arr[np.argmax(values)]
    peak_error = abs(t_peak - source.t0) / source.t0

    print(f"\nPeak location check:")
    print(f"  t0 (set)      = {source.t0 * 1e12:.2f} ps")
    print(f"  t_peak (meas) = {t_peak * 1e12:.2f} ps")
    print(f"  Error         = {peak_error*100:.3f}%")

    assert peak_error < 0.01, \
        f"Peak not at t0: error = {peak_error*100:.3f}%"

    print("  [PASS] Peak located at t0")

    # ------------------------------------------------------------------ #
    # Step 6: Plot — also test a second source at 5 GHz for comparison
    # ------------------------------------------------------------------ #
    source_5ghz = make_source_for_fmax(5e9)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plt.suptitle('Test 01 — Gaussian Source Waveforms', fontsize=13)

    plot_source_waveform(source,       grid.dt, n_steps, ax=axes[0])
    axes[0].set_title(f'f_max = 10 GHz  |  t0 = {source.t0*1e12:.1f} ps  '
                      f'|  width = {source.width*1e12:.1f} ps')

    plot_source_waveform(source_5ghz,  grid.dt, n_steps, ax=axes[1])
    axes[1].set_title(f'f_max = 5 GHz  |  t0 = {source_5ghz.t0*1e12:.1f} ps  '
                      f'|  width = {source_5ghz.width*1e12:.1f} ps')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'test_01_output.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close('all')

    print(f"\n{'='*60}")
    print(f"TEST 01 PASSED ✓")
    print(f"Output saved to: {out_path}")
    print(f"{'='*60}")

    return True


if __name__ == '__main__':
    test_01_source_viz()
