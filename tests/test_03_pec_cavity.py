"""
test_03_pec_cavity.py — PEC rectangular cavity resonance.

Validates the PEC domain-wall boundary condition (apply_pec_faces) and the
ability of the solver to support undamped standing-wave eigenmodes. With CPML
removed and all four faces PEC, the 2D TMz cavity (fields Ez, Hx, Hy) is
lossless, so a broadband pulse rings forever as a superposition of cavity
eigenmodes. Their frequencies are read from the FFT of field monitors.

Setup:
    Grid:     Nx=100, Ny=80, Nz=1, dx=1 mm   (nominal 10 cm x 8 cm cavity)
    Material: vacuum
    Boundary: apply_pec_faces on x0,x1,y0,y1 — NO CPML (resonance must not decay)
    Source:   soft Ez injection at off-centre cell (23, 17, 0), short Gaussian
    Monitors: 3 FieldMonitors at off-node points; SnapshotMonitor every 50 steps
    Run:      10000 timesteps

Analytic TMz resonances of a 2D rectangular PEC cavity:
    f_mn = (c/2) * sqrt((m/a)^2 + (n/b)^2),   m,n >= 1

    IMPORTANT — effective cavity size (Yee-grid subtlety):
    apply_pec_faces zeros Ez on the node planes i=0, i=Nx-1 (and j=0, j=Ny-1).
    The standing wave therefore spans the distance *between those nodes*:
        a_eff = (Nx-1)*dx = 0.099 m   (not 0.100 m)
        b_eff = (Ny-1)*dy = 0.079 m   (not 0.080 m)
    The half-cell Ez stagger in y cancels in the node-to-node distance.
    Using the nominal 0.1 x 0.08 m injects a built-in ~1% bias; the effective
    dimensions are the physically correct cavity span and are used below.

Validation checks:
    1. At least 3 analytic modes are matched by a measured FFT peak within 1%.
    2. The three lowest-order modes (1,1),(2,1),(1,2) are all matched within 1%.

Pass criteria: both checks pass; PNG saved for visual inspection of the
standing-wave snapshots and the labelled spectrum.

Run (Windows console may need UTF-8):
    set PYTHONIOENCODING=utf-8
    python tests\test_03_pec_cavity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from fdtd.grid import create_grid
from fdtd.materials import set_vacuum
from fdtd.update import update_H, update_E
from fdtd.pec import apply_pec_faces, apply_pec_mask
from fdtd.sources import GaussianSource, gaussian_pulse
from fdtd.monitors import (FieldMonitor, SnapshotMonitor,
                           record_field, record_snapshot)
from fdtd.constants import C0


def _parabolic_refine(idx, x, y):
    """Sub-bin peak location via parabolic interpolation around index idx."""
    if idx <= 0 or idx >= len(y) - 1:
        return x[idx]
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if denom == 0.0:
        return x[idx]
    delta = 0.5 * (y0 - y2) / denom
    return x[idx] + delta * (x[1] - x[0])


def _mode_shape(snaps, times, f):
    """
    Extract a single mode's spatial pattern from the recorded snapshots via a
    temporal DFT at frequency f:  shape(x,y) = | sum_k Ez_k * exp(-i 2 pi f t_k) |.

    This isolates one eigenmode from the multi-mode superposition that any raw
    snapshot shows, yielding a clean standing-wave (m x n lobe) pattern.
    """
    stack = np.stack(snaps, axis=0)                       # (Nframes, Nx, Ny)
    phase = np.exp(-2j * np.pi * f * np.asarray(times))   # (Nframes,)
    proj  = np.tensordot(phase, stack, axes=(0, 0))       # (Nx, Ny) complex
    return np.abs(proj)


def run():
    print("=" * 60)
    print("TEST 03 — PEC Cavity Resonance")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    Nx, Ny, Nz = 100, 80, 1
    dx = 1e-3   # 1 mm

    grid = create_grid(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx)
    grid = set_vacuum(grid)
    # No CPML: a lossless cavity must not absorb its own standing waves.

    print(f"\nGrid: {Nx}x{Ny}x{Nz}, dx={dx*1e3:.1f} mm")
    print(f"Nominal cavity:   {Nx*dx*1e2:.1f} cm x {Ny*grid.dy*1e2:.1f} cm")

    # Effective cavity span = distance between the zeroed Ez node planes.
    a_eff = (Nx - 1) * dx
    b_eff = (Ny - 1) * grid.dy
    print(f"Effective cavity: {a_eff*1e2:.1f} cm x {b_eff*1e2:.1f} cm "
          f"(node-to-node)")
    print(f"dt = {grid.dt*1e12:.4f} ps")

    # Source — short broadband Gaussian, off-centre to excite many modes.
    f_max  = 7e9
    width  = 1.0 / (2.0 * np.pi * f_max)
    t0     = 4.0 * width
    source = GaussianSource(t0=t0, width=width)
    i_src, j_src = 23, 17
    print(f"Source: f_max={f_max/1e9:.0f} GHz, t0={t0*1e12:.1f} ps, "
          f"width={width*1e12:.1f} ps, at ({i_src}, {j_src}, 0)")

    # Monitors — 3 off-node points (avoid source cell and low-order nodes).
    mon_pts = [(61, 29), (47, 53), (73, 11)]
    field_mons = [FieldMonitor(component='Ez', i=p[0], j=p[1], k=0)
                  for p in mon_pts]
    snap_mon = SnapshotMonitor(component='Ez', k_slice=0, interval=50)
    print(f"Monitors at: {mon_pts}")

    # ------------------------------------------------------------------ #
    # Time loop  (CPML omitted; PEC faces enforced every step)
    # ------------------------------------------------------------------ #
    N_STEPS = 10000
    print(f"\nRunning {N_STEPS} timesteps...")

    for n in range(N_STEPS):
        t = n * grid.dt

        grid = update_H(grid)
        grid = update_E(grid)

        # Enforce PEC walls after every E update (no CPML correction here).
        grid = apply_pec_faces(grid, faces=('x0', 'x1', 'y0', 'y1'))
        grid = apply_pec_mask(grid)   # no-op — pec_mask is None

        grid.Ez[i_src, j_src, 0] += gaussian_pulse(source, t)

        for m in field_mons:
            record_field(m, grid)
        record_snapshot(snap_mon, grid)

        grid.time_step += 1

        if (n + 1) % 2000 == 0:
            print(f"  step {n+1}/{N_STEPS}")

    print("  Done.\n")

    # ------------------------------------------------------------------ #
    # Spectral analysis
    # ------------------------------------------------------------------ #
    # Skip the driven transient; after the pulse ends the signal is a pure
    # superposition of cavity eigenmodes.
    n_skip = 300
    Nfft   = 1 << 18   # heavy zero-pad -> ~1.6 MHz bin (sub-bin via parabola)

    freqs = np.fft.rfftfreq(Nfft, grid.dt)
    spectrum = np.zeros_like(freqs)
    for m in field_mons:
        sig = np.array(m.values[n_skip:])
        sig = sig - sig.mean()                  # drop DC
        sig = sig * np.hanning(len(sig))        # reduce spectral leakage
        F = np.fft.rfft(sig, n=Nfft)
        spectrum += np.abs(F) ** 2              # sum power over monitors

    # Analyse band of interest.
    band      = (freqs >= 1.5e9) & (freqs <= 5.3e9)
    f_band    = freqs[band]
    s_band    = spectrum[band]
    bin_hz    = freqs[1] - freqs[0]
    min_dist  = max(1, int(round(60e6 / bin_hz)))   # >=60 MHz between peaks

    pk_idx, _ = find_peaks(s_band,
                           height=s_band.max() * 0.01,
                           distance=min_dist)
    measured = np.array([_parabolic_refine(int(i), f_band, s_band)
                         for i in pk_idx])
    measured.sort()
    print(f"Measured FFT peaks (GHz): "
          f"{', '.join(f'{f/1e9:.3f}' for f in measured)}\n")

    # ------------------------------------------------------------------ #
    # Analytic mode table (effective dimensions)
    # ------------------------------------------------------------------ #
    modes = []
    for m_i in range(1, 5):
        for n_i in range(1, 5):
            f = 0.5 * C0 * np.sqrt((m_i / a_eff) ** 2 + (n_i / b_eff) ** 2)
            if f <= 5.2e9:
                modes.append((m_i, n_i, f))
    modes.sort(key=lambda x: x[2])

    print("--- Analytic vs measured (effective cavity dims) ---")
    print(f"  {'mode':>6} {'analytic':>11} {'measured':>11} {'error':>8}")
    matched = []          # (m,n,err%)
    for (m_i, n_i, f_a) in modes:
        if len(measured) == 0:
            break
        k = int(np.argmin(np.abs(measured - f_a)))
        f_m = measured[k]
        err = abs(f_m - f_a) / f_a * 100.0
        hit = err < 1.0
        if hit:
            matched.append((m_i, n_i, err))
        print(f"  ({m_i},{n_i})  {f_a/1e9:8.4f} GHz {f_m/1e9:8.4f} GHz "
              f"{err:6.2f}%  {'<--' if hit else ''}")

    n_matched = len(matched)
    print(f"\n  Modes matched within 1%: {n_matched}")

    # ------------------------------------------------------------------ #
    # Validation 1 — at least 3 modes matched within 1%
    # ------------------------------------------------------------------ #
    print("\n--- Validation 1: >= 3 resonances within 1% ---")
    print(f"  Matched {n_matched} modes  (must be >= 3)")
    assert n_matched >= 3, \
        f"Only {n_matched} modes matched within 1% (need >= 3)"
    print("  [PASS]\n")

    # ------------------------------------------------------------------ #
    # Validation 2 — the three lowest modes are all matched
    # ------------------------------------------------------------------ #
    print("--- Validation 2: lowest modes (1,1),(2,1),(1,2) within 1% ---")
    matched_set = {(m, n) for (m, n, _) in matched}
    for need in [(1, 1), (2, 1), (1, 2)]:
        ok = need in matched_set
        print(f"  mode {need}: {'matched' if ok else 'MISSING'}")
        assert ok, f"Fundamental mode {need} not matched within 1%"
    print("  [PASS]\n")

    # ------------------------------------------------------------------ #
    # Output plots
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Test 03 — PEC Cavity Resonance', fontsize=14)

    # (a) Field monitor time series
    ax1 = fig.add_subplot(2, 3, 1)
    for m, p in zip(field_mons, mon_pts):
        t_ns = np.array(m.times) * 1e9
        ax1.plot(t_ns, np.array(m.values), lw=0.6, label=f'{p}')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Ez (V/m)')
    ax1.set_title('Field monitors (undamped ringing)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # (b) Spectrum with analytic lines
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(f_band / 1e9, s_band / s_band.max(), lw=1.0, color='C0')
    for (m_i, n_i, f_a) in modes:
        ax2.axvline(f_a / 1e9, color='gray', ls='--', lw=0.8)
        ax2.text(f_a / 1e9, 1.02, f'{m_i}{n_i}', rotation=90,
                 fontsize=7, ha='center', va='bottom', color='gray')
    ax2.plot(measured / 1e9,
             np.interp(measured, f_band, s_band) / s_band.max(),
             'rv', ms=6, label='measured peak')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Power (norm.)')
    ax2.set_title('Spectrum vs analytic modes')
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # (c-f) Extracted mode shapes for the four lowest matched modes.
    # A raw snapshot is a superposition of ALL excited modes and looks like
    # interference noise; projecting the recorded snapshots onto a single
    # eigenfrequency recovers that mode's clean m x n standing-wave pattern.
    extent = [0, Nx * dx * 1e2, 0, Ny * grid.dy * 1e2]
    # Drop the driven-transient snapshots before projecting.
    settled = [(s, t) for s, t in zip(snap_mon.snapshots, snap_mon.snap_times)
               if t >= n_skip * grid.dt]
    sm_snaps = [s for s, _ in settled]
    sm_times = [t for _, t in settled]

    plot_modes = [(1, 1), (2, 1), (1, 2), (2, 2)]
    for ax_pos, (m_i, n_i) in zip([3, 4, 5, 6], plot_modes):
        f_a = 0.5 * C0 * np.sqrt((m_i / a_eff) ** 2 + (n_i / b_eff) ** 2)
        f_use = measured[int(np.argmin(np.abs(measured - f_a)))] \
            if len(measured) else f_a
        shape = _mode_shape(sm_snaps, sm_times, f_use)
        vmax  = max(shape.max(), 1e-30)
        ax = fig.add_subplot(2, 3, ax_pos)
        im = ax.imshow(shape.T, origin='lower', extent=extent,
                       cmap='inferno', aspect='equal', vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Mode ({m_i},{n_i})  |Ez| @ {f_use/1e9:.3f} GHz')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'test_03_output.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close('all')

    # ------------------------------------------------------------------ #
    # Animation — raw Ez field over all snapshots (multi-mode interference)
    # ------------------------------------------------------------------ #
    print("Saving animation...")
    import matplotlib.animation as animation

    snaps = snap_mon.snapshots
    times = snap_mon.snap_times
    # Fixed global colour scale (settled portion) so the standing-wave
    # sloshing is visible without the early transient saturating it.
    vmax = max((np.max(np.abs(s)) for s in sm_snaps), default=1e-30)
    vmax = max(vmax, 1e-30)

    fig_anim, ax_anim = plt.subplots(figsize=(6, 5))
    im = ax_anim.imshow(snaps[0].T, origin='lower', extent=extent,
                        cmap='RdBu_r', aspect='equal',
                        vmin=-vmax, vmax=vmax, animated=True)
    plt.colorbar(im, ax=ax_anim, label='Ez (V/m)', pad=0.02)
    ax_anim.plot(i_src * dx * 1e2, j_src * grid.dy * 1e2, 'k+', ms=9, mew=2)
    for p in mon_pts:
        ax_anim.plot(p[0] * dx * 1e2, p[1] * grid.dy * 1e2, 'gx', ms=5, mew=1.2)
    ax_anim.set_xlabel('x (cm)')
    ax_anim.set_ylabel('y (cm)')
    title = ax_anim.set_title('')

    def _update(frame):
        im.set_data(snaps[frame].T)
        title.set_text(f'Ez  t = {times[frame]*1e9:.2f} ns  '
                       f'(frame {frame+1}/{len(snaps)})')
        return im, title

    anim = animation.FuncAnimation(
        fig_anim, _update, frames=len(snaps), interval=50, blit=True
    )
    gif_path = os.path.join(os.path.dirname(__file__), 'test_03_animation.gif')
    anim.save(gif_path, writer='pillow', fps=20)
    plt.close(fig_anim)
    print(f"Animation saved to: {gif_path}")

    print("=" * 60)
    print("TEST 03 PASSED")
    print(f"Output saved to: {out_path}")
    print("=" * 60)


if __name__ == '__main__':
    run()
