"""
test_04_waveguide.py — Rectangular waveguide dominant-mode dispersion.

Validates guided-wave propagation: a parallel-PEC-wall waveguide supports a
dominant mode with a half-sine (n=1) transverse profile and a cutoff
frequency below which the field is evanescent (no propagation) and above
which it propagates with the waveguide phase velocity.

Geometry (2D TMz: fields Ez, Hx, Hy; propagation in x, cross-section in y):
    Grid:     Nx=200, Ny=50, Nz=1, dx=0.5 mm   (10 cm long, 25 mm wide)
    Material: vacuum
    Boundary: apply_pec_faces on y0,y1 (waveguide walls);
              CPML on x0,x1 ONLY (y-CPML neutralised — see note below)
    Source:   soft Ez line injection at x=20, all j (excites the n=1 mode);
              narrowband Gaussian-modulated sinusoid at a chosen carrier f0
    Run:      3000 timesteps per carrier

    Effective width (Yee subtlety, as in Test 03): apply_pec_faces zeros Ez at
    node planes j=0 and j=Ny-1, so the mode spans b_eff=(Ny-1)*dy=24.5 mm, not
    25 mm. Dominant-mode cutoff f_c = c/(2*b_eff) ~ 6.12 GHz (not 6.0 GHz).

Dispersion (n=1 mode):
    beta(f) = (2 pi / c) * sqrt(f^2 - f_c^2)            (f > f_c, propagating)
    v_ph(f) = c / sqrt(1 - (f_c/f)^2)                   (phase velocity)
    alpha(f) = (2 pi / c) * sqrt(f_c^2 - f^2)           (f < f_c, evanescent)

Validation checks:
    1. Below cutoff (f0=4 GHz): field decays exponentially in x; measured
       decay constant alpha within 15% of theory; far/near amplitude < 5%.
    2. Above cutoff (f0=9 GHz): measured phase velocity within 2% of theory.
    3. Above cutoff: transverse |Ez(y)| profile matches the half-sine n=1
       mode shape (correlation > 0.99).

Pass criteria: all three checks pass. PNG + side-by-side GIF saved.

Run (Windows console may need UTF-8):
    set PYTHONIOENCODING=utf-8
    python tests\test_04_waveguide.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavesim.grid import create_grid
from wavesim.materials import set_vacuum
from wavesim.update import update_H, update_E
from wavesim.pml import init_cpml, update_H_pml, update_E_pml
from wavesim.pec import apply_pec_faces, apply_pec_mask
from wavesim.monitors import FieldMonitor, SnapshotMonitor, record_field, record_snapshot
from wavesim.constants import C0

# ---------------------------------------------------------------------- #
# Geometry / problem constants
# ---------------------------------------------------------------------- #
NX, NY, NZ = 200, 50, 1
DX = 0.5e-3
I_SRC = 20
N_STEPS = 3000
SNAP_INTERVAL = 20

# Effective width and dominant-mode cutoff (node-to-node span).
B_EFF = (NY - 1) * DX
F_C = C0 / (2.0 * B_EFF)


def _modulated_gaussian(t, f0, t0, tau):
    """Narrowband source: sinusoid at f0 under a Gaussian envelope."""
    return np.sin(2.0 * np.pi * f0 * (t - t0)) * np.exp(-0.5 * ((t - t0) / tau) ** 2)


def simulate(f0, mon_xs=()):
    """
    Run the waveguide with a narrowband carrier at f0.

    Returns
    -------
    snaps, snap_times : list of (Nx,Ny) Ez slices and their times
    mons              : list of FieldMonitor at the requested centreline x's
    """
    grid = create_grid(Nx=NX, Ny=NY, Nz=NZ, dx=DX)
    grid = set_vacuum(grid)
    # Waveguide: y-faces are PEC walls, not absorbers — CPML on the x-faces only.
    cpml = init_cpml(grid, d_pml=10, faces=('x0', 'x1'))

    # Narrowband pulse: ~6 carrier cycles under the envelope.
    tau = 6.0 / f0
    t0  = 3.5 * tau

    j_c  = NY // 2
    mons = [FieldMonitor(component='Ez', i=x, j=j_c, k=0) for x in mon_xs]
    snap = SnapshotMonitor(component='Ez', k_slice=0, interval=SNAP_INTERVAL)

    for n in range(N_STEPS):
        t = n * grid.dt

        grid = update_H(grid)
        grid, cpml = update_H_pml(grid, cpml)
        grid = update_E(grid)
        grid, cpml = update_E_pml(grid, cpml)

        grid = apply_pec_faces(grid, faces=('y0', 'y1'))   # waveguide walls
        grid = apply_pec_mask(grid)                         # no-op

        grid.Ez[I_SRC, :, 0] += _modulated_gaussian(t, f0, t0, tau)

        for m in mons:
            record_field(m, grid)
        record_snapshot(snap, grid)
        grid.time_step += 1

    return snap.snapshots, snap.snap_times, mons, grid.dt


def _envelope_x(snaps, j):
    """Peak |Ez| over time along x at transverse index j (the field envelope)."""
    stack = np.stack(snaps, axis=0)        # (Nframes, Nx, Ny)
    return np.max(np.abs(stack[:, :, j]), axis=0)


def run():
    print("=" * 60)
    print("TEST 04 — Rectangular Waveguide Dominant Mode")
    print("=" * 60)
    print(f"\nGrid: {NX}x{NY}x{NZ}, dx={DX*1e3:.2f} mm "
          f"({NX*DX*1e2:.1f} cm long, {NY*DX*1e3:.1f} mm wide)")
    print(f"Effective width b_eff = {B_EFF*1e3:.2f} mm  ->  "
          f"f_c = {F_C/1e9:.3f} GHz")

    j_c = NY // 2

    # ------------------------------------------------------------------ #
    # Below-cutoff run (evanescent)
    # ------------------------------------------------------------------ #
    f_below = 4.0e9
    print(f"\n[1/2] Below cutoff: f0 = {f_below/1e9:.1f} GHz "
          f"(< {F_C/1e9:.2f} GHz)  -> expect evanescent decay")
    snaps_b, times_b, _, dt = simulate(f_below)

    env_b = _envelope_x(snaps_b, j_c)
    # Fit ln(env) vs x in a clean window (clear of source and right PML).
    x_lo, x_hi = 30, 95
    xs = np.arange(x_lo, x_hi)
    logy = np.log(env_b[x_lo:x_hi] + 1e-30)
    slope = np.polyfit(xs, logy, 1)[0]            # per cell
    alpha_meas = -slope / DX                      # per metre
    alpha_th = (2.0 * np.pi / C0) * np.sqrt(F_C**2 - f_below**2)
    alpha_err = abs(alpha_meas - alpha_th) / alpha_th * 100.0
    far_near = env_b[160] / (env_b[40] + 1e-30)

    print(f"  alpha measured = {alpha_meas:7.1f} /m  "
          f"(decay length {1e3/alpha_meas:.2f} mm)")
    print(f"  alpha theory   = {alpha_th:7.1f} /m  (error {alpha_err:.1f}%)")
    print(f"  far/near amplitude ratio (x=160 vs x=40): {far_near:.2e}")

    # ------------------------------------------------------------------ #
    # Above-cutoff run (propagating)
    # ------------------------------------------------------------------ #
    f_above = 9.0e9
    x_a, x_b = 70, 110                              # dx_ab < half guide-wavelength
    print(f"\n[2/2] Above cutoff: f0 = {f_above/1e9:.1f} GHz "
          f"(> {F_C/1e9:.2f} GHz)  -> expect propagation")
    snaps_a, times_a, mons, dt = simulate(f_above, mon_xs=(x_a, x_b))

    # Phase velocity from the cross-spectrum phase at f0 (two centreline taps).
    va = np.array(mons[0].values)
    vb = np.array(mons[1].values)
    w  = np.hanning(len(va))
    Nfft = 1 << 15
    Fa = np.fft.rfft(va * w, n=Nfft)
    Fb = np.fft.rfft(vb * w, n=Nfft)
    freqs = np.fft.rfftfreq(Nfft, dt)
    k = int(np.argmin(np.abs(freqs - f_above)))
    dphi = np.angle(Fb[k] / Fa[k])                 # = -beta * dx_ab (wrapped)
    dx_ab = (x_b - x_a) * DX
    beta_meas = -dphi / dx_ab
    if beta_meas < 0:                              # unwrap guard
        beta_meas += 2.0 * np.pi / dx_ab
    vph_meas = 2.0 * np.pi * f_above / beta_meas
    vph_th = C0 / np.sqrt(1.0 - (F_C / f_above) ** 2)
    vph_err = abs(vph_meas - vph_th) / vph_th * 100.0

    print(f"  beta measured = {beta_meas:7.2f} rad/m")
    print(f"  v_ph measured = {vph_meas/1e8:.4f} x1e8 m/s")
    print(f"  v_ph theory   = {vph_th/1e8:.4f} x1e8 m/s  (error {vph_err:.2f}%)")

    # Above-cutoff envelope (propagating: should NOT decay strongly).
    env_a = _envelope_x(snaps_a, j_c)
    prop_ratio = env_a[160] / (env_a[60] + 1e-30)
    print(f"  far/near amplitude ratio (x=160 vs x=60): {prop_ratio:.3f}")

    # Transverse profile at a probe far from the source (pure n=1; higher
    # modes are evanescent at 9 GHz and have decayed away).
    x_probe = 150
    stack_a = np.stack(snaps_a, axis=0)
    env_y = np.max(np.abs(stack_a[:, x_probe, :]), axis=0)
    j = np.arange(NY)
    model = np.sin(np.pi * j / (NY - 1))           # half-sine, zero at walls
    en = env_y / (env_y.max() + 1e-30)
    mn = model / model.max()
    corr = float(np.corrcoef(en, mn)[0, 1])
    print(f"  transverse |Ez(y)| correlation with half-sine: {corr:.4f}")

    # ------------------------------------------------------------------ #
    # Validations
    # ------------------------------------------------------------------ #
    print("\n--- Validation 1: below-cutoff evanescent decay ---")
    print(f"  alpha error {alpha_err:.1f}% (must be < 15%), "
          f"far/near {far_near:.2e} (must be < 0.05)")
    assert alpha_err < 15.0, f"alpha error {alpha_err:.1f}% too large"
    assert far_near < 0.05, f"field did not decay (ratio {far_near:.2e})"
    print("  [PASS]\n")

    print("--- Validation 2: above-cutoff phase velocity ---")
    print(f"  v_ph error {vph_err:.2f}% (must be < 2%), "
          f"propagation far/near {prop_ratio:.3f} (must be > 0.5)")
    assert vph_err < 2.0, f"phase velocity error {vph_err:.2f}% too large"
    assert prop_ratio > 0.5, f"wave did not propagate (ratio {prop_ratio:.3f})"
    print("  [PASS]\n")

    print("--- Validation 3: TE10 half-sine transverse profile ---")
    print(f"  correlation {corr:.4f} (must be > 0.99)")
    assert corr > 0.99, f"transverse profile correlation {corr:.4f} too low"
    print("  [PASS]\n")

    # ------------------------------------------------------------------ #
    # Static summary figure
    # ------------------------------------------------------------------ #
    x_cm = np.arange(NX) * DX * 1e2
    extent = [0, NX * DX * 1e2, 0, NY * DX * 1e3]   # x in cm, y in mm
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Test 04 — Rectangular Waveguide Dominant Mode', fontsize=14)

    # (a) Below-cutoff envelope (log) with fit + theory
    ax = fig.add_subplot(2, 3, 1)
    ax.semilogy(x_cm, env_b / env_b.max(), 'C0', lw=1.2, label='|Ez| envelope')
    fit_line = np.exp(np.polyval(np.polyfit(xs, np.log(env_b[x_lo:x_hi] + 1e-30), 1),
                                 np.arange(NX))) / env_b.max()
    ax.semilogy(x_cm[x_lo:x_hi], fit_line[x_lo:x_hi], 'r--', lw=1.5, label='exp fit')
    ax.axvline(I_SRC * DX * 1e2, color='gray', ls=':', lw=1, label='source')
    ax.set_xlabel('x (cm)'); ax.set_ylabel('|Ez| (norm.)')
    ax.set_title(f'Below cutoff {f_below/1e9:.0f} GHz — evanescent')
    ax.set_ylim(1e-4, 2); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (b) Above-cutoff envelope (linear) — propagation
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(x_cm, env_a / env_a.max(), 'C1', lw=1.2)
    ax.axvline(I_SRC * DX * 1e2, color='gray', ls=':', lw=1)
    ax.set_xlabel('x (cm)'); ax.set_ylabel('|Ez| (norm.)')
    ax.set_title(f'Above cutoff {f_above/1e9:.0f} GHz — propagating')
    ax.grid(True, alpha=0.3)

    # (c) Transverse profile vs half-sine
    ax = fig.add_subplot(2, 3, 3)
    y_mm = j * DX * 1e3
    ax.plot(y_mm, en, 'C1o', ms=3, label='measured')
    ax.plot(y_mm, mn, 'k--', lw=1.2, label='sin(pi y / b)')
    ax.set_xlabel('y (mm)'); ax.set_ylabel('|Ez| (norm.)')
    ax.set_title(f'Transverse profile (x={x_probe}), corr={corr:.4f}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (d, e) Representative snapshots near each peak
    def _peak_frame(snaps):
        return int(np.argmax([np.max(np.abs(s)) for s in snaps]))
    for ax_pos, snaps, ttl, f0 in [
            (4, snaps_b, 'Below cutoff', f_below),
            (5, snaps_a, 'Above cutoff', f_above)]:
        ax = fig.add_subplot(2, 3, ax_pos)
        fr = _peak_frame(snaps)
        s = snaps[fr]
        vmax = max(np.max(np.abs(s)), 1e-30)
        im = ax.imshow(s.T, origin='lower', extent=extent, cmap='RdBu_r',
                       aspect='auto', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        ax.axvline(I_SRC * DX * 1e2, color='k', ls=':', lw=1)
        ax.set_xlabel('x (cm)'); ax.set_ylabel('y (mm)')
        ax.set_title(f'{ttl} {f0/1e9:.0f} GHz  Ez')

    # (f) dispersion diagram with the two operating points
    ax = fig.add_subplot(2, 3, 6)
    ff = np.linspace(F_C * 1.001, 14e9, 300)
    ax.plot(ff / 1e9, C0 / np.sqrt(1 - (F_C / ff) ** 2) / 1e8, 'k', lw=1.2,
            label='v_ph theory')
    ax.plot(f_above / 1e9, vph_meas / 1e8, 'C1*', ms=14, label='measured')
    ax.axvline(F_C / 1e9, color='gray', ls='--', lw=1, label='cutoff')
    ax.axhline(C0 / 1e8, color='C2', ls=':', lw=1, label='c')
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('v_ph (x1e8 m/s)')
    ax.set_title('Phase velocity dispersion')
    ax.set_ylim(0, 12); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'test_04_output.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close('all')

    # ------------------------------------------------------------------ #
    # Side-by-side animation: evanescent (left) vs propagating (right)
    # ------------------------------------------------------------------ #
    print("Saving animation...")
    import matplotlib.animation as animation

    nframes = min(len(snaps_b), len(snaps_a))
    vmax_b = max((np.max(np.abs(s)) for s in snaps_b), default=1e-30)
    vmax_a = max((np.max(np.abs(s)) for s in snaps_a), default=1e-30)

    fig_a, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))
    imL = axL.imshow(snaps_b[0].T, origin='lower', extent=extent, cmap='RdBu_r',
                     aspect='auto', vmin=-vmax_b, vmax=vmax_b, animated=True)
    imR = axR.imshow(snaps_a[0].T, origin='lower', extent=extent, cmap='RdBu_r',
                     aspect='auto', vmin=-vmax_a, vmax=vmax_a, animated=True)
    for ax, ttl in [(axL, f'Below cutoff {f_below/1e9:.0f} GHz (evanescent)'),
                    (axR, f'Above cutoff {f_above/1e9:.0f} GHz (propagating)')]:
        ax.axvline(I_SRC * DX * 1e2, color='k', ls=':', lw=1)
        ax.set_xlabel('x (cm)'); ax.set_ylabel('y (mm)'); ax.set_title(ttl)
    plt.colorbar(imL, ax=axL, pad=0.02, fraction=0.046)
    plt.colorbar(imR, ax=axR, pad=0.02, fraction=0.046)
    sup = fig_a.suptitle('')

    def _update(fr):
        imL.set_data(snaps_b[fr].T)
        imR.set_data(snaps_a[fr].T)
        sup.set_text(f't = {times_a[fr]*1e9:.3f} ns  (frame {fr+1}/{nframes})')
        return imL, imR, sup

    anim = animation.FuncAnimation(fig_a, _update, frames=nframes,
                                   interval=60, blit=True)
    gif_path = os.path.join(os.path.dirname(__file__), 'test_04_animation.gif')
    anim.save(gif_path, writer='pillow', fps=18)
    plt.close(fig_a)
    print(f"Animation saved to: {gif_path}")

    print("=" * 60)
    print("TEST 04 PASSED")
    print(f"Output saved to: {out_path}")
    print("=" * 60)


if __name__ == '__main__':
    run()
