"""
test_06_box_cavity_3d.py — Rectangular PEC cavity (genuinely volumetric 3D).

Test 05 (coax) runs full 3D code paths, but its TEM mode is a transverse field
*extruded* along z: the mode shape does not vary with z. This test closes that
gap with a mode that varies in ALL THREE axes against an analytic ground truth —
the rectangular PEC cavity, the 3D analogue of Test 03.

A box of perfectly conducting walls is a lossless resonator: a broadband pulse
rings forever as a superposition of cavity eigenmodes whose frequencies are

    f_mnp = (c/2) * sqrt( (m/a)^2 + (n/b)^2 + (p/d)^2 )

for both TE and TM families (m,n,p integers, at least two of them >= 1). Modes
with p >= 1 carry a half-integer standing wave ALONG z — the quantity Test 05
could not test. We read the resonances from the FFT of field monitors and match
them to the analytic table, then require that several matched modes have p >= 1.

Effective cavity size (same Yee-grid subtlety as Test 03):
    apply_pec_faces zeros the tangential E nodes on the integer boundary planes
    (i=0,Nx-1; j=0,Ny-1; k=0,Nz-1), so each standing wave spans node-to-node:
        a = (Nx-1)*dx,  b = (Ny-1)*dy,  d = (Nz-1)*dz.

Setup:
    Grid:     Nx=48, Ny=44, Nz=40, dx=dy=dz=2 mm   (distinct dims -> well
              separated, non-degenerate modes)
    Material: vacuum
    Boundary: apply_pec_faces on all SIX faces — NO CPML (must not decay)
    Source:   soft point pulse on Ex, Ey AND Ez at one off-node cell, to excite
              both TE (Ez=0) and TM (Ez!=0) families
    Monitors: Ex/Ey/Ez at 3 off-node points (power summed for the spectrum);
              full 3D Ez snapshots for the temporal-DFT mode-shape extraction
    Run:      5000 timesteps

Validation checks:
    1. At least 5 analytic modes matched by a measured FFT peak within 1.5%.
    2. At least 2 genuinely 3D modes (p >= 1) matched, INCLUDING the lowest
       z-varying mode — i.e. the z-axis carries real standing-wave physics.

Pass criteria: both checks pass; a multi-panel PNG (spectrum + 3D mode-shape
triptych via wavesim.viz.plot_field_slices_3d) and an animation (XZ + XY
sloshing via wavesim.viz.animate_field_slices_3d) are saved.

Run (Windows console may need UTF-8):
    set PYTHONIOENCODING=utf-8
    python tests\test_06_box_cavity_3d.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from wavesim.grid import create_grid
from wavesim.materials import set_vacuum
from wavesim.update import update_H, update_E
from wavesim.pec import apply_pec_faces, apply_pec_mask
from wavesim.sources import GaussianSource, gaussian_pulse
from wavesim.monitors import FieldMonitor, record_field
from wavesim.constants import C0
from wavesim import viz

# ---------------------------------------------------------------------- #
# Geometry / run constants
# ---------------------------------------------------------------------- #
NX, NY, NZ = 48, 44, 40
DX = 2.0e-3                       # uniform cubic cells

F_MAX = 6.0e9                    # source bandwidth
N_STEPS = 5000
SNAP_INTERVAL = 20               # 3D-Ez snapshot cadence (Nyquist ~6.5 GHz)
N_SKIP = 300                     # drop the driven transient before the FFT

BAND = (2.0e9, 5.0e9)            # spectral band analysed
TOL = 1.5                        # mode-match tolerance (%)

SRC_CELL = (13, 11, 9)           # off-node, asymmetric source cell
MON_PTS = [(31, 17, 23), (19, 29, 13), (37, 9, 27)]


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


def _analytic_modes(a, b, d, f_lo, f_hi):
    """
    Cavity resonances in [f_lo, f_hi]. A valid TE/TM mode needs at least two of
    (m,n,p) >= 1; combos with two zeros are not resonances. Frequencies that
    coincide (degenerate) are kept once, tagged with all contributing indices.
    """
    out = {}
    for m in range(0, 5):
        for n in range(0, 5):
            for p in range(0, 5):
                if (m > 0) + (n > 0) + (p > 0) < 2:
                    continue
                f = 0.5 * C0 * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)
                if f_lo <= f <= f_hi:
                    key = round(f / 1e6)        # 1 MHz dedupe key
                    out.setdefault(key, (f, []))[1].append((m, n, p))
    modes = [(f, idx) for f, idx in out.values()]
    modes.sort(key=lambda x: x[0])
    return modes


def simulate():
    """Run the lossless 3D PEC cavity; return monitors + 3D Ez snapshots."""
    grid = create_grid(Nx=NX, Ny=NY, Nz=NZ, dx=DX)
    grid = set_vacuum(grid)
    # Lossless cavity: every face is PEC, NO CPML (resonances must not decay).

    source = GaussianSource(t0=4.0 / (2 * np.pi * F_MAX),
                            width=1.0 / (2 * np.pi * F_MAX), amplitude=1.0)
    si, sj, sk = SRC_CELL

    # Record all three E components so both TE and TM families are seen.
    mons = []
    for (i, j, k) in MON_PTS:
        for comp in ('Ex', 'Ey', 'Ez'):
            mons.append(FieldMonitor(comp, i, j, k))

    ez_snaps, snap_times = [], []

    for n in range(N_STEPS):
        t = n * grid.dt

        grid = update_H(grid)
        grid = update_E(grid)

        # PEC on all six walls after every E update (no CPML correction here).
        grid = apply_pec_faces(grid, faces=('x0', 'x1', 'y0', 'y1', 'z0', 'z1'))
        grid = apply_pec_mask(grid)          # no-op (no interior conductor)

        amp = gaussian_pulse(source, t)
        grid.Ex[si, sj, sk] += amp
        grid.Ey[si, sj, sk] += amp
        grid.Ez[si, sj, sk] += amp

        for mm in mons:
            record_field(mm, grid)

        if grid.time_step % SNAP_INTERVAL == 0:
            ez_snaps.append(grid.Ez.copy())
            snap_times.append(grid.time_step * grid.dt)

        grid.time_step += 1

    return dict(grid=grid, dt=grid.dt, mons=mons,
                ez_snaps=ez_snaps, snap_times=snap_times)


def _settled(res):
    """Snapshots/times past the driven transient (t >= N_SKIP*dt)."""
    t_cut = N_SKIP * res['dt']
    pairs = [(s, t) for s, t in zip(res['ez_snaps'], res['snap_times'])
             if t >= t_cut]
    return [s for s, _ in pairs], [t for _, t in pairs]


def _mode_shape_3d(snaps, times, f):
    """3D |Ez| pattern at frequency f via a temporal DFT of the snapshots.

    Per-cell mean subtraction removes the static Ez bias the soft source leaves
    at its cell (a Gaussian pulse injects a net-positive offset that the finite
    record would otherwise leak into the projection); a Hann window suppresses
    spectral leakage from neighbouring modes.
    """
    stack = np.stack(snaps, axis=0)                       # (Nf, Nx, Ny, Nz)
    stack = stack - stack.mean(axis=0, keepdims=True)     # drop per-cell DC
    w = np.hanning(len(snaps))                            # (Nf,)
    phase = np.exp(-2j * np.pi * f * np.asarray(times)) * w
    proj = np.tensordot(phase, stack, axes=(0, 0))        # (Nx, Ny, Nz) complex
    return np.abs(proj)


def run():
    print("=" * 60)
    print("TEST 06 — Rectangular PEC Cavity (volumetric 3D)")
    print("=" * 60)

    a = (NX - 1) * DX
    b = (NY - 1) * DX
    d = (NZ - 1) * DX
    print(f"\nGrid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm")
    print(f"Effective cavity (node-to-node): "
          f"{a*1e3:.1f} x {b*1e3:.1f} x {d*1e3:.1f} mm")

    res = simulate()
    dt = res['dt']
    print(f"dt = {dt*1e12:.4f} ps,  {N_STEPS} steps, "
          f"record length {N_STEPS*dt*1e9:.1f} ns")

    # ------------------------------------------------------------------ #
    # Spectrum — sum |FFT|^2 over every monitor/component (post-transient)
    # ------------------------------------------------------------------ #
    Nfft = 1 << 18
    freqs = np.fft.rfftfreq(Nfft, dt)
    spectrum = np.zeros_like(freqs)
    for mm in res['mons']:
        sig = np.array(mm.values[N_SKIP:])
        if not np.any(sig):
            continue
        sig = sig - sig.mean()
        sig = sig * np.hanning(len(sig))
        F = np.fft.rfft(sig, n=Nfft)
        spectrum += np.abs(F) ** 2

    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    f_band, s_band = freqs[band], spectrum[band]
    bin_hz = freqs[1] - freqs[0]
    min_dist = max(1, int(round(40e6 / bin_hz)))
    pk_idx, _ = find_peaks(s_band, height=s_band.max() * 0.02, distance=min_dist)
    measured = np.array(sorted(_parabolic_refine(int(i), f_band, s_band)
                               for i in pk_idx))
    print(f"\nMeasured FFT peaks (GHz): "
          f"{', '.join(f'{f/1e9:.3f}' for f in measured)}")

    # ------------------------------------------------------------------ #
    # Match against the analytic mode table
    # ------------------------------------------------------------------ #
    modes = _analytic_modes(a, b, d, *BAND)
    print(f"\n--- Analytic vs measured (tol {TOL}%) ---")
    print(f"  {'f_analytic':>11}  {'modes(mnp)':>16}  {'f_meas':>9}  {'err':>6}")
    matched = []                          # (f, indices, err%)
    for (f_a, idxs) in modes:
        if len(measured) == 0:
            break
        k = int(np.argmin(np.abs(measured - f_a)))
        f_m = measured[k]
        err = abs(f_m - f_a) / f_a * 100.0
        hit = err < TOL
        tag = '/'.join(f'{m}{n}{p}' for (m, n, p) in idxs)
        if hit:
            matched.append((f_a, idxs, err))
        print(f"  {f_a/1e9:8.4f}GHz  {tag:>16}  {f_m/1e9:6.4f}GHz "
              f"{err:5.2f}%  {'<--' if hit else ''}")

    # A mode is "3D" (z-varying) if every contributing index set has p >= 1.
    def _is_3d(idxs):
        return all(p >= 1 for (_, _, p) in idxs)
    matched_3d = [mt for mt in matched if _is_3d(mt[1])]
    n_matched = len(matched)
    n_3d = len(matched_3d)
    print(f"\n  Modes matched within {TOL}%: {n_matched}  "
          f"(of which z-varying p>=1: {n_3d})")

    # Lowest z-varying analytic mode in band (must be among the matches).
    lowest_3d_f = next((f for (f, idxs) in modes if _is_3d(idxs)), None)
    lowest_3d_matched = any(abs(mt[0] - lowest_3d_f) < 1.0 for mt in matched_3d) \
        if lowest_3d_f is not None else False

    # ------------------------------------------------------------------ #
    # Assertions
    # ------------------------------------------------------------------ #
    print("\n" + "-" * 60)
    print("--- Check 1: >= 5 resonances within tol ---")
    assert n_matched >= 5, f"only {n_matched} modes matched (need >= 5)"
    print(f"  matched {n_matched}  [PASS]")
    print("--- Check 2: >= 2 z-varying (p>=1) modes, incl. the lowest ---")
    assert n_3d >= 2, f"only {n_3d} z-varying modes matched (need >= 2)"
    assert lowest_3d_matched, \
        f"lowest z-varying mode ({lowest_3d_f/1e9:.3f} GHz) not matched"
    print(f"  matched {n_3d} z-varying modes; lowest "
          f"({lowest_3d_f/1e9:.3f} GHz) present  [PASS]")
    print("-" * 60)

    _make_figure(res, f_band, s_band, modes, measured, matched_3d,
                 lowest_3d_f, a, b, d)
    _make_animation(res)

    print("\n" + "=" * 60)
    print("TEST 06 PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------- #
# Plotting (uses the new 3D viz helpers)
# ---------------------------------------------------------------------- #
def _make_figure(res, f_band, s_band, modes, measured, matched_3d,
                 lowest_3d_f, a, b, d):
    grid = res['grid']

    # Spectrum panel on its own figure row.
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle('Test 06 — Rectangular PEC Cavity (volumetric 3D)', fontsize=14)

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(f_band / 1e9, s_band / s_band.max(), lw=1.0, color='C0')
    for (f_a, idxs) in modes:
        is3d = all(p >= 1 for (_, _, p) in idxs)
        ax.axvline(f_a / 1e9, color=('C3' if is3d else 'gray'),
                   ls='--', lw=0.8, alpha=0.7)
        ax.text(f_a / 1e9, 1.02, '/'.join(f'{m}{n}{p}' for (m, n, p) in idxs),
                rotation=90, fontsize=6, ha='center', va='bottom',
                color=('C3' if is3d else 'gray'))
    ax.plot(measured / 1e9, np.interp(measured, f_band, s_band) / s_band.max(),
            'kv', ms=6, label='measured peak')
    ax.plot([], [], color='C3', ls='--', label='z-varying mode (p>=1)')
    ax.plot([], [], color='gray', ls='--', label='z-invariant mode (p=0)')
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('Power (norm.)')
    ax.set_title('Cavity spectrum vs analytic modes '
                 '(red = genuinely 3D, p>=1)')
    ax.set_ylim(0, 1.18); ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3D mode-shape triptych, drawn into the bottom row via the shared helper.
    # We visualise |Ez|, so we must pick a mode that HAS Ez: that means a TM-type
    # mode with m,n >= 1 (TE modes have Ez = 0), and p >= 1 for a half-wave along
    # z — i.e. lobes in all three axes. (The lowest z-varying mode overall is
    # TE101, Ez = 0, which would render blank.)
    def _has_ez_3d(idxs):
        return any(m >= 1 and n >= 1 and p >= 1 for (m, n, p) in idxs)
    ez_modes = [mt for mt in matched_3d if _has_ez_3d(mt[1])]
    # Among the Ez-carrying 3D modes, pick the one whose peak is most ISOLATED
    # from every other analytic mode. Projecting at a blended peak detunes the
    # target by tens of MHz and the DFT integral cancels it (-> noise). Isolation
    # guarantees f_use lands on a single clean eigenmode.
    all_f = np.array([f for (f, _) in modes])
    def _isolation(f):
        return float(np.min(np.abs(all_f[all_f != f] - f))) if len(all_f) > 1 else 1e9
    f_target = (max((mt[0] for mt in ez_modes), key=_isolation)
                if ez_modes else lowest_3d_f)
    f_use = measured[int(np.argmin(np.abs(measured - f_target)))]
    # Project only the SETTLED snapshots: during the pulse the source cell
    # dominates Ez and would swamp the eigenmode (cf. Test 03).
    s_snaps, s_times = _settled(res)
    shape3d = _mode_shape_3d(s_snaps, s_times, f_use)
    # Cut through the global antinode so no panel lands on a node plane.
    pi, pj, pk = np.unravel_index(int(np.argmax(shape3d)), shape3d.shape)
    ax_xy = fig.add_subplot(2, 3, 4)
    ax_xz = fig.add_subplot(2, 3, 5)
    ax_yz = fig.add_subplot(2, 3, 6)
    viz.plot_field_slices_3d(
        shape3d, grid,
        component=f'|Ez| @ {f_use/1e9:.3f} GHz (TM mode, m,n,p>=1)',
        i=int(pi), j=int(pj), k=int(pk), axes=(ax_xy, ax_xz, ax_yz))

    out = os.path.join(os.path.dirname(__file__), 'test_06_output.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"\nFigure saved to: {out}")


def _make_animation(res):
    print("Saving animation...")
    grid = res['grid']
    mm = 1e3
    jc, kc = NY // 2, NZ // 2
    # Drop the driven transient so the standing-wave sloshing sets the colour
    # scale (the initial pulse at the source is far brighter).
    snaps, times = _settled(res)

    # Two orthogonal cuts of the raw Ez field as the standing wave sloshes.
    xz_frames = [s[:, jc, :] for s in snaps]                 # (Nx, Nz) -> z horiz
    xy_frames = [s[:, :, kc].T for s in snaps]               # (Ny, Nx) for imshow

    panels = [
        dict(frames=xz_frames,
             extent=[0, NZ * DX * mm, 0, NX * DX * mm],
             xlabel='z (mm)', ylabel='x (mm)',
             title=f'Ez — XZ slice (j={jc})', cmap='RdBu_r',
             symmetric=True, aspect='equal'),
        dict(frames=xy_frames,
             extent=[0, NX * DX * mm, 0, NY * DX * mm],
             xlabel='x (mm)', ylabel='y (mm)',
             title=f'Ez — XY slice (k={kc})', cmap='RdBu_r',
             symmetric=True, aspect='equal'),
    ]
    anim = viz.animate_field_slices_3d(panels, times=times, interval_ms=60,
                                       suptitle='Cavity standing wave')
    gif = os.path.join(os.path.dirname(__file__), 'test_06_animation.gif')
    anim.save(gif, writer='pillow', fps=18)
    plt.close('all')
    print(f"Animation saved to: {gif}")


if __name__ == '__main__':
    run()
