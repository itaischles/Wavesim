"""
test_05_coax_tem.py — Coaxial TEM mode (first full 3D run).

This is the first test that uses Nz > 1: it exercises the 3D curl in update.py
and the z-face CPML in pml.py for real.

A coaxial line supports a TEM mode that propagates ALONG THE AXIS with purely
transverse fields:
    - radial electric field      E_r(r) ~ 1/r      (between the conductors)
    - azimuthal magnetic field   H_phi(r) ~ 1/r
    - in phase, with |E_r| / |H_phi| = eta0 = 377 ohm
    - phase/group velocity = c (no cutoff, dispersionless)

Because the TEM mode propagates along the axis, the axis MUST be resolved. We
therefore orient the coax along z (Nz > 1) and put the cross-section in XY:

    Grid:     Nx=Ny=64, Nz=100, dx=dy=dz=0.4 mm  (25.6 mm wide, 40 mm long)
    Geometry: set_coax(cx, cy, r_inner=6 cells, r_outer=18 cells), vacuum fill
              -> Z0 = (eta0 / 2pi) * ln(b/a) = 60*ln(3) ~ 66 ohm
    Boundary: CPML on z0,z1 ONLY. The outer PEC conductor shields the x/y
              domain walls, so no x/y absorber (or PEC face) is needed.
    Source:   soft, additive, azimuthally-symmetric RADIAL E field injected on
              the vacuum annulus at one z-plane (k=K_SRC), with a Gaussian time
              pulse. A uniform-in-r radial drive is used deliberately: it is not
              the 1/r eigenshape, so a clean 1/r profile downstream is a genuine
              result, not an artefact of the source. All azimuthally-symmetric
              non-TEM coax modes (TM0n) are well above cutoff here and stay
              evanescent, so only the TEM mode propagates.
    Run:      600 timesteps

Validation checks:
    1. Radial profile far from the source decays as 1/r: log-log slope of
       |E_r|(r) is -1 (within 0.2), correlation with 1/r > 0.97.
    2. Wave impedance |E_r| / |H_phi| ~ eta0 at every monitor radius (mean
       within 8%, each radius within 12%).
    3. Axial propagation velocity ~ c (within 4%), measured by the time lag of
       the pulse between two z-planes.

Pass criteria: all three checks pass. PNG + propagation GIF saved.

Run (Windows console may need UTF-8):
    set PYTHONIOENCODING=utf-8
    python tests\test_05_coax_tem.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavesim.grid import create_grid
from wavesim.materials import set_vacuum, set_coax
from wavesim.update import update_H, update_E
from wavesim.pml import init_cpml, update_H_pml, update_E_pml
from wavesim.pec import apply_pec_mask
from wavesim.sources import GaussianSource, gaussian_pulse
from wavesim.monitors import FieldMonitor, record_field
from wavesim.constants import C0, ETA0
from wavesim import viz

# ---------------------------------------------------------------------- #
# Geometry / problem constants (all lengths in metres unless _C suffix)
# ---------------------------------------------------------------------- #
NX, NY, NZ = 80, 80, 110
DX = 0.4e-3                       # uniform cubic cells
CX_C, CY_C = NX // 2, NY // 2     # cross-section centre (cell indices)
R_INNER_C, R_OUTER_C = 10, 30     # conductor radii in cells
R_INNER, R_OUTER = R_INNER_C * DX, R_OUTER_C * DX

D_PML = 10                       # z-face CPML thickness (cells)
K_SRC = 22                       # source z-plane
K_MEAS = 68                      # plane for 1/r profile + impedance (past near field)
# Axial taps for the phase-velocity fit: a line of probes spanning more than a
# half wavelength at F_PROBE, so a least-squares phase-vs-z fit averages out any
# residual CPML standing-wave ripple. Kept clear of source and far CPML.
K_VEL = np.arange(44, 93, 2)
K_A, K_B = int(K_VEL[0]), int(K_VEL[-1])

F_MAX = 12.0e9                   # pulse bandwidth (broadband baseband Gaussian)
N_STEPS = 600
SNAP_INTERVAL = 6

# The pulse is deliberately broadband, so it also contains content above the
# first higher-order coax cutoff (TM01, ~ c/(2(b-a)) ~ 19 GHz here). That
# content excites slow, dispersive TM01 and would corrupt a time-domain
# (envelope/peak) measurement. We therefore evaluate ALL three validation
# quantities in the FREQUENCY DOMAIN at a single low probe frequency, where the
# coax supports ONLY the TEM mode (v = c, |E_r|/|H_phi| = eta0, E_r ~ 1/r).
F_PROBE = 10.0e9
F_TM01_CUTOFF = C0 / (2.0 * (R_OUTER - R_INNER))

# Analytic characteristic impedance of the (vacuum) coax.
Z0_THEORY = (ETA0 / (2.0 * np.pi)) * np.log(R_OUTER / R_INNER)

# Radial sample line: +x axis from the centre, clear of both staircased walls.
R_PROF_C = np.arange(R_INNER_C + 2, R_OUTER_C - 1)     # monitored radii (cells)
R_FIT_C = np.arange(R_INNER_C + 4, R_OUTER_C - 2)      # cells used in the 1/r fit
R_IMP_C = np.array([14, 19, 24])                        # cells reported for impedance


def _radial_unit_field(cx_c, cy_c):
    """Unit radial vectors (ux, uy) at every cell centre; 0 at the centre."""
    ix = np.arange(NX)[:, None]
    iy = np.arange(NY)[None, :]
    xr = (ix - cx_c).astype(float)
    yr = (iy - cy_c).astype(float)
    r = np.sqrt(xr**2 + yr**2)
    r_safe = np.where(r == 0.0, 1.0, r)
    return xr / r_safe, yr / r_safe, r          # (Nx,Ny) each


def simulate():
    """Run the 3D coax and return everything the validation/plots need."""
    grid = create_grid(Nx=NX, Ny=NY, Nz=NZ, dx=DX)
    grid = set_vacuum(grid)
    grid = set_coax(grid, cx=CX_C * DX, cy=CY_C * DX,
                    r_inner=R_INNER, r_outer=R_OUTER, eps_r_fill=1.0)

    # Coax runs along z; only the z-faces are absorbing ports. The outer PEC
    # conductor shields the transverse (x/y) domain walls.
    cpml = init_cpml(grid, d_pml=D_PML, faces=('z0', 'z1'))

    # Soft radial-E source on the vacuum annulus at one z-plane.
    ux, uy, r_c = _radial_unit_field(CX_C, CY_C)
    annulus = (r_c > R_INNER_C + 0.5) & (r_c < R_OUTER_C - 0.5)
    pec_xy = grid.pec_mask[:, :, K_SRC]
    src = annulus & (~pec_xy)                    # (Nx,Ny) where to drive
    src_ux = np.where(src, ux, 0.0)
    src_uy = np.where(src, uy, 0.0)

    source = GaussianSource(t0=4.0 / (2 * np.pi * F_MAX),
                            width=1.0 / (2 * np.pi * F_MAX),
                            amplitude=1.0)

    # Radial-line time-series monitors at K_MEAS: Ex (=E_r) and Hy (=H_phi)
    # along +x. Full radial line -> 1/r profile AND impedance from one FFT pass.
    prof_Ex = [FieldMonitor('Ex', CX_C + int(m), CY_C, K_MEAS) for m in R_PROF_C]
    prof_Hy = [FieldMonitor('Hy', CX_C + int(m), CY_C, K_MEAS) for m in R_PROF_C]
    # Axial tap line (same radius) for the phase velocity at F_PROBE.
    m_mid = int(np.median(R_IMP_C))
    vel_line = [FieldMonitor('Ex', CX_C + m_mid, CY_C, int(k)) for k in K_VEL]

    # Peak-over-time envelopes (accumulated cheaply; no need to store frames).
    env_Ex_meas = np.zeros((NX, NY))             # |Ex| over XY at K_MEAS
    env_Et_meas = np.zeros((NX, NY))             # |E_transverse| over XY at K_MEAS

    # Frames for the propagation animation (XZ slice through the centre row).
    xz_snaps, xz_times = [], []
    xy_snaps = []                                # |E_t| over XY at K_MEAS

    for n in range(N_STEPS):
        t = n * grid.dt

        grid = update_H(grid)
        grid, cpml = update_H_pml(grid, cpml)
        grid = update_E(grid)
        grid, cpml = update_E_pml(grid, cpml)

        grid = apply_pec_mask(grid)              # inner + outer conductors

        amp = gaussian_pulse(source, t)
        grid.Ex[:, :, K_SRC] += amp * src_ux
        grid.Ey[:, :, K_SRC] += amp * src_uy

        # Envelopes at the measurement plane.
        ex = np.abs(grid.Ex[:, :, K_MEAS])
        et = np.sqrt(grid.Ex[:, :, K_MEAS]**2 + grid.Ey[:, :, K_MEAS]**2)
        np.maximum(env_Ex_meas, ex, out=env_Ex_meas)
        np.maximum(env_Et_meas, et, out=env_Et_meas)

        for m in prof_Ex:
            record_field(m, grid)
        for m in prof_Hy:
            record_field(m, grid)
        for m in vel_line:
            record_field(m, grid)

        if grid.time_step % SNAP_INTERVAL == 0:
            xz_snaps.append(grid.Ex[:, CY_C, :].copy())
            xz_times.append(grid.time_step * grid.dt)
            xy_snaps.append(np.sqrt(grid.Ex[:, :, K_MEAS]**2 +
                                    grid.Ey[:, :, K_MEAS]**2))

        grid.time_step += 1

    return dict(grid=grid, dt=grid.dt,
                prof_Ex=prof_Ex, prof_Hy=prof_Hy, vel_line=vel_line,
                env_Ex_meas=env_Ex_meas, env_Et_meas=env_Et_meas,
                xz_snaps=xz_snaps, xz_times=xz_times, xy_snaps=xy_snaps)


def run():
    print("=" * 60)
    print("TEST 05 — Coaxial TEM Mode (first full 3D run, Nz > 1)")
    print("=" * 60)
    print(f"\nGrid: {NX}x{NY}x{NZ}, dx={DX*1e3:.2f} mm "
          f"({NX*DX*1e3:.1f} mm wide, {NZ*DX*1e3:.1f} mm long)")
    print(f"Coax: a={R_INNER*1e3:.2f} mm, b={R_OUTER*1e3:.2f} mm  "
          f"-> Z0 (theory) = {Z0_THEORY:.1f} ohm")

    res = simulate()
    dt = res['dt']

    print(f"\nProbe frequency = {F_PROBE/1e9:.1f} GHz  "
          f"(TM01 cutoff ~ {F_TM01_CUTOFF/1e9:.1f} GHz -> pure TEM here)")

    # All three quantities are read from the steady single-frequency response
    # at F_PROBE, isolating the TEM mode from any broadband higher-order content.
    def _phasor(mon):
        """Complex spectral amplitude of a monitor time series at F_PROBE."""
        v = np.asarray(mon.values)
        v = v - v.mean()
        w = np.hanning(len(v))
        Nfft = 1 << 15
        V = np.fft.rfft(v * w, n=Nfft)
        freqs = np.fft.rfftfreq(Nfft, dt)
        k = int(np.argmin(np.abs(freqs - F_PROBE)))
        return V[k]

    r_prof = R_PROF_C * DX
    Ex_f = np.array([_phasor(m) for m in res['prof_Ex']])
    Hy_f = np.array([_phasor(m) for m in res['prof_Hy']])
    e_r = np.abs(Ex_f)

    # ------------------------------------------------------------------ #
    # Check 1 — 1/r radial profile (|E_r| at F_PROBE vs r)
    # ------------------------------------------------------------------ #
    fit_mask = np.isin(R_PROF_C, R_FIT_C)
    r_m = r_prof[fit_mask]
    e_fit = e_r[fit_mask]
    coeffs = np.polyfit(np.log(r_m), np.log(e_fit), 1)
    slope = coeffs[0]
    model = np.exp(np.polyval(coeffs, np.log(r_m)))
    corr_invr = float(np.corrcoef(e_fit, 1.0 / r_m)[0, 1])

    print(f"\n--- Check 1: radial profile ~ 1/r (plane k={K_MEAS}) ---")
    print(f"  log-log slope     = {slope:+.3f}   (ideal -1.000)")
    print(f"  corr with 1/r law = {corr_invr:.4f}")

    # ------------------------------------------------------------------ #
    # Check 2 — wave impedance |E_r| / |H_phi| ~ eta0
    # ------------------------------------------------------------------ #
    z_all = np.abs(Ex_f) / (np.abs(Hy_f) + 1e-30)
    print(f"\n--- Check 2: wave impedance |E_r|/|H_phi| ~ eta0 "
          f"({ETA0:.1f} ohm) ---")
    z_meas = []
    for m in R_IMP_C:
        idx = int(np.where(R_PROF_C == m)[0][0])
        z = float(z_all[idx])
        z_meas.append(z)
        err = abs(z - ETA0) / ETA0 * 100.0
        print(f"  r = {m:2d} cells ({m*DX*1e3:.1f} mm): "
              f"Z = {z:6.1f} ohm   (err {err:4.1f}%)")
    z_meas = np.array(z_meas)
    z_mean = float(np.mean(z_meas))
    z_mean_err = abs(z_mean - ETA0) / ETA0 * 100.0
    z_max_err = float(np.max(np.abs(z_meas - ETA0) / ETA0 * 100.0))
    print(f"  mean Z = {z_mean:.1f} ohm  (mean err {z_mean_err:.1f}%, "
          f"worst {z_max_err:.1f}%)")

    # ------------------------------------------------------------------ #
    # Check 3 — axial phase velocity ~ c
    # Fit the F_PROBE phase against z over the whole tap line; the slope is the
    # propagation constant beta. The least-squares fit over > lambda/2 averages
    # out residual CPML standing-wave ripple that biases a 2-point estimate.
    # ------------------------------------------------------------------ #
    z_line = K_VEL * DX
    phase = np.unwrap(np.array([np.angle(_phasor(m)) for m in res['vel_line']]))
    pslope, _ = np.polyfit(z_line, phase, 1)
    beta = abs(pslope)
    v_meas = 2.0 * np.pi * F_PROBE / beta
    v_err = abs(v_meas - C0) / C0 * 100.0
    print("\n--- Check 3: axial phase velocity ~ c ---")
    print(f"  beta = {beta:7.2f} rad/m  (fit over {len(K_VEL)} taps, "
          f"z={K_A}-{K_B} cells)")
    print(f"  v_ph = {v_meas/1e8:.4f} x1e8 m/s   "
          f"(c = {C0/1e8:.4f}, err {v_err:.2f}%)")

    # ------------------------------------------------------------------ #
    # Assertions
    # ------------------------------------------------------------------ #
    print("\n" + "-" * 60)
    assert abs(slope + 1.0) < 0.1, f"radial slope {slope:.3f} not ~ -1"
    assert corr_invr > 0.99, f"1/r correlation {corr_invr:.4f} too low"
    print("Check 1 [PASS] — radial profile is 1/r")
    assert z_mean_err < 6.0, f"mean impedance error {z_mean_err:.1f}% too large"
    assert z_max_err < 8.0, f"worst impedance error {z_max_err:.1f}% too large"
    print("Check 2 [PASS] — wave impedance ~ eta0")
    assert v_err < 2.0, f"phase velocity error {v_err:.2f}% too large"
    print("Check 3 [PASS] — axial phase velocity ~ c")
    print("-" * 60)

    _make_figure(res, slope, corr_invr, r_m, e_fit, model,
                 z_meas, z_mean, v_meas, v_err, dt)
    _make_animation(res, dt)

    print("\n" + "=" * 60)
    print("TEST 05 PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------- #
# Plotting
# ---------------------------------------------------------------------- #
def _conductor_circles(ax, scale, lw=1.2):
    """Overlay inner/outer conductor radii (scale: metres -> plot units)."""
    for rc in (R_INNER, R_OUTER):
        ax.add_patch(plt.Circle((CX_C * DX * scale, CY_C * DX * scale),
                                 rc * scale, fill=False, color='k',
                                 lw=lw, ls='--'))


def _make_figure(res, slope, corr_invr, r_m, e_r, model,
                 z_meas, z_mean, v_meas, v_err, dt):
    mm = 1e3
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Test 05 — Coaxial TEM Mode (3D)', fontsize=14)

    # (a) transverse |E_t| pattern at K_MEAS
    ax = fig.add_subplot(2, 3, 1)
    ext_xy = [0, NX * DX * mm, 0, NY * DX * mm]
    im = ax.imshow(res['env_Et_meas'].T, origin='lower', extent=ext_xy,
                   cmap='inferno', aspect='equal')
    plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    _conductor_circles(ax, mm, lw=1.0)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(f'Transverse |E| envelope (z plane k={K_MEAS})')

    # (b) radial profile vs 1/r
    ax = fig.add_subplot(2, 3, 2)
    ax.loglog(r_m * mm, e_r / e_r.max(), 'C0o', ms=4, label='measured |E_r|')
    ax.loglog(r_m * mm, model / e_r.max(), 'r--', lw=1.5,
              label=f'fit slope={slope:.2f}')
    ax.set_xlabel('r (mm)'); ax.set_ylabel('|E_r| (norm.)')
    ax.set_title(f'Radial profile (corr with 1/r = {corr_invr:.3f})')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

    # (c) wave impedance vs radius
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(R_IMP_C * DX * mm, z_meas, 'C1o-', ms=6, label='|E_r|/|H_phi|')
    ax.axhline(ETA0, color='k', ls='--', lw=1.2, label=f'eta0 = {ETA0:.0f}')
    ax.set_xlabel('r (mm)'); ax.set_ylabel('Z (ohm)')
    ax.set_ylim(0, max(ETA0 * 1.4, z_meas.max() * 1.1))
    ax.set_title(f'Wave impedance (mean {z_mean:.0f} ohm)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (d) XZ propagation snapshot at the peak frame
    ax = fig.add_subplot(2, 3, 4)
    fr = int(np.argmax([np.max(np.abs(s)) for s in res['xz_snaps']]))
    s = res['xz_snaps'][fr]
    vmax = max(np.max(np.abs(s)), 1e-30)
    ext_xz = [0, NZ * DX * mm, 0, NX * DX * mm]      # z horizontal, x vertical
    im = ax.imshow(s, origin='lower', extent=ext_xz, cmap='RdBu_r',
                   aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    for kk, c in [(K_SRC, 'k'), (K_A, 'g'), (K_B, 'g'), (K_MEAS, 'm')]:
        ax.axvline(kk * DX * mm, color=c, ls=':', lw=1)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('x (mm)')
    ax.set_title(f'Ex on XZ centre slice (t={res["xz_times"][fr]*1e9:.2f} ns)')

    # (e) axial taps: pulse arriving later at the farther plane
    ax = fig.add_subplot(2, 3, 5)
    near, far = res['vel_line'][0], res['vel_line'][-1]
    ta = np.arange(len(near.values)) * dt * 1e9
    ax.plot(ta, near.values, 'C0', lw=1.0, label=f'Ex @ k={K_A}')
    ax.plot(ta, far.values, 'C3', lw=1.0, label=f'Ex @ k={K_B}')
    ax.set_xlabel('t (ns)'); ax.set_ylabel('Ex (V/m)')
    ax.set_title(f'Axial taps: v_ph={v_meas/1e8:.3f}e8 m/s (err {v_err:.1f}%)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (f) cross-section geometry (PEC mask)
    ax = fig.add_subplot(2, 3, 6)
    pec = res['grid'].pec_mask[:, :, K_MEAS].astype(float)
    ax.imshow(pec.T, origin='lower', extent=ext_xy, cmap='Greys',
              aspect='equal')
    _conductor_circles(ax, mm, lw=1.0)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title('Coax cross-section (PEC = black)')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'test_05_output.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"\nFigure saved to: {out}")


def _make_animation(res, dt):
    """Axial-propagation (XZ) + transverse-pattern (XY) animation.

    Built with the shared wavesim.viz.animate_field_slices_3d helper (the same
    multi-plane animator Test 06 uses), instead of a bespoke FuncAnimation.
    """
    print("Saving animation...")
    mm = 1e3
    panels = [
        dict(frames=res['xz_snaps'],                      # Ex on XZ centre slice
             extent=[0, NZ * DX * mm, 0, NX * DX * mm],
             xlabel='z (mm)', ylabel='x (mm)',
             title='Ex — axial propagation (XZ slice)',
             cmap='RdBu_r', symmetric=True, aspect='auto',
             vlines=[(K_SRC * DX * mm, 'k'), (K_MEAS * DX * mm, 'm')]),
        dict(frames=[s.T for s in res['xy_snaps']],       # |E_t| transverse
             extent=[0, NX * DX * mm, 0, NY * DX * mm],
             xlabel='x (mm)', ylabel='y (mm)',
             title=f'|E| transverse pattern (k={K_MEAS})',
             cmap='inferno', symmetric=False, aspect='equal'),
    ]
    anim = viz.animate_field_slices_3d(panels, times=res['xz_times'],
                                       interval_ms=60)
    gif = os.path.join(os.path.dirname(__file__), 'test_05_animation.gif')
    anim.save(gif, writer='pillow', fps=18)
    plt.close('all')
    print(f"Animation saved to: {gif}")


if __name__ == '__main__':
    run()
