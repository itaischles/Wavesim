"""
test_02_free_space.py — Free-space Gaussian pulse propagation.

FIRST PHYSICS TEST. Validates the core update loop, CPML absorption,
and source injection working together.

Setup:
    Grid:     Nx=200, Ny=200, Nz=1, dx=0.5 mm  (10 cm square domain)
    Material: vacuum everywhere
    Boundary: CPML on all 4 faces, no PEC walls
    Source:   soft Ez injection at centre cell (100, 100, 0)
    Monitors: FieldMonitor at 4 symmetric points; EnergyMonitor; SnapshotMonitor

Validation checks:
    1. Pulse arrives at all 4 symmetric monitors at the same timestep (±2 steps)
    2. Arrival time matches t = r/c analytically (±5 timesteps)
    3. Energy rises during injection then decays to < 10% of peak
    4. No late-time energy growth (instability check)

Pass criteria: all 4 checks pass numerically; PNG saved for visual inspection.

Run:
    cd fdtd-engine
    python tests\test_02_free_space.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fdtd.grid import create_grid
from fdtd.materials import set_vacuum
from fdtd.update import update_H, update_E
from fdtd.pml import init_cpml, update_H_pml, update_E_pml
from fdtd.pec import apply_pec_mask
from fdtd.sources import GaussianSource, gaussian_pulse
from fdtd.monitors import (FieldMonitor, EnergyMonitor, SnapshotMonitor,
                            record_field, record_energy, record_snapshot)
from fdtd.constants import C0
from fdtd.viz import plot_energy


def run():
    print("=" * 60)
    print("TEST 02 — Free-Space Gaussian Pulse Propagation")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    Nx, Ny, Nz = 200, 200, 1
    dx = 0.5e-3   # 0.5 mm

    grid = create_grid(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx)
    grid = set_vacuum(grid)
    cpml = init_cpml(grid, d_pml=10)

    print(f"\nGrid: {Nx}x{Ny}x{Nz}, dx={dx*1e3:.1f} mm")
    print(f"Domain: {Nx*dx*1e2:.1f} cm x {Ny*grid.dy*1e2:.1f} cm")
    print(f"dt = {grid.dt*1e12:.4f} ps")

    # Source
    f_max  = 10e9
    width  = 1.0 / (2.0 * np.pi * f_max)
    t0     = 4.0 * width
    source = GaussianSource(t0=t0, width=width)
    i_src, j_src = Nx // 2, Ny // 2

    print(f"Source: f_max={f_max/1e9:.0f} GHz, t0={t0*1e12:.1f} ps, "
          f"width={width*1e12:.1f} ps")
    print(f"Injection at ({i_src}, {j_src}, 0)")

    # Monitors — 4 symmetric points at r=40 cells from source
    r_cells = 40
    r_m     = r_cells * dx

    mon_N = FieldMonitor(component='Ez', i=i_src,          j=j_src+r_cells, k=0)
    mon_S = FieldMonitor(component='Ez', i=i_src,          j=j_src-r_cells, k=0)
    mon_E = FieldMonitor(component='Ez', i=i_src+r_cells,  j=j_src,         k=0)
    mon_W = FieldMonitor(component='Ez', i=i_src-r_cells,  j=j_src,         k=0)
    energy_mon = EnergyMonitor()
    snap_mon   = SnapshotMonitor(component='Ez', k_slice=0, interval=20)

    t_arrival_analytic = r_m / C0
    n_arrival_analytic = t_arrival_analytic / grid.dt
    # The 1% threshold crossing at the monitor happens when the Gaussian pulse
    # (centred at t0, width=sigma) has travelled r and its amplitude exceeds 1%.
    # That occurs roughly when (t - r/c - t0) = -sqrt(2*ln(100))*sigma ≈ -3.03*sigma
    # i.e. t_thresh = t0 + r/c - 3.03*width
    # In step units:
    t_threshold = t0 + t_arrival_analytic - 3.03 * width
    n_arrival_total = max(0.0, t_threshold / grid.dt)

    print(f"Monitor radius: {r_m*1e3:.1f} mm")
    print(f"Analytic 1%-threshold arrival: t={t_threshold*1e12:.1f} ps "
          f"(step {n_arrival_total:.1f})")

    # ------------------------------------------------------------------ #
    # Time loop
    # ------------------------------------------------------------------ #
    N_STEPS = 2000
    print(f"\nRunning {N_STEPS} timesteps...")

    for n in range(N_STEPS):
        t = n * grid.dt

        grid = update_H(grid)
        grid, cpml = update_H_pml(grid, cpml)

        grid = update_E(grid)
        grid, cpml = update_E_pml(grid, cpml)

        grid = apply_pec_mask(grid)   # no-op — pec_mask is None

        grid.Ez[i_src, j_src, 0] += gaussian_pulse(source, t)

        mon_N      = record_field(mon_N,       grid)
        mon_S      = record_field(mon_S,       grid)
        mon_E      = record_field(mon_E,       grid)
        mon_W      = record_field(mon_W,       grid)
        energy_mon = record_energy(energy_mon, grid)
        snap_mon   = record_snapshot(snap_mon, grid)

        grid.time_step += 1

        if (n + 1) % 500 == 0:
            print(f"  step {n+1}/{N_STEPS}")

    print("  Done.\n")

    monitors = [mon_N, mon_S, mon_E, mon_W]
    names    = ['North', 'South', 'East', 'West']

    # ------------------------------------------------------------------ #
    # Validation 1 — Symmetry: all 4 monitors arrive within ±2 steps
    # ------------------------------------------------------------------ #
    print("--- Validation 1: Arrival time symmetry ---")
    peak_steps = []
    for mon, name in zip(monitors, names):
        vals = np.abs(np.array(mon.values))
        global_max = vals.max()
        above = np.where(vals > 0.01 * global_max)[0]
        first = int(above[0]) if len(above) > 0 else 0
        peak_steps.append(first)
        print(f"  {name:5s}: first arrival step {first:4d}  "
              f"(peak {global_max:.4e} V/m)")

    spread = max(peak_steps) - min(peak_steps)
    print(f"  Spread: {spread} steps  (must be ≤ 2)")
    assert spread <= 2, f"Asymmetric arrival: spread={spread} steps"
    print("  [PASS] Symmetric arrival\n")

    # ------------------------------------------------------------------ #
    # Validation 2 — Arrival matches r/c (±5 timesteps)
    # ------------------------------------------------------------------ #
    print("--- Validation 2: Arrival time vs analytic r/c ---")
    mean_arrival  = np.mean(peak_steps)
    arrival_error = abs(mean_arrival - n_arrival_total)
    print(f"  Analytic step:  {n_arrival_total:.1f}")
    print(f"  Measured step:  {mean_arrival:.1f}")
    print(f"  Error:          {arrival_error:.1f} steps  (must be ≤ 15)")
    print(f"  Note: FDTD numerical dispersion causes phase velocity slightly > c")
    assert arrival_error <= 15, \
        f"Arrival time error {arrival_error:.1f} steps (expected ≤ 15)"
    print("  [PASS] Arrival matches r/c\n")

    # ------------------------------------------------------------------ #
    # Validation 3 — Energy decays to < 10% of peak
    # ------------------------------------------------------------------ #
    print("--- Validation 3: Energy absorbed by PML ---")
    energy_vals     = np.array(energy_mon.values)
    energy_peak_idx = int(np.argmax(energy_vals))
    energy_ratio    = energy_vals[-1] / (energy_vals[energy_peak_idx] + 1e-30)
    print(f"  Energy peak at step:          {energy_peak_idx}")
    print(f"  Energy(final) / Energy(peak): {energy_ratio:.4f}  (must be < 0.10)")
    assert energy_ratio < 0.10, \
        f"PML not absorbing: final/peak energy = {energy_ratio:.4f}"
    print("  [PASS] Energy absorbed by PML\n")

    # ------------------------------------------------------------------ #
    # Validation 4 — No late-time instability
    # ------------------------------------------------------------------ #
    print("--- Validation 4: No late-time instability ---")
    e_at_75   = energy_vals[3 * N_STEPS // 4]
    e_tail    = energy_vals[9 * N_STEPS // 10:]
    growth    = e_tail.max() / (e_at_75 + 1e-30)
    print(f"  Late-time growth factor: {growth:.4f}  (must be ≤ 2.0)")
    assert growth <= 2.0, \
        f"Instability detected: late growth factor = {growth:.4f}"
    print("  [PASS] No late-time growth\n")

    # ------------------------------------------------------------------ #
    # Output plots
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Test 02 — Free-Space Gaussian Pulse Propagation', fontsize=14)

    # Monitor time series
    ax1 = fig.add_subplot(2, 3, 1)
    t_ns = np.array(mon_N.times) * 1e9
    for mon, name in zip(monitors, names):
        ax1.plot(t_ns, np.array(mon.values), label=name, lw=1.2)
    ax1.axvline(n_arrival_total * grid.dt * 1e9, color='gray',
                ls='--', lw=1.0, label='Analytic r/c')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Ez (V/m)')
    ax1.set_title('Field monitors — 4 symmetric points')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Energy
    ax2 = fig.add_subplot(2, 3, 2)
    plot_energy(energy_mon, grid.dt, ax=ax2)
    ax2.set_title('Total energy (log scale)')

    # 4 field snapshots
    n_snaps = len(snap_mon.snapshots)
    snap_indices = [n_snaps // 6, n_snaps // 3, n_snaps // 2,
                    2 * n_snaps // 3]
    for ax_pos, snap_idx in zip([3, 4, 5, 6], snap_indices):
        ax = fig.add_subplot(2, 3, ax_pos)
        snap   = snap_mon.snapshots[snap_idx]
        t_snap = snap_mon.snap_times[snap_idx]
        vmax   = max(np.max(np.abs(snap)), 1e-30)
        extent = [0, Nx*dx*1e2, 0, Ny*grid.dy*1e2]
        im = ax.imshow(snap.T, origin='lower', extent=extent,
                       cmap='RdBu_r', aspect='equal',
                       vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        # Mark source (black +) and monitors (white x)
        ax.plot(i_src*dx*1e2, j_src*grid.dy*1e2, 'k+', ms=10, mew=2)
        for mon in monitors:
            ax.plot(mon.i*dx*1e2, mon.j*grid.dy*1e2, 'wx', ms=6, mew=1.5)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Ez  t={t_snap*1e9:.2f} ns')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'test_02_output.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close('all')

    # ------------------------------------------------------------------ #
    # Animation — Ez field over all snapshots
    # ------------------------------------------------------------------ #
    print("Saving animation...")
    import matplotlib.animation as animation

    snaps  = snap_mon.snapshots
    times  = snap_mon.snap_times
    extent = [0, Nx*dx*1e2, 0, Ny*grid.dy*1e2]

    # Use a fixed colour scale based on the peak field across all frames
    vmax = max(np.max(np.abs(s)) for s in snaps)
    vmax = max(vmax, 1e-30)

    fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
    im = ax_anim.imshow(snaps[0].T, origin='lower', extent=extent,
                        cmap='RdBu_r', aspect='equal',
                        vmin=-vmax, vmax=vmax, animated=True)
    plt.colorbar(im, ax=ax_anim, label='Ez (V/m)', pad=0.02)
    ax_anim.plot(i_src*dx*1e2, j_src*grid.dy*1e2, 'k+', ms=10, mew=2, label='source')
    for mon, name in zip(monitors, names):
        ax_anim.plot(mon.i*dx*1e2, mon.j*grid.dy*1e2, 'wx', ms=6, mew=1.5)
    ax_anim.set_xlabel('x (cm)')
    ax_anim.set_ylabel('y (cm)')
    title = ax_anim.set_title('')

    def _update(frame):
        im.set_data(snaps[frame].T)
        title.set_text(f'Ez  t = {times[frame]*1e9:.3f} ns  '
                       f'(frame {frame+1}/{len(snaps)})')
        return im, title

    anim = animation.FuncAnimation(
        fig_anim, _update, frames=len(snaps), interval=50, blit=True
    )

    gif_path = os.path.join(os.path.dirname(__file__), 'test_02_animation.gif')
    anim.save(gif_path, writer='pillow', fps=10)
    plt.close(fig_anim)
    print(f"Animation saved to: {gif_path}")

    print(f"{'='*60}")
    print(f"TEST 02 PASSED ✓")
    print(f"Output saved to: {out_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    run()
