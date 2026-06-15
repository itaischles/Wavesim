"""
test_07_simulation_api.py — Tutorial + smoke test for the v2 Simulation/Source API.

Tests 02-06 write the FDTD time loop out by hand (the canonical v1 pattern). This
test does the *same physics* — a soft Gaussian point source radiating into open
(CPML-absorbed) free space — but drives it entirely through the v2 orchestration
layer, so it doubles as a worked tutorial for the new classes:

    wavesim.sources.Source / PointSource / ArraySource   (the "what to inject")
    wavesim.simulation.Simulation                         (the canonical loop)

Read top-to-bottom: every step is commented as a tutorial. The validation at the
end confirms the orchestrated run behaves correctly:

    1. Energy rises while the pulse is driven, then decays toward zero as the
       wavefront is swallowed by the CPML (open-boundary absorption works).
    2. The radiated pattern from a centred source in vacuum is four-fold
       symmetric (the loop order / source injection is correct).
    3. The result is bit-for-bit identical to the equivalent hand-written loop
       (the class only orchestrates the existing pure functions — no physics
       changes).

Run (Windows console may need UTF-8):
    set PYTHONIOENCODING=utf-8
    python tests\test_07_simulation_api.py
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
from wavesim.pml import init_cpml
from wavesim.sources import PointSource, make_source_for_fmax
from wavesim.monitors import SnapshotMonitor, EnergyMonitor, FieldMonitor
from wavesim.simulation import Simulation
from wavesim import viz

# ---------------------------------------------------------------------- #
# Problem setup
# ---------------------------------------------------------------------- #
N = 200                 # square domain, Nx = Ny = N, single z-slice (Nz=1)
DX = 0.5e-3             # 0.5 mm cells
F_MAX = 10.0e9          # pulse content up to ~10 GHz
N_STEPS = 1200
SNAP_INTERVAL = 20
CENTER = N // 2


def build_simulation():
    """
    Assemble a Simulation the v2 way and return it (plus the monitors we want
    to inspect afterwards). This is the part worth copying into your own script.
    """
    # 1. Grid + material — same as the functional API.
    grid = create_grid(Nx=N, Ny=N, Nz=1, dx=DX)   # dt set automatically (CFL)
    grid = set_vacuum(grid)

    # 2. Absorbing boundaries on all four faces (open free space).
    cpml = init_cpml(grid, d_pml=10)

    # 3. One object owns the grid + cpml and will run the canonical loop.
    sim = Simulation(grid, cpml=cpml)

    # 4. A source is a reusable object: which component, where, and a waveform.
    #    make_source_for_fmax(...) returns a GaussianSource, which is itself
    #    callable — so it slots straight in as the PointSource waveform.
    #    (For a distributed/line/shaped drive use ArraySource(component, profile,
    #    waveform); for a custom excitation subclass Source directly.)
    sim.add_source(PointSource('Ez', CENTER, CENTER, 0,
                               make_source_for_fmax(F_MAX)))

    # 5. Monitors are registered with the sim; it records them every step.
    #    add_monitor returns the monitor so we can read its data after the run.
    snap = sim.add_monitor(SnapshotMonitor('Ez', k_slice=0,
                                           interval=SNAP_INTERVAL))
    energy = sim.add_monitor(EnergyMonitor())
    probe = sim.add_monitor(FieldMonitor('Ez', CENTER + 40, CENTER, 0))

    return sim, snap, energy, probe


def run_via_simulation():
    sim, snap, energy, probe = build_simulation()

    # 6. Run the loop. That's it — no hand-written update/CPML/PEC/inject/record.
    #    (A callback is available for progress, e.g.
    #     sim.run(N_STEPS, callback=lambda s, n: ...).)
    sim.run(N_STEPS)

    return sim.grid, snap, energy, probe


# ---------------------------------------------------------------------- #
# Equivalence check: the class must match a hand-written loop exactly
# ---------------------------------------------------------------------- #
def run_manual_reference():
    """The v1 hand-written loop, identical physics — used only to prove the
    Simulation class changes nothing about the result."""
    from wavesim.update import update_H, update_E
    from wavesim.pml import update_H_pml, update_E_pml
    from wavesim.pec import apply_pec_mask
    from wavesim.sources import gaussian_pulse

    grid = create_grid(Nx=N, Ny=N, Nz=1, dx=DX)
    grid = set_vacuum(grid)
    cpml = init_cpml(grid, d_pml=10)
    src = make_source_for_fmax(F_MAX)

    for n in range(N_STEPS):
        t = n * grid.dt
        grid = update_H(grid);  grid, cpml = update_H_pml(grid, cpml)
        grid = update_E(grid);  grid, cpml = update_E_pml(grid, cpml)
        grid = apply_pec_mask(grid)
        grid.Ez[CENTER, CENTER, 0] += gaussian_pulse(src, t)
        grid.time_step += 1
    return grid


def run():
    print("=" * 60)
    print("TEST 07 — v2 Simulation / Source API (tutorial + smoke test)")
    print("=" * 60)
    print(f"\nGrid: {N}x{N}x1, dx={DX*1e3:.2f} mm, {N_STEPS} steps")

    grid, snap, energy, probe = run_via_simulation()

    # ------------------------------------------------------------------ #
    # Check 1 — energy rises then is absorbed by the CPML
    # ------------------------------------------------------------------ #
    e = np.array(energy.values)
    e_peak = e.max()
    e_final = e[-1]
    decay = e_final / e_peak
    print("\n--- Check 1: open-boundary absorption (energy) ---")
    print(f"  peak energy   = {e_peak:.3e} J")
    print(f"  final energy  = {e_final:.3e} J  ({decay*100:.2f}% of peak)")

    # ------------------------------------------------------------------ #
    # Check 2 — four-fold symmetry of the radiated pattern (centred source)
    # ------------------------------------------------------------------ #
    # Use the strongest recorded snapshot (mid-run, ring well inside the domain)
    # — by the final step the field is fully absorbed and only numerical noise
    # remains, where a symmetry ratio is meaningless. Flip about the SOURCE node
    # (an odd-length window centred on CENTER), not the array midpoint: the
    # staggered Yee Ez pattern is symmetric about the driven node.
    fr_sym = int(np.argmax([np.max(np.abs(s)) for s in snap.snapshots]))
    field = snap.snapshots[fr_sym]
    m = min(CENTER, N - 1 - CENTER)
    win = field[CENTER - m:CENTER + m + 1, CENTER - m:CENTER + m + 1]
    sym_lr = np.max(np.abs(win - win[::-1, :]))
    sym_ud = np.max(np.abs(win - win[:, ::-1]))
    scale = np.max(np.abs(win)) + 1e-30
    print("\n--- Check 2: four-fold symmetry (centred source, vacuum) ---")
    print(f"  left-right asym = {sym_lr/scale:.2e} (relative)")
    print(f"  up-down   asym = {sym_ud/scale:.2e} (relative)")

    # ------------------------------------------------------------------ #
    # Check 3 — bit-identical to the hand-written loop
    # ------------------------------------------------------------------ #
    grid_ref = run_manual_reference()
    identical = np.array_equal(grid.Ez, grid_ref.Ez)
    max_diff = float(np.max(np.abs(grid.Ez - grid_ref.Ez)))
    print("\n--- Check 3: Simulation == hand-written loop ---")
    print(f"  identical = {identical}  (max |diff| = {max_diff:.1e})")

    # ------------------------------------------------------------------ #
    # Assertions
    # ------------------------------------------------------------------ #
    print("\n" + "-" * 60)
    assert e_peak > 0, "no energy was injected"
    assert decay < 0.05, f"energy not absorbed (final {decay*100:.1f}% of peak)"
    print("Check 1 [PASS] — wavefront absorbed by CPML")
    assert sym_lr / scale < 1e-9, "pattern not left-right symmetric"
    assert sym_ud / scale < 1e-9, "pattern not up-down symmetric"
    print("Check 2 [PASS] — radiated pattern is four-fold symmetric")
    assert identical, f"Simulation differs from manual loop (max {max_diff:.1e})"
    print("Check 3 [PASS] — orchestration is bit-identical to the manual loop")
    print("-" * 60)

    _make_figure(snap, energy, probe)
    _make_animation(snap, grid)

    print("\n" + "=" * 60)
    print("TEST 07 PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------- #
# Plotting
# ---------------------------------------------------------------------- #
def _make_figure(snap, energy, probe):
    mm = 1e3
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Test 07 — v2 Simulation / Source API', fontsize=14)

    # (a) a mid-run snapshot of the radiating pulse
    ax = fig.add_subplot(1, 3, 1)
    fr = min(len(snap.snapshots) - 1,
             int(0.35 * len(snap.snapshots)))     # while the ring is in-domain
    s = snap.snapshots[fr]
    vmax = np.max(np.abs(s)) + 1e-30
    ext = [0, N * DX * mm, 0, N * DX * mm]
    im = ax.imshow(s.T, origin='lower', extent=ext, cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='equal')
    plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(f'Ez snapshot (t={snap.snap_times[fr]*1e9:.2f} ns)')

    # (b) energy: rises with the drive, then absorbed by the CPML
    ax = fig.add_subplot(1, 3, 2)
    et = np.array(energy.times) * 1e9
    ax.semilogy(et, np.array(energy.values) + 1e-30, 'C0')
    ax.set_xlabel('t (ns)'); ax.set_ylabel('total energy (J)')
    ax.set_title('Energy: injected then absorbed')
    ax.grid(True, which='both', alpha=0.3)

    # (c) field at an off-centre probe: the pulse passes, then quiet
    ax = fig.add_subplot(1, 3, 3)
    pt = np.array(probe.times) * 1e9
    ax.plot(pt, probe.values, 'C3')
    ax.set_xlabel('t (ns)'); ax.set_ylabel('Ez (V/m)')
    ax.set_title(f'Probe at ({probe.i},{probe.j})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'test_07_output.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"\nFigure saved to: {out}")


def _make_animation(snap, grid):
    print("Saving animation...")
    anim = viz.animate_snapshots(snap, grid, interval_ms=50)
    gif = os.path.join(os.path.dirname(__file__), 'test_07_animation.gif')
    anim.save(gif, writer='pillow', fps=20)
    plt.close('all')
    print(f"Animation saved to: {gif}")


if __name__ == '__main__':
    run()
