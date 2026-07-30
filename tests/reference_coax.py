"""The conformal-PEC reference case (plan §7/§8) — build, run, measure.

Air coax, a = 3 mm, b = 9 mm, 100 mm long, dz = 1 mm, Gaussian fmax = 10 GHz,
a ModalPort at each end (amplitude 1 and 0). Analytic Z₀ = 59.9585·ln(b/a) =
65.871 Ω.

Not a pytest module — it is the measurement harness behind the baseline table,
run directly:

    python tests/reference_coax.py [cell_mm ...]

The coax is built **analytically** (``pec_mask`` from r² on cell centres) rather
than from a voxelised ``materials.npz``, so the case is independent of the
FreeCAD workbench. Two details from §8 that are easy to get wrong:

* ``pec_faces`` must be passed to the ``Simulation``, not applied once before the
  loop — PEC enforcement belongs after *every* E update, and applying it once
  gives a completely different, unstable-looking answer;
* ``backend='numba'`` — the NumPy path is far too slow at these sizes.

Modal purity (§8) is measured without any solver change, by comparing a
VoltageMonitor on a radial line against a CurrentMonitor looping the inner
conductor. ``∮H·dl`` measures net enclosed current, which is zero for an m=1
azimuthal pattern, so a TE11 parasitic shows up in V only; a TEM residue shows
equally in both. The split between the two floors is therefore a non-TEM
detector.

Conformal mode (``--conformal``) runs the same case with the analytic cut-cell
fractions from :mod:`conformal_shapes`, which is how V3-V5 are re-measured after
S5. It forces ``backend='numpy'``: **S2 has landed only in the NumPy reference**,
so the Numba and CUDA H updates still integrate the full face area while
``apply_pec_mask`` zeroes E by the conformal rule — E and H would see different
geometry and the answer would be quietly wrong rather than slow. Drop the
override once phase 4 lands.
"""

import sys
import numpy as np

sys.path.insert(0, r"c:\Users\itais\Desktop\Wavesim")
sys.path.insert(0, r"c:\Users\itais\Desktop\Wavesim\tests")

import wavesim as ws
from wavesim.constants import C0
from wavesim.mode_solver import solve_tem_modes
from conformal_shapes import coax_fractions

A_IN = 3.0e-3           # inner radius
B_OUT = 9.0e-3          # outer radius (shield)
LENGTH = 100.0e-3       # line length
DZ = 1.0e-3
FMAX = 10e9
Z0_ANALYTIC = 59.9585 * np.log(B_OUT / A_IN)      # 65.871 Ω


def build(cell, eps_r=1.0, conformal=False):
    """Grid + analytic coax PEC mask. ``cell`` is the transverse cell size (m).

    With ``conformal`` the six cut-cell open-fraction arrays are attached too,
    so the H update, the E masking and the mode solve all see the same cut
    geometry. ``pec_mask`` is still set — it stays the fully-covered test.
    """
    n = int(round(2 * B_OUT / cell))
    nz = int(round(LENGTH / DZ))
    grid = ws.create_grid(Nx=n, Ny=n, Nz=nz, dx=cell, dy=cell, dz=DZ)
    ws.set_vacuum(grid)
    for axis in 'xyz':
        getattr(grid, 'eps_' + axis)[...] = eps_r

    cx = cy = 0.5 * n * cell
    xc = grid.xc[:, None, None] - cx
    yc = grid.yc[None, :, None] - cy
    r2 = xc ** 2 + yc ** 2
    grid.pec_mask = np.broadcast_to(
        (r2 < A_IN ** 2) | (r2 > B_OUT ** 2), (grid.Nx, grid.Ny, grid.Nz)).copy()
    if conformal:
        ws.set_material_arrays(
            grid, grid.eps_x, grid.eps_y, grid.eps_z,
            grid.mu_x, grid.mu_y, grid.mu_z,
            **coax_fractions(grid, cx, cy, A_IN, B_OUT))
    return grid, cx, cy


def _floor_db(values, tail=0.35):
    """Late-time floor relative to the peak, in dB."""
    v = np.abs(np.asarray(values, dtype=float))
    peak = v.max()
    if peak <= 0:
        return np.nan
    late = v[int((1.0 - tail) * len(v)):]
    return 20.0 * np.log10(max(late.max(), 1e-300) / peak)


def run(cell, fmax=FMAX, steps_factor=6.0, verbose=True, conformal=False,
        backend=None, waveform=None):
    """One resolution: returns the §7 row plus the §8 V/I purity split."""
    grid, cx, cy = build(cell, conformal=conformal)
    # Conformal geometry is honoured only by the NumPy H update until phase 4;
    # see the module docstring. Silently stepping it on Numba would mismatch E
    # and H, which is a wrong answer rather than a slow one.
    backend = backend or ('numpy' if conformal else 'numba')

    # Modes on the two end planes. The launch plane sits one cell in from the
    # low face: a low-index sheet writes its ghost H at k-1, so k=0 is unusable.
    k_lo, k_hi = 1, grid.Nz - 1
    mode_lo = solve_tem_modes(grid, normal='z', position=k_lo * DZ,
                              compute_params=True)[0]
    mode_hi = solve_tem_modes(grid, normal='z', position=k_hi * DZ,
                              compute_params=True)[0]

    wave = waveform or ws.GaussianPulse.for_fmax(fmax)
    launch = ws.ModalPort(mode_lo, amplitude=1.0, waveform=wave)
    absorb = ws.ModalPort(mode_hi, amplitude=0.0)

    # Mid-line probes: V across the gap on a radial line, I around the inner
    # conductor. Endpoints sit inside the metal so the line spans the full gap.
    k_mid = grid.Nz // 2
    z_mid = k_mid * DZ
    vmon = ws.VoltageMonitor(path=((cx + 0.5 * A_IN, cy, z_mid),
                                   (cx + 1.15 * B_OUT, cy, z_mid)))
    imon = ws.CurrentMonitor(path=ws.circular_path(
        cx, cy, z_mid, radius=0.5 * (A_IN + B_OUT), normal='z', n_points=128))

    sim = ws.Simulation(grid, sources=[], monitors=[vmon, imon],
                        backend=backend, pec_faces=('x0', 'x1', 'y0', 'y1'))
    sim.add_boundary(launch)
    sim.add_boundary(absorb)

    transit = LENGTH / C0
    sim.run(int(steps_factor * transit / grid.dt))

    v = np.asarray(vmon.values, dtype=float)
    i = np.asarray(imon.values, dtype=float)
    v_peak = float(np.abs(v).max())
    i_peak = float(np.abs(i).max())

    row = dict(
        cell_mm=cell * 1e3,
        n_transverse=grid.Nx,
        conformal=conformal,
        backend=backend,
        Z0_mode=float(mode_lo.impedance),
        Z0_err_pct=100.0 * (mode_lo.impedance / Z0_ANALYTIC - 1.0),
        Z0_ratio=v_peak / i_peak if i_peak else np.nan,
        v_peak=v_peak,
        v_floor_db=_floor_db(v),
        i_floor_db=_floor_db(i),
        eps_eff=float(mode_lo.eps_eff),
        stable=bool(np.all(np.isfinite(v)) and np.all(np.isfinite(i))),
    )
    row['purity_split_db'] = row['v_floor_db'] - row['i_floor_db']
    row['v'], row['i'], row['t'] = v, i, np.asarray(vmon.times, dtype=float)

    if verbose:
        print(f"  cell {row['cell_mm']:.4f} mm  ({grid.Nx}x{grid.Ny}x{grid.Nz})"
              f"  Z0 = {row['Z0_mode']:.3f} ohm ({row['Z0_err_pct']:+.2f}%)"
              f"  Vpk = {row['v_peak']:.4f}"
              f"  Vfloor = {row['v_floor_db']:.1f} dB"
              f"  Ifloor = {row['i_floor_db']:.1f} dB"
              f"  split = {row['purity_split_db']:+.1f} dB"
              f"  eps_eff = {row['eps_eff']:.6f}")
    return row


if __name__ == '__main__':
    args = sys.argv[1:]
    conformal = '--conformal' in args
    backend = 'numpy' if '--numpy' in args else None
    cells = [float(a) * 1e-3 for a in args if not a.startswith('--')] or \
        [0.5e-3, 0.375e-3, 0.25e-3, 0.1875e-3]
    print(f"Reference coax: a = {A_IN*1e3} mm, b = {B_OUT*1e3} mm, "
          f"L = {LENGTH*1e3} mm, dz = {DZ*1e3} mm, fmax = {FMAX/1e9} GHz")
    print(f"Analytic Z0 = {Z0_ANALYTIC:.3f} ohm   "
          f"({'CONFORMAL' if conformal else 'staircase'})\n")
    rows = [run(c, conformal=conformal, backend=backend) for c in cells]

    print("\n| cell (mm) | Z0 (ohm) | err | V peak | V floor | I floor | split |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['cell_mm']:.4f} | {r['Z0_mode']:.2f} | "
              f"{r['Z0_err_pct']:+.2f}% | {r['v_peak']:.3f} | "
              f"{r['v_floor_db']:.1f} dB | {r['i_floor_db']:.1f} dB | "
              f"{r['purity_split_db']:+.1f} dB |")
