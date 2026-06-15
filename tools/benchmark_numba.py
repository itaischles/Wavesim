"""
benchmark_numba.py — NumPy vs Numba speedup & multicore scaling (ROADMAP §3).

Quantifies the Numba backend (wavesim/backend_numba.py) against the pure-NumPy
reference on the same canonical 3D loop (CPML on all six faces — the realistic
cost). Produces three figures that answer the questions the migration was meant
to settle:

  1. throughput   — us/cell-step vs grid size, NumPy vs Numba(all cores)
  2. multicore    — speedup vs thread count at a fixed size (the <10%-CPU fix)
  3. speedup      — Numba/NumPy speedup factor vs grid size

Timing is honest:
  * Numba's one-time JIT compile is triggered and DISCARDED in a warmup before any
    timed region (reported separately, once).
  * Each timed run advances the real loop for `--steps` steps after warmup.
  * us/cell-step = wall / (cells * steps) is the size-independent figure of merit.

Run:
    python tools\benchmark_numba.py
    python tools\benchmark_numba.py --sizes 48,64,96,128 --steps 60
    python tools\benchmark_numba.py --threads 1,2,4,8,12 --scale-size 96
"""

import sys
import os
import gc
import time
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import numba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavesim.grid import create_grid
from wavesim.materials import set_vacuum
from wavesim.pml import init_cpml
from wavesim.pec import apply_pec_mask
from wavesim.sources import GaussianSource, gaussian_pulse

import wavesim.update as np_update
import wavesim.pml as np_pml
import wavesim.backend_numba as nb

DEFAULT_SIZES = [48, 64, 80, 96, 112]
DEFAULT_STEPS = 60
WARMUP = 3
D_PML = 10


def _make(N):
    grid = create_grid(Nx=N, Ny=N, Nz=N, dx=1.0e-3)
    grid = set_vacuum(grid)
    cpml = init_cpml(grid, d_pml=D_PML)
    return grid, cpml


def _loop(uH, uE, uHp, uEp, grid, cpml, n_steps, ic, src):
    for n in range(n_steps):
        grid = uH(grid)
        grid, cpml = uHp(grid, cpml)
        grid = uE(grid)
        grid, cpml = uEp(grid, cpml)
        grid = apply_pec_mask(grid)
        grid.Ez[ic, ic, ic] += gaussian_pulse(src, n * grid.dt)
        grid.time_step += 1
    return grid, cpml


def _time_backend(funcs, N, n_steps):
    """Return (sec_per_step, us_per_cellstep) for one backend at size N, timed
    after a warmup that absorbs JIT compilation and cache warmth."""
    gc.collect()
    grid, cpml = _make(N)
    src = GaussianSource(t0=30 * grid.dt, width=10 * grid.dt)
    ic = N // 2
    # Warmup (also triggers + discards Numba compilation)
    _loop(*funcs, grid, cpml, WARMUP, ic, src)
    t0 = time.perf_counter()
    _loop(*funcs, grid, cpml, n_steps, ic, src)
    elapsed = time.perf_counter() - t0
    cells = N ** 3
    del grid, cpml
    return elapsed / n_steps, elapsed / (cells * n_steps) * 1e6


NPY = (np_update.update_H, np_update.update_E, np_pml.update_H_pml, np_pml.update_E_pml)
NB = (nb.update_H, nb.update_E, nb.update_H_pml, nb.update_E_pml)


def _measure_compile(N=48):
    """Wall time of the first (compiling) Numba step set minus a warmed step set."""
    gc.collect()
    grid, cpml = _make(N)
    src = GaussianSource(t0=30 * grid.dt, width=10 * grid.dt)
    ic = N // 2
    t0 = time.perf_counter()
    _loop(*NB, grid, cpml, 1, ic, src)         # includes JIT compile
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    _loop(*NB, grid, cpml, 1, ic, src)         # warmed
    warm = time.perf_counter() - t0
    del grid, cpml
    return max(first - warm, 0.0)


def sweep_sizes(sizes, steps):
    print("-" * 74)
    print(f"Size sweep (NumPy vs Numba, {numba.get_num_threads()} threads, "
          f"{steps} steps/size after warmup):")
    rows = []
    for N in sizes:
        s_np, us_np = _time_backend(NPY, N, steps)
        s_nb, us_nb = _time_backend(NB, N, steps)
        speedup = s_np / s_nb
        rows.append(dict(N=N, mcell=N ** 3 / 1e6,
                         us_np=us_np, us_nb=us_nb,
                         ms_np=s_np * 1e3, ms_nb=s_nb * 1e3, speedup=speedup))
        print(f"  N={N:<4d} {N**3/1e6:6.3f} Mcell | "
              f"NumPy {us_np:6.3f} us/cs ({s_np*1e3:7.1f} ms/step) | "
              f"Numba {us_nb:6.3f} us/cs ({s_nb*1e3:7.1f} ms/step) | "
              f"speedup {speedup:5.2f}x")
    return rows


def sweep_threads(threads, N, steps):
    print("-" * 74)
    print(f"Thread scaling at N={N} ({N**3/1e6:.3f} Mcell), {steps} steps/point:")
    # NumPy baseline (single-threaded by nature) for reference.
    s_np, _ = _time_backend(NPY, N, steps)
    rows = []
    base = None
    for nt in threads:
        numba.set_num_threads(nt)
        s_nb, _ = _time_backend(NB, N, steps)
        if base is None:
            base = s_nb
        rows.append(dict(threads=nt, ms=s_nb * 1e3,
                         scaling=base / s_nb, vs_numpy=s_np / s_nb))
        print(f"  threads={nt:<3d} {s_nb*1e3:8.1f} ms/step | "
              f"scaling {base/s_nb:5.2f}x (vs 1 thread) | "
              f"{s_np/s_nb:5.2f}x vs NumPy")
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    return rows, s_np


def _plot(size_rows, thread_rows, s_np_at_scale, scale_N, n_threads, out_dir):
    mcell = np.array([r['mcell'] for r in size_rows])
    us_np = np.array([r['us_np'] for r in size_rows])
    us_nb = np.array([r['us_nb'] for r in size_rows])
    speedup = np.array([r['speedup'] for r in size_rows])

    th = np.array([r['threads'] for r in thread_rows])
    scaling = np.array([r['scaling'] for r in thread_rows])
    vs_np = np.array([r['vs_numpy'] for r in thread_rows])

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(f'Wavesim — Numba vs NumPy (3D, CPML all 6 faces, '
                 f'{n_threads} cores)', fontsize=13)

    a1, a2, a3 = axs
    a1.plot(mcell, us_np, 'o-', color='C3', label='NumPy (1 core)')
    a1.plot(mcell, us_nb, 's-', color='C0', label=f'Numba ({n_threads} cores)')
    a1.set_xlabel('grid size (Mcell)'); a1.set_ylabel('us / cell-step')
    a1.set_title('Throughput (lower = faster)')
    a1.grid(True, alpha=0.3); a1.legend(fontsize=9)
    a1.set_ylim(bottom=0)

    # Multicore scaling: measured vs ideal-linear.
    a2.plot(th, scaling, 'o-', color='C2', label='measured')
    a2.plot(th, th, '--', color='gray', alpha=0.6, label='ideal linear')
    a2.set_xlabel('threads'); a2.set_ylabel('speedup vs 1 thread')
    a2.set_title(f'Multicore scaling (N={scale_N})')
    a2.grid(True, alpha=0.3); a2.legend(fontsize=9)
    for x, y in zip(th, vs_np):
        a2.annotate(f'{y:.1f}x vs NumPy', (x, scaling[list(th).index(x)]),
                    fontsize=7, xytext=(4, -10), textcoords='offset points')

    a3.bar([str(r['N']) for r in size_rows], speedup, color='C0', alpha=0.85)
    a3.axhline(1.0, color='C3', lw=1, ls='--', label='NumPy baseline')
    a3.set_xlabel('cube size N'); a3.set_ylabel('speedup factor')
    a3.set_title('Numba / NumPy speedup vs size')
    a3.grid(True, alpha=0.3, axis='y'); a3.legend(fontsize=9)
    for i, v in enumerate(speedup):
        a3.text(i, v, f'{v:.1f}x', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = os.path.join(out_dir, 'benchmark_numba.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close('all')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=str, default=None)
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--threads', type=str, default=None)
    ap.add_argument('--scale-size', type=int, default=96)
    args = ap.parse_args()

    sizes = ([int(s) for s in args.sizes.split(',')] if args.sizes else DEFAULT_SIZES)
    n_threads = numba.config.NUMBA_NUM_THREADS
    threads = ([int(t) for t in args.threads.split(',')] if args.threads
               else sorted({1, 2, 4, 8, n_threads}))
    threads = [t for t in threads if t <= n_threads]

    print("=" * 74)
    print("Wavesim — Numba acceleration benchmark")
    print("=" * 74)
    print(f"numba {numba.__version__} | numpy {np.__version__} | "
          f"NUMBA_NUM_THREADS={n_threads}")

    compile_s = _measure_compile()
    print(f"One-time Numba JIT compile: ~{compile_s:.1f} s "
          f"(amortised over the whole run; cache=True persists it across runs)\n")

    size_rows = sweep_sizes(sizes, args.steps)
    print()
    thread_rows, s_np_scale = sweep_threads(threads, args.scale_size, args.steps)

    out = _plot(size_rows, thread_rows, s_np_scale, args.scale_size,
                n_threads, os.path.dirname(__file__))
    print("\n" + "-" * 74)
    best = max(r['speedup'] for r in size_rows)
    print(f"Headline: up to {best:.1f}x faster than NumPy at the sizes tested; "
          f"{thread_rows[-1]['scaling']:.1f}x scaling across {threads[-1]} cores.")
    print(f"Figure saved to: {out}")
    print("=" * 74)


if __name__ == '__main__':
    main()
