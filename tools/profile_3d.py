"""
profile_3d.py — Memory & runtime profiling for the pure-NumPy 3D solver.

Answers the ROADMAP question: "Profile memory and runtime at representative 3D
sizes; document practical grid-size limits for pure NumPy." It runs the canonical
time loop (with CPML on all six faces — the realistic 3D cost) over a sweep of
cube sizes, measures throughput, and extrapolates how many steps fit a 3- and
5-minute wall-clock budget at each size.

Two numbers matter for planning:
    * throughput   — microseconds per cell-step (size-independent figure of merit)
    * memory       — bytes per cell, and where they go (grid vs CPML psi arrays)

Memory is reported BOTH analytically (exact array bytes — deterministic) and as
the measured process working set, which includes interpreter + library overhead.

Run:
    python tools\profile_3d.py
    python tools\profile_3d.py --sizes 40,64,96 --steps 120
"""

import sys
import os
import gc
import time
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wavesim.grid import create_grid, FDTDGrid
from wavesim.materials import set_vacuum
from wavesim.update import update_H, update_E
from wavesim.pml import init_cpml, update_H_pml, update_E_pml, CPMLArrays
from wavesim.pec import apply_pec_mask
from wavesim.sources import GaussianSource, gaussian_pulse

DEFAULT_SIZES = [32, 48, 64, 80, 96, 112]
DEFAULT_STEPS = 100          # timed steps (after warmup) per size
WARMUP = 5
D_PML = 10


# ---------------------------------------------------------------------- #
# Windows process-memory query (stdlib only; no psutil dependency)
# ---------------------------------------------------------------------- #
def _working_set_mb():
    """Current process working-set size in MiB (Windows PSAPI)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        # Pin handle/pointer types — the GetCurrentProcess pseudo-handle is a
        # 64-bit value and is silently truncated if left as the default c_int.
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if not psapi.GetProcessMemoryInfo(
                k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float('nan')
        return pmc.WorkingSetSize / 1024**2
    except Exception:
        return float('nan')


def _array_bytes(obj):
    """Sum nbytes over every ndarray attribute of a dataclass-like object."""
    total = 0
    for v in vars(obj).values():
        if isinstance(v, np.ndarray):
            total += v.nbytes
    return total


# ---------------------------------------------------------------------- #
# One profiled run at a given cube size
# ---------------------------------------------------------------------- #
def profile_size(N, n_steps):
    gc.collect()
    grid = create_grid(Nx=N, Ny=N, Nz=N, dx=1.0e-3)
    grid = set_vacuum(grid)
    cpml = init_cpml(grid, d_pml=D_PML)          # all six faces — full 3D cost
    source = GaussianSource(t0=30 * grid.dt, width=10 * grid.dt)
    ic = N // 2

    grid_bytes = _array_bytes(grid)
    cpml_bytes = _array_bytes(cpml)
    total_bytes = grid_bytes + cpml_bytes

    def _step(n):
        nonlocal grid, cpml
        grid = update_H(grid)
        grid, cpml = update_H_pml(grid, cpml)
        grid = update_E(grid)
        grid, cpml = update_E_pml(grid, cpml)
        grid = apply_pec_mask(grid)
        grid.Ez[ic, ic, ic] += gaussian_pulse(source, n * grid.dt)
        grid.time_step += 1

    for n in range(WARMUP):                       # JIT-free, but warms caches
        _step(n)
    t0 = time.perf_counter()
    for n in range(n_steps):
        _step(WARMUP + n)
    elapsed = time.perf_counter() - t0

    ws_mb = _working_set_mb()
    cells = N ** 3
    us_per_cellstep = elapsed / (cells * n_steps) * 1e6
    sec_per_step = elapsed / n_steps

    del grid, cpml
    gc.collect()

    return dict(N=N, cells=cells, elapsed=elapsed, sec_per_step=sec_per_step,
                us_per_cellstep=us_per_cellstep,
                grid_mb=grid_bytes / 1024**2, cpml_mb=cpml_bytes / 1024**2,
                total_mb=total_bytes / 1024**2, ws_mb=ws_mb,
                bytes_per_cell=total_bytes / cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=str, default=None,
                    help='comma-separated cube sizes, e.g. 40,64,96')
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    args = ap.parse_args()
    sizes = ([int(s) for s in args.sizes.split(',')]
             if args.sizes else DEFAULT_SIZES)

    print("=" * 74)
    print("Wavesim — 3D memory & runtime profile (NumPy, CPML on all 6 faces)")
    print("=" * 74)
    print(f"Timed steps per size: {args.steps}  (warmup {WARMUP})\n")

    rows = []
    for N in sizes:
        r = profile_size(N, args.steps)
        rows.append(r)
        print(f"  N={N:<4d} {r['cells']/1e6:6.3f} Mcell | "
              f"{r['us_per_cellstep']:6.3f} us/cell-step | "
              f"{r['sec_per_step']*1e3:7.2f} ms/step | "
              f"mem {r['total_mb']:7.1f} MB ({r['bytes_per_cell']:.0f} B/cell, "
              f"ws {r['ws_mb']:.0f} MB)")

    # ------------------------------------------------------------------ #
    # Table: how many steps fit a 3- and 5-minute budget at each size
    # ------------------------------------------------------------------ #
    print("\n" + "-" * 74)
    print("Budget table (steps that fit a wall-clock target at each size):")
    print(f"  {'N':>4} {'Mcell':>7} {'mem(MB)':>9} {'B/cell':>7} "
          f"{'steps/3min':>11} {'steps/5min':>11}")
    for r in rows:
        s3 = 180.0 / r['sec_per_step']
        s5 = 300.0 / r['sec_per_step']
        print(f"  {r['N']:>4} {r['cells']/1e6:>7.3f} {r['total_mb']:>9.1f} "
              f"{r['bytes_per_cell']:>7.0f} {s3:>11.0f} {s5:>11.0f}")

    # Memory split (validates the 'CPML psi arrays are ~half the footprint' note)
    rlast = rows[-1]
    frac = rlast['cpml_mb'] / rlast['total_mb'] * 100.0
    print("\n" + "-" * 74)
    print(f"Memory split at N={rlast['N']}: "
          f"grid {rlast['grid_mb']:.1f} MB + CPML psi {rlast['cpml_mb']:.1f} MB "
          f"= {rlast['total_mb']:.1f} MB")
    print(f"  CPML psi arrays are {frac:.0f}% of the footprint — now allocated as "
          f"boundary slabs (compressed along each derivative axis to the active "
          f"PML cells), not full volume.")

    _plot(rows)
    print("=" * 74)


def _plot(rows):
    N = np.array([r['N'] for r in rows])
    mcell = np.array([r['cells'] for r in rows]) / 1e6
    us = np.array([r['us_per_cellstep'] for r in rows])
    mem = np.array([r['total_mb'] for r in rows])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle('Wavesim 3D profile (NumPy, CPML all 6 faces)', fontsize=13)

    a1.plot(mcell, us, 'o-', color='C0')
    a1.set_xlabel('grid size (Mcell)'); a1.set_ylabel('us / cell-step')
    a1.set_title('Throughput (lower = faster)')
    a1.grid(True, alpha=0.3)
    for x, y, n in zip(mcell, us, N):
        a1.annotate(f'{n}', (x, y), fontsize=7, xytext=(3, 3),
                    textcoords='offset points')

    a2.plot(mcell, mem, 's-', color='C3', label='measured array bytes')
    a2.set_xlabel('grid size (Mcell)'); a2.set_ylabel('memory (MB)')
    a2.set_title('Memory footprint (grid + CPML psi)')
    a2.grid(True, alpha=0.3); a2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'profile_3d.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"\nProfile plot saved to: {out}")


if __name__ == '__main__':
    main()
