"""
Test 08 — Numba backend parity (ROADMAP §3).

The Numba backend (wavesim/backend_numba.py) reimplements the four hot solver
functions as multithreaded JIT kernels. This test is the "nothing is broken"
guarantee: it runs the *same* physical setup through both the validated NumPy
reference and the Numba backend and asserts the field arrays agree to within
floating-point rounding.

Because both backends do identical per-cell float64 arithmetic with no parallel
reductions, the result is expected to be **bit-identical** (max|diff| == 0). We
assert a tiny tolerance rather than exact equality only to stay robust to future
fast-math / reassociation changes.

Two configurations exercise both code paths:
  * Check 1 — 2D slice (Nz=1): the z-derivative fast path.
  * Check 2 — full 3D (Nz>1): the volumetric path + z-face CPML.

Run:
    python tests/test_08_numba_parity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from wavesim.grid import create_grid
from wavesim.materials import set_vacuum
from wavesim.pml import init_cpml
from wavesim.pec import apply_pec_mask
from wavesim.sources import GaussianSource, gaussian_pulse

import wavesim.update as np_update
import wavesim.pml as np_pml
import wavesim.backend_numba as nb

FIELDS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
RTOL = 1e-12
ATOL = 1e-12


def _run(uH, uE, uHp, uEp, N, Nz, n_steps):
    """Run the canonical loop with a given set of update functions; return the
    final grid plus the recorded Ez time-series at a probe point."""
    grid = create_grid(Nx=N, Ny=N, Nz=Nz, dx=1.0e-3)
    grid = set_vacuum(grid)
    cpml = init_cpml(grid, d_pml=8)
    src = GaussianSource(t0=20 * grid.dt, width=6 * grid.dt)
    ic, kc = N // 2, Nz // 2
    probe = ic // 2 + 1                       # an off-centre cell, inside the PML-free core
    series = np.empty(n_steps)
    for n in range(n_steps):
        grid = uH(grid)
        grid, cpml = uHp(grid, cpml)
        grid = uE(grid)
        grid, cpml = uEp(grid, cpml)
        grid = apply_pec_mask(grid)
        grid.Ez[ic, ic, kc] += gaussian_pulse(src, n * grid.dt)
        grid.time_step += 1
        series[n] = grid.Ez[probe, probe, kc]
    return grid, series


def _compare(label, N, Nz, n_steps):
    ga, sa = _run(np_update.update_H, np_update.update_E,
                  np_pml.update_H_pml, np_pml.update_E_pml, N, Nz, n_steps)
    gb, sb = _run(nb.update_H, nb.update_E,
                  nb.update_H_pml, nb.update_E_pml, N, Nz, n_steps)

    field_err = max(np.max(np.abs(getattr(ga, f) - getattr(gb, f))) for f in FIELDS)
    field_scale = max(np.max(np.abs(getattr(ga, f))) for f in FIELDS)
    series_err = np.max(np.abs(sa - sb))

    print(f"  {label:12s} N={N:<3d} Nz={Nz:<3d} steps={n_steps}: "
          f"max|field diff|={field_err:.3e} (scale {field_scale:.3e}), "
          f"max|probe diff|={series_err:.3e}")

    for f in FIELDS:
        assert np.allclose(getattr(ga, f), getattr(gb, f), rtol=RTOL, atol=ATOL), \
            f"{label}: field {f} differs beyond tolerance (max {field_err:.3e})"
    assert np.allclose(sa, sb, rtol=RTOL, atol=ATOL), \
        f"{label}: probe time-series differs beyond tolerance"
    return field_err, series_err


def main():
    print("=" * 70)
    print("Test 08 — Numba backend parity vs NumPy reference")
    print("=" * 70)
    print("(first run pays a one-time Numba JIT compile cost)\n")

    print("Check 1 — 2D slice path (Nz=1):")
    _compare("2D slice", N=28, Nz=1, n_steps=150)

    print("\nCheck 2 — full 3D path (Nz>1):")
    _compare("full 3D", N=24, Nz=24, n_steps=150)

    print("\n" + "=" * 70)
    print("PASS — Numba backend reproduces the NumPy solver to float64 rounding.")
    print("=" * 70)


if __name__ == '__main__':
    main()
