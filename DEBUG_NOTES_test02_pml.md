# Debug Notes — CPML / `pml.py` (Test 02, free-space propagation)

**Date:** 2026-06-11
**Subsystem:** `fdtd/pml.py` (Convolutional PML, Roden–Gedney)
**Test:** `tests/test_02_free_space.py` — Gaussian Ez pulse in 2D vacuum (200×200, `Nz=1`)
**Status:** ✅ All four validations pass; both reported anomalies explained and resolved.

---

## Summary

`pml.py` was rewritten from scratch to make Test 02 pass. Two distinct bugs were
found and fixed, and one apparent "issue" was shown to be normal, expected
behaviour.

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | PML did not absorb; energy never decayed (Validation 3 failed) | CPML correction coefficient omitted `MU0` / `EPS0` | Use `dt/(MU0·mu)` and `dt/(EPS0·eps)`, matching `update.py` |
| 2 | Reflections from x0/y0 walls phase-reversed vs x1/y1 walls | Staggered-grid PML profile sampled at `(i+½)` instead of `(i−½)` | Staggered coordinate = `(i−0.5)·ds` |
| — | Faint residual reflections still visible in the animation | Inherent finite-PML residual (≈ −67 dB), amplified by global colour scale | None needed (already excellent) |

---

## Bug 1 — Missing `MU0` / `EPS0` in the CPML correction (no absorption)

### What happened
The main field updates in `fdtd/update.py` use **physical** coefficients, because
the material arrays store *relative* permittivity/permeability:

```python
Hx -= (dt / (MU0  * mu_x))  * (curl E)      # update_H
Ex += (dt / (EPS0 * eps_x)) * (curl H)      # update_E
```

The CPML convolution correction is part of that *same* curl, so it must use the
*identical* coefficient. The original `pml.py` applied the correction with
`dt/mu_x` and `dt/eps_x` — dropping `MU0` and `EPS0`. In vacuum
(`mu_x = eps_x = 1`) this makes the correction smaller by a factor of
`MU0 ≈ 1.26e-6` (H) and `EPS0 ≈ 8.85e-12` (E).

### Consequence
The convolution term `ψ` was effectively never added back into the curl, so the
PML behaved as if absent. Outgoing waves reflected off the domain edge, total
energy never decayed below 10 % of peak, and Validation 3 failed.

### Fix
Apply the correction with the same physical coefficient as `update.py`:
`dt/(MU0·mu)` for H corrections and `dt/(EPS0·eps)` for E corrections.

---

## Bug 2 — Phase-reversed reflections (staggered-grid coordinate sign)

### What happened
A phase flip between *opposite* walls (x0 vs x1, y0 vs y1) is the signature of an
**anti-symmetric** error — something offset by `+Δ` at one end and effectively
`−Δ` at the other.

`update.py` advances the E-fields with **backward** differences (this is the
convention stated in the plan):

```
Ez[i] += coef * ( (Hy[i] - Hy[i-1])/dx - (Hx[i] - Hx[i-1])/dy )
```

`Hy` lives at integer-x, so `Hy[i] − Hy[i-1]` is physically centred at
`x = (i − ½)·dx`. The CPML absorption coefficient that multiplies this term must
therefore be sampled at `(i − ½)·dx`. The original profile sampled the
**staggered (E-grid)** coordinate at `(i + ½)·dx` — a full cell off, and in
*opposite directions* relative to the two walls (toward the interior on the low
wall, toward the boundary on the high wall). That is exactly what flips the
reflection phase between opposite walls.

The non-staggered (H-grid) coordinate `i·ds` was already correct: `Hz` is driven
by `Ey[i+1] − Ey[i]`, centred at `x = i·dx`.

### Fix
In `_calc_profile_1d`, the staggered branch uses `coord = (i − 0.5)·ds`
(was `(i + 0.5)·ds`). The non-staggered branch is unchanged (`i·ds`).

### Evidence (reference-subtraction measurement)
To isolate *only* the PML reflection, the same run was repeated on an oversized
reference grid whose walls are far enough away that, within the capture window,
no wall reflection has reached the measurement region. `reflection =
small_grid − aligned_reference`. The signed mirror-correlation of that reflection
field, in a band just inside each PML, captured at the moment the first
reflection forms (step 240):

| Staggered coord | x0 ↔ x1 correlation | y0 ↔ y1 correlation |
|-----------------|--------------------:|--------------------:|
| `(i + ½)` (old, buggy) | **−0.982** (phase-reversed) | **−0.982** |
| `(i − ½)` (fixed)      | **+1.000** (in-phase)       | **+1.000** |

`+1` = symmetric/in-phase (correct); `−1` = phase-reversed (the reported bug).

---

## Non-issue — the faint residual reflections

The residual reflected back into the interior is **≈ −67 dB** relative to the
incident wavefront (measured late-time `|Ez|` residual / incident amplitude), and
total energy decays to `< 1e-4` of peak. This is a healthy 10-cell CPML; a small
residual is inherent to any finite PML.

It *looks* visible in `test_02_animation.gif` only because the animation uses a
single **global** colour scale (`vmax` = peak over all frames). Once the main
pulse is absorbed, the autoscaled colormap stretches even a −60 dB residual into a
visible colour. It is not an energy leak.

**If smaller residual is ever wanted:** increase `d_pml` (e.g. 12–16) or raise the
`σ_max` factor in `_calc_profile_1d`. Not necessary for correct physics.

---

## Final Test 02 results (after both fixes)

```
Validation 1 — arrival symmetry : spread 0 steps      (≤ 2)      PASS
Validation 2 — arrival vs r/c   : error 5.2 steps     (≤ 15)     PASS
Validation 3 — PML absorption   : final/peak 0.0000   (< 0.10)   PASS
Validation 4 — stability        : growth 0.46         (≤ 2.0)    PASS
```

(The ~5-step early arrival in Validation 2 is FDTD numerical dispersion — phase
velocity slightly above `c` on a coarse grid — not a bug.)

---

## Conventions to preserve (so this does not regress)

These three facts are coupled. If `update.py`'s differencing is ever changed,
`pml.py` must change with it.

1. **Material arrays are relative** (`eps_r`, `mu_r`). Every curl coefficient —
   in `update.py` *and* `pml.py` — carries `MU0` (H) or `EPS0` (E).
2. **E updates use backward differences, H updates use forward differences**
   (per the plan). Therefore:
   - E-field CPML profile samples the **staggered** coordinate `(i − ½)·ds`.
   - H-field CPML profile samples the **integer** coordinate `i·ds`.
3. **CPML is additive on top of the interior update.** The loop order is
   `update_H → update_H_pml → update_E → update_E_pml`; the `_pml` functions only
   add the `ψ` correction (with `κ = 1`, the `(1/κ)·∂F` part is already in the
   interior update).

---

## How to reproduce the diagnostics

- Full test: `python tests/test_02_free_space.py`
  (Windows console: set `PYTHONIOENCODING=utf-8` first, or the `≤` glyph in the
  output crashes the cp1252 console — a printing quirk, not a code bug.)
- Environment: the dedicated conda env `fdtd`
  (`C:\Users\itais\miniconda3\envs\fdtd\python.exe`), which has NumPy/Matplotlib;
  the conda `base` env does not.
- The reflection phase-symmetry was measured with a temporary reference-subtraction
  script (oversized reference grid, capture at step 240, signed mirror correlation
  of `small − reference` in a band inside each PML). It was removed after use; the
  method is described in the *Bug 2 → Evidence* section above.
