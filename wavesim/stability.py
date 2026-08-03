"""
stability.py — will this conformal grid actually run? (conformal-PEC plan, S7)

The Dey–Mittra small-cut threshold (S4, :func:`wavesim.pec.build_conformal_geometry`)
bounds the ``1/A_open`` coefficient, but the default of 0.4 is **not safe by
itself**: the plan's reference coax diverges at a transverse cell of 0.25 mm with
no sources and no ports at all, while the *finer* 0.1875 mm mesh is fine.
Stability is not monotone in resolution, so no amount of testing at one mesh
licenses another and a user cannot be asked to guess. This module measures it
instead.

Two independent measurements, and they agree
--------------------------------------------

:func:`probe_growth` — seed a random field, step the **real** scheme with no
sources, and fit the per-step amplification. Cheap (a few hundred steps), uses
the production kernels, and is what :class:`~wavesim.simulation.Simulation` runs
by default on a conformal grid.

:func:`max_stable_dt` — the definition. The leapfrog is stable iff
``dt² λ_max ≤ 4`` for ``λ_max`` of the discrete curl-curl operator
``M_ε⁻¹ Cᵀ M_ν C``, which :func:`max_stable_dt` obtains by Lanczos on the actual
``update_H``/``update_E`` pair, so it can never disagree with the kernels about
the geometry. Accurate but expensive — minutes on a 10⁶-cell grid.

Measured on the reference coax (``tests/reference_coax.py``), against the
recorded stable/diverges outcomes in the plan's S4 table:

| cell (mm) | threshold | ``max_stable_dt``/dt | free-run growth | outcome |
|---|---|---|---|---|
| 0.5000 | 0.4 | 1.00585 | 1.000 | stable |
| 0.5000 | 0.5 | 1.01234 | 1.000 | stable |
| 0.2500 | 0.4 | **0.99795** | **1.1358** | **diverges** |
| 0.2500 | 0.5 | 1.01072 | 1.000 | stable |

Both call all four correctly, and on the failing case they agree *numerically*:
a margin of 0.99795 predicts a per-step amplification of 1.13657, and the free
run measures 1.13581 (inverted through :func:`_margin_from_growth`: 0.99798).
That is one number arrived at from an eigenvalue and from a time-domain run.

Note how little headroom a working conformal run has: **1.006** at 0.5 mm. The
CFL of a cut-cell model is set by its slivers, not by its mesh, which is the
whole of R2 in one number.

What does *not* work
--------------------

* **The smallest open fraction does not predict it**, and W3's warning points at
  a number that cannot. Once a face falls below the threshold it is clamped to
  ``threshold·A_full`` *exactly*, so a 0.0015 sliver and a 0.0044 one get an
  identical coefficient — the difference between the 0.25 mm and 0.375 mm meshes
  is the open *edge* lengths around those clamped faces, which are not clamped.
* **A Gershgorin bound is too loose.** It is a genuine upper bound on ``λ_max``
  and costs one array pass, but its overshoot is not constant: 1.34× on a uniform
  vacuum mesh against 1.51× on the cut coax. It cannot resolve margins that live
  within 1% of unity.
"""

import copy
import math
import warnings
from dataclasses import dataclass

import numpy as np

from wavesim.constants import EPS0, MU0
from wavesim.grid import FDTDGrid
from wavesim.pec import apply_pec_faces, apply_pec_mask


# Probe defaults. 500 steps at chunk 50 gives ten samples; the log-linear fit
# then reads |rate - 1| <= 7.4e-6 across stable cut-cell, vacuum and PEC-walled
# grids, so the 5e-5 trigger has ~7x headroom. An instability weaker than the
# trigger is not detected — raise ``steps`` if a very long run must be
# certified, since what matters is growth over the run, not per step.
PROBE_STEPS = 500
PROBE_CHUNK = 50
GROWTH_TRIGGER = 5e-5

# Auto-raise ladder. The accuracy cost is real (V1 read +0.21% at 0.4 against
# +1.03% at 0.5), so climb in coarse steps and stop early rather than search for
# a tight optimum that buys nothing.
LADDER_STEP = 0.1
MAX_THRESHOLD = 0.9


@dataclass(frozen=True)
class StabilityProbe:
    """What :func:`probe_growth` measured.

    ``growth`` is the per-step amplification of the seeded free run: 1.0 for a
    lossless scheme that conserves its energy, ``inf`` if the fields overflowed.
    ``margin`` is the implied ``dt_max/dt``, which the growth rate pins **only
    when the run is unstable** — a stable leapfrog amplifies by exactly 1
    whatever its headroom, so a stable probe leaves ``margin`` at ``None``. Use
    :func:`stability_margin` when the actual number is wanted.
    """
    growth: float
    stable: bool
    steps: int
    margin: float = None
    norms: tuple = ()


# ====================================================================== #
# The fast probe — run the real scheme and watch
# ====================================================================== #

def probe_growth(grid: FDTDGrid, *, steps: int = PROBE_STEPS,
                 chunk: int = PROBE_CHUNK, pec_faces: tuple = (),
                 cpml=None, backend: str = 'numpy', seed: int = 0,
                 trigger: float = GROWTH_TRIGGER) -> StabilityProbe:
    """Seed noise, step the scheme with no sources, and fit the growth rate.

    The grid is **not** touched: the fields are replaced on a shallow copy, so
    the material and geometry arrays are shared but ``Ex..Hz`` are the probe's
    own. ``cpml`` is deep-copied for the same reason — its ψ recursion carries
    state.

    A random E field excites every mode of the discretisation at once, which is
    the point: a cut-cell instability lives on a handful of faces and a physical
    source may not reach it for thousands of steps, while noise reaches it
    immediately. The run is stopped as soon as the norm has grown 10⁶×, so
    detecting an unstable grid is *cheaper* than certifying a stable one.

    Two details that the noise floor caught, both of which would have shown up
    as slow "growth" on a perfectly good grid:

    * **The seed is pinned on the edges the kernels never update.** ``update_E``
      writes ``Ex[:, 1:, 1:]`` and its cyclic partners, so the low-index planes
      keep whatever they were given — and a frozen E edge feeds Faraday every
      step, ramping H linearly forever. Seeding them left the total energy 9×
      higher after 500 steps on an open grid; pinning them leaves it flat to
      3.8%.
    * **The tracked quantity is the energy** ``ε₀‖E‖² + μ₀‖H‖²``, not ``‖E‖``.
      The two exchange energy every step, so ``‖E‖`` alone wobbles and biases
      the fit an order of magnitude higher (+2e-5 per step against +7e-6).

    Growth is fitted by least squares on the log over the samples after the
    first (the seed is not a physical field and the first chunk carries its
    transient). Stable grids read 1.000000 ± 7.4e-6; the reference coax at
    0.25 mm and threshold 0.4 reads 1.135809.
    """
    from wavesim.simulation import Simulation      # circular at module scope

    if chunk < 1 or steps < 2 * chunk:
        raise ValueError(f"need steps >= 2*chunk >= 2, got {steps}/{chunk}")

    probe = copy.copy(grid)
    rng = np.random.default_rng(seed)
    for name in ('Ex', 'Ey', 'Ez'):
        setattr(probe, name, rng.standard_normal(getattr(grid, name).shape)
                .astype(grid.Ex.dtype))
    for name in ('Hx', 'Hy', 'Hz'):
        setattr(probe, name, np.zeros_like(getattr(grid, name)))
    _pin_unupdated_edges(probe)
    probe.time_step = 0

    # conformal_stability='off': this *is* the stability check, and the inner
    # Simulation must not recurse into it.
    sim = Simulation(probe, cpml=copy.deepcopy(cpml), pec_faces=pec_faces,
                     backend=backend, conformal_stability='off')

    norms = []
    n_done = 0
    with np.errstate(over='ignore', invalid='ignore'):
        while n_done < steps:
            sim.run(chunk)
            n_done += chunk
            norms.append(_energy_norm(probe))
            if not math.isfinite(norms[-1]) or norms[-1] > 1e6 * norms[0]:
                break

    growth = _fit_growth(norms, chunk)
    stable = growth <= 1.0 + trigger
    return StabilityProbe(
        growth=growth, stable=stable, steps=n_done, norms=tuple(norms),
        margin=None if stable else _margin_from_growth(growth))


def _pin_unupdated_edges(grid: FDTDGrid) -> None:
    """Zero the E edges no update ever writes, so they cannot drive anything.

    ``update_E`` writes ``Ex[:, 1:, 1:]``, ``Ey[1:, :, 1:]``, ``Ez[1:, 1:, :]``
    (and drops the ``k`` restriction on the ``Nz == 1`` fast path). Everything
    else keeps its initial value for the whole run, which is harmless when that
    value is zero and a permanent Faraday source when it is not.
    """
    grid.Ex[:, 0, :] = 0.0
    grid.Ey[0, :, :] = 0.0
    grid.Ez[0, :, :] = 0.0
    grid.Ez[:, 0, :] = 0.0
    if grid.Nz > 1:
        grid.Ex[:, :, 0] = 0.0
        grid.Ey[:, :, 0] = 0.0


def _energy_norm(grid: FDTDGrid) -> float:
    """``sqrt(ε₀‖E‖² + μ₀‖H‖²)`` — flat under a stable lossless leapfrog.

    Not the scheme's exact invariant (that pairs H at two half steps), but it is
    free of the E↔H exchange that makes ``‖E‖`` alone wobble, and it is what
    keeps the growth fit's noise floor at 7e-6 per step.
    """
    e = sum(float(np.dot(a.ravel(), a.ravel()))
            for a in (grid.Ex, grid.Ey, grid.Ez))
    h = sum(float(np.dot(a.ravel(), a.ravel()))
            for a in (grid.Hx, grid.Hy, grid.Hz))
    total = EPS0 * e + MU0 * h
    return math.sqrt(total) if math.isfinite(total) and total >= 0 else math.inf


def _fit_growth(norms, chunk: int) -> float:
    """Per-step amplification from the sampled norms (least squares on the log)."""
    n = np.asarray(norms, dtype=float)
    if not np.all(np.isfinite(n)) or np.any(n <= 0.0):
        return math.inf
    fit = n[1:] if len(n) > 2 else n            # drop the seed's transient
    if len(fit) < 2:
        return float((n[-1] / n[0]) ** (1.0 / (chunk * (len(n) - 1))))
    x = np.arange(len(fit), dtype=float) * chunk
    slope = np.polyfit(x, np.log(fit), 1)[0]
    return float(math.exp(slope))


def _margin_from_growth(growth: float) -> float:
    """``dt_max/dt`` implied by a measured per-step amplification.

    The leapfrog's amplification factor satisfies ``z + 1/z = 2 - dt²λ``. Past
    the limit ``z`` is real and negative, so ``|z| = g`` inverts to
    ``dt²λ/4 = (2 + g + 1/g)/4`` and the margin is its inverse square root.
    Only meaningful for ``g > 1``: below the limit ``|z| = 1`` for every stable
    mode and the run tells you nothing about how much room is left.
    """
    if not math.isfinite(growth):
        return 0.0
    if growth <= 1.0:
        return math.nan
    return 1.0 / math.sqrt((2.0 + growth + 1.0 / growth) / 4.0)


# ====================================================================== #
# The definition — lambda_max of the discrete curl-curl
# ====================================================================== #

def max_stable_dt(grid: FDTDGrid, *, pec_faces: tuple = (), tol: float = 1e-4,
                  seed: int = 0) -> float:
    """Largest timestep this grid's spatial operator supports, in seconds.

    ``dt_max = 2/√λ_max`` with ``λ_max`` the top eigenvalue of
    ``M_ε⁻¹ Cᵀ M_ν C``, which is symmetric and positive semi-definite in the
    ``M_ε`` inner product, so Lanczos applies. The operator is never assembled:
    one matrix-vector product is ``update_H`` on a zeroed H followed by
    ``update_E`` on a zeroed E, at ``dt = 1``, with the run's PEC constraints
    projected out. It therefore measures the *kernels*, including every clamped
    face and open edge length, and cannot drift from them.

    On uniform vacuum this returns the textbook ``h/(c√3)`` to within 0.6%; the
    remainder is the finite domain, whose frozen outermost slabs put λ_max just
    below the infinite-grid ``12c²/h²``.

    **Expensive.** The top of the Yee spectrum is a dense cluster, so Lanczos
    needs thousands of products: ~40 s on a 36×36×100 grid and ~8 minutes on
    72×72×100. :func:`probe_growth` answers the yes/no question in seconds; this
    is for when the *margin* itself is the quantity of interest.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    op, size = _curl_curl_operator(grid, pec_faces)
    rng = np.random.default_rng(seed)
    lam = eigsh(LinearOperator((size, size), matvec=op, dtype=np.float64),
                k=1, which='LA', tol=tol, return_eigenvectors=False,
                v0=rng.standard_normal(size))
    lam_max = float(lam[-1])
    return math.inf if lam_max <= 0.0 else 2.0 / math.sqrt(lam_max)


def stability_margin(grid: FDTDGrid, **kwargs) -> float:
    """``max_stable_dt(grid)/grid.dt`` — above 1 is stable, below 1 diverges."""
    return max_stable_dt(grid, **kwargs) / grid.dt


def _curl_curl_operator(grid: FDTDGrid, pec_faces: tuple):
    """``(matvec, size)`` for the curl-curl operator, built out of the kernels."""
    from wavesim.update import update_E, update_H

    work = copy.copy(grid)
    for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        setattr(work, name, np.zeros(getattr(grid, name).shape, dtype=float))
    work.dt = 1.0
    shape = work.Ex.shape
    size = 3 * int(np.prod(shape))

    def project():
        if pec_faces:
            apply_pec_faces(work, faces=pec_faces)
        apply_pec_mask(work)
        # The never-updated edges are zero in every output, so leaving them live
        # in the input would make the matrix non-symmetric — Lanczos assumes it
        # is not. Projecting them out both ways restricts the operator to the
        # subspace the scheme actually moves.
        _pin_unupdated_edges(work)

    def matvec(vec):
        v = np.asarray(vec, dtype=float).reshape(3, *shape)
        work.Ex[...], work.Ey[...], work.Ez[...] = v[0], v[1], v[2]
        project()
        work.Hx[...] = work.Hy[...] = work.Hz[...] = 0.0
        update_H(work)
        work.Ex[...] = work.Ey[...] = work.Ez[...] = 0.0
        update_E(work)
        project()
        # update_H writes -A_H E and update_E writes +A_E H, so the pair lands
        # on -P v.
        return -np.concatenate([work.Ex.ravel(), work.Ey.ravel(),
                                work.Ez.ravel()])

    return matvec, size


# ====================================================================== #
# The remedy — raise the threshold until the probe is quiet
# ====================================================================== #

def safe_area_threshold(grid: FDTDGrid, *, ladder_step: float = LADDER_STEP,
                        max_threshold: float = MAX_THRESHOLD,
                        **probe_kw) -> tuple:
    """Smallest clamp threshold at or above the grid's own that probes stable.

    Returns ``(threshold, first_probe)`` — the probe of the threshold the grid
    *arrived* with, so a caller can report what was wrong with it rather than
    the uninformative 1.0 the passing rung reports. The grid is left exactly as
    it was found, including its threshold: deciding is separate from applying
    (see :func:`ensure_stable_threshold`).

    Climbing the threshold is S7 option 2 and it is a real trade, not a free
    fix — over-clamping costs accuracy (V1: +0.21% at 0.4, +1.03% at 0.5). The
    ladder is coarse for that reason: it stops at the first rung that works
    rather than bisecting toward a tight bound whose extra precision buys
    nothing measurable.
    """
    if not grid.is_conformal:
        raise ValueError("safe_area_threshold needs a conformal grid; this one "
                         "carries no open-fraction arrays.")

    original = float(grid.conformal_area_threshold)
    candidate = original
    first = None
    try:
        while True:
            grid.conformal_area_threshold = candidate
            probe = probe_growth(grid, **probe_kw)
            first = first or probe
            if probe.stable:
                return candidate, first
            if candidate >= max_threshold:
                raise RuntimeError(
                    f"conformal grid is unstable at every clamp threshold up to "
                    f"{max_threshold:.2f} (growth {probe.growth:.4f}/step, "
                    f"implied dt_max/dt {probe.margin:.5f}). Clamping cannot fix "
                    f"this geometry — merge the cut cells (BCK), move the "
                    f"conductor off the sliver, or change the mesh.")
            candidate = min(round(candidate + ladder_step, 6), max_threshold)
    finally:
        grid.conformal_area_threshold = original


def ensure_stable_threshold(grid: FDTDGrid, **kwargs) -> float:
    """Raise ``grid.conformal_area_threshold`` in place until the run is stable.

    Returns the threshold in force afterwards — **read this, not the one that
    was asked for**, when recording what a run did. A staircase grid is returned
    unchanged and never probed, so nothing that does not carry cut cells pays
    for this.

    A no-op whenever the requested threshold already works, which is the normal
    case: the probe runs, agrees, and the geometry is untouched.
    """
    if not grid.is_conformal:
        return float(grid.conformal_area_threshold)

    original = float(grid.conformal_area_threshold)
    threshold, probe = safe_area_threshold(grid, **kwargs)
    grid.conformal_area_threshold = threshold
    if threshold != original:
        warnings.warn(
            f"conformal PEC: clamp threshold raised {original:.2f} -> "
            f"{threshold:.2f}. The run diverges at {original:.2f} — the seeded "
            f"free run grows {probe.growth:.4g} per step, implying "
            f"dt_max/dt = {probe.margin:.5f}. The smallest cut faces are now "
            f"clamped harder, which costs accuracy; cut-cell merging (BCK) "
            f"would remove the sliver instead of clamping around it.",
            RuntimeWarning, stacklevel=2)
    return threshold
