"""
loss.py — electric conductivity: the two-coefficient E update.

Scope: **lossy dielectrics only.** A finite ``sigma`` here models a material
whose conduction current is a perturbation on its displacement current — FR4 and
other tan δ substrates, silicon, seawater, resistive films, lumped resistors.
It is *not* a way to model metal; see "What this is not" below.

The coefficients
----------------
Ampere's law with a conduction term, ``ε ∂E/∂t = ∇×H − σE``, is discretised with
the conduction term averaged over the two time levels it straddles (Taflove
§3.6.3) — E lives at integer steps, so ``σE`` at ``n+½`` is ``σ(E^{n+1}+E^n)/2``.
Solving for ``E^{n+1}`` turns the one-coefficient lossless update into two:

    E += (Δt/ε₀ε)·curl H          ->          E = Ca·E + Cb·curl H

    k  = σΔt / (2 ε₀ ε)
    Ca = (1 − k) / (1 + k)
    Cb = (Δt/ε₀ε) / (1 + k)

``Ca`` is the (1,1) Padé approximant of ``exp(−2k) = exp(−Δt/τ)`` for the
dielectric relaxation time ``τ = ε₀ε/σ``, so the damping is second-order accurate
in Δt like the rest of the scheme, and ``|Ca| < 1`` for every ``k > 0``: the loss
term is unconditionally stable and **does not change the CFL limit**. What it
does change is accuracy, sharply, once ``k > 1`` — see the warning below.

Why the H update is untouched
-----------------------------
No magnetic loss. ``σ_m`` would damp H by the mirror-image coefficients, but a
lossy *dielectric* has none, and adding an unused pair of arrays would cost the
H kernel its bandwidth for nothing.

Exactness at σ = 0
------------------
``Cb`` is built from ``base = dt / (EPS0 * eps)`` — character for character the
expression :mod:`wavesim.update` uses — and divided by ``1 + k``. At ``σ = 0``,
``k`` is exactly ``0.0``, so ``Ca`` is exactly ``1.0`` and ``Cb`` is exactly
``base``: a grid carrying all-zero σ arrays steps bit-identically to one carrying
none. That is a stronger guarantee than the conformal-PEC path offers (which is
exact only in the *absent* case), and it holds in any dtype because the
coefficients are built in the arrays' own precision rather than promoted to
float64 and cast back.

The absent case is of course still exact by dispatch: no σ arrays means
:func:`wavesim.update.update_E` never reaches this module.

What this is not
----------------
**A volumetric σ is not a model of a good conductor.** Copper's skin depth is
0.66 µm at 10 GHz, so a mesh that resolves it is out of reach; and at any usable
cell size ``k`` runs to ~10⁵–10⁶, which puts ``Ca`` a hair above ``−1``. The
update stays stable — but it alternates sign every step and bleeds off over ~10⁵
steps, where the physical field dies inside one cell. The result looks like a
run and is not one. Good conductors belong in ``pec_mask`` (or, eventually, a
surface-impedance boundary condition on it), never here.

:func:`build_loss_coefficients` warns when any cell exceeds ``k > 1``, which is
exactly the boundary where ``Ca`` goes negative and the update stops resembling
a decay. Equivalently: Δt has outrun twice the material's relaxation time.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from wavesim.constants import EPS0
from wavesim.grid import FDTDGrid


@dataclass(frozen=True)
class LossCoefficients:
    """The per-component ``(Ca, Cb)`` pair of the lossy E update.

    ``Ca_x`` multiplies the old ``Ex`` and ``Cb_x`` the curl of H, and likewise
    for y/z. Both have the grid's shape and dtype. Built by
    :func:`build_loss_coefficients`, cached by :func:`loss_coefficients`.
    """
    Ca_x: np.ndarray
    Cb_x: np.ndarray
    Ca_y: np.ndarray
    Cb_y: np.ndarray
    Ca_z: np.ndarray
    Cb_z: np.ndarray


def build_loss_coefficients(grid: FDTDGrid) -> LossCoefficients:
    """Compute ``(Ca, Cb)`` for all three components from σ, ε and dt.

    Three full-volume array pairs, so this is not something to do per step —
    go through :func:`loss_coefficients`, which caches.
    """
    dt = grid.dt
    out = {}
    worst = 0.0
    for c in ('x', 'y', 'z'):
        eps = getattr(grid, 'eps_' + c)
        sigma = getattr(grid, 'sigma_' + c)
        # Exactly update.py's lossless coefficient, so k == 0 reproduces it.
        base = dt / (EPS0 * eps)
        k = 0.5 * sigma * base
        worst = max(worst, float(np.max(k)) if k.size else 0.0)
        out['Ca_' + c] = (1.0 - k) / (1.0 + k)
        out['Cb_' + c] = base / (1.0 + k)

    if worst > 1.0:
        warnings.warn(
            f"lossy dielectric: sigma*dt/(2*eps) reaches {worst:.3g} > 1, so Ca "
            f"is negative and the field alternates sign each step instead of "
            f"decaying. The timestep has outrun twice the dielectric relaxation "
            f"time eps/sigma. This is the regime a metal-like sigma lands in, "
            f"and wavesim models good conductors as PEC (pec_mask), not as a "
            f"volumetric sigma — see wavesim.loss.",
            RuntimeWarning, stacklevel=3)
    return LossCoefficients(**out)


def loss_coefficients(grid: FDTDGrid):
    """Cached :class:`LossCoefficients` for ``grid``, or ``None`` if lossless.

    Cached on the grid and keyed on the *identity* of the three σ and three ε
    arrays plus the value of ``dt`` — the same scheme (and the same caveat) as
    :func:`wavesim.pec.conformal_geometry`: replacing an array invalidates the
    cache automatically, mutating one **in place** does not, so clear
    ``grid._loss_cache`` yourself if you need to do that mid-run.

    ``dt`` is in the key because it is not a property of the material and callers
    do change it on a copied grid — :func:`wavesim.stability._curl_curl_operator`
    sets ``dt = 1`` on a shallow copy, which would otherwise inherit coefficients
    built for the real timestep.
    """
    if not grid.is_lossy:
        return None

    arrays = (grid.eps_x, grid.eps_y, grid.eps_z,
              grid.sigma_x, grid.sigma_y, grid.sigma_z)
    dt = float(grid.dt)

    cache = getattr(grid, '_loss_cache', None)
    if (cache is None or cache[1] != dt
            or any(a is not b for a, b in zip(cache[0], arrays))):
        cache = (arrays, dt, build_loss_coefficients(grid))
        grid._loss_cache = cache
    return cache[2]
