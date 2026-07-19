"""
backend_cuda.py — CUDA (NVIDIA GPU) drop-in for the hot solver functions.

Relationship to the other backends
-----------------------------------
This is a third backend alongside the pure-NumPy reference (update.py / pml.py)
and the CPU Numba backend (backend_numba.py). It exposes the same four
signature-compatible wrappers — ``update_H``, ``update_E``, ``update_H_pml``,
``update_E_pml`` — so ``Simulation(backend='cuda')`` swaps them in transparently.

Faithfulness
------------
Every kernel below is a 1:1 port of one ``@njit(parallel=True)`` loop nest in
backend_numba.py, which is itself a faithful port of the NumPy slice form. The
mapping is:

    numba:  for i in prange(Nx):  for j in range(...):  for k in range(...): <cell>
    cuda:   i, j, k = cuda.grid(3);  if <bounds>: <cell>

i.e. the outer ``prange`` and the sequential inner loops all become one thread
per cell, with the numba loop bounds turned into ``if`` guards. Signs,
staggering, physical coefficients, and the ``Nz>1`` / ``Nz==1`` branch structure
are identical, so results match the oracle to floating-point tolerance (not
bit-identical: the GPU may fuse/round differently, and the float32 path is a
deliberate lower-precision mode).

Parallel-write safety
---------------------
* Field kernels: thread (i,j,k) writes only Hx/Hy/Hz[i,j,k] (or Ex/Ey/Ez[i,j,k]);
  every cell is written by exactly one thread. Reads are from the *other* field
  (not mutated in the same kernel). Race-free without atomics.
* CPML kernels: each fuses a psi recursion with its field correction. A thread
  advances only its own psi cell and then applies a correction that depends on
  *that same* psi cell — no cross-thread dependency. The six H-CPML kernels read
  only E (+ their own psi) and write only H (+ their own psi); the six E-CPML
  kernels are the mirror. They are therefore mutually independent, and the only
  cross-kernel overlaps are commutative ``+=`` / ``-=`` accumulations onto shared
  corner cells — order-independent. Launch ordering on the default stream makes
  the whole sequence equivalent to the numba sequential form.

Precision
---------
Kernels are dtype-generic (Numba compiles one specialisation per field dtype).
The wrappers cast the scalar coefficients (dt/MU0, dt/EPS0, dx, dy, dz) and the
CPML (b, c) profiles to the field dtype so a float32 grid does genuine float32
arithmetic — important on consumer GPUs where float64 runs at a small fraction
of float32 throughput.

Transfers
---------
For now these wrappers copy the touched arrays host<->device on every call, so
``backend='cuda'`` is *correct* but transfer-bound (a validation / fallback
path, not yet the fast path). A device-resident field lifecycle that keeps the
arrays on the GPU across the whole time loop is the follow-up optimisation.
"""

import os

# Numba selects its CUDA binding at import time. The default (cuda.core) binding
# ships a native DLL that Windows Smart App Control blocks on this machine, so we
# force the legacy ctypes binding, which is unaffected. Harmless where SAC is off.
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "0")

import numpy as np
from numba import cuda

from wavesim.grid import FDTDGrid
from wavesim.pml import CPMLArrays
from wavesim.pec import build_pec_edge_masks
from wavesim.constants import MU0, EPS0

# Default 3D thread-block. 256 threads/block suits Turing (CC 7.5) occupancy.
_TPB = (8, 8, 4)


# ====================================================================== #
# Launch helper
# ====================================================================== #
def _launch(kernel, ext, args):
    """Launch ``kernel`` over a 3D index space ``ext`` (a tuple of extents),
    covering it with ``_TPB`` blocks. Skips the launch entirely if any extent is
    zero (e.g. an empty CPML slab), which would otherwise be an illegal 0-block
    grid."""
    ex, ey, ez = int(ext[0]), int(ext[1]), int(ext[2])
    if ex == 0 or ey == 0 or ez == 0:
        return
    bpg = ((ex + _TPB[0] - 1) // _TPB[0],
           (ey + _TPB[1] - 1) // _TPB[1],
           (ez + _TPB[2] - 1) // _TPB[2])
    kernel[bpg, _TPB](*args)


# ====================================================================== #
# Field updates (curl) — ports of _update_H / _update_E
# ====================================================================== #
@cuda.jit
def _k_update_H(Ex, Ey, Ez, Hx, Hy, Hz,
                mu_x, mu_y, mu_z, cH, dx, dy, dz, Nx, Ny, Nz):
    i, j, k = cuda.grid(3)
    if i >= Nx or j >= Ny or k >= Nz:
        return

    if Nz > 1:
        # Hx[:, :-1, :-1] -= coef * (dEz/dy - dEy/dz)
        if j < Ny - 1 and k < Nz - 1:
            dEz_dy = (Ez[i, j + 1, k] - Ez[i, j, k]) / dy
            dEy_dz = (Ey[i, j, k + 1] - Ey[i, j, k]) / dz
            Hx[i, j, k] -= (cH / mu_x[i, j, k]) * (dEz_dy - dEy_dz)
        # Hy[:-1, :, :-1] -= coef * (dEx/dz - dEz/dx)
        if i < Nx - 1 and k < Nz - 1:
            dEx_dz = (Ex[i, j, k + 1] - Ex[i, j, k]) / dz
            dEz_dx = (Ez[i + 1, j, k] - Ez[i, j, k]) / dx
            Hy[i, j, k] -= (cH / mu_y[i, j, k]) * (dEx_dz - dEz_dx)
    else:
        if j < Ny - 1:
            dEz_dy = (Ez[i, j + 1, 0] - Ez[i, j, 0]) / dy
            Hx[i, j, 0] -= (cH / mu_x[i, j, 0]) * dEz_dy
        if i < Nx - 1:
            dEz_dx = (Ez[i + 1, j, 0] - Ez[i, j, 0]) / dx
            Hy[i, j, 0] -= (cH / mu_y[i, j, 0]) * (-dEz_dx)

    # Hz[:-1, :-1, :] -= coef * (dEy/dx - dEx/dy)   (identical for all Nz)
    if i < Nx - 1 and j < Ny - 1:
        dEy_dx = (Ey[i + 1, j, k] - Ey[i, j, k]) / dx
        dEx_dy = (Ex[i, j + 1, k] - Ex[i, j, k]) / dy
        Hz[i, j, k] -= (cH / mu_z[i, j, k]) * (dEy_dx - dEx_dy)


@cuda.jit
def _k_update_E(Ex, Ey, Ez, Hx, Hy, Hz,
                eps_x, eps_y, eps_z, cE, dx, dy, dz, Nx, Ny, Nz):
    i, j, k = cuda.grid(3)
    if i >= Nx or j >= Ny or k >= Nz:
        return

    if Nz > 1:
        # Ex[:, 1:, 1:] += coef * (dHz/dy - dHy/dz)
        if j >= 1 and k >= 1:
            dHz_dy = (Hz[i, j, k] - Hz[i, j - 1, k]) / dy
            dHy_dz = (Hy[i, j, k] - Hy[i, j, k - 1]) / dz
            Ex[i, j, k] += (cE / eps_x[i, j, k]) * (dHz_dy - dHy_dz)
        # Ey[1:, :, 1:] += coef * (dHx/dz - dHz/dx)
        if i >= 1 and k >= 1:
            dHx_dz = (Hx[i, j, k] - Hx[i, j, k - 1]) / dz
            dHz_dx = (Hz[i, j, k] - Hz[i - 1, j, k]) / dx
            Ey[i, j, k] += (cE / eps_y[i, j, k]) * (dHx_dz - dHz_dx)
    else:
        if j >= 1:
            dHz_dy = (Hz[i, j, 0] - Hz[i, j - 1, 0]) / dy
            Ex[i, j, 0] += (cE / eps_x[i, j, 0]) * dHz_dy
        if i >= 1:
            dHz_dx = (Hz[i, j, 0] - Hz[i - 1, j, 0]) / dx
            Ey[i, j, 0] += (cE / eps_y[i, j, 0]) * (-dHz_dx)

    # Ez[1:, 1:, :] += coef * (dHy/dx - dHx/dy)   (identical for all Nz)
    if i >= 1 and j >= 1:
        dHy_dx = (Hy[i, j, k] - Hy[i - 1, j, k]) / dx
        dHx_dy = (Hx[i, j, k] - Hx[i, j - 1, k]) / dy
        Ez[i, j, k] += (cE / eps_z[i, j, k]) * (dHy_dx - dHx_dy)


# ====================================================================== #
# CPML H-field corrections — ports of _update_H_pml loop nests
# Each kernel fuses one psi recursion with its field correction.
# ====================================================================== #
@cuda.jit
def _k_HX_Y(Ez, Hx, mu_x, psi, b, c, sel, cH, dy, Nx, nP, Nz):
    """psi_Ez_y (y axis), correction -psi onto Hx."""
    i, p, k = cuda.grid(3)
    if i >= Nx or p >= nP or k >= Nz:
        return
    j = sel[p]
    dEz_dy = (Ez[i, j + 1, k] - Ez[i, j, k]) / dy
    psi[i, p, k] = b[p] * psi[i, p, k] + c[p] * dEz_dy
    if Nz > 1:
        if k < Nz - 1:
            Hx[i, j, k] -= (cH / mu_x[i, j, k]) * psi[i, p, k]
    else:
        Hx[i, j, 0] -= (cH / mu_x[i, j, 0]) * psi[i, p, 0]


@cuda.jit
def _k_HX_Z(Ey, Hx, mu_x, psi, b, c, sel, cH, dz, Nx, Ny, nP):
    """psi_Ey_z (z axis), correction +psi onto Hx (Nz>1 only)."""
    i, j, p = cuda.grid(3)
    if i >= Nx or j >= Ny or p >= nP:
        return
    k = sel[p]
    dEy_dz = (Ey[i, j, k + 1] - Ey[i, j, k]) / dz
    psi[i, j, p] = b[p] * psi[i, j, p] + c[p] * dEy_dz
    if j < Ny - 1:
        Hx[i, j, k] += (cH / mu_x[i, j, k]) * psi[i, j, p]


@cuda.jit
def _k_HY_X(Ez, Hy, mu_y, psi, b, c, sel, cH, dx, nP, Ny, Nz):
    """psi_Ez_x (x axis), correction +psi onto Hy."""
    p, j, k = cuda.grid(3)
    if p >= nP or j >= Ny or k >= Nz:
        return
    i = sel[p]
    dEz_dx = (Ez[i + 1, j, k] - Ez[i, j, k]) / dx
    psi[p, j, k] = b[p] * psi[p, j, k] + c[p] * dEz_dx
    if Nz > 1:
        if k < Nz - 1:
            Hy[i, j, k] += (cH / mu_y[i, j, k]) * psi[p, j, k]
    else:
        Hy[i, j, 0] += (cH / mu_y[i, j, 0]) * psi[p, j, 0]


@cuda.jit
def _k_HY_Z(Ex, Hy, mu_y, psi, b, c, sel, cH, dz, Nx, Ny, nP):
    """psi_Ex_z (z axis), correction -psi onto Hy (Nz>1 only)."""
    i, j, p = cuda.grid(3)
    if i >= Nx or j >= Ny or p >= nP:
        return
    k = sel[p]
    dEx_dz = (Ex[i, j, k + 1] - Ex[i, j, k]) / dz
    psi[i, j, p] = b[p] * psi[i, j, p] + c[p] * dEx_dz
    if i < Nx - 1:
        Hy[i, j, k] -= (cH / mu_y[i, j, k]) * psi[i, j, p]


@cuda.jit
def _k_HZ_X(Ey, Hz, mu_z, psi, b, c, sel, cH, dx, nP, Ny, Nz):
    """psi_Ey_x (x axis), correction -psi onto Hz."""
    p, j, k = cuda.grid(3)
    if p >= nP or j >= Ny or k >= Nz:
        return
    i = sel[p]
    dEy_dx = (Ey[i + 1, j, k] - Ey[i, j, k]) / dx
    psi[p, j, k] = b[p] * psi[p, j, k] + c[p] * dEy_dx
    if j < Ny - 1:
        Hz[i, j, k] -= (cH / mu_z[i, j, k]) * psi[p, j, k]


@cuda.jit
def _k_HZ_Y(Ex, Hz, mu_z, psi, b, c, sel, cH, dy, Nx, nP, Nz):
    """psi_Ex_y (y axis), correction +psi onto Hz."""
    i, p, k = cuda.grid(3)
    if i >= Nx or p >= nP or k >= Nz:
        return
    j = sel[p]
    dEx_dy = (Ex[i, j + 1, k] - Ex[i, j, k]) / dy
    psi[i, p, k] = b[p] * psi[i, p, k] + c[p] * dEx_dy
    if i < Nx - 1:
        Hz[i, j, k] += (cH / mu_z[i, j, k]) * psi[i, p, k]


# ====================================================================== #
# CPML E-field corrections — ports of _update_E_pml loop nests
# ====================================================================== #
@cuda.jit
def _k_EX_Y(Hz, Ex, eps_x, psi, b, c, sel, cE, dy, Nx, nP, Nz):
    """psi_Hz_y (y axis), correction +psi onto Ex."""
    i, p, k = cuda.grid(3)
    if i >= Nx or p >= nP or k >= Nz:
        return
    j = sel[p]
    dHz_dy = (Hz[i, j, k] - Hz[i, j - 1, k]) / dy
    psi[i, p, k] = b[p] * psi[i, p, k] + c[p] * dHz_dy
    if Nz > 1:
        if k >= 1:
            Ex[i, j, k] += (cE / eps_x[i, j, k]) * psi[i, p, k]
    else:
        Ex[i, j, 0] += (cE / eps_x[i, j, 0]) * psi[i, p, 0]


@cuda.jit
def _k_EX_Z(Hy, Ex, eps_x, psi, b, c, sel, cE, dz, Nx, Ny, nP):
    """psi_Hy_z (z axis), correction -psi onto Ex (Nz>1 only)."""
    i, j, p = cuda.grid(3)
    if i >= Nx or j >= Ny or p >= nP:
        return
    k = sel[p]
    dHy_dz = (Hy[i, j, k] - Hy[i, j, k - 1]) / dz
    psi[i, j, p] = b[p] * psi[i, j, p] + c[p] * dHy_dz
    if j >= 1:
        Ex[i, j, k] -= (cE / eps_x[i, j, k]) * psi[i, j, p]


@cuda.jit
def _k_EY_X(Hz, Ey, eps_y, psi, b, c, sel, cE, dx, nP, Ny, Nz):
    """psi_Hz_x (x axis), correction -psi onto Ey."""
    p, j, k = cuda.grid(3)
    if p >= nP or j >= Ny or k >= Nz:
        return
    i = sel[p]
    dHz_dx = (Hz[i, j, k] - Hz[i - 1, j, k]) / dx
    psi[p, j, k] = b[p] * psi[p, j, k] + c[p] * dHz_dx
    if Nz > 1:
        if k >= 1:
            Ey[i, j, k] -= (cE / eps_y[i, j, k]) * psi[p, j, k]
    else:
        Ey[i, j, 0] -= (cE / eps_y[i, j, 0]) * psi[p, j, 0]


@cuda.jit
def _k_EY_Z(Hx, Ey, eps_y, psi, b, c, sel, cE, dz, Nx, Ny, nP):
    """psi_Hx_z (z axis), correction +psi onto Ey (Nz>1 only)."""
    i, j, p = cuda.grid(3)
    if i >= Nx or j >= Ny or p >= nP:
        return
    k = sel[p]
    dHx_dz = (Hx[i, j, k] - Hx[i, j, k - 1]) / dz
    psi[i, j, p] = b[p] * psi[i, j, p] + c[p] * dHx_dz
    if i >= 1:
        Ey[i, j, k] += (cE / eps_y[i, j, k]) * psi[i, j, p]


@cuda.jit
def _k_EZ_X(Hy, Ez, eps_z, psi, b, c, sel, cE, dx, nP, Ny, Nz):
    """psi_Hy_x (x axis), correction +psi onto Ez."""
    p, j, k = cuda.grid(3)
    if p >= nP or j >= Ny or k >= Nz:
        return
    i = sel[p]
    dHy_dx = (Hy[i, j, k] - Hy[i - 1, j, k]) / dx
    psi[p, j, k] = b[p] * psi[p, j, k] + c[p] * dHy_dx
    if j >= 1:
        Ez[i, j, k] += (cE / eps_z[i, j, k]) * psi[p, j, k]


@cuda.jit
def _k_EZ_Y(Hx, Ez, eps_z, psi, b, c, sel, cE, dy, Nx, nP, Nz):
    """psi_Hx_y (y axis), correction -psi onto Ez."""
    i, p, k = cuda.grid(3)
    if i >= Nx or p >= nP or k >= Nz:
        return
    j = sel[p]
    dHx_dy = (Hx[i, j, k] - Hx[i, j - 1, k]) / dy
    psi[i, p, k] = b[p] * psi[i, p, k] + c[p] * dHx_dy
    if i >= 1:
        Ez[i, j, k] -= (cE / eps_z[i, j, k]) * psi[i, p, k]


# ====================================================================== #
# Thin wrappers — signature-compatible with update.py / pml.py
# (per-call host<->device transfer; correctness path, not yet the fast path)
# ====================================================================== #
def _scalars(grid):
    """Coefficient scalars cast to the field dtype (genuine float32 math on a
    float32 grid)."""
    T = grid.Hx.dtype.type
    return (T(grid.dt / MU0), T(grid.dt / EPS0),
            T(grid.dx), T(grid.dy), T(grid.dz))


def update_H(grid: FDTDGrid) -> FDTDGrid:
    """CUDA drop-in for :func:`wavesim.update.update_H`."""
    g = grid
    cH, _cE, dx, dy, dz = _scalars(g)
    dEx, dEy, dEz = (cuda.to_device(g.Ex), cuda.to_device(g.Ey),
                     cuda.to_device(g.Ez))
    dHx, dHy, dHz = (cuda.to_device(g.Hx), cuda.to_device(g.Hy),
                     cuda.to_device(g.Hz))
    dmux, dmuy, dmuz = (cuda.to_device(g.mu_x), cuda.to_device(g.mu_y),
                        cuda.to_device(g.mu_z))
    _launch(_k_update_H, (g.Nx, g.Ny, g.Nz),
            (dEx, dEy, dEz, dHx, dHy, dHz, dmux, dmuy, dmuz,
             cH, dx, dy, dz, g.Nx, g.Ny, g.Nz))
    dHx.copy_to_host(g.Hx); dHy.copy_to_host(g.Hy); dHz.copy_to_host(g.Hz)
    return g


def update_E(grid: FDTDGrid) -> FDTDGrid:
    """CUDA drop-in for :func:`wavesim.update.update_E`."""
    g = grid
    _cH, cE, dx, dy, dz = _scalars(g)
    dEx, dEy, dEz = (cuda.to_device(g.Ex), cuda.to_device(g.Ey),
                     cuda.to_device(g.Ez))
    dHx, dHy, dHz = (cuda.to_device(g.Hx), cuda.to_device(g.Hy),
                     cuda.to_device(g.Hz))
    depx, depy, depz = (cuda.to_device(g.eps_x), cuda.to_device(g.eps_y),
                        cuda.to_device(g.eps_z))
    _launch(_k_update_E, (g.Nx, g.Ny, g.Nz),
            (dEx, dEy, dEz, dHx, dHy, dHz, depx, depy, depz,
             cE, dx, dy, dz, g.Nx, g.Ny, g.Nz))
    dEx.copy_to_host(g.Ex); dEy.copy_to_host(g.Ey); dEz.copy_to_host(g.Ez)
    return g


def _dev_ravel(a, dtype):
    """CPML (b, c) slab arrays are stored reshaped to broadcast; the kernels want
    them 1D in slab order and in the field dtype."""
    return cuda.to_device(np.ascontiguousarray(a).ravel().astype(dtype, copy=False))


def update_H_pml(grid: FDTDGrid, cpml: CPMLArrays) -> tuple[FDTDGrid, CPMLArrays]:
    """CUDA drop-in for :func:`wavesim.pml.update_H_pml`."""
    g = grid
    dtype = g.Hx.dtype
    cH, _cE, dx, dy, dz = _scalars(g)
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz

    dEx, dEy, dEz = (cuda.to_device(g.Ex), cuda.to_device(g.Ey),
                     cuda.to_device(g.Ez))
    dHx, dHy, dHz = (cuda.to_device(g.Hx), cuda.to_device(g.Hy),
                     cuda.to_device(g.Hz))
    dmux, dmuy, dmuz = (cuda.to_device(g.mu_x), cuda.to_device(g.mu_y),
                        cuda.to_device(g.mu_z))

    sxH = cuda.to_device(cpml.sel_xH)
    syH = cuda.to_device(cpml.sel_yH)
    szH = cuda.to_device(cpml.sel_zH)
    nxH, nyH, nzH = cpml.sel_xH.shape[0], cpml.sel_yH.shape[0], cpml.sel_zH.shape[0]

    bxH, cxH = _dev_ravel(cpml.bxH_s, dtype), _dev_ravel(cpml.cxH_s, dtype)
    byH, cyH = _dev_ravel(cpml.byH_s, dtype), _dev_ravel(cpml.cyH_s, dtype)
    bzH, czH = _dev_ravel(cpml.bzH_s, dtype), _dev_ravel(cpml.czH_s, dtype)

    p_Ez_y = cuda.to_device(cpml.psi_Ez_y)
    p_Ey_z = cuda.to_device(cpml.psi_Ey_z)
    p_Ex_z = cuda.to_device(cpml.psi_Ex_z)
    p_Ez_x = cuda.to_device(cpml.psi_Ez_x)
    p_Ey_x = cuda.to_device(cpml.psi_Ey_x)
    p_Ex_y = cuda.to_device(cpml.psi_Ex_y)

    _launch(_k_HX_Y, (Nx, nyH, Nz),
            (dEz, dHx, dmux, p_Ez_y, byH, cyH, syH, cH, dy, Nx, nyH, Nz))
    if Nz > 1:
        _launch(_k_HX_Z, (Nx, Ny, nzH),
                (dEy, dHx, dmux, p_Ey_z, bzH, czH, szH, cH, dz, Nx, Ny, nzH))
    _launch(_k_HY_X, (nxH, Ny, Nz),
            (dEz, dHy, dmuy, p_Ez_x, bxH, cxH, sxH, cH, dx, nxH, Ny, Nz))
    if Nz > 1:
        _launch(_k_HY_Z, (Nx, Ny, nzH),
                (dEx, dHy, dmuy, p_Ex_z, bzH, czH, szH, cH, dz, Nx, Ny, nzH))
    _launch(_k_HZ_X, (nxH, Ny, Nz),
            (dEy, dHz, dmuz, p_Ey_x, bxH, cxH, sxH, cH, dx, nxH, Ny, Nz))
    _launch(_k_HZ_Y, (Nx, nyH, Nz),
            (dEx, dHz, dmuz, p_Ex_y, byH, cyH, syH, cH, dy, Nx, nyH, Nz))

    dHx.copy_to_host(g.Hx); dHy.copy_to_host(g.Hy); dHz.copy_to_host(g.Hz)
    p_Ez_y.copy_to_host(cpml.psi_Ez_y); p_Ey_z.copy_to_host(cpml.psi_Ey_z)
    p_Ex_z.copy_to_host(cpml.psi_Ex_z); p_Ez_x.copy_to_host(cpml.psi_Ez_x)
    p_Ey_x.copy_to_host(cpml.psi_Ey_x); p_Ex_y.copy_to_host(cpml.psi_Ex_y)
    return g, cpml


def update_E_pml(grid: FDTDGrid, cpml: CPMLArrays) -> tuple[FDTDGrid, CPMLArrays]:
    """CUDA drop-in for :func:`wavesim.pml.update_E_pml`."""
    g = grid
    dtype = g.Ex.dtype
    _cH, cE, dx, dy, dz = _scalars(g)
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz

    dEx, dEy, dEz = (cuda.to_device(g.Ex), cuda.to_device(g.Ey),
                     cuda.to_device(g.Ez))
    dHx, dHy, dHz = (cuda.to_device(g.Hx), cuda.to_device(g.Hy),
                     cuda.to_device(g.Hz))
    depx, depy, depz = (cuda.to_device(g.eps_x), cuda.to_device(g.eps_y),
                        cuda.to_device(g.eps_z))

    sxE = cuda.to_device(cpml.sel_xE)
    syE = cuda.to_device(cpml.sel_yE)
    szE = cuda.to_device(cpml.sel_zE)
    nxE, nyE, nzE = cpml.sel_xE.shape[0], cpml.sel_yE.shape[0], cpml.sel_zE.shape[0]

    bxE, cxE = _dev_ravel(cpml.bxE_s, dtype), _dev_ravel(cpml.cxE_s, dtype)
    byE, cyE = _dev_ravel(cpml.byE_s, dtype), _dev_ravel(cpml.cyE_s, dtype)
    bzE, czE = _dev_ravel(cpml.bzE_s, dtype), _dev_ravel(cpml.czE_s, dtype)

    p_Hz_y = cuda.to_device(cpml.psi_Hz_y)
    p_Hy_z = cuda.to_device(cpml.psi_Hy_z)
    p_Hx_z = cuda.to_device(cpml.psi_Hx_z)
    p_Hz_x = cuda.to_device(cpml.psi_Hz_x)
    p_Hy_x = cuda.to_device(cpml.psi_Hy_x)
    p_Hx_y = cuda.to_device(cpml.psi_Hx_y)

    _launch(_k_EX_Y, (Nx, nyE, Nz),
            (dHz, dEx, depx, p_Hz_y, byE, cyE, syE, cE, dy, Nx, nyE, Nz))
    if Nz > 1:
        _launch(_k_EX_Z, (Nx, Ny, nzE),
                (dHy, dEx, depx, p_Hy_z, bzE, czE, szE, cE, dz, Nx, Ny, nzE))
    _launch(_k_EY_X, (nxE, Ny, Nz),
            (dHz, dEy, depy, p_Hz_x, bxE, cxE, sxE, cE, dx, nxE, Ny, Nz))
    if Nz > 1:
        _launch(_k_EY_Z, (Nx, Ny, nzE),
                (dHx, dEy, depy, p_Hx_z, bzE, czE, szE, cE, dz, Nx, Ny, nzE))
    _launch(_k_EZ_X, (nxE, Ny, Nz),
            (dHy, dEz, depz, p_Hy_x, bxE, cxE, sxE, cE, dx, nxE, Ny, Nz))
    _launch(_k_EZ_Y, (Nx, nyE, Nz),
            (dHx, dEz, depz, p_Hx_y, byE, cyE, syE, cE, dy, Nx, nyE, Nz))

    dEx.copy_to_host(g.Ex); dEy.copy_to_host(g.Ey); dEz.copy_to_host(g.Ez)
    p_Hz_y.copy_to_host(cpml.psi_Hz_y); p_Hy_z.copy_to_host(cpml.psi_Hy_z)
    p_Hx_z.copy_to_host(cpml.psi_Hx_z); p_Hz_x.copy_to_host(cpml.psi_Hz_x)
    p_Hy_x.copy_to_host(cpml.psi_Hy_x); p_Hx_y.copy_to_host(cpml.psi_Hx_y)
    return g, cpml


# ====================================================================== #
# Device-side PEC kernels (for the resident runner)
# ====================================================================== #
@cuda.jit
def _k_zero_x(arr, i0, Ny, Nz):
    j, k = cuda.grid(2)
    if j < Ny and k < Nz:
        arr[i0, j, k] = 0.0


@cuda.jit
def _k_zero_y(arr, j0, Nx, Nz):
    i, k = cuda.grid(2)
    if i < Nx and k < Nz:
        arr[i, j0, k] = 0.0


@cuda.jit
def _k_zero_z(arr, k0, Nx, Ny):
    i, j = cuda.grid(2)
    if i < Nx and j < Ny:
        arr[i, j, k0] = 0.0


@cuda.jit
def _k_pec_mask(Ex, Ey, Ez, mex, mey, mez, Nx, Ny, Nz):
    """Zero each E component on its own edge mask.

    The masks are the per-component ones from
    :func:`wavesim.pec.build_pec_edge_masks`, computed on the host and uploaded
    once — a cell's twelve edges are indexed partly by its neighbours, so a
    single cell-wise mask would leave nine of them alive (see that function).
    Deriving them host-side keeps this bit-identical to the NumPy/Numba path.
    """
    i, j, k = cuda.grid(3)
    if i < Nx and j < Ny and k < Nz:
        if mex[i, j, k]:
            Ex[i, j, k] = 0.0
        if mey[i, j, k]:
            Ey[i, j, k] = 0.0
        if mez[i, j, k]:
            Ez[i, j, k] = 0.0


_TPB2 = (16, 16)


def _launch2d(kernel, ext, args):
    ex, ey = int(ext[0]), int(ext[1])
    if ex == 0 or ey == 0:
        return
    bpg = ((ex + _TPB2[0] - 1) // _TPB2[0], (ey + _TPB2[1] - 1) // _TPB2[1])
    kernel[bpg, _TPB2](*args)


# ====================================================================== #
# Device-resident runner
# ====================================================================== #
class CudaResident:
    """Keeps the fields, materials and CPML state resident on the GPU for the
    whole run, so the curl/CPML/PEC updates advance with **no per-step
    host<->device transfer** — this is the fast path.

    Built once from a ``(grid, cpml)`` pair. :meth:`step_evolution` runs the
    H, CPML-H, E, CPML-E and PEC updates entirely on the device.
    :meth:`download_EH` / :meth:`upload_EH` sync only the E/H fields, used by the
    caller to run host-side per-step hooks (sources, monitors) around the device
    loop; when there are no such hooks nothing is transferred until the end.

    Materials and the CPML psi/coefficient arrays never leave the device during a
    run. The compute dtype is the grid's dtype (float32 recommended on consumer
    GPUs).
    """

    def __init__(self, grid: FDTDGrid, cpml: CPMLArrays = None,
                 pec_faces: tuple = ()):
        g = grid
        self.g = g
        self.dtype = g.Hx.dtype
        self.Nx, self.Ny, self.Nz = g.Nx, g.Ny, g.Nz
        self.cH, self.cE, self.dx, self.dy, self.dz = _scalars(g)
        self.pec_faces = tuple(pec_faces)

        # Resident field + material arrays.
        self.dEx, self.dEy, self.dEz = map(cuda.to_device, (g.Ex, g.Ey, g.Ez))
        self.dHx, self.dHy, self.dHz = map(cuda.to_device, (g.Hx, g.Hy, g.Hz))
        self.dmux, self.dmuy, self.dmuz = map(cuda.to_device,
                                              (g.mu_x, g.mu_y, g.mu_z))
        self.depx, self.depy, self.depz = map(cuda.to_device,
                                               (g.eps_x, g.eps_y, g.eps_z))
        if g.pec_mask is not None:
            mex, mey, mez = build_pec_edge_masks(g.pec_mask)
            self.dmask = tuple(map(cuda.to_device, (mex, mey, mez)))
        else:
            self.dmask = None

        self.cpml = cpml
        if cpml is not None:
            self._init_cpml_device(cpml)

    def _init_cpml_device(self, c: CPMLArrays):
        dt = self.dtype
        self.sxH = cuda.to_device(c.sel_xH); self.syH = cuda.to_device(c.sel_yH)
        self.szH = cuda.to_device(c.sel_zH)
        self.sxE = cuda.to_device(c.sel_xE); self.syE = cuda.to_device(c.sel_yE)
        self.szE = cuda.to_device(c.sel_zE)
        self.nxH, self.nyH, self.nzH = (c.sel_xH.shape[0], c.sel_yH.shape[0],
                                        c.sel_zH.shape[0])
        self.nxE, self.nyE, self.nzE = (c.sel_xE.shape[0], c.sel_yE.shape[0],
                                        c.sel_zE.shape[0])
        self.bxH, self.cxH = _dev_ravel(c.bxH_s, dt), _dev_ravel(c.cxH_s, dt)
        self.byH, self.cyH = _dev_ravel(c.byH_s, dt), _dev_ravel(c.cyH_s, dt)
        self.bzH, self.czH = _dev_ravel(c.bzH_s, dt), _dev_ravel(c.czH_s, dt)
        self.bxE, self.cxE = _dev_ravel(c.bxE_s, dt), _dev_ravel(c.cxE_s, dt)
        self.byE, self.cyE = _dev_ravel(c.byE_s, dt), _dev_ravel(c.cyE_s, dt)
        self.bzE, self.czE = _dev_ravel(c.bzE_s, dt), _dev_ravel(c.czE_s, dt)
        self.p_Ez_y = cuda.to_device(c.psi_Ez_y)
        self.p_Ey_z = cuda.to_device(c.psi_Ey_z)
        self.p_Ex_z = cuda.to_device(c.psi_Ex_z)
        self.p_Ez_x = cuda.to_device(c.psi_Ez_x)
        self.p_Ey_x = cuda.to_device(c.psi_Ey_x)
        self.p_Ex_y = cuda.to_device(c.psi_Ex_y)
        self.p_Hz_y = cuda.to_device(c.psi_Hz_y)
        self.p_Hy_z = cuda.to_device(c.psi_Hy_z)
        self.p_Hx_z = cuda.to_device(c.psi_Hx_z)
        self.p_Hz_x = cuda.to_device(c.psi_Hz_x)
        self.p_Hy_x = cuda.to_device(c.psi_Hy_x)
        self.p_Hx_y = cuda.to_device(c.psi_Hx_y)

    # ------------------------------------------------------------------ #
    def _run_H(self):
        _launch(_k_update_H, (self.Nx, self.Ny, self.Nz),
                (self.dEx, self.dEy, self.dEz, self.dHx, self.dHy, self.dHz,
                 self.dmux, self.dmuy, self.dmuz,
                 self.cH, self.dx, self.dy, self.dz, self.Nx, self.Ny, self.Nz))

    def _run_E(self):
        _launch(_k_update_E, (self.Nx, self.Ny, self.Nz),
                (self.dEx, self.dEy, self.dEz, self.dHx, self.dHy, self.dHz,
                 self.depx, self.depy, self.depz,
                 self.cE, self.dx, self.dy, self.dz, self.Nx, self.Ny, self.Nz))

    def _run_H_pml(self):
        Nx, Ny, Nz, cH = self.Nx, self.Ny, self.Nz, self.cH
        _launch(_k_HX_Y, (Nx, self.nyH, Nz),
                (self.dEz, self.dHx, self.dmux, self.p_Ez_y, self.byH, self.cyH,
                 self.syH, cH, self.dy, Nx, self.nyH, Nz))
        if Nz > 1:
            _launch(_k_HX_Z, (Nx, Ny, self.nzH),
                    (self.dEy, self.dHx, self.dmux, self.p_Ey_z, self.bzH,
                     self.czH, self.szH, cH, self.dz, Nx, Ny, self.nzH))
        _launch(_k_HY_X, (self.nxH, Ny, Nz),
                (self.dEz, self.dHy, self.dmuy, self.p_Ez_x, self.bxH, self.cxH,
                 self.sxH, cH, self.dx, self.nxH, Ny, Nz))
        if Nz > 1:
            _launch(_k_HY_Z, (Nx, Ny, self.nzH),
                    (self.dEx, self.dHy, self.dmuy, self.p_Ex_z, self.bzH,
                     self.czH, self.szH, cH, self.dz, Nx, Ny, self.nzH))
        _launch(_k_HZ_X, (self.nxH, Ny, Nz),
                (self.dEy, self.dHz, self.dmuz, self.p_Ey_x, self.bxH, self.cxH,
                 self.sxH, cH, self.dx, self.nxH, Ny, Nz))
        _launch(_k_HZ_Y, (Nx, self.nyH, Nz),
                (self.dEx, self.dHz, self.dmuz, self.p_Ex_y, self.byH, self.cyH,
                 self.syH, cH, self.dy, Nx, self.nyH, Nz))

    def _run_E_pml(self):
        Nx, Ny, Nz, cE = self.Nx, self.Ny, self.Nz, self.cE
        _launch(_k_EX_Y, (Nx, self.nyE, Nz),
                (self.dHz, self.dEx, self.depx, self.p_Hz_y, self.byE, self.cyE,
                 self.syE, cE, self.dy, Nx, self.nyE, Nz))
        if Nz > 1:
            _launch(_k_EX_Z, (Nx, Ny, self.nzE),
                    (self.dHy, self.dEx, self.depx, self.p_Hy_z, self.bzE,
                     self.czE, self.szE, cE, self.dz, Nx, Ny, self.nzE))
        _launch(_k_EY_X, (self.nxE, Ny, Nz),
                (self.dHz, self.dEy, self.depy, self.p_Hz_x, self.bxE, self.cxE,
                 self.sxE, cE, self.dx, self.nxE, Ny, Nz))
        if Nz > 1:
            _launch(_k_EY_Z, (Nx, Ny, self.nzE),
                    (self.dHx, self.dEy, self.depy, self.p_Hx_z, self.bzE,
                     self.czE, self.szE, cE, self.dz, Nx, Ny, self.nzE))
        _launch(_k_EZ_X, (self.nxE, Ny, Nz),
                (self.dHy, self.dEz, self.depz, self.p_Hy_x, self.bxE, self.cxE,
                 self.sxE, cE, self.dx, self.nxE, Ny, Nz))
        _launch(_k_EZ_Y, (Nx, self.nyE, Nz),
                (self.dHx, self.dEz, self.depz, self.p_Hx_y, self.byE, self.cyE,
                 self.syE, cE, self.dy, Nx, self.nyE, Nz))

    def _run_pec(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        for face in self.pec_faces:
            if face == 'x0':
                _launch2d(_k_zero_x, (Ny, Nz), (self.dEy, 0, Ny, Nz))
                _launch2d(_k_zero_x, (Ny, Nz), (self.dEz, 0, Ny, Nz))
            elif face == 'x1':
                _launch2d(_k_zero_x, (Ny, Nz), (self.dEy, Nx - 1, Ny, Nz))
                _launch2d(_k_zero_x, (Ny, Nz), (self.dEz, Nx - 1, Ny, Nz))
            elif face == 'y0':
                _launch2d(_k_zero_y, (Nx, Nz), (self.dEx, 0, Nx, Nz))
                _launch2d(_k_zero_y, (Nx, Nz), (self.dEz, 0, Nx, Nz))
            elif face == 'y1':
                _launch2d(_k_zero_y, (Nx, Nz), (self.dEx, Ny - 1, Nx, Nz))
                _launch2d(_k_zero_y, (Nx, Nz), (self.dEz, Ny - 1, Nx, Nz))
            elif face == 'z0':
                _launch2d(_k_zero_z, (Nx, Ny), (self.dEx, 0, Nx, Ny))
                _launch2d(_k_zero_z, (Nx, Ny), (self.dEy, 0, Nx, Ny))
            elif face == 'z1':
                _launch2d(_k_zero_z, (Nx, Ny), (self.dEx, Nz - 1, Nx, Ny))
                _launch2d(_k_zero_z, (Nx, Ny), (self.dEy, Nz - 1, Nx, Ny))
            else:
                raise ValueError(f"Unknown face {face!r}")
        if self.dmask is not None:
            _launch(_k_pec_mask, (Nx, Ny, Nz),
                    (self.dEx, self.dEy, self.dEz) + self.dmask + (Nx, Ny, Nz))

    def step_evolution(self):
        """One full field step on the device: H, CPML-H, E, CPML-E, PEC."""
        self._run_H()
        if self.cpml is not None:
            self._run_H_pml()
        self._run_E()
        if self.cpml is not None:
            self._run_E_pml()
        self._run_pec()

    # ------------------------------------------------------------------ #
    def download_EH(self, grid: FDTDGrid):
        """Copy the resident E/H fields back into ``grid`` (host)."""
        self.dEx.copy_to_host(grid.Ex); self.dEy.copy_to_host(grid.Ey)
        self.dEz.copy_to_host(grid.Ez)
        self.dHx.copy_to_host(grid.Hx); self.dHy.copy_to_host(grid.Hy)
        self.dHz.copy_to_host(grid.Hz)

    def upload_EH(self, grid: FDTDGrid):
        """Push ``grid``'s host E/H fields onto the device (after host hooks)."""
        self.dEx.copy_to_device(grid.Ex); self.dEy.copy_to_device(grid.Ey)
        self.dEz.copy_to_device(grid.Ez)
        self.dHx.copy_to_device(grid.Hx); self.dHy.copy_to_device(grid.Hy)
        self.dHz.copy_to_device(grid.Hz)

    def sync(self):
        """Block until all queued device work has finished."""
        cuda.synchronize()
