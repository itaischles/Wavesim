"""
grid.py — FDTDGrid dataclass and construction helpers.

The FDTDGrid is the central state object passed to every function in the
FDTD engine. All field and material data live here.

Design decisions:
- All arrays are full 3D: shape (Nx, Ny, Nz), even for Nz=1 slices.
- dt is always computed from the full 3D CFL condition (correct for Nz>1 too).
- Material arrays are split into x/y/z components to support future
  anisotropic materials.
"""

from dataclasses import dataclass, field
import numpy as np
from wavesim.constants import C0


@dataclass
class FDTDGrid:
    # ------------------------------------------------------------------ #
    # Field arrays — shape (Nx, Ny, Nz) always
    # ------------------------------------------------------------------ #
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Hx: np.ndarray
    Hy: np.ndarray
    Hz: np.ndarray

    # ------------------------------------------------------------------ #
    # Material arrays — shape (Nx, Ny, Nz)
    # Stored as separate x/y/z tensors for future anisotropic support.
    # eps_x is the permittivity seen by Ex, etc.
    # ------------------------------------------------------------------ #
    eps_x: np.ndarray   # relative permittivity seen by Ex
    eps_y: np.ndarray
    eps_z: np.ndarray
    mu_x: np.ndarray    # relative permeability seen by Hx
    mu_y: np.ndarray
    mu_z: np.ndarray

    # ------------------------------------------------------------------ #
    # Grid spacing (metres)
    # ------------------------------------------------------------------ #
    dx: float
    dy: float
    dz: float           # 3D-UPGRADE: set dz = dx for uniform 3D; for Nz=1 dz=dx is fine
    dt: float           # Computed from CFL condition — do not set manually

    # ------------------------------------------------------------------ #
    # Domain size (stored for convenience — derived from array shapes)
    # ------------------------------------------------------------------ #
    Nx: int
    Ny: int
    Nz: int             # 3D-UPGRADE: set Nz > 1 for full 3D

    # ------------------------------------------------------------------ #
    # PEC body mask — shape (Nx, Ny, Nz), dtype bool
    # True = cell is a perfect electric conductor.
    # E components inside PEC cells are zeroed after every E update.
    # Set by pec.py geometry helpers or directly by the future CAD importer.
    # ------------------------------------------------------------------ #
    pec_mask: np.ndarray = field(default=None)

    # ------------------------------------------------------------------ #
    # Simulation time
    # ------------------------------------------------------------------ #
    time_step: int = 0


def create_grid(Nx: int, Ny: int, Nz: int,
                dx: float, dy: float = None, dz: float = None) -> FDTDGrid:
    """
    Allocate a grid with all fields and materials initialised to zero/vacuum.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of cells along each axis. Use Nz=1 for 2D-in-3D operation.
    dx : float
        Cell size in x (metres). If dy/dz are omitted they default to dx
        (uniform cubic cells).

    Returns
    -------
    FDTDGrid
        Fully initialised grid. All fields are zero; all materials are vacuum
        (eps_r = mu_r = 1); dt is set from the 3D CFL condition.

    Notes
    -----
    dt is computed from the conservative 3D CFL condition:
        dt = CFL / (c * sqrt(1/dx² + 1/dy² + 1/dz²))
    with CFL = 0.99. This formula is already correct for full 3D — no
    changes needed when Nz is later increased.
    # 3D-UPGRADE: CFL formula is already correct — no changes needed.
    """
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx

    # 3D CFL stability criterion (Taflove eq. 4.78)
    CFL = 0.99
    dt = CFL / (C0 * np.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))

    shape = (Nx, Ny, Nz)
    zeros = lambda: np.zeros(shape, dtype=np.float64)
    ones  = lambda: np.ones(shape,  dtype=np.float64)

    return FDTDGrid(
        # Fields — initialised to zero
        Ex=zeros(), Ey=zeros(), Ez=zeros(),
        Hx=zeros(), Hy=zeros(), Hz=zeros(),

        # Materials — initialised to vacuum (eps_r = mu_r = 1)
        eps_x=ones(), eps_y=ones(), eps_z=ones(),
        mu_x=ones(),  mu_y=ones(),  mu_z=ones(),

        # Grid spacing
        dx=dx, dy=dy, dz=dz, dt=dt,

        # Domain size
        Nx=Nx, Ny=Ny, Nz=Nz,

        # PEC mask starts as None (no interior conductors)
        pec_mask=None,

        time_step=0,
    )
