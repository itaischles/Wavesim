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
    dz: float
    dt: float           # Computed from CFL condition — do not set manually

    # ------------------------------------------------------------------ #
    # Domain size (stored for convenience — derived from array shapes)
    # ------------------------------------------------------------------ #
    Nx: int
    Ny: int
    Nz: int

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

    # ------------------------------------------------------------------ #
    # Coordinate conversion — metres -> cell index
    # ------------------------------------------------------------------ #
    # Every public API speaks metres; these helpers are the single place
    # physical positions are snapped to the Yee grid. Sources, monitors, and
    # viz all route through them so the rounding convention is consistent.
    # ------------------------------------------------------------------ #
    def axis_index(self, axis: str, pos: float) -> int:
        """Nearest cell index along ``axis`` ('x'/'y'/'z') for ``pos`` (metres)."""
        ds = {'x': self.dx, 'y': self.dy, 'z': self.dz}.get(axis)
        if ds is None:
            raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
        return int(round(pos / ds))

    def position_to_index(self, x: float, y: float, z: float) -> tuple:
        """Nearest cell index ``(i, j, k)`` for a physical ``(x, y, z)`` in metres."""
        return (int(round(x / self.dx)),
                int(round(y / self.dy)),
                int(round(z / self.dz)))


def create_grid(Nx: int, Ny: int, Nz: int,
                dx: float, dy: float = None, dz: float = None,
                dtype=np.float64) -> FDTDGrid:
    """
    Allocate a grid with all fields and materials initialised to zero/vacuum.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of cells along each axis. Use Nz=1 for 2D-in-3D operation.
    dx : float
        Cell size in x (metres). If dy/dz are omitted they default to dx
        (uniform cubic cells).
    dtype : numpy dtype, optional
        Storage dtype for all field and material arrays (default
        ``np.float64``). Pass ``np.float32`` for the GPU (``backend='cuda'``)
        path: it halves memory traffic and, on consumer NVIDIA cards, avoids the
        heavily throttled float64 arithmetic. The NumPy/Numba CPU backends run
        correctly in either precision but are validated in float64.
        :func:`wavesim.pml.init_cpml` follows this dtype automatically.

    Returns
    -------
    FDTDGrid
        Fully initialised grid. All fields are zero; all materials are vacuum
        (eps_r = mu_r = 1); dt is set from the 3D CFL condition.

    Notes
    -----
    dt is computed from the conservative 3D CFL condition:
        dt = CFL / (c * sqrt(1/dx² + 1/dy² + 1/dz²))
    with CFL = 0.99.
    """
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx

    # 3D CFL stability criterion (Taflove eq. 4.78)
    CFL = 0.99
    dt = CFL / (C0 * np.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))

    shape = (Nx, Ny, Nz)
    zeros = lambda: np.zeros(shape, dtype=dtype)
    ones  = lambda: np.ones(shape,  dtype=dtype)

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
