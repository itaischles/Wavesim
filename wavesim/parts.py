"""
parts.py — named PEC parts: giving one conductor an addressable identity.

``grid.pec_mask`` answers "is this cell metal?" and that is all the FDTD update
ever needs to know — a perfect conductor is a boundary condition, and boundary
conditions do not have names. Other solvers do need names. The electrostatic
solver (:mod:`wavesim.electrostatics`) must hold *the signal trace* at 5 V and
*the ground plane* at 0 V, which it cannot express against a single boolean array
in which the two are indistinguishable.

This module adds that identity as a labelling *of* ``pec_mask``:

    ``grid.pec_id[i,j,k]``  — part number owning the cell, 0 = unnamed metal
    ``grid.pec_names``      — ``{name: part number}``, numbering from 1

Both stay ``None`` until the first part is named, so nothing changes for a model
that never uses them, and the FDTD path never reads either.

Why names rather than connected components
------------------------------------------
:func:`wavesim.mode_solver.solve_tem_modes` identifies conductors by running
:func:`scipy.ndimage.label` over the plane and addressing them by label number.
That is fine there — a cross-section has two or three conductors and the caller
is a script. It is the wrong primitive here for two reasons.

First, **label numbers are not stable**. They are assigned in raster order over
whatever the voxeliser produced, so refining the mesh, nudging a part, or adding
a screw somewhere else in the model silently renumbers everything, and a saved
"conductor 2 is at 5 V" now energises a different piece of metal. A name assigned
at placement time survives all of that. The FreeCAD workbench is the primary
caller and its solids already carry names, so this also removes a translation
step that could only ever lose information.

Second, **a name distinguishes an error from a coincidence**. Two named parts
whose voxels touch are one electrical body, and asking for them to sit at two
different potentials is not a solvable problem — it is a modelling mistake, and
usually a meshing one (two solids that clear each other in CAD by less than a
cell). Connected-component numbering cannot report that, because after labelling
the two parts *are* one component and the evidence that they were ever meant to
be separate is gone. :func:`check_shorts` recovers it from the names.

An unnamed conductor is still a conductor; it simply has no potential assigned to
it, and the electrostatic solver grounds it. :func:`list_conductors` enumerates
both kinds, because a user cannot select a part they cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from wavesim.grid import FDTDGrid


# ====================================================================== #
# Conductor inventory
# ====================================================================== #

@dataclass(frozen=True)
class Conductor:
    """One addressable conductor in the model.

    A *named* part is one conductor by declaration (``name`` set, ``id`` ≥ 1),
    even in the rare case its voxels come out in two disjoint pieces — the user
    said it was one part, and holding both pieces at one potential is exactly
    what that means. Unnamed metal is instead reported one **connected body** at
    a time (``name is None``, ``id == 0``), since nothing else distinguishes it.
    """
    name: str            # None for unnamed metal
    id: int              # pec_id value; 0 for unnamed metal
    n_cells: int
    bbox: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    centroid: Tuple[float, float, float]
    touches_edge: bool   # does the body reach the domain boundary?

    def __str__(self) -> str:
        what = self.name if self.name is not None else "<unnamed>"
        (x0, x1), (y0, y1), (z0, z1) = self.bbox
        edge = ", touches domain edge" if self.touches_edge else ""
        return (f"{what}: {self.n_cells} cells, "
                f"bbox x[{x0:.4g},{x1:.4g}] y[{y0:.4g},{y1:.4g}] "
                f"z[{z0:.4g},{z1:.4g}] m{edge}")


def list_conductors(grid: FDTDGrid) -> List[Conductor]:
    """Enumerate every conductor in the model, named parts first.

    Named parts come out in part-number order, followed by one entry per
    connected body of unnamed metal. Returns an empty list for a model with no
    PEC at all.

    This is the discovery call: it is how a user (or the workbench UI) finds out
    what there is to assign a potential to, and how they check that the part
    they named is the size and in the place they expected. ``touches_edge``
    matters when choosing boundary conditions — a conductor running into a
    Dirichlet face is being shorted to that face's potential whether or not that
    was intended.
    """
    if grid.pec_mask is None or not grid.pec_mask.any():
        return []

    out = []
    for name, pid in sorted((grid.pec_names or {}).items(), key=lambda kv: kv[1]):
        mask = grid.pec_id == pid
        if mask.any():
            out.append(_describe(grid, mask, name, pid))

    unnamed = unnamed_pec_mask(grid)
    if unnamed.any():
        labels, n = ndimage.label(unnamed)
        for lab in range(1, n + 1):
            out.append(_describe(grid, labels == lab, None, 0))
    return out


def describe_conductors(grid: FDTDGrid) -> str:
    """Human-readable inventory of :func:`list_conductors`, one per line."""
    conductors = list_conductors(grid)
    if not conductors:
        return "no PEC in this model"
    return "\n".join(str(c) for c in conductors)


def _describe(grid: FDTDGrid, mask: np.ndarray, name, pid: int) -> Conductor:
    """Measure one conductor body from its cell mask."""
    idx = np.nonzero(mask)
    nodes = (grid.x, grid.y, grid.z)
    centres = (grid.xc, grid.yc, grid.zc)
    extent = (grid.Nx, grid.Ny, grid.Nz)

    # A cell spans node[i] .. node[i+1], so the body's extent runs from the low
    # node of its lowest cell to the high node of its highest — the true
    # physical box, not the span of cell centres.
    bbox = tuple((float(nodes[a][idx[a].min()]),
                  float(nodes[a][idx[a].max() + 1])) for a in range(3))
    centroid = tuple(float(centres[a][idx[a]].mean()) for a in range(3))
    touches = any(idx[a].min() == 0 or idx[a].max() == extent[a] - 1
                  for a in range(3))
    return Conductor(name=name, id=int(pid), n_cells=int(mask.sum()),
                     bbox=bbox, centroid=centroid, touches_edge=touches)


# ====================================================================== #
# Naming
# ====================================================================== #

def name_pec_region(grid: FDTDGrid, mask: np.ndarray, name: str) -> FDTDGrid:
    """Mark ``mask`` as PEC belonging to the part called ``name``.

    The production-path entry point, and the one the CAD importer wants: it
    takes a voxelised boolean mask straight from the mesher. The scaffolding
    helpers (``set_box``, ``set_cylinder``) route their ``name=`` argument here.

    Both ``pec_mask`` and ``pec_id`` are written, because a named part is metal
    first and named second — the identity is additional information about a
    conductor, never a way to declare one the FDTD update will not see.

    Re-using a ``name`` **extends** that part rather than starting a new one, so
    a conductor assembled from several primitives is one part. Where a new part
    overlaps an existing one the new part takes the cells, matching the
    last-writer-wins rule the material placement helpers already follow.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"part name must be a non-empty string, got {name!r}")
    mask = np.asarray(mask, dtype=bool)
    shape = (grid.Nx, grid.Ny, grid.Nz)
    if mask.shape != shape:
        raise ValueError(f"mask: expected shape {shape}, got {mask.shape}")

    _ensure_part_arrays(grid)
    pid = grid.pec_names.get(name)
    if pid is None:
        # Number from 1: 0 is reserved for "unnamed metal" in pec_id.
        pid = max(grid.pec_names.values(), default=0) + 1
        grid.pec_names[name] = pid

    grid.pec_mask[mask] = True
    grid.pec_id[mask] = pid
    return grid


def _ensure_part_arrays(grid: FDTDGrid) -> None:
    """Allocate ``pec_mask`` / ``pec_id`` / ``pec_names`` on first use."""
    shape = (grid.Nx, grid.Ny, grid.Nz)
    if grid.pec_mask is None:
        grid.pec_mask = np.zeros(shape, dtype=bool)
    if grid.pec_id is None:
        grid.pec_id = np.zeros(shape, dtype=np.int32)
    if grid.pec_names is None:
        grid.pec_names = {}


def part_id(grid: FDTDGrid, name: str) -> int:
    """Part number for ``name``, or ``KeyError`` listing what does exist."""
    names = grid.pec_names or {}
    if name not in names:
        known = ", ".join(sorted(names)) or "none"
        raise KeyError(f"no PEC part named {name!r}; named parts are: {known}")
    return int(names[name])


def part_mask(grid: FDTDGrid, name: str) -> np.ndarray:
    """Boolean cell mask of the named part."""
    return grid.pec_id == part_id(grid, name)


def unnamed_pec_mask(grid: FDTDGrid) -> np.ndarray:
    """Cells that are PEC but belong to no named part.

    The electrostatic solver grounds these. Non-empty is not an error — a shield
    or an enclosure is usually most naturally left unnamed and at 0 V — but it
    is worth reporting, since it is also what an unnoticed typo in a part name
    looks like.
    """
    shape = (grid.Nx, grid.Ny, grid.Nz)
    if grid.pec_mask is None:
        return np.zeros(shape, dtype=bool)
    if grid.pec_id is None:
        return grid.pec_mask.copy()
    return grid.pec_mask & (grid.pec_id == 0)


# ====================================================================== #
# Consistency
# ====================================================================== #

def check_shorts(grid: FDTDGrid) -> List[Tuple[str, ...]]:
    """Find sets of named parts that are electrically one body.

    Returns one sorted tuple of names per connected run of metal containing two
    or more distinct named parts; an empty list means every named part is its
    own island.

    Touching is decided by 6-connectivity (face-sharing) over ``pec_mask``,
    which is what "the same conductor" means on a Yee grid — two cells meeting
    only along an edge or at a corner share no E-edge and conduct nothing
    between them in the FDTD update, so counting them as connected here would
    contradict the solver they are meant to describe. Unnamed metal is included
    in the connectivity search, because a floating unnamed bracket bridging two
    named parts shorts them exactly as directly as contact would.

    Callers decide what to do about it: parts fused deliberately (a name per
    sub-piece of one electrode) are legitimate, so this reports rather than
    raises. The electrostatic solver escalates only when the shorted parts have
    been given *different* potentials, which has no solution.
    """
    if grid.pec_mask is None or grid.pec_id is None or not grid.pec_names:
        return []

    labels, n = ndimage.label(grid.pec_mask)
    by_id = {pid: name for name, pid in grid.pec_names.items()}

    out = []
    for lab in range(1, n + 1):
        ids = np.unique(grid.pec_id[labels == lab])
        names = sorted(by_id[int(i)] for i in ids if int(i) in by_id)
        if len(names) > 1:
            out.append(tuple(names))
    return out
