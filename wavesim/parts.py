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
from typing import List, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from wavesim.grid import FDTDGrid
from wavesim.pec import build_pec_edge_masks, build_conformal_edge_masks


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

# ====================================================================== #
# Electrical connectivity
#
# "Which cells are the same conductor?" has to be answered the way the *field
# solver* answers it, not the way the geometry looks, or a solver built on this
# will disagree with the FDTD run it is supposed to describe.
#
# The field solver's answer is in the E-edge masks: ``apply_pec_mask`` zeroes a
# set of Yee edges, and E = 0 along an edge means its two end nodes are at the
# same potential. So the conductor is the *graph* of zeroed edges, one connected
# component per electrical body, and the nodes are where potential lives.
#
# This is not the same as 6-connectivity of ``pec_mask``, and the difference is
# not academic. The staircase rule (:func:`~wavesim.pec.build_pec_edge_masks`)
# zeroes an edge when *any* of the four cells around it is PEC — deliberate
# over-zeroing that buys E/H consistency. Two cells meeting only at a corner
# share no cell face, but that dilation zeroes edges on both sides of the shared
# *node*, so the FDTD does hold them at one potential: they are shorted. Reading
# it off the edge masks gets this right without anyone having to reason it out,
# and switches to the conformal rule by itself when the grid carries cut cells —
# where the same corner contact is genuinely *not* a short, because the exact
# rule zeroes an edge only when its open length is zero.
# ====================================================================== #

def _edge_masks(grid: FDTDGrid, cell_mask: np.ndarray = None) -> tuple:
    """The three per-component zeroed-E masks the FDTD would apply.

    Delegates to whichever rule the grid is actually running — conformal cut
    cells if it carries them, the staircase dilation otherwise — so everything
    built on top inherits that choice instead of re-deciding it.

    ``cell_mask`` restricts the question to one part: "which edges would be
    zeroed if *this* were the only metal in the model". The conformal path has
    no such restriction available (its open fractions are a property of the
    whole assembled geometry, not of one solid), so a part-wise query there
    falls back to the staircase rule on that part's cells — used only to decide
    which body a named part lands in, never to build the operator.
    """
    if cell_mask is None:
        if grid.is_conformal:
            return build_conformal_edge_masks(grid)
        if grid.pec_mask is None:
            shape = (grid.Nx, grid.Ny, grid.Nz)
            return tuple(np.zeros(shape, dtype=bool) for _ in range(3))
        return build_pec_edge_masks(grid.pec_mask)
    return build_pec_edge_masks(np.asarray(cell_mask, dtype=bool))


def pec_node_mask(grid: FDTDGrid, cell_mask: np.ndarray = None) -> np.ndarray:
    """Nodes lying on a conductor — the end points of every zeroed E-edge.

    Shape ``(Nx, Ny, Nz)``, indexed by node, following the same convention as
    every other array: node ``(i,j,k)`` is the low corner of cell ``(i,j,k)``,
    and the ``N``-th node of each axis is not carried (there are ``N+1`` nodes
    but ``N`` array slots, exactly as in :mod:`wavesim.mode_solver`).

    For a staircased block this comes out as the *closed* node box of the block,
    surface nodes included — which is the point. Slicing ``pec_mask`` and
    calling it a node mask instead keeps the low-side surface nodes and drops
    the high-side ones, an asymmetry that is the same off-by-half-a-cell trap
    :func:`~wavesim.pec.apply_pec_mask` documents.
    """
    ex, ey, ez = _edge_masks(grid, cell_mask)
    out = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool)
    for arr, ax in ((ex, 0), (ey, 1), (ez, 2)):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax] = slice(0, -1)
        hi[ax] = slice(1, None)
        # An edge contributes both of its end nodes. Edges on the last index of
        # their own axis reach a node the arrays do not carry; their low end
        # still counts.
        out |= arr
        out[tuple(hi)] |= arr[tuple(lo)]
    return out


def conductor_bodies(grid: FDTDGrid) -> Tuple[np.ndarray, int]:
    """Label every node by which electrical body it belongs to.

    Returns ``(labels, n_bodies)`` with ``labels`` of shape ``(Nx, Ny, Nz)``
    over nodes, 0 for nodes that are not on a conductor and 1..``n_bodies``
    otherwise. A *body* is a connected component of the zeroed-edge graph: the
    set of nodes the field solver forces to one potential.

    Connectivity runs over the edge graph rather than over the node mask,
    because the two are not the same. Two conductors separated by a single cell
    have node masks that are adjacent — the near-side surface nodes sit one
    index apart — while the edge between them lies in the gap and is not
    zeroed. Labelling the node mask directly would fuse them; walking the edges
    keeps them apart.
    """
    shape = (grid.Nx, grid.Ny, grid.Nz)
    n_nodes = int(np.prod(shape))
    idx = np.arange(n_nodes, dtype=np.int64).reshape(shape)

    rows, cols = [], []
    for arr, ax in zip(_edge_masks(grid), range(3)):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax] = slice(0, -1)
        hi[ax] = slice(1, None)
        live = arr[tuple(lo)]
        rows.append(idx[tuple(lo)][live])
        cols.append(idx[tuple(hi)][live])

    rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    labels = np.zeros(shape, dtype=np.int32)
    if rows.size == 0:
        return labels, 0

    graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)),
                       shape=(n_nodes, n_nodes))
    _, comp = connected_components(graph, directed=False)

    # connected_components gives every isolated node its own component; keep
    # only those an edge actually touched, then renumber them from 1.
    on_metal = pec_node_mask(grid)
    keep = np.unique(comp.reshape(shape)[on_metal])
    renumber = np.zeros(comp.max() + 1, dtype=np.int32)
    renumber[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    labels[on_metal] = renumber[comp.reshape(shape)[on_metal]]
    return labels, int(keep.size)


def body_parts(grid: FDTDGrid) -> List[List[str]]:
    """Named parts occupying each electrical body, in body order.

    ``body_parts(grid)[b - 1]`` is the sorted list of part names sharing body
    ``b``; an empty list means that body carries no named part at all (a shield
    or an enclosure nobody bothered to name). This is the lookup the
    electrostatic solver needs — it assigns potentials per *body*, not per part,
    because a body is what physically holds one potential.
    """
    labels, n = conductor_bodies(grid)
    out = [[] for _ in range(n)]
    for name in sorted(grid.pec_names or {}):
        touched = np.unique(labels[pec_node_mask(grid, part_mask(grid, name))])
        for b in touched:
            if b:
                out[int(b) - 1].append(name)
    return out


def check_shorts(grid: FDTDGrid) -> List[Tuple[str, ...]]:
    """Find sets of named parts that are electrically one body.

    Returns one sorted tuple of names per body containing two or more distinct
    named parts; an empty list means every named part is its own island.
    Unnamed metal participates in the connectivity, because a floating unnamed
    bracket bridging two named parts shorts them exactly as directly as contact
    would — it just does not get a name in the report.

    Callers decide what to do about it: parts fused deliberately (a name per
    sub-piece of one electrode) are legitimate, so this reports rather than
    raises. The electrostatic solver escalates only when the shorted parts have
    been given *different* potentials, which has no solution.
    """
    if grid.pec_mask is None or grid.pec_id is None or not grid.pec_names:
        return []
    return [tuple(names) for names in body_parts(grid) if len(names) > 1]
