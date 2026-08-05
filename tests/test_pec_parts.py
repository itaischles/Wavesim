"""
test_pec_parts.py — named PEC parts (wavesim.parts).

The property under test throughout is that naming a conductor is *purely
additive*: it records who owns a cell without changing what the cell is. So the
first test is the one that matters most — placing the same geometry with and
without names must produce the identical ``pec_mask``, because the FDTD solver
reads that array and nothing else, and a naming feature that perturbs a field
run would be a bug dressed as a convenience.
"""

import numpy as np
import pytest

import wavesim as ws


def _grid(Nx=20, Ny=20, Nz=10, ds=1e-3):
    return ws.set_vacuum(ws.create_grid(Nx, Ny, Nz, ds, ds, ds))


# ====================================================================== #
# Naming is additive
# ====================================================================== #

def test_naming_does_not_change_the_pec_mask():
    """The mask the FDTD update sees is identical with and without names."""
    anon = _grid()
    ws.set_box(anon, 2e-3, 8e-3, 2e-3, 8e-3, 2e-3, 5e-3, 1.0, pec=True)
    ws.set_cylinder(anon, 12e-3, 12e-3, 3e-3, 1e-3, 6e-3, 1.0, pec=True)

    named = _grid()
    ws.set_box(named, 2e-3, 8e-3, 2e-3, 8e-3, 2e-3, 5e-3, 1.0, pec=True,
               name="pad")
    ws.set_cylinder(named, 12e-3, 12e-3, 3e-3, 1e-3, 6e-3, 1.0, pec=True,
                    name="via")

    assert np.array_equal(anon.pec_mask, named.pec_mask)
    assert anon.pec_id is None and anon.pec_names is None


def test_unnamed_model_carries_no_part_state():
    g = _grid()
    ws.set_box(g, 2e-3, 8e-3, 2e-3, 8e-3, 2e-3, 5e-3, 1.0, pec=True)
    assert g.pec_id is None
    assert g.pec_names is None
    assert ws.list_conductors(g)[0].name is None


def test_named_cells_are_always_masked():
    """The pec_id ⊆ pec_mask invariant, checked cell by cell."""
    g = _grid()
    ws.set_box(g, 2e-3, 8e-3, 2e-3, 8e-3, 2e-3, 5e-3, 1.0, pec=True, name="a")
    ws.set_cylinder(g, 15e-3, 15e-3, 3e-3, 1e-3, 6e-3, 1.0, pec=True, name="b")
    assert not np.any((g.pec_id != 0) & ~g.pec_mask)


def test_part_numbering_starts_at_one():
    """0 must stay available to mean 'unnamed metal'."""
    g = _grid()
    ws.set_box(g, 2e-3, 5e-3, 2e-3, 5e-3, 2e-3, 5e-3, 1.0, pec=True, name="a")
    ws.set_box(g, 10e-3, 15e-3, 2e-3, 5e-3, 2e-3, 5e-3, 1.0, pec=True, name="b")
    assert sorted(g.pec_names.values()) == [1, 2]
    assert 0 not in g.pec_names.values()


# ====================================================================== #
# Part identity semantics
# ====================================================================== #

def test_repeating_a_name_extends_the_same_part():
    """A conductor built from two primitives is one part, not two."""
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 5e-3, 2e-3, 5e-3, 1.0, pec=True, name="L")
    ws.set_box(g, 2e-3, 4e-3, 5e-3, 12e-3, 2e-3, 5e-3, 1.0, pec=True, name="L")

    assert list(g.pec_names) == ["L"]
    conductors = [c for c in ws.list_conductors(g) if c.name == "L"]
    assert len(conductors) == 1
    # The part spans the union of both boxes, not just the last one.
    (_, _), (y0, y1), _ = conductors[0].bbox
    assert y0 == pytest.approx(2e-3) and y1 == pytest.approx(12e-3)


def test_overlapping_parts_last_writer_wins():
    """Matches the rule the material placement helpers already follow."""
    g = _grid()
    ws.set_box(g, 2e-3, 10e-3, 2e-3, 10e-3, 2e-3, 5e-3, 1.0, pec=True, name="a")
    ws.set_box(g, 5e-3, 10e-3, 2e-3, 10e-3, 2e-3, 5e-3, 1.0, pec=True, name="b")

    overlap = np.zeros_like(g.pec_mask)
    overlap[5:10, 2:10, 2:5] = True
    assert np.all(g.pec_id[overlap] == ws.part_id(g, "b"))


def test_a_named_part_in_two_pieces_is_still_one_conductor():
    """The user's declaration outranks the geometry's connectivity."""
    g = _grid()
    ws.set_box(g, 2e-3, 5e-3, 2e-3, 5e-3, 2e-3, 5e-3, 1.0, pec=True, name="split")
    ws.set_box(g, 14e-3, 18e-3, 2e-3, 5e-3, 2e-3, 5e-3, 1.0, pec=True, name="split")
    assert len([c for c in ws.list_conductors(g) if c.name == "split"]) == 1


def test_part_mask_and_lookup():
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 3e-3, 8e-3, 1e-3, 4e-3, 1.0, pec=True, name="trace")
    mask = ws.part_mask(g, "trace")
    assert mask.sum() == 4 * 5 * 3
    assert np.array_equal(mask, g.pec_mask)


def test_unknown_name_lists_what_exists():
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="gnd")
    with pytest.raises(KeyError, match="gnd"):
        ws.part_id(g, "grnd")


def test_unnamed_pec_mask_is_the_leftover_metal():
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="a")
    ws.set_box(g, 12e-3, 16e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True)

    leftover = ws.unnamed_pec_mask(g)
    assert np.array_equal(leftover, g.pec_mask & ~ws.part_mask(g, "a"))
    assert leftover.sum() == 4 * 4 * 3


# ====================================================================== #
# Inventory
# ====================================================================== #

def test_list_conductors_reports_named_then_unnamed_bodies():
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="a")
    # Two disjoint anonymous blocks -> two separate unnamed conductors.
    ws.set_box(g, 10e-3, 12e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True)
    ws.set_box(g, 16e-3, 19e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True)

    conductors = ws.list_conductors(g)
    assert [c.name for c in conductors] == ["a", None, None]
    assert [c.id for c in conductors] == [1, 0, 0]


def test_bbox_spans_nodes_not_cell_centres():
    """A body's extent is the physical box it occupies."""
    g = _grid()
    ws.set_box(g, 3e-3, 7e-3, 2e-3, 5e-3, 1e-3, 4e-3, 1.0, pec=True, name="b")
    (x0, x1), (y0, y1), (z0, z1) = ws.list_conductors(g)[0].bbox
    assert (x0, x1) == pytest.approx((3e-3, 7e-3))
    assert (y0, y1) == pytest.approx((2e-3, 5e-3))
    assert (z0, z1) == pytest.approx((1e-3, 4e-3))


def test_touches_edge_flags_a_conductor_running_into_the_boundary():
    g = _grid()
    ws.set_box(g, 0.0, 4e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="wall")
    ws.set_box(g, 10e-3, 14e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="free")
    flags = {c.name: c.touches_edge for c in ws.list_conductors(g)}
    assert flags == {"wall": True, "free": False}


def test_empty_model_lists_nothing():
    assert ws.list_conductors(_grid()) == []
    assert "no PEC" in ws.describe_conductors(_grid())


def test_describe_conductors_names_each_body():
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="sig")
    ws.set_box(g, 12e-3, 16e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True)
    text = ws.describe_conductors(g)
    assert "sig" in text and "<unnamed>" in text
    assert len(text.splitlines()) == 2


# ====================================================================== #
# Shorts between named parts
# ====================================================================== #

def test_touching_parts_are_reported_as_shorted():
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="a")
    ws.set_box(g, 6e-3, 10e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="b")
    assert ws.check_shorts(g) == [("a", "b")]


def test_separated_parts_are_not_shorted():
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="a")
    ws.set_box(g, 8e-3, 12e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="b")
    assert ws.check_shorts(g) == []


def test_diagonal_contact_is_not_a_short():
    """6-connectivity: cells meeting at an edge share no E-edge, so no current.

    Counting a corner touch as a connection would contradict the FDTD update
    this labelling exists to describe.
    """
    g = _grid()
    ws.set_box(g, 2e-3, 5e-3, 2e-3, 5e-3, 2e-3, 5e-3, 1.0, pec=True, name="a")
    ws.set_box(g, 5e-3, 8e-3, 5e-3, 8e-3, 2e-3, 5e-3, 1.0, pec=True, name="b")
    assert ws.check_shorts(g) == []


def test_unnamed_metal_can_bridge_two_named_parts():
    """A floating bracket shorts as effectively as direct contact."""
    g = _grid()
    ws.set_box(g, 2e-3, 5e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="a")
    ws.set_box(g, 8e-3, 11e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True, name="b")
    assert ws.check_shorts(g) == []
    ws.set_box(g, 5e-3, 8e-3, 2e-3, 6e-3, 2e-3, 5e-3, 1.0, pec=True)  # bridge
    assert ws.check_shorts(g) == [("a", "b")]


# ====================================================================== #
# Guard rails
# ====================================================================== #

def test_name_without_pec_is_refused():
    g = _grid()
    with pytest.raises(ValueError, match="requires pec=True"):
        ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 5e-3, 4.0, name="oops")
    with pytest.raises(ValueError, match="requires pec=True"):
        ws.set_cylinder(g, 5e-3, 5e-3, 2e-3, 1e-3, 4e-3, 4.0, name="oops")


@pytest.mark.parametrize("bad", ["", None, 3])
def test_empty_or_non_string_names_are_refused(bad):
    g = _grid()
    mask = np.zeros((g.Nx, g.Ny, g.Nz), dtype=bool)
    mask[2:5, 2:5, 2:5] = True
    with pytest.raises(ValueError, match="non-empty string"):
        ws.name_pec_region(g, mask, bad)


def test_name_pec_region_checks_shape():
    g = _grid()
    with pytest.raises(ValueError, match="expected shape"):
        ws.name_pec_region(g, np.ones((3, 3, 3), dtype=bool), "a")


def test_set_coax_names_both_conductors():
    g = _grid(40, 40, 4)
    ws.set_coax(g, 20e-3, 20e-3, 4e-3, 15e-3, eps_r_fill=2.3,
                name_inner="core", name_outer="shield")
    assert set(g.pec_names) == {"core", "shield"}
    named = {c.name: c for c in ws.list_conductors(g)}
    assert named["shield"].n_cells > named["core"].n_cells
    # Both extend through the whole z depth (set_coax extrudes a cross-section),
    # so both touch a domain face; only the shield reaches the transverse edge.
    (sx0, sx1), _, _ = named["shield"].bbox
    (cx0, cx1), _, _ = named["core"].bbox
    assert (sx0, sx1) == pytest.approx((0.0, 40e-3))
    assert cx0 > 0.0 and cx1 < 40e-3


# ====================================================================== #
# Production path (the CAD importer's entry point)
# ====================================================================== #

def _arrays(g):
    return dict(eps_x=g.eps_x, eps_y=g.eps_y, eps_z=g.eps_z,
                mu_x=g.mu_x, mu_y=g.mu_y, mu_z=g.mu_z)


def test_set_material_arrays_round_trips_a_labelling():
    g = _grid()
    mask = np.zeros((g.Nx, g.Ny, g.Nz), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    pid = np.zeros((g.Nx, g.Ny, g.Nz), dtype=np.int32)
    pid[2:6, 2:6, 2:6] = 7

    ws.set_material_arrays(g, **_arrays(g), pec_mask=mask,
                           pec_id=pid, pec_names={"solid": 7})
    assert ws.part_id(g, "solid") == 7
    assert ws.part_mask(g, "solid").sum() == 64


def test_set_material_arrays_rejects_labels_outside_the_mask():
    g = _grid()
    mask = np.zeros((g.Nx, g.Ny, g.Nz), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    pid = np.zeros((g.Nx, g.Ny, g.Nz), dtype=np.int32)
    pid[10:12, 2:6, 2:6] = 1  # labelled, but not metal

    with pytest.raises(ValueError, match="not PEC"):
        ws.set_material_arrays(g, **_arrays(g), pec_mask=mask,
                               pec_id=pid, pec_names={"ghost": 1})


def test_set_material_arrays_rejects_a_half_supplied_labelling():
    g = _grid()
    pid = np.zeros((g.Nx, g.Ny, g.Nz), dtype=np.int32)
    with pytest.raises(ValueError, match="together"):
        ws.set_material_arrays(g, **_arrays(g), pec_id=pid)
    with pytest.raises(ValueError, match="together"):
        ws.set_material_arrays(g, **_arrays(g), pec_names={"a": 1})


def test_set_material_arrays_rejects_unnamed_labels():
    g = _grid()
    mask = np.zeros((g.Nx, g.Ny, g.Nz), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    pid = np.zeros((g.Nx, g.Ny, g.Nz), dtype=np.int32)
    pid[2:6, 2:6, 2:6] = 4
    with pytest.raises(ValueError, match="no name"):
        ws.set_material_arrays(g, **_arrays(g), pec_mask=mask,
                               pec_id=pid, pec_names={"other": 9})


def test_set_material_arrays_rejects_zero_part_numbers():
    g = _grid()
    mask = np.zeros((g.Nx, g.Ny, g.Nz), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    pid = np.zeros((g.Nx, g.Ny, g.Nz), dtype=np.int32)
    with pytest.raises(ValueError, match="start at 1"):
        ws.set_material_arrays(g, **_arrays(g), pec_mask=mask,
                               pec_id=pid, pec_names={"zero": 0})


def test_replacing_the_mask_discards_a_stale_labelling():
    """Stale names are worse than absent ones — absent raises, stale energises."""
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 6e-3, 1.0, pec=True, name="old")
    new_mask = np.zeros((g.Nx, g.Ny, g.Nz), dtype=bool)
    new_mask[12:16, 12:16, 2:6] = True

    ws.set_material_arrays(g, **_arrays(g), pec_mask=new_mask)
    assert g.pec_id is None and g.pec_names is None


def test_set_material_arrays_keeps_parts_when_the_mask_is_untouched():
    g = _grid()
    ws.set_box(g, 2e-3, 6e-3, 2e-3, 6e-3, 2e-3, 6e-3, 1.0, pec=True, name="keep")
    ws.set_material_arrays(g, **_arrays(g))
    assert ws.part_id(g, "keep") == 1
