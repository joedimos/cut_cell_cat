"""Exact regressions for the algebra, moving geometry, and interface witnesses."""
from fractions import Fraction as F
from itertools import product
import unittest

import numpy as np

from cutcell.categorical_numerics import incidence
from cutcell.research.conservation import (
    ChainMap, OrientedMesh, Slab, balance_defect, glue_fluxes, internal_correction, overlap_remap,
)
from cutcell.research.subgrid_capacity import PlanarBedCutCell


class MeshCategoryTests(unittest.TestCase):
    def test_matches_existing_path_convention(self):
        mesh = OrientedMesh(3, ((None, 0), (0, 1), (1, 2), (2, None)))
        np.testing.assert_array_equal(mesh.incidence, incidence(3))

    def test_composition_and_augmented_exterior_budget(self):
        mesh = OrientedMesh(4, ((None, 0), (0, 1), (2, 1), (1, 2), (2, 3), (3, None)))
        first = mesh.quotient((0, 0, 1, 2))
        second = first.target.quotient((0, 0, 1))
        third = second.target.quotient((0, 0))
        self.assertEqual(first.then(second), mesh.quotient((0, 0, 0, 1)))
        self.assertEqual(first.then(second).then(third), first.then(second.then(third)))
        self.assertEqual(ChainMap.identity(mesh).then(first), first)
        self.assertEqual(first.then(ChainMap.identity(first.target)), first)
        # Opposite fine orientations aggregate with opposite signs.
        np.testing.assert_array_equal(first.faces[1], [0, 0, -1, 1, 0, 0])
        flux = np.array([1, 10, -2, 4, 7, 3], dtype=object)
        self.assertEqual((mesh.incidence @ flux).sum(), -2)
        self.assertEqual((first.target.incidence @ (first.faces @ flux)).sum(), -2)

    def test_noncommuting_and_nonconservative_maps_are_rejected(self):
        mesh = OrientedMesh(2, ((0, 1),))
        with self.assertRaisesRegex(ValueError, "commute"):
            ChainMap(mesh, mesh, [[1, 0], [0, 1]], [[-1]])
        with self.assertRaisesRegex(ValueError, "preserve"):
            ChainMap(mesh, mesh, [[2, 0], [0, 2]], [[2]])
        with self.assertRaisesRegex(ValueError, "exact"):
            ChainMap(mesh, mesh, [[1., 0.], [0., 1.]], [[1]])
        with self.assertRaises(ValueError):
            OrientedMesh(1, ((None, None),))
        empty = OrientedMesh(0, ())
        self.assertEqual(ChainMap.identity(empty), empty.quotient(()))

    def test_large_integral_chain_coefficients_do_not_wrap(self):
        mesh = OrientedMesh(2, ((0, 1),))
        k = 2**80
        transfer = ChainMap(mesh, mesh, [[k, 0], [1-k, 1]], [[k]])
        self.assertEqual(transfer.then(transfer).faces[0, 0], k*k)


class MovingMeshTests(unittest.TestCase):
    def test_polygon_capacity_wets_and_dries_with_uniform_scalar(self):
        polygon = PlanarBedCutCell(((0, 0), (1, 0), (1, 1), (0, 1)), (1, 0, 0))
        volume = F.from_float(polygon.volume_below(.5))
        self.assertEqual(volume, F(1, 8))
        mesh = OrientedMesh(2, ((0, 1),))
        wet = Slab(mesh, (volume, 0), (0, volume), {0})
        dry = Slab(mesh, wet.after, wet.after, set())
        np.testing.assert_array_equal(wet.geometry_residual([volume], [0, 0]), [0, 0])
        np.testing.assert_array_equal(dry.geometry_residual([0], [0, 0]), [0, 0])
        c = F(2, 3)
        np.testing.assert_array_equal(wet.residual([c*volume, 0], [0, c*volume],
                                                  [c*volume], [0, 0]), [0, 0])
        with self.assertRaisesRegex(ValueError, "closed face"):
            dry.geometry_residual([volume], [0, 0])
        np.testing.assert_array_equal(wet.geometry_residual([0], [0, 0]), [-volume, volume])

    def test_overlap_remap_conserves_mass_and_constant_at_newly_wet_cell(self):
        o = [[1, 0, 0], [2, 0, 1], [0, 0, 1]]
        before, after = (3, 0, 2), (1, 3, 1)
        mass = overlap_remap(o, before, after, (9, 0, 8))
        np.testing.assert_array_equal(mass, [3, 10, 4])
        self.assertEqual(sum(mass), 17)
        np.testing.assert_array_equal(overlap_remap(o, before, after, (6, 0, 4)), [2, 6, 2])
        with self.assertRaisesRegex(ValueError, "margins"):
            overlap_remap(o, before, (2, 2, 1), (6, 0, 4))
        with self.assertRaisesRegex(ValueError, "dry source"):
            overlap_remap(o, before, after, (6, 1, 4))

    def test_direct_overlap_can_disagree_with_composed_remaps(self):
        mass = (1, 0)
        merged = overlap_remap([[1, 1]], (1, 1), (2,), mass)
        split = overlap_remap([[1], [1]], (2,), (1, 1), merged)
        direct = overlap_remap([[1, 0], [0, 1]], (1, 1), (1, 1), mass)
        np.testing.assert_array_equal(split, [F(1, 2), F(1, 2)])
        np.testing.assert_array_equal(direct, mass)

    def test_diffusion_cannot_descend_to_block_masses(self):
        mesh = OrientedMesh(3, ((0, 1), (1, 2)))
        a = mesh.quotient((0, 0, 1)).cells
        states = np.array([[2, 0, 0], [0, 2, 0]], dtype=object)
        np.testing.assert_array_equal(a @ states[0], a @ states[1])
        b = mesh.incidence
        for state, expected in zip(states, ([2, 0], [F(3, 2), F(1, 2)])):
            next_state = state - b @ (F(1, 4) * (b.T @ state))
            self.assertTrue(all(next_state >= 0))
            np.testing.assert_array_equal(a @ next_state, expected)

    def test_state_square_factorization_and_masked_obstruction(self):
        mesh = OrientedMesh(3, ((0, 1), (1, 2)))
        r = mesh.quotient((0, 0, 1))
        fine = Slab(mesh, (1, 2, 0), (1, 1, 1), {0, 1})
        coarse = Slab(r.target, (3, 0), (3, 0), set())
        q, s, d = balance_defect(r, fine, coarse, [0, 1], [0], [0, 0, 0], [0, 0])
        np.testing.assert_array_equal(q, [1])
        np.testing.assert_array_equal(s, [0, 0])
        np.testing.assert_array_equal(d, r.cells @ fine.after - coarse.after)
        self.assertEqual(sum(d), 0)
        with self.assertRaisesRegex(ValueError, "component"):
            internal_correction(coarse, d)
        np.testing.assert_array_equal(internal_correction(Slab(r.target, (3, 0), (3, 0), {0}), d), [1])

    def test_component_criterion_exhaustive_on_small_cycle_with_external_face(self):
        mesh = OrientedMesh(4, ((1, 0), (1, 2), (2, 0), (None, 3)))
        for mask in range(8):
            opened = {j for j in range(3) if mask & (1 << j)} | {3}
            slab = Slab(mesh, (1,)*4, (1,)*4, opened)
            generated = set()
            for x in product(range(-3, 4), repeat=3):
                if all(x[j] == 0 or j in opened for j in range(3)):
                    generated.add(tuple(mesh.incidence @ (*x, 0)))
            for d in product(range(-1, 2), repeat=4):
                if d not in generated:
                    with self.assertRaises(ValueError):
                        internal_correction(slab, d)
                else:
                    q = internal_correction(slab, d)
                    self.assertEqual(q[-1], 0)
                    np.testing.assert_array_equal(mesh.incidence @ slab.face_vector(q), d)

    def test_natural_balance_can_hide_circulation_and_source_compensation(self):
        mesh = OrientedMesh(3, ((0, 1), (1, 2), (2, 0)))
        slab = Slab(mesh, (1, 1, 1), (1, 1, 1), {0, 1, 2})
        r = ChainMap.identity(mesh)
        q, _, d = balance_defect(r, slab, slab, [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0])
        np.testing.assert_array_equal(q, [1, 1, 1])
        np.testing.assert_array_equal(d, [0, 0, 0])
        _, s, d = balance_defect(r, slab, slab, [1, 0, 0], [0, 0, 0], [1, -1, 0], [0, 0, 0])
        np.testing.assert_array_equal(s, [1, -1, 0])
        np.testing.assert_array_equal(d, [0, 0, 0])


class FluxSheafTests(unittest.TestCase):
    def test_glue_with_exterior_port_and_local_balance(self):
        mesh = OrientedMesh(2, ((0, 1), (0, None), (None, 1)))
        slab = Slab(mesh, (3, 0), (1, 3), {0, 1, 2})
        atoms = {0: ("first", "second"), 1: ("all",), 2: ("all",)}
        outward = {(0, 0, "first"): F(1, 2), (1, 0, "first"): -F(1, 2),
                   (0, 0, "second"): 1, (1, 0, "second"): -1,
                   (0, 1, "all"): F(1, 2), (1, 2, "all"): -F(3, 2)}
        flux, mismatch = glue_fluxes(slab, atoms, outward)
        self.assertEqual(mismatch, {})
        np.testing.assert_array_equal(flux, [F(3, 2), F(1, 2), F(3, 2)])
        np.testing.assert_array_equal(slab.geometry_residual(flux, [0, 0]), [0, 0])
        with self.assertRaisesRegex(ValueError, "cover"):
            glue_fluxes(slab, atoms, {})
        with self.assertRaisesRegex(ValueError, "distinct"):
            glue_fluxes(slab, {**atoms, 1: ("all", "all")}, outward)

    def test_aggregate_agreement_masks_equal_and_opposite_atom_defects(self):
        mesh = OrientedMesh(2, ((0, 1),))
        slab = Slab(mesh, (1, 0), (0, 1), {0})
        q, mismatches = glue_fluxes(slab, {0: (0, 1)}, {(0, 0, 0): 1, (1, 0, 0): 0,
                                                       (0, 0, 1): 0, (1, 0, 1): -1})
        self.assertIsNone(q)
        self.assertEqual(mismatches, {(0, 0): 1, (0, 1): -1})
        self.assertEqual(sum(mismatches.values()), 0)

    def test_closed_face_has_no_trace_slots(self):
        slab = Slab(OrientedMesh(2, ((0, 1),)), (1, 0), (1, 0), set())
        q, errors = glue_fluxes(slab, {}, {})
        np.testing.assert_array_equal(q, [0])
        self.assertEqual(errors, {})
        with self.assertRaisesRegex(ValueError, "open faces"):
            glue_fluxes(slab, {0: (0,)}, {})


if __name__ == "__main__":
    unittest.main()
