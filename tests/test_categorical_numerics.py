import unittest
import numpy as np
from cutcell import CutCellGrid, DiffusionOperator, BoundaryCondition
from cutcell.categorical_numerics import incidence, Coarsening, closed_diffusion_diagram


class CategoricalNumerics(unittest.TestCase):
    def test_ported_incidence_is_actual_solver_balance(self):
        grid = CutCellGrid([0., .1, .4, 1.], volume_fractions=[.2, 1., .5])
        op = DiffusionOperator(grid, .3, left=BoundaryCondition('value', 2.), right=BoundaryCondition('flux', .7))
        state = np.array([1., 3., -2.])
        source = np.array([.2, -.1, .6])
        b, flux = incidence(grid.size), op.flux(state)
        np.testing.assert_allclose(grid.volumes * op.tendency(state, source), b @ flux + grid.volumes * source)
        np.testing.assert_array_equal(np.ones(grid.size, dtype=int) @ b, [1, 0, 0, -1])

    def test_closed_diagram_operator_adjoint_energy_and_homology(self):
        grids = [CutCellGrid.uniform(5), CutCellGrid.partial_bottom(np.linspace(0, 1, 7), .31),
                 CutCellGrid([0., .1, .2, .5, 1.], apertures=[1, 1, 0, 1, 1]),
                 CutCellGrid.uniform(1)]
        rng = np.random.default_rng(623)
        for grid in grids:
            for k in (.3, 0.):
                op = DiffusionOperator(grid, k)
                d = closed_diffusion_diagram(op)
                wet, b, l, v = d['wet_cells'], d['boundary'], d['generator'], d['volumes']
                c = rng.normal(size=grid.size)
                x = c[wet]
                np.testing.assert_allclose(l @ x, op.tendency(c)[wet], atol=1e-13)
                np.testing.assert_allclose(v[:, None]*l, (v[:, None]*l).T, atol=1e-13)
                np.testing.assert_allclose(l @ np.ones(len(wet)), 0., atol=1e-13)
                self.assertAlmostEqual(float(x @ (v * (l @ x))), -float(np.dot(d['conductance'], (b.T @ x)**2)))
                self.assertLessEqual(float(x @ (v * (l @ x))), 1e-13)
                np.testing.assert_array_equal(d['component_cocycles'] @ b, 0)
                self.assertEqual(d['betti_0'], len(wet) - np.linalg.matrix_rank(b) if b.size else len(wet))
                self.assertEqual(d['betti_1'], 0)

    def test_chain_map_for_every_contiguous_partition(self):
        grid = CutCellGrid([0., .1, .25, .3, .7, 1.])
        for mask in range(1 << (grid.size-1)):
            cuts = (0,) + tuple(i for i in range(1, grid.size) if mask & (1 << (i-1))) + (grid.size,)
            c = Coarsening(grid, cuts)
            self.assertTrue(c.chain_map_holds())
            np.testing.assert_allclose(c.restrict @ c.prolong, np.eye(len(cuts)-1))
            np.testing.assert_allclose(c.coarse_volumes[:, None] * c.restrict,
                                       c.prolong.T * c.fine_volumes)
            np.testing.assert_allclose(c.coarse_volumes @ c.restrict, grid.volumes)
            projection = c.prolong @ c.restrict
            np.testing.assert_allclose(projection @ projection, projection)

    def test_coarsening_composes_functorially(self):
        fine = CutCellGrid([0., .1, .2, .4, .6, 1.])
        middle = CutCellGrid(fine.faces[[0, 2, 3, 5]])
        a = Coarsening(fine, (0, 2, 3, 5))
        b = Coarsening(middle, (0, 2, 3))
        direct = Coarsening(fine, (0, 3, 5))
        np.testing.assert_array_equal(b.aggregate @ a.aggregate, direct.aggregate)
        np.testing.assert_array_equal(b.face_map @ a.face_map, direct.face_map)
        np.testing.assert_allclose(b.restrict @ a.restrict, direct.restrict)
        np.testing.assert_array_equal(a.prolong @ b.prolong, direct.prolong)

    def test_partial_cell_restriction_ignores_dry_placeholders(self):
        grid = CutCellGrid.partial_bottom(np.linspace(0, 1, 5), .3)
        c = Coarsening(grid, (0, 2, 4))
        np.testing.assert_allclose(c.restrict_state([1e100, 2, 4, 6]), [2, 5])
        np.testing.assert_allclose(c.prolong_state([2, 5]), [0, 2, 5, 5])
        self.assertTrue(c.chain_map_holds())
        with self.assertRaises(ValueError):
            Coarsening(grid, (0, 1, 4))  # all-solid coarse block

    def test_mass_compatibility_does_not_imply_diffusion_naturality(self):
        fine, coarse = CutCellGrid.uniform(4), CutCellGrid.uniform(2)
        c = Coarsening(fine, (0, 2, 4))
        lf = closed_diffusion_diagram(DiffusionOperator(fine, 1.))['generator']
        lc = closed_diffusion_diagram(DiffusionOperator(coarse, 1.))['generator']
        self.assertTrue(c.chain_map_holds())
        self.assertGreater(np.linalg.norm(c.dynamics_defect(lf, lc)), 1.)
        identity = Coarsening(fine, (0, 1, 2, 3, 4))
        np.testing.assert_array_equal(identity.dynamics_defect(lf, lf), 0)

    def test_reject_invalid_and_affine_inputs(self):
        grid = CutCellGrid.uniform(3)
        for cuts in ((0, 0, 3), (1, 3), (0, 4, 3), (0, 1.5, 3), (False, 3), ()):
            with self.assertRaises(ValueError):
                Coarsening(grid, cuts)
        for boundary in (BoundaryCondition('flux', 1.), BoundaryCondition('value', 0.)):
            with self.assertRaises(ValueError):
                closed_diffusion_diagram(DiffusionOperator(grid, left=boundary))
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                incidence(n)
        with self.assertRaises(ValueError):
            Coarsening(grid, (0, 3)).dynamics_defect(np.eye(2), np.eye(1))
