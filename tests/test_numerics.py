import unittest
import numpy as np
from cutcell import CutCellGrid, DiffusionOperator, DiffusionModel, BoundaryCondition as BC


class GeometryTests(unittest.TestCase):
    def test_uniform_centers_are_cell_centers(self):
        grid = CutCellGrid.uniform(4)
        np.testing.assert_allclose(grid.centers, [.125, .375, .625, .875])
        self.assertAlmostEqual(sum(grid.volumes), 1)

    def test_geometry_copies_inputs(self):
        faces = np.array([0., .5, 1.])
        grid = CutCellGrid(faces)
        faces[1] = .1
        self.assertEqual(grid.faces[1], .5)
        with self.assertRaises(ValueError):
            grid.volumes[0] = 0

    def test_invalid_geometry(self):
        for faces in ([0, 0, 1], [0, np.nan], [0, 1, .5], [0]):
            with self.subTest(faces=faces), self.assertRaises(ValueError):
                CutCellGrid(faces)
        for fractions in ([0, 0], [-1, 1], [1, 2], [np.nan, 1]):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                CutCellGrid([0, .5, 1], fractions)
        with self.assertRaises(ValueError):
            CutCellGrid([0, .5, 1], [0, 1], apertures=1)
        with self.assertRaises(ValueError):
            CutCellGrid([0, .5, 1], centers=[.6, .8])
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                CutCellGrid.uniform(size)

    def test_partial_bottom_matches_upstream_metric_formula(self):
        grid = CutCellGrid.partial_bottom([0, 1, 2, 3], 1.9, minimum_fraction=.2)
        np.testing.assert_allclose(grid.volumes, [0, .2, 1])
        self.assertAlmostEqual(grid.centers[1], 1.9)
        # Upstream delta at face above bottom = half full + half partial height.
        self.assertAlmostEqual(grid.centers[2] - grid.centers[1], .5 + .2 / 2)
        np.testing.assert_allclose(grid.apertures, [0, 0, 1, 1])

    def test_bottom_inside_first_cell_is_impermeable(self):
        grid = CutCellGrid.partial_bottom([0, 1, 2], .9)
        self.assertEqual(grid.apertures[0], 0)
        op = DiffusionOperator(grid, 1, BC('flux', 100))
        self.assertEqual(op.flux([1, 2])[0], 0)

    def test_exact_face_bottom(self):
        grid = CutCellGrid.partial_bottom([0, 1, 2], 1)
        np.testing.assert_allclose(grid.volumes, [0, 1])
        np.testing.assert_allclose(grid.apertures, [0, 0, 1])


class OperatorTests(unittest.TestCase):
    def test_one_cell(self):
        model = DiffusionModel(CutCellGrid.uniform(1), [4.5])
        self.assertTrue(np.isinf(model.operator.stable_dt()))
        model.step(1)
        np.testing.assert_allclose(model.state, [4.5])

    def test_two_cell_exchange(self):
        model = DiffusionModel(CutCellGrid.uniform(2), [0, 1], 1, method='euler')
        model.step(.1)
        np.testing.assert_allclose(model.state, [.4, .6])
        self.assertAlmostEqual(model.mass(), .5)

    def test_uniform_neumann_stencil(self):
        op = DiffusionOperator(CutCellGrid.uniform(4), .3)
        c = np.array([1., 4., 2., 7.])
        expected = .3 * 16 * np.array([3, -5, 7, -5])
        np.testing.assert_allclose(op.tendency(c), expected)
        self.assertEqual(op.flux(c)[0], 0)
        self.assertEqual(op.flux(c)[-1], 0)

    def test_heterogeneous_face_flux(self):
        op = DiffusionOperator(CutCellGrid([0, 1, 2, 3], apertures=[1, .5, 1, 1]), [0, 2, 3, 0])
        np.testing.assert_allclose(op.flux([0, 1, 3]), [0, -1, -6, 0])

    def test_discrete_divergence_theorem_random_geometry(self):
        rng = np.random.default_rng(2026)
        for _ in range(20):
            grid = CutCellGrid(np.r_[0, np.cumsum(rng.uniform(.1, 1, 12))],
                               rng.uniform(.01, 1, 12), rng.uniform(.01, 1, 13))
            op = DiffusionOperator(grid, rng.random(13), BC('flux', .2), BC('flux', -.3))
            c = rng.normal(size=12)
            flux = op.flux(c)
            self.assertAlmostEqual(np.dot(grid.volumes, op.tendency(c)), flux[0] - flux[-1], places=12)

    def test_self_adjoint_negative_semidefinite_operator(self):
        grid = CutCellGrid([0, .1, .4, 1], [.2, .8, 1], [1, .5, .9, 1])
        op = DiffusionOperator(grid, [1, 2, .3, 1])
        matrix = np.column_stack([op.tendency(v) for v in np.eye(3)])
        weighted = grid.volumes[:, None] * matrix
        np.testing.assert_allclose(weighted, weighted.T, atol=1e-14)
        self.assertLessEqual(np.linalg.eigvalsh(weighted).max(), 1e-12)
        np.testing.assert_allclose(matrix @ np.ones(3), 0, atol=1e-12)

    def test_linear_dirichlet_steady_state_nonuniform(self):
        grid = CutCellGrid([0, .1, .3, .7, 1])
        op = DiffusionOperator(grid, .7, BC('value', 2), BC('value', 5))
        np.testing.assert_allclose(op.tendency(2 + 3 * grid.centers), 0, atol=1e-12)

    def test_small_cells_reduce_stability_bound(self):
        full = DiffusionOperator(CutCellGrid.uniform(3), 1).stable_dt()
        cut = DiffusionOperator(CutCellGrid.uniform(3, volume_fractions=[1, .001, 1]), 1).stable_dt()
        self.assertAlmostEqual(cut / full, .001)

    def test_invalid_operator_inputs(self):
        for k in (-1, np.nan, [1, 2]):
            with self.assertRaises(ValueError):
                DiffusionOperator(CutCellGrid.uniform(2), k)
        with self.assertRaises(ValueError):
            BC('neuman', 0)
        with self.assertRaises(ValueError):
            BC('flux', np.inf)


class EvolutionTests(unittest.TestCase):
    def test_mass_bounds_energy_no_clipping(self):
        rng = np.random.default_rng(4)
        for method in ('euler', 'ssprk3'):
            grid = CutCellGrid.uniform(20, volume_fractions=rng.uniform(.02, 1, 20))
            initial = rng.uniform(-3, 5, 20)
            model = DiffusionModel(grid, initial, method=method)
            mass, energy = model.mass(), np.dot(grid.volumes, initial**2) / 2
            for _ in range(100):
                b = model.step(1)
                self.assertLessEqual(b.energy, energy + 1e-12)
                self.assertGreaterEqual(b.minimum, initial.min() - 1e-12)
                self.assertLessEqual(b.maximum, initial.max() + 1e-12)
                self.assertAlmostEqual(model.mass(), mass, places=12)
                energy = b.energy

    def test_solid_cells_split_independent_components(self):
        grid = CutCellGrid.uniform(5, volume_fractions=[1, 1, 0, .2, 1])
        model = DiffusionModel(grid, [0, 1, 999, 4, 2])
        left_mass = np.dot(grid.volumes[:2], model.state[:2])
        right_mass = np.dot(grid.volumes[3:], model.state[3:])
        for _ in range(30):
            model.step(.01)
        self.assertEqual(model.state[2], 999)
        self.assertAlmostEqual(np.dot(grid.volumes[:2], model.state[:2]), left_mass)
        self.assertAlmostEqual(np.dot(grid.volumes[3:], model.state[3:]), right_mass)

    def test_sources_and_signed_boundary_budget(self):
        grid = CutCellGrid.uniform(8)
        model = DiffusionModel(grid, np.ones(8), left=BC('flux', .2), right=BC('flux', .05), source=.4)
        initial = model.mass()
        model.run_until(.03)
        self.assertAlmostEqual(model.mass(), initial + (.2 - .05 + .4) * .03, places=12)
        self.assertTrue(all(b.passed for b in model.history))

    def test_dirichlet_rk_stage_weighted_budget(self):
        model = DiffusionModel(CutCellGrid.uniform(10), np.zeros(10), left=BC('value', 1))
        model.run_until(.01, max_dt=.01)
        self.assertAlmostEqual(model.mass(), sum(b.boundary_exchange for b in model.history), places=13)
        self.assertGreater(model.mass(), 0)

    def test_clock_reports_actual_dt(self):
        model = DiffusionModel(CutCellGrid.uniform(10), np.ones(10))
        b = model.step(100)
        self.assertLess(b.dt, 100)
        self.assertEqual(model.time, b.dt)
        target = model.time + .012345
        model.run_until(target, max_dt=100)
        self.assertEqual(model.time, target)
        self.assertAlmostEqual(sum(b.dt for b in model.history), target)

    def test_constant_state_and_zero_diffusivity(self):
        model = DiffusionModel(CutCellGrid.uniform(5), [-2, 0, 4, 8, 10], 0)
        original = model.state.copy()
        model.step(10)
        np.testing.assert_allclose(model.state, original)

    def test_invalid_evolution_inputs(self):
        grid = CutCellGrid.uniform(2)
        for kwargs in ({'method':'rk4'}, {'safety':0}, {'atol':-1}, {'source':np.nan}):
            with self.assertRaises(ValueError):
                DiffusionModel(grid, [1, 2], **kwargs)
        model = DiffusionModel(grid, [1, 2])
        for dt in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                model.step(dt)
        self.assertEqual(model.iteration, 0)
        with self.assertRaises(ValueError):
            model.run_until(-1)
        with self.assertRaises(RuntimeError):
            model.run_until(10, max_steps=1)

    def test_nonfinite_state_step_is_not_committed(self):
        model = DiffusionModel(CutCellGrid.uniform(2), [1, 2])
        model.state[0] = np.nan
        with self.assertRaises(ValueError):
            model.step(.01)
        self.assertEqual(model.time, 0)
        self.assertEqual(len(model.history), 0)

    def test_second_order_cosine_cell_average_convergence(self):
        errors = []
        for n in (16, 32, 64):
            grid = CutCellGrid.uniform(n)
            # Exact cell averages; no conflation with point samples.
            c0 = .5 + .2 * np.cos(np.pi * grid.centers) * np.sinc(1 / (2*n))
            model = DiffusionModel(grid, c0, .1)
            model.run_until(.03, max_dt=1)
            exact = .5 + (c0 - .5) * np.exp(-.1 * np.pi**2 * model.time)
            errors.append(np.sqrt(np.dot(grid.volumes, (model.state-exact)**2)))
        self.assertTrue(all(a / b > 3.8 for a, b in zip(errors, errors[1:])), errors)

    def test_ssprk3_third_order_temporal_convergence(self):
        grid = CutCellGrid.uniform(8)
        initial = np.cos(np.pi * grid.centers)
        decay = -.1 * 4 * 8**2 * np.sin(np.pi/16)**2
        errors = []
        for dt in (.02, .01, .005):
            model = DiffusionModel(grid, initial)
            model.run_until(.1, max_dt=dt)
            errors.append(np.linalg.norm(model.state - initial * np.exp(decay * .1)))
        self.assertTrue(all(a/b > 7.5 for a,b in zip(errors, errors[1:])), errors)


if __name__ == '__main__':
    unittest.main()
