import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from cutcell import CutCellGrid, DiffusionModel, DiffusionOperator, BoundaryCondition as BC
from cutcell.io import atomic_json
from verified_simulator import VerifiedCategoricalSimulator


class ProductionTests(unittest.TestCase):
    def test_local_error_cannot_hide_in_global_cancellation(self):
        model = DiffusionModel(CutCellGrid.uniform(2), [1, 1], method='euler')
        before = model.state.copy()
        with patch.object(model.operator, 'tendency', return_value=np.array([1., -1.])):
            with self.assertRaisesRegex(ArithmeticError, 'local cell'):
                model.step(.01)
        np.testing.assert_array_equal(model.state, before)
        self.assertEqual(model.time, 0)
        self.assertEqual(model.total_exchange, 0)
        self.assertEqual(model.history, [])

    def test_cumulative_ledger_catches_external_mass_change(self):
        model = DiffusionModel(CutCellGrid.uniform(2), [1, 2])
        model.step(.001)
        model.state += 1
        with self.assertRaisesRegex(ArithmeticError, 'cumulative'):
            model.step(.001)
        self.assertEqual(model.iteration, 1)

    def test_history_retention_preserves_global_ledger(self):
        model = DiffusionModel(CutCellGrid.uniform(8), np.ones(8), source=.2,
                               left=BC('flux', .03), history_limit=8)
        for _ in range(20000):
            model.step(.0001)
        self.assertEqual(len(model.history), 8)
        self.assertEqual(model.iteration, 20000)
        self.assertAlmostEqual(model.total_exchange, 2 * .23, places=12)
        self.assertLess(abs(model.history[-1].cumulative_residual), 1e-11)

    def test_nonnegative_policy_rejects_excessive_sink_without_clipping(self):
        model = DiffusionModel(CutCellGrid.uniform(1), [1], source=-2, require_nonnegative=True)
        with self.assertRaisesRegex(ArithmeticError, 'negative concentration'):
            model.step(1)
        self.assertEqual(model.state[0], 1)
        self.assertEqual(model.iteration, 0)

    def test_invalid_retention_and_unrepresentable_metrics(self):
        for value in (0, True, 2.5):
            with self.assertRaises(ValueError):
                DiffusionModel(CutCellGrid.uniform(1), [1], history_limit=value)
        with self.assertRaises(ValueError):
            CutCellGrid([0, 1e-200], volume_fractions=1e-200)
        with self.assertRaises(ValueError):
            DiffusionOperator(CutCellGrid([0, 1e-200]), diffusivity=1e200)
        with self.assertRaises(TypeError):
            DiffusionOperator(CutCellGrid.uniform(1), left=0)

    def test_atomic_output_retains_previous_result_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            atomic_json(path, {'valid':1})
            with self.assertRaises(ValueError):
                atomic_json(path, {'invalid':float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'valid':1})
            with patch('cutcell.io.os.replace', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    atomic_json(path, {'valid':2})
            self.assertEqual(json.loads(path.read_text()), {'valid':1})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_require_lean_fails_closed(self):
        with patch('lean_verification.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                VerifiedCategoricalSimulator(2, require_lean=True)

    def test_facade_retained_history_and_export_counts(self):
        sim = VerifiedCategoricalSimulator(4, history_limit=3)
        sim.run_verified(8)
        self.assertEqual(len(sim.model.history), 3)
        self.assertTrue(all(len(v) == 3 for v in sim.verification_history.values()))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'result.json'
            sim.save_results(p)
            data = json.loads(p.read_text())
        self.assertEqual(data['total_steps'], 8)
        self.assertEqual(data['retained_steps'], 3)

    def test_disconnected_extreme_placeholders_do_not_corrupt_flux(self):
        grid = CutCellGrid.uniform(3, volume_fractions=[1, 0, 1])
        op = DiffusionOperator(grid)
        with np.errstate(all='raise'):
            np.testing.assert_array_equal(op.flux([1e308, -1e308, 1e308]), np.zeros(4))

    def test_large_signed_field_does_not_fail_on_mass_cancellation(self):
        grid = CutCellGrid.uniform(8)
        model = DiffusionModel(grid, 1e8*np.cos(np.pi*grid.centers))
        model.run_until(.03)
        self.assertLess(abs(model.history[-1].cumulative_residual), 1e-6)
        self.assertTrue(all(b.passed for b in model.history))

    def test_randomized_evolution_invariants(self):
        rng = np.random.default_rng(491)
        for case in range(40):
            n = int(rng.integers(2, 30))
            grid = CutCellGrid(np.r_[0, np.cumsum(rng.uniform(.02, .5, n))],
                               rng.uniform(.0001, 1, n), rng.uniform(0, 1, n+1))
            model = DiffusionModel(grid, rng.uniform(0, 10, n), rng.uniform(0, 3, n+1),
                                   method='euler' if case % 2 else 'ssprk3')
            mass, lo, hi = model.mass(), model.state.min(), model.state.max()
            for _ in range(20):
                b = model.step(1)
                self.assertLess(abs(model.mass()-mass), 1e-11)
                self.assertGreaterEqual(b.minimum, lo-1e-12)
                self.assertLessEqual(b.maximum, hi+1e-12)
                self.assertLess(b.max_local_residual, 1e-12)


if __name__ == '__main__':
    unittest.main()

class IndependentReferenceTests(unittest.TestCase):
    def test_cut_cell_evolution_against_independent_matrix_exponential(self):
        # Assemble a mass-symmetric graph generator without calling the solver's
        # flux or tendency routines, then exponentiate its eigenvalues exactly.
        grid = CutCellGrid([0, .15, .4, .7, 1], [.3, .8, 1, .6], [.0, .4, .8, .6, 0.])
        diffusivity = np.array([.1, .2, .15, .3, .1])
        n = grid.size
        stiffness = np.zeros((n, n))
        for face in range(1, n):
            g = grid.apertures[face]*diffusivity[face]/(grid.centers[face]-grid.centers[face-1])
            i, j = face-1, face
            stiffness[i, i] -= g
            stiffness[j, j] -= g
            stiffness[i, j] += g
            stiffness[j, i] += g
        root_volume = np.sqrt(grid.volumes)
        symmetric = stiffness/root_volume[:, None]/root_volume[None, :]
        eigenvalues, basis = np.linalg.eigh(symmetric)
        initial = np.array([.2, 1.3, .7, .1])
        time = .04
        exact = (basis @ (np.exp(time*eigenvalues) * (basis.T @ (root_volume*initial))))/root_volume
        errors = []
        for dt in (.001, .0005, .00025):
            model = DiffusionModel(grid, initial, diffusivity)
            model.run_until(time, max_dt=dt)
            errors.append(np.linalg.norm(model.state-exact))
        self.assertLess(errors[-1], 1e-8)
        self.assertTrue(all(a/b > 7 for a,b in zip(errors, errors[1:])), errors)
