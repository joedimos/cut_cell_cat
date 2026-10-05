import json
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from cutcell import CutCellGrid, DiffusionModel
from lean_verification import LeanVerificationServer, LeanCodeGenerator
from verified_simulator import VerifiedCategoricalSimulator
from semantic_search import SemanticPatternSearch
from knowledge_graph import CategoricalKnowledgeGraph


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.budget = DiffusionModel(CutCellGrid.uniform(2), [0, 1]).step(.01)

    def test_numerical_success_is_not_a_theorem(self):
        passed, error, meta = LeanVerificationServer(mock_mode=True).verify_budget(self.budget)
        self.assertTrue(passed)
        self.assertFalse(meta['lean_proven'])
        self.assertEqual(meta['backend'], 'numerical')

    def test_tampered_budget_fails(self):
        bad = replace(self.budget, mass_after=self.budget.mass_after+.1)
        passed, error, meta = LeanVerificationServer(mock_mode=True).verify_budget(bad)
        self.assertFalse(passed)
        self.assertGreater(error, .09)
        self.assertFalse(meta['lean_proven'])

    def test_nonfinite_budget_rejected(self):
        with self.assertRaises(ValueError):
            LeanVerificationServer(mock_mode=True).verify_budget(replace(self.budget, mass_after=float('nan')))

    def test_snapshot_fluxes_cannot_prove_conservation(self):
        passed, _, meta = LeanVerificationServer(mock_mode=True).verify_conservation([1, 0], [1, -1])
        self.assertFalse(passed)
        self.assertFalse(meta['lean_proven'])

    def test_certificate_does_not_truncate_tiny_error(self):
        budget = replace(self.budget, mass_before=0., mass_after=1e-9,
                         boundary_exchange=0., source_exchange=0., tolerance=1e-12)
        code = LeanVerificationServer._certificate(budget)
        self.assertNotIn('sorry', code)
        self.assertNotIn('axiom', code)
        self.assertNotIn('native_decide', code)
        self.assertIn('Nat', code)
        import re
        lhs, rhs = re.search(r'\((\d+) : Nat\) ≤ (\d+)', code).groups()
        self.assertGreater(int(lhs), int(rhs))

    def test_lean_failure_timeout_and_unavailable_do_not_prove(self):
        for result in (subprocess.CompletedProcess([], 1, '', 'error'),
                       subprocess.CompletedProcess([], 0, 'warning', ''),
                       subprocess.TimeoutExpired('lean', 1), FileNotFoundError()):
            with self.subTest(result=result):
                server = LeanVerificationServer(lean_path='/configured/lean')
                with patch('lean_verification.subprocess.run') as run:
                    if isinstance(result, Exception):
                        run.side_effect = result
                    else:
                        run.return_value = result
                    passed, _, meta = server.verify_budget(self.budget)
                self.assertTrue(passed)
                self.assertFalse(meta['lean_proven'])

    def test_successful_process_status_and_explicit_executable(self):
        # Orchestration test only. The actual Lean test below requires Lean.
        server = LeanVerificationServer(lean_path='/configured/lean')
        with patch('lean_verification.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')) as run:
            _, _, meta = server.verify_budget(self.budget)
        self.assertTrue(meta['lean_proven'])
        self.assertEqual(run.call_args.args[0][0], '/configured/lean')

    @unittest.skipUnless(shutil.which('lean'), 'Lean executable not installed')
    def test_real_lean_accepts_valid_and_rejects_invalid_certificate(self):
        server = LeanVerificationServer()
        _, _, meta = server.verify_budget(self.budget)
        self.assertTrue(meta['lean_proven'], meta)
        bad = replace(self.budget, mass_after=self.budget.mass_after+1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'Invalid.lean'
            path.write_text(server._certificate(bad))
            process = subprocess.run([server.lean_path, str(path)], capture_output=True, timeout=10)
        self.assertNotEqual(process.returncode, 0)

    def test_legacy_generator_evolves_full_requested_time(self):
        generator = LeanCodeGenerator(LeanVerificationServer(mock_mode=True))
        step = generator.generate_verified_evolution_step(1., .5)
        result = step([0, 1])
        self.assertAlmostEqual(sum(result), 1)
        self.assertGreater(result[0], .45)

    def test_petri_duplicate_input_requires_multiple_tokens(self):
        generator = LeanCodeGenerator(LeanVerificationServer(mock_mode=True))
        step = generator.generate_petri_net_step()
        self.assertEqual(step([1, 0], [([0, 0], [1])]), [1, 0])
        self.assertEqual(step([2, 0], [([0, 0], [1])]), [0, 1])


class IntegrationTests(unittest.TestCase):
    def test_facade_serialization_and_actual_clock(self):
        sim = VerifiedCategoricalSimulator(12, dt=10)
        sim.run_verified(3)
        sim.run_verified(2)
        self.assertEqual(sim.model.iteration, 5)
        self.assertTrue(all(sim.verification_history['conservation_verified']))
        self.assertEqual(sum(sim.verification_history['lean_theorems_proven']), 0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'results.json'
            sim.save_results(path)
            data = json.loads(path.read_text())
        self.assertEqual(data['schema_version'], 2)
        self.assertEqual(len(data['final_fluxes']), 13)
        self.assertEqual(data['time'], sum(b['dt'] for b in data['budgets']))
        self.assertFalse(data['lean_used'])
        self.assertEqual(len(data['reference_commit']), 40)

    def test_zero_steps_and_multiscale(self):
        sim = VerifiedCategoricalSimulator(2)
        self.assertEqual(sim.run_verified(0), [])
        with self.assertRaises(NotImplementedError):
            VerifiedCategoricalSimulator(use_multiscale=True)

    def test_graph_bounded_and_uses_simulation_time(self):
        sim = VerifiedCategoricalSimulator(4)
        sim.kg = CategoricalKnowledgeGraph(max_patterns=4)
        sim.run_verified(10)
        self.assertEqual(len(sim.kg.patterns), 4)
        self.assertEqual(len(sim.kg.semantic_search.pattern_database), 4)
        self.assertEqual(sim.kg.patterns[-1]['time'], sim.model.time)
        ids = {p['id'] for p in sim.kg.patterns}
        for k, v in sim.kg.relationships.items():
            self.assertIn(k, ids)
            self.assertIn(v['previous_observation'], ids)
        self.assertTrue(sim.search_patterns('mass balance'))
        self.assertFalse(any(p['type'] == 'flux_composition' for p in sim.kg.patterns))

    def test_fixed_feature_coordinates_and_search(self):
        index = SemanticPatternSearch(2)
        index.add_pattern('a', {'type':'gradient', 'gradient':2}, 'High gradient')
        index.add_pattern('b', {'type':'conservation', 'conservation_error':2}, 'Mass conservation')
        a, b = [p['embedding'] for p in index.pattern_database]
        self.assertEqual(a[1], 0)
        self.assertEqual(b[0], 0)
        self.assertEqual(index.search('slope')[0]['pattern']['id'], 'a')
        index.add_pattern('b', {'type':'conservation'}, 'Mass conservation')
        self.assertEqual(len(index.pattern_database), 2)
        self.assertEqual(index.search(''), [])
        with self.assertRaises(ValueError):
            index.search('mass', -1)


if __name__ == '__main__':
    unittest.main()
