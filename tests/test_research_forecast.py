import unittest
import numpy as np
from cutcell import CutCellGrid
from cutcell.categorical_numerics import Coarsening
from cutcell.research import crps, field_score, multimodal_score, ConservativeFluxEnsemble
from cutcell.category import FiniteSet, FiniteMap, FiniteKernel


class ForecastScores(unittest.TestCase):
    def test_sorted_crps_against_independent_pairwise_definition(self):
        rng = np.random.default_rng(835)
        for members in (1, 2, 7, 32):
            x, y = rng.normal(size=(members, 3, 2)), rng.normal(size=(3, 2))
            pairs = np.abs(x[:, None]-x[None, :]).sum(axis=(0, 1))
            expected = np.mean(np.abs(x-y), axis=0)-pairs/(2*members**2)
            np.testing.assert_allclose(crps(x, y), expected, atol=1e-14)
            if members > 1:
                fair = np.mean(np.abs(x-y), axis=0)-pairs/(2*members*(members-1))
                np.testing.assert_allclose(crps(x, y, fair=True), fair, atol=1e-14)
        self.assertEqual(float(crps([0., 2.], 1.)), .5)
        self.assertEqual(float(crps([0., 2.], 1., fair=True)), 0.)
        self.assertEqual(float(crps([2.], 3.)), 1.)

    def test_empirical_replication_and_fair_sampling_assumption(self):
        x = np.array([0., 2.])
        self.assertEqual(float(crps(np.repeat(x, 2), 1.)), float(crps(x, 1.)))
        # Duplicating members changes fair CRPS: duplicated draws are not iid.
        self.assertNotEqual(float(crps(np.repeat(x, 2), 1., fair=True)), float(crps(x, 1., fair=True)))
        np.testing.assert_allclose(crps(x+100, 101), crps(x, 1.))

    def test_marginals_do_not_identify_joint_distribution(self):
        correlated = np.array([[-1., -1.], [1., 1.]])
        anticorrelated = np.array([[-1., 1.], [1., -1.]])
        a = field_score(correlated, [0, 0], [1, 1], pooled_weight=.3)
        b = field_score(anticorrelated, [0, 0], [1, 1], pooled_weight=.3)
        self.assertEqual(a['marginal'], b['marginal'])
        self.assertGreater(a['pooled_mean'], b['pooled_mean'])
        self.assertGreater(a['total'], b['total'])
        # Pool samples first; pooling pointwise scores would miss this.
        self.assertEqual(b['pooled_mean'], 0.)

    def test_modality_normalization_masks_and_unequal_area_weights(self):
        data = {'grid': {'ensemble': [[0, 2]], 'observation': [1, 4], 'weights': [1, 3], 'coefficient': 2.},
                'station': {'ensemble': [[0, np.nan]], 'observation': [3, np.nan], 'weights': [1, 1],
                            'valid': [True, False], 'coefficient': .5}}
        result = multimodal_score(data)
        self.assertEqual(result['total'], 2*(1/4+3*2/4)+.5*3)
        data['grid'] = {'ensemble': [[0, 0, 2, 2]], 'observation': [1, 1, 4, 4],
                        'weights': [1, 1, 3, 3], 'coefficient': 2.}
        self.assertEqual(multimodal_score(data)['total'], result['total'])

    def test_invalid_scores_fail_closed(self):
        for x, y, fair in [([], 0., False), ([1.], 0., True), ([np.nan], 0., False), ([[1., 2.]], [0.], False)]:
            with self.assertRaises(ValueError):
                crps(x, y, fair=fair)
        for weights in ([0, 0], [-1, 2], [1, np.inf]):
            with self.assertRaises(ValueError):
                field_score([[0, 1]], [1, 1], weights)
        with self.assertRaises(ValueError):
            multimodal_score({})
        with self.assertRaises(ValueError):
            multimodal_score({'empty': {'ensemble': [[np.nan]], 'observation': [np.nan], 'weights': [1],
                                        'valid': [False], 'coefficient': 1}})


class ConservativeSamples(unittest.TestCase):
    def test_memberwise_mass_and_pushforward_coarse_mass(self):
        grid = CutCellGrid([0, .1, .4, .6, 1.])
        modes = np.array([[0, 0], [.1, .2], [-.1, .3], [.1, -.4], [0, 0]])
        model = ConservativeFluxEnsemble(grid, modes)
        noise = np.random.default_rng(39).normal(size=(100, 2))
        state = np.array([.4, .5, .8, 1.])
        result = model.sample(state, 10., noise)
        np.testing.assert_allclose(result['mass_residuals'], 0, atol=1e-14)
        self.assertLess(result['actual_dt'], 10.)
        restriction = Coarsening(grid, (0, 2, 4))
        coarse = result['members'] @ restriction.restrict.T
        np.testing.assert_allclose(coarse @ restriction.coarse_volumes, state @ grid.volumes, atol=1e-14)
        # Shared functional noise creates a correlated field of rank <= 2.
        self.assertLessEqual(np.linalg.matrix_rank(np.cov(result['members'], rowvar=False), tol=1e-12), 2)
        again = model.sample(state, 10., noise)
        np.testing.assert_array_equal(result['members'], again['members'])

    def test_disconnected_components_and_no_clipping(self):
        grid = CutCellGrid.uniform(4, apertures=[0, 1, 0, 1, 0])
        modes = np.array([[0], [1], [0], [-2], [0]])
        model = ConservativeFluxEnsemble(grid, modes, diffusivity=0.)
        result = model.sample([1, 2, 3, 4], .1, [[1], [-1]])
        for cells in ([0, 1], [2, 3]):
            np.testing.assert_allclose(result['members'][:, cells] @ grid.volumes[cells],
                                       np.asarray([1, 2, 3, 4])[cells] @ grid.volumes[cells])
        with self.assertRaises(ArithmeticError):
            model.sample([0, 0, 0, 0], 1., [[1]], require_nonnegative=True)
        self.assertTrue(np.any(model.sample([0, 0, 0, 0], 1., [[1]])['members'] < 0))
        bad_modes = modes.copy(); bad_modes[2] = 1
        with self.assertRaises(ValueError):
            ConservativeFluxEnsemble(grid, bad_modes)
        with self.assertRaises(ValueError):
            model.sample([1, 1, 1, 1], .1, [[1, 2]])


class StochasticCategoryTests(unittest.TestCase):
    def test_composition_associativity_and_expectation_duality(self):
        obj = FiniteSet((0, 1))
        k = FiniteKernel(obj, obj, [[.2, .8], [.5, .5]])
        l = FiniteKernel(obj, obj, [[.8, .2], [.1, .9]])
        identity = FiniteKernel.identity(obj)
        np.testing.assert_allclose(k.then(l).then(k).matrix, k.then(l.then(k)).matrix)
        np.testing.assert_array_equal(k.then(identity).matrix, k.matrix)
        p, f = np.array([.3, .7]), np.array([1., 4.])
        self.assertAlmostEqual(k.pushforward(p) @ f, p @ k.expectation(f))
        np.testing.assert_allclose(k.then(l).expectation(f), k.expectation(l.expectation(f)))
        np.testing.assert_allclose(k.tensor(l).pushforward(np.kron(p, p)), np.kron(k.pushforward(p), l.pushforward(p)))

    def test_deterministic_embedding_preserves_composition(self):
        a, b = FiniteSet((0, 1, 2)), FiniteSet(('a', 'b'))
        f, g = FiniteMap(a, b, ('a', 'a', 'b')), FiniteMap(b, a, (2, 1))
        np.testing.assert_array_equal(FiniteKernel.deterministic(f.then(g)).matrix,
                                     FiniteKernel.deterministic(f).then(FiniteKernel.deterministic(g)).matrix)

    def test_lumpability_is_an_extra_condition(self):
        fine, coarse = FiniteSet((0, 1, 2)), FiniteSet(('a', 'b'))
        q = FiniteMap(fine, coarse, ('a', 'a', 'b'))
        k = FiniteKernel(fine, fine, [[.2, .3, .5], [.4, .1, .5], [.1, .2, .7]])
        kc = FiniteKernel(coarse, coarse, [[.5, .5], [.3, .7]])
        np.testing.assert_allclose(k.lumpability_defect(q, kc), 0, atol=1e-15)
        bad = FiniteKernel(fine, fine, [[.2, .4, .4], [.4, .1, .5], [.1, .2, .7]])
        self.assertGreater(np.max(np.abs(bad.lumpability_defect(q, kc))), .09)

    def test_invalid_probabilities_are_not_normalized_or_clipped(self):
        a = FiniteSet((0, 1))
        for matrix in ([[.2, .7], [.2, .8]], [[-.1, 1.1], [0, 1]], [[np.nan, 0], [0, 1]]):
            with self.assertRaises(ValueError):
                FiniteKernel(a, a, matrix)
        k = FiniteKernel.identity(a)
        with self.assertRaises(ValueError):
            k.pushforward([1, 1])
        with self.assertRaises(ValueError):
            k.expectation([np.nan, 0])
        empty = FiniteSet(())
        self.assertEqual(FiniteKernel.identity(empty).matrix.shape, (0, 0))


class ResearchIntegration(unittest.TestCase):
    def test_report_source_versions_and_validation_gates(self):
        from cutcell.research.showcase import showcase
        report = showcase()
        self.assertEqual(report['sources']['korn_2026']['arxiv'], '2608.25679v3')
        self.assertEqual(report['sources']['weathernext3']['arxiv'], '2609.03582v1')
        self.assertTrue(all(x <= report['residual_absolute_tolerance'] for x in report['residuals'].values()))
        self.assertGreater(abs(report['physical_tracer_content_change_not_conserved']), .1)
        self.assertEqual(report['forecast_members'], 64)
