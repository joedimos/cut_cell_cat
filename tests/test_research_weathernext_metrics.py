"""WeatherNext 3 pooled-CRPS conventions from Appendix A.2.2."""
import unittest
import numpy as np
from cutcell.research import pooled_crps


class PooledCRPSTests(unittest.TestCase):
    def test_average_pooling_scores_pooled_fields_not_pooled_point_scores(self):
        ensemble = np.array(((-1., -1., 1., 1.), (1., 1., -1., -1.)))
        observation = np.zeros(4)
        pools = ((0, 1), (1, 2), (2, 3), (3, 0))
        result = pooled_crps(ensemble, observation, pools, reducer='mean', weights=(1, 2, 2, 1))
        expected_x = np.array(((-1., 0., 1., 0.), (1., 0., -1., 0.)))
        np.testing.assert_allclose(result['pooled_values'], expected_x)
        np.testing.assert_allclose(result['pooled_observation'], 0.)
        self.assertGreater(result['weighted_score'], 0.)

    def test_max_pooling_detects_extreme_structure(self):
        ensemble = np.array(((0., 2., 0.), (1., 0., 3.)))
        observation = np.array((0., 1., 1.))
        result = pooled_crps(ensemble, observation, ((0, 1), (1, 2)), reducer='max')
        np.testing.assert_allclose(result['pooled_values'], ((2., 2.), (1., 3.)))
        np.testing.assert_allclose(result['pooled_observation'], (1., 1.))
        self.assertEqual(result['reducer'], 'max')

    def test_latitude_like_weights_change_only_spatial_aggregation(self):
        ensemble = np.array(((0., 0., 4.), (0., 2., 0.)))
        observation = np.zeros(3)
        pools = ((0,), (1,), (2,))
        uniform = pooled_crps(ensemble, observation, pools, weights=(1, 1, 1))
        weighted = pooled_crps(ensemble, observation, pools, weights=(4, 2, 1))
        np.testing.assert_allclose(uniform['scores'], weighted['scores'])
        self.assertNotEqual(uniform['weighted_score'], weighted['weighted_score'])

    def test_invalid_pooled_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            pooled_crps([[0, 1]], [0, 1], (), reducer='mean')
        with self.assertRaises(ValueError):
            pooled_crps([[0, 1]], [0, 1], ((0,),), reducer='median')
        with self.assertRaises(ValueError):
            pooled_crps([[0, 1]], [0, 1], ((2,),))
        with self.assertRaises(ValueError):
            pooled_crps([[0, np.nan]], [0, 1], ((0,),))


if __name__ == '__main__':
    unittest.main()
