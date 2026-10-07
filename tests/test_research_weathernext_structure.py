"""WeatherNext 3 structural equations represented as finite stochastic diagrams."""
import unittest
import numpy as np
from cutcell.category import FiniteSet, FiniteMap, FiniteKernel, product
from cutcell.research.weathernext import (
    SecondOrderWeatherKernel, FunctionalGeneratorKernel, MultimodalLatentDiagram,
)


class SecondOrderTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.state = FiniteSet((0, 1))
        pair, _, _ = product(self.state, self.state)
        self.transition = FiniteKernel(pair, self.state, (
            (.9, .1), (.3, .7), (.6, .4), (.2, .8),
        ))
        self.model = SecondOrderWeatherKernel(self.state, self.transition)

    def test_pair_lift_is_row_stochastic_and_shifts_history(self):
        lifted = self.model.lifted()
        np.testing.assert_allclose(lifted.matrix.sum(axis=1), 1.)
        pair_index = {xy: i for i, xy in enumerate(self.model.pair_state.elements)}
        row = lifted.matrix[pair_index[(0, 1)]]
        expected = np.zeros(4)
        expected[pair_index[(1, 0)]] = .3
        expected[pair_index[(1, 1)]] = .7
        np.testing.assert_allclose(row, expected)

    def test_trajectory_probability_is_second_order_factorization(self):
        self.assertAlmostEqual(self.model.trajectory_probability((0, 1), (1, 0)), .14)
        self.assertEqual(self.model.trajectory_probability((0, 1), ()), 1.)

    def test_rollout_equals_repeated_pair_kernel_composition(self):
        p0 = np.array((0., 1., 0., 0.))
        lifted = self.model.lifted()
        np.testing.assert_allclose(self.model.rollout(p0, 2), p0 @ lifted.matrix @ lifted.matrix)
        np.testing.assert_array_equal(self.model.rollout(p0, 0), p0)

    def test_invalid_second_order_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            SecondOrderWeatherKernel(self.state, FiniteKernel.identity(self.state))
        with self.assertRaises(ValueError):
            self.model.rollout((1., 1., 0., 0.), 1)
        with self.assertRaises(ValueError):
            self.model.rollout((1., 0., 0., 0.), -1)
        with self.assertRaises(ValueError):
            self.model.trajectory_probability((0,), (1,))


class FunctionalGeneratorTests(unittest.TestCase):
    def test_shared_noise_marginalizes_to_one_field_valued_kernel(self):
        condition = FiniteSet(("cold", "warm"))
        noise = FiniteSet(("z0", "z1"))
        # Each target is an entire two-location field, not one point.
        target = FiniteSet(((0, 0), (1, 1), (0, 1), (1, 0)))
        domain, _, _ = product(condition, noise)
        response = FiniteMap(domain, target, ((0, 0), (1, 1), (0, 1), (1, 0)))
        generator = FunctionalGeneratorKernel(condition, noise, target, response, (.25, .75))
        kernel = generator.kernel()
        np.testing.assert_allclose(kernel.matrix[0], (.25, .75, 0, 0))
        np.testing.assert_allclose(kernel.matrix[1], (0, 0, .25, .75))
        # The construction never creates independently mixed fields (0,1)/(1,0)
        # for the cold condition; one noise draw chooses a complete target field.
        self.assertEqual(kernel.matrix[0, 2], 0.)
        self.assertEqual(kernel.matrix[0, 3], 0.)

    def test_functional_generator_validation(self):
        condition = FiniteSet((0,))
        noise = FiniteSet((0, 1))
        target = FiniteSet(("a", "b"))
        domain, _, _ = product(condition, noise)
        response = FiniteMap(domain, target, ("a", "b"))
        with self.assertRaises(ValueError):
            FunctionalGeneratorKernel(condition, noise, target, response, (1., 1.))
        with self.assertRaises(ValueError):
            FunctionalGeneratorKernel(condition, noise, target,
                                      FiniteMap(condition, target, ("a",)), (.5, .5))


class MultimodalProcessorTests(unittest.TestCase):
    def setUp(self):
        self.fine = FiniteSet(("f0", "f1"))
        self.coarse = FiniteSet(("c0", "c1"))
        self.latent = FiniteSet(("z0", "z1"))
        fine_enc = FiniteKernel.deterministic(FiniteMap(self.fine, self.latent, ("z0", "z1")))
        fine_dec = FiniteKernel.deterministic(FiniteMap(self.latent, self.fine, ("f0", "f1")))
        coarse_enc = FiniteKernel.deterministic(FiniteMap(self.coarse, self.latent, ("z0", "z1")))
        coarse_dec = FiniteKernel.deterministic(FiniteMap(self.latent, self.coarse, ("c0", "c1")))
        self.diagram = MultimodalLatentDiagram(
            self.latent,
            {"fine": fine_enc, "coarse": coarse_enc},
            {"fine": fine_dec, "coarse": coarse_dec},
        )
        self.processor = FiniteKernel(self.latent, self.latent, ((.8, .2), (.1, .9)))

    def test_shared_processor_composes_native_encoders_and_decoders(self):
        translated = self.diagram.translate("fine", "coarse", self.processor)
        np.testing.assert_allclose(translated.matrix, self.processor.matrix)
        self.assertEqual(translated.source, self.fine)
        self.assertEqual(translated.target, self.coarse)
        np.testing.assert_allclose(self.diagram.roundtrip_defect("fine"), 0.)

    def test_resolution_naturality_is_extra_not_architectural_given(self):
        restriction = FiniteKernel.deterministic(FiniteMap(self.fine, self.coarse, ("c0", "c1")))
        np.testing.assert_allclose(
            self.diagram.resolution_naturality_defect("fine", "coarse", restriction, self.processor),
            0., atol=1e-15)
        swapped = FiniteKernel.deterministic(FiniteMap(self.latent, self.coarse, ("c1", "c0")))
        nonnatural = MultimodalLatentDiagram(
            self.latent, self.diagram.encoders,
            {"fine": self.diagram.decoders["fine"], "coarse": swapped})
        defect = nonnatural.resolution_naturality_defect("fine", "coarse", restriction, self.processor)
        self.assertGreater(np.max(np.abs(defect)), .5)

    def test_invalid_latent_diagrams_are_rejected(self):
        with self.assertRaises(ValueError):
            MultimodalLatentDiagram(self.latent, {}, {})
        with self.assertRaises(ValueError):
            MultimodalLatentDiagram(self.latent,
                                    {"fine": FiniteKernel.identity(self.fine)},
                                    {"fine": self.diagram.decoders["fine"]})
        with self.assertRaises(ValueError):
            self.diagram.translate("fine", "coarse", FiniteKernel.identity(self.fine))


if __name__ == "__main__":
    unittest.main()
