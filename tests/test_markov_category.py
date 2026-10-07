"""Finite Markov-category laws, Bayesian inversion, and strong lumpability."""
import unittest
import numpy as np
from cutcell.category import FiniteSet, FiniteMap, product, FiniteKernel, UNIT


class MarkovCategoryTests(unittest.TestCase):
    def setUp(self):
        self.x = FiniteSet(("dry", "wet"))
        self.y = FiniteSet(("no-rain", "rain"))
        self.k = FiniteKernel(self.x, self.y, ((.9, .1), (.2, .8)))

    def test_state_pushforward_and_joint(self):
        prior = np.array((.7, .3))
        q = self.k.pushforward(prior)
        self.assertTrue(np.allclose(q, (.69, .31)))
        obj, joint = self.k.joint(prior)
        expected = np.array((.63, .07, .06, .24))
        self.assertEqual(obj, product(self.x, self.y)[0])
        self.assertTrue(np.allclose(joint, expected))
        self.assertAlmostEqual(joint.sum(), 1.)
        state = FiniteKernel.state(self.x, prior)
        self.assertEqual(state.source, UNIT)
        self.assertTrue(np.allclose(state.then(self.k).matrix[0], q))

    def test_all_kernels_discard_but_only_deterministic_kernels_copy(self):
        discard_lhs = self.k.then(FiniteKernel.discard(self.y))
        self.assertTrue(np.allclose(discard_lhs.matrix,
                                    FiniteKernel.discard(self.x).matrix))
        deterministic = FiniteKernel.deterministic(
            FiniteMap(self.x, self.y, ("no-rain", "rain")))
        self.assertTrue(deterministic.is_deterministic())
        self.assertTrue(np.allclose(deterministic.copy_defect(), 0.))
        self.assertFalse(self.k.is_deterministic())
        self.assertGreater(np.max(np.abs(self.k.copy_defect())), .1)

    def test_tensor_symmetry_and_associator_are_deterministic_isomorphism_data(self):
        z = FiniteSet((0, 1, 2))
        swap = FiniteKernel.swap(self.x, self.y)
        back = FiniteKernel.swap(self.y, self.x)
        self.assertTrue(np.allclose(swap.then(back).matrix,
                                    FiniteKernel.identity(swap.source).matrix))
        associator = FiniteKernel.associator(self.x, self.y, z)
        self.assertTrue(associator.is_deterministic())
        self.assertEqual(len(associator.source), len(associator.target))
        self.assertEqual(len(set(associator.matrix.argmax(axis=1))),
                         len(associator.target))

    def test_bayes_inverse_reconstructs_joint_on_supported_outputs(self):
        prior = np.array((.7, .3))
        q = self.k.pushforward(prior)
        reverse = self.k.bayes_inverse(prior)
        forward_joint = prior[:, None] * self.k.matrix
        reverse_joint = q[:, None] * reverse.matrix
        self.assertTrue(np.allclose(reverse_joint, forward_joint.T))
        self.assertTrue(np.allclose(reverse.pushforward(q), prior))

    def test_bayes_inverse_exposes_null_event_choice(self):
        target = FiniteSet(("seen", "never"))
        kernel = FiniteKernel(self.x, target, ((1., 0.), (1., 0.)))
        prior = np.array((.4, .6))
        with self.assertRaisesRegex(ValueError, "zero-probability"):
            kernel.bayes_inverse(prior)
        reverse = kernel.bayes_inverse(prior, null_policy="prior")
        self.assertTrue(np.allclose(reverse.matrix[1], prior))

    def test_likelihood_conditioning(self):
        prior = np.array((.7, .3))
        posterior = self.k.posterior(prior, (0., 1.))
        expected = np.array((.07, .24)) / .31
        self.assertTrue(np.allclose(posterior, expected))
        with self.assertRaisesRegex(ValueError, "zero probability"):
            self.k.posterior(prior, (0., 0.))

    def test_strong_lumpability_constructs_unique_coarse_kernel(self):
        fine = FiniteSet((0, 1, 2, 3))
        coarse = FiniteSet(("a", "b"))
        partition = FiniteMap(fine, coarse, ("a", "a", "b", "b"))
        kernel = FiniteKernel(fine, fine, (
            (.3, .2, .1, .4),
            (.4, .1, .2, .3),
            (.1, .2, .4, .3),
            (.2, .1, .3, .4),
        ))
        lumped = kernel.lumped(partition)
        self.assertTrue(np.allclose(lumped.matrix, ((.5, .5), (.3, .7))))
        self.assertTrue(np.allclose(kernel.lumpability_defect(partition, lumped), 0.))

    def test_non_lumpable_partition_is_rejected(self):
        fine = FiniteSet((0, 1, 2, 3))
        coarse = FiniteSet(("a", "b"))
        partition = FiniteMap(fine, coarse, ("a", "a", "b", "b"))
        kernel = FiniteKernel(fine, fine, (
            (.3, .2, .1, .4),
            (.8, .1, .05, .05),
            (.1, .2, .4, .3),
            (.2, .1, .3, .4),
        ))
        with self.assertRaisesRegex(ValueError, "not strongly lumpable"):
            kernel.lumped(partition)

    def test_invalid_partition_and_tolerances_are_rejected(self):
        fine = FiniteSet((0, 1))
        coarse = FiniteSet(("a", "b"))
        non_surjective = FiniteMap(fine, coarse, ("a", "a"))
        kernel = FiniteKernel.identity(fine)
        with self.assertRaisesRegex(ValueError, "surjective"):
            kernel.lumped(non_surjective)
        with self.assertRaises(ValueError):
            self.k.is_deterministic(atol=-1.)


if __name__ == "__main__":
    unittest.main()
