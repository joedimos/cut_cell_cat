"""Korn 2608.25679v3 physical versus compressibility energy dissipation."""
import unittest
import numpy as np
from cutcell.research import energy_dissipation_diagnostics


class EnergyDissipationTests(unittest.TestCase):
    def test_equations_46_and_47_are_kept_separate(self):
        volumes = np.array((1., 2., .5))
        relative_density = np.array((1., 1.01, .99))
        nu_h = np.array((2., 2., 2.))
        omega_z = np.array((1., -2., .5))
        nu_v = np.array((3., 3., 3.))
        omega_h_squared = np.array((4., 1., 9.))
        nu_d = np.array((5., 5., 5.))
        divergence = np.array((.1, -.2, .3))
        result = energy_dissipation_diagnostics(
            volumes, relative_density, nu_h, omega_z,
            nu_v, omega_h_squared, nu_d, divergence)
        physical = np.sum(volumes*relative_density*(nu_h*omega_z**2 + nu_v*omega_h_squared))
        compressibility = np.sum(volumes*relative_density*nu_d*divergence**2)
        self.assertAlmostEqual(result['physical_dissipation'], physical)
        self.assertAlmostEqual(result['compressibility_dissipation'], compressibility)
        self.assertGreater(result['physical_dissipation'], result['compressibility_dissipation'])

    def test_zero_divergence_has_no_compressibility_dissipation(self):
        result = energy_dissipation_diagnostics(
            [1., 1.], [1., 1.], [1., 1.], [1., 2.],
            [1., 1.], [3., 4.], [7., 8.], [0., 0.])
        self.assertEqual(result['compressibility_dissipation'], 0.)
        self.assertGreater(result['physical_dissipation'], 0.)

    def test_invalid_energy_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            energy_dissipation_diagnostics([], [], [], [], [], [], [], [])
        with self.assertRaises(ValueError):
            energy_dissipation_diagnostics([1], [1], [-1], [1], [1], [1], [1], [1])
        with self.assertRaises(ValueError):
            energy_dissipation_diagnostics([1], [0], [1], [1], [1], [1], [1], [1])


if __name__ == '__main__':
    unittest.main()
