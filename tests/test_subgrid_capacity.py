"""Exact planar bed capacities and implicit frozen-flux research contracts.

These tests are not claims about full native MITgcm or moving-mask dynamics.
"""
import unittest

import numpy as np

from cutcell.research.subgrid_capacity import (
    NonlinearCapacityPressure,
    PlanarBedCutCell,
    required_vertical_exchange,
)


SQUARE = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
TRIANGLE = ((0., 0.), (1., 0.), (0., 1.))


class CapacityTests(unittest.TestCase):
    def test_rectangular_linear_bed_exact_volume_area_and_faces(self):
        cell = PlanarBedCutCell(SQUARE, (1., 0., 0.))
        for eta in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5):
            expected_volume = 0.5 * eta**2 if eta <= 1 else eta - 0.5
            expected_area = min(eta, 1.)
            self.assertAlmostEqual(cell.volume_below(eta), expected_volume, places=13)
            self.assertAlmostEqual(cell.wetted_area(eta), expected_area, places=13)
            self.assertAlmostEqual(cell.face_section((0, 0), (1, 0), eta),
                                   expected_volume, places=13)
            self.assertAlmostEqual(cell.face_section((0, 0), (0, 1), eta),
                                   eta, places=13)
        self.assertEqual(cell.volume_below(-1.), 0.)
        self.assertEqual(cell.wetted_area(-1.), 0.)
        self.assertEqual(cell.face_section((1, 0), (1, 1), 0.7), 0.)

    def test_triangular_sloping_capacity_is_cubic(self):
        cell = PlanarBedCutCell(TRIANGLE, (1., 1., 0.))
        for eta in (0.1, 0.4, 0.9, 1.):
            self.assertAlmostEqual(cell.wetted_area(eta), eta**2/2, places=13)
            self.assertAlmostEqual(cell.volume_below(eta), eta**3/6, places=13)
        self.assertAlmostEqual(cell.volume_below(1.5), 5./12., places=13)
        self.assertAlmostEqual(cell.volume_below(1.e-5), (1.e-5)**3/6,
                               delta=5.e-26)

    def test_fixed_planform_fraction_does_not_capture_shoreline_capacity(self):
        cell = PlanarBedCutCell(SQUARE, (1., 0., 0.))
        ratio = cell.volume_below(.75) / cell.volume_below(.25)
        self.assertAlmostEqual(ratio, 9.)
        # A fixed-area linear capacity fitted at eta=0.25 predicts ratio 3.
        self.assertGreater(abs(ratio - .75/.25), 5.)

    def test_layer_volumes_telescope(self):
        cell = PlanarBedCutCell(SQUARE, (1., 1., 0.))
        interfaces = np.array([-0.5, .25, .5, 1., 2.])
        layers = cell.layer_volumes(1.35, interfaces)
        self.assertTrue(np.all(layers >= 0.))
        self.assertAlmostEqual(np.sum(layers), cell.volume_below(1.35), places=13)
        self.assertAlmostEqual(cell.layer_volumes(.2, interfaces)[-1], 0.)

    def test_exact_hydrostatic_potential_has_correct_conjugate(self):
        cell = PlanarBedCutCell(SQUARE, (1., 0.4, -0.15))
        gravity = 9.81
        for eta in (.1, .3, .8, 1.4):
            h = 1.e-5
            numerator = (cell.hydrostatic_potential(eta+h, gravity=gravity)
                         - cell.hydrostatic_potential(eta-h, gravity=gravity))
            denominator = cell.volume_below(eta+h) - cell.volume_below(eta-h)
            self.assertAlmostEqual(numerator / denominator, gravity * eta, delta=3.e-7)

    def test_invalid_and_immutable_geometry(self):
        with self.assertRaises(ValueError):
            PlanarBedCutCell(tuple(reversed(SQUARE)), (1., 0., 0.))
        with self.assertRaises(ValueError):
            PlanarBedCutCell(SQUARE, (np.nan, 0., 0.))
        cell = PlanarBedCutCell(SQUARE, (1., 0., 0.))
        with self.assertRaises(ValueError):
            cell.vertices[0, 0] = 3.
        with self.assertRaises(ValueError):
            cell.layer_volumes(.8, (0., .5, .5, 1.))
        with self.assertRaises(ValueError):
            cell.face_section((0., 0.), (0., 0.), .5)


class PressureTests(unittest.TestCase):
    def test_exact_flat_capacity_reduces_to_linear_implicit_system(self):
        a = PlanarBedCutCell(SQUARE, (0., 0., 0.))
        solver = NonlinearCapacityPressure((a, a), [0], [1], [1.5], gravity=2.)
        before = np.array([1.2, 0.8])
        dt = 0.1
        result = solver.step(before, [0.], dt)
        expected_difference = .4 / (1. + 2*2.*1.5*dt**2)
        self.assertAlmostEqual(result.surface[0] + result.surface[1], 2., places=13)
        self.assertAlmostEqual(result.surface[0] - result.surface[1], expected_difference,
                               places=13)
        self.assertLess(np.max(np.abs(result.residual)), 1.e-11)
        self.assertLess(solver.energy(result.surface, result.face_flux),
                        solver.energy(before, [0.]))

    def test_nonlinear_capacity_conserves_actual_volume_and_dissipates_energy(self):
        cell = PlanarBedCutCell(SQUARE, (1., 0., 0.))
        solver = NonlinearCapacityPressure((cell, cell), [0], [1], [.4])
        old_surface = np.array([.8, .3])
        old_flux = np.array([.01])
        dt = .15
        start_volume = sum(c.volume_below(z) for c,z in zip(solver.cells, old_surface))
        result = solver.step(old_surface, old_flux, dt)
        self.assertGreaterEqual(np.min(result.volumes), 0.)
        self.assertAlmostEqual(sum(result.volumes), start_volume, places=12)
        change = result.volumes - np.array(
            [c.volume_below(z) for c,z in zip(solver.cells, old_surface)])
        np.testing.assert_allclose(change, dt * solver.incidence @ result.face_flux,
                                   rtol=0, atol=1.e-11)
        self.assertLessEqual(solver.energy(result.surface, result.face_flux),
                             solver.energy(old_surface, old_flux) + 1.e-10)

    def test_one_face_zero_conductance_cannot_carry_transport(self):
        cell = PlanarBedCutCell(SQUARE, (0., 0., 0.))
        solver = NonlinearCapacityPressure((cell, cell), [0], [1], [0.])
        with self.assertRaisesRegex(ValueError, 'closed face'):
            solver.step([1., 1.], [.02], .1)
        with self.assertRaisesRegex(ValueError, 'closed-face flux'):
            solver.energy([1., 1.], [.02])

    def test_unique_internal_vertical_reconciliation(self):
        old = np.array([[.1, .2, .3], [.2, .3, .4]])
        new = np.array([[.15, .25, .20], [.1, .3, .5]])
        horizontal = np.zeros_like(old)
        exchanges = required_vertical_exchange(old, new, horizontal)
        self.assertEqual(exchanges.shape, (2, 4))
        np.testing.assert_array_equal(exchanges[:, 0], [0., 0.])
        np.testing.assert_array_equal(exchanges[:, -1], [0., 0.])
        np.testing.assert_allclose(new-old, horizontal+exchanges[:, :-1]-exchanges[:, 1:],
                                   atol=1.e-15, rtol=0)
        broken = new.copy()
        broken[0, 0] += .02
        with self.assertRaisesRegex(ValueError, 'does not close'):
            required_vertical_exchange(old, broken, horizontal)


if __name__ == '__main__':
    unittest.main()
