import unittest
import numpy as np
from cutcell import CutCellGrid
from cutcell.research import pseudo_density, consistent_tracer_step


class TransportTests(unittest.TestCase):
    def test_pseudo_mass_content_constants_and_bounds(self):
        grid = CutCellGrid([0, .1, .4, 1.])
        r = pseudo_density([-.1, .2, .05], 1.)
        c = np.array([1., 3., 2.])
        flux = np.array([0., .04, -.03, 0.])
        for _ in range(10):
            result = consistent_tracer_step(grid, r, c, flux, .1)
            self.assertAlmostEqual(result['pseudo_mass_residual'], 0., places=14)
            self.assertAlmostEqual(result['tracer_content_residual'], 0., places=14)
            self.assertGreaterEqual(result['tracer'].min(), c.min()-1e-14)
            self.assertLessEqual(result['tracer'].max(), c.max()+1e-14)
            self.assertLessEqual(result['weighted_second_moment_change'], 1e-14)
            r, c = result['relative_density'], result['tracer']
        constant = consistent_tracer_step(grid, r, np.full(3, 7.), flux, .1)
        np.testing.assert_allclose(constant['tracer'], 7., atol=1e-14)

    def test_physical_and_pseudo_contents_are_not_interchangeable(self):
        grid = CutCellGrid.uniform(2)
        result = consistent_tracer_step(grid, [1., 1.], [1., 3.], [0., .1, 0.], 1.)
        self.assertAlmostEqual(result['tracer_content_residual'], 0.)
        self.assertGreater(abs(result['physical_content_change']), .1)
        # Incorrectly holding density fixed preserves a different quantity and
        # fails constant-tracer preservation under nonzero divergence.
        bad_constant = np.ones(2)-np.diff([0., .1, 0.])/grid.volumes
        self.assertFalse(np.allclose(bad_constant, 1.))

    def test_solid_cells_and_component_conservation(self):
        grid = CutCellGrid.uniform(5, volume_fractions=[1, 1, 0, 1, 1])
        c = np.array([1., 2., 99., 3., 4.])
        result = consistent_tracer_step(grid, np.ones(5), c, [0, .1, 0, 0, -.05, 0], .1)
        new_mass = grid.volumes*result['relative_density']
        for ids in ([0, 1], [3, 4]):
            self.assertAlmostEqual(new_mass[ids] @ result['tracer'][ids], grid.volumes[ids] @ c[ids])
        self.assertEqual(result['tracer'][2], 99.)

    def test_flux_cfl_density_and_boundary_rejections(self):
        grid = CutCellGrid.uniform(2)
        for flux in ([.1, 0, 0], [0, 0, .1], [0, 1., 0]):
            with self.assertRaises(ValueError):
                consistent_tracer_step(grid, [1, 1], [1, 1], flux, 1.)
        for psi in ([-1., 0.], [np.inf, 0.]):
            with self.assertRaises(ValueError):
                pseudo_density(psi, 1.)
        with self.assertRaises(ValueError):
            pseudo_density([0., 0.], 0.)
