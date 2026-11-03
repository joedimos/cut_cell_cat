import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cutcell.research.mitgcm import BarotropicGrid, ghost_kernel, half_plane_fractions, reflected_ghost


def rect_grid(ny=1, nx=2):
    metric = np.ones((ny, nx))
    west = metric.copy()
    south = metric.copy()
    west[:, 0] = 0
    south[0, :] = 0
    return BarotropicGrid(rac=metric, dxg=metric, dyg=metric, dxc=metric, dyc=metric,
                         drf=[1.], hfac_c=metric[None], hfac_w=west[None], hfac_s=south[None])


class MitgcmGeometryTests(unittest.TestCase):
    def test_mds_vertical_thickness_and_horizontal_aperture_are_independent(self):
        fields = dict(RAC=[[10., 20.]], DXG=[[2., 3.]], DYG=[[4., 5.]], DXC=[[1., 2.]],
                      DYC=[[3., 4.]], DRF=[[2.], [8.]], hFacC=[[[1., 1.]], [[0.5, 1.]]],
                      hFacW=[[[0., 1.]], [[0., 0.25]]], hFacS=np.zeros((2, 1, 2)))
        grid = BarotropicGrid.from_mds(fields.__getitem__, area_fraction=[[0.5, 0.75]],
                                       west_aperture=[[0., 0.25]], south_aperture=[[0., 0.]])
        assert_array_equal(grid.area, [5., 15.])
        assert_array_equal(grid.depth, [6., 10.])
        assert_array_equal(grid.cross_section, [5.])
        assert_array_equal(grid.distance, [2.])
        assert_array_equal(grid.boundary, [[-1], [1]])
        assert_array_equal(grid.pack_velocity([[50., 2.]], [[30., 40.]]), [2.])
        fields['hFacW'][0][0][0] = 1.
        with self.assertRaisesRegex(ValueError, 'periodic'):
            BarotropicGrid.from_mds(fields.__getitem__)

    def test_oblique_geometry_and_grid_aligned_wall(self):
        area, west, south = half_plane_fractions([0., 1., 2.], [0., 1.], (1., 1.), 1.5)
        assert_array_equal(area, [[0.125, 0.875]])
        assert_array_equal(west, [[0., 0.5]])
        assert_array_equal(south, [[0., 0.5]])
        area, west, south = half_plane_fractions([0., 1., 2.], [0., 1.], (1., 0.), 1.)
        assert_array_equal(area, [[0., 1.]])
        assert_array_equal(west, [[0., 0.]])
        assert_array_equal(south, [[0., 1.]])

    def test_full_face_geometry_includes_east_and_north_exterior_segments(self):
        area, west, south = half_plane_fractions([0., 1., 2.], [0., 1.], (1., 1.), 1.5, full_faces=True)
        assert_array_equal(area, [[0.125, 0.875]])
        assert_array_equal(west, [[0., 0.5, 1.]])
        assert_array_equal(south, [[0., 0.5], [0.5, 1.]])

    def test_quotient_merges_opposite_crossings_with_orientation(self):
        grid = rect_grid(2, 2)
        a, q, b_coarse = grid.quotient([1, 0, 0, 1])
        assert_array_equal(a @ grid.boundary, b_coarse @ q)
        assert_array_equal(q, [[-1, 1, -1, 1]])
        _, internal, _ = grid.quotient([0, 0, 0, 0])
        self.assertEqual(internal.shape, (0, 4))


class BarotropicTests(unittest.TestCase):
    def test_two_cell_closed_form_and_energy_identity(self):
        grid = rect_grid()
        eta, velocity, flux, defect = grid.step([1., -1.], [0.], 2., gravity=1.)
        assert_allclose(eta, [1/9, -1/9], atol=2e-15, rtol=0)
        assert_allclose(velocity, [4/9], atol=2e-15, rtol=0)
        assert_array_equal(flux, velocity)
        assert_allclose(defect, 0., atol=2e-15, rtol=0)
        energy = grid.energy(eta, velocity, gravity=1.)
        increment_energy = grid.energy(eta - [1., -1.], velocity, gravity=1.)
        assert_allclose(energy + increment_energy, 1., atol=2e-15, rtol=0)

    def test_zero_face_basin(self):
        grid = rect_grid(1, 1)
        eta, velocity, flux, defect = grid.step(0.2, [], 5.)
        assert_allclose(eta, [0.2])
        assert_array_equal(defect, 0.)
        self.assertEqual(velocity.size + flux.size, 0)
        a, q, coarse = grid.quotient([6])
        assert_array_equal(a @ grid.boundary, coarse @ q)

    def test_tracer_moves_conservatively_into_and_out_of_dry_cells(self):
        grid = rect_grid()
        volume, tracer = grid.tracer_step([1., 0.], [0.3, 9.], [1.], 1.)
        assert_array_equal(volume, [0., 1.])
        assert_array_equal(tracer, [0., 0.3])
        volume, tracer = grid.tracer_step(volume, tracer, [-0.5], 1.)
        assert_array_equal(volume, [0.5, 0.5])
        assert_array_equal(tracer, [0.3, 0.3])
        with self.assertRaisesRegex(ValueError, 'donor CFL'):
            grid.tracer_step(volume, tracer, [1.], 1.)


class GhostTests(unittest.TestCase):
    def test_conditioned_donors_and_wall_reflection(self):
        kernel = ghost_kernel(0.5, 0.5, [True, True, True, False], [True, False, True, True])
        assert_array_equal(kernel.matrix, [[0.5, 0., 0.5, 0.]])
        discharge = [[2., 4.], [500., 600.], [4., 2.], [-100., 200.]]
        result = reflected_ghost(kernel, (1., 0.), [2., 7., 2., 90.], [0., 7., 1., 30.], discharge)
        self.assertEqual(result['surface'], 2.)
        self.assertEqual(result['bed'], 0.5)
        self.assertEqual(result['depth'], 1.5)
        assert_array_equal(result['discharge'], [-3., 3.])
        self.assertIsNone(ghost_kernel(0., 0., [False, True, False, False]))
        self.assertIsNone(ghost_kernel(0.01, 0., [True, True, True, True], [False, True, False, False]))
        self.assertIsNotNone(ghost_kernel(0.01, 0., [False, True, False, False]))


if __name__ == '__main__':
    unittest.main()
