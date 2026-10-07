import unittest
import numpy as np
from cutcell.research import ColumnPressureSplit, dispersion


def dense_vertical(grid):
    z = grid.z
    n = z.size
    stiffness = np.zeros((n, n))
    for j in range(n-1):
        k = 1/(z.centers[j+1]-z.centers[j])
        stiffness[j:j+2, j:j+2] += k*np.array([[1., -1.], [-1., 1.]])
    stiffness[-1, -1] += 1/(z.faces[-1]-z.centers[-1])
    return -stiffness/z.volumes[:, None]


def dense_horizontal(grid):
    x = grid.x
    stiffness = np.zeros((x.size, x.size))
    for i in range(x.size-1):
        k = 1/(x.centers[i+1]-x.centers[i])
        stiffness[i:i+2, i:i+2] += k*np.array([[1., -1.], [-1., 1.]])
    return -stiffness/x.volumes[:, None]


class PressureTests(unittest.TestCase):
    def test_column_and_aggregated_pressure_against_independent_dense_solve(self):
        grid = ColumnPressureSplit([0, .2, .8, 2.], [-1., -.8, -.3, 0.])
        source = np.random.default_rng(73).normal(size=grid.shape)
        split = grid.split(source)
        vertical, horizontal = dense_vertical(grid), dense_horizontal(grid)
        q = split['column_pressure_over_density']
        np.testing.assert_allclose(q, np.linalg.solve(vertical, source.T).T, atol=1e-13)
        np.testing.assert_allclose(grid.vertical(q), q @ vertical.T, atol=1e-13)
        np.testing.assert_allclose(grid.horizontal(q), horizontal @ q, atol=1e-13)
        np.testing.assert_allclose(split['column_residual'], 0, atol=1e-13)
        np.testing.assert_allclose(split['split_identity_residual'], 0, atol=1e-13)
        full = np.kron(horizontal, np.eye(grid.z.size))+np.kron(np.eye(grid.x.size), vertical)
        correction = np.linalg.solve(full, split['residual_source'].ravel()).reshape(grid.shape)
        reference = np.linalg.solve(full, source.ravel()).reshape(grid.shape)
        np.testing.assert_allclose(q+correction, reference, atol=1e-13)
        np.testing.assert_allclose(full @ reference.ravel(), source.ravel(), atol=1e-13)

    def test_single_cell_column_boundary_conditions(self):
        grid = ColumnPressureSplit([0, 1], [-2, 0])
        # q_zz=3, bottom derivative=0, q(top)=0; FV one-cell coefficient -1/2.
        result = grid.split([[3.]])
        np.testing.assert_allclose(result['column_pressure_over_density'], [[-6.]])
        np.testing.assert_array_equal(result['residual_source'], [[0.]])

    def test_manufactured_pressure_spatial_convergence(self):
        errors = []
        for n in (12, 24, 48):
            grid = ColumnPressureSplit(np.linspace(0, 2, n+1), np.linspace(-1, 0, n+1))
            x, z = grid.x.centers[:, None], grid.z.centers[None, :]
            q = np.cos(np.pi*x/2)*np.cos(np.pi*(z+1)/2)
            source = -(np.pi/2)**2*q
            recovered = grid.column_solve(source)
            errors.append(np.max(np.abs(recovered-q)))
        self.assertTrue(all(a/b > 3.9 for a, b in zip(errors, errors[1:])))

    def test_acoustic_wave_time_convergence_and_vertical_cfl(self):
        grid = ColumnPressureSplit([0, 1], np.linspace(-1, 0, 9))
        lz = dense_vertical(grid)
        eigenvalues, eigenvectors = np.linalg.eigh(-lz)
        mode = eigenvectors[:, 0][None, :]
        alpha, stop = 2., .15
        exact = mode*np.cos(np.sqrt(alpha*eigenvalues[0])*stop)
        errors = []
        for steps in (4, 8, 16):
            p, u, w = mode.copy(), np.zeros((2, 8)), np.zeros((1, 9))
            for _ in range(steps):
                result = grid.acoustic_stage(u, w, p, stop/steps, alpha)
                p, u, w = result['pressure_over_density'], result['u'], result['w']
            errors.append(np.max(np.abs(p-exact)))
        self.assertTrue(all(a/b > 3.9 for a, b in zip(errors, errors[1:])))
        thin = ColumnPressureSplit([0, 1], np.linspace(-.1, 0, 9))
        a = grid.acoustic_stage(np.zeros((2, 8)), np.zeros((1, 9)), mode, .2, 2.)
        b = thin.acoustic_stage(np.zeros((2, 8)), np.zeros((1, 9)), mode, .2, 2.)
        self.assertGreater(b['substeps'], 5*a['substeps'])
        self.assertLess(a['substep_dt']**2*2*a['spectral_bound'], 4.)

    def test_verlet_time_reversibility(self):
        grid = ColumnPressureSplit([0, .2, 1], [-1, -.4, 0])
        rng = np.random.default_rng(41)
        p, u, w = rng.normal(size=(2, 2)), np.zeros((3, 2)), np.zeros((2, 3))
        u[1] = rng.normal(size=2)
        w[:, 1:] = rng.normal(size=(2, 2))
        result = grid.acoustic_stage(u, w, p, .1, 10.)
        reverse = grid.acoustic_stage(-result['u'], -result['w'], result['pressure_over_density'], .1, 10.)
        np.testing.assert_allclose(reverse['pressure_over_density'], p, atol=1e-13)
        np.testing.assert_allclose(reverse['u'], -u, atol=1e-13)
        np.testing.assert_allclose(reverse['w'], -w, atol=1e-13)

    def test_dispersion_matches_full_fourier_system(self):
        for kx, kz, n, f, alpha in [(1., 4., 2., .5, 100.), (3., 1., 2., .2, 1000.), (0., 3., 1., .4, 50.)]:
            d = dispersion(kx, kz, n, f, alpha)
            # Independent 5-variable Fourier dynamics, Korn (82)-(86).
            matrix = np.zeros((5, 5), dtype=complex)  # u,v,w,b,p
            matrix[0, 1] = f*(1-kx*kx/(kz*kz))
            matrix[0, 3], matrix[0, 4] = -kx/kz, -1j*kx
            matrix[1, 0] = -f
            matrix[2, 1], matrix[2, 4] = -kx*f/kz, -1j*kz
            matrix[3, 2] = -n*n
            matrix[4, 0], matrix[4, 2] = -1j*alpha*kx, -1j*alpha*kz
            eigenvalues = np.linalg.eigvals(matrix)
            np.testing.assert_allclose(eigenvalues.real, 0, atol=1e-11)
            frequencies = sorted(v.imag**2 for v in eigenvalues if v.imag > 1e-8)
            np.testing.assert_allclose(frequencies, [d['acdc_slow_squared'], d['acdc_fast_squared']], rtol=1e-10)

    def test_dispersion_asymptotic_error_is_not_exact(self):
        errors = []
        for alpha in (100., 200., 400.):
            d = dispersion(1., 3., 2., .5, alpha)
            actual = d['acdc_slow_squared']-d['incompressible_omega_squared']
            self.assertGreater(actual, 0)
            self.assertNotEqual(actual, d['acdc_leading_error_squared'])
            errors.append(actual)
            self.assertAlmostEqual(actual/d['acdc_leading_error_squared'], 1., delta=.01)
            pure = d['pure_ac_slow_squared']-d['incompressible_omega_squared']
            self.assertLess(pure, 0)
            self.assertAlmostEqual(actual/-pure, 1/9, delta=.002)
        self.assertTrue(all(1.99 < a/b < 2.01 for a, b in zip(errors, errors[1:])))
        self.assertEqual(dispersion(1., 2., 0., 0., 10.)['acdc_slow_squared'], 0.)

    def test_pressure_validation(self):
        grid = ColumnPressureSplit([0, 1], [-1, 0])
        for source in ([1.], [[np.nan]]):
            with self.assertRaises(ValueError):
                grid.split(source)
        for args in [(1, 0, 1, 0, 1), (1, 1, -1, 0, 1), (1, 1, 1, 0, .001), (1, 1, 1, 0, -1)]:
            with self.assertRaises(ValueError):
                dispersion(*args)
        with self.assertRaises(ValueError):
            grid.acoustic_stage([[1.], [0.]], [[0., 0.]], [[0.]], 1., 1.)
        with self.assertRaises(ValueError):
            grid.acoustic_stage([[0.], [0.]], [[0., 0.]], [[0.]], 100., 100., max_substeps=1)
