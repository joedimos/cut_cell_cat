"""Equation-level diagnostics from Korn, arXiv:2608.25679v3."""
import unittest
import numpy as np
from cutcell.research.ocean_diagnostics import (
    thin_fluid_calibration, flux_corrected_reconstruction,
    tracer_variance_diagnostics, osborn_cox_diffusivity, mixing_diagnostics,
)


class ThinFluidCalibrationTests(unittest.TestCase):
    def test_equations_9_and_10(self):
        result = thin_fluid_calibration(1025., 9.81, 4000., 1.)
        self.assertAlmostEqual(result["alpha"], 1025.*9.81*4000.)
        self.assertAlmostEqual(result["acoustic_speed"], np.sqrt(9.81*4000.))
        self.assertAlmostEqual(result["barotropic_froude"], 1./np.sqrt(9.81*4000.))
        self.assertAlmostEqual(result["barotropic_froude_squared"],
                               result["barotropic_froude"]**2)
        self.assertLess(result["barotropic_froude_squared"], 3e-5)

    def test_bad_calibration_inputs_are_rejected(self):
        for args in ((0, 9.81, 4000, 1), (1025, -9.81, 4000, 1),
                     (1025, 9.81, 0, 1), (1025, 9.81, 4000, -1)):
            with self.assertRaises(ValueError):
                thin_fluid_calibration(*args)


class TracerVarianceTests(unittest.TestCase):
    def setUp(self):
        self.c = np.array((1., 3., 2., 5.))
        self.flux = np.array((2., -1., 3.))
        self.volumes = np.array((1., 2., 1.5, .5))
        self.relative_density = np.array((1., 1.01, .99, 1.))
        self.kappa = np.array((.2, .1, .3))
        self.geometry = np.array((2., 4., 1.))

    def test_equations_39_to_43_separate_physical_and_upwind_sinks(self):
        upwind = flux_corrected_reconstruction(self.c, self.flux, np.zeros(3))
        report = tracer_variance_diagnostics(
            self.volumes, self.relative_density, self.c, self.flux,
            self.kappa, self.geometry, upwind, rho0=1025.)
        jump = np.diff(self.c)
        expected_num = .5*np.sum(np.abs(self.flux)*jump**2)
        expected_chi = 2*np.sum(self.kappa*self.geometry*jump**2)
        self.assertAlmostEqual(report["numerical_sink"], expected_num)
        self.assertAlmostEqual(report["upwind_sink_upper_bound"], expected_num)
        self.assertAlmostEqual(report["chi_integral"], expected_chi)
        self.assertAlmostEqual(report["variance_tendency"],
                               -1025.*expected_num-.5*1025.*expected_chi)

    def test_centered_reconstruction_has_zero_numerical_sink(self):
        centered = flux_corrected_reconstruction(self.c, self.flux, np.ones(3))
        report = tracer_variance_diagnostics(
            self.volumes, self.relative_density, self.c, self.flux,
            self.kappa, self.geometry, centered)
        self.assertAlmostEqual(report["numerical_sink"], 0.)
        self.assertLessEqual(report["variance_tendency"], 0.)

    def test_limited_sink_matches_corollary_3_2(self):
        limiter = np.array((.25, .5, .75))
        reconstructed = flux_corrected_reconstruction(self.c, self.flux, limiter)
        report = tracer_variance_diagnostics(
            self.volumes, self.relative_density, self.c, self.flux,
            np.zeros(3), self.geometry, reconstructed)
        jump = np.diff(self.c)
        expected = .5*np.sum((1-limiter)*np.abs(self.flux)*jump**2)
        self.assertAlmostEqual(report["numerical_sink"], expected)
        self.assertGreaterEqual(report["numerical_sink"], 0.)
        self.assertLessEqual(report["numerical_sink"], report["upwind_sink_upper_bound"])

    def test_osborn_cox_excludes_the_numerical_sink(self):
        self.assertAlmostEqual(osborn_cox_diffusivity(12., 3., 2.), .5)
        with self.assertRaises(ValueError):
            osborn_cox_diffusivity(1., 1., 0.)

    def test_general_reconstruction_sink_is_not_forced_positive(self):
        centered = .5*(self.c[:-1]+self.c[1:])
        anti_upwind = centered + np.sign(self.flux)*np.diff(self.c)
        report = tracer_variance_diagnostics(
            self.volumes, self.relative_density, self.c, self.flux,
            np.zeros(3), self.geometry, anti_upwind)
        self.assertLess(report["numerical_sink"], 0.)


class MixingDiagnosticsTests(unittest.TestCase):
    def test_equations_54_to_56(self):
        result = mixing_diagnostics(.2, 1., .01,
                                    compressibility_error=.01,
                                    advection_error=-.02,
                                    closure_residual=.03)
        self.assertAlmostEqual(result["flux_coefficient_gamma"], .2)
        self.assertAlmostEqual(result["flux_richardson"], 1/6)
        self.assertAlmostEqual(result["diapycnal_diffusivity"], 20.)
        self.assertAlmostEqual(result["numerical_contamination_ratio"], .06)
        gamma = result["flux_coefficient_gamma"]
        rf = result["flux_richardson"]
        self.assertAlmostEqual(gamma, rf/(1-rf))
        self.assertAlmostEqual(rf, gamma/(1+gamma))

    def test_physical_dissipation_denominator_must_be_positive(self):
        with self.assertRaises(ValueError):
            mixing_diagnostics(.1, 0., 1.)
        with self.assertRaises(ValueError):
            mixing_diagnostics(-.1, 1., 1.)
        with self.assertRaises(ValueError):
            mixing_diagnostics(.1, 1., -1.)


if __name__ == "__main__":
    unittest.main()
