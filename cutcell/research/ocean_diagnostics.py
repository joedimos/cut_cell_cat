"""Source-traceable diagnostics from Korn, arXiv:2608.25679v3.

These routines expose algebraic quantities from Sections 2.1--2.2 and 3.1--3.3.
They are diagnostics and finite-volume face formulas, not a complete AC/DC ocean
integrator. Physical and numerical dissipation are kept separate by construction.
"""
import numpy as np


def _positive_scalar(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _nonnegative_scalar(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(value)


def _vector(value, size, name, *, nonnegative=False, positive=False):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array.copy()


def thin_fluid_calibration(rho0, gravity, full_depth, velocity_scale):
    """Korn (9)--(10): alpha=rho0*g*H and c_AC=sqrt(gH)."""
    rho0 = _positive_scalar(rho0, "rho0")
    gravity = _positive_scalar(gravity, "gravity")
    full_depth = _positive_scalar(full_depth, "full_depth")
    velocity_scale = _nonnegative_scalar(velocity_scale, "velocity_scale")
    alpha = rho0 * gravity * full_depth
    acoustic_speed = float(np.sqrt(alpha / rho0))
    froude = velocity_scale / acoustic_speed
    return {"alpha": alpha, "acoustic_speed": acoustic_speed,
            "barotropic_froude": froude,
            "barotropic_froude_squared": froude*froude}


def flux_corrected_reconstruction(tracer, pseudo_mass_flux, limiter):
    """Korn (42): blend upwind and centred face values with 0 <= lambda <= 1."""
    c = np.asarray(tracer, dtype=float)
    if c.ndim != 1 or len(c) < 2 or not np.all(np.isfinite(c)):
        raise ValueError("tracer must be a finite vector with at least two cells")
    flux = _vector(pseudo_mass_flux, len(c)-1, "pseudo_mass_flux")
    lam = _vector(limiter, len(c)-1, "limiter")
    if np.any(lam < 0) or np.any(lam > 1):
        raise ValueError("limiter coefficients must lie in [0, 1]")
    centered = .5 * (c[:-1] + c[1:])
    upwind = np.where(flux >= 0, c[:-1], c[1:])
    return upwind + lam * (centered - upwind)


def tracer_variance_diagnostics(volumes, relative_density, tracer, pseudo_mass_flux,
                                diffusivity, geometric_factor, reconstruction, *, rho0=1.):
    """Korn (38)--(43): explicit physical chi and reconstruction sink D_num."""
    rho0 = _positive_scalar(rho0, "rho0")
    c = np.asarray(tracer, dtype=float)
    if c.ndim != 1 or len(c) < 2 or not np.all(np.isfinite(c)):
        raise ValueError("tracer must be a finite vector with at least two cells")
    n = len(c)
    volumes = _vector(volumes, n, "volumes", positive=True)
    relative_density = _vector(relative_density, n, "relative_density", positive=True)
    flux = _vector(pseudo_mass_flux, n-1, "pseudo_mass_flux")
    diffusivity = _vector(diffusivity, n-1, "diffusivity", nonnegative=True)
    geometry = _vector(geometric_factor, n-1, "geometric_factor", positive=True)
    reconstruction = _vector(reconstruction, n-1, "reconstruction")
    jump = np.diff(c)
    centered = .5 * (c[:-1] + c[1:])
    weighted_variance = .5 * rho0 * float(np.dot(volumes * relative_density, c*c))
    chi_integral = 2. * float(np.sum(diffusivity * geometry * jump*jump))
    numerical_sink = -float(np.sum(flux * jump * (reconstruction - centered)))
    variance_tendency = -rho0 * numerical_sink - .5 * rho0 * chi_integral
    upwind_upper_bound = .5 * float(np.sum(np.abs(flux) * jump*jump))
    return {"weighted_variance": weighted_variance,
            "chi_integral": chi_integral,
            "numerical_sink": numerical_sink,
            "variance_tendency": variance_tendency,
            "upwind_sink_upper_bound": upwind_upper_bound}


def energy_dissipation_diagnostics(volumes, relative_density, nu_h, omega_z,
                                   nu_v, omega_h_squared, nu_d, divergence):
    """Korn (46)--(47): physical viscous and compressibility dissipation.

    All arrays are cell-local. ``relative_density`` is rho_AC/rho0,
    ``omega_h_squared`` is |omega_H|^2. The two returned rates are kept separate:
    physical dissipation epsilon is not inflated by the AC divergence reservoir.
    """
    volumes = np.asarray(volumes, dtype=float)
    if volumes.ndim != 1 or len(volumes) == 0 or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("volumes must be a finite positive vector")
    n = len(volumes)
    r = _vector(relative_density, n, "relative_density", positive=True)
    nu_h = _vector(nu_h, n, "nu_h", nonnegative=True)
    omega_z = _vector(omega_z, n, "omega_z")
    nu_v = _vector(nu_v, n, "nu_v", nonnegative=True)
    omega_h_squared = _vector(omega_h_squared, n, "omega_h_squared", nonnegative=True)
    nu_d = _vector(nu_d, n, "nu_d", nonnegative=True)
    divergence = _vector(divergence, n, "divergence")
    physical = float(np.sum(volumes * r * (nu_h*omega_z*omega_z + nu_v*omega_h_squared)))
    compressibility = float(np.sum(volumes * r * nu_d * divergence*divergence))
    return {"physical_dissipation": physical,
            "compressibility_dissipation": compressibility}


def osborn_cox_diffusivity(chi_integral, domain_volume, mean_gradient):
    """Korn (45): Osborn--Cox diffusivity under the stated steady/local assumptions."""
    chi_integral = _nonnegative_scalar(chi_integral, "chi_integral")
    domain_volume = _positive_scalar(domain_volume, "domain_volume")
    if (isinstance(mean_gradient, (bool, np.bool_)) or not np.isscalar(mean_gradient)
            or not np.isfinite(mean_gradient) or mean_gradient == 0):
        raise ValueError("mean_gradient must be finite and nonzero")
    mean_gradient = float(mean_gradient)
    return chi_integral / (2. * domain_volume * mean_gradient*mean_gradient)


def mixing_diagnostics(epsilon_b, epsilon, stratification_n2, *,
                       compressibility_error=0., advection_error=0., closure_residual=0.):
    """Korn (54)--(56): q, Gamma, R_f and K_rho from separated budget terms."""
    epsilon_b = _nonnegative_scalar(epsilon_b, "epsilon_b")
    epsilon = _positive_scalar(epsilon, "epsilon")
    n2 = _positive_scalar(stratification_n2, "stratification_n2")
    epsilon_c = _nonnegative_scalar(compressibility_error, "compressibility_error")
    if (isinstance(advection_error, (bool, np.bool_)) or not np.isscalar(advection_error)
            or not np.isfinite(advection_error)):
        raise ValueError("advection_error must be finite")
    if (isinstance(closure_residual, (bool, np.bool_)) or not np.isscalar(closure_residual)
            or not np.isfinite(closure_residual)):
        raise ValueError("closure_residual must be finite")
    gamma = epsilon_b / epsilon
    flux_richardson = epsilon_b / (epsilon_b + epsilon)
    contamination_ratio = (epsilon_c + abs(float(advection_error))
                           + abs(float(closure_residual))) / epsilon
    k_rho = epsilon_b / n2
    return {"flux_coefficient_gamma": gamma,
            "flux_richardson": flux_richardson,
            "diapycnal_diffusivity": k_rho,
            "numerical_contamination_ratio": contamination_ratio}
