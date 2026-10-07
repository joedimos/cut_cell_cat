"""Stationary 1-D specialization of Korn (78)-(79), hypotheses H1-H4.

Updates pseudo-mass and tracer content with identical face fluxes and forward
Euler time levels. This module consumes a flux; it does not compute AC dynamics.
"""
import math
import numpy as np
from ..grid import vector
from .pressure import positive


def pseudo_density(psi, alpha):
    alpha = positive(alpha, 'alpha')
    psi = np.asarray(psi, dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        r = 1 + psi / alpha
    if psi.ndim != 1 or not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError('psi must be a finite vector with 1+psi/alpha > 0')
    return r


def consistent_tracer_step(grid, relative_density, tracer, pseudo_mass_flux, dt):
    """Closed-boundary upwind update, returning a new state without mutation.

    Flux is integrated over face area and points in the positive coordinate
    direction. It must be pseudo-mass flux r_face*u_face*area, not volume flux.
    Outgoing pseudo-mass must be strictly less than donor mass during the step.
    The upwind update is then a convex combination with no artificial clipping.
    """
    dt = positive(dt, 'dt')
    r = vector(relative_density, grid.size, 'relative_density')
    c = vector(tracer, grid.size, 'tracer')
    flux = vector(pseudo_mass_flux, grid.size+1, 'pseudo_mass_flux')
    if np.any(r <= 0) or flux[0] != 0 or flux[-1] != 0 or np.any(flux[grid.apertures == 0] != 0):
        raise ValueError('positive density and closed external/impermeable face fluxes required')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        mass = grid.volumes*r
        outflow = np.maximum(flux[1:], 0) + np.maximum(-flux[:-1], 0)
        if np.any(dt*outflow[grid.active] >= mass[grid.active]):
            raise ValueError('outgoing pseudo-mass violates the strict donor CFL bound')
        face_tracer = np.zeros(grid.size+1)
        face_tracer[1:-1] = np.where(flux[1:-1] >= 0, c[:-1], c[1:])
        new_mass = mass-dt*np.diff(flux)
        content = mass*c
        new_content = content-dt*np.diff(flux*face_tracer)
        if np.any(new_mass[grid.active] <= 0):
            raise ArithmeticError('nonpositive pseudo-mass')
        new_c, new_r = c.copy(), r.copy()
        new_c[grid.active] = new_content[grid.active]/new_mass[grid.active]
        new_r[grid.active] = new_mass[grid.active]/grid.volumes[grid.active]
        variance_before = float(np.dot(mass, c*c))
        variance_after = float(np.dot(new_mass, new_c*new_c))
    if not all(np.all(np.isfinite(a)) for a in (new_c, new_r, new_content)):
        raise FloatingPointError('transport result not representable')
    total = lambda a: math.fsum(float(x) for x in a)
    return {'relative_density': new_r, 'tracer': new_c,
            'pseudo_mass_residual': total(new_mass)-total(mass),
            'tracer_content_residual': total(new_content)-total(content),
            'weighted_second_moment_change': variance_after-variance_before,
            'physical_content_change': total(grid.volumes*new_c)-total(grid.volumes*c)}
