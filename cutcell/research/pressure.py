"""Korn 2608.25679v3: column split (16)-(21), and dispersion (87)-(89).

A separable, stationary 2-D pressure audit, not a global ocean model or a full
AC/DC time integrator. Pressure variables below mean p/rho0. See the source map.
"""
import numpy as np
from ..grid import CutCellGrid


def finite(value, shape, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must have shape {shape} and finite entries')
    return result


def positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


class ColumnPressureSplit:
    """Tensor-product FV Laplacian: x walls Neumann; z bottom Neumann, top zero.

    Fields have shape (nx, nz); z_faces increase from bottom to surface. The
    vertical solve is one batched tridiagonal elimination, O(nx*nz) memory/work.
    No global pressure solve is used by split(). No topographic cut or moving
    coordinate is implied by this bounded rectangular reference problem.
    """
    def __init__(self, x_faces, z_faces):
        self.x = CutCellGrid(x_faces)
        self.z = CutCellGrid(z_faces)
        self.shape = (self.x.size, self.z.size)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            self.kx = 1 / np.diff(self.x.centers)
            self.kz = np.r_[1 / np.diff(self.z.centers), 1 / (self.z.faces[-1] - self.z.centers[-1])]
            self.diagonal = self.kz + np.r_[0., self.kz[:-1]]
        if not all(np.all(np.isfinite(a)) for a in (self.kx, self.kz, self.diagonal)):
            raise ValueError('pressure conductance is not representable')
        for a in (self.kx, self.kz, self.diagonal):
            a.setflags(write=False)

    def horizontal(self, field):
        q = finite(field, self.shape, 'pressure')
        # Flux here is +grad(q), unlike the solver's diffusive -grad(c).
        flux = np.zeros((self.shape[0]+1, self.shape[1]))
        flux[1:-1] = self.kx[:, None] * np.diff(q, axis=0)
        return np.diff(flux, axis=0) / self.x.volumes[:, None]

    def vertical(self, field):
        q = finite(field, self.shape, 'pressure')
        flux = np.zeros((self.shape[0], self.shape[1]+1))
        flux[:, 1:-1] = self.kz[:-1] * np.diff(q, axis=1)
        flux[:, -1] = -self.kz[-1] * q[:, -1]
        return np.diff(flux, axis=1) / self.z.volumes

    def gradient(self, pressure):
        q = finite(pressure, self.shape, 'pressure')
        gx = np.zeros((self.shape[0]+1, self.shape[1]))
        gz = np.zeros((self.shape[0], self.shape[1]+1))
        gx[1:-1] = self.kx[:, None]*np.diff(q, axis=0)
        gz[:, 1:-1] = self.kz[:-1]*np.diff(q, axis=1)
        gz[:, -1] = -self.kz[-1]*q[:, -1]
        return gx, gz

    def divergence(self, horizontal_velocity, vertical_velocity):
        u = finite(horizontal_velocity, (self.shape[0]+1, self.shape[1]), 'u')
        w = finite(vertical_velocity, (self.shape[0], self.shape[1]+1), 'w')
        if np.any(u[[0, -1]] != 0) or np.any(w[:, 0] != 0):
            raise ValueError('normal velocity must vanish at side and bottom walls')
        return np.diff(u, axis=0)/self.x.volumes[:, None] + np.diff(w, axis=1)/self.z.volumes

    def acoustic_stage(self, u, w, pressure, interval, alpha_over_rho0, safety=.9, max_substeps=100000):
        """Explicit S3-b (107), p=psi/rho0, with a conservative spectral CFL bound.

        Both horizontal and vertical spacings enter the bound. This undamped
        wave evolution is NOT an iterative Poisson solver and need not decrease
        divergence monotonically. Uses fixed geometry and unit relative density.
        """
        interval = positive(interval, 'interval')
        alpha = positive(alpha_over_rho0, 'alpha_over_rho0')
        if not np.isfinite(safety) or not 0 < safety < 1:
            raise ValueError('safety must lie strictly between zero and one')
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError('max_substeps must be a positive integer')
        u = finite(u, (self.shape[0]+1, self.shape[1]), 'u')
        w = finite(w, (self.shape[0], self.shape[1]+1), 'w')
        p = finite(pressure, self.shape, 'pressure')
        self.divergence(u, w)  # validate walls before changing state
        # Gershgorin absolute row-sum bound for -L. The top Dirichlet
        # contribution appears only once; each internal conductance twice.
        hx = 2*(np.r_[0., self.kx]+np.r_[self.kx, 0.])/self.x.volumes
        vz = (self.diagonal + np.r_[0., self.kz[:-1]] + np.r_[self.kz[:-1], 0.])/self.z.volumes
        bound = float(np.max(hx)+np.max(vz))
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            max_dt = safety*2/np.sqrt(alpha*bound)
            required = interval/max_dt
        if not np.isfinite(required) or required > max_substeps:
            raise ValueError('required acoustic substeps exceed max_substeps')
        count = max(1, int(np.ceil(required)))
        dt = interval/count
        for _ in range(count):
            half = p-.5*dt*alpha*self.divergence(u, w)
            gx, gz = self.gradient(half)
            u -= dt*gx
            w -= dt*gz
            p = half-.5*dt*alpha*self.divergence(u, w)
        if not all(np.all(np.isfinite(a)) for a in (u, w, p)):
            raise FloatingPointError('nonfinite acoustic stage')
        return {'u': u, 'w': w, 'pressure_over_density': p,
                'substeps': count, 'substep_dt': dt, 'spectral_bound': bound}

    def column_solve(self, source):
        """Solve Lz q=S (16), using -Mz Lz as an SPD tridiagonal matrix."""
        s = finite(source, self.shape, 'source')
        diagonal = self.diagonal.copy()
        off = -self.kz[:-1]
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            rhs = -s * self.z.volumes
            for j in range(1, self.shape[1]):
                multiplier = off[j-1] / diagonal[j-1]
                diagonal[j] -= multiplier * off[j-1]
                rhs[:, j] -= multiplier * rhs[:, j-1]
            if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
                raise ArithmeticError('column factorization lost positive pivots')
            q = np.empty_like(rhs)
            q[:, -1] = rhs[:, -1] / diagonal[-1]
            for j in range(self.shape[1]-2, -1, -1):
                q[:, j] = (rhs[:, j] - off[j]*q[:, j+1]) / diagonal[j]
        return q

    def split(self, source):
        s = finite(source, self.shape, 'source')
        q = self.column_solve(s)
        residual = s - self.vertical(q) - self.horizontal(q)
        expected = -self.horizontal(q)
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError('pressure residual is not finite')
        return {'column_pressure_over_density': q, 'residual_source': residual,
                'column_residual': self.vertical(q)-s,
                'split_identity_residual': residual-expected}


def dispersion(kx, kz, buoyancy_frequency, coriolis, alpha_over_rho0):
    """Exact real frequency-squared branches, with stable quadratic evaluation.

    Implements Korn (87) and the pure-AC polynomial printed immediately after
    it. Rejects complex/negative branches instead of reporting them as stable
    waves. The leading 1/alpha errors are asymptotic, not exact identities.
    """
    alpha = positive(alpha_over_rho0, 'alpha_over_rho0')
    vals = np.asarray([kx, kz, buoyancy_frequency, coriolis], dtype=float)
    if not np.all(np.isfinite(vals)) or kz == 0 or buoyancy_frequency < 0:
        raise ValueError('finite wavenumbers, kz != 0 and N >= 0 required')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        x, z, n, f = vals**2
        k2 = x+z
        omega0 = (n*x+f*z)/k2
        a = alpha*k2+f*(1-x/z)
        b = alpha*(n*x+f*z)-(x/z)*n*f
        ap = alpha*k2+n+f
        bp = alpha*(n*x+f*z)+n*f
        def roots(a, b):
            if a <= 0 or b < 0:
                raise ValueError('parameters do not give two nonnegative wave branches')
            # Scaling avoids cancellation in the small root and overflow in a*a.
            scaled = (b/a)/a
            if not np.isfinite(scaled) or scaled > .25:
                raise ValueError('dispersion branches are complex')
            high = .5*a*(1+np.sqrt(1-4*scaled))
            return float(b/high), float(high)
        slow, fast = roots(a, b)
        pure_slow, pure_fast = roots(ap, bp)
        leading = x*x*(n-f)**2/(alpha*k2**3)
        pure_leading = -x*z*(n-f)**2/(alpha*k2**3)
    return {'incompressible_omega_squared': float(omega0),
            'acdc_slow_squared': slow, 'acdc_fast_squared': fast,
            'pure_ac_slow_squared': pure_slow, 'pure_ac_fast_squared': pure_fast,
            'acdc_leading_error_squared': float(leading),
            'pure_ac_leading_error_squared': float(pure_leading)}
