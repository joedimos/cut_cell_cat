"""WeatherNext-referenced ensemble mathematics, not its neural architecture.

CRPS conventions follow WN3 2609.03582v1 A.1.1/A.2.1. The conservative flux
ensemble is this repository's construction, not a claim about WN conservation.
"""
import numpy as np
from ..grid import vector
from ..model import DiffusionModel
from .pressure import finite, positive


def crps(ensemble, observation, *, fair=False):
    """CRPS per target; ensemble axis 0. O(m log m) time, O(m) memory per target.

    Empirical: mean|X-y| - sum_ij|Xi-Xj|/(2m^2).
    Fair: same first term, denominator 2m(m-1); needs iid members to be unbiased
    for the underlying distribution. Default evaluates the delivered ensemble.
    """
    x, y = np.asarray(ensemble, dtype=float), np.asarray(observation, dtype=float)
    if not isinstance(fair, bool):
        raise ValueError('fair must be boolean')
    if x.ndim < 1 or x.shape[0] < 1 or x.shape[1:] != y.shape:
        raise ValueError('ensemble must have shape (members, *observation.shape)')
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError('CRPS inputs must be finite; mask missing reports before scoring')
    m = x.shape[0]
    if fair and m < 2:
        raise ValueError('fair CRPS needs at least two members')
    with np.errstate(over='raise', invalid='raise'):
        ordered = np.sort(x, axis=0)
        # Sum over i<j = sum_k (2k-m+1) x_(k), k zero-based.
        # Centering reduces cancellation for large common offsets.
        ordered = ordered-ordered[0]
        weights = (2*np.arange(m)-m+1).reshape((m,)+(1,)*(x.ndim-1))
        pair_sum = np.sum(weights*ordered, axis=0)
        denominator = m*(m-1) if fair else m*m
        return np.mean(np.abs(x-y), axis=0)-pair_sum/denominator


def spatial_weights(weights, size):
    w = finite(weights, (size,), 'spatial weights')
    total = float(np.sum(w))
    if np.any(w < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError('nonnegative weights with positive finite sum required')
    return w/total


def field_score(ensemble, observation, weights, *, fair=False, pooled_weight=0.):
    """Weighted marginal CRPS plus optional CRPS of each member's spatial mean.

    WN3 uses coefficient 0.3 for selected gridded variables, not station targets.
    Caller explicitly chooses whether it is appropriate; defaults to disabled.
    """
    x, y = np.asarray(ensemble, dtype=float), np.asarray(observation, dtype=float)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError('field shapes must be (members, locations) and (locations,)')
    if not np.isfinite(pooled_weight) or pooled_weight < 0:
        raise ValueError('pooled_weight must be finite and nonnegative')
    marginal = crps(x, y, fair=fair)
    w = spatial_weights(weights, len(y))
    local = float(w @ marginal)
    pooled = float(crps(x @ w, y @ w, fair=fair))
    total = local+pooled_weight*pooled
    if not all(np.isfinite(v) for v in (local, pooled, total)):
        raise FloatingPointError('field score is not finite')
    return {'marginal': local, 'pooled_mean': pooled, 'total': total}


def multimodal_score(modalities, *, fair=False):
    """Sum lambda_i * mean(score_i), normalized within each modality (A.4).

    Each entry has ensemble, observation, weights, coefficient, optional valid
    boolean mask and pooled_weight. An empty modality is rejected rather than
    silently changing relative loss weights. Masked-out NaNs are permitted.
    """
    if not modalities:
        raise ValueError('at least one modality required')
    total, reports = 0., {}
    for name, item in modalities.items():
        y = np.asarray(item['observation'], dtype=float)
        x = np.asarray(item['ensemble'], dtype=float)
        weights = np.asarray(item['weights'], dtype=float)
        valid = np.asarray(item.get('valid', np.ones(y.shape, dtype=bool)))
        coefficient = float(item['coefficient'])
        if (y.ndim != 1 or x.ndim != 2 or x.shape[1:] != y.shape or weights.shape != y.shape
                or valid.shape != y.shape or valid.dtype != np.bool_ or not np.any(valid)
                or not np.isfinite(coefficient) or coefficient < 0):
            raise ValueError('invalid modality shape, mask or coefficient')
        report = field_score(x[:, valid], y[valid], weights[valid], fair=fair,
                             pooled_weight=item.get('pooled_weight', 0.))
        reports[name] = report
        total += coefficient*report['total']
    if not np.isfinite(total):
        raise FloatingPointError('multimodal loss is not finite')
    return {'total': float(total), 'modalities': reports}


class ConservativeFluxEnsemble:
    """A shared low-dimensional noise vector generates a whole flux field.

    This is a fixed linear stochastic residual around the existing diffusion
    solver, not a trained FGN. Zero boundary/closed-face modes ensure memberwise
    conservation on each connected component. Positivity is not implied.
    """
    def __init__(self, grid, face_modes, diffusivity=.1):
        modes = np.array(face_modes, dtype=float, copy=True)
        if (modes.ndim != 2 or modes.shape[0] != grid.size+1 or modes.shape[1] < 1
                or not np.all(np.isfinite(modes))):
            raise ValueError('face_modes must be finite with shape (n+1, latent_dimension)')
        if np.any(modes[[0, -1]] != 0) or np.any(modes[grid.apertures == 0] != 0):
            raise ValueError('modes must vanish on exterior and impermeable faces')
        # Validate and copy diffusivity using the actual solver.
        template = DiffusionModel(grid, np.zeros(grid.size), diffusivity)
        self.grid, self.diffusivity = grid, template.operator.diffusivity
        modes.setflags(write=False)
        self.face_modes = modes

    def sample(self, state, dt, noise, *, require_nonnegative=False):
        dt = positive(dt, 'dt')
        if not isinstance(require_nonnegative, bool):
            raise ValueError('require_nonnegative must be boolean')
        z = np.asarray(noise, dtype=float)
        if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] != self.face_modes.shape[1] or not np.all(np.isfinite(z)):
            raise ValueError('noise must have shape (members, latent_dimension) and be finite')
        c = vector(state, self.grid.size, 'state')
        model = DiffusionModel(self.grid, c, self.diffusivity, require_nonnegative=require_nonnegative)
        budget = model.step(dt)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            fluxes = z @ self.face_modes.T
            changes = -budget.dt*np.diff(fluxes, axis=1)
            increments = np.zeros_like(changes)
            np.divide(changes, self.grid.volumes, out=increments, where=self.grid.active)
            members = model.state+increments
        if require_nonnegative and np.any(members[:, self.grid.active] < 0):
            raise ArithmeticError('ensemble violates positivity; reduce perturbation, no clipping applied')
        if not np.all(np.isfinite(members)):
            raise FloatingPointError('nonfinite ensemble')
        return {'members': members, 'actual_dt': budget.dt,
                'mass_residuals': members @ self.grid.volumes-model.mass(c)}
