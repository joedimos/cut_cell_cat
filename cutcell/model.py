"""Simulation clock, SSP time integration, and auditable step budgets."""

from dataclasses import asdict, dataclass
import math
import numpy as np
from .grid import vector
from .operators import DiffusionOperator


@dataclass(frozen=True)
class StepBudget:
    time: float
    dt: float
    mass_before: float
    mass_after: float
    boundary_exchange: float
    source_exchange: float
    residual: float
    tolerance: float
    passed: bool
    minimum: float
    maximum: float
    energy: float
    max_local_residual: float
    cumulative_residual: float

    def to_dict(self):
        return asdict(self)


class DiffusionModel:
    def __init__(self, grid, state, diffusivity=0.1, *, left=None, right=None,
                 source=0.0, method="ssprk3", safety=0.9, atol=1e-12, rtol=1e-10,
                 history_limit=10000, require_nonnegative=False):
        self.grid = grid
        self.operator = DiffusionOperator(grid, diffusivity, left, right)
        self.state = vector(state, grid.size, "state")
        self.source = vector(source, grid.size, "source")
        if np.any(self.source[~grid.active] != 0):
            raise ValueError("source must be zero in solid cells")
        if method not in ("euler", "ssprk3"):
            raise ValueError("method must be euler or ssprk3")
        if not all(np.isfinite(x) and x >= 0 for x in (atol, rtol)):
            raise ValueError("tolerances must be finite and nonnegative")
        self.method, self.safety, self.atol, self.rtol = method, safety, atol, rtol
        self.operator.stable_dt(safety)
        self.time, self.iteration = 0.0, 0
        if isinstance(history_limit, bool) or not isinstance(history_limit, int) or history_limit < 1:
            raise ValueError("history_limit must be a positive integer")
        if not isinstance(require_nonnegative, bool):
            raise ValueError("require_nonnegative must be a boolean")
        self.history_limit = history_limit
        self.require_nonnegative = require_nonnegative
        self.history = []
        self.initial_mass = None
        self.total_exchange = 0.0
        self._exchange_correction = 0.0

    def mass(self, state=None):
        c = self.state if state is None else vector(state, self.grid.size, "state")
        return math.fsum(float(v) * float(x) for v, x in zip(self.grid.volumes, c))

    def step(self, requested_dt):
        """Advance by min(requested_dt, stability bound), returning actual dt."""
        if not np.isfinite(requested_dt) or requested_dt <= 0:
            raise ValueError("requested_dt must be positive and finite")
        dt = float(min(requested_dt, self.operator.stable_dt(self.safety)))
        if self.time + dt == self.time or not np.isfinite(self.time + dt):
            raise FloatingPointError("time step cannot advance the floating-point clock")
        c = vector(self.state, self.grid.size, "state")
        if self.require_nonnegative and np.any(c[self.grid.active] < 0):
            raise ValueError("negative initial concentration")
        before = self.mass(c)
        operator = self.operator
        f0 = operator.flux(c)
        c1 = c + dt * operator.tendency(c, self.source)
        if self.method == "euler":
            new, weighted_flux = c1, f0
        else:
            f1 = operator.flux(c1)
            c2 = 0.75 * c + 0.25 * (c1 + dt * operator.tendency(c1, self.source))
            f2 = operator.flux(c2)
            new = c / 3 + (2 / 3) * (c2 + dt * operator.tendency(c2, self.source))
            weighted_flux = (f0 + f1) / 6 + (2 / 3) * f2
        # Solid-cell values are placeholders, never evolved.
        new[~self.grid.active] = c[~self.grid.active]
        if not np.all(np.isfinite(new)):
            raise FloatingPointError("non-finite state; step was not committed")
        boundary = float(dt * (weighted_flux[0] - weighted_flux[-1]))
        source = dt * self.mass(self.source)
        after = self.mass(new)
        residual = math.fsum([after, -before, -boundary, -source])
        tolerance = self.atol + self.rtol * max(abs(before), abs(after), abs(boundary), abs(source))
        volumes = self.grid.volumes
        local_change = volumes * (new - c)
        local_exchange = -dt * np.diff(weighted_flux) + dt * volumes * self.source
        local_residual = local_change - local_exchange
        local_scale = np.maximum.reduce([np.abs(volumes*c), np.abs(volumes*new), np.abs(local_exchange)])
        local_tolerance = self.atol * volumes / math.fsum(volumes) + self.rtol * local_scale
        if not np.all(np.isfinite(local_residual)) or np.any(np.abs(local_residual) > local_tolerance):
            raise ArithmeticError("local cell balance failed; step was not committed")
        # Compensated cumulative exchange, independent of retained history.
        exchange = math.fsum([boundary, source]) - self._exchange_correction
        total = self.total_exchange + exchange
        correction = (total - self.total_exchange) - exchange
        initial = before if self.initial_mass is None else self.initial_mass
        cumulative = math.fsum([after, -initial, -total])
        cumulative_tolerance = self.atol + self.rtol * max(abs(initial), abs(after), abs(total))
        if not np.isfinite(cumulative) or abs(cumulative) > cumulative_tolerance:
            raise ArithmeticError("cumulative mass balance failed; step was not committed")
        wet = new[self.grid.active]
        if self.require_nonnegative and np.any(wet < 0):
            raise ArithmeticError("negative concentration; reduce forcing or timestep; step was not committed")
        budget = StepBudget(self.time + dt, dt, before, after, boundary, source,
                            residual, tolerance, abs(residual) <= tolerance,
                            float(wet.min()), float(wet.max()),
                            float(np.dot(self.grid.volumes[self.grid.active], wet * wet) / 2),
                            float(np.max(np.abs(local_residual))), cumulative)
        if not all(np.isfinite(x) for x in (before, after, boundary, source, residual, tolerance, budget.energy)):
            raise FloatingPointError("non-finite budget; step was not committed")
        if not budget.passed:
            raise ArithmeticError(f"mass budget failed: {residual}; step was not committed")
        self.state, self.time = new, budget.time
        self.initial_mass = initial
        self.total_exchange, self._exchange_correction = total, correction
        self.iteration += 1
        self.history.append(budget)
        if len(self.history) > self.history_limit:
            del self.history[0]
        return budget

    def run_until(self, stop_time, max_dt=0.001, max_steps=1_000_000):
        if not np.isfinite(stop_time) or stop_time < self.time:
            raise ValueError("stop_time must be finite and at or after current time")
        if not np.isfinite(max_dt) or max_dt <= 0:
            raise ValueError("max_dt must be finite and positive")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        count = 0
        while self.time < stop_time:
            if count >= max_steps:
                raise RuntimeError("maximum step count exceeded")
            self.step(min(max_dt, stop_time - self.time))
            count += 1
        return self.history
