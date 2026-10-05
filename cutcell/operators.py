"""Shared face fluxes and volume-weighted finite-volume divergence."""

from dataclasses import dataclass
import numpy as np
from .grid import vector


@dataclass(frozen=True)
class BoundaryCondition:
    """Flux values are flux density in the positive coordinate direction.

    Therefore positive left flux adds mass and positive right flux removes it.
    Dirichlet values are concentrations at the physical outer faces.
    """
    kind: str = "flux"
    value: float = 0.0

    def __post_init__(self):
        if self.kind not in ("flux", "value") or not np.isfinite(self.value):
            raise ValueError("boundary must be 'flux' or 'value', with finite value")


class DiffusionOperator:
    def __init__(self, grid, diffusivity=0.1, left=None, right=None):
        self.grid = grid
        self.diffusivity = vector(diffusivity, grid.size + 1, "face diffusivity")
        if np.any(self.diffusivity < 0):
            raise ValueError("diffusivity must be nonnegative")
        self.diffusivity.setflags(write=False)
        self.left = BoundaryCondition() if left is None else left
        self.right = BoundaryCondition() if right is None else right
        if not isinstance(self.left, BoundaryCondition) or not isinstance(self.right, BoundaryCondition):
            raise TypeError("left and right must be BoundaryCondition instances")
        distance = np.r_[grid.centers[0] - grid.faces[0], np.diff(grid.centers),
                         grid.faces[-1] - grid.centers[-1]]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            self.conductance = grid.apertures * self.diffusivity / distance
        if not np.all(np.isfinite(self.conductance)):
            raise ValueError("face conductance is not representable; rescale geometry or diffusivity")
        self.conductance.setflags(write=False)

    def flux(self, state):
        """Return n+1 integrated face fluxes F = -aperture * kappa * grad(c)."""
        c = vector(state, self.grid.size, "state")
        flux = np.zeros(self.grid.size + 1)
        # Never evaluate differences across closed faces: inactive placeholder
        # values must not introduce 0 * inf or NaN into a disconnected component.
        connected = self.conductance[1:-1] > 0
        flux[1:-1][connected] = -self.conductance[1:-1][connected] * (c[1:][connected] - c[:-1][connected])
        flux[0] = (self.grid.apertures[0] * self.left.value if self.left.kind == "flux"
                   else self.conductance[0] * (self.left.value - c[0]))
        flux[-1] = (self.grid.apertures[-1] * self.right.value if self.right.kind == "flux"
                    else self.conductance[-1] * (c[-1] - self.right.value))
        if not np.all(np.isfinite(flux)):
            raise FloatingPointError("non-finite face flux")
        return flux

    def tendency(self, state, source=0.0):
        """dc/dt = (F_left - F_right)/V + source, zero in solid cells."""
        source = vector(source, self.grid.size, "source")
        flux = self.flux(state)
        result = np.zeros(self.grid.size)
        np.divide(-np.diff(flux), self.grid.volumes, out=result, where=self.grid.active)
        result[self.grid.active] += source[self.grid.active]
        return result

    def stable_dt(self, safety=0.9):
        """Monotone forward-Euler bound min V/(sum of draining conductances).

        SSPRK3 has SSP coefficient one and uses this same sufficient bound.
        Prescribed source/flux forcing may still drive values below zero.
        """
        if not np.isfinite(safety) or not 0 < safety <= 1:
            raise ValueError("safety must be in (0, 1]")
        conductance = self.conductance.copy()
        if self.left.kind == "flux":
            conductance[0] = 0
        if self.right.kind == "flux":
            conductance[-1] = 0
        drain = conductance[:-1] + conductance[1:]
        limited = self.grid.active & (drain > 0)
        return float(safety * np.min(self.grid.volumes[limited] / drain[limited])) if np.any(limited) else np.inf
