"""Exact planar-bed cut-cell capacity and a frozen-flux implicit reference.

Research-only: this is not a MITgcm core solver, a nonlinear momentum
discretization, or a proof of full wet/dry/rStar stability.
"""
from dataclasses import dataclass

import numpy as np


def _finite_scalar(value, name):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _area_centroid(points):
    """Area and centroid of a CCW polygon by anchored triangle moments."""
    if len(points) < 3:
        return 0.0, np.zeros(2)
    points = np.asarray(points, dtype=float)
    origin = points[0]
    edges = points[1:] - origin
    twice = edges[:-1, 0] * edges[1:, 1] - edges[:-1, 1] * edges[1:, 0]
    weights = 0.5 * twice
    area = float(np.sum(weights))
    if area <= 0.0:
        return 0.0, np.zeros(2)
    centroids = (origin + points[1:-1] + points[2:]) / 3.0
    return area, np.sum(weights[:, None] * centroids, axis=0) / area


@dataclass(frozen=True)
class PlanarBedCutCell:
    """Convex fluid polygon in metres, with bed b(x,y)=ax+by+c."""

    vertices: object
    bed: object

    def __post_init__(self):
        vertices = np.array(self.vertices, dtype=float, copy=True)
        bed = np.array(self.bed, dtype=float, copy=True)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("vertices must be an (N,2) polygon with N>=3")
        if bed.shape != (3,) or not (np.all(np.isfinite(vertices)) and np.all(np.isfinite(bed))):
            raise ValueError("bed and polygon must be finite")
        vectors = np.roll(vertices, -1, axis=0) - vertices
        area, _ = _area_centroid(vertices)
        scale = np.max(np.abs(vectors))
        cross = (vectors[:, 0] * np.roll(vectors[:, 1], -1)
                 - vectors[:, 1] * np.roll(vectors[:, 0], -1))
        if area <= 0.0 or scale <= 0.0 or np.any(cross < -1e-12 * scale**2):
            raise ValueError("fluid polygon must be nondegenerate, convex and CCW")
        vertices.setflags(write=False)
        bed.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "bed", bed)

    @property
    def area(self):
        return _area_centroid(self.vertices)[0]

    @property
    def minimum_bed(self):
        return float(np.min(self.vertices @ self.bed[:2] + self.bed[2]))

    @property
    def maximum_bed(self):
        return float(np.max(self.vertices @ self.bed[:2] + self.bed[2]))

    def _wet_polygon(self, level):
        level = _finite_scalar(level, "level")
        vertices = self.vertices
        heights = vertices @ self.bed[:2] + self.bed[2] - level
        if np.all(heights <= 0.0):
            return vertices
        if np.all(heights >= 0.0):
            return np.empty((0, 2))
        result = []
        for index, point in enumerate(vertices):
            next_index = (index + 1) % len(vertices)
            next_point = vertices[next_index]
            a, b = heights[index], heights[next_index]
            if a <= 0.0:
                result.append(point)
            if (a < 0.0 and b > 0.0) or (a > 0.0 and b < 0.0):
                fraction = a / (a - b)
                result.append(point + fraction * (next_point - point))
        return np.asarray(result, dtype=float).reshape(-1, 2)

    def wetted_area(self, level):
        return _area_centroid(self._wet_polygon(level))[0]

    def volume_below(self, level):
        """Integral_polygon max(level - bed(x,y),0) dx dy in cubic metres."""
        level = _finite_scalar(level, "level")
        area, centroid = _area_centroid(self._wet_polygon(level))
        if area == 0.0:
            return 0.0
        return max(0.0, float(area * (level - self.bed[:2] @ centroid - self.bed[2])))

    def layer_volumes(self, surface, interfaces):
        """Exact volumes within fixed z-interfaces; order bottom to top."""
        surface = _finite_scalar(surface, "surface")
        interfaces = np.asarray(interfaces, dtype=float)
        if interfaces.ndim != 1 or len(interfaces) < 2 or not np.all(np.isfinite(interfaces)):
            raise ValueError("interfaces must be finite 1-D with at least two levels")
        if np.any(np.diff(interfaces) <= 0.0):
            raise ValueError("interfaces must be strictly increasing")
        capacity = np.array([self.volume_below(min(surface, z)) for z in interfaces])
        differences = np.diff(capacity)
        if np.any(differences < -1e-10 * max(1.0, np.max(capacity))):
            raise ArithmeticError("negative layer volume from geometric cancellation")
        return np.maximum(0.0, differences)

    def face_section(self, start, end, level):
        """Integral along an open face segment of positive water depth."""
        p = np.asarray(start, dtype=float)
        q = np.asarray(end, dtype=float)
        level = _finite_scalar(level, "level")
        if p.shape != (2,) or q.shape != (2,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            raise ValueError("face endpoints must be finite two-vectors")
        length = float(np.linalg.norm(q - p))
        if length <= 0.0:
            raise ValueError("face endpoints must be distinct")
        a = level - (p @ self.bed[:2] + self.bed[2])
        b = level - (q @ self.bed[:2] + self.bed[2])
        if a <= 0.0 and b <= 0.0:
            return 0.0
        if a >= 0.0 and b >= 0.0:
            return float(length * 0.5 * (a + b))
        if a > 0.0:
            return float(0.5 * length * a * a / (a - b))
        return float(0.5 * length * b * b / (b - a))


@dataclass(frozen=True)
class PressureResult:
    surface: np.ndarray
    volumes: np.ndarray
    face_flux: np.ndarray
    residual: np.ndarray
    iterations: int


class NonlinearCapacityPressure:
    """Implicit volume balance with exact bed capacity and frozen face K.

    Face orientation: tail=-1, head=+1. K is an input, not a constructed
    ocean flux law. The method establishes a geometric conservation
    reference, not a stable nonlinear momentum or dynamic-mask scheme.
    """

    def __init__(self, cells, tail, head, conductance, *, gravity=9.81):
        self.cells = tuple(cells)
        if not self.cells or any(not isinstance(c, PlanarBedCutCell) for c in self.cells):
            raise ValueError("cells must be nonempty PlanarBedCutCell sequence")
        self.tail = np.asarray(tail, dtype=int)
        self.head = np.asarray(head, dtype=int)
        self.k = np.asarray(conductance, dtype=float)
        count = len(self.cells)
        if (any(a.ndim != 1 for a in (self.tail, self.head, self.k))
                or len(self.tail) != len(self.head) or len(self.head) != len(self.k)
                or np.any(self.tail < 0) or np.any(self.head < 0)
                or np.any(self.tail >= count) or np.any(self.head >= count)
                or np.any(self.tail == self.head)
                or np.any(~np.isfinite(self.k)) or np.any(self.k < 0.0)):
            raise ValueError("invalid oriented faces or conductances")
        self.g = _finite_scalar(gravity, "gravity")
        if self.g <= 0.0:
            raise ValueError("gravity must be positive")
        self.incidence = np.zeros((count, len(self.k)), dtype=float)
        for f, (tail_id, head_id) in enumerate(zip(self.tail, self.head)):
            self.incidence[tail_id, f] = -1.0
            self.incidence[head_id, f] = 1.0
        self.laplacian = (self.incidence * self.k) @ self.incidence.T

    def _volume(self, surface):
        return np.array([c.volume_below(v) for c, v in zip(self.cells, surface)])

    def _area(self, surface):
        return np.array([c.wetted_area(v) for c, v in zip(self.cells, surface)])

    def step(self, old_surface, old_face_flux, dt, *, tol=1e-11, max_iter=50):
        old_surface = np.asarray(old_surface, dtype=float)
        old_face_flux = np.asarray(old_face_flux, dtype=float)
        dt = _finite_scalar(dt, "dt")
        if (dt <= 0.0 or old_surface.shape != (len(self.cells),)
                or old_face_flux.shape != (len(self.k),)
                or not np.all(np.isfinite(old_surface))
                or not np.all(np.isfinite(old_face_flux))):
            raise ValueError("invalid old state, flux or timestep")
        before = self._volume(old_surface)
        rhs = before + dt * (self.incidence @ old_face_flux)
        factor = self.g * dt * dt
        eta = old_surface.copy()
        scale = max(1.0, float(np.max(np.abs(rhs))))
        for iteration in range(max_iter + 1):
            volumes = self._volume(eta)
            residual = volumes + factor * (self.laplacian @ eta) - rhs
            magnitude = float(np.max(np.abs(residual)))
            if magnitude <= tol * scale:
                next_flux = old_face_flux - self.g * dt * self.k * (self.incidence.T @ eta)
                return PressureResult(eta, volumes, next_flux, residual, iteration)
            if iteration == max_iter:
                break
            jacobian = np.diag(self._area(eta)) + factor * self.laplacian
            try:
                delta = np.linalg.solve(jacobian, residual)
            except np.linalg.LinAlgError as error:
                raise ValueError("singular pressure capacity: dry disconnected component") from error
            accepted = False
            for power in range(35):
                trial = eta - (0.5**power) * delta
                trial_residual = self._volume(trial) + factor * (self.laplacian @ trial) - rhs
                if np.max(np.abs(trial_residual)) < magnitude:
                    eta = trial
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError("capacity Newton iteration failed line search")
        raise RuntimeError("capacity Newton iteration failed to converge")


def required_vertical_exchange(old_layers, new_layers, horizontal_inflow, *, atol=1e-10):
    """Return unique bottom-to-top internal volume transfers, if feasible.

    Rows are cells, columns are bottom-up layers. H is signed horizontal
    inflow (already integrated over dt) per layer. For vertical transfers
    G[k] upward across interface k, dV[k] = H[k]+G[k]-G[k+1].
    A zero bottom and top boundary is possible iff column sums match.
    """
    old = np.asarray(old_layers, dtype=float)
    new = np.asarray(new_layers, dtype=float)
    inflow = np.asarray(horizontal_inflow, dtype=float)
    if (old.ndim != 2 or not old.size or old.shape != new.shape or old.shape != inflow.shape
            or not all(np.all(np.isfinite(a)) for a in (old, new, inflow))
            or np.any(old < 0.0) or np.any(new < 0.0)):
        raise ValueError("invalid finite nonnegative layer volumes or inflows")
    change = new - old
    deficit = inflow - change
    mismatch = np.sum(deficit, axis=1)
    size = np.maximum(1.0, np.sum(old + new + np.abs(inflow), axis=1))
    if np.any(np.abs(mismatch) > atol * size):
        raise ValueError("column does not close: no purely internal vertical exchange exists")
    result = np.zeros((len(old), old.shape[1] + 1))
    result[:, 1:] = np.cumsum(deficit, axis=1)
    result[:, -1] = 0.0
    return result
