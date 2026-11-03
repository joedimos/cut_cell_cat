"""Closed, static z-coordinate C-grid verification and cut-cell pressure stage."""
from dataclasses import dataclass

import numpy as np

from ..category.finite_sets import FiniteSet
from ..category.stochastic import FiniteKernel
from ..grid import vector
from .pressure import finite, positive


def _fraction(values, shape, name):
    values = finite(values, shape, name)
    if np.any((values < 0) | (values > 1)):
        raise ValueError(f'{name} must lie in [0, 1]')
    return values


def _incidence(size, tail, head):
    boundary = np.zeros((size, tail.size), dtype=np.int64)
    faces = np.arange(tail.size)
    boundary[tail, faces], boundary[head, faces] = -1, 1
    return boundary


@dataclass(frozen=True, init=False)
class BarotropicGrid:
    """Compact wet cells and open faces for a closed rectangular C-grid.

    Native MITgcm arrays are (level, y, x) or (y, x), with U on west faces
    and V on south faces. Edges point east/north. There is no periodic seam.
    The dense pressure stage is intended for verification grids.
    """

    shape: tuple
    wet: np.ndarray
    west: np.ndarray
    south: np.ndarray
    area: np.ndarray
    depth: np.ndarray
    cross_section: np.ndarray
    distance: np.ndarray
    boundary: np.ndarray
    tail: np.ndarray
    head: np.ndarray

    def __init__(self, *, rac, dxg, dyg, dxc, dyc, drf, hfac_c, hfac_w, hfac_s,
                 area_fraction=None, west_aperture=None, south_aperture=None):
        rac = np.asarray(rac, dtype=float)
        shape = rac.shape
        if rac.ndim != 2 or not rac.size:
            raise ValueError('RAC must be a nonempty (y, x) array')
        drf = np.asarray(drf, dtype=float)
        if drf.ndim != 1 or not drf.size:
            raise ValueError('DRF must be a nonempty level vector')
        dxg, dyg, dxc, dyc = (finite(value, shape, name) for value, name in
                              zip((dxg, dyg, dxc, dyc), ('DXG', 'DYG', 'DXC', 'DYC')))
        if any(not np.all(np.isfinite(a) & (a > 0)) for a in (rac, dxg, dyg, dxc, dyc, drf)):
            raise ValueError('native metric arrays must be finite and positive')
        levels = (drf.size, *shape)
        hfac_c, hfac_w, hfac_s = (_fraction(value, levels, name) for value, name in
                                 zip((hfac_c, hfac_w, hfac_s), ('hFacC', 'hFacW', 'hFacS')))
        if np.any(hfac_w[:, :, 0]) or np.any(hfac_s[:, 0, :]):
            raise ValueError('open or periodic exterior seams are unsupported')
        if (np.any((hfac_w[:, :, 1:] > 0) & ((hfac_c[:, :, :-1] == 0) | (hfac_c[:, :, 1:] == 0)))
                or np.any((hfac_s[:, 1:, :] > 0) & ((hfac_c[:, :-1, :] == 0) | (hfac_c[:, 1:, :] == 0)))):
            raise ValueError('a native open face level must connect two wet cells')
        geometry = (area_fraction, west_aperture, south_aperture)
        if any(a is None for a in geometry) and not all(a is None for a in geometry):
            raise ValueError('provide cell area fractions and both face apertures together')
        area_fraction, west_aperture, south_aperture = (
            np.ones(shape) if value is None else _fraction(value, shape, name)
            for value, name in zip(geometry, ('area_fraction', 'west_aperture', 'south_aperture')))
        thickness = drf[:, None, None]
        with np.errstate(over='raise', invalid='raise'):
            depth = np.sum(thickness * hfac_c, axis=0)
            area = rac * area_fraction
            cross_w = dyg * np.sum(thickness * hfac_w, axis=0) * west_aperture
            cross_s = dxg * np.sum(thickness * hfac_s, axis=0) * south_aperture
        wet = (depth > 0) & (area > 0)
        if not np.any(wet):
            raise ValueError('the grid has no wet cells')
        west, south = cross_w > 0, cross_s > 0
        j_w, i_w = np.nonzero(west)
        j_s, i_s = np.nonzero(south)
        cell_id = np.full(shape, -1, dtype=int)
        cell_id[wet] = np.arange(np.count_nonzero(wet))
        tail = np.concatenate((cell_id[j_w, i_w - 1], cell_id[j_s - 1, i_s]))
        head = np.concatenate((cell_id[west], cell_id[south]))
        if np.any(tail < 0) or np.any(head < 0):
            raise ValueError('an open face must connect two positive-area wet cells')
        arrays = {'wet': wet, 'west': west, 'south': south, 'area': area[wet],
                  'depth': depth[wet], 'cross_section': np.r_[cross_w[west], cross_s[south]],
                  'distance': np.r_[dxc[west], dyc[south]], 'tail': tail, 'head': head,
                  'boundary': _incidence(np.count_nonzero(wet), tail, head)}
        object.__setattr__(self, 'shape', shape)
        for name, values in arrays.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @classmethod
    def from_mds(cls, read, **horizontal_geometry):
        """Read static geometry through a caller-supplied MITgcmutils rdmds wrapper."""
        metrics = {name.lower(): read(name) for name in ('RAC', 'DXG', 'DYG', 'DXC', 'DYC')}
        drf = np.atleast_1d(np.squeeze(read('DRF')))
        fractions = {}
        for suffix in 'CWS':
            values = np.asarray(read(f'hFac{suffix}'))
            fractions[f'hfac_{suffix.lower()}'] = values[None] if drf.size == 1 and values.ndim == 2 else values
        return cls(**metrics, **fractions, drf=drf, **horizontal_geometry)

    def pack_velocity(self, u, v):
        """Extract one barotropic velocity per open face, in incidence order."""
        u = finite(u, self.shape, 'U')
        v = finite(v, self.shape, 'V')
        return np.r_[u[self.west], v[self.south]]

    def step(self, eta, velocity, dt, *, gravity=9.81):
        """Fully implicit linear wave step, returning the shared face transport."""
        eta = vector(eta, self.area.size, 'eta')
        velocity = vector(velocity, self.distance.size, 'velocity')
        dt, gravity = positive(dt, 'dt'), positive(gravity, 'gravity')
        b = self.boundary
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            transport = self.cross_section * velocity
            k = self.cross_section / self.distance
            lhs = np.diag(self.area) + dt**2 * gravity * ((b * k) @ b.T)
            rhs = self.area * eta + dt * (b @ transport)
            next_eta = np.linalg.solve(lhs, rhs)
            next_velocity = velocity - dt * gravity * (b.T @ next_eta) / self.distance
            next_transport = self.cross_section * next_velocity
            continuity = self.area * (next_eta - eta) - dt * (b @ next_transport)
        if not all(np.all(np.isfinite(a)) for a in (next_eta, next_velocity, next_transport, continuity)):
            raise FloatingPointError('pressure result is not representable')
        return next_eta, next_velocity, next_transport, continuity

    def tracer_step(self, volume, tracer, transport, dt):
        """Upwind concentration using the same integrated volume flux as continuity.

        This transport stage permits zero initial/final volume. Its donor bound
        is separate from pressure stability. Concentration is zero where dry.
        """
        size = self.area.size
        volume = vector(volume, size, 'volume')
        tracer = vector(tracer, size, 'tracer')
        transport = vector(transport, self.distance.size, 'transport')
        dt = positive(dt, 'dt')
        if np.any(volume < 0):
            raise ValueError('volume must be nonnegative')
        donor = np.where(transport >= 0, self.tail, self.head)
        receiver = np.where(transport >= 0, self.head, self.tail)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            transferred = dt * np.abs(transport)
            remaining = volume - np.bincount(donor, transferred, minlength=size)
            if np.any(remaining < 0):
                raise ValueError('outgoing volume violates the donor CFL bound')
            next_volume = remaining + np.bincount(receiver, transferred, minlength=size)
            content = remaining * tracer + np.bincount(receiver, transferred * tracer[donor], minlength=size)
            next_tracer = np.divide(content, next_volume, out=np.zeros(size), where=next_volume > 0)
        if not np.all(np.isfinite(next_volume) & np.isfinite(next_tracer)):
            raise FloatingPointError('tracer result is not representable')
        return next_volume, next_tracer

    def energy(self, eta, velocity, *, gravity=9.81):
        eta = vector(eta, self.area.size, 'eta')
        velocity = vector(velocity, self.distance.size, 'velocity')
        gravity = positive(gravity, 'gravity')
        return 0.5 * (gravity * np.dot(self.area, eta**2)
                      + np.dot(self.cross_section * self.distance, velocity**2))

    def quotient(self, blocks):
        """Return integer A, Q, Bc with A @ Bf == Bc @ Q for any cell partition."""
        blocks = np.asarray(blocks)
        if blocks.shape != self.area.shape or blocks.dtype.kind not in 'iu':
            raise ValueError('provide one integer block label per wet cell')
        labels, group = np.unique(blocks, return_inverse=True)
        coarse = len(labels)
        aggregate = np.eye(coarse, dtype=np.int64)[group].T
        start, end = group[self.tail], group[self.head]
        crossing = start != end
        pairs = np.column_stack((np.minimum(start, end), np.maximum(start, end)))
        edges, inverse = np.unique(pairs[crossing], axis=0, return_inverse=True)
        face_map = np.zeros((len(edges), len(start)), dtype=np.int64)
        face_map[inverse, np.flatnonzero(crossing)] = np.sign(end[crossing] - start[crossing])
        return aggregate, face_map, _incidence(coarse, edges[:, 0], edges[:, 1])


def half_plane_fractions(x_edges, y_edges, normal, offset):
    """Clip Cartesian cells and shared faces to normal dot (x,y) > offset.

    Returns cell, west and south fractions on the same (y, x) indexing. The
    polygon/segment clipping is exact in real arithmetic and uses float64 here.
    """
    x, y, normal = (np.asarray(a, dtype=float) for a in (x_edges, y_edges, normal))
    if any(a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a))
           or not np.all(a[1:] > a[:-1]) for a in (x, y)):
        raise ValueError('x_edges and y_edges must be finite and strictly increasing')
    if normal.shape != (2,) or not np.any(normal) or not np.all(np.isfinite(normal)):
        raise ValueError('normal must be a finite nonzero 2-vector')
    offset = finite(offset, (), 'offset').item()
    with np.errstate(over='raise', invalid='raise'):
        signed = normal[0] * x[None, :] + normal[1] * y[:, None] - offset

    def crossing_ratio(a, b):
        scale = np.maximum(np.abs(a), np.abs(b))
        a, b = np.abs(a) / scale, np.abs(b) / scale
        return a / (a + b)

    def segment_fraction(a, b):
        same_side = (a > 0) == (b > 0)
        fraction = np.zeros_like(a)
        fraction[same_side & (a > 0)] = 1.
        cross = ~same_side
        fraction[cross] = crossing_ratio(np.maximum(a[cross], b[cross]), np.minimum(a[cross], b[cross]))
        return fraction

    west = segment_fraction(signed[:-1, :-1], signed[1:, :-1])
    south = segment_fraction(signed[:-1, :-1], signed[:-1, 1:])
    corners = np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    values = np.stack((signed[:-1, :-1], signed[:-1, 1:], signed[1:, 1:], signed[1:, :-1]), axis=-1)
    area = np.all(values > 0, axis=-1).astype(float)
    cut = np.any(values > 0, axis=-1) & ~np.all(values > 0, axis=-1)
    for j, i in zip(*np.nonzero(cut)):
        polygon = []
        for k, a in enumerate(values[j, i]):
            l = (k + 1) % 4
            b = values[j, i, l]
            if a > 0:
                polygon.append(corners[k])
            if (a > 0) != (b > 0):
                ratio = crossing_ratio(a, b)
                polygon.append(corners[k] + ratio * (corners[l] - corners[k]))
        # Translate before computing the area to avoid cancellation for tiny corners.
        p = np.asarray(polygon) - polygon[0]
        area[j, i] = np.clip(0.5 * abs(np.dot(p[:-1, 0], p[1:, 1])
                                      - np.dot(p[:-1, 1], p[1:, 0])), 0., 1.)
    return area, west, south


def ghost_kernel(x, y, fluid, wet=None, *, min_wet_weight=0.05):
    """One bilinear image-point stencil conditioned on eligible wet donors.

    Donor order is SW, SE, NW, NE. ``wet`` defaults to ``fluid``. Returns
    None for a dry or geometrically unsupported stencil. The caller chooses
    the dry state or a geometric fallback and supplies neighbor eligibility.
    """
    x, y = (_fraction(a, (), name).item() for a, name in ((x, 'x'), (y, 'y')))
    minimum = _fraction(min_wet_weight, (), 'min_wet_weight').item()
    fluid = np.asarray(fluid)
    wet = fluid if wet is None else np.asarray(wet)
    if any(a.shape != (4,) or a.dtype.kind != 'b' for a in (fluid, wet)):
        raise ValueError('fluid and wet must be boolean masks in SW, SE, NW, NE order')
    weights = np.array(((1-x)*(1-y), x*(1-y), (1-x)*y, x*y))
    fluid_weight = np.sum(weights[fluid])
    weights *= fluid & wet
    wet_weight = np.sum(weights)
    if wet_weight <= minimum * fluid_weight:
        return None
    donors = FiniteSet(('SW', 'SE', 'NW', 'NE'))
    return FiniteKernel.state(donors, weights / wet_weight)


def reflected_ghost(kernel, normal, surface, bed, discharge):
    """Use one donor law for surface, bed and reflected two-component discharge."""
    if kernel.source.elements != ((),) or kernel.target.elements != ('SW', 'SE', 'NW', 'NE'):
        raise ValueError('expected a single bilinear donor law')
    normal = finite(normal, (2,), 'normal')
    scale = np.max(np.abs(normal))
    if scale == 0:
        raise ValueError('wall normal must be nonzero')
    normal = normal / scale
    normal /= np.linalg.norm(normal)
    surface = finite(surface, (4,), 'surface')
    bed = finite(bed, (4,), 'bed')
    discharge = finite(discharge, (4, 2), 'discharge')
    weights = kernel.matrix[0]
    with np.errstate(over='raise', invalid='raise'):
        support = weights > 0
        depth = surface[support] - bed[support]
        if np.any(depth <= 0):
            raise ValueError('supported donors must have positive depth')
        image = weights @ discharge
        reflected = image - 2. * (image @ normal) * normal
        return {'surface': weights @ surface, 'bed': weights @ bed,
                'depth': weights[support] @ depth, 'discharge': reflected}
