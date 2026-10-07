"""Dense educational views of the solver's incidence and coarse-grid diagrams.

The time-stepping path remains sparse/stencil based. These matrices are for
small audit examples; their floating-point identities are tolerance checks.
"""
from dataclasses import dataclass
import numpy as np
from .grid import vector


def incidence(size):
    """B F = F_left - F_right, including the two external boundary ports."""
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    b = np.zeros((size, size + 1), dtype=np.int64)
    i = np.arange(size)
    b[i, i], b[i, i + 1] = 1, -1
    return b


@dataclass(frozen=True, init=False)
class Coarsening:
    """A contiguous partition with exact integer chain map A Bf = Bc Q.

    A sums cell masses; Q selects block boundary fluxes. R averages
    concentrations by volume; P injects a block constant into wet cells.
    Dry placeholders are zero under P and absent from the weighted spaces.
    """
    boundaries: tuple
    aggregate: np.ndarray
    face_map: np.ndarray
    prolong: np.ndarray
    restrict: np.ndarray
    fine_volumes: np.ndarray
    coarse_volumes: np.ndarray

    def __init__(self, grid, boundaries):
        boundaries = tuple(boundaries)
        if (len(boundaries) < 2 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in boundaries)
                or boundaries[0] != 0 or boundaries[-1] != grid.size
                or any(b <= a for a, b in zip(boundaries, boundaries[1:]))):
            raise ValueError("boundaries must increase from 0 to grid.size")
        m = len(boundaries) - 1
        a = np.zeros((m, grid.size), dtype=np.int64)
        for j, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            a[j, left:right] = 1
        q = np.zeros((m + 1, grid.size + 1), dtype=np.int64)
        q[np.arange(m + 1), list(boundaries)] = 1
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            coarse = a @ grid.volumes
            r = (a * grid.volumes) / coarse[:, None]
        if not np.all(np.isfinite(coarse)) or np.any(coarse <= 0) or not np.all(np.isfinite(r)):
            raise ValueError("every coarse cell needs positive finite volume and representable weights")
        p = a.T * grid.active[:, None]
        object.__setattr__(self, 'boundaries', boundaries)
        for name, array in (('aggregate', a), ('face_map', q), ('prolong', p),
                            ('restrict', r), ('fine_volumes', grid.volumes.copy()), ('coarse_volumes', coarse)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def chain_map_holds(self):
        return bool(np.array_equal(self.aggregate @ incidence(len(self.fine_volumes)),
                                   incidence(len(self.coarse_volumes)) @ self.face_map))

    def restrict_state(self, state):
        return self.restrict @ vector(state, len(self.fine_volumes), 'state')

    def prolong_state(self, state):
        return self.prolong @ vector(state, len(self.coarse_volumes), 'coarse state')

    def dynamics_defect(self, fine_generator, coarse_generator):
        """R Lf - Lc R; zero is the extra condition for evolution naturality.

        Generators use full grid indexing. For grids with solids, embed the
        wet-only generator from closed_diffusion_diagram in a zero matrix first.
        """
        n, m = len(self.fine_volumes), len(self.coarse_volumes)
        fine = np.asarray(fine_generator, dtype=float)
        coarse = np.asarray(coarse_generator, dtype=float)
        if fine.shape != (n, n) or coarse.shape != (m, m) or not (np.all(np.isfinite(fine)) and np.all(np.isfinite(coarse))):
            raise ValueError("finite square generators with matching dimensions required")
        return self.restrict @ fine - coarse @ self.restrict


def closed_diffusion_diagram(operator):
    """Return B, K, M and L=-M^-1 B K B^T on wet cells and conducting faces.

    Only homogeneous prescribed-flux outer boundaries are admitted. Dirichlet
    data and nonzero boundary forcing require an affine/ported formulation.
    """
    if any(b.kind != 'flux' or b.value != 0 for b in (operator.left, operator.right)):
        raise ValueError("closed diagram requires zero prescribed boundary fluxes")
    grid = operator.grid
    wet = np.flatnonzero(grid.active)
    edges = np.flatnonzero(operator.conductance[1:-1] > 0) + 1
    b = incidence(grid.size)[np.ix_(wet, edges)]
    volumes = grid.volumes[wet].copy()
    k = operator.conductance[edges].copy()
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        stiffness = (b * k) @ b.T
        generator = -stiffness / volumes[:, None]
    if not np.all(np.isfinite(generator)):
        raise ValueError("generator is not representable; rescale inputs")
    # Component basis of H_0 for the conducting dual graph, without rank tolerances.
    labels = {int(i): j for j, i in enumerate(wet)}
    parent = list(range(len(wet)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for edge in edges:
        a, c = labels[int(edge - 1)], labels[int(edge)]
        parent[root(c)] = root(a)
    roots = [root(i) for i in range(len(wet))]
    unique = tuple(dict.fromkeys(roots))
    components = np.array([[int(r == u) for r in roots] for u in unique], dtype=np.int64)
    return {'wet_cells': wet, 'conducting_faces': edges, 'boundary': b,
            'conductance': k, 'volumes': volumes, 'generator': generator,
            'component_cocycles': components, 'betti_0': len(unique),
            'betti_1': len(edges) - len(wet) + len(unique)}
