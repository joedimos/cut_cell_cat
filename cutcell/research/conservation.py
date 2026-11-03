"""Exact, small finite-volume certificates on oriented dual meshes.

Topology uses unbounded integers. Ledger entries use rational numbers. Convert
floating-point model output explicitly, and assess its residual with tolerances
appropriate to the model and the units.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np


def _exact_array(values, shape, scalar_type, name):
    array = np.array(values, dtype=object)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if any(isinstance(x, bool) or not isinstance(x, scalar_type) for x in array.flat):
        raise ValueError(f"{name} must have exact {scalar_type.__name__} entries")
    cast = int if scalar_type is Integral else Fraction
    return np.array([cast(x) for x in array.flat], dtype=object).reshape(shape)


@dataclass(frozen=True)
class OrientedMesh:
    """Dual cells and oriented faces (tail, head), with None for exterior.

    Parallel faces are distinct. Internal columns have a -1 and a +1;
    exterior columns have one nonzero entry. Isolated cells are permitted.
    """
    cells: int
    faces: tuple
    _incidence: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if isinstance(self.cells, bool) or not isinstance(self.cells, Integral) or self.cells < 0:
            raise ValueError("cell count must be a nonnegative integer")
        faces = tuple(tuple(face) for face in self.faces)
        b = np.zeros((self.cells, len(faces)), dtype=object)
        for j, (tail, head) in enumerate(faces):
            if tail == head:
                raise ValueError("a face needs distinct endpoints")
            for cell, sign in ((tail, -1), (head, 1)):
                if cell is not None:
                    if isinstance(cell, bool) or not isinstance(cell, Integral) or not 0 <= cell < self.cells:
                        raise ValueError("invalid face endpoint")
                    b[cell, j] = sign
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "_incidence", b)

    @property
    def incidence(self):
        return self._incidence.copy()

    def quotient(self, partition):
        """Sum cells and consistently oriented faces between the same blocks.

        Labels must be contiguous from zero. Internal faces of a block map
        to zero. For exterior faces, the aggregate orientation is inward.
        """
        blocks = tuple(partition)
        if (len(blocks) != self.cells
                or any(isinstance(x, bool) or not isinstance(x, Integral) for x in blocks)
                or set(blocks) != set(range(len(set(blocks))))):
            raise ValueError("partition needs a contiguous nonnegative label for each cell")
        groups = len(set(blocks))
        a = np.zeros((groups, self.cells), dtype=object)
        for i, group in enumerate(blocks):
            a[group, i] = 1
        faces = {}
        mappings = []
        for tail, head in self.faces:
            left = None if tail is None else blocks[tail]
            right = None if head is None else blocks[head]
            if left == right:
                mappings.append(None)
                continue
            forward = left is None or (right is not None and left < right)
            face = (left, right) if forward else (right, left)
            mappings.append((faces.setdefault(face, len(faces)), 1 if forward else -1))
        q = np.zeros((len(faces), len(self.faces)), dtype=object)
        for j, mapping in enumerate(mappings):
            if mapping is not None:
                i, sign = mapping
                q[i, j] = sign
        return ChainMap(self, OrientedMesh(groups, tuple(faces)), a, q)


@dataclass(frozen=True, eq=False)
class ChainMap:
    """Total-preserving integral chain map, with A B_source = B_target Q."""
    source: OrientedMesh
    target: OrientedMesh
    _cells: np.ndarray = field(repr=False)
    _faces: np.ndarray = field(repr=False)

    def __post_init__(self):
        a = _exact_array(self._cells, (self.target.cells, self.source.cells), Integral, "A")
        q = _exact_array(self._faces, (len(self.target.faces), len(self.source.faces)), Integral, "Q")
        if np.any(a.sum(axis=0) != 1):
            raise ValueError("A must preserve total extensive mass")
        if not np.array_equal(a @ self.source.incidence, self.target.incidence @ q):
            raise ValueError("A and Q do not commute with incidence")
        object.__setattr__(self, "_cells", a)
        object.__setattr__(self, "_faces", q)

    @classmethod
    def identity(cls, mesh):
        return cls(mesh, mesh, np.eye(mesh.cells, dtype=int), np.eye(len(mesh.faces), dtype=int))

    def __eq__(self, other):
        return (isinstance(other, ChainMap) and self.source == other.source and self.target == other.target
                and np.array_equal(self._cells, other._cells) and np.array_equal(self._faces, other._faces))

    @property
    def cells(self):
        return self._cells.copy()

    @property
    def faces(self):
        return self._faces.copy()

    def then(self, other):
        if self.target != other.source:
            raise ValueError("chain maps have incompatible endpoints")
        return ChainMap(self.source, other.target, other._cells @ self._cells, other._faces @ self._faces)


@dataclass(frozen=True, eq=False)
class Slab:
    """One time interval on the ambient mesh, with a fixed allowed-face set.

    All face values are time-integrated and signed in the mesh orientation.
    Capacity is measured at both time boundaries. A topology event can split
    a step into slabs; opening is explicit and does not infer a flux.
    """
    mesh: OrientedMesh
    before: tuple
    after: tuple
    open_faces: frozenset

    def __post_init__(self):
        object.__setattr__(self, "before", tuple(self.cell_vector(self.before, "initial capacity")))
        object.__setattr__(self, "after", tuple(self.cell_vector(self.after, "final capacity")))
        opened = tuple(self.open_faces)
        if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(self.mesh.faces)
               for i in opened):
            raise ValueError("invalid open face")
        if any(x < 0 for x in (*self.before, *self.after)):
            raise ValueError("capacities must be nonnegative")
        object.__setattr__(self, "open_faces", frozenset(opened))

    def cell_vector(self, values, name):
        return _exact_array(values, (self.mesh.cells,), Rational, name)

    def face_vector(self, values):
        flux = _exact_array(values, (len(self.mesh.faces),), Rational, "face flux")
        if any(x != 0 for i, x in enumerate(flux) if i not in self.open_faces):
            raise ValueError("nonzero flux on a closed face")
        return flux

    def residual(self, before, after, flux, sources):
        """Return the local balance residual for any extensive quantity."""
        return (self.cell_vector(after, "final mass") - self.cell_vector(before, "initial mass")
                - self.mesh.incidence @ self.face_vector(flux) - self.cell_vector(sources, "sources"))

    def geometry_residual(self, sweep, sources):
        """Check the discrete geometric conservation law for capacity."""
        return self.residual(self.before, self.after, sweep, sources)


def overlap_remap(overlap, before, after, mass):
    """Remap an extensive field with given intersection volumes O[target, source].

    Exact row and column margins certify complete coverage and disjoint
    accounting in volume. Actual intersection geometry is the caller's
    responsibility. A zero-volume source must carry zero mass.
    """
    before = _exact_array(before, (len(before),), Rational, "source capacities")
    after = _exact_array(after, (len(after),), Rational, "target capacities")
    mass = _exact_array(mass, before.shape, Rational, "source mass")
    overlap = _exact_array(overlap, (len(after), len(before)), Rational, "overlap")
    if np.any(overlap < 0) or not np.array_equal(overlap.sum(axis=0), before) or not np.array_equal(
            overlap.sum(axis=1), after):
        raise ValueError("nonnegative overlaps must match both capacity margins")
    if any(v == 0 and m != 0 for m, v in zip(mass, before)):
        raise ValueError("dry source cells cannot carry mass")
    concentration = np.array([m / v if v else 0 for m, v in zip(mass, before)], dtype=object)
    return overlap @ concentration


def balance_defect(resolution, fine, coarse, fine_flux, coarse_flux, fine_source, coarse_source):
    """Return (flux mismatch, source mismatch, induced coarse balance defect).

    Assumes synchronized slabs. Fine endpoint storage may be checked against
    coarse storage with A separately, and the recorded balance with residual.
    """
    if fine.mesh != resolution.source or coarse.mesh != resolution.target:
        raise ValueError("slabs must match the resolution map")
    delta_flux = resolution.faces @ fine.face_vector(fine_flux) - coarse.face_vector(coarse_flux)
    delta_source = (resolution.cells @ fine.cell_vector(fine_source, "fine sources")
                    - coarse.cell_vector(coarse_source, "coarse sources"))
    return delta_flux, delta_source, coarse.mesh.incidence @ delta_flux + delta_source


def internal_correction(slab, defect):
    """Find B q = defect using only open internal faces, or reject the defect.

    A spanning forest gives one solution with all non-tree fluxes zero.
    This algebraic lift imposes no CFL, capacity, or positivity constraint.
    """
    pending = slab.cell_vector(defect, "cell defect")
    mesh = slab.mesh
    adjacent = [[] for _ in range(mesh.cells)]
    for j in sorted(slab.open_faces):
        tail, head = mesh.faces[j]
        if tail is not None and head is not None:
            adjacent[tail].append((head, j, 1))
            adjacent[head].append((tail, j, -1))
    flux = np.zeros(len(mesh.faces), dtype=object)
    seen = set()
    for root in range(mesh.cells):
        if root in seen:
            continue
        seen.add(root)
        tree = [(root, None, None, None)]
        for cell, _, _, _ in tree:
            for child, face, sign in adjacent[cell]:
                if child not in seen:
                    seen.add(child)
                    tree.append((child, cell, face, sign))
        for child, parent, face, sign in reversed(tree[1:]):
            flux[face] = sign * pending[child]
            pending[parent] += pending[child]
        if pending[root] != 0:
            raise ValueError(f"unbalanced defect on open component rooted at cell {root}")
    return flux


def glue_fluxes(slab, face_atoms, outward):
    """Glue outward local flux records on a specified common space-time partition.

    face_atoms maps every open face to its atom IDs. outward maps triples
    (cell, face, atom) to exact integrated outward fluxes. Exterior faces
    have one record per atom. Uncovered and duplicate atom IDs are rejected.
    Returns face totals and the interface residuals (one per internal atom);
    no global flux is returned when any interface residual is nonzero.
    """
    if set(face_atoms) != slab.open_faces:
        raise ValueError("atom partitions must cover precisely the open faces")
    atoms = {face: tuple(ids) for face, ids in face_atoms.items()}
    if any(not ids or len(set(ids)) != len(ids) for ids in atoms.values()):
        raise ValueError("each face partition needs nonempty distinct atom IDs")
    expected = {(cell, face, atom) for face, ids in atoms.items() for atom in ids
                for cell in slab.mesh.faces[face] if cell is not None}
    if set(outward) != expected:
        raise ValueError("local trace keys must cover every incident cell and atom exactly")
    if any(isinstance(x, bool) or not isinstance(x, Rational) for x in outward.values()):
        raise ValueError("local traces must be exact rational integrals")

    flux = np.zeros(len(slab.mesh.faces), dtype=object)
    mismatch = {}
    for face in sorted(atoms):
        tail, head = slab.mesh.faces[face]
        for atom in atoms[face]:
            a = Fraction(outward[tail, face, atom]) if tail is not None else None
            b = Fraction(outward[head, face, atom]) if head is not None else None
            if a is not None and b is not None and a + b != 0:
                mismatch[face, atom] = a + b
            flux[face] += a if a is not None else -b
    return (None if mismatch else flux), mismatch
