"""Exact finite-set maps and universal constructions (small examples only)."""
from dataclasses import dataclass
from itertools import product as cartesian


def elements(values):
    """Preserve supplied order; reject duplicate/unhashable element labels."""
    result = tuple(values)
    if len(set(result)) != len(result):
        raise ValueError("element labels must be distinct")
    return result


@dataclass(frozen=True)
class FiniteSet:
    elements: tuple

    def __post_init__(self):
        object.__setattr__(self, 'elements', elements(self.elements))

    def __contains__(self, value):
        return value in self.elements

    def __len__(self):
        return len(self.elements)


@dataclass(frozen=True)
class FiniteMap:
    """A total function, with images in the order of source.elements.

    Sets are represented by ordered enumerations; identical carriers in a
    different order are isomorphic but are not the same encoded object.
    """
    source: FiniteSet
    target: FiniteSet
    images: tuple

    def __post_init__(self):
        object.__setattr__(self, 'images', tuple(self.images))
        if len(self.images) != len(self.source) or any(y not in self.target for y in self.images):
            raise ValueError("a map must assign every source element a target element")

    def __call__(self, x):
        return self.images[self.source.elements.index(x)]

    @classmethod
    def identity(cls, obj):
        return cls(obj, obj, obj.elements)

    def then(self, other):
        """other ∘ self; composition order is explicit throughout the API."""
        if self.target != other.source:
            raise ValueError("composition requires matching intermediate objects")
        return FiniteMap(self.source, other.target, tuple(other(y) for y in self.images))


def all_maps(source, target):
    """Enumerate Hom(source, target); includes the unique empty-source map."""
    return tuple(FiniteMap(source, target, xs)
                 for xs in cartesian(target.elements, repeat=len(source)))


def product(left, right):
    obj = FiniteSet(tuple(cartesian(left.elements, right.elements)))
    return (obj, FiniteMap(obj, left, tuple(x for x, _ in obj.elements)),
            FiniteMap(obj, right, tuple(y for _, y in obj.elements)))


def pair(f, g):
    if f.source != g.source:
        raise ValueError("pairing requires a common source")
    obj, _, _ = product(f.target, g.target)
    return FiniteMap(f.source, obj, tuple(zip(f.images, g.images)))


def coproduct(left, right):
    obj = FiniteSet(tuple((0, x) for x in left.elements) + tuple((1, y) for y in right.elements))
    return (obj, FiniteMap(left, obj, tuple((0, x) for x in left.elements)),
            FiniteMap(right, obj, tuple((1, y) for y in right.elements)))


def copair(f, g):
    if f.target != g.target:
        raise ValueError("copairing requires a common target")
    obj, _, _ = coproduct(f.source, g.source)
    return FiniteMap(obj, f.target, f.images + g.images)


def pullback(f, g):
    if f.target != g.target:
        raise ValueError("pullback requires a common target")
    obj = FiniteSet(tuple((x, y) for x in f.source.elements for y in g.source.elements if f(x) == g(y)))
    return (obj, FiniteMap(obj, f.source, tuple(x for x, _ in obj.elements)),
            FiniteMap(obj, g.source, tuple(y for _, y in obj.elements)))


def pullback_lift(f, g, u, v):
    obj, _, _ = pullback(f, g)
    if u.source != v.source or u.then(f) != v.then(g):
        raise ValueError("pullback cone must commute")
    return FiniteMap(u.source, obj, tuple(zip(u.images, v.images)))


def _quotient(obj, relations):
    # Equivalence closure, including transitivity, not just pairwise merging.
    classes = [{x} for x in obj.elements]
    for x, y in relations:
        a = next(c for c in classes if x in c)
        b = next(c for c in classes if y in c)
        if a is not b:
            a.update(b)
            classes.remove(b)
    quotient = FiniteSet(tuple(frozenset(c) for c in classes))
    projection = FiniteMap(obj, quotient, tuple(next(c for c in quotient.elements if x in c) for x in obj.elements))
    return quotient, projection


def pushout(f, g):
    if f.source != g.source:
        raise ValueError("pushout requires a common source")
    union, left, right = coproduct(f.target, g.target)
    obj, q = _quotient(union, (((0, f(x)), (1, g(x))) for x in f.source.elements))
    return obj, left.then(q), right.then(q)


def pushout_descend(f, g, u, v):
    obj, _, _ = pushout(f, g)
    if u.target != v.target or f.then(u) != g.then(v):
        raise ValueError("pushout cocone must commute")
    mapping = copair(u, v)
    # Independence of representative follows from the cocone equation.
    return FiniteMap(obj, u.target, tuple(mapping(next(iter(c))) for c in obj.elements))


def equalizer(f, g):
    _parallel(f, g)
    obj = FiniteSet(tuple(x for x in f.source.elements if f(x) == g(x)))
    return obj, FiniteMap(obj, f.source, obj.elements)


def coequalizer(f, g):
    _parallel(f, g)
    return _quotient(f.target, zip(f.images, g.images))


def _parallel(f, g):
    if f.source != g.source or f.target != g.target:
        raise ValueError("maps must be parallel")


def exponential(base, exponent):
    """base**exponent, encoded by image tuples, and evaluation B**A × A → B."""
    obj = FiniteSet(tuple(f.images for f in all_maps(exponent, base)))
    domain, _, _ = product(obj, exponent)
    evaluation = FiniteMap(domain, base, tuple(xs[exponent.elements.index(a)] for xs, a in domain.elements))
    return obj, evaluation


def curry(f, parameter, argument):
    domain, _, _ = product(parameter, argument)
    if f.source != domain:
        raise ValueError("curry requires the encoded product as source")
    obj, _ = exponential(f.target, argument)
    return FiniteMap(parameter, obj, tuple(tuple(f((x, a)) for a in argument.elements) for x in parameter.elements))


def uncurry(f, argument, result):
    obj, evaluation = exponential(result, argument)
    if f.target != obj:
        raise ValueError("uncurry requires the specified exponential as target")
    domain, p, q = product(f.source, argument)
    return pair(p.then(f), q).then(evaluation)


@dataclass(frozen=True)
class Cospan:
    """A → N ← B; gluing uses a pushout in FinSet.

    Associativity is up to canonical apex isomorphism, not tuple equality.
    No decoration or structured-cospan functor is asserted here.
    """
    left: FiniteMap
    right: FiniteMap

    def __post_init__(self):
        if self.left.target != self.right.target:
            raise ValueError("cospan legs must share an apex")

    def then(self, other):
        if self.right.source != other.left.source:
            raise ValueError("cospan interfaces must match")
        _, i, j = pushout(self.right, other.left)
        return Cospan(self.left.then(i), other.right.then(j))

    def tensor(self, other):
        _, i, j = coproduct(self.left.target, other.left.target)
        return Cospan(copair(self.left.then(i), other.left.then(j)),
                      copair(self.right.then(i), other.right.then(j)))
