"""Thin categories: adjunctions, idempotent monads, and pointwise Kan extensions."""
from dataclasses import dataclass
from itertools import combinations
from .core import FiniteCategory, Functor, freeze


def powerset(values):
    values = tuple(values)
    if len(set(values)) != len(values):
        raise ValueError("distinct labels required")
    return tuple(frozenset(xs) for n in range(len(values) + 1) for xs in combinations(values, n))


@dataclass(frozen=True)
class FinitePoset:
    category: FiniteCategory

    def __post_init__(self):
        c = self.category
        for x in c.objects:
            for y in c.objects:
                if len(c.hom(x, y)) > 1 or (x != y and c.hom(x, y) and c.hom(y, x)):
                    raise ValueError("category must be a partial order")

    @classmethod
    def from_relation(cls, objects, leq):
        return cls(FiniteCategory.poset(objects, leq))

    @classmethod
    def subsets(cls, values):
        return cls.from_relation(powerset(values), lambda a, b: a <= b)

    @property
    def objects(self):
        return self.category.objects

    def leq(self, x, y):
        return bool(self.category.hom(x, y))

    def join(self, values):
        values = tuple(values)
        if any(x not in self.objects for x in values):
            raise ValueError("unknown element")
        upper = [u for u in self.objects if all(self.leq(x, u) for x in values)]
        least = [u for u in upper if all(self.leq(u, v) for v in upper)]
        if len(least) != 1:
            raise ValueError("join does not exist")
        return least[0]

    def meet(self, values):
        values = tuple(values)
        if any(x not in self.objects for x in values):
            raise ValueError("unknown element")
        lower = [u for u in self.objects if all(self.leq(u, x) for x in values)]
        greatest = [u for u in lower if all(self.leq(v, u) for v in lower)]
        if len(greatest) != 1:
            raise ValueError("meet does not exist")
        return greatest[0]


@dataclass(frozen=True)
class MonotoneMap:
    source: FinitePoset
    target: FinitePoset
    images: dict

    def __post_init__(self):
        freeze(self, 'images', self.images)
        if set(self.images) != set(self.source.objects) or any(x not in self.target.objects for x in self.images.values()):
            raise ValueError("map must be total and land in the target")
        if any(self.source.leq(x, y) and not self.target.leq(self(x), self(y))
               for x in self.source.objects for y in self.source.objects):
            raise ValueError("map is not monotone")

    def __call__(self, x):
        return self.images[x]

    def then(self, other):
        if self.target != other.source:
            raise ValueError("map endpoints do not match")
        return MonotoneMap(self.source, other.target, {x: other(self(x)) for x in self.source.objects})

    def as_functor(self):
        return Functor(self.source.category, self.target.category, self.images,
            {f: self.target.category.hom(self(x), self(y))[0]
             for f, (x, y) in self.source.category.arrows.items()})


@dataclass(frozen=True)
class GaloisConnection:
    """L ⊣ R iff L(x) ≤ y iff x ≤ R(y), checked for every pair."""
    left: MonotoneMap
    right: MonotoneMap

    def __post_init__(self):
        l, r = self.left, self.right
        if l.source != r.target or l.target != r.source:
            raise ValueError("adjoint endpoints do not match")
        if any(l.target.leq(l(x), y) != l.source.leq(x, r(y))
               for x in l.source.objects for y in l.target.objects):
            raise ValueError("adjunction equivalence fails")

    @property
    def monad(self):
        """The closure endofunctor R L, with unit x ≤ RLx and μ: RLRLx = RLx."""
        return self.left.then(self.right)

    @property
    def comonad(self):
        """The interior endofunctor L R; counit LRy ≤ y."""
        return self.right.then(self.left)


def image_adjunction(mapping):
    """For f:A→B, direct image f_! is left adjoint to inverse image f*."""
    a, b = FinitePoset.subsets(mapping.source.elements), FinitePoset.subsets(mapping.target.elements)
    left = MonotoneMap(a, b, {s: frozenset(mapping(x) for x in s) for s in a.objects})
    right = MonotoneMap(b, a, {t: frozenset(x for x in mapping.source.elements if mapping(x) in t) for t in b.objects})
    return GaloisConnection(left, right)


def left_kan(along, diagram):
    """Lan_K F(d) = join {F(c) | K(c) ≤ d}; fails if a required join is absent."""
    if along.source != diagram.source:
        raise ValueError("K and F must have common source")
    return MonotoneMap(along.target, diagram.target,
        {d: diagram.target.join(diagram(c) for c in along.source.objects if along.target.leq(along(c), d))
         for d in along.target.objects})


def right_kan(along, diagram):
    """Ran_K F(d) = meet {F(c) | d ≤ K(c)} in finite posets."""
    if along.source != diagram.source:
        raise ValueError("K and F must have common source")
    return MonotoneMap(along.target, diagram.target,
        {d: diagram.target.meet(diagram(c) for c in along.source.objects if along.target.leq(d, along(c)))
         for d in along.target.objects})


def section_presheaf(cells, values):
    """U ↦ values**U, contravariant on patches; a sheaf on the discrete space.

    Restriction is genuine function restriction, not a diffusion operator.
    The power-set site and all sections grow exponentially; use tiny examples.
    """
    from .core import SetFunctor
    from .finite_sets import FiniteSet, FiniteMap, all_maps
    cells = FiniteSet(tuple(cells))
    patches = FinitePoset.subsets(cells.elements).category.opposite()
    domains = {u: FiniteSet(tuple(x for x in cells.elements if x in u)) for u in patches.objects}
    sets = {u: FiniteSet(tuple(f.images for f in all_maps(domains[u], values))) for u in patches.objects}
    maps = {}
    for arrow, (u, v) in patches.arrows.items():
        indices = [domains[u].elements.index(x) for x in domains[v].elements]
        maps[arrow] = FiniteMap(sets[u], sets[v], tuple(tuple(s[i] for i in indices) for s in sets[u].elements))
    return SetFunctor(patches, sets, maps)


def glue_sections(patch, sections):
    """Glue dictionaries on a covering family, rejecting holes and disagreements."""
    patch = frozenset(patch)
    result = {}
    for section in sections:
        for cell, value in section.items():
            if cell not in patch:
                raise ValueError("section extends outside the patch")
            if cell in result and result[cell] != value:
                raise ValueError("sections disagree on an overlap")
            result[cell] = value
    if set(result) != patch:
        raise ValueError("sections do not cover the patch")
    return result
