"""Finite categories and functors with exhaustive exact law validation."""
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from .finite_sets import FiniteSet, FiniteMap, elements


def freeze(obj, name, mapping):
    object.__setattr__(obj, name, MappingProxyType(dict(mapping)))


@dataclass(frozen=True)
class FiniteCategory:
    """composition[(f, g)] denotes g ∘ f; all and only composable pairs required."""
    objects: tuple
    arrows: dict  # label -> (source, target)
    identities: dict
    composition: dict

    def __post_init__(self):
        object.__setattr__(self, 'objects', elements(self.objects))
        for name in ('arrows', 'identities', 'composition'):
            freeze(self, name, getattr(self, name))
        if set(self.identities) != set(self.objects):
            raise ValueError("one identity per object required")
        for endpoints in self.arrows.values():
            if not isinstance(endpoints, tuple) or len(endpoints) != 2 or any(x not in self.objects for x in endpoints):
                raise ValueError("arrow endpoints must be objects")
        for x, identity in self.identities.items():
            if identity not in self.arrows or self.arrows[identity] != (x, x):
                raise ValueError("identity has wrong endpoints")
        pairs = {(f, g) for f, (a, b) in self.arrows.items()
                 for g, (c, d) in self.arrows.items() if b == c}
        if set(self.composition) != pairs:
            raise ValueError("composition must cover exactly the composable pairs")
        for (f, g), h in self.composition.items():
            if h not in self.arrows or self.arrows[h] != (self.arrows[f][0], self.arrows[g][1]):
                raise ValueError("composite has wrong endpoints")
        for f, (a, b) in self.arrows.items():
            if self.then(self.identities[a], f) != f or self.then(f, self.identities[b]) != f:
                raise ValueError("identity law fails")
        for f, g, h in product(self.arrows, repeat=3):
            if (f, g) in pairs and (g, h) in pairs:
                if self.then(self.then(f, g), h) != self.then(f, self.then(g, h)):
                    raise ValueError("associativity fails")

    def then(self, f, g):
        try:
            return self.composition[f, g]
        except KeyError as exc:
            raise ValueError("arrows are not composable") from exc

    def hom(self, a, b):
        if a not in self.objects or b not in self.objects:
            raise ValueError("unknown object")
        return tuple(f for f, endpoints in self.arrows.items() if endpoints == (a, b))

    def opposite(self):
        return FiniteCategory(self.objects, {f: (b, a) for f, (a, b) in self.arrows.items()},
                              self.identities, {(g, f): h for (f, g), h in self.composition.items()})

    @classmethod
    def poset(cls, objects, leq):
        objects = elements(objects)
        arrows = {(x, y): (x, y) for x in objects for y in objects if leq(x, y)}
        if any(x != y and (y, x) in arrows for x, y in arrows):
            raise ValueError("relation is not antisymmetric")
        if any((x, x) not in arrows for x in objects):
            raise ValueError("relation is not reflexive")
        if any((x, z) not in arrows for x, y in arrows for yy, z in arrows if y == yy):
            raise ValueError("relation is not transitive")
        return cls(objects, arrows, {x: (x, x) for x in objects},
                   {((x, y), (yy, z)): (x, z) for x, y in arrows for yy, z in arrows if y == yy})


@dataclass(frozen=True)
class Functor:
    source: FiniteCategory
    target: FiniteCategory
    objects: dict
    arrows: dict

    def __post_init__(self):
        freeze(self, 'objects', self.objects)
        freeze(self, 'arrows', self.arrows)
        if set(self.objects) != set(self.source.objects) or any(x not in self.target.objects for x in self.objects.values()):
            raise ValueError("functor must map every object")
        if set(self.arrows) != set(self.source.arrows):
            raise ValueError("functor must map every arrow")
        for f, (a, b) in self.source.arrows.items():
            if self.target.arrows.get(self.arrows[f]) != (self.objects[a], self.objects[b]):
                raise ValueError("functor arrow has wrong endpoints")
        for a, identity in self.source.identities.items():
            if self.arrows[identity] != self.target.identities[self.objects[a]]:
                raise ValueError("functor must preserve identities")
        for (f, g), h in self.source.composition.items():
            if self.arrows[h] != self.target.then(self.arrows[f], self.arrows[g]):
                raise ValueError("functor must preserve composition")

    @classmethod
    def identity(cls, category):
        return cls(category, category, {x: x for x in category.objects}, {f: f for f in category.arrows})

    def then(self, other):
        if self.target != other.source:
            raise ValueError("functor endpoints do not match")
        return Functor(self.source, other.target,
                       {x: other.objects[y] for x, y in self.objects.items()},
                       {f: other.arrows[g] for f, g in self.arrows.items()})


@dataclass(frozen=True)
class NaturalTransformation:
    source: Functor
    target: Functor
    components: dict

    def __post_init__(self):
        freeze(self, 'components', self.components)
        f, g = self.source, self.target
        if f.source != g.source or f.target != g.target or set(self.components) != set(f.source.objects):
            raise ValueError("natural transformations require parallel functors and all components")
        for x, a in self.components.items():
            if f.target.arrows.get(a) != (f.objects[x], g.objects[x]):
                raise ValueError("component has wrong endpoints")
        for a, (x, y) in f.source.arrows.items():
            if f.target.then(f.arrows[a], self.components[y]) != f.target.then(self.components[x], g.arrows[a]):
                raise ValueError("naturality square does not commute")

    @classmethod
    def identity(cls, functor):
        return cls(functor, functor, {x: functor.target.identities[y] for x, y in functor.objects.items()})

    def then(self, other):
        if self.target != other.source:
            raise ValueError("transformation endpoints do not match")
        return NaturalTransformation(self.source, other.target,
            {x: self.source.target.then(a, other.components[x]) for x, a in self.components.items()})


    def horizontal(self, other):
        """For α:F⇒G and β:H⇒K, return β*α:HF⇒KG.

        Component H(α_x) followed by β_(Gx); naturality equates the other route.
        """
        if self.source.target != other.source.source:
            raise ValueError("horizontal composition requires adjacent categories")
        return NaturalTransformation(self.source.then(other.source), self.target.then(other.target),
            {x: other.source.target.then(other.source.arrows[a], other.components[self.target.objects[x]])
             for x, a in self.components.items()})


@dataclass(frozen=True)
class SetFunctor:
    """A covariant finite-set-valued functor on a finite category."""
    source: FiniteCategory
    objects: dict
    arrows: dict

    def __post_init__(self):
        freeze(self, 'objects', self.objects)
        freeze(self, 'arrows', self.arrows)
        if set(self.objects) != set(self.source.objects) or any(not isinstance(x, FiniteSet) for x in self.objects.values()):
            raise ValueError("one finite set per object required")
        if set(self.arrows) != set(self.source.arrows):
            raise ValueError("one finite map per arrow required")
        for a, (x, y) in self.source.arrows.items():
            f = self.arrows[a]
            if not isinstance(f, FiniteMap) or f.source != self.objects[x] or f.target != self.objects[y]:
                raise ValueError("set functor arrow has wrong endpoints")
        for x, identity in self.source.identities.items():
            if self.arrows[identity] != FiniteMap.identity(self.objects[x]):
                raise ValueError("set functor must preserve identities")
        for (a, b), c in self.source.composition.items():
            if self.arrows[a].then(self.arrows[b]) != self.arrows[c]:
                raise ValueError("set functor must preserve composition")


@dataclass(frozen=True)
class SetTransformation:
    source: SetFunctor
    target: SetFunctor
    components: dict

    def __post_init__(self):
        freeze(self, 'components', self.components)
        f, g = self.source, self.target
        if f.source != g.source or set(self.components) != set(f.source.objects):
            raise ValueError("parallel diagrams and all components required")
        for x, a in self.components.items():
            if not isinstance(a, FiniteMap) or a.source != f.objects[x] or a.target != g.objects[x]:
                raise ValueError("component has wrong endpoints")
        for a, (x, y) in f.source.arrows.items():
            if f.arrows[a].then(self.components[y]) != self.components[x].then(g.arrows[a]):
                raise ValueError("naturality square does not commute")


def representable(category, at):
    """Covariant Hom(at, -). Contravariant representables use category.opposite()."""
    if at not in category.objects:
        raise ValueError("representing object must belong to the category")
    objects = {x: FiniteSet(category.hom(at, x)) for x in category.objects}
    arrows = {f: FiniteMap(objects[x], objects[y], tuple(category.then(h, f) for h in objects[x].elements))
              for f, (x, y) in category.arrows.items()}
    return SetFunctor(category, objects, arrows)


def yoneda_lift(functor, at, element):
    """x ∈ F(at) ↦ (h ↦ F(h)(x)) in Nat(Hom(at,-), F)."""
    hom = representable(functor.source, at)
    if element not in functor.objects[at]:
        raise ValueError("element must lie in F(at)")
    return SetTransformation(hom, functor,
        {x: FiniteMap(hom.objects[x], functor.objects[x],
                     tuple(functor.arrows[h](element) for h in hom.objects[x].elements))
         for x in functor.source.objects})


def yoneda_evaluate(transformation, at):
    if transformation.source != representable(transformation.target.source, at):
        raise ValueError("source must be Hom(at,-)")
    return transformation.components[at](transformation.target.source.identities[at])
