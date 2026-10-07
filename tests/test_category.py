"""Exact categorical laws, universal-property uniqueness, and counterexamples."""
import itertools
import unittest
from cutcell.category import *


def cyclic(n):
    return FiniteCategory(('*',), {i: ('*', '*') for i in range(n)}, {'*': 0},
                          {(i, j): (i + j) % n for i in range(n) for j in range(n)})


def permutations_category():
    arrows = tuple(itertools.permutations(range(3)))
    return FiniteCategory(('*',), {p: ('*', '*') for p in arrows}, {'*': (0, 1, 2)},
                          {(p, q): tuple(q[p[i]] for i in range(3)) for p in arrows for q in arrows})


class CategoryLaws(unittest.TestCase):
    def test_opposite_and_composition_on_noncommutative_group(self):
        c = permutations_category()
        self.assertEqual(c.opposite().opposite(), c)
        f, g = (1, 0, 2), (0, 2, 1)
        self.assertNotEqual(c.then(f, g), c.then(g, f))
        self.assertEqual(c.opposite().then(g, f), c.then(f, g))
        identity = Functor.identity(c)
        self.assertEqual(identity.then(identity), identity)
        eta = NaturalTransformation.identity(identity)
        self.assertEqual(eta.then(eta), eta)

    def test_malformed_categories_rejected(self):
        c = cyclic(3)
        for field, replacement in [('identities', {'*': 1}), ('composition', {}),
                                    ('arrows', {0: ('*', 'missing')})]:
            args = dict(objects=c.objects, arrows=c.arrows, identities=c.identities, composition=c.composition)
            args[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                FiniteCategory(**args)
        bad = dict(c.composition)
        bad[1, 2] = 1  # all endpoints and units still valid, associativity fails
        with self.assertRaisesRegex(ValueError, 'associativity'):
            FiniteCategory(c.objects, c.arrows, c.identities, bad)
        with self.assertRaises(ValueError):
            FiniteCategory.poset((0, 1, 2), lambda x, y: x == y or y == x + 1)
        with self.assertRaises(ValueError):
            FiniteCategory.poset((0, 1), lambda x, y: True)

    def test_functor_rejects_nonhomomorphism_and_wrong_identity(self):
        c, d = cyclic(2), cyclic(3)
        with self.assertRaisesRegex(ValueError, 'composition'):
            Functor(c, d, {'*': '*'}, {0: 0, 1: 1})
        with self.assertRaisesRegex(ValueError, 'identities'):
            Functor(c, c, {'*': '*'}, {0: 1, 1: 0})
        f = Functor(c, d, {'*': '*'}, {0: 0, 1: 0})
        self.assertEqual(Functor.identity(c).then(f).then(Functor.identity(d)), f)

    def test_non_natural_component_rejected(self):
        c = permutations_category()
        identity = Functor.identity(c)
        with self.assertRaisesRegex(ValueError, 'naturality'):
            NaturalTransformation(identity, identity, {'*': (1, 0, 2)})
        chain = FiniteCategory.poset((0, 1, 2), lambda x, y: x <= y)
        f = Functor.identity(chain)
        g = Functor(chain, chain, {0: 1, 1: 2, 2: 2},
                    {a: (min(a[0]+1, 2), min(a[1]+1, 2)) for a in chain.arrows})
        eta = NaturalTransformation(f, g, {x: (x, g.objects[x]) for x in chain.objects})
        self.assertEqual(eta.then(NaturalTransformation.identity(g)), eta)

    def test_two_category_interchange_law(self):
        c = FiniteCategory.poset((0, 1, 2), lambda x, y: x <= y)
        def shift(n):
            return Functor(c, c, {x: min(x+n, 2) for x in c.objects},
                           {a: (min(a[0]+n, 2), min(a[1]+n, 2)) for a in c.arrows})
        f, g, h = shift(0), shift(1), shift(2)
        alpha = NaturalTransformation(f, g, {x: (f.objects[x], g.objects[x]) for x in c.objects})
        beta = NaturalTransformation(g, h, {x: (g.objects[x], h.objects[x]) for x in c.objects})
        self.assertEqual(alpha.then(beta).horizontal(alpha.then(beta)),
                         alpha.horizontal(alpha).then(beta.horizontal(beta)))
        self.assertEqual(alpha.horizontal(NaturalTransformation.identity(f)), alpha)
        self.assertEqual(NaturalTransformation.identity(f).horizontal(alpha), alpha)

    def test_input_maps_are_copied_and_read_only(self):
        c = cyclic(2)
        images = {0: 0, 1: 1}
        f = Functor(c, c, {'*': '*'}, images)
        images[1] = 0
        self.assertEqual(f.arrows[1], 1)
        with self.assertRaises(TypeError):
            f.arrows[1] = 0

    def test_yoneda_bijection_enumerates_all_natural_transformations(self):
        # A nontrivial group action catches reversed composition and variance.
        c = permutations_category()
        values = FiniteSet((0, 1, 2))
        action = SetFunctor(c, {'*': values}, {p: FiniteMap(values, values, p) for p in c.arrows})
        hom = representable(c, '*')
        natural = []
        for candidate in all_maps(hom.objects['*'], values):
            try:
                natural.append(SetTransformation(hom, action, {'*': candidate}))
            except ValueError:
                pass
        self.assertEqual(len(natural), len(values))
        for eta in natural:
            x = yoneda_evaluate(eta, '*')
            self.assertEqual(yoneda_lift(action, '*', x), eta)
        for x in values.elements:
            self.assertEqual(yoneda_evaluate(yoneda_lift(action, '*', x), '*'), x)
        with self.assertRaises(ValueError):
            yoneda_lift(action, '*', 4)

    def test_empty_category_has_no_representing_object(self):
        empty = FiniteCategory((), {}, {}, {})
        self.assertEqual(empty.opposite(), empty)
        with self.assertRaises(ValueError):
            representable(empty, 'missing')

    def test_set_functor_rejects_invalid_laws(self):
        c, a = cyclic(2), FiniteSet((0, 1))
        with self.assertRaises(ValueError):
            SetFunctor(c, {'*': a}, {0: FiniteMap.identity(a), 1: FiniteMap(a, a, (0, 0))})
        with self.assertRaises(ValueError):
            SetFunctor(c, {'*': a}, {0: FiniteMap(a, a, (1, 0)), 1: FiniteMap.identity(a)})


class UniversalProperties(unittest.TestCase):
    def setUp(self):
        self.empty, self.one, self.two = FiniteSet(()), FiniteSet(('x',)), FiniteSet((0, 1))

    def test_total_maps_and_composition(self):
        with self.assertRaises(ValueError):
            FiniteSet((0, 0))
        with self.assertRaises(ValueError):
            FiniteMap(self.two, self.one, ('x',))
        with self.assertRaises(ValueError):
            FiniteMap(self.two, self.one, ('x', 'y'))
        with self.assertRaises(ValueError):
            FiniteMap.identity(self.two).then(FiniteMap.identity(self.one))
        for f in all_maps(self.two, self.two):
            for g in all_maps(self.two, self.two):
                for h in all_maps(self.two, self.two):
                    self.assertEqual(f.then(g).then(h), f.then(g.then(h)))

    def test_initial_terminal_and_empty_exponentials(self):
        for obj in (self.empty, self.one, self.two):
            self.assertEqual(len(all_maps(self.empty, obj)), 1)
            self.assertEqual(len(all_maps(obj, self.one)), 1)
            self.assertEqual(len(exponential(obj, self.empty)[0]), 1)
        self.assertEqual(len(exponential(self.empty, self.two)[0]), 0)
        self.assertEqual(len(all_maps(self.one, self.empty)), 0)

    def test_product_and_coproduct_unique_mediators(self):
        for a in (self.empty, self.one, self.two):
            for b in (self.empty, self.one, self.two):
                p, p1, p2 = product(a, b)
                s, i1, i2 = coproduct(a, b)
                for f in all_maps(self.two, a):
                    for g in all_maps(self.two, b):
                        candidates = [h for h in all_maps(self.two, p) if h.then(p1) == f and h.then(p2) == g]
                        self.assertEqual(candidates, [pair(f, g)])
                for f in all_maps(a, self.two):
                    for g in all_maps(b, self.two):
                        candidates = [h for h in all_maps(s, self.two) if i1.then(h) == f and i2.then(h) == g]
                        self.assertEqual(candidates, [copair(f, g)])

    def test_pullback_universal_property_for_all_two_element_maps(self):
        for f in all_maps(self.two, self.two):
            for g in all_maps(self.two, self.two):
                obj, p, q = pullback(f, g)
                self.assertEqual(p.then(f), q.then(g))
                for u in all_maps(self.two, self.two):
                    for v in all_maps(self.two, self.two):
                        if u.then(f) == v.then(g):
                            candidates = [h for h in all_maps(self.two, obj) if h.then(p) == u and h.then(q) == v]
                            self.assertEqual(candidates, [pullback_lift(f, g, u, v)])
                        else:
                            with self.assertRaises(ValueError):
                                pullback_lift(f, g, u, v)

    def test_pushout_universal_property_for_all_two_element_maps(self):
        for f in all_maps(self.two, self.two):
            for g in all_maps(self.two, self.two):
                obj, i, j = pushout(f, g)
                self.assertEqual(f.then(i), g.then(j))
                for u in all_maps(self.two, self.two):
                    for v in all_maps(self.two, self.two):
                        if f.then(u) == g.then(v):
                            candidates = [h for h in all_maps(obj, self.two) if i.then(h) == u and j.then(h) == v]
                            self.assertEqual(candidates, [pushout_descend(f, g, u, v)])
                        else:
                            with self.assertRaises(ValueError):
                                pushout_descend(f, g, u, v)

    def test_equalizer_and_coequalizer_universal_properties(self):
        for f in all_maps(self.two, self.two):
            for g in all_maps(self.two, self.two):
                eq, inclusion = equalizer(f, g)
                coeq, projection = coequalizer(f, g)
                self.assertEqual(inclusion.then(f), inclusion.then(g))
                self.assertEqual(f.then(projection), g.then(projection))
                for h in all_maps(self.two, self.two):
                    if h.then(f) == h.then(g):
                        self.assertEqual(sum(k.then(inclusion) == h for k in all_maps(self.two, eq)), 1)
                    if f.then(h) == g.then(h):
                        self.assertEqual(sum(projection.then(k) == h for k in all_maps(coeq, self.two)), 1)
        a, b = FiniteSet((0, 1)), FiniteSet(('a', 'b', 'c'))
        obj, q = coequalizer(FiniteMap(a, b, ('a', 'b')), FiniteMap(a, b, ('b', 'c')))
        self.assertEqual(len(obj), 1)  # transitive closure must identify all three

    def test_exponential_adjunction_both_inverse_laws(self):
        for x, a, b in itertools.product((self.empty, self.one, self.two), repeat=3):
            domain, _, _ = product(x, a)
            exp, _ = exponential(b, a)
            for f in all_maps(domain, b):
                self.assertEqual(uncurry(curry(f, x, a), a, b), f)
            for g in all_maps(x, exp):
                self.assertEqual(curry(uncurry(g, a, b), x, a), g)

    def test_cospan_gluing_associativity_up_to_interface_preserving_iso(self):
        port, apex = self.one, self.two
        segment = Cospan(FiniteMap(port, apex, (0,)), FiniteMap(port, apex, (1,)))
        left = segment.then(segment).then(segment)
        right = segment.then(segment.then(segment))
        self.assertNotEqual(left.left.target, right.left.target)
        isos = [f for f in all_maps(left.left.target, right.left.target)
                if len(set(f.images)) == len(right.left.target)
                and left.left.then(f) == right.left and left.right.then(f) == right.right]
        self.assertTrue(isos)
        identity = Cospan(FiniteMap.identity(port), FiniteMap.identity(port))
        glued = identity.then(segment)
        self.assertTrue(any(glued.left.then(f) == segment.left and glued.right.then(f) == segment.right
                            for f in all_maps(glued.left.target, apex) if len(set(f.images)) == len(apex)))
        tensor = segment.tensor(segment)
        self.assertEqual(len(tensor.left.source), 2)
        self.assertEqual(len(tensor.left.target), 4)
