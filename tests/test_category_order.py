import itertools
import unittest
from cutcell.category import (FiniteSet, FiniteMap, FinitePoset, MonotoneMap,
    GaloisConnection, image_adjunction, left_kan, right_kan, section_presheaf, glue_sections)


def monotone_maps(source, target):
    for values in itertools.product(target.objects, repeat=len(source.objects)):
        try:
            yield MonotoneMap(source, target, dict(zip(source.objects, values)))
        except ValueError:
            pass


class OrderTheory(unittest.TestCase):
    def test_adjunction_unit_counit_monad_and_comonad_laws(self):
        f = FiniteMap(FiniteSet((0, 1, 2)), FiniteSet(('a', 'b', 'unused')), ('a', 'a', 'b'))
        adj = image_adjunction(f)
        l, r, t, g = adj.left, adj.right, adj.monad, adj.comonad
        for x in l.source.objects:
            self.assertTrue(l.source.leq(x, t(x)))
            self.assertEqual(t(t(x)), t(x))
            self.assertEqual(l(r(l(x))), l(x))  # triangle equality in a poset
        for y in l.target.objects:
            self.assertTrue(l.target.leq(g(y), y))
            self.assertEqual(g(g(y)), g(y))
            self.assertEqual(r(l(r(y))), r(y))
        self.assertEqual(t(frozenset((0,))), frozenset((0, 1)))
        # Monad algebras are exactly the saturated (fixed) subsets.
        self.assertEqual(sum(t(x) == x for x in l.source.objects), 4)
        self.assertEqual(l.as_functor().then(r.as_functor()), t.as_functor())
        for x in l.source.objects:
            for y in l.source.objects:
                self.assertEqual(l(x | y), l(x) | l(y))
        for x in l.target.objects:
            for y in l.target.objects:
                self.assertEqual(r(x & y), r(x) & r(y))
        # Direct images need not preserve intersections.
        self.assertNotEqual(l(frozenset((0,)) & frozenset((1,))), l(frozenset((0,))) & l(frozenset((1,))))

    def test_invalid_adjunction_and_missing_bounds(self):
        p = FinitePoset.from_relation((0, 1), lambda x, y: x <= y)
        identity = MonotoneMap(p, p, {0: 0, 1: 1})
        zero = MonotoneMap(p, p, {0: 0, 1: 0})
        with self.assertRaises(ValueError):
            GaloisConnection(identity, zero)
        with self.assertRaises(ValueError):
            MonotoneMap(p, p, {0: 1, 1: 0})
        discrete = FinitePoset.from_relation((0, 1), lambda x, y: x == y)
        for method in (discrete.join, discrete.meet):
            for values in ((), (0, 1), (5,)):
                with self.assertRaises(ValueError):
                    method(values)
        self.assertEqual(p.join(()), 0)
        self.assertEqual(p.meet(()), 1)

    def test_kan_extension_universal_properties_exhaustively(self):
        c = FinitePoset.from_relation((0, 1), lambda x, y: x <= y)
        d = FinitePoset.from_relation((0, 1, 2), lambda x, y: x <= y)
        for k in monotone_maps(c, d):
            for f in monotone_maps(c, d):
                lan, ran = left_kan(k, f), right_kan(k, f)
                for h in monotone_maps(d, d):
                    self.assertEqual(all(d.leq(lan(x), h(x)) for x in d.objects),
                                     all(d.leq(f(x), h(k(x))) for x in c.objects))
                    self.assertEqual(all(d.leq(h(x), ran(x)) for x in d.objects),
                                     all(d.leq(h(k(x)), f(x)) for x in c.objects))
        # Empty comma category requires a bottom; absence must not invent one.
        discrete = FinitePoset.from_relation(('a', 'b'), lambda x, y: x == y)
        k = MonotoneMap(c, d, {0: 1, 1: 2})
        f = MonotoneMap(c, discrete, {0: 'a', 1: 'a'})
        with self.assertRaises(ValueError):
            left_kan(k, f)

    def test_presheaf_restriction_and_unique_gluing(self):
        f = section_presheaf((0, 1, 2), FiniteSet(('cold', 'hot')))
        full, left, right = frozenset((0, 1, 2)), frozenset((0, 1)), frozenset((1, 2))
        self.assertEqual(len(f.objects[frozenset()]), 1)
        for section in f.objects[full].elements:
            ls = f.arrows[(left, full)](section)  # opposite category reverses arrows
            rs = f.arrows[(right, full)](section)
            glued = glue_sections(full, (dict(zip((0, 1), ls)), dict(zip((1, 2), rs))))
            self.assertEqual(tuple(glued[x] for x in (0, 1, 2)), section)
            candidates = [s for s in f.objects[full].elements
                          if f.arrows[(left, full)](s) == ls and f.arrows[(right, full)](s) == rs]
            self.assertEqual(candidates, [section])
        self.assertEqual(glue_sections((), ()), {})
        for sections in (({0: 'cold'},), ({0: 0, 1: 0}, {1: 1, 2: 0}), ({0: 0, 1: 0, 2: 0, 3: 0},)):
            with self.assertRaises(ValueError):
                glue_sections(full, sections)


class ShowcaseIntegration(unittest.TestCase):
    def test_showcase_reports_checks_and_counterexample(self):
        from cutcell.category.showcase import showcase
        result = showcase()
        self.assertTrue(all(result['exact_checks'].values()))
        self.assertTrue(all(x <= result['numerical_absolute_tolerance']
                            for x in result['numerical_residuals'].values()))
        self.assertFalse(result['counterexample']['valid'])
        self.assertGreater(result['counterexample']['commutator_frobenius_norm'], 1.)
