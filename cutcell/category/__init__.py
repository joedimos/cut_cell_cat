"""Exact, exhaustively checked finite examples; see docs/CATEGORY_THEORY.md.

This is an educational categorical layer, not formal verification of the solver.
"""
from .core import (FiniteCategory, Functor, NaturalTransformation, SetFunctor,
                   SetTransformation, representable, yoneda_lift, yoneda_evaluate)
from .finite_sets import (FiniteSet, FiniteMap, all_maps, product, pair, coproduct,
                         copair, pullback, pullback_lift, pushout, pushout_descend,
                         equalizer, coequalizer, exponential, curry, uncurry, Cospan)
from .order import (FinitePoset, MonotoneMap, GaloisConnection, image_adjunction,
                    left_kan, right_kan, section_presheaf, glue_sections)
from .stochastic import FiniteKernel, UNIT
