"""A connected, executable category-theory tour based on a four-cell partition."""
import argparse
import json
import numpy as np
from . import (FiniteSet, FiniteMap, FinitePoset, MonotoneMap, image_adjunction,
               section_presheaf, glue_sections, yoneda_lift, yoneda_evaluate,
               left_kan, right_kan, pullback, pushout, Cospan)
from .. import CutCellGrid, DiffusionOperator
from ..categorical_numerics import Coarsening, closed_diffusion_diagram
from ..io import atomic_json


def showcase():
    fine, coarse = FiniteSet((0, 1, 2, 3)), FiniteSet(('left', 'right'))
    partition = FiniteMap(fine, coarse, ('left', 'left', 'right', 'right'))
    adj = image_adjunction(partition)
    field = section_presheaf(fine.elements, FiniteSet(('cold', 'hot')))
    full = frozenset(fine.elements)
    section = ('cold', 'cold', 'hot', 'hot')
    eta = yoneda_lift(field, full, section)
    glued = glue_sections(full, ({0: 'cold', 1: 'cold', 2: 'hot'}, {2: 'hot', 3: 'hot'}))
    # The kernel pair identifies exactly cells in the same partition block.
    kernel_pair, p, q = pullback(partition, partition)
    interface, patch = FiniteSet(('port',)), FiniteSet(('start', 'end'))
    segment = Cospan(FiniteMap(interface, patch, ('start',)), FiniteMap(interface, patch, ('end',)))
    joined = segment.then(segment)
    push, i, j = pushout(segment.right, segment.left)
    # Ordered levels: extend data from selected resolution levels.
    c = FinitePoset.from_relation((0, 2), lambda x, y: x <= y)
    d = FinitePoset.from_relation((0, 1, 2), lambda x, y: x <= y)
    k = MonotoneMap(c, d, {0: 0, 2: 2})
    f = MonotoneMap(c, d, {0: 0, 2: 2})
    lan, ran = left_kan(k, f), right_kan(k, f)
    exact = {
        'yoneda_round_trip': yoneda_evaluate(eta, full) == section,
        'unique_section_gluing': tuple(glued[x] for x in fine.elements) == section,
        'kernel_pair_commutes': p.then(partition) == q.then(partition),
        'pushout_commutes': segment.right.then(i) == segment.left.then(j),
        'monad_is_extensive': all(adj.left.source.leq(x, adj.monad(x)) for x in adj.left.source.objects),
        'monad_is_idempotent': all(adj.monad(adj.monad(x)) == adj.monad(x) for x in adj.left.source.objects),
        'comonad_is_idempotent': all(adj.comonad(adj.comonad(x)) == adj.comonad(x) for x in adj.left.target.objects),
    }
    grid, coarse_grid = CutCellGrid.uniform(4), CutCellGrid.uniform(2)
    numerical = Coarsening(grid, (0, 2, 4))
    op = DiffusionOperator(grid, 1.)
    diagram = closed_diffusion_diagram(op)
    lf = diagram['generator']
    lc = closed_diffusion_diagram(DiffusionOperator(coarse_grid, 1.))['generator']
    state = np.array([0., 1., 3., 2.])
    exact['incidence_chain_map'] = numerical.chain_map_holds()
    residuals = {
        'operator_factorization': float(np.max(np.abs(lf @ state - op.tendency(state)))),
        'mass_preservation': float(abs(numerical.coarse_volumes @ numerical.restrict_state(state) - grid.volumes @ state)),
        'weighted_adjoint': float(np.max(np.abs(numerical.coarse_volumes[:, None] * numerical.restrict - numerical.prolong.T * grid.volumes))),
    }
    defect = float(np.linalg.norm(numerical.dynamics_defect(lf, lc)))
    if not all(exact.values()) or any(v > 1e-12 for v in residuals.values()) or defect <= 1.:
        raise ArithmeticError('category showcase validation failed')
    return {
        'schema_version': 1,
        'scope': 'exact finite examples and floating-point numerical diagnostics; no Lean category proof',
        'story': 'four fine cells -> two coarse blocks -> patch fields -> gluing -> conservative dynamics',
        'exact_checks': exact,
        'constructions': {
            'kernel_pair_size': len(kernel_pair),
            'glued_cospan_apex_size': len(joined.left.target),
            'parallel_cospan_apex_size': len(segment.tensor(segment).left.target),
            'saturation_of_cell_0': sorted(adj.monad(frozenset((0,)))),
            'left_kan_levels': [lan(x) for x in d.objects],
            'right_kan_levels': [ran(x) for x in d.objects],
            'dual_graph_betti_0': diagram['betti_0'],
            'dual_graph_betti_1': diagram['betti_1'],
        },
        'numerical_residuals': residuals,
        'numerical_absolute_tolerance': 1e-12,
        'counterexample': {'claim': 'mass-preserving restriction commutes with diffusion',
                          'valid': False, 'commutator_frobenius_norm': defect},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='optional JSON report path; otherwise print JSON')
    args = parser.parse_args()
    try:
        result = showcase()
        if args.output:
            atomic_json(args.output, result)
            print(f'Category showcase passed; report={args.output}')
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, ArithmeticError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
