"""Bounded diagnostic evidence graph; observations are not mathematical proofs."""
from collections import Counter
import numpy as np
from semantic_search import SemanticPatternSearch


class CategoricalKnowledgeGraph:
    def __init__(self, max_patterns=1000):
        if isinstance(max_patterns, bool) or not isinstance(max_patterns, int) or max_patterns < 1:
            raise ValueError('max_patterns must be a positive integer')
        self.max_patterns = max_patterns
        self.patterns, self.relationships = [], {}
        self.semantic_search = SemanticPatternSearch(max_patterns)
        self.pattern_counter = 0

    def _add(self, data, description):
        pattern_id = f'pattern_{self.pattern_counter}'
        data = dict(data, id=pattern_id)
        previous = self.patterns[-1]['id'] if self.patterns else None
        self.patterns.append(data)
        self.semantic_search.add_pattern(pattern_id, data, description)
        if previous:
            self.relationships[pattern_id] = {'previous_observation': previous}
        self.pattern_counter += 1
        if len(self.patterns) > self.max_patterns:
            removed = self.patterns.pop(0)['id']
            self.relationships.pop(removed, None)
            self.relationships = {k: v for k, v in self.relationships.items()
                                  if v['previous_observation'] != removed}

    def update(self, complex_obj):
        grid = complex_obj.grid
        states = np.array([c.value for c in complex_obj.cell_states])
        time = complex_obj.time
        connected = grid.apertures[1:-1] > 0
        if np.any(connected):
            gradients = np.abs(np.diff(states) / np.diff(grid.centers))
            gradients[~connected] = 0
            location = int(np.argmax(gradients)) + 1
            self._add({'type': 'gradient', 'gradient': float(gradients.max()),
                       'location': float(grid.faces[location]), 'time': time},
                      f'Maximum connected-face gradient {gradients.max():.6g}')
        budget = complex_obj.last_budget
        if budget is not None:
            self._add({'type': 'conservation' if budget.passed else 'budget_failure',
                       'conservation_error': abs(budget.residual), 'time': time,
                       'mass': budget.mass_after, 'tolerance': budget.tolerance},
                      f'Numerical mass conservation residual {abs(budget.residual):.6g}')
        self._add({'type': 'flux', 'time': time,
                   'max_flux': float(np.max(np.abs(complex_obj.operator.flux(states))))},
                  'Shared face flux observation')

    def search_patterns(self, query, top_k=10):
        return self.semantic_search.search(query, top_k)

    def get_statistics(self):
        return {'num_patterns': len(self.patterns),
                'pattern_types': dict(Counter(p['type'] for p in self.patterns)),
                'graph_density': len(self.relationships) / max(1, len(self.patterns)),
                'num_relationships': len(self.relationships)}
