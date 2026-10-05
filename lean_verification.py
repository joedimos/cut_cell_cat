"""Numerical diagnostics and optional, explicitly scoped Lean evidence."""
import math
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

class CategoryTheoryFramework(Enum):
    """Categorical frameworks for verification."""
    CUT_CELL = "cut_cell"  
    STRUCTURED_COSPAN = "structured_cospan" 
    PETRI_NET = "petri_net"  
    DOUBLE_PUSHOUT = "double_pushout"  
    OPERAD = "operad" 

@dataclass
class CategoryTheorySignature:
    """Semantic signature for categorical theories."""
    framework: CategoryTheoryFramework
    objects: List[str]  
    morphisms: List[Tuple[str, str, str]] 
    composition_laws: List[str]  
    conservation_properties: List[str] 
    
    def semantic_distance(self, other: 'CategoryTheorySignature') -> float:
        """Calculate semantic similarity between theories."""
        
        framework_score = 1.0 if self.framework == other.framework else 0.5
        
        # Object overlap
        obj_overlap = len(set(self.objects) & set(other.objects)) / max(len(self.objects), len(other.objects), 1)
        
        # Morphism compatibility
        morph_overlap = len(set(m[0] for m in self.morphisms) & set(m[0] for m in other.morphisms)) / \
                       max(len(self.morphisms), len(other.morphisms), 1)
        
        # Conservation property overlap
        cons_overlap = len(set(self.conservation_properties) & set(other.conservation_properties)) / \
                      max(len(self.conservation_properties), len(other.conservation_properties), 1)
        
        return 0.3 * framework_score + 0.2 * obj_overlap + 0.3 * morph_overlap + 0.2 * cons_overlap

class TheoryRegistry:
    """Registry of categorical theory definitions."""
    
    def __init__(self):
        self.theories: Dict[str, CategoryTheorySignature] = {}
        self._initialize_builtin_theories()
    
    def _initialize_builtin_theories(self):
        """Initialize built-in categorical theories."""
        
        # Cut-cell finite volume theory
        self.register_theory("cut_cell_conservation", CategoryTheorySignature(
            framework=CategoryTheoryFramework.CUT_CELL,
            objects=["Cell", "Face", "Flux"],
            morphisms=[
                ("boundary", "Cell", "Face"),
                ("flow", "Face", "Flux"),
                ("divergence", "Flux", "Cell")
            ],
            composition_laws=[
                "V dc/dt = B F + V s",
                "internal face columns of B sum to zero"
            ],
            conservation_properties=["scalar_mass_balance"]
        ))
        
        # Structured cospan (stock-flow) theory
        self.register_theory("stock_flow", CategoryTheorySignature(
            framework=CategoryTheoryFramework.STRUCTURED_COSPAN,
            objects=["Stock", "Flow", "Rate"],
            morphisms=[
                ("inflow", "Flow", "Stock"),
                ("outflow", "Stock", "Flow"),
                ("rate_law", "Rate", "Flow")
            ],
            composition_laws=[
                "d(stock)/dt = ∑(inflows) - ∑(outflows)",
                "flow_composition: (f ∘ g)(t) = f(g(t))"
            ],
            conservation_properties=["total_stock", "flow_balance"]
        ))
        
        # Petri net (chemical reaction) theory
        self.register_theory("petri_net", CategoryTheorySignature(
            framework=CategoryTheoryFramework.PETRI_NET,
            objects=["Place", "Transition", "Token"],
            morphisms=[
                ("consume", "Place", "Transition"),
                ("produce", "Transition", "Place"),
                ("fire", "Transition", "Transition")
            ],
            composition_laws=[
                "firing_rule: enabled(t) → fire(t)",
                "token_conservation: ∑(tokens) = constant"
            ],
            conservation_properties=["token_count", "stoichiometry"]
        ))
        
        # Double pushout (graph rewriting) theory
        self.register_theory("graph_rewrite", CategoryTheorySignature(
            framework=CategoryTheoryFramework.DOUBLE_PUSHOUT,
            objects=["Graph", "Interface", "Rule"],
            morphisms=[
                ("match", "Interface", "Graph"),
                ("rewrite", "Graph", "Graph"),
                ("glue", "Interface", "Graph")
            ],
            composition_laws=[
                "pushout_square: match ∘ glue = rewrite",
                "interface_preservation"
            ],
            conservation_properties=["connectivity", "node_types"]
        ))
    
    def register_theory(self, name: str, signature: CategoryTheorySignature):
        """Register a new theory."""
        self.theories[name] = signature
    
    def find_similar_theories(self, signature: CategoryTheorySignature, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Find theories similar to the given signature."""
        similarities = []
        for name, theory in self.theories.items():
            distance = signature.semantic_distance(theory)
            if distance >= threshold:
                similarities.append((name, distance))
        return sorted(similarities, key=lambda x: x[1], reverse=True)


class LeanVerificationServer:
    """Optional Lean certificates of concrete mass budgets, not solver proofs.

    Numeric success and a kernel-checked certificate are separate facts. A
    missing or failing Lean process never counts as a theorem.
    """

    def __init__(self, lean_path=None, mock_mode=False, timeout=10.0):
        self.lean_path = lean_path or shutil.which('lean')
        self.mock_mode = bool(mock_mode or not self.lean_path)
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        self.timeout = timeout
        self.theory_registry = TheoryRegistry()
        self.current_theory = self.theory_registry.theories['cut_cell_conservation']

    def set_theory(self, theory_name):
        if theory_name not in self.theory_registry.theories:
            raise ValueError(f'unknown theory: {theory_name}')
        self.current_theory = self.theory_registry.theories[theory_name]

    def suggest_theories(self, objects, morphisms):
        signature = CategoryTheorySignature(CategoryTheoryFramework.CUT_CELL,
                     objects, [(m, '', '') for m in morphisms], [], [])
        return [name for name, _ in self.theory_registry.find_similar_theories(signature, 0.3)]

    @staticmethod
    def _certificate(budget):
        # Encode the supplied finite IEEE values exactly as rationals. No
        # decimal truncation, relaxed threshold, sorry, axiom, or native_decide.
        q = lambda x: Fraction.from_float(float(x))
        residual = abs(q(budget.mass_after) - q(budget.mass_before)
                       - q(budget.boundary_exchange) - q(budget.source_exchange))
        tolerance = q(budget.tolerance)
        lhs = residual.numerator * tolerance.denominator
        rhs = tolerance.numerator * residual.denominator
        return (f'-- Concrete snapshot only; not a proof of the numerical solver.\n'
                f'theorem snapshot_budget : ({lhs} : Nat) ≤ {rhs} := by decide\n')

    def verify_budget(self, budget):
        values = [budget.mass_after, budget.mass_before, budget.boundary_exchange,
                  budget.source_exchange, budget.tolerance]
        if not all(np.isfinite(x) for x in values) or budget.tolerance < 0:
            raise ValueError('budget values must be finite with nonnegative tolerance')
        error = abs(math.fsum([budget.mass_after, -budget.mass_before,
                              -budget.boundary_exchange, -budget.source_exchange]))
        passed = error <= budget.tolerance
        metadata = {'theory': 'cut_cell', 'conservation_properties': ['scalar_mass_balance'],
                    'numeric_passed': passed, 'lean_proven': False,
                    'proof_scope': 'concrete_snapshot_budget', 'backend': 'numerical',
                    'lean_status': 'disabled_or_unavailable'}
        if not self.mock_mode:
            metadata['lean_status'] = 'not_run_numeric_failure'
            if passed:
                try:
                    with tempfile.TemporaryDirectory(prefix='cutcell-lean-') as directory:
                        path = Path(directory) / 'Budget.lean'
                        path.write_text(self._certificate(budget), encoding='utf-8')
                        result = subprocess.run([self.lean_path, str(path)], capture_output=True,
                                                text=True, timeout=self.timeout, check=False)
                    # Generated code has no holes; warnings still fail closed.
                    proved = result.returncode == 0 and not (result.stdout.strip() or result.stderr.strip())
                    metadata.update(lean_proven=proved, backend='lean' if proved else 'numerical',
                                    lean_status='proved' if proved else 'failed')
                except subprocess.TimeoutExpired:
                    metadata['lean_status'] = 'timeout'
                except OSError:
                    metadata['lean_status'] = 'unavailable'
        return passed, error, metadata

    def verify_conservation(self, cell_states, fluxes, theory_name=None):
        """Deprecated snapshot API cannot establish an evolution mass budget."""
        if theory_name:
            self.set_theory(theory_name)
        return False, float('inf'), {
            'theory': self.current_theory.framework.value, 'lean_proven': False,
            'conservation_properties': [], 'backend': 'unverified',
            'reason': 'Use verify_budget with before/after mass and integrated exchanges.'}

    def export_theory_to_lean(self, theory_name, output_path):
        """Export descriptive comments only; registry entries are not theorems."""
        theory = self.theory_registry.theories[theory_name]
        lines = [f'-- Descriptive schema: {theory_name}; NOT a verified theory.']
        lines += [f'-- {law}' for law in theory.composition_laws]
        Path(output_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


class LeanCodeGenerator:
    """Legacy numerical factories; generated Python has no formal certificate."""

    def __init__(self, verification_server):
        self.server = verification_server

    def generate_verified_flux_computation(self, tolerance=1e-10):
        from cutcell import CutCellGrid, DiffusionOperator
        def compute_flux(states, n):
            if not 0 <= n <= len(states):
                raise IndexError('face index out of range')
            return float(DiffusionOperator(CutCellGrid.uniform(len(states))).flux(states)[n])
        return compute_flux

    def generate_verified_evolution_step(self, diffusion, dt):
        from cutcell import CutCellGrid, DiffusionModel
        def evolve_step(states, diffusion=diffusion, dt=dt):
            model = DiffusionModel(CutCellGrid.uniform(len(states)), states, diffusion)
            # Subcycle to the requested elapsed time instead of silently reducing it.
            model.run_until(dt, max_dt=dt)
            return model.state.tolist()
        return evolve_step

    def generate_stock_flow_dynamics(self, dt=0.01):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        def stock_flow_step(stocks, flows):
            new = np.asarray(stocks, dtype=float).copy()
            if not np.all(np.isfinite(new)) or np.any(new < 0):
                raise ValueError('stocks must be finite and nonnegative')
            for source, target, rate in flows:
                if not (0 <= source < len(new) and 0 <= target < len(new)) or not np.isfinite(rate) or rate < 0:
                    raise ValueError('invalid flow')
                amount = min(rate * dt, new[source])
                new[source] -= amount
                new[target] += amount
            return new.tolist()
        return stock_flow_step

    def generate_petri_net_step(self):
        from collections import Counter
        def petri_step(tokens, transitions):
            new = list(tokens)
            if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in new):
                raise ValueError('tokens must be nonnegative integers')
            for inputs, outputs in transitions:
                if any(not isinstance(i, int) or not 0 <= i < len(new) for i in inputs + outputs):
                    raise ValueError('invalid place')
                required = Counter(inputs)
                if all(new[i] >= count for i, count in required.items()):
                    for i in inputs:
                        new[i] -= 1
                    for i in outputs:
                        new[i] += 1
            return new
        return petri_step
