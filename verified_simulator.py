"""Compatibility facade for the geometry-aware finite-volume implementation."""
from dataclasses import asdict, dataclass
from cutcell.io import atomic_json
import time
import numpy as np
from cutcell import CutCellGrid, DiffusionModel
from lean_verification import LeanVerificationServer, LeanCodeGenerator
from knowledge_graph import CategoricalKnowledgeGraph


@dataclass
class CellState:
    value: float = 0.0


class CutCellComplex:
    def __init__(self, num_cells, grid=None, diffusivity=0.1):
        self.grid = grid if grid is not None else CutCellGrid.uniform(num_cells)
        self.num_cells = self.grid.size
        from cutcell import DiffusionOperator
        self.operator = DiffusionOperator(self.grid, diffusivity)
        x = (self.grid.centers - self.grid.faces[0]) / np.ptp(self.grid.faces)
        self.cell_states = [CellState(float(v)) for v in 0.5 + 0.2 * np.cos(np.pi * x)]
        self.time = 0.0
        self.last_budget = None

    def compute_flux(self, i):
        if not 0 <= i <= self.num_cells:
            raise IndexError('face index out of range')
        return float(self.operator.flux([c.value for c in self.cell_states])[i])


class CategoricalSimulator:
    def __init__(self, resolution=50, use_multiscale=False, *, grid=None, diffusivity=0.1):
        if use_multiscale:
            raise NotImplementedError('multiscale coupling is not implemented')
        self.complex = CutCellComplex(resolution, grid, diffusivity)
        self.resolution = self.complex.num_cells
        self.use_multiscale = False
        self.kg = CategoricalKnowledgeGraph()


class VerifiedCategoricalSimulator(CategoricalSimulator):
    """Numerically budget-checked simulation, with optional Lean snapshot checks."""
    def __init__(self, resolution=50, use_multiscale=False, *, grid=None, diffusivity=0.1,
                 dt=0.001, method='ssprk3', use_lean=False, lean_path=None,
                 require_lean=False, history_limit=10000, require_nonnegative=False):
        super().__init__(resolution, use_multiscale, grid=grid, diffusivity=diffusivity)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        self.dt = dt
        self.model = DiffusionModel(self.complex.grid, [c.value for c in self.complex.cell_states],
                                    diffusivity, method=method, history_limit=history_limit,
                                    require_nonnegative=require_nonnegative)
        self.complex.operator = self.model.operator
        self.lean_server = LeanVerificationServer(lean_path, mock_mode=not (use_lean or require_lean))
        self.require_lean = require_lean
        self.lean_certificate_count = 0
        if require_lean and self.lean_server.mock_mode:
            raise RuntimeError("Lean is required but unavailable")
        self.code_generator = LeanCodeGenerator(self.lean_server)
        self.verified_flux_compute = self.code_generator.generate_verified_flux_computation()
        self.verified_evolve_step = self.code_generator.generate_verified_evolution_step(diffusivity, dt)
        self.verification_history = {key: [] for key in (
            'conservation_verified', 'max_errors', 'lean_theorems_proven',
            'verification_times', 'categorical_errors', 'theories_used', 'lean_status')}

    def run_verified(self, steps=30):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        for _ in range(steps):
            self._verified_evolution_step(self.model.iteration + 1)
            start = time.perf_counter()
            result = self._verify_current_state()
            result['verification_time'] = time.perf_counter() - start
            self._record_verification(result)
            if self.require_lean and not result['theorems_proven']:
                raise RuntimeError('required Lean certificate failed after numerical step; no successful run output')
            self.kg.update(self.complex)
        return self.model.history

    def _verified_evolution_step(self, step):
        # Preserve the historical public cell_states initialization interface.
        self.model.state = np.array([c.value for c in self.complex.cell_states])
        budget = self.model.step(self.dt)
        for cell, value in zip(self.complex.cell_states, self.model.state):
            cell.value = float(value)
        self.complex.last_budget = budget
        self.complex.time = self.model.time

    def _verify_current_state(self):
        if self.complex.last_budget is None:
            raise RuntimeError('no completed step to verify')
        passed, error, meta = self.lean_server.verify_budget(self.complex.last_budget)
        return {'conservation_verified': passed, 'max_conservation_error': error,
                'theorems_proven': int(meta['lean_proven']), 'categorical_error': error,
                'theory_used': meta['theory'], 'lean_status': meta['lean_status'],
                'conservation_properties': meta['conservation_properties']}

    def _record_verification(self, result):
        keys = {'conservation_verified': 'conservation_verified', 'max_errors': 'max_conservation_error',
                'lean_theorems_proven': 'theorems_proven', 'verification_times': 'verification_time',
                'categorical_errors': 'categorical_error', 'theories_used': 'theory_used',
                'lean_status': 'lean_status'}
        self.lean_certificate_count += result['theorems_proven']
        for target, source in keys.items():
            self.verification_history[target].append(result[source])
            if len(self.verification_history[target]) > self.model.history_limit:
                del self.verification_history[target][0]

    def search_patterns(self, query):
        return self.kg.search_patterns(query)

    def visualize(self, output_path='categorical_simulation.png'):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
        grid = self.complex.grid
        axes[0].plot(grid.centers[grid.active], self.model.state[grid.active])
        axes[0].set(xlabel='Position', ylabel='Concentration', title='Active cell states')
        axes[1].plot(grid.faces, self.model.operator.flux(self.model.state))
        axes[1].set(xlabel='Face position', ylabel='Integrated flux')
        axes[2].plot([b.time for b in self.model.history], [b.residual for b in self.model.history])
        axes[2].set(xlabel='Actual elapsed time', ylabel='Mass budget residual')
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def save_results(self, output_path='categorical_results.json'):
        from cutcell.reference import OCEANANIGANS_COMMIT
        grid = self.complex.grid
        result = {'schema_version': 2, 'reference_commit': OCEANANIGANS_COMMIT,
                  'method': self.model.method, 'requested_dt': self.dt, 'time': self.model.time,
                  'configuration': {'diffusivity': self.model.operator.diffusivity.tolist(),
                                    'source': self.model.source.tolist(),
                                    'left': asdict(self.model.operator.left),
                                    'right': asdict(self.model.operator.right),
                                    'safety': self.model.safety, 'atol': self.model.atol,
                                    'rtol': self.model.rtol,
                                    'require_nonnegative': self.model.require_nonnegative},
                  'grid': {k: getattr(grid, k).tolist() for k in
                           ('faces', 'centers', 'volumes', 'volume_fractions', 'apertures')},
                  'final_state': self.model.state.tolist(),
                  'final_fluxes': self.model.operator.flux(self.model.state).tolist(),
                  'budgets': [b.to_dict() for b in self.model.history],
                  'verification_history': self.verification_history,
                  'lean_used': self.lean_certificate_count > 0,
                  'lean_certificate_count': self.lean_certificate_count,
                  'history_limit': self.model.history_limit,
                  'retained_steps': len(self.model.history),
                  'initial_mass': self.model.initial_mass,
                  'total_exchange': self.model.total_exchange,
                  'proof_scope': 'concrete_snapshot_budget_only',
                  'total_steps': self.model.iteration, 'patterns_detected': len(self.kg.patterns)}
        atomic_json(output_path, result)


def main_verified():
    sim = VerifiedCategoricalSimulator(resolution=30)
    sim.run_verified(30)
    sim.save_results()
    print(f'{sim.model.iteration} steps; t={sim.model.time:.6g}; numerical mass budgets passed; '
          f'Lean certificates={sum(sim.verification_history["lean_theorems_proven"])}')
    return sim
