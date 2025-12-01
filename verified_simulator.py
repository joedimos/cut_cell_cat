import numpy as np
from typing import List, Dict, Any
import time
import os
import sys
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lean_verification import LeanVerificationServer, LeanCodeGenerator
from knowledge_graph import CategoricalKnowledgeGraph

# Core simulation classes
class CellState:
    def __init__(self, value: float = 0.0):
        self.value = value

class CutCellComplex:
    def __init__(self, num_cells: int):
        self.num_cells = num_cells
        # Initialize with a distribution that better matches categorical structure
        x = np.linspace(0, 1, num_cells)
        # Use a smoother initial condition that satisfies conservation better
        self.cell_states = [CellState(0.5 + 0.2 * np.sin(2 * np.pi * xi)) for xi in x]
        
    def compute_flux(self, i: int) -> float:
        """Compute flux at cell i - categorical interpretation."""
        if 1 <= i < self.num_cells:
            return self.cell_states[i].value - self.cell_states[i-1].value
        return 0.0

class CategoricalSimulator:
    def __init__(self, resolution: int = 50, use_multiscale: bool = False):
        self.resolution = resolution
        self.use_multiscale = use_multiscale
        self.complex = CutCellComplex(resolution)
        self.kg = CategoricalKnowledgeGraph()


class VerifiedCategoricalSimulator(CategoricalSimulator):
    """Enhanced simulator with categorical Lean verification."""
    
    def __init__(self, resolution: int = 50, use_multiscale: bool = False):
        super().__init__(resolution, use_multiscale)
        
        # Initialize Lean integration with categorical verification
        self.lean_server = LeanVerificationServer(mock_mode=False)
        self.code_generator = LeanCodeGenerator(self.lean_server)
        
        # Generate verified algorithms
        self.verified_flux_compute = self.code_generator.generate_verified_flux_computation()
        self.verified_evolve_step = self.code_generator.generate_verified_evolution_step(0.1, 0.01)
        
        # Enhanced monitoring
        self.verification_history = {
            'conservation_verified': [],
            'max_errors': [],
            'lean_theorems_proven': [],
            'verification_times': [],
            'categorical_errors': [],
            'theories_used': []
        }
        
        if not self.lean_server.mock_mode:
            print("✓ CATEGORICAL Lean integration initialized")
        else:
            print(" Using enhanced categorical verification (no Lean)")
    
    def run_verified(self, steps: int = 30):
        """Run simulation with categorical Lean verification."""
        print(f"\nRunning CATEGORICAL Lean-verified simulation for {steps} steps...")
        print("Categorical flux composition verification active")
        print("-" * 70)
        
        total_verification_time = 0
        
        for step in range(1, steps + 1):
            # Perform verified evolution
            self._verified_evolution_step(step)
            
            # Continuous categorical verification with timing
            start_time = time.time()
            verification_result = self._verify_current_state()
            verification_time = time.time() - start_time
            total_verification_time += verification_time
            
            verification_result['verification_time'] = verification_time
            self._record_verification(verification_result)
            
            # Update knowledge graph with categorical patterns
            self.kg.update(self.complex)
            
            # Progress reporting with categorical status
            if step % 5 == 0:
                self._report_verified_progress(step, verification_result)
        
        print("-" * 70)
        self._final_verification_report(total_verification_time)
    
    def _verified_evolution_step(self, step: int):
        """Perform evolution step that preserves categorical structure."""
        current_states = [cell.value for cell in self.complex.cell_states]
        
        # Use conservative evolution that respects flux composition
        new_states = self.verified_evolve_step(current_states, 0.1, 0.001)
        
        for i, cell in enumerate(self.complex.cell_states):
            cell.value = new_states[i]
    
    def _verify_current_state(self) -> Dict[str, Any]:
        """Verify current state against categorical properties."""
        cell_states = [cell.value for cell in self.complex.cell_states]
        fluxes = [self.complex.compute_flux(i) for i in range(self.complex.num_cells)]
        
        # Use categorical Lean verification - FIXED: now handles 3 return values
        conservation_verified, categorical_error, metadata = self.lean_server.verify_conservation(
            cell_states, fluxes
        )
        
        return {
            'conservation_verified': conservation_verified,
            'max_conservation_error': categorical_error,
            'theorems_proven': 1 if conservation_verified else 0,
            'categorical_error': categorical_error,
            'theory_used': metadata.get('theory', 'unknown'),
            'conservation_properties': metadata.get('conservation_properties', [])
        }
    
    def _record_verification(self, result: Dict[str, Any]):
        """Record verification results."""
        self.verification_history['conservation_verified'].append(
            bool(result['conservation_verified'])
        )
        self.verification_history['max_errors'].append(
            float(result['max_conservation_error'])
        )
        self.verification_history['lean_theorems_proven'].append(
            int(result['theorems_proven'])
        )
        self.verification_history['verification_times'].append(
            float(result.get('verification_time', 0))
        )
        self.verification_history['categorical_errors'].append(
            float(result.get('categorical_error', 0))
        )
        self.verification_history['theories_used'].append(
            result.get('theory_used', 'unknown')
        )
    
    def _report_verified_progress(self, step: int, verification: Dict[str, Any]):
        """Report progress with categorical verification status."""
        icon = "✓" if verification['conservation_verified'] else "✗"
        time_str = f"{verification.get('verification_time', 0):.2f}s"
        cat_error = verification.get('categorical_error', 0)
        theory = verification.get('theory_used', 'unknown')
        patterns = len(self.kg.patterns)
        print(f"  Step {step:3d}: {icon} {theory:15} | Error: {cat_error:.2e} "
              f"| Time: {time_str} | Patterns: {patterns}")
    
    def _final_verification_report(self, total_verification_time: float):
        """Comprehensive final categorical verification report."""
        print("\n" + "=" * 70)
        print("CATEGORICAL LEAN VERIFICATION SUMMARY")
        print("=" * 70)
        
        total_steps = len(self.verification_history['conservation_verified'])
        verified_steps = sum(self.verification_history['conservation_verified'])
        verification_rate = verified_steps / total_steps * 100 if total_steps > 0 else 0
        
        print(f"\nCategorical Conservation Law Verification:")
        print(f"  Steps verified: {verified_steps}/{total_steps} ({verification_rate:.1f}%)")
        
        if self.verification_history['categorical_errors']:
            final_error = self.verification_history['categorical_errors'][-1]
            avg_error = np.mean(self.verification_history['categorical_errors'])
            print(f"  Final categorical error: {final_error:.2e}")
            print(f"  Average categorical error: {avg_error:.2e}")
        
        total_theorems = sum(self.verification_history['lean_theorems_proven'])
        print(f"\nCategorical Theorem Proving:")
        print(f"  Total theorems proven: {total_theorems}")
        
        if self.verification_history['verification_times']:
            avg_verification_time = np.mean(self.verification_history['verification_times'])
            print(f"  Total verification time: {total_verification_time:.2f}s")
            print(f"  Average verification time per step: {avg_verification_time:.2f}s")
        
        # Theory usage statistics
        theory_counts = {}
        for theory in self.verification_history['theories_used']:
            theory_counts[theory] = theory_counts.get(theory, 0) + 1
        
        print(f"\nTheory Usage:")
        for theory, count in theory_counts.items():
            percentage = count / total_steps * 100
            print(f"  {theory}: {count} steps ({percentage:.1f}%)")
        
        kg_stats = self.kg.get_statistics()
        print(f"\nCategorical Pattern Analysis:")
        print(f"  Total patterns detected: {kg_stats['num_patterns']}")
        if 'pattern_types' in kg_stats:
            for p_type, count in kg_stats['pattern_types'].items():
                print(f"    {p_type}: {count}")
        
        if not self.lean_server.mock_mode:
            print(f"\n✓ REAL categorical Lean theorem proving was used")
        else:
            print(f"Enhanced categorical verification was used (no Lean)")
    
    def search_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Semantic search over categorical patterns."""
        return self.kg.search_patterns(query, top_k=10)
    
    def visualize(self, output_path: str = 'categorical_simulation.png'):
        """Visualize simulation results with categorical insights."""
        try:
            import matplotlib.pyplot as plt
            
            states = [cell.value for cell in self.complex.cell_states]
            fluxes = [self.complex.compute_flux(i) for i in range(1, self.complex.num_cells)]
            
            plt.figure(figsize=(12, 10))
            
            # State plot
            plt.subplot(3, 1, 1)
            plt.plot(states, 'b-', linewidth=2, label='Cell States')
            plt.xlabel('Cell Index')
            plt.ylabel('State Value')
            plt.title('Categorical Cut-Cell Simulation - Cell States')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Flux plot
            plt.subplot(3, 1, 2)
            plt.plot(range(1, self.complex.num_cells), fluxes, 'r-', linewidth=2, label='Fluxes')
            plt.xlabel('Cell Interface')
            plt.ylabel('Flux Value')
            plt.title('Inter-cell Fluxes (categorical morphisms)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Flux composition plot (categorical structure) - FIXED
            plt.subplot(3, 1, 3)
            flux_compositions = []
            # Fix: Ensure same dimensions for x and y
            x_indices = []
            for i in range(1, len(fluxes)):  # Start from 1 to have flux[i] and flux[i-1]
                composition = fluxes[i] * fluxes[i-1]  # flux n ≫ flux (n-1)
                flux_compositions.append(abs(composition))
                x_indices.append(i)  # Interface where composition is defined
            
            plt.plot(x_indices, flux_compositions, 'g-', linewidth=2, 
                    label='Flux Compositions (should be ≈ 0)')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Interface Index')
            plt.ylabel('|Flux Composition|')
            plt.title('Categorical Flux Composition: |flux n ≫ flux (n-1)| ≈ 0')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Categorical visualization saved to {output_path}")
        except ImportError:
            print(f"  ⚠ Matplotlib not available, skipping visualization")
        except Exception as e:
            print(f"  ⚠ Visualization error: {e}")
    
    def save_results(self, output_path: str = 'categorical_results.json'):
        """Save categorical simulation results."""
        results = {
            'verification_history': {
                'conservation_verified': [bool(v) for v in self.verification_history['conservation_verified']],
                'max_errors': [float(e) for e in self.verification_history['max_errors']],
                'lean_theorems_proven': [int(t) for t in self.verification_history['lean_theorems_proven']],
                'verification_times': [float(t) for t in self.verification_history['verification_times']],
                'categorical_errors': [float(e) for e in self.verification_history['categorical_errors']],
                'theories_used': self.verification_history['theories_used']
            },
            'final_state': [float(cell.value) for cell in self.complex.cell_states],
            'final_fluxes': [float(self.complex.compute_flux(i)) for i in range(self.complex.num_cells)],
            'categorical_structure_used': True,
            'lean_used': not self.lean_server.mock_mode,
            'total_steps': len(self.verification_history['conservation_verified']),
            'patterns_detected': len(self.kg.patterns)
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"  ✓ Categorical results saved to {output_path}")


def main_verified():
    """Run verified demonstration with categorical Lean integration."""
    print("\n" + "=" * 70)
    print("CATEGORICAL LEAN-VERIFIED CUT-CELL SYSTEM")
    print("Categorical Theorem Proving & Structure Preservation")
    print("=" * 70)
    
    sim = VerifiedCategoricalSimulator(resolution=30, use_multiscale=False)
    sim.run_verified(steps=30)
    
    # Demonstrate categorical pattern search
    print("\n" + "=" * 70)
    print("CATEGORICAL PATTERN SEARCH")
    print("=" * 70)
    
    # Search for different types of patterns
    queries = ["flux composition", "conservation", "gradient", "oscillation"]
    for query in queries:
        results = sim.search_patterns(query)
        print(f"\nFound {len(results)} patterns matching '{query}':")
        for i, result in enumerate(results[:3]):  # Show top 3
            pattern = result['pattern']['data']
            desc = result['pattern']['description']
            score = result['score']
            print(f"  {i+1}. {desc} (score: {score:.1f})")
    
    sim.visualize(output_path='categorical_verification.png')
    sim.save_results('categorical_results.json')
    
    print("\n" + "=" * 70)
    print("CATEGORICAL LEAN VERIFICATION COMPLETE!")
    print("=" * 70)