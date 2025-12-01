import numpy as np
from collections import defaultdict
from typing import Any, List, Dict

class CategoricalKnowledgeGraph:
    """Knowledge graph for tracking categorical structures and patterns."""
    
    def __init__(self):
        self.patterns = []
        self.relationships = defaultdict(list)
        from semantic_search import SemanticPatternSearch
        self.semantic_search = SemanticPatternSearch()
        self.pattern_counter = 0
        
    def update(self, complex_obj: Any):
        """Update knowledge graph with current system state."""
        # Detect patterns in current state
        if hasattr(complex_obj, 'cell_states'):
            states = [cell.value for cell in complex_obj.cell_states]
            
            # Debug: Print state statistics
            state_range = max(states) - min(states) if states else 0
            print(f"    [Debug] State range: {state_range:.3f}, Max: {max(states):.3f}, Min: {min(states):.3f}")
            
            # Detect high-gradient regions with lower threshold
            gradients = []
            for i in range(1, len(states)):
                gradient = abs(states[i] - states[i-1])
                gradients.append(gradient)
            
            if gradients:
                max_grad = max(gradients)
                avg_grad = np.mean(gradients)
                max_grad_idx = np.argmax(gradients)
                
                print(f"    [Debug] Gradients - Max: {max_grad:.3e}, Avg: {avg_grad:.3e}, At: {max_grad_idx}")
                
                # Lower threshold and detect multiple gradient patterns
                gradient_threshold = max(avg_grad * 2, 1e-3)  # Dynamic threshold
                
                if max_grad > gradient_threshold:
                    pattern = {
                        'type': 'high_gradient',
                        'location': max_grad_idx,
                        'gradient': max_grad,
                        'average_gradient': avg_grad,
                        'threshold': gradient_threshold,
                        'time': self.pattern_counter
                    }
                    self.patterns.append(pattern)
                    self.semantic_search.add_pattern(
                        f"gradient_pattern_{self.pattern_counter}",
                        pattern,
                        f"High gradient {max_grad:.3e} at location {max_grad_idx} (avg: {avg_grad:.3e})"
                    )
                    self.pattern_counter += 1
                    print(f"    [Pattern] Added gradient pattern at {max_grad_idx}")
                
                # Also detect all gradients above threshold
                high_gradient_locations = []
                for i, grad in enumerate(gradients):
                    if grad > gradient_threshold:
                        high_gradient_locations.append((i, grad))
                
                if len(high_gradient_locations) > 0:
                    pattern = {
                        'type': 'multiple_gradients',
                        'locations': high_gradient_locations,
                        'count': len(high_gradient_locations),
                        'max_gradient': max_grad,
                        'time': self.pattern_counter
                    }
                    self.patterns.append(pattern)
                    self.semantic_search.add_pattern(
                        f"multi_gradient_{self.pattern_counter}",
                        pattern,
                        f"Multiple high gradients ({len(high_gradient_locations)} locations), max: {max_grad:.3e}"
                    )
                    self.pattern_counter += 1
            
            # Detect flux composition patterns (categorical structure)
            fluxes = []
            for i in range(1, len(states)):
                flux = states[i] - states[i-1]
                fluxes.append(flux)
            
            if len(fluxes) >= 3:
                total_flux = sum(fluxes)
                max_flux = max(abs(f) for f in fluxes) if fluxes else 0
                avg_flux = np.mean([abs(f) for f in fluxes]) if fluxes else 0
                
                print(f"    [Debug] Fluxes - Total: {total_flux:.3e}, Max: {max_flux:.3e}, Avg: {avg_flux:.3e}")
                
                # Check flux composition: flux n ≫ flux (n-1) ≈ 0
                flux_compositions = []
                for i in range(1, len(fluxes)):
                    composition = fluxes[i] * fluxes[i-1]  # flux n ≫ flux (n-1)
                    flux_compositions.append(abs(composition))
                
                max_composition = max(flux_compositions) if flux_compositions else 0
                avg_composition = np.mean(flux_compositions) if flux_compositions else 0
                
                print(f"    [Debug] Flux compositions - Max: {max_composition:.3e}, Avg: {avg_composition:.3e}")
                
                # Detect flux composition patterns with very low threshold
                composition_threshold = 1e-10
                if max_composition > composition_threshold:
                    max_comp_idx = np.argmax(flux_compositions)
                    pattern = {
                        'type': 'flux_composition',
                        'location': max_comp_idx,
                        'composition_value': max_composition,
                        'average_composition': avg_composition,
                        'max_flux': max_flux,
                        'time': self.pattern_counter
                    }
                    self.patterns.append(pattern)
                    self.semantic_search.add_pattern(
                        f"flux_composition_{self.pattern_counter}",
                        pattern,
                        f"Flux composition {max_composition:.2e} at interface {max_comp_idx}"
                    )
                    self.pattern_counter += 1
                    print(f"    [Pattern] Added flux composition pattern at {max_comp_idx}")
                
                # Detect conservation patterns (total flux near zero)
                conservation_error = abs(total_flux)
                conservation_threshold = 1e-8
                
                if conservation_error < conservation_threshold:
                    conservation_level = "exact" if conservation_error < 1e-12 else "good"
                    pattern = {
                        'type': f'{conservation_level}_conservation',
                        'conservation_error': conservation_error,
                        'total_flux': total_flux,
                        'time': self.pattern_counter
                    }
                    self.patterns.append(pattern)
                    self.semantic_search.add_pattern(
                        f"conservation_{self.pattern_counter}",
                        pattern,
                        f"{conservation_level.title()} conservation with error {conservation_error:.2e}"
                    )
                    self.pattern_counter += 1
                    print(f"    [Pattern] Added {conservation_level} conservation pattern")
            
            # Detect oscillation patterns
            if len(states) >= 5:
                oscillations = []
                for i in range(1, len(states)-1):
                    # Check for local maxima/minima (more sensitive detection)
                    is_maxima = states[i] > states[i-1] and states[i] > states[i+1]
                    is_minima = states[i] < states[i-1] and states[i] < states[i+1]
                    
                    if is_maxima or is_minima:
                        oscillation_strength = abs(states[i] - (states[i-1] + states[i+1])/2)
                        if oscillation_strength > 1e-4:  # Only count significant oscillations
                            oscillations.append((i, "maxima" if is_maxima else "minima", oscillation_strength))
                
                if len(oscillations) >= 2:
                    pattern = {
                        'type': 'oscillation',
                        'oscillation_count': len(oscillations),
                        'locations': [o[0] for o in oscillations],
                        'types': [o[1] for o in oscillations],
                        'strengths': [o[2] for o in oscillations],
                        'time': self.pattern_counter
                    }
                    self.patterns.append(pattern)
                    self.semantic_search.add_pattern(
                        f"oscillation_{self.pattern_counter}",
                        pattern,
                        f"Oscillation with {len(oscillations)} turning points"
                    )
                    self.pattern_counter += 1
                    print(f"    [Pattern] Added oscillation pattern with {len(oscillations)} points")
            
            # Detect boundary patterns
            if len(states) >= 2:
                left_boundary = states[0]
                right_boundary = states[-1]
                boundary_difference = abs(left_boundary - right_boundary)
                
                if boundary_difference > 0.1:
                    pattern = {
                        'type': 'boundary_difference',
                        'left_value': left_boundary,
                        'right_value': right_boundary,
                        'difference': boundary_difference,
                        'time': self.pattern_counter
                    }
                    self.patterns.append(pattern)
                    self.semantic_search.add_pattern(
                        f"boundary_{self.pattern_counter}",
                        pattern,
                        f"Boundary difference: {boundary_difference:.3f} (left: {left_boundary:.3f}, right: {right_boundary:.3f})"
                    )
                    self.pattern_counter += 1
    
    def search_patterns(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search over detected patterns."""
        results = self.semantic_search.search(query, top_k)
        print(f"    [Search] Query: '{query}', Found: {len(results)} results")
        for i, result in enumerate(results):
            print(f"      {i+1}. {result['pattern']['description']} (score: {result['score']:.1f})")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        pattern_types = {}
        for pattern in self.patterns:
            p_type = pattern.get('type', 'unknown')
            pattern_types[p_type] = pattern_types.get(p_type, 0) + 1
        
        return {
            'num_patterns': len(self.patterns),
            'pattern_types': pattern_types,
            'graph_density': len(self.relationships) / max(1, len(self.patterns))
        }