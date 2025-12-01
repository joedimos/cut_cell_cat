from typing import List, Dict, Any
import numpy as np
from collections import defaultdict

class SemanticPatternSearch:
   
    
    def __init__(self):
        self.pattern_embeddings = {}
        self.pattern_database = []
        self.index_built = False
        
    def add_pattern(self, pattern_id: str, pattern_data: Dict[str, Any], 
                   description: str = ""):
        """Add a pattern to the searchable database."""
        
        embedding = self._compute_pattern_embedding(pattern_data, description)
        
        self.pattern_database.append({
            'id': pattern_id,
            'data': pattern_data,
            'description': description,
            'embedding': embedding
        })
        
        self.index_built = False
    
    def _compute_pattern_embedding(self, pattern_data: Dict[str, Any], description: str) -> np.ndarray:
        """Compute comprehensive embedding vector for a pattern."""
        features = []
        
      
        if 'gradient' in pattern_data:
            features.extend([pattern_data['gradient'], pattern_data.get('average_gradient', 0)])
        if 'composition_value' in pattern_data:
            features.extend([pattern_data['composition_value'], pattern_data.get('average_composition', 0)])
        if 'conservation_error' in pattern_data:
            features.extend([pattern_data['conservation_error']])
        if 'location' in pattern_data:
            # Normalize location to [0,1]
            features.extend([pattern_data['location'] / 100.0])
        if 'oscillation_count' in pattern_data:
            features.extend([pattern_data['oscillation_count']])
        if 'difference' in pattern_data:
            features.extend([pattern_data['difference']])
        
        # Add description-based features
        desc_lower = description.lower()
        if 'gradient' in desc_lower:
            features.extend([1.0, 0.0, 0.0])  # Gradient feature vector
        elif 'flux' in desc_lower or 'composition' in desc_lower:
            features.extend([0.0, 1.0, 0.0])  # Flux feature vector
        elif 'conservation' in desc_lower:
            features.extend([0.0, 0.0, 1.0])  # Conservation feature vector
        elif 'oscillation' in desc_lower:
            features.extend([0.5, 0.5, 0.0])  # Oscillation feature vector
        else:
            features.extend([0.0, 0.0, 0.0])  # Unknown type
        
        # Add pattern type encoding
        pattern_type = pattern_data.get('type', 'unknown')
        type_encoding = {
            'high_gradient': [1, 0, 0, 0],
            'flux_composition': [0, 1, 0, 0], 
            'exact_conservation': [0, 0, 1, 0],
            'good_conservation': [0, 0, 0.5, 0],
            'oscillation': [0, 0, 0, 1],
            'boundary_difference': [0.5, 0, 0, 0.5],
            'multiple_gradients': [1, 0, 0, 0]
        }
        features.extend(type_encoding.get(pattern_type, [0, 0, 0, 0]))
        
        # Pad to fixed dimension (20 features)
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced search for patterns matching the query."""
        results = []
        
        query_lower = query.lower()
        keywords = query_lower.split()
        
        print(f"      [Search Debug] Query: '{query}', Keywords: {keywords}")
        
        for pattern in self.pattern_database:
            score = 0.0
            desc_lower = pattern['description'].lower()
            pattern_type = pattern['data'].get('type', 'unknown')
            
            # Exact keyword matching with weights
            for keyword in keywords:
                if keyword in desc_lower:
                    score += 2.0  # Higher weight for exact matches
                if keyword in pattern_type:
                    score += 3.0  # Even higher weight for type matches
            
            # Semantic concept matching
            if any(word in query_lower for word in ['gradient', 'slope', 'steep']):
                if any(word in desc_lower for word in ['gradient', 'steep', 'slope']):
                    score += 3.0
                if pattern_type in ['high_gradient', 'multiple_gradients']:
                    score += 4.0
            
            if any(word in query_lower for word in ['flux', 'composition', 'morphism']):
                if any(word in desc_lower for word in ['flux', 'composition', 'morphism']):
                    score += 3.0
                if pattern_type == 'flux_composition':
                    score += 4.0
            
            if any(word in query_lower for word in ['conservation', 'preservation', 'zero']):
                if any(word in desc_lower for word in ['conservation', 'zero', 'preservation']):
                    score += 3.0
                if 'conservation' in pattern_type:
                    score += 4.0
            
            if any(word in query_lower for word in ['oscillation', 'wave', 'periodic']):
                if any(word in desc_lower for word in ['oscillation', 'wave', 'periodic']):
                    score += 3.0
                if pattern_type == 'oscillation':
                    score += 4.0
            
            if any(word in query_lower for word in ['boundary', 'edge', 'border']):
                if any(word in desc_lower for word in ['boundary', 'edge', 'border']):
                    score += 3.0
                if pattern_type == 'boundary_difference':
                    score += 4.0
            
            # Feature-based matching for numerical queries
            if 'high' in query_lower and 'gradient' in query_lower:
                if pattern['data'].get('gradient', 0) > 0.01:
                    score += 2.0
            if 'low' in query_lower and 'gradient' in query_lower:
                if pattern['data'].get('gradient', 0) < 0.001:
                    score += 2.0
            
            if score > 0:
                results.append({
                    'pattern': pattern,
                    'score': score
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"      [Search Debug] Total matches before filtering: {len(results)}")
        
        return results[:top_k]
    
    def find_similar_patterns(self, pattern_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find patterns similar to the given pattern."""
        # Find the query pattern
        query_pattern = None
        for pattern in self.pattern_database:
            if pattern['id'] == pattern_id:
                query_pattern = pattern
                break
        
        if query_pattern is None:
            return []
        
        query_embedding = query_pattern['embedding']
        
        # Compute similarities
        similarities = []
        for pattern in self.pattern_database:
            if pattern['id'] == pattern_id:
                continue
            
            # Cosine similarity
            similarity = np.dot(query_embedding, pattern['embedding'])
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(pattern['embedding'])
            
            if norm_product > 0:
                similarity /= norm_product
            
            similarities.append({
                'pattern': pattern,
                'similarity': similarity
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
