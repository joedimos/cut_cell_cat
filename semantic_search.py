"""Deterministic keyword/concept retrieval with fixed-coordinate numeric features.

This is a local diagnostic index, not a language-model embedding service.
"""
import re
import numpy as np


class SemanticPatternSearch:
    FEATURES = ('gradient', 'conservation_error', 'location', 'max_flux', 'mass')
    TYPES = ('gradient', 'conservation', 'budget_failure', 'flux')
    ALIASES = {'slope': 'gradient', 'steep': 'gradient', 'balance': 'conservation',
               'preservation': 'conservation', 'transport': 'flux'}

    def __init__(self, max_patterns=1000):
        if isinstance(max_patterns, bool) or not isinstance(max_patterns, int) or max_patterns < 1:
            raise ValueError('max_patterns must be a positive integer')
        self.max_patterns = max_patterns
        self.pattern_database = []

    @classmethod
    def _tokens(cls, text):
        return {cls.ALIASES.get(t, t) for t in re.findall(r'[a-z0-9]+', text.lower())}

    def _compute_pattern_embedding(self, data, description):
        values = np.array([float(data.get(key, 0)) for key in self.FEATURES])
        if not np.all(np.isfinite(values)):
            raise ValueError('pattern features must be finite')
        values = np.sign(values) * np.log1p(np.abs(values))
        return np.r_[values, [float(data.get('type') == kind) for kind in self.TYPES]]

    def add_pattern(self, pattern_id, pattern_data, description=''):
        data = dict(pattern_data)
        pattern = {'id': pattern_id, 'data': data, 'description': description,
                   'embedding': self._compute_pattern_embedding(data, description)}
        self.pattern_database = [p for p in self.pattern_database if p['id'] != pattern_id]
        self.pattern_database.append(pattern)
        self.pattern_database = self.pattern_database[-self.max_patterns:]

    @staticmethod
    def _validate_top_k(top_k):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')

    def search(self, query, top_k=10):
        self._validate_top_k(top_k)
        tokens = self._tokens(query)
        results = []
        for pattern in self.pattern_database:
            description = self._tokens(pattern['description'])
            kind = self._tokens(pattern['data'].get('type', ''))
            score = 2 * len(tokens & description) + 3 * len(tokens & kind)
            if score:
                results.append({'pattern': pattern, 'score': float(score)})
        return sorted(results, key=lambda r: (-r['score'], r['pattern']['id']))[:top_k]

    def find_similar_patterns(self, pattern_id, top_k=5):
        self._validate_top_k(top_k)
        query = next((p for p in self.pattern_database if p['id'] == pattern_id), None)
        if query is None:
            return []
        results = []
        for pattern in self.pattern_database:
            if pattern['id'] == pattern_id:
                continue
            a, b = query['embedding'], pattern['embedding']
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            score = float(np.dot(a, b) / norm) if norm else 0.0
            results.append({'pattern': pattern, 'similarity': score})
        return sorted(results, key=lambda r: -r['similarity'])[:top_k]
