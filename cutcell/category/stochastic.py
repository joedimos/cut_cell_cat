"""Finite stochastic kernels: a probabilistic extension of the exact set layer.

Matrices are row-stochastic: K[x,y]=P(Y=y|X=x). This is the finite-distribution
Kleisli category, evaluated in floating point rather than exact arithmetic.
"""
from dataclasses import dataclass
import numpy as np
from .finite_sets import FiniteSet, FiniteMap, product


@dataclass(frozen=True, eq=False)
class FiniteKernel:
    source: FiniteSet
    target: FiniteSet
    matrix: np.ndarray

    def __post_init__(self):
        matrix = np.array(self.matrix, dtype=float, copy=True)
        if (matrix.shape != (len(self.source), len(self.target)) or not np.all(np.isfinite(matrix))
                or np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1., atol=1e-12, rtol=0.)):
            raise ValueError('kernel must be finite, nonnegative and row-stochastic')
        matrix.setflags(write=False)
        object.__setattr__(self, 'matrix', matrix)

    @classmethod
    def deterministic(cls, mapping):
        matrix = np.zeros((len(mapping.source), len(mapping.target)))
        for i, y in enumerate(mapping.images):
            matrix[i, mapping.target.elements.index(y)] = 1.
        return cls(mapping.source, mapping.target, matrix)

    @classmethod
    def identity(cls, obj):
        return cls.deterministic(FiniteMap.identity(obj))

    def then(self, other):
        if self.target != other.source:
            raise ValueError('kernel endpoints must match')
        return FiniteKernel(self.source, other.target, self.matrix @ other.matrix)

    def tensor(self, other):
        """Independent product of kernels; independence is an explicit assumption."""
        source, _, _ = product(self.source, other.source)
        target, _, _ = product(self.target, other.target)
        return FiniteKernel(source, target, np.kron(self.matrix, other.matrix))

    def pushforward(self, probability):
        p = np.asarray(probability, dtype=float)
        if (p.shape != (len(self.source),) or not np.all(np.isfinite(p)) or np.any(p < 0)
                or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0.)):
            raise ValueError('a probability distribution on the source is required')
        return p @ self.matrix

    def expectation(self, observable):
        """Pull an observable back by conditional expectation (the dual action)."""
        f = np.asarray(observable, dtype=float)
        if f.shape != (len(self.target),) or not np.all(np.isfinite(f)):
            raise ValueError('a finite observable on the target is required')
        return self.matrix @ f

    def lumpability_defect(self, partition, coarse_kernel):
        """K Q - Q Kc; vanishing is strong lumpability for this deterministic Q."""
        if self.source != self.target or partition.source != self.source:
            raise ValueError('a source endokernel and matching partition are required')
        if coarse_kernel.source != partition.target or coarse_kernel.target != partition.target:
            raise ValueError('coarse endokernel must act on partition target')
        q = FiniteKernel.deterministic(partition)
        return self.then(q).matrix - q.then(coarse_kernel).matrix
