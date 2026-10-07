"""Finite stochastic kernels and explicit Markov-category structure.

Matrices are row-stochastic: ``K[x, y] = P(Y=y | X=x)``. Objects are finite
sets and morphisms are finite Markov kernels. The tensor is cartesian product
on objects and independent product on kernels. Copy/discard maps give the
standard Markov-category comonoid structure; only deterministic kernels preserve
copy in general.

All numerical equalities are evaluated in floating point. Conditioning on
zero-probability evidence is rejected unless an explicit null-event policy is
chosen.
"""
from dataclasses import dataclass
import numpy as np
from .finite_sets import FiniteSet, FiniteMap, product

UNIT = FiniteSet(((),))


def _probability(value, size, name="probability"):
    p = np.array(value, dtype=float, copy=True)
    if (p.shape != (size,) or not np.all(np.isfinite(p)) or np.any(p < 0)
            or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0.)):
        raise ValueError(f"{name} must be a finite probability vector of length {size}")
    return p


def _tolerance(value, name="atol"):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(value)


@dataclass(frozen=True, eq=False)
class FiniteKernel:
    source: FiniteSet
    target: FiniteSet
    matrix: np.ndarray

    def __post_init__(self):
        matrix = np.array(self.matrix, dtype=float, copy=True)
        if (matrix.shape != (len(self.source), len(self.target)) or not np.all(np.isfinite(matrix))
                or np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1., atol=1e-12, rtol=0.)):
            raise ValueError("kernel must be finite, nonnegative and row-stochastic")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def deterministic(cls, mapping):
        matrix = np.zeros((len(mapping.source), len(mapping.target)))
        for i, y in enumerate(mapping.images):
            matrix[i, mapping.target.elements.index(y)] = 1.
        return cls(mapping.source, mapping.target, matrix)

    @classmethod
    def identity(cls, obj):
        return cls.deterministic(FiniteMap.identity(obj))

    @classmethod
    def state(cls, obj, probability):
        """A state I -> X, where I is the monoidal unit and rows encode P(X)."""
        return cls(UNIT, obj, _probability(probability, len(obj))[None, :])

    @classmethod
    def copy(cls, obj):
        """Diagonal X -> X tensor X used by the finite Markov-category structure."""
        target, _, _ = product(obj, obj)
        return cls.deterministic(FiniteMap(obj, target, tuple((x, x) for x in obj.elements)))

    @classmethod
    def discard(cls, obj):
        """Unique causal map X -> I."""
        return cls.deterministic(FiniteMap(obj, UNIT, tuple(() for _ in obj.elements)))

    @classmethod
    def swap(cls, left, right):
        """Symmetry X tensor Y -> Y tensor X."""
        source, _, _ = product(left, right)
        target, _, _ = product(right, left)
        return cls.deterministic(FiniteMap(source, target,
                                           tuple((y, x) for x, y in source.elements)))

    @classmethod
    def associator(cls, left, middle, right):
        """Canonical ((X tensor Y) tensor Z) -> (X tensor (Y tensor Z)) associator."""
        left_middle, _, _ = product(left, middle)
        source, _, _ = product(left_middle, right)
        middle_right, _, _ = product(middle, right)
        target, _, _ = product(left, middle_right)
        images = tuple((x, (y, z)) for (x, y), z in source.elements)
        return cls.deterministic(FiniteMap(source, target, images))

    def then(self, other):
        if self.target != other.source:
            raise ValueError("kernel endpoints must match")
        return FiniteKernel(self.source, other.target, self.matrix @ other.matrix)

    def tensor(self, other):
        """Independent product of kernels; independence is an explicit assumption."""
        source, _, _ = product(self.source, other.source)
        target, _, _ = product(self.target, other.target)
        return FiniteKernel(source, target, np.kron(self.matrix, other.matrix))

    def pushforward(self, probability):
        p = _probability(probability, len(self.source))
        return p @ self.matrix

    def expectation(self, observable):
        """Pull an observable back by conditional expectation (the dual action)."""
        f = np.asarray(observable, dtype=float)
        if f.shape != (len(self.target),) or not np.all(np.isfinite(f)):
            raise ValueError("a finite observable on the target is required")
        return self.matrix @ f

    def is_deterministic(self, *, atol=1e-12):
        """Whether every row is a Dirac mass, up to the requested audit tolerance."""
        atol = _tolerance(atol)
        if len(self.source) == 0:
            return True
        maxima = np.max(self.matrix, axis=1)
        return bool(np.all(np.isclose(maxima, 1., atol=atol, rtol=0.)))

    def copy_defect(self):
        """f;Delta_Y - Delta_X;(f tensor f). It vanishes for deterministic kernels."""
        copied_after = self.then(FiniteKernel.copy(self.target))
        independent_after = FiniteKernel.copy(self.source).then(self.tensor(self))
        return copied_after.matrix - independent_after.matrix

    def joint(self, prior):
        """Return X tensor Y and P(x,y)=P(x)K(y|x), in product-object order."""
        p = _probability(prior, len(self.source), "prior")
        joint_object, _, _ = product(self.source, self.target)
        joint_probability = (p[:, None] * self.matrix).reshape(-1)
        return joint_object, joint_probability

    def bayes_inverse(self, prior, *, null_policy="error"):
        """Bayesian inverse Y -> X relative to a prior on X.

        For q(y)>0 this is p(x)K(y|x)/q(y). A conditional distribution on a
        q-null output is mathematically undetermined. ``null_policy='error'``
        exposes that fact; ``'prior'`` chooses the prior as an explicit version
        on null outputs.
        """
        p = _probability(prior, len(self.source), "prior")
        if null_policy not in {"error", "prior"}:
            raise ValueError("null_policy must be 'error' or 'prior'")
        q = p @ self.matrix
        reverse = np.empty((len(self.target), len(self.source)), dtype=float)
        supported = q > 0.
        if np.any(supported):
            reverse[supported] = ((self.matrix[:, supported].T * p[None, :])
                                  / q[supported, None])
        if np.any(~supported):
            if null_policy == "error":
                raise ValueError("Bayesian inverse is undefined on a zero-probability output")
            reverse[~supported] = p
        return FiniteKernel(self.target, self.source, reverse)

    def posterior(self, prior, likelihood):
        """Condition X using a nonnegative likelihood on Y after the channel X -> Y."""
        p = _probability(prior, len(self.source), "prior")
        likelihood = np.array(likelihood, dtype=float, copy=True)
        if (likelihood.shape != (len(self.target),) or not np.all(np.isfinite(likelihood))
                or np.any(likelihood < 0)):
            raise ValueError("likelihood must be finite and nonnegative on the target")
        evidence_given_x = self.matrix @ likelihood
        unnormalized = p * evidence_given_x
        normalizer = float(np.sum(unnormalized))
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("conditioning evidence has zero probability")
        return unnormalized / normalizer

    def lumped(self, partition, *, atol=1e-12):
        """Construct the strong-lumpability quotient kernel, or reject it.

        ``partition`` is a surjective deterministic map X -> C. For an
        endokernel K on X, rows in each fibre must induce the same distribution
        over coarse blocks. The returned Kc is then the unique kernel satisfying
        K;Q = Q;Kc within the requested tolerance.
        """
        if self.source != self.target or partition.source != self.source:
            raise ValueError("lumping requires an endokernel and a matching partition")
        atol = _tolerance(atol)
        if len(partition.target) == 0:
            return FiniteKernel(partition.target, partition.target, np.empty((0, 0)))
        q = FiniteKernel.deterministic(partition)
        aggregate = self.then(q).matrix
        rows = []
        for block in partition.target.elements:
            indices = [i for i, x in enumerate(partition.source.elements)
                       if partition(x) == block]
            if not indices:
                raise ValueError("partition must be surjective onto every coarse block")
            reference = aggregate[indices[0]]
            defect = max((float(np.max(np.abs(aggregate[i] - reference)))
                          for i in indices), default=0.)
            if defect > atol:
                raise ValueError(f"kernel is not strongly lumpable; fibre defect {defect:.3e}")
            rows.append(reference)
        return FiniteKernel(partition.target, partition.target, np.vstack(rows))

    def lumpability_defect(self, partition, coarse_kernel):
        """K Q - Q Kc; vanishing is strong lumpability for this deterministic Q."""
        if self.source != self.target or partition.source != self.source:
            raise ValueError("a source endokernel and matching partition are required")
        if coarse_kernel.source != partition.target or coarse_kernel.target != partition.target:
            raise ValueError("coarse endokernel must act on partition target")
        q = FiniteKernel.deterministic(partition)
        return self.then(q).matrix - q.then(coarse_kernel).matrix
