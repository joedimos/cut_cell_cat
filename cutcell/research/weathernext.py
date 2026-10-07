"""Finite categorical semantics for selected WeatherNext 3 equations.

This module does not reproduce the WeatherNext neural network. It gives exact
finite models of two structural statements in arXiv:2609.03582v1:

* Eq. (3): six-hour forecast windows form a second-order Markov process.
* Eqs. (A.2)-(A.3): modality-specific encode/process/decode maps communicate
  through a shared latent representation, while stochasticity defines a kernel.

The constructions are deliberately finite so their categorical laws can be
checked exactly or to explicit floating tolerances in unit tests.
"""
from dataclasses import dataclass
import numpy as np
from ..category.finite_sets import FiniteSet, product
from ..category.stochastic import FiniteKernel


def _distribution(value, size, name):
    p = np.asarray(value, dtype=float)
    if (p.shape != (size,) or not np.all(np.isfinite(p)) or np.any(p < 0)
            or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0.)):
        raise ValueError(f"{name} must be a finite probability vector of length {size}")
    return p.copy()


@dataclass(frozen=True)
class SecondOrderWeatherKernel:
    """Second-order Markov window model lifted to a first-order pair-state kernel.

    If K : X×X -> X gives p(x_{t+1}|x_{t-1},x_t), the lifted kernel
    Khat : X×X -> X×X is

        Khat((a,b),(b,c)) = K((a,b),c),

    and zero otherwise. Iterating Khat is exactly the finite analogue of WN3
    Eq. (3), after replacing each weather window by an element of X.
    """
    state: FiniteSet
    transition: FiniteKernel

    def __post_init__(self):
        pair, _, _ = product(self.state, self.state)
        if self.transition.source != pair or self.transition.target != self.state:
            raise ValueError("transition must be a kernel X×X -> X for the supplied state")

    @property
    def pair_state(self):
        return self.transition.source

    def lifted(self):
        pair = self.pair_state
        matrix = np.zeros((len(pair), len(pair)), dtype=float)
        target_index = {xy: i for i, xy in enumerate(pair.elements)}
        for i, (_, current) in enumerate(pair.elements):
            for k, nxt in enumerate(self.state.elements):
                matrix[i, target_index[(current, nxt)]] = self.transition.matrix[i, k]
        return FiniteKernel(pair, pair, matrix)

    def rollout(self, initial_pair_probability, steps):
        """Distribution on the two most recent windows after ``steps`` transitions."""
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        p = _distribution(initial_pair_probability, len(self.pair_state), "initial pair")
        lifted = self.lifted()
        for _ in range(steps):
            p = lifted.pushforward(p)
        return p

    def trajectory_probability(self, history, future):
        """Conditional probability of a finite future given exactly two history windows."""
        history, future = tuple(history), tuple(future)
        if len(history) != 2 or any(x not in self.state for x in history):
            raise ValueError("history must contain exactly two valid states")
        if any(x not in self.state for x in future):
            raise ValueError("future contains a state outside the state space")
        pair_index = {xy: i for i, xy in enumerate(self.pair_state.elements)}
        state_index = {x: i for i, x in enumerate(self.state.elements)}
        a, b = history
        probability = 1.0
        for c in future:
            probability *= self.transition.matrix[pair_index[(a, b)], state_index[c]]
            a, b = b, c
        return float(probability)


@dataclass(frozen=True)
class MultimodalLatentDiagram:
    """Finite schematic of WN3's native encoders/decoders and shared processor mesh.

    ``encoders[name]`` maps the native modality object into one common latent
    object. ``decoders[name]`` maps the latent object to that modality's native
    target object. The real WN3 maps are learned neural operators; finite kernels
    here expose only the compositional/categorical shape.
    """
    latent: FiniteSet
    encoders: dict
    decoders: dict

    def __post_init__(self):
        encoders, decoders = dict(self.encoders), dict(self.decoders)
        if not encoders or set(encoders) != set(decoders):
            raise ValueError("encoders and decoders must have the same nonempty modality keys")
        for name in encoders:
            if not isinstance(encoders[name], FiniteKernel) or encoders[name].target != self.latent:
                raise ValueError(f"encoder {name!r} must target the shared latent object")
            if not isinstance(decoders[name], FiniteKernel) or decoders[name].source != self.latent:
                raise ValueError(f"decoder {name!r} must start at the shared latent object")
        object.__setattr__(self, "encoders", encoders)
        object.__setattr__(self, "decoders", decoders)

    def translate(self, source_modality, target_modality, processor=None):
        """Encode one modality, optionally process in latent space, then decode another."""
        if source_modality not in self.encoders or target_modality not in self.decoders:
            raise KeyError("unknown modality")
        kernel = self.encoders[source_modality]
        if processor is not None:
            if (not isinstance(processor, FiniteKernel) or processor.source != self.latent
                    or processor.target != self.latent):
                raise ValueError("processor must be a latent endokernel")
            kernel = kernel.then(processor)
        return kernel.then(self.decoders[target_modality])

    def roundtrip_defect(self, modality, processor=None):
        """Deviation of decode∘process∘encode from identity on one native modality."""
        roundtrip = self.translate(modality, modality, processor)
        if roundtrip.source != roundtrip.target:
            raise ValueError("roundtrip defect requires matching native input/output objects")
        return roundtrip.matrix - FiniteKernel.identity(roundtrip.source).matrix

    def resolution_naturality_defect(self, fine, coarse, restriction, processor):
        """Compare process-then-restrict with restrict-then-process across resolutions.

        Vanishing is an additional resolution-consistency condition. WN3 does not
        claim this square commutes exactly; the diagnostic prevents that property
        from being inferred merely because modalities share a processor mesh.
        """
        if not isinstance(restriction, FiniteKernel):
            raise ValueError("restriction must be a finite kernel")
        fine_step = self.translate(fine, fine, processor)
        coarse_step = self.translate(coarse, coarse, processor)
        if restriction.source != fine_step.source or restriction.target != coarse_step.source:
            raise ValueError("restriction endpoints do not match fine/coarse modalities")
        return fine_step.then(restriction).matrix - restriction.then(coarse_step).matrix
