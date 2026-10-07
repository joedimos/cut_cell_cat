# Finite Markov-category layer

This note documents the stochastic categorical structure implemented in
`cutcell.category.stochastic`. It complements the deterministic material in
`CATEGORY_THEORY.md` and the forecast mathematics in
`OCEAN_WEATHER_MATHEMATICS.md`.

## 1. Objects and morphisms

Objects are finite sets. A morphism `K : X -> Y` is a row-stochastic matrix

\[
K_{xy} = \Pr(Y=y\mid X=x), \qquad K_{xy}\ge 0, \qquad \sum_y K_{xy}=1.
\]

Composition is matrix multiplication,

\[
(L\circ K)_{xz}=\sum_y K_{xy}L_{yz},
\]

which is exactly marginalization over the intermediate variable. Identity
morphisms are Dirac kernels induced by identity functions. The implementation
therefore realizes the finite-distribution Kleisli category in floating point.

## 2. Symmetric monoidal structure

The tensor product of objects is Cartesian product. For kernels
`K : X -> Y` and `L : A -> B`, the implemented tensor is the independent
product

\[
(K\otimes L)((x,a),(y,b)) = K(x,y)L(a,b).
\]

The code includes the symmetry and associator as deterministic kernels. They
are not treated as literal tuple equalities because the encoded carriers
`((X x Y) x Z)` and `(X x (Y x Z))` differ structurally.

## 3. Markov-category copy and discard

Every object has a deterministic copy map

\[
\Delta_X : X\to X\otimes X, \qquad x\mapsto (x,x),
\]

and a discard map

\[
!_X : X\to I,
\]

where `I` is the singleton monoidal unit. Every stochastic kernel preserves
discard. Copy is different: a general stochastic kernel does **not** satisfy

\[
K;\Delta_Y = \Delta_X;(K\otimes K).
\]

The left side samples once and duplicates the result; the right side creates
two conditionally independent samples. Equality holds precisely for
deterministic finite kernels. `FiniteKernel.copy_defect()` exposes this
structural distinction numerically.

This matters for forecast modeling: an ensemble-producing stochastic map cannot
be silently substituted for a deterministic state transformation in diagrams
that duplicate information.

## 4. States, joints, and Bayesian inversion

A probability distribution `p` on `X` is represented as a state `I -> X`.
Given a channel `K : X -> Y`, the induced output distribution is

\[
q(y)=\sum_x p(x)K(y\mid x).
\]

The joint distribution is

\[
p(x,y)=p(x)K(y\mid x).
\]

For every output with `q(y)>0`, the Bayesian inverse is

\[
K^\dagger_p(x\mid y)=\frac{p(x)K(y\mid x)}{q(y)}.
\]

A regular conditional distribution is not uniquely determined on `q`-null
outputs. The implementation makes this explicit: the default policy rejects
such a Bayesian inverse; an optional `null_policy="prior"` chooses the prior as
a particular version on null outputs. That choice is documented as a convention,
not as a theorem.

`FiniteKernel.posterior()` additionally performs likelihood conditioning. For a
nonnegative likelihood `ell(y)`,

\[
p(x\mid \ell) \propto p(x)\sum_y K(y\mid x)\ell(y).
\]

Zero-evidence events are rejected rather than normalized silently.

## 5. Strong lumpability and coarse graining

Let `Q : X -> C` be a surjective deterministic partition and let
`K : X -> X` be a Markov kernel. A coarse kernel `K_C : C -> C` is an exact
Markov quotient when

\[
K;Q = Q;K_C.
\]

For a deterministic partition this is the classical strong-lumpability
condition: all fine states in the same fibre of `Q` must induce the same
probability distribution over coarse blocks. `FiniteKernel.lumped()` checks that
condition and constructs the unique encoded quotient kernel when it holds.

This is the stochastic analogue of the repository's existing distinction
between conservative aggregation and falsely assumed commuting fine/coarse
dynamics. A coarse forecast transition model is admitted only when the diagram
actually commutes to tolerance.

## 6. Relation to the ocean/forecast layer

The repository's WeatherNext-referenced code implements probabilistic scoring
and conservative stochastic residual examples. The finite Markov-category layer
provides a mathematical language for finite stochastic transitions,
conditioning, and coarse graining around such forecast objects.

No claim is made that WeatherNext 3 itself is implemented as a finite Markov
category, nor that the neural architecture in that work reduces to the kernels
here. The connection is structural: probabilistic forecast transformations can
be represented and audited as stochastic morphisms once a finite state
abstraction has been chosen.

Likewise, strong lumpability is not asserted for the Oceananigans or Korn
operators in general. It is an explicit testable property of a chosen finite
stochastic abstraction.

## 7. Checked laws

`tests/test_markov_category.py` verifies:

- state/channel composition and joint-distribution construction;
- causality through discard;
- preservation of copy for deterministic kernels and its failure for genuinely
  stochastic kernels;
- symmetry and associator data for the tensor product;
- reconstruction of the joint law by Bayesian inversion;
- explicit handling of zero-probability outputs;
- likelihood conditioning and zero-evidence rejection;
- construction of exact strongly lumpable quotients;
- rejection of non-lumpable or non-surjective coarse-graining data.

The tests establish these identities for the finite encoded examples. They are
not a machine-checked proof of the abstract theory and do not turn the numerical
solver into a formally verified probabilistic program.
