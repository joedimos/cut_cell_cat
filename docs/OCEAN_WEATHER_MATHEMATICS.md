# Ocean and probabilistic forecast mathematics: source contract

The extension has two pinned primary references:

- Peter Korn, [2608.25679v3](https://arxiv.org/pdf/2608.25679v3),
  *Foundations of Global Ocean Climate Modelling at all Scales*, 5 September 2026.
  The user's unversioned URL resolves to this version at review time.
- Rasp et al., [WeatherNext 3, 2609.03582v1](https://arxiv.org/html/2609.03582v1),
  3 September 2026. DeepMind's [model documentation](https://developers.google.com/weathernext/guides/models)
  identifies the current family member. WN3 builds on WeatherNext 2's FGN approach.

Equation-to-function mappings, test names, versions, and explicit exclusions are
machine-readable in [references.json](../cutcell/research/references.json).
Nothing in the numerical kernels is a copy of a trained weather model.

## Source-to-implementation map

| Source location | Mathematical component | Implementation / restriction |
|---|---|---|
| Korn §2.2 (16)–(21), §6.2 (99)–(100) | Column pressure and residual split | `ColumnPressureSplit`; fixed separable 2-D grid |
| Korn §6.2.2 (107a–c) | Explicit acoustic Störmer–Verlet stage | `acoustic_stage`; constant coefficient S3-b only |
| Korn §5.2.2 (78)–(79), H1–H4 | Shared pseudo-mass and tracer flux | `consistent_tracer_step`; closed stationary 1-D upwind Euler |
| Korn §5.3 (82)–(89) | Wave dispersion | `dispersion`; real stable branches and separate asymptotic errors |
| WN3 A.1.1 (A.4)–(A.5), A.2.1 | Fair training / empirical evaluation CRPS | `crps`, `multimodal_score`; within-modality normalization |
| WN3 A.1.2 global mean loss, A.2.2 | Score after spatial pooling | `field_score`; global weighted mean only |
| WN3 A.1.1 (A.2) | Shared noise in a field-valued prediction | `ConservativeFluxEnsemble`; original linear flux adaptation |
| Original categorical extension | Kernel composition and coarse transition compatibility | `FiniteKernel`; finite stochastic matrices |

WN3 trains with fair two-sample CRPS and evaluates its mixed-seed ensemble with
empirical CRPS. Its optional global-mean term has coefficient 0.3 for selected
gridded variables and is omitted for sparse station targets. Our functions
expose those choices; they do not apply them indiscriminately.

## Reproduce the audit

```sh
python -m pip install .
cut-cell-research --output research-report.json
python -m unittest discover -s tests -p 'test_research_*.py' -v
```

From a checkout use `python -m cutcell.research.showcase`. Every reported forecast
score is synthetic, on a scalar diffusion example. It measures no real weather
skill. The new modules leave the existing diffusion solver and CLI unchanged.

## Discrete pressure contract

`ColumnPressureSplit(x_faces,z_faces)` stores an `(nx,nz)` field. Coordinates
increase left-to-right and bottom-to-top. Internal pressure gradients use
center distances; divergence uses cell widths. Side and bottom normal gradients
vanish; the top pressure is zero using its actual center-to-face distance.
These choices define linear maps Gx,Gz,Dx,Dz with

\[
L_H=D_xG_x,\qquad L_z=D_zG_z,\qquad L=L_H+L_z.
\]

The column calculation solves `Lz q=S`, where q is pressure divided by reference
density. It returns `S-Lq` and independently checks equality with `-LHq`.
If a correction e satisfies `Le=S-Lq`, then `L(q+e)=S`. This algebra is tested
against an independently assembled dense full solve; the dense solve is a test
oracle, not an operation in `split()`.

For the code's vertical discretization, `-Mz Lz` has positive diagonal and
negative adjacent off-diagonal entries. Its quadratic form is a sum of weighted
squared adjacent differences plus the top-boundary square, hence positive
definite. Batched Thomas elimination solves each column in O(nz), with total
O(nx*nz) storage/work. This differs from assembling a full global inverse.
The field geometry here is rectangular, not a global sphere or a moving cut mesh.

### Explicit acoustic stage

Using p=psi/rho0 and a=alpha/rho0, the implementation advances
`p_dot=-a Dv`, `v_dot=-Gp` by a pressure half-step, velocity step, and pressure
half-step. This is a linear, isolated stage. Its eigenmodes satisfy
`p_ddot=a Lp` and have squared frequencies from `a*(-L)`.

An absolute row-sum bound Lambda for `-L` gives a sufficient substep condition

\[
\delta t^2 a\Lambda<4.
\]

The code picks a strictly interior safety factor, subcycles over exactly the
requested interval, and caps the permitted work. Both directions contribute;
refining the vertical spacing increases the work. It does not claim the
horizontal-only restriction associated with the unimplemented S3-a split.
Because this is an undamped oscillator, it is reversible and need not reduce
divergence monotonically. Calling it an iterative Poisson convergence algorithm
would be wrong. The top acoustic boundary is the same zero-pressure boundary;
there is no prognostic free-surface displacement in this example.

Tests compare a low-frequency mode with its analytic cosine evolution and
verify second-order time refinement and time reversal. Spatial tests recover a
manufactured column pressure at approximately second order.

### Dispersion: exact roots versus asymptotics

The implementation computes the two roots for frequency squared of the source
biquadratics. It uses the large root followed by their product to recover the
small root, avoiding subtractive cancellation. Nonfinite inputs, zero vertical
wavenumber, complex branches, and negative squared frequencies are rejected.

A separate test constructs the five-variable Fourier generator for
(u,v,w,buoyancy,p), computes its eigenvalues, and compares the oscillation
frequencies. Reported leading errors are explicitly **asymptotic**. They are not
substituted for exact differences: the test verifies their discrepancy and
1/alpha refinement. The alpha argument is a squared-speed coefficient in this
normalized equation; no automatic mesh-dependent calibration is inferred from
other coefficient conventions in the paper.

## Consistent transport is a commuting conservation diagram

Let B be the existing signed incidence, positive for incoming face flux, and
let m_i=V_i*r_i, with positive relative pseudo-density r. The new transport
kernel uses one supplied integrated pseudo-mass flux F for both quantities:

\[
m^+=m+\Delta t BF,\qquad
q^+=mC+\Delta t B(F C_f),\qquad C^+=q^+/m^+.
\]

Closed faces have zero F. The same upwind donor reconstruction and timestep are
used throughout. Consequently, summing the incidence rows proves conservation
of total m and total q. For constant C=c, q^+=c*m^+, so constants remain
constant even when the velocity has nonzero divergence.

The strict outgoing-flux bound leaves positive donor mass. Each updated tracer
is a convex combination of the old local/donor values, so bounds are preserved.
Jensen's inequality applied to these combinations and summed over cells proves
nonincrease of the pseudo-mass-weighted second moment. No clipping is used.
The tests include disconnected wet regions, solids, nonuniform volumes, and
repeated steps.

For any block aggregation A, the inherited identity `AB=BcQ` applies separately
to m and q. Coarse concentration is therefore **Aq/Am**, not an unweighted
average. This is the categorical content: a compatible map transports the two
conservation laws together. Physical content `sum(V*C)` is another quantity;
a two-cell counterexample in the report demonstrates that it can change while
pseudo-mass content is exactly conserved. The kernel does not assert any global
long-time physical-error bound beyond its stated hypotheses.

## Probabilistic operators and conservative support

A field-valued conditional prediction is a probability kernel K(c,dc'). The
new fixed-mode example constructs samples with

\[
c'=\Phi_{\Delta t}(c)+\Delta t M^{-1}B\,W\xi.
\]

Here Phi is one actual SSPRK3 solver step, with its returned stable timestep;
W supplies zero flux at external and impermeable faces. One low-dimensional
noise vector drives the whole field. Therefore every member preserves each
closed component's scalar mass, regardless of noise distribution, up to roundoff.
Its covariance is spatially coupled and has rank bounded by the noise dimension.
The caller supplies noise, making sampling and reproducibility explicit.

This construction is an original constrained stochastic residual. It has no
learned transformer, normalization-layer noise injection, observation encoder,
or neural training. Gaussian perturbations do not guarantee positivity;
`require_nonnegative=True` rejects violations and never clips samples to claim
conservation. Trained forecast skill requires data and a separate evaluation.

`FiniteKernel` represents the finite-distribution Kleisli category:
row-stochastic matrices compose by multiplication, deterministic maps embed as
one-hot kernels, and tensor products represent independent coupling. A prior
pushes forward as pK; observables pull back by Kf. The identity
`(pK)f=p(Kf)` is tested, as are composition and the tower property.

For a deterministic partition Q, a coarse transition Kc is compatible only if

\[
K_fQ=QK_c.
\]

This is strong lumpability. `lumpability_defect` reports its failure. Merely
preserving mass or applying restriction to each forecast sample does not prove
that a closed Markov model exists on the coarse state. An ensemble may be pushed
forward consistently even when an autonomous coarse dynamic does not exist.

## Scores that preserve the intended statistical question

For scalar observations, empirical CRPS evaluates the finite forecast actually
delivered. Fair CRPS changes the pairwise denominator from m² to m(m-1),
requiring iid sampling for its unbiased population interpretation. The sorted
implementation avoids allocating an m-by-m pairwise array. Tests compare it
with an independent pairwise formula, including one-member, two-member,
replicated-member, and translation cases.

`multimodal_score` computes a weighted sum of separately normalized modality
scores, so duplicating grid locations with proportionate weights cannot drown
out stations. Boolean masks are applied before scoring; an entirely missing
modality is rejected instead of silently changing the loss definition. Variable
units/scales must be normalized by the caller; combining raw unlike units is
not automatic.

Pooling is applied **to each ensemble member and the observation before
scoring**, not to pointwise scores. Two joint ensembles with identical marginals
but opposite spatial dependence demonstrate why this matters. A pooled mean
still does not identify every feature of a joint law; it is one additional
observable, not a proof of calibration or physical consistency. Spatial weights
are supplied explicitly: volume weights suit this scalar example, area weights
suit a spherical field, and station weights answer a different sampling question.

## Work still required for a coupled scientific model

The available kernels now have equations, hypotheses, independent oracles, and
source locators. Further integration must supply nonlinear momentum, Coriolis
and buoyancy coupling, pressure/pseudo-density synchronization, consistent
integrated acoustic tracer fluxes, free-surface evolution, geometric operators,
and suitable benchmark cases. The independent transport and acoustic components
must not be advertised as an end-to-end AC/DC implementation.

A WeatherNext-scale extension separately requires licensed and versioned data,
multimodal observation processing, the neural architecture and optimization,
training/inference uncertainty semantics, geographical/time holdouts, joint and
marginal calibration, extremes, and operational latency evaluation. None of the
paper's forecast-accuracy or hardware-speed claims transfer to these kernels.

See [CATEGORY_THEORY.md](CATEGORY_THEORY.md) for the exact finite category layer
and [NUMERICS.md](NUMERICS.md) for the existing scalar solver's contract.
