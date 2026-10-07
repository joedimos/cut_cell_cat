# cut_cell_cat

`cut_cell_cat` is a conservative finite-volume research framework with an executable category-theory layer and source-traceable mathematical interfaces to:

- **CliMA/Oceananigans.jl** for the bounded scalar-diffusion numerical architecture,
- **Peter Korn, arXiv:2608.25679v3**, *Foundations of Global Ocean Climate Modelling at all Scales*, for selected AC/DC ocean mathematics,
- **WeatherNext 3, arXiv:2609.03582v1**, for selected probabilistic trajectory, multimodal, functional-noise, and forecast-scoring structure.

The repository is intentionally explicit about scope. It is **not** an Oceananigans port, a complete AC/DC ocean circulation model, a trained WeatherNext implementation, or a formal proof of either paper.

## Integrated mathematical architecture

The current release treats three kinds of maps separately:

1. **Deterministic conservative dynamics** — finite-volume updates and exact finite-set maps.
2. **Probabilistic dynamics** — finite Markov kernels, Bayesian inversion, shared functional noise, and second-order autoregressive weather trajectories.
3. **Representation maps** — cell coarsening, native-resolution modality encoders/decoders, latent processors, and spatial pooling operators.

This separation matters because conservation, stochastic calibration, and cross-resolution consistency are different mathematical claims.

Read:

- [Research synthesis](docs/RESEARCH_SYNTHESIS.md)
- [Ocean + WeatherNext source contract](docs/OCEAN_WEATHER_MATHEMATICS.md)
- [Finite Markov-category model](docs/MARKOV_CATEGORY.md)
- [Category-theory guide](docs/CATEGORY_THEORY.md)
- [Numerical contract](docs/NUMERICS.md)
- [Oceananigans reference map](docs/OCEANANIGANS_REFERENCE.md)
- [Production validation contract](docs/PRODUCTION_READINESS.md)

The equation-to-code mapping is machine-readable in `cutcell/research/references.json`.

## Category theory

The category layer covers finite categories, functors, natural transformations, Yoneda, finite limits/colimits, cartesian closure, cospans, adjunctions, idempotent monads/comonads, finite-poset Kan extensions, presheaf/sheaf examples, chain complexes, weighted coarse maps, and finite Markov kernels.

```sh
python -m cutcell.category.showcase
# installed entry point
cut-cell-category --output category-report.json
```

Important distinctions are enforced by tests:

- conservative aggregation does not imply that diffusion commutes with coarsening;
- pushing a stochastic ensemble through a partition does not imply a closed coarse Markov model;
- an autonomous coarse stochastic model exists only when the strong-lumpability square `KQ = QKc` holds;
- stochastic kernels preserve discard, while copy preservation characterizes deterministic behavior in the finite examples.

## WeatherNext 3 structure

The WeatherNext layer now reflects substantially more than CRPS.

### Second-order trajectory factorization

WeatherNext 3 models the forecast trajectory as a **second-order Markov process over six-hour windows**. `SecondOrderWeatherKernel` represents

```text
K : X × X -> X
```

and lifts it to a first-order kernel on pair states:

```text
Khat((a,b),(b,c)) = K((a,b),c).
```

This is the finite analogue of the factorization in WeatherNext 3 equation (3). The code therefore does not silently treat WN3 as first-order on a single forecast window.

### Multimodal native-resolution diagram

`MultimodalLatentDiagram` represents the structural form

```text
native modality_i -> shared latent processor -> native modality_j
```

with separate encoders and decoders around one shared latent object. Round-trip and cross-resolution naturality defects are measurable; sharing a processor does not make those diagrams commute automatically.

### Functional stochasticity

`FunctionalGeneratorKernel` models a finite shared-noise construction

```text
Condition × Noise -> complete Target field
```

and marginalizes the noise law to obtain a stochastic kernel. One noise draw selects one complete target field/function, preserving the distinction between functional/joint stochasticity and independent pointwise perturbations.

This is a categorical abstraction of functional-noise semantics, **not** the learned FGN neural architecture, conditional normalization, transformer/GNN processor, seed ensemble, or epistemic dropout used by WeatherNext.

### WeatherNext scoring

The research layer includes:

- empirical and fair CRPS,
- separately normalized multimodal scoring,
- optional global-mean CRPS term for selected gridded variables,
- **average- and max-pooled CRPS** after spatial pooling.

`pooled_crps` requires explicit pool neighborhoods and spatial weights. WeatherNext 3 uses approximately equi-area latitude-longitude patches centered at every grid point and latitude weighting in the final average; this repository does not invent those geographical neighborhoods for a 1-D scalar grid.

## Korn AC/DC mathematics

The Korn layer now spans the main mathematical themes relevant to this repository rather than only pressure splitting.

### Thin-fluid calibration and pseudo-density

`thin_fluid_calibration` implements the paper's calibration

```text
alpha = rho0 g H
c_AC = sqrt(alpha / rho0) = sqrt(g H)
```

and reports the associated barotropic Froude number. `pseudo_density` implements the corresponding artificial-compressibility density map used by the transport specialization.

### Pressure split and acoustic stage

`ColumnPressureSplit` implements a stationary separable 2-D specialization of the hydrostatic/non-hydrostatic pressure decomposition, and its acoustic stage implements the tested constant-coefficient S3-b Störmer-Verlet specialization.

`dispersion` evaluates the corresponding stable dispersion branches and keeps exact roots separate from asymptotic errors.

### Consistent pseudo-mass/tracer transport

`consistent_tracer_step` advances pseudo-mass and tracer content with the same pseudo-mass flux:

```text
m+ = m + dt B F
q+ = q + dt B(F C_f)
C+ = q+ / m+
```

This yields an actual conservation diagram. Coarse concentration is aggregate content divided by aggregate pseudo-mass, not an unweighted average.

### Tracer variance and numerical mixing

`tracer_variance_diagnostics` implements Korn equations (38)-(43):

```text
∫ chi dV = 2 Σ_f kappa_f gamma_f [C]_f^2
D_num = -Σ_f Phi_f [C]_f ((RC)_f - <C>_f)
```

and the pseudo-density-weighted variance tendency

```text
dV_h/dt = -rho0 D_num - rho0/2 ∫ chi dV.
```

The implementation distinguishes centered, upwind, and flux-corrected reconstruction. Numerical mixing is computed face-by-face rather than inferred from the residual of a global budget.

`osborn_cox_diffusivity` uses the explicit physical `chi` term only, excluding the numerical reconstruction sink.

### Physical versus AC energy dissipation

`energy_dissipation_diagnostics` keeps the paper's rotational viscous dissipation and divergence/compressibility dissipation as separate reservoirs. The physical denominator used by mixing efficiency is therefore not automatically inflated by AC damping.

`mixing_diagnostics` exposes:

- numerical contamination ratio `q`,
- flux coefficient `Gamma = epsilon_b / epsilon`,
- flux Richardson number `R_f = epsilon_b / (epsilon_b + epsilon)`,
- diapycnal diffusivity `K_rho = epsilon_b / N^2`.

Advection mismatch and time-integration closure residual remain explicit inputs because the full mimetic AC/DC energy operator is not implemented here.

## Research showcase

```sh
python -m pip install .
cut-cell-research --output research-report.json
```

The report exercises:

- pressure splitting and acoustic residuals,
- Korn thin-fluid calibration,
- tracer-variance and energy-dissipation diagnostics,
- mixing-efficiency quantities,
- second-order WeatherNext trajectory semantics,
- shared functional-noise kernels,
- marginal and pooled forecast scores,
- conservative stochastic samples,
- stochastic strong-lumpability diagnostics.

All forecast scores are synthetic. They are **not WeatherNext skill measurements**.

## Install and validate

Python 3.10 or newer:

```sh
python -m pip install .
python -m unittest discover -s tests -v
python validation/convergence.py
cut-cell-cat --cells 64 --steps 100 --method ssprk3 --output results.json
cut-cell-category --output category-report.json
cut-cell-research --output research-report.json
```

Plotting is optional:

```sh
python -m pip install '.[plot]'
cut-cell-cat --plot simulation.png
```

## Core numerical scope

The production scalar solver provides:

- bounded 1-D finite-volume scalar diffusion,
- nonuniform volumes and face diffusivities,
- explicit cut-cell geometry and partial-bottom examples,
- zero-flux, prescribed-flux, and Dirichlet outer boundaries,
- stationary sources,
- forward Euler and SSPRK3,
- small-cell stability control,
- local/global/cumulative mass budgets,
- optional exact-rational Lean certificates of supplied budget snapshots,
- atomic result export.

Closed unforced diffusion preserves weighted mass, respects the documented bound-preserving conditions, and dissipates weighted quadratic energy under the stated timestep restriction.

## Architecture

| Layer | Responsibility |
|---|---|
| `cutcell/category/` | Finite categories, universal constructions, order structures, sheaves, cospans, finite Markov kernels |
| `cutcell/categorical_numerics.py` | Incidence, homology, chain maps, weighted coarse maps, dynamics defects |
| `cutcell/research/weathernext.py` | Second-order trajectory kernels, functional-noise semantics, multimodal latent diagrams |
| `cutcell/research/forecast.py` | CRPS, multimodal/global-mean scores, pooled CRPS, conservative stochastic residual example |
| `cutcell/research/ocean_diagnostics.py` | Korn thin-fluid, variance, physical/compressibility dissipation, mixing diagnostics |
| `cutcell/research/pressure.py` | Column pressure split, acoustic stage, dispersion |
| `cutcell/research/transport.py` | Pseudo-density and consistent tracer transport |
| `cutcell/grid.py`, `operators.py`, `model.py` | Scalar finite-volume runtime |
| `lean_verification.py` | Optional snapshot certificates; not a proof of the solver or papers |
| `tests/`, `validation/` | Mathematical law tests, numerical invariants, convergence and upstream checks |

## Explicit non-claims

The repository does not currently implement:

- global spherical ocean circulation,
- full nonlinear AC/DC momentum, Coriolis/buoyancy coupling, moving coordinates, or free-surface dynamics,
- Korn's full mimetic C-grid energy system or Section 7-8 benchmark/performance reproduction,
- WeatherNext 3 neural architecture, trained weights, raw satellite/station data ingestion, continuous coordinate-conditioned station head, training curriculum, or operational ensemble,
- WeatherNext's reported forecast accuracy or hardware performance,
- automatic construction of approximately equi-area spherical pooling windows,
- proof that learned WeatherNext heads satisfy exact categorical naturality,
- proof that arbitrary ocean coarse-graining is strongly lumpable.

These limits are part of the mathematical contract, not deferred assumptions.

## Version notes

- **0.4.0** — executable finite category-theory layer.
- **0.5.0** — first source-traceable Korn/WeatherNext research kernels.
- **0.6.0** — finite Markov-category, Bayesian inversion, and strong-lumpability layer.
- **0.7.0** — integrated Korn/WeatherNext synthesis: second-order weather kernels, multimodal latent diagrams, functional-noise semantics, pooled CRPS, thin-fluid calibration, tracer-variance/numerical-mixing separation, physical/compressibility energy diagnostics, and mixing-efficiency observables.
