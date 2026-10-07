# Ocean and WeatherNext mathematics: source contract

This repository pins two primary research references:

- Peter Korn, **arXiv:2608.25679v3**, *Foundations of Global Ocean Climate Modelling at all Scales* (5 September 2026).
- Rasp et al., **arXiv:2609.03582v1**, *WeatherNext 3: Increasing resolution and performance of global weather models with raw observations* (3 September 2026).

The integrated interpretation is developed in [RESEARCH_SYNTHESIS.md](RESEARCH_SYNTHESIS.md). The exact source locators, implementation symbols, validation tests, and non-claims are machine-readable in `cutcell/research/references.json`.

Nothing here is a copy of a trained WeatherNext model or a complete implementation of Korn's AC/DC global ocean model.

## Source-to-implementation map

| Source | Mathematical component | Implementation | Scope |
|---|---|---|---|
| Korn §2.1-2.2 (9)-(14) | thin-fluid AC calibration and pseudo-density | `thin_fluid_calibration`, `pseudo_density` | scalar calibration and density map |
| Korn §2.2 (16)-(21), §6.2 (99)-(100) | pressure decomposition | `ColumnPressureSplit` | stationary separable 2-D specialization |
| Korn §6.2.2 (107a-c) | explicit acoustic S3-b stage | `ColumnPressureSplit.acoustic_stage` | linear constant-coefficient stage |
| Korn §5.2.2 (78)-(79), H1-H4 | shared pseudo-mass/tracer flux | `consistent_tracer_step` | closed stationary 1-D specialization |
| Korn §3.1 (38)-(43) | physical tracer dissipation and numerical reconstruction sink | `flux_corrected_reconstruction`, `tracer_variance_diagnostics` | face-local finite-volume diagnostics |
| Korn §3.1 (45) | Osborn-Cox diffusivity | `osborn_cox_diffusivity` | caller supplies the steady/local assumptions |
| Korn §3.2 (46)-(48) | physical vs AC energy dissipation | `energy_dissipation_diagnostics` | local algebraic diagnostic; incomplete energy core |
| Korn §3.3 (54)-(56) | q, Gamma, R_f, K_rho | `mixing_diagnostics` | separated supplied budget terms |
| Korn §5.3 (82)-(89) | dispersion | `dispersion` | exact stable roots + distinct asymptotic errors |
| WN3 §2.1 (1)-(3) | second-order 6-hour trajectory factorization | `SecondOrderWeatherKernel` | exact finite-state analogue |
| WN3 §2.2, A.1.1 (A.1)-(A.3) | native modality encoders/decoders and shared processor | `MultimodalLatentDiagram` | finite compositional structure only |
| WN3 stochastic functional generation | shared functional noise | `FunctionalGeneratorKernel` | finite shared-noise field semantics, not neural FGN |
| WN3 A.1.1 (A.4)-(A.5), A.2.1 | CRPS conventions and multimodal loss | `crps`, `multimodal_score` | score semantics, no training pipeline |
| WN3 A.1.2 | global mean loss | `field_score` | optional global mean term |
| WN3 A.2.2 | average/max pooled CRPS | `pooled_crps` | caller supplies pools and latitude-like weights |

## Korn: conservative carrier and information gain

### Thin-fluid calibration

The paper calibrates the artificial-compressibility modulus to the full-depth barotropic gravity-wave scale:

\[
\alpha=\rho_0 g H,\qquad c_{AC}=\sqrt{\alpha/\rho_0}=\sqrt{gH}.
\]

`thin_fluid_calibration` reports this pair and the corresponding barotropic Froude number. It does not choose a global ocean mesh or derive the full AC/DC timestep hierarchy.

### Pseudo-density and consistent tracer transport

The transport specialization uses the same integrated pseudo-mass flux for pseudo-density and tracer content. For cell pseudo-mass `m` and content `q=mC`,

\[
m^+=m+\Delta t BF,\qquad q^+=q+\Delta t B(FC_f).
\]

This makes conservation a property of a common incidence map, not a post-hoc residual. Constant tracers are preserved under the documented donor-CFL hypotheses, and block aggregation transports both conserved quantities together.

### Tracer variance and numerical mixing

Korn §3.1 separates explicit physical tracer dissipation from reconstruction-induced numerical mixing. The code evaluates

\[
\int_\Omega \chi\,dV=2\sum_f \kappa_f\gamma_f[C]_f^2,
\]

and

\[
D_{num}=-\sum_f \Phi_f[C]_f\left((RC)_f-\langle C\rangle_f\right).
\]

The pseudo-density-weighted variance budget is

\[
\frac{dV_h}{dt}=-\rho_0D_{num}-\frac{\rho_0}{2}\int_\Omega\chi\,dV.
\]

For centered reconstruction, `D_num=0`. For upwind,

\[
D_{num}=\frac12\sum_f|\Phi_f|[C]_f^2.
\]

For the limited upwind/centered blend,

\[
D_{num}=\frac12\sum_f(1-\lambda_f)|\Phi_f|[C]_f^2.
\]

The tests verify these identities directly. A general reconstruction may have either sign, and the API does not clip that diagnostic to force dissipation.

`osborn_cox_diffusivity` uses only the explicit physical `chi` contribution. The numerical sink is deliberately excluded, following the paper's central distinction between diagnosed physical mixing and discretization removal.

### Energy dissipation and mixing efficiency

`energy_dissipation_diagnostics` evaluates separate discrete analogues of Korn equations (46) and (47): rotational viscous dissipation is physical, whereas divergence damping is the artificial-compressibility reservoir.

`mixing_diagnostics` then reports the contamination ratio

\[
q=\frac{\epsilon_c+|E_{adv}|+|r|}{\epsilon},
\]

and the related mixing quantities

\[
\Gamma=\frac{\epsilon_b}{\epsilon},\qquad
R_f=\frac{\epsilon_b}{\epsilon_b+\epsilon},\qquad
K_\rho=\frac{\epsilon_b}{N^2}.
\]

The full mimetic momentum/energy discretization is not implemented, so `E_adv` and the time-integration residual remain explicit inputs.

## WeatherNext 3: trajectory, modalities, stochastic functions, and scores

### Second-order trajectory law

WN3's autoregressive trajectory is second-order in six-hour windows. The finite implementation therefore uses

\[
K:X\times X\to X
\]

and the pair-state lift

\[
\widehat K((a,b),(b,c))=K((a,b),c).
\]

This is an exact finite analogue of the factorization in WN3 equation (3). It avoids replacing the paper's second-order process with an unjustified first-order kernel on `X`.

### Native-resolution modalities through one processor

WN3 uses separate encoders and decoders around a shared processor. `MultimodalLatentDiagram` represents

\[
M_i\xrightarrow{E_i}Z\xrightarrow{P}Z\xrightarrow{D_j}M_j.
\]

The repository explicitly measures two possible failures:

- decoder/encoder round trips need not be identities;
- fine/coarse representation maps need not commute with processed dynamics.

Thus a shared latent processor is a compositional architecture, not a proof of exact resolution naturality.

### Shared functional noise

`FunctionalGeneratorKernel` begins with a deterministic finite map

\[
G:\mathrm{Condition}\times\mathrm{Noise}\to\mathrm{Target}
\]

and marginalizes a supplied noise law. If a target element represents a full spatial field, one noise draw selects one whole field. This captures the distinction between a stochastic function/field and independent pointwise random variables.

It does **not** implement WN3's learned FGN conditional normalization, neural processor, dropout, or learned weights.

### CRPS and pooled spatial evaluation

`crps` implements empirical evaluation CRPS and the fair alternative. `multimodal_score` normalizes within each modality before applying modality weights. `field_score` optionally adds the globally pooled mean term used for selected WN3 gridded targets.

`pooled_crps` implements the evaluation order in WN3 Appendix A.2.2: every ensemble member and the observation are first average- or max-pooled, then CRPS is computed, then pooled-location scores are spatially weighted.

WN3 defines approximately equi-area latitude-longitude patches centered at every grid point and applies latitude weighting to the final spatial average. This library requires callers to supply those pools/weights explicitly rather than fabricating spherical geometry from a 1-D example.

## Categorical synthesis

The repository does not claim that Korn or WeatherNext 3 are category-theory papers. Category theory is used here to make compatibility statements precise:

- deterministic numerical maps embed into stochastic kernels;
- WN3 second-order evolution becomes first-order on a product object;
- multimodal maps factor through a common latent object;
- functional noise is marginalization over a shared noise object;
- autonomous stochastic coarse dynamics require the strong-lumpability square `KQ=QKc`;
- conservative pseudo-density/tracer aggregation is a chain-map statement;
- pooled scores factor through spatial observables and answer a different question from pointwise marginals.

Each claimed commuting law has a corresponding test; where no theorem is justified, the code exposes a defect instead.

## Reproduce the audits

```sh
python -m pip install .
python -m unittest discover -s tests -p 'test_research_*.py' -v
cut-cell-research --output research-report.json
```

The report is synthetic. It is a mathematical audit of implemented identities and abstractions, not a weather-skill or ocean-climate benchmark.

## Work still required

A genuine AC/DC ocean implementation still requires nonlinear momentum, Coriolis and buoyancy coupling, free-surface evolution, the complete mimetic C-grid operator suite, coupled pressure/pseudo-density synchronization, and the benchmark/performance experiments of Korn Sections 7-8.

A genuine WeatherNext 3 reproduction still requires the trained neural architecture and weights, raw multimodal observation processing, continuous station-coordinate head, stochastic conditional-normalization mechanism, dropout/seed uncertainty, training curriculum, versioned datasets, operational ensemble construction, geographical holdouts, and forecast-skill evaluation.
