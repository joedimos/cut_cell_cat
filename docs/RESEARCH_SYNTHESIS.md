# Research synthesis: category theory, AC/DC ocean dynamics, and WeatherNext 3

This document states the integrated mathematical interpretation used by the repository after the v0.7 refinement. It is deliberately narrower than either primary source: the goal is to make every implemented categorical statement traceable to an actual equation or structural claim, while refusing to identify a finite abstraction with a trained global model.

Primary sources:

- Peter Korn, *Foundations of Global Ocean Climate Modelling at all Scales*, arXiv:2608.25679v3.
- Rasp et al., *WeatherNext 3: Increasing resolution and performance of global weather models with raw observations*, arXiv:2609.03582v1.

The machine-readable equation map is `cutcell/research/references.json`.

## 1. Three different kinds of morphism

The repository now distinguishes three structures that were previously too loosely adjacent.

1. **Deterministic physical evolution.** A conservative finite-volume update is a deterministic map between state spaces. When embedded into `FiniteKernel`, each row is a Dirac mass. It preserves copy in the Markov-category sense.
2. **Probabilistic forecast evolution.** A WeatherNext-style conditional forecast is a stochastic kernel. In general it preserves discard but not copy, reflecting the distinction between stochastic and deterministic morphisms.
3. **Observation or representation maps.** Native-resolution encoders, decoders, coarse-graining maps, and pooling maps change representation. Their interaction with dynamics is an additional naturality or lumpability condition, never automatic.

Keeping these separate prevents conservation, statistical calibration, and representational compatibility from being conflated.

## 2. WeatherNext 3 trajectory semantics

WN3 writes the autoregressive trajectory distribution as a second-order process over six-hour windows: the next window depends on the two preceding windows. A first-order Markov representation therefore lives on the pair state `X × X`, not on `X` itself.

`SecondOrderWeatherKernel` accepts a finite kernel

`K : X × X -> X`

and constructs the lifted kernel

`Khat((a,b),(b,c)) = K((a,b),c)`.

Iterating `Khat` reproduces the finite analogue of the factorization in WN3 equation (3). Tests verify both the trajectory product and repeated pair-state composition. This avoids the incorrect simplification that WeatherNext 3 is first-order on single forecast windows.

## 3. Multimodal native grids and the shared processor

WN3 does not force every data source onto one common input/output grid. It uses modality-specific encoders and decoders with a shared processor representation. The finite abstraction is therefore a diagram

`M_i --E_i--> Z --P--> Z --D_j--> M_j`.

`MultimodalLatentDiagram` exposes exactly this shape. A native modality can be encoded to a shared latent object, processed, and decoded to another native target object.

Two properties are intentionally *not* assumed:

- `D_i E_i = id` need not hold; `roundtrip_defect` measures failure.
- resolution change need not commute with latent processing; `resolution_naturality_defect` measures the square `F_fine ; R - R ; F_coarse`.

Sharing a processor mesh therefore does not itself prove resolution-independent dynamics.

## 4. Functional stochasticity and joint fields

WN3 builds on the Functional Generative Network idea: a noise realization selects a coherent stochastic forecast function rather than perturbing every output independently. `FunctionalGeneratorKernel` captures the finite categorical core of this statement.

Given

`G : Condition × Noise -> Target`

and a probability law on `Noise`, marginalizing the shared noise object yields a kernel

`Condition -> Target`.

If each target element denotes a complete spatial field, one noise draw selects one complete field. The test suite includes a counterexample showing that independently mixing target coordinates would create field combinations absent from the functional generator.

This is only the semantics of shared functional noise. The repository does not implement WN3's learned conditional normalization, transformer/GNN processor, neural station MLP, seed ensemble, or dropout.

## 5. WeatherNext 3 scoring structure

The forecast layer now distinguishes four statistical questions.

- `crps`: empirical or fair scalar/marginal CRPS.
- `multimodal_score`: separately normalized modality scores, preserving modality weights.
- `field_score`: local marginal CRPS plus the optional global-mean term used by WN3 for selected gridded variables.
- `pooled_crps`: CRPS after average or max spatial pooling, matching the evaluation order in WN3 Appendix A.2.2.

For pooled CRPS, the library requires explicit patch memberships and spatial weights. WN3 uses approximately equi-area latitude-longitude patches centered at every grid point and latitude weighting in the final spatial average. The repository does not synthesize those geographical neighborhoods from an abstract 1-D grid.

The crucial categorical/statistical distinction is that marginal scoring factors through pointwise projections, while pooled scoring first applies a many-to-one spatial observable to every ensemble member and the observation. Equal marginals therefore do not imply equal pooled skill or equal joint covariance structure.

## 6. Korn AC/DC: conservative carrier

Korn's AC/DC analysis replaces exact incompressibility by a pseudo-density continuity law. The implemented transport specialization preserves this coupling by advancing pseudo-mass and tracer content with the same face flux.

For pseudo-mass `m_i = V_i r_i` and tracer content `q_i = m_i C_i`, the discrete update is

`m+ = m + dt B F`

`q+ = q + dt B(F C_f)`.

The common incidence map makes componentwise conservation a commuting diagram. Coarse concentration is therefore obtained from aggregated content divided by aggregated pseudo-mass; unweighted averaging is not the corresponding morphism.

## 7. Thin-fluid calibration and acoustic scale

`thin_fluid_calibration` implements the scalar calibration from Korn equations (9)-(10):

`alpha = rho0 g H`,

`c_AC = sqrt(alpha/rho0) = sqrt(gH)`.

The associated barotropic Froude number `U/c_AC` quantifies the thin-fluid compressibility scale. The existing pressure module then supplies the column split and the explicit acoustic Störmer-Verlet specialization already mapped to Korn Sections 2.2 and 6.2.

## 8. Tracer variance: physical versus numerical mixing

Korn's Section 3.1 is now represented directly instead of being summarized only by transport conservation.

For jump `[C]_f`, centered mean `<C>_f`, reconstruction `(RC)_f`, pseudo-mass flux `Phi_f`, diffusivity `kappa_f`, and geometry `gamma_f`, the code evaluates

`integral chi dV = 2 sum_f kappa_f gamma_f [C]_f^2`,

`D_num = -sum_f Phi_f [C]_f ((RC)_f - <C>_f)`.

The pseudo-density-weighted variance tendency is then

`dV_h/dt = -rho0 D_num - rho0/2 integral chi dV`.

For centered reconstruction, `D_num = 0`. For upwind reconstruction,

`D_num = 1/2 sum |Phi_f| [C]_f^2`.

For the flux-corrected blend with limiter `lambda_f`,

`D_num = 1/2 sum (1-lambda_f)|Phi_f|[C]_f^2`.

Tests verify all three cases, including the fact that a completely general reconstruction need not have a nonnegative numerical sink.

`osborn_cox_diffusivity` then evaluates Korn equation (45) only from the explicit physical `chi` term. Numerical reconstruction dissipation is intentionally excluded.

## 9. Energy dissipation and mixing efficiency

`energy_dissipation_diagnostics` implements the separation in Korn equations (46)-(47): rotational viscous dissipation is physical, while divergence damping is an AC compressibility reservoir. They are returned as separate quantities.

`mixing_diagnostics` then exposes equations (54)-(56):

- numerical contamination ratio `q`,
- flux coefficient `Gamma = epsilon_b/epsilon`,
- flux Richardson number `R_f = epsilon_b/(epsilon_b+epsilon)`,
- diapycnal diffusivity `K_rho = epsilon_b/N^2 = Gamma epsilon/N^2`.

The advection mismatch and time-integration closure residual remain explicit inputs because the repository does not implement the complete mimetic AC/DC energy operator from Korn Sections 3.2 and 5.1.

## 10. Where category theory actually adds value

The synthesis is not that both papers are "categorical." The useful categorical statements are narrower:

- deterministic numerical updates embed in stochastic kernels;
- WN3 second-order dynamics become first-order on a product object;
- multimodal encoders/decoders compose through a shared latent object;
- functional noise is represented by marginalizing a shared noise object;
- coarse stochastic dynamics require strong lumpability, not merely pushforward of samples;
- conservative pseudo-mass/tracer transport is a chain-map/aggregation statement;
- pooled forecast evaluation is composition with a spatial observable and does not commute automatically with marginal scoring.

These are executable diagrams with explicit defects or laws. They are not metaphors for category theory.

## 11. Explicit non-claims

The repository still does not provide:

- a complete global AC/DC ocean circulation model;
- Korn's full C-grid mimetic momentum/energy discretization or Section 7-8 benchmark suite;
- WeatherNext 3 neural architecture, trained weights, observation ingestion, station coordinate network, operational ensemble, or reported skill;
- an automatic geographical reconstruction of WN3's approximately equi-area pooling windows;
- a proof that a physical ocean state coarse-graining is strongly lumpable;
- a proof that WN3's learned native-resolution heads satisfy exact categorical naturality.

Those omissions are encoded in `references.json` and are part of the repository's mathematical contract.
