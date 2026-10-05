# Strict Oceananigans reference mapping

## Pinned evidence

Reference repository: [CliMA/Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl).
Inspected commit: **819a245b837041d01fcf273d74a8b45db81ea14a**.
The machine-readable [manifest](oceananigans_reference.json) records every inspected
path and Git blob SHA. References below point to that exact revision; future
upstream changes do not silently change the meaning of this comparison.

This revision independently implements a bounded, one-dimensional scalar-diffusion
subset in Python. Source correspondence is not a claim of runtime equivalence,
Oceananigans endorsement, or feature parity. No upstream source is vendored.

## Source-to-implementation-to-test mapping

| Upstream source at the pinned revision | Adopted principle | Local implementation | Evidence / explicit difference |
|---|---|---|---|
| [finite_volume.md](https://github.com/CliMA/Oceananigans.jl/blob/819a245b837041d01fcf273d74a8b45db81ea14a/docs/src/numerical_implementation/finite_volume.md) | Cell averages, face-centered fluxes, cell volumes | `cutcell/grid.py`, `cutcell/operators.py` | Cell-center placement and cell-average cosine convergence tests; only scalar 1-D fields |
| [divergence_operators.jl](https://github.com/CliMA/Oceananigans.jl/blob/819a245b837041d01fcf273d74a8b45db81ea14a/src/Operators/divergence_operators.jl) | Area-weighted flux differences divided by cell volume | `DiffusionOperator.tendency` | Random-geometry divergence theorem; volume-weighted symmetry; internal shared-face cancellation |
| [partial_cell_bottom.jl](https://github.com/CliMA/Oceananigans.jl/blob/819a245b837041d01fcf273d74a8b45db81ea14a/src/ImmersedBoundaries/partial_cell_bottom.jl) | Minimum bottom-cell height and adjusted adjacent center distance | `CutCellGrid.partial_bottom` | Explicit height/centroid/face-distance regression; one column only, rejects out-of-domain bottom instead of upstream clamping |
| [abstract_scalar_diffusivity_closure.jl](https://github.com/CliMA/Oceananigans.jl/blob/819a245b837041d01fcf273d74a8b45db81ea14a/src/TurbulenceClosures/abstract_scalar_diffusivity_closure.jl) | Constitutive diffusivity separated from geometry/operators | `DiffusionOperator.diffusivity` and `flux` | Nonnegative scalar or n+1 face coefficient input; no closure-field interpolation or turbulence model |
| [time_step_wizard.jl](https://github.com/CliMA/Oceananigans.jl/blob/819a245b837041d01fcf273d74a8b45db81ea14a/src/Simulations/time_step_wizard.jl) | Stability-aware timestep control | `stable_dt`, `DiffusionModel.step` | Direct local conductance bound and small-cell regression; no wizard growth/min-change limits, and safety bound is never overridden by a minimum dt |
| [runge_kutta_3.jl](https://github.com/CliMA/Oceananigans.jl/blob/819a245b837041d01fcf273d74a8b45db81ea14a/src/TimeSteppers/runge_kutta_3.jl) | Staged tendencies and actual clock advancement | `DiffusionModel.step` | Deliberately uses SSPRK3, **not** upstream low-storage Wray RK3; temporal-order and stage-weighted budget tests |
| [one_dimensional_diffusion.jl](https://github.com/CliMA/Oceananigans.jl/blob/819a245b837041d01fcf273d74a8b45db81ea14a/examples/one_dimensional_diffusion.jl) | Bounded 1-D diffusion, no-flux boundaries, explicit diffusion timescale | `main.py`, facade, optional Julia comparison | End-to-end Python smoke test; optional Julia harness uses pinned API |
| [OneDimensionalCosineAdvectionDiffusion.jl](https://github.com/CliMA/Oceananigans.jl/blob/819a245b837041d01fcf273d74a8b45db81ea14a/validation/convergence_tests/src/OneDimensionalCosineAdvectionDiffusion.jl) | Analytic modal decay and grid-refinement validation | `validation/convergence.py` | Bounded zero-advection cosine cell averages on [0,1], rather than upstream periodic advection-diffusion on [0,2pi] |

## Mathematical alignment

The common structural rule is **metric-aware divergence of shared face fluxes**.
Upstream multiplies velocity/flux densities by face areas, differences those
integrated fluxes, then divides by the destination control volume. This project
uses the same conservation structure for scalar diffusion. The resulting global
mass balance follows from cancellation of internal oriented faces.

Partial-bottom handling is tied to both volume and gradient spacing. Merely
multiplying a volume by a fraction while leaving the bottom-cell centroid unchanged
would not reproduce the inspected vertical metric. The factory adjusts both. It
does not implement upstream horizontal area calculations for multi-column terrain.

## Differences requiring separate validation

- Oceananigans supports fluid velocity fields, pressure projection, multiple
  topologies, GPU/distributed kernels, turbulence closures and higher dimensions.
  None of those capabilities is implied by this implementation.
- SSPRK3 here has an explicit positivity-preserving Euler bound. Upstream's Wray
  RK3 has different stage coefficients. For the **linear autonomous regular-grid
  diffusion** comparison only, both share the third-degree order-three stability
  polynomial; this does not establish equivalence for nonlinear or coupled models.
- Sources and boundary values in this core are stationary. Time-dependent forcing
  would require correct stage times and quadrature before being added.
- No explicit tiny-cell merging/redistribution or implicit solve is supplied.
  Stability is maintained by reducing dt; a small-cell workload can be expensive.
- Formal/categorical verification is not provided by the inspected Oceananigans
  code. This project's Lean snapshot layer is independent and limited in scope.

## Validation recorded for this revision

The local Python run completed 39 tests: 38 passed, one real-Lean test
skipped because Lean was absent. Analytic refinement on 16,32,64,128 cells gave
L2 errors approximately 1.3023e-5, 3.2628e-6, 8.1613e-7, 2.0406e-7, with successive
orders 1.9969, 1.9992, 1.9998. Maximum mass drift was approximately 1.8e-15.
The reproducible command is `python validation/convergence.py`.

The direct Julia/Oceananigans comparison and real Lean check now run as dedicated
CI gates. They remain unavailable in the local Python-only environment. The
initial local results above are historical; inspect the current commit workflow
for release evidence.
Mocked subprocess tests validate status handling only and are not Lean evidence.
Python CI runs multiple Python versions; its remote results must be inspected
separately from the local validation above.
