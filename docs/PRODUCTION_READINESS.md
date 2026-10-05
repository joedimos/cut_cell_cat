# Production validation contract

The supported deployment scope is stationary, bounded, one-dimensional scalar
**diffusion** with positive representable active volumes, nonnegative finite face
diffusivities, constant sources and flux/value boundary conditions. This does not
include a full ocean circulation model, arbitrary moving cut geometry, or a formal
proof of the solver. Production acceptance is limited to this declared scope.

## Required release evidence

Every change must pass the `numerical-validation` workflow:

- Python unit/invariant tests, including adversarial fault injection and a
  20,000-step forced mass-balance run, on Linux, Windows, and macOS.
- Uniform, smooth nonuniform, and partial-bottom cosine cell-average refinement.
- A real Lean 4.19.0 kernel accepts valid snapshot certificates and rejects a
  deliberately false certificate. The strict CLI must certify every requested step.
- A real Julia run installs the pinned Oceananigans source commit and compares
  the bounded regular-grid diffusion profile, coordinates, and clock to Python.

No skipped or missing integration job counts as release evidence. GitHub workflow
success is necessary for release; it is not a proof of correctness for untested
physics or every representable input. These jobs are CI gates, not repository
branch-protection settings. Repository owners must enforce required checks if
manual merges should be blocked automatically.

## Runtime safeguards

Each step checks finite state and budget values, every individual cell's integrated
mass balance, global mass balance, and cumulative mass balance since the first step.
A failed numerical check leaves state, clock, iteration, and ledger unchanged.
External mass-changing state edits after evolution begins are detected by the ledger;
construct a new model to intentionally restart with a different state.

`require_nonnegative=True` (CLI `--nonnegative`) rejects negative active
concentrations rather than clipping away mass. It does not automatically repair
unphysical sinks. Generic signed scalar fields remain supported by default.

Extremely small active volumes that underflow, or conductances that overflow, are
rejected. Closed faces do not evaluate differences across disconnected placeholder
values. The time-step controller rejects inability to advance the floating clock.

History defaults to the latest 10,000 steps (`history_limit` / `--history-limit`).
The cumulative exchange ledger and iteration count cover the whole run, independent
of retention. JSON includes retained and total counts, configuration, geometry,
actual clock, local/global/cumulative residuals, and certificate totals.

JSON serialization is finite-only and atomic: serialization or replacement failure
preserves the previous result. Atomic replacement is not a distributed transaction
or a guarantee against every network-filesystem/power-loss behavior.

`--require-lean` fails when Lean is missing or any snapshot certificate fails.
Numerical evolution precedes certificate checking; on certificate failure the model
may contain the numerically accepted step, but the CLI exits nonzero without writing
a successful result. A certificate proves a concrete rational budget inequality,
not the Python program or the PDE discretization.

## Accuracy acceptance

The analytic studies compare true cell averages, not pointwise values. Uniform and
smooth nonuniform grids should approach second order; the partial-bottom study
uses the actual numerical domain after any minimum-height adjustment. Tests do not
establish second order for arbitrary independently supplied fractions/centroids.
Such inputs describe a conservative network unless their geometry is consistent.

Before consequential use, verify grid/time refinement for the actual application,
its units, boundary conventions, forcing, and acceptable error. Do not infer
momentum/physical-energy conservation from a scalar mass budget.

## Running the gates locally

```sh
python -m pip install .
python -m unittest discover -s tests -v
python validation/convergence.py
python validation/geometry_convergence.py
lean --version
cut-cell-cat --cells 16 --steps 10 --require-lean --output lean-results.json
julia --startup-file=no validation/oceananigans_reference.jl oceananigans.csv
python validation/compare_oceananigans.py oceananigans.csv
```

Python-only environments intentionally skip the real-Lean unit test. The dedicated
CI job requires `lean --version` and the strict CLI, so this skip cannot silently
stand in for real integration validation there.
