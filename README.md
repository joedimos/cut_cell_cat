# cut_cell_cat

A one-dimensional, conservative finite-volume scalar diffusion research framework,
with explicit cut-cell geometry, auditable mass budgets, diagnostic pattern search,
and optional Lean certificates for **concrete budget snapshots**.

The numerical architecture is referenced to **CliMA/Oceananigans.jl at
`819a245b837041d01fcf273d74a8b45db81ea14a`**. Read the
[source-by-source comparison](docs/OCEANANIGANS_REFERENCE.md),
[mathematical contract](docs/NUMERICS.md), and [migration notes](docs/MIGRATION.md), and [production validation contract](docs/PRODUCTION_READINESS.md).
This is an independent Python implementation of a limited scalar-diffusion scope.
It is not an Oceananigans port, a 3-D ocean model, or a formally verified solver.

## Install and run

Python 3.10 or newer:

```sh
python -m pip install .
cut-cell-cat --cells 64 --steps 100 --method ssprk3 --output results.json
python -m unittest discover -s tests -v
python validation/convergence.py
```

From a checkout, `python main.py` is also supported. Plotting is optional:

```sh
python -m pip install '.[plot]'
cut-cell-cat --plot simulation.png
```

Results contain geometry, all n+1 face fluxes, actual elapsed time, per-step
boundary/source exchanges, tolerances, residuals, verification status, and the
upstream reference commit. `--dt` is a maximum requested step; stability can reduce
it. `--steps` counts accepted steps, not a prescribed elapsed duration.

## Partial-bottom example

```python
import numpy as np
from cutcell import CutCellGrid, DiffusionModel

grid = CutCellGrid.partial_bottom(np.linspace(0, 1, 65), bottom=0.233,
                                  minimum_fraction=0.2)
initial = np.exp(-((grid.centers - 0.6) / 0.1)**2)
model = DiffusionModel(grid, initial, diffusivity=0.1)
model.run_until(0.02, max_dt=0.001)
print(model.time, model.mass(), model.history[-1].residual)
```

The bottom is impermeable. Below-bottom cells are inactive. Tiny bottom cells
are enlarged to the requested minimum fraction, changing the numerical bottom;
inspect returned volumes and centers. Generic fractions and face apertures are
also supported, but do not by themselves reconstruct a physical multidimensional cut.

## Architecture

| Layer | Responsibility |
|---|---|
| `cutcell/grid.py` | Cell centers, volumes, face apertures, partial-bottom geometry |
| `cutcell/operators.py` | Shared diffusive face fluxes, boundary conditions, stability bound |
| `cutcell/model.py` | Euler / SSPRK3, actual clock, stage-weighted mass budgets |
| `lean_verification.py` | Numerical budget checks and optional Lean snapshot certificates |
| `verified_simulator.py` | Compatibility facade, results export, optional plots |
| `knowledge_graph.py`, `semantic_search.py` | Bounded diagnostic observations and deterministic retrieval |
| `tests/`, `validation/` | Invariant tests, analytic convergence, optional upstream comparison |

## Verification meaning

A passing numerical budget is never counted as a Lean theorem. `--lean` requests
an installed Lean executable; missing binaries, timeouts, errors, or warnings are
reported without proof credit. A successful certificate checks an exact rational
inequality constructed from the supplied floating-point budget values. It does
**not** prove the Python implementation, the PDE, or a categorical theory.

The historical theory registry is descriptive metadata. Arbitrary products of
neighboring fluxes are not morphism composition or a conservation law.

## Validated scope and limits

- Bounded 1-D scalar diffusion; nonuniform volumes and face diffusivities.
- Zero-flux, prescribed-flux, and Dirichlet outer boundaries; stationary sources.
- Shared internal flux cancellation, weighted mass conservation, no state clipping.
- Small-cell stability control; SSPRK3 and forward Euler.
- Closed unforced diffusion preserves bounds and dissipates weighted quadratic energy
  under the stated stability condition. Sources and prescribed fluxes can remove
  enough mass to make concentrations negative; there is no artificial clipping.
- No advection, pressure projection, velocity dynamics, GPU execution, moving
  boundaries, periodic topology, or adaptive multiscale coupling yet.
- Python tests and analytic convergence run on Linux, Windows, and macOS in CI.
  Separate release jobs install and run real Julia/Oceananigans and Lean.

## Optional direct Oceananigans comparison

```sh
julia validation/oceananigans_reference.jl /tmp/oceananigans.csv
python validation/compare_oceananigans.py /tmp/oceananigans.csv
```

The Julia script installs the exact referenced commit into a temporary Julia
project and writes a bounded 1-D cosine-diffusion reference. This needs network
access and may take several minutes. The Python comparison requires the pinned
commit in the CSV header and tests the full profile and elapsed time. It runs in the dedicated CI job; the local development environment has no Julia
installation. Consult the exact commit's CI results for pass/fail evidence.

## Runtime policies

- Every step enforces local, global, and cumulative mass budgets.
- `--nonnegative` rejects negative concentrations, including forcing-induced negatives.
- `--require-lean` requires real certificates and exits nonzero on failure.
- `--history-limit 10000` bounds retained diagnostics; total step/certificate counts
  and the cumulative ledger are independent of retention.
- JSON results include full numerical configuration and are replaced atomically.

Version 0.3.0 adds these controls. Successful CI validates the documented 1-D
scope and reference cases; it is not certification for unimplemented physics.
