# Migration from the original prototype

| Original behavior | Revised behavior |
|---|---|
| `resolution` endpoint samples on [0,1] | `resolution` cell averages with n+1 faces |
| Endpoint states frozen under a claimed Neumann condition | Zero boundary flux; endpoint cell averages evolve |
| Laplacian update clipped to [0,1] | Conservative face-flux update without clipping |
| Uniform geometry despite the cut-cell name | Explicit volumes, apertures, centroids, solids, partial bottom |
| Hidden timestep reduction without elapsed-time accounting | Accepted dt and cumulative time recorded |
| Sum of internal gradients called conservation | Before/after volume-weighted mass with boundary/source exchange |
| Neighbor flux products called categorical composition | Removed; internal incidence cancellation is documented |
| Numerical/mock success counted as a Lean theorem | Distinct numerical and actual Lean-certificate status |
| Fixed shared temporary Lean filenames | Isolated temporary directory per invocation |
| Unbounded/debug-print-heavy pattern accumulation | Bounded observations with actual times and temporal links |
| Feature coordinate meanings shifted between patterns | Fixed named feature coordinates |
| `use_multiscale=True` silently did nothing | Explicit NotImplementedError |

`VerifiedCategoricalSimulator`, `CellState`, `CutCellComplex`, `run_verified`,
`search_patterns`, `visualize`, and `save_results` remain available. Legacy
`LeanCodeGenerator` diffusion factories now use the conservative core and
subcycle for the full requested interval; the historical names confer no proof.
`verify_conservation(states, fluxes)` returns unverified with a migration reason,
because an isolated snapshot cannot establish an evolution budget. Use
`verify_budget(StepBudget)` instead.

Serialized output is schema version 2. `lean_theorems_proven` counts concrete
snapshot certificates only. `categorical_errors` remains a legacy alias for the
numerical mass residual. `final_fluxes` now includes both boundary faces (n+1).
The `lean_used` flag means at least one successful certificate, not just executable
availability. Plot panel three is mass residual versus time.

The theory registry and stock-flow/Petri factories are retained for compatibility;
they are separate exploratory tools, not Oceananigans-derived physics. Duplicate
Petri inputs now require their full token multiplicity. Search is deterministic
keyword/concept retrieval, not learned semantic embedding inference.

Future scope should add independent operator validation before advection, pressure
projection, moving geometry, or dimensional expansion. A diffusion-only test suite
cannot establish correctness of those future systems.

## Version 0.4 categorical API

Use `cutcell.category` for exact constructions and
`cutcell.categorical_numerics` for dense numerical audit diagrams. The historical
simulator names and `categorical_errors` alias are unchanged. `use_multiscale`
continues to reject unsupported adaptive coupling: a conservative chain map is
not sufficient to commute diffusion dynamics. The new `cut-cell-category` CLI
has its own report schema, separate from solver results and Lean snapshots.

Registry descriptions now state the necessary hypotheses for Petri invariants
and DPO rewriting rather than asserting unconditional token/connectivity
conservation. See [the mathematical guide](CATEGORY_THEORY.md).
