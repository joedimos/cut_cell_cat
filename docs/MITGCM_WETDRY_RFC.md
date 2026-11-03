# RFC: horizontal cut-cell geometry alongside MITgcm pkg/wetdry

Status: **experimental design; no merged upstream implementation**. Updated 2026-11-03.  
Upstream discussion target: [MITgcm/MITgcm](https://github.com/MITgcm/MITgcm/issues) · related open [PR #1048, pkg/wetdry](https://github.com/MITgcm/MITgcm/pull/1048).  
Reference prototype: [mitgcm-cut-cell-bridge at f8884f2](https://github.com/joedimos/cut_cell_cat/commit/f8884f263232597ce223ec9c925f1bcc5773183a).

## Scope and non-claims

The existing experimental `ccMode=3` feeds fixed horizontal cut geometry into a restricted stock, three-layer `rStar` MITgcm calculation. It does **not** activate/deactivate columns in the stock multilayer model. The separate `ccMode=2` performs depth-averaged wetting/drying using a standalone shallow-water scheme; it does **not** establish compatibility with the new stock `pkg/wetdry`. Present code requires a Cartesian hydrostatic domain, a limited EOS and physics configuration, and one tile and thread per rank.

The latest upstream `pkg/wetdry` PR changes `maskW/S` from a thresholded free-surface/bed-crest rule; `UPDATE_R_STAR` or `UPDATE_SURF_DR` then applies a dynamic face mask to `hFacW/S`. Its `WETDRY_LIMIT_OUTFLOW` routine uses the physical `rA` and `dyG/dxG` metrics. Horizontal cut fractions are **not** supplied by that PR. Cut-cell compatibility must be validated separately against the same PR revision; it should not be inferred from our tests against pinned `checkpoint69s`.

## Geometric contract

A cut wall is vertical in the first implementation: it removes a fraction of the horizontal footprint and a fraction of the **shared horizontal face**, without modifying vertical partial-cell definitions.

| Quantity | Static source | Proposed geometry-effective value |
|---|---|---|
| Area | `RAC(i,j)` | `RAC(i,j) * lambdaC(i,j)` |
| W cross-section width | `DYG(i,j)` | `DYG(i,j) * alphaW(i,j)` |
| S cross-section width | `DXG(i,j)` | `DXG(i,j) * alphaS(i,j)` |
| Vertical layer fraction | `hFacC/W/S(i,j,k)` | Keep vertical meaning; combine with dynamic `rStar`/wetdry masks at use sites |
| Gradient spacing | `DXC`, `DYC` | Keep original center-to-center metric |

Requirements:

* `0 <= lambdaC, alphaW, alphaS <= 1`, all finite, no open internal face incident to a zero-area cell, and a single shared aperture value per oriented face.
* Each internal face has one signed total water transfer `F_f`, entering the tail and head with opposite signs. All layerwise tracer transfers use exactly this geometric opening once; *never* fold both `alpha` and an already effective width into the same flux again.
* The static cut-face mask and the `wdMaskW/S` dynamic face-open mask remain **distinct**. Reopening a wetdry face cannot reopen a permanently closed embedded wall.
* Positive arbitrarily small `lambdaC` is not sufficient for usable explicit time stepping: either impose documented geometric admissibility/merge thresholds or design small-cell-safe redistribution and validate it before general use.

Define the physical layer face section, *before* applying its velocity, as

```text
A_face[k] = L_native * alpha_static * hFac_vertical[k] * DRF[k] * thickness_factor[k]
A_face_effective[k] = A_face[k] * wdMask_dynamic
```

where the `thickness_factor` is the selected native nonlinear free-surface/rStar factor and `wdMask_dynamic` is 0 or 1. If the native `hFacW/S` has **already** been dynamically masked at this call site, do not multiply by `wdMask` again. The interface must declare whether an input is raw or effective.

Planform storage `A_cell = RAC * lambdaC`. Under a film-depth constraint, `V_available = A_cell * max(0, H + eta - wdMinDepth)`. In `WETDRY_LIMIT_OUTFLOW`, both the numerator (available storage) **and** the corresponding face transports must use effective cut geometry; otherwise a small cut cell can violate its depth floor. Test dynamic masks both at pressure coefficient assembly and after a face closes/opens.

## Conservation and pressure operator

Let `B` be oriented cell-face incidence (-1 tail, +1 head), `M=diag(A_cell)`, `S=diag(L_f H_f)`, and `D=diag(d_f)` for positive retained wet cells and open faces. A fixed-geometry backward-Euler implicit barotropic stage has

```text
M(eta^{n+1}-eta^n) = dt B S u^{n+1}
D(u^{n+1}-u^n)     = -g dt B^T eta^{n+1}
[M + g dt^2 B S D^{-1} B^T] eta^{n+1} = M eta^n + dt B S u^n.
```

Because `1^T B=0` for closed internal faces, shared transports cancel in global volume, and the linear system is symmetric positive definite on positive-storage wet cells. These are useful consistency checks **for fixed geometry only**. They do not establish well-balancing, monotonicity, stability, or conservation across new wet-mask transitions, which require separate proofs/measurements.

## Integration seams identified in #1048

1. `WETDRY_UPDATE_MASKS`: calculate the threshold mask from the native bed/free surface; combine it with permanent cut-face connectivity without erasing either component. The program must not use `lambdaC>0` alone as a face-connectivity test.
2. `UPDATE_R_STAR` / `UPDATE_SURF_DR`: establish exactly one multiplication by each static/dynamic geometric factor, with consistent wet/dry inverse thickness fields and fresh operator coefficients after mask transitions.
3. `WETDRY_LIMIT_OUTFLOW`: limit using `RAC*lambdaC` and cut-effective `dyG/dxG`; handle the case where one face is outgoing from one column and incoming to its neighbor, including MPI seam faces.
4. `INTEGR_CONTINUITY` and temperature/salt transport: audit water and scalar content against the same oriented face ledger and consistent dynamic layer volumes. Make sure surface forcing volume scales with cut area.
5. `CG2D` solve: invalidate/reassemble affected coefficients as required for dynamic face changes, and test positive definiteness/conditioning after wet/dry transitions.
6. MPI/pickups: store immutable planform/face geometry with a versioned checksum; verify consistency on restart across a different process decomposition; exchange dynamic masks and fluxes with single ownership. Native wetdry pickup and cut geometry must be checked against the same step.
7. Preserve **feature-disabled** baseline; reject unsupported configurations explicitly rather than claiming full physics-package compatibility.

These are proposed interface requirements. No code in the published experimental branch currently implements the combined `pkg/cutcell + pkg/wetdry` semantics.

## Staged contributions

**Phase A — fixed geometry upstream:** Create an opt-in `pkg/cutcell` or equally isolated geometry interface with static area and aperture fields and a single native/geometry coupling contract. Add a closed oblique basin case, tracer budgets and a no-feature regression. No moving wet mask.

**Phase B — nonlinear multilayer co-existence:** Rebase the package against the revision of PR #1048 accepted by maintainers; update native `rStar` and dynamic `CG2D` coefficients on face transitions; add conservative film/outflow checks with tiny cut fractions and three-layer tracer content.

**Phase C — validation and generalization:** Reproduce lake at rest over slope, oblique Ritter dam break, tidal shoreline run-up/dry-out, MPI seam transfer, 1->4 and 4->1 process-grid restarts, area-weighted heat/salt budgets, and systematic mesh convergence; expand beyond the restricted physics matrix only after separate tests. Explicitly test failure paths for mismatched geometry and invalid configurations.

### Minimal acceptance matrix

| Test | Baseline requirement | Cut-cell addition |
|---|---|---|
| Feature disabled | Native MITgcm diagnostics identical | No change to base integration |
| Lake at rest | Native no-flow | No mass flux through oblique wall; no spurious wall current |
| Linear impulse | Stock implicit operator | Cell-volume balance within scale-aware roundoff |
| Fixed three-layer rStar | Native T/S, W, Eta | Effective volumes, salt/heat budgets, stratified initial conditions |
| Dynamic wet/dry | PR #1048 film depth | Small-cut film positivity, no double-scaled flux |
| MPI | One and four ranks | Shared aperture and face-owner consistency |
| Restart | Native pickup | Geometry hash and decomposition-independent consistency |
| Refinement | Defined reference | Errors/convergence and timestep condition across cut sizes |

### Suggested upstream maintainer question

Would a **fixed-horizontal-geometry-only**, opt-in `pkg/cutcell` contribution be acceptable as a first step, designed to compose later with PR #1048's `pkg/wetdry`? In particular, should the effective `rA/dyG/dxG` be established centrally before coefficient assembly, or should immutable apertures be carried as independent arrays and used explicitly by the stock pressure/continuity/transport paths to avoid interaction with other packages?

### References

- [Yaegashi, arXiv:2610.16148 (2026)](https://arxiv.org/abs/2610.16148), ghost cells, shared finite-volume cut flux and wetting/drying in shallow water.
- [MITgcm PR #1048](https://github.com/MITgcm/MITgcm/pull/1048), proposed native wet/dry masks and positivity limiter.
- [Existing prototype description](./MITGCM_CUT_CELLS.md), including limited numerical experiments and reproduction instructions.
