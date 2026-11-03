# Nonlinear subgrid capacities: novelty gate and reproducible reference

**Status: mathematical reference experiment, NOT a validated novel method.**  
Branch: `research/nonlinear-cutcell-capacity` (based on `mitgcm-cut-cell-bridge`).  
Last reviewed: 2026-11-03.

## Prior-art review (explicitly limits originality)

1. Begnudelli & Sanders (2006), [unstructured finite-volume shallow-water/scalar wetting and drying](https://doi.org/10.1061/(ASCE)0733-9429(2006)132:4(371)), introduced volume/free-surface consistency in partially submerged cells.
2. Casulli (2009), [high-resolution wetting and drying](https://doi.org/10.1002/fld.1896), analyzed a mildly nonlinear free-surface system and its mass balance through dry/wet transitions, including existence/uniqueness and an iterative scheme.
3. Adcroft (2013), [porous barriers and thin-wall topography](https://doi.org/10.1016/j.ocemod.2013.03.002), already considers cut cells, subgrid-scale thin walls, and elevation-dependent transport openings in ocean circulation models.
4. Brus et al. (2026), [subgrid coastal inundation in MPAS-Ocean / E3SM](https://doi.org/10.1029/2026MS005842), already use cell and edge wet-fraction integrals and subgrid hypsometry in a C-grid ocean model. **Their presented experiments use a single-layer barotropic configuration**, not stock MITgcm multilayer `rStar`.
5. Ersing, Goldberg & Winters (2025), [entropy-stable hydrostatic reconstructions for one- and multilayer shallow water](https://doi.org/10.1016/j.jcp.2025.113802), already combines positivity, well-balancing and entropy stability for wet/dry interfaces.
6. Bhat et al. (2024), [spatiotemporally varying multilayer ratios near wet/dry fronts](https://doi.org/10.1002/fld.5249), explicitly treats multilayer transitions.
7. Candy (2017), [implicit baroclinic/non-hydrostatic wetting and drying](https://doi.org/10.1016/j.advwatres.2017.02.004), already addresses three-dimensional, mass-conservative implicit inundation with baroclinic physics and conditioning.
8. [MOM6 model capabilities](https://www.gfdl.noaa.gov/mom-ocean-model/) include ALE vertical remapping and conservative wetting/drying via vanishing layers; 3-D layer remapping cannot be treated as a missing general idea.
9. [MITgcm nonlinear-free-surface documentation](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html) describes native `rStar` and conservation-aware transport.
10. MITgcm [PR #1048](https://github.com/MITgcm/MITgcm/pull/1048) proposes native nonlinear wetting/drying with per-column thin-film constraints, changing faces and outgoing flux limitation (still under review when inspected).

Therefore **exact planform capacity, elevation-dependent edge opening, conservative wetting/drying, well-balanced multilayer wet/dry reconstruction, and baroclinic implicit wetting/drying are NOT new results**. Neither does implementing a convex implicit equation automatically yield a new theorem; these belong to established monotone free-surface / mixed finite-volume machinery.

## Exact reference capacity

For an immutable convex, counter-clockwise horizontal polygon `P_i` and an affine bed `b_i(x,y)=a x+c y+d`, define for sea level `eta`:

```text
V_i(eta) = integral_Pi max(eta - b_i(x,y), 0) dA
A_i(eta) = measure({(x,y) in P_i | b_i(x,y) < eta})
S_f(eta) = integral_face max(eta - b_i(x,y), 0) ds
```

`V_i` is continuous, nondecreasing and differentiable away from changes in wet polygon topology; the one-sided derivative is `A_i` there. A linear bed gives exact piecewise-polynomial capacity without tabulated numerical quadrature. The implementation polygon-clips the wet domain, integrates `eta-b` via its centroid and integrates potential via triangle second moments.

For the unit square with bed `b=x` and `0 <= eta <=1`:

```text
A(eta) = eta
V(eta) = eta^2 / 2
```

For the triangle `x>=0, y>=0, x+y<=1` with bed `b=x+y` and `0<=eta<=1`:

```text
A(eta) = eta^2 / 2
V(eta) = eta^3 / 6
```

These are **verification identities, not new mathematics**. A constant planform fraction `lambda_i` cannot fit either shoreline volume at more than one elevation because `dV/deta=A(eta)` is not constant.

## Frozen-conductance implicit reference

Let `B` be the oriented cell/face incidence (-1 tail, +1 head), `K=diag(k_f)` a fixed nonnegative symmetric face conductance, `g>0`, and `F^n` the signed *volume flux*, not velocity. Solve

```text
V(eta^{n+1}) - V(eta^n) = dt B F^{n+1}
F^{n+1} = F^n - g dt K B^T eta^{n+1}

R(eta^{n+1}) := V(eta^{n+1}) + g dt^2 B K B^T eta^{n+1}
                - V(eta^n) - dt B F^n = 0.
```

A damped Newton iteration uses

```text
DR(eta) = diag(A_i(eta_i)) + g dt^2 B K B^T.
```

If each connected positive-conductance component contains at least one cell with positive wet capacity derivative, the Jacobian is positive definite. A component fully dry and disconnected from wet storage is a singular case and must be rejected or handled by a mathematically specified dry-state convention. No claim is made for nonlinear-momentum coupling, changing `K` during the step, or conservative boundary sources.

### Mass and conditional energy identities

Since `1^T B=0` on a closed cell graph, the discrete update conserves total physical volume `sum_i V_i`.

Define the hydrostatic potential relative to a fixed vertical datum:

```text
U_i(eta) = (g/2) * integral_(P_i and b<eta) (eta^2 - b_i(x,y)^2) dA.
```

Its derivative with respect to volume is `dU_i/dV_i=g eta` wherever `A_i>0`; as a function of `V_i`, the potential is convex. With positive fixed `K` and the quadratic flux energy

```text
E = sum_i U_i(eta_i) + (1/2) F^T K^{-1} F,
```

the backward-Euler update satisfies

```text
E^{n+1}-E^n <= -(1/2) ||F^{n+1}-F^n||^2_(K^{-1}) <= 0,
```

provided the step solves the implicit equations and the potential is well-defined on the examined storage states. This follows from convexity of `U(V)` and the shared incidence transpose; it is an established-style convexity argument, **not asserted to be original or applicable to the moving-mask full ocean solver**. Zero-conductance faces require zero flux and are excluded from the inverse quadratic form.

## Multilayer bookkeeping, not yet a stratified model

Let `V_{ik}` be exact water volume between prescribed fixed z-interfaces and `H_{ik}` a signed *integrated horizontal inflow* during a timestep. For upward internal transfers `G_{i,k+1/2}`,

```text
Delta V_ik = H_ik + G_i,k-1/2 - G_i,k+1/2,
G_i,bottom = G_i,top = 0.
```

The internal exchanges are uniquely determined by cumulative deficits **only if** `sum_k Delta V_ik = sum_k H_ik`. `required_vertical_exchange` checks precisely this consistency. It **does not** solve vertical advection, conservative temperature/salinity remap, `rStar` layer movement, pressure-gradient errors, density stability, or newly active column momentum.

## Novelty hypotheses to test (NOT proven)

**H0 (existing methods suffice).** The geometry/implicit-pressure idea is already covered by the works above, and the only additional value is an MITgcm port. In this case, no fundamental research novelty is claimed.

**H1 (conditional candidate).** It is possible to construct a *fully coupled three-dimensional, stratified, dynamic-mask* scheme that simultaneously has:
- nonnegative and properly wetted **every-layer** volumes and tracer concentrations;
- physical planform/face storage through arbitrary stationary embedded boundaries and time-varying shorelines;
- conservative layer, heat and salt inventories under `rStar` and dynamic masks, including a consistent newly wet layer remap;
- a proved discrete mechanical/available-potential energy or entropy inequality in a stated regime;
- robustness to small cut fractions without an unbounded timestep penalty or singular pressure system;
- MPI seam and restart invariance with feature-disabled native parity;
- demonstrable advantage over **Adcroft porous barriers, Brus et al. subgrid MPAS-Ocean, Candy's implicit baroclinic method, MOM6 ALE/vanishing-layer treatment, and MITgcm PR #1048** on comparable 3-D benchmarks.

These criteria should be checked against further literature (including 3-D schemes not found in this initial review) before any novelty claim. Merely reproducing `V(eta)` or `A(eta)`, or demonstrating budget cancellation, **does not pass the gate**.

## Current executable artifact

- `cutcell/research/subgrid_capacity.py`: exact polygon capacity, wetted area, face section and potential; implicit *frozen-conductance* pressure reference; vertical exchange accounting.
- `tests/test_subgrid_capacity.py`: analytical 2-D and triangle identities, potential derivative, nonlinear conservation/energy, zero-face checks, and layer-budget consistency.
- Execution: `python -m unittest tests.test_subgrid_capacity -v` or `python -m unittest discover -s tests -p test_subgrid_capacity.py -v`.

These are component tests, not 3-D MITgcm validation. The exact geometry algebra is only a baseline; its numerical benefit must be compared to existing methods before investment in native integration.

### Stop / go rule

**Current decision: NO GO on asserting scientific novelty or posting a broad upstream RFC.** The code provides a verified-or-falsifiable reference component. Proceed to a full 3-D research implementation *only* when an explicit missing theorem, counterexample, or material benchmark advantage not addressed by prior work has been identified.
