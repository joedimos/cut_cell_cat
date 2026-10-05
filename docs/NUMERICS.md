# Numerical contract

## 1. Geometry and units

A cell stores a scalar concentration c_i. Faces are ordered in increasing x.
Unit cross-sectional area is assumed. Cell volume V_i = alpha_i * delta_x_i,
with 0 <= alpha_i <= 1. Face aperture a_f lies in [0,1]. Inactive cells have V_i=0
and closed adjacent faces. Each shared face has one aperture and one diffusivity.
Geometry is stationary. Arrays are copied and marked read-only at construction.

`partial_bottom` fits the lowest wet interval and its centroid together, using
h = max(requested_wet_height, minimum_fraction * underlying_height). This matches
the upstream minimum-height policy, including its deliberate change to numerical
bathymetry. The factory closes the bottom face. Other submerged cells are solid.
Unlike upstream, an out-of-domain bottom is rejected rather than clamped.

For generic fractional cells, default centroids remain at underlying midpoints.
Users requiring geometric fidelity must supply consistent centroids and apertures.
No geometric reconstruction or second-order guarantee on arbitrary cuts is claimed.

## 2. Flux and divergence

For an internal face f joining L and R, let d_f = x_R - x_L and
G_f = a_f * kappa_f / d_f. The integrated flux is

    F_f = -G_f * (c_R - c_L)
    V_i * dc_i/dt = F_i - F_(i+1) + V_i * s_i.

Interior contributions enter neighboring cells with opposite signs. For
M = sum_i V_i c_i, telescoping gives

    dM/dt = F_left - F_right + sum_i V_i s_i.

This is an incidence-map conservation identity. It does not assert products of
neighboring scalar fluxes vanish. The descriptive category registry is not a
formal construction of a category or chain complex.

Prescribed boundary values use center-to-face distances. Prescribed fluxes are
flux densities in the **positive coordinate direction**, multiplied by aperture.
Thus positive left flux adds mass and positive right flux removes it. This API
convention is explicit; it is not asserted identical to every Oceananigans BC API.
A closed face ignores a prescribed value/flux because its area is zero.

## 3. Stability and time integration

For each active cell, R_i is the sum of adjacent internal conductances plus
Dirichlet boundary conductances. Prescribed flux boundaries contribute forcing,
not a diagonal conductance. The sufficient monotonicity bound is

    dt <= safety * min_(R_i > 0) V_i / R_i,  0 < safety <= 1.

For regular interior cells this reduces to safety * dx^2/(2*kappa).
If R_i=0 everywhere, diffusion does not limit dt. The clock advances by the
**accepted** dt, and `run_until` shortens the last step to the requested stop time.
A maximum step count prevents tiny cells causing unbounded work.

Euler: c_new = c + dt L(c).

SSPRK3:

    c1 = c + dt L(c)
    c2 = 3/4 c + 1/4 (c1 + dt L(c1))
    c_new = 1/3 c + 2/3 (c2 + dt L(c2)).

The flux integrated across a step uses weights (1/6, 1/6, 2/3) at c,c1,c2,
including Dirichlet exchange. Constant-in-time source exchange is dt sum(V s).
SSPRK3 inherits the Euler positivity bound for closed, unforced diffusion.
It is a deliberate difference from Oceananigans' low-storage Wray RK3.

## 4. Budgets and energy

Each accepted step records

    residual = M_after - M_before - boundary_exchange - source_exchange
    tolerance = atol + rtol * max(|M_before|, |M_after|, |boundary|, |source|).

Mass sums use compensated summation. A non-finite state or failed budget raises
before committing state, clock, or history. No clipping disguises mass loss.

For closed unforced diffusion, W=diag(V) and L obey W L = (W L)^T and
c^T W L c = -sum_internal G_f (c_R-c_L)^2 <= 0. Hence the semidiscrete weighted
quadratic energy E=1/2 sum(V c^2) dissipates. Under the monotone timestep bound,
Euler and its SSP convex combinations also dissipate this convex energy. Energy
is diagnostic, not a claimed conserved physical energy of fluid dynamics.

## 5. Formal evidence boundary

Numerical checks use the budget equation, independent of the stored `passed`
flag. Optional Lean input converts each floating value to its exact rational
representation and cross-multiplies the inequality into natural numbers. There
is no decimal rounding to millionths, no relaxed 0.01 threshold, and no `sorry`,
axiom, or `native_decide` in generated certificates.

A successful process must exit zero with no output/warnings. A failed or absent
process leaves `lean_proven=false`. A certificate concerns the provided budget
numbers only. It does not certify that those numbers came from the correct code,
or that the discretization is consistent. Those require separate tests/proofs.

## 6. Validation ladder

1. Input/geometry validation and tiny-cell geometry metrics.
2. Shared flux / discrete divergence identity on seeded random nonuniform grids.
3. Volume-weighted symmetry, negative semidefiniteness, constant nullspace.
4. Closed mass/bounds/energy and disconnected-component conservation.
5. Forced/Dirichlet budgets with stage-consistent quadrature.
6. Cosine cell-average refinement (second-order uniform spatial convergence).
7. Semidiscrete cosine decay (third-order temporal SSPRK3 convergence).
8. Optional pinned Oceananigans regular-grid profile comparison.
9. Optional real Lean acceptance/rejection test, separately marked when skipped.
