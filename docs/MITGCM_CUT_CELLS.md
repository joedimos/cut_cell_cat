# MITgcm and horizontal cut-cell continuity

`cutcell.research.mitgcm` connects the project's incidence, volume-weighted transport, and finite Markov kernels to a native MITgcm verification case and an experimental oblique wall. The native comparison covers a closed, static, Cartesian, hydrostatic, one-layer basin with a linear free surface, `exactConserv`, fully implicit surface pressure and divergence, and no external forcing, Coriolis, momentum advection, or viscosity. The Python stage solves the same linear pressure–continuity subsystem directly from MITgcm's MDS grid and state files. [MITgcm checkpoint69s](https://github.com/MITgcm/MITgcm/tree/fef697a599dd2ad9918d2876d17b68764d32f871) · [MITgcm algorithm](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html)

Yaegashi's paper keeps ghost-cell momentum and replaces depth continuity with a shared, aperture-weighted flux. It uses hydrostatic reconstruction, signed sliver and pool bookkeeping, and a depth-floor redistribution in an explicit shallow-water solver. Its ghost reconstruction masks solid and dry donors and applies a common normalized stencil to surface, bed, and discharge. The bridge takes the shared-flux construction into a linear implicit C-grid pressure solve. The ghost stencil is a separate reconstruction primitive for future wall momentum and shoreline work. [Yaegashi, arXiv:2610.16148v1, §§2.2–2.7](https://arxiv.org/html/2610.16148v1)

## Geometry and source mapping

| Quantity | MITgcm source | Horizontal cut extension |
| --- | --- | --- |
| Surface storage \(M_i\) | `RAC` on wet columns | `RAC` × fluid planform fraction \(\Lambda_i\) |
| Reference column depth \(H_i\) | \(\sum_k \mathrm{DRF}_k\,\mathrm{hFacC}_{ik}\) | Same depth inside the remaining footprint |
| Open west cross-section \(S_w\) | `DYG` × \(\sum_k \mathrm{DRF}_k\,\mathrm{hFacW}_{wk}\) | Multiply once by shared horizontal aperture \(\alpha_w\) |
| Open south cross-section \(S_s\) | `DXG` × \(\sum_k \mathrm{DRF}_k\,\mathrm{hFacS}_{sk}\) | Multiply once by shared horizontal aperture \(\alpha_s\) |
| Gradient distance \(d_f\) | `DXC` west and `DYC` south | Same background-grid center spacing |

The product geometry assumes a vertical wall extruded through the water column. `half_plane_fractions` clips one Cartesian half-plane and produces one aperture per face. The intersections are computed analytically in floating point. MITgcm's `hFac` fields retain their vertical-thickness meaning. A cut portion with positive fluid area participates in storage even when the geometric center lies outside the fluid, provided the underlying MITgcm column is wet. The pressure gradient still uses the background-grid spacing, so accuracy at oblique walls needs a separate convergence study. [MITgcm vertical grid](https://mitgcm.readthedocs.io/en/latest/algorithm/vert-grid.html) · [Yaegashi, §2.6](https://arxiv.org/html/2610.16148v1)

Native wet masks and metric arrays come from `write_grid.F`. `integr_continuity.F` uses the signed divergence of depth-integrated face transports, `calc_div_ghat.F` forms the barotropic right-hand side, and `ini_cg2d.F` assembles the coefficients proportional to open cross-section over center spacing. `solve_for_pressure.F` applies the implicit pressure solve and velocity correction. [Grid output](https://github.com/MITgcm/MITgcm/blob/fef697a599dd2ad9918d2876d17b68764d32f871/model/src/write_grid.F) · [Continuity](https://github.com/MITgcm/MITgcm/blob/fef697a599dd2ad9918d2876d17b68764d32f871/model/src/integr_continuity.F) · [Pressure RHS](https://github.com/MITgcm/MITgcm/blob/fef697a599dd2ad9918d2876d17b68764d32f871/model/src/calc_div_ghat.F) · [CG2D coefficients](https://github.com/MITgcm/MITgcm/blob/fef697a599dd2ad9918d2876d17b68764d32f871/model/src/ini_cg2d.F) · [Pressure solve](https://github.com/MITgcm/MITgcm/blob/fef697a599dd2ad9918d2876d17b68764d32f871/model/src/solve_for_pressure.F)

## Conservation, pressure, and coarse grids

Use the incoming incidence \(B\) with \(-1\) at the tail and \(+1\) at the head of every eastward or northward open face. With \(M=\mathrm{diag}(M_i)\), \(S=\mathrm{diag}(S_f)\), and \(D=\mathrm{diag}(d_f)\), the fully implicit update is

\[
M(\eta^+-\eta)=\Delta t\,B S u^+,\qquad
D(u^+-u)=-g\Delta t\,B^\top\eta^+.
\]

Eliminating velocity yields \([M+g\Delta t^2 B S D^{-1} B^\top]\eta^+=M\eta+\Delta t B S u\). Its matrix is positive definite for positive wet areas, including disconnected basins and isolated wet columns. Each shared face enters continuity with opposite signs, so \(\mathbf{1}^\top B=0\) on each connected basin. Constant surface and zero velocity are an equilibrium even over variable reference depth.

The quadratic wave energy per reference density is \(E(\eta,u)=(g\eta^\top M\eta+u^\top S D u)/2\). Pair the update equations with \(g\eta^+\) and \(S u^+\), respectively, so the face work cancels. Backward Euler then gives \(E(\eta^+,u^+)-E(\eta,u)=-E(\eta^+-\eta,u^+-u)\). This is an energy estimate for the linear stage with fixed metrics. It imposes neither a positive nonlinear depth nor a tracer donor CFL bound.

The cell partition \(A\) and oriented face quotient \(Q\) satisfy the exact integer chain-map identity \(A B_f=B_c Q\). Each within-block face maps to zero. Each crossing maps to the signed edge between its two blocks, with parallel crossings summed. Hence aggregation preserves the finite-volume balance for every supplied face flux. With \(R=\mathrm{diag}(AM\mathbf{1})^{-1}AM\), the validation independently measures the defect \(R L_f-L_cR\), where \(L_f=-M^{-1}B S D^{-1}B^\top\) and the coarse stiffness is Galerkin with prolongation \(A^\top\). The tested partition conserves volume and has a nonzero pressure-operator closure defect. This extends the existing 1-D categorical construction to a 2-D C-grid graph.

## Tracers and ghost donors

`tracer_step` advects water volume and tracer content with the same face transport. The new concentration is a convex combination of retained donor content and incoming content when outgoing volume is bounded by initial donor volume. A dry cell can receive water and tracer from an externally supplied flux. The oblique case uses two tracer substeps with the solved volume flux held fixed, so their total volume change equals the pressure step's change. This does not simulate a moving shoreline.

`ghost_kernel` interprets bilinear image-point weights as a categorical distribution and conditions them on eligible wet donors. The default cutoff is 5% of the geometrically eligible fluid weight. `reflected_ghost` uses the same donor law for free surface, bed elevation, and discharge and applies the slip reflection \(I-2nn^\top\) to discharge. The null-stencil result is explicit. Geometric fallback, first-layer face-neighbor classification, and dry ghost bed elevation remain caller decisions. The ghost primitive is tested separately from the aperture-based pressure solve. [Yaegashi, §§2.3–2.5](https://arxiv.org/html/2610.16148v1)

## Reproduce the results

From the `cut_cell_cat` root with Python, NumPy, GNU Fortran, and GNU Make installed on Linux x86-64

```sh
git clone https://github.com/MITgcm/MITgcm.git ../MITgcm
git -C ../MITgcm checkout fef697a599dd2ad9918d2876d17b68764d32f871
python -m pip install .
python validation/compare_mitgcm.py ../MITgcm validation/results/mitgcm --jobs 4
```

The command builds MITgcm via `genmake2` and runs five 15-second steps in an unforced basin with nonuniform spacing, an island, and ten partial-bottom columns. It reads native 64-bit `Eta`, `U`, and `V` snapshots, checks each step against the bridge using the preceding native state, and audits the native continuity budget separately. A second native run starts from uniform surface height. The oblique wall is a separate Python experiment. Metrics are written to `validation/results/mitgcm/report.json`.

| Check | Observed result |
| --- | ---: |
| Maximum native pulse surface difference | \(7.56\times10^{-16}\,\mathrm{m}\) |
| Maximum native pulse velocity difference | \(2.16\times10^{-16}\,\mathrm{m/s}\) |
| Maximum native pulse volume drift | \(1.39\times10^{-13}\,\mathrm{m^3}\) |
| Native lake-at-rest volume drift | \(0\,\mathrm{m^3}\) |
| Smallest oblique fluid area fraction | \(5.0\times10^{-5}\) |
| Maximum oblique cell-volume budget defect | \(4.03\times10^{-14}\,\mathrm{m^3}\) |
| Oblique constant-tracer maximum difference | \(1.11\times10^{-16}\) |

## Extending to moving shorelines in MITgcm

The next testable extension is a nonlinear storage law \(V_i(\eta)=\int_{P_i\cap\mathrm{fluid}}(\eta-z_b(x,y))_+\,dA\), paired with one wetted cross-section per face and a stagewise shared flux. Validate a dry front crossing an oblique wall over a sloped bed before testing a dam break. With dynamic apertures, the pressure matrix, velocity correction, `exactConserv` continuity, and tracer fluxes need the same time-level geometry. `update_surf_dr.F` and `update_r_star.F` already update vertical thickness for nonlinear free surfaces, so a future native implementation would keep lateral geometry separately and audit when each operator reads it. Nonlinear wetting and drying, curved-wall convergence, and a native Fortran cut-cell option remain open. [MITgcm nonlinear free surface](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html) · [Yaegashi's test suite](https://arxiv.org/html/2610.16148v1#S3)
