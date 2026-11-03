# MITgcm and horizontal cut-cell continuity

`cutcell.research.mitgcm` connects the project's incidence, volume-weighted transport, and finite Markov kernels to MITgcm. The experiment-local Fortran overlay adds a fixed-geometry mode to the native implicit pressure step and a nonlinear shallow-water mode for wetting and drying. The native comparison covers a closed, static, Cartesian, hydrostatic, one-layer basin with a linear free surface, `exactConserv`, fully implicit surface pressure and divergence, and no external forcing, Coriolis, momentum advection, or viscosity. The Python stage solves the same linear pressure–continuity subsystem directly from MITgcm's MDS grid and state files. [MITgcm checkpoint69s](https://github.com/MITgcm/MITgcm/tree/fef697a599dd2ad9918d2876d17b68764d32f871) · [MITgcm algorithm](https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html)

Yaegashi's paper keeps ghost-cell momentum and replaces depth continuity with a shared, aperture-weighted flux. It uses hydrostatic reconstruction, signed sliver and pool bookkeeping, and a depth-floor redistribution in an explicit shallow-water solver. Its ghost reconstruction masks solid and dry donors and applies a common normalized stencil to surface, bed, and discharge. The fixed-geometry mode takes the shared-flux construction into MITgcm's implicit C-grid pressure solve. The nonlinear mode uses the same geometric conservation principle with hydrostatic reconstruction and a slip-reflected ghost state at each wall. The masked bilinear ghost stencil is also available as a separate primitive. [Yaegashi, arXiv:2610.16148v1, §§2.2–2.7](https://arxiv.org/html/2610.16148v1)

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

The command builds MITgcm via `genmake2` and runs five 15-second steps in an unforced basin with nonuniform spacing, an island, and ten partial-bottom columns. It reads native 64-bit `Eta`, `U`, and `V` snapshots, checks each step against the bridge using the preceding native state, and audits the native continuity budget separately. A second native run starts from uniform surface height. This first command also runs the independent Python oblique-wall audit. Metrics are written to `validation/results/mitgcm/report.json`.

| Check | Observed result |
| --- | ---: |
| Maximum native pulse surface difference | \(7.56\times10^{-16}\,\mathrm{m}\) |
| Maximum native pulse velocity difference | \(2.16\times10^{-16}\,\mathrm{m/s}\) |
| Maximum native pulse volume drift | \(1.39\times10^{-13}\,\mathrm{m^3}\) |
| Native lake-at-rest volume drift | \(0\,\mathrm{m^3}\) |
| Smallest oblique fluid area fraction | \(5.0\times10^{-5}\) |
| Maximum oblique cell-volume budget defect | \(4.03\times10^{-14}\,\mathrm{m^3}\) |
| Oblique constant-tracer maximum difference | \(1.11\times10^{-16}\) |

## Native Fortran option

The validation overlay builds against the pinned `checkpoint69s` source via `genmake2 -mods`. It injects an initialization hook after `INI_MASKS_ETC` and a dispatch hook around `FORWARD_STEP`. In the absence of `data.cutcell` it follows the normal MITgcm step. An enabled case selects one of two serial, one-tile, one-level modes on a closed Cartesian domain.

| Mode | Native integration | Advected quantities |
| --- | --- | --- |
| `ccMode=1` | Replace wet `rA` and open `dyG`/`dxG` and their reciprocals before `INI_CG2D`. Stock `CG2D`, pressure-gradient correction, and `exactConserv` use the effective geometry. `DXC`, `DYC` and the vertical `hFac` fields retain their original values. | The overlay advects passive tracer with the final native `U`/`V` volume transport. The donor bound is enforced before updating tracer. |
| `ccMode=2` | Advance a nonlinear, depth-averaged finite-volume state inside MITgcm's native timestep loop and write named MDS diagnostics. This mode has its own pressure flux and momentum update. | Water volume, horizontal discharge and passive tracer content. Dry cells can receive water. |

Supply `&CUTCELL_PARM ccMode=1, /` or `ccMode=2` in `data.cutcell`. The verifier generates both cases and the binary inputs. To prepare custom cases, write IEEE big-endian binary64 arrays in the following order with x varying fastest and no record markers.

| File | Consecutive fields |
| --- | --- |
| `cut_geometry.bin` | Planform area fraction `(nx, ny)`, west/east face apertures `(nx+1, ny)`, south/north face apertures `(nx, ny+1)` |
| `cut_initial.bin` | Bed elevation, water depth, x and y velocities, and passive tracer concentration, each `(nx, ny)` |

Mode 1 takes initial depth from native `Eta` and bathymetry and requires the cut cells and open faces to agree with the fixed native masks. Its perimeter is dry. Mode 2 takes its state from `cut_initial.bin`, closes the box boundaries, and allows at most one straight immersed wall segment per cell. Geometry is fixed in planform while water depth and reconstructed wetted face cross-section evolve. The output fields are `CCDepth` and `CCTracer` in both modes and `CCQx` and `CCQy` for nonlinear discharge. These fields describe the overlay state. For mode 1, standard `Eta`, `U`, and `V` are the native implicit solution.

### Nonlinear conservation, wetting and drying

With piecewise-constant bed, the cell's stored volume is $V_i=A_i h_i$ and tracer content is $V_i c_i$. Each face has one geometric opening and one signed hydrostatically reconstructed Rusanov volume flux. At a bed step, reconstruct the depths against the higher bed, $h_i^*=\max(0,\eta_i-\max(z_i,z_j))$, capped by the local depth for roundoff. The matching hydrostatic pressure correction and the immersed-wall pressure use the outward-normal closure of the clipped cell. The reflected ghost state reverses only normal discharge. This gives exactly zero wall mass flux and leaves the lake at rest over a discontinuous bed, including when a shoreline crosses a wall. [Audusse et al., §2](https://publications.imp.fu-berlin.de/478/) · [Yaegashi, §§2.3–2.6](https://arxiv.org/html/2610.16148v1)

For each face with Rusanov wave bound $a_f$, an outgoing mass flux is bounded by $a_f h_i^*$ times its open length. Choosing the substep from the wet polygon area divided by the sum of face wave bounds times open lengths, with Courant factor 0.4, bounds total outflow by the available donor volume. The solver updates water and tracer using the same transfers and retains water and tracer in shallow cells. A $10^{-10}\,\mathrm{m}$ denominator floor regularizes velocity at vanishing depth. Small cut fractions reduce the explicit substep. The scheme is first-order and subdivides each model clock interval using a grid-wide CFL bound.

### Native verification command

```sh
python validation/verify_native_cutcell.py ../MITgcm validation/results/mitgcm_native --jobs 4
```

The verifier builds an unforced linear cut basin with a fixed wet mask and checks stock native pressure, velocity, and continuity against the independent Python operators. It checks native tracer against the shared-flux update, verifies the opt-out run against the uncut operators, and confirms that the cut geometry changes the result. The nonlinear suite uses three resolutions in a 45-degree channel. It checks a sloped-bed lake at rest, a disturbed sloped shoreline, and a dry-bed dam break against the Ritter cell-average solution. The reference integrates depth over each clipped polygon by splitting the integral at all geometric and wavefront transitions. The verifier checks MITgcm's reported model clock, positivity, water and tracer budgets, tracer bounds and convergence. [Yaegashi's channel experiment, §3.3](https://arxiv.org/html/2610.16148v1#S3.SS3) · [SWASHES, §4.1.2](https://doi.org/10.1002/fld.3741)

| Native benchmark | Observed result |
| --- | ---: |
| Linear cut basin, maximum surface / velocity difference versus Python | $1.45\times10^{-15}$ m / $9.02\times10^{-17}$ m/s |
| Linear cut basin, maximum transport depth mismatch against native continuity | $4.88\times10^{-15}$ m |
| Sloped-bed lake at rest, maximum depth / discharge drift across three grids | $1.11\times10^{-16}$ m / $1.66\times10^{-16}$ m²/s |
| Disturbed shoreline, wetting / drying events at a 1 mm diagnostic threshold | 13 / 7 |
| Disturbed shoreline, maximum absolute water-volume drift | $5.33\times10^{-15}$ m³ |
| Dry-bed dam break, area-weighted depth error at 0.5, 0.25 and 0.125 m spacing | 0.02467 m, 0.01785 m, 0.01199 m |
| Dry-bed dam break, observed refinement orders | 0.47, 0.57 |

The overlay is an experimental barotropic option. A general upstream package, MPI and tile exchange, pickups, moving immersed walls, the stock multilayer nonlinear free-surface solver, and laboratory-benchmark reproduction are separate work. [MITgcm nonlinear free-surface formulation](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html)
