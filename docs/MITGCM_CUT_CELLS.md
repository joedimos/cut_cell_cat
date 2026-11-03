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

The validation overlay builds against the pinned `checkpoint69s` source via `genmake2 -mods`. It injects a geometry hook after `INI_MASKS_ETC`, an initialization hook around `INI_FIELDS`, and a dispatch hook around `FORWARD_STEP`. In the absence of `data.cutcell` it follows the normal MITgcm step. An enabled case selects one of three modes on a closed Cartesian domain. Each MPI process owns one tile and uses one thread.

| Mode | Native integration | Advected quantities |
| --- | --- | --- |
| `ccMode=1` | Replace wet `rA` and open `dyG`/`dxG` and their reciprocals before `INI_CG2D`. Stock `CG2D`, pressure-gradient correction, and `exactConserv` use the effective geometry. `DXC`, `DYC` and the vertical `hFac` fields retain their original values. | The overlay advects passive tracer with the final native `U`/`V` volume transport. The donor bound is enforced before updating tracer. |
| `ccMode=2` | Advance a nonlinear, depth-averaged finite-volume state inside MITgcm's native timestep loop. This mode has its own pressure flux and momentum update. | Water volume, horizontal discharge and passive tracer content. Dry cells can receive water. |
| `ccMode=3` | Feed effective horizontal areas and widths into the stock multilayer hydrostatic `FORWARD_STEP`, with `select_rStar=2` and `nonlinFreeSurf=4`. Stock `UPDATE_R_STAR` and `UPDATE_CG2D` update the layer thickness and pressure coefficients. | Stock layer temperature and salinity. The wet mask stays fixed while layer thickness changes. |

The call paths are explicit in `cc_driver.F`. Mode 1 applies geometry in `CC_SETUP`, enters the stock `FORWARD_STEP`, and calls `CC_TRANSFER` with the resulting `U` and `V`. Mode 2 dispatches from `CC_FORWARD_STEP` to `CC_SWE_ADVANCE`, which calls `CC_TRANSFER` once per explicit CFL substep. Mode 3 returns to the stock `FORWARD_STEP` after applying the geometry. [Driver](../validation/mitgcm_cutcell/cc_driver.F) · [Nonlinear step and shared transfer](../validation/mitgcm_cutcell/cc_shallow_water.F)

Select the mode with `&CUTCELL_PARM ccMode=1, /` (or `2` or `3`) in `data.cutcell`. The verifier generates both cases and the binary inputs. To prepare custom cases, write IEEE big-endian binary64 arrays in the following order with x varying fastest and no record markers.

| File | Consecutive fields |
| --- | --- |
| `cut_geometry.bin` | Planform area fraction `(nx, ny)`, west/east face apertures `(nx+1, ny)`, south/north face apertures `(nx, ny+1)` |
| `cut_initial.bin` | Bed elevation, water depth, x and y velocities, and passive tracer concentration, each `(nx, ny)` |

Mode 1 takes initial depth from native `Eta` and bathymetry and requires the cut cells and open faces to agree with the fixed native masks. Its perimeter is dry. Mode 2 takes its state from `cut_initial.bin`, closes the box boundaries, and allows at most one straight immersed wall segment per cell. Geometry is fixed in planform while water depth and reconstructed wetted face cross-section evolve. The output fields are `CCDepth` and `CCTracer` in both modes and `CCQx` and `CCQy` for nonlinear discharge. These fields describe the overlay state. For mode 1, standard `Eta`, `U`, and `V` are the native implicit solution. Mode 3 takes standard MITgcm `data` and initialization files and writes standard `Eta`, `U`, `V`, `W`, `T`, and `S`. It uses `cut_geometry.bin` as its only overlay input.

### Nonlinear conservation, wetting and drying

With piecewise-constant bed, the cell's stored volume is $V_i=A_i h_i$ and tracer content is $V_i c_i$. Each face has one geometric opening and one signed hydrostatically reconstructed Rusanov volume flux. At a bed step, reconstruct the depths against the higher bed, $h_i^*=\max(0,\eta_i-\max(z_i,z_j))$, capped by the local depth for roundoff. The matching hydrostatic pressure correction and the immersed-wall pressure use the outward-normal closure of the clipped cell. The reflected ghost state reverses only normal discharge. This gives exactly zero wall mass flux and leaves the lake at rest over a discontinuous bed, including when a shoreline crosses a wall. [Audusse et al., §2](https://publications.imp.fu-berlin.de/478/) · [Yaegashi, §§2.3–2.6](https://arxiv.org/html/2610.16148v1)

For each face with Rusanov wave bound $a_f$, an outgoing mass flux is bounded by $a_f h_i^*$ times its open length. Choosing the substep from the wet polygon area divided by the sum of face wave bounds times open lengths, with Courant factor 0.4, bounds total outflow by the available donor volume. The solver updates water and tracer using the same transfers and retains water and tracer in shallow cells. A $10^{-10}\,\mathrm{m}$ denominator floor regularizes velocity at vanishing depth. Small cut fractions reduce the explicit substep. The scheme is first-order and subdivides each model clock interval using a grid-wide CFL bound.

### Audit a cell from native outputs

For modes 1 and 2, set `ccWriteFluxes=.TRUE.` in `data.cutcell` to emit four extra MDS fields at the end of each clock step. The verifier enables this setting. Each field records the **signed amount transferred over the interval**, in m³ for water and m³ times concentration for tracer. For mode 2 the field sums the CFL substeps. The donor-volume check still runs on each individual substep.

| MDS field | Face and sign |
| --- | --- |
| `CCVolW`, `CCTrW` | West face of cell `(i,j)`, positive eastward |
| `CCVolS`, `CCTrS` | South face of cell `(i,j)`, positive northward |

The east face of `(i,j)` is the west face of `(i+1,j)`. The north face is the south face of `(i,j+1)`. The omitted exterior east/north faces are closed. The checker reconstructs both balances cell by cell from $A h$ and $A h c$. It checks $\Delta V=F_W-F_E+F_S-F_N$ and the same expression for tracer content. Every recorded closed face must carry exactly zero transfer. The allowed residual is 512 binary64 epsilons times the local scale formed by old and new storage and the absolute transfers, with a minimum scale of one in the stored units.

These are the actual Fortran outputs for two cells at iteration 1. The face columns use the coordinate signs above. The time step is 15 s for the linear case and 0.2 s for shoreline run-up.

| Case and one-based cell `(i,j)` | West (m³) | East (m³) | South (m³) | North (m³) | Change in stored water (m³) | Absolute residual (m³) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear `(5,3)` | −34.212160 | −77.942425 | −9.772283 | 0 | +33.957982 | $3.12\times10^{-11}$ |
| Initially dry wall-cut cell `(35,30)` | $2.179173\times10^{-4}$ | 0 | $9.524823\times10^{-5}$ | 0 | $3.131655\times10^{-4}$ | $1.08\times10^{-19}$ |

### Native verification command

```sh
python validation/verify_native_cutcell.py ../MITgcm validation/results/mitgcm_native --jobs 4
jq '.linear.cut_cell_trace, .shoreline_runup.wetting_cell_trace, .dam_break' validation/results/mitgcm_native/report.json
```

The verifier builds an unforced linear cut basin with a fixed wet mask and checks stock native pressure, velocity, and continuity against the independent Python operators. It checks each recorded native tracer transfer against the donor concentration and native water transfer. The nonlinear suite uses three resolutions in a 45-degree channel. It checks a sloped-bed lake at rest, a disturbed sloped shoreline, and a dry-bed dam break against the Ritter cell averages. The reference integrates depth and streamwise discharge over each clipped polygon by splitting the integral at geometric and wavefront transitions. It also checks the analytic reservoir volume and the integrated streamwise momentum from the upstream hydrostatic pressure. The verifier checks MITgcm's reported clock, positivity, local and global water and tracer budgets, and tracer bounds. [Yaegashi's channel experiment, §3.3](https://arxiv.org/html/2610.16148v1#S3.SS3) · [SWASHES, §4.1.2](https://doi.org/10.1002/fld.3741)

| Native benchmark | Observed result |
| --- | ---: |
| Linear cut basin, maximum surface / velocity difference versus Python | $1.45\times10^{-15}$ m / $9.02\times10^{-17}$ m/s |
| Linear cut basin, maximum transport depth mismatch against native continuity | $4.88\times10^{-15}$ m |
| Linear cut basin, maximum local water-budget residual | $8.48\times10^{-11}$ m³ |
| Sloped-bed lake at rest, maximum depth / discharge drift across three grids | $1.11\times10^{-16}$ m / $1.66\times10^{-16}$ m²/s |
| Disturbed shoreline, wetting / drying events at a 1 mm diagnostic threshold | 13 / 7 overall, 4 / 2 in wall-cut cells |
| Disturbed shoreline, maximum absolute water-volume drift | $5.33\times10^{-15}$ m³ |
| Disturbed shoreline, maximum local water-budget residual | $1.21\times10^{-16}$ m³ |

| Grid spacing (m) | Dam-break depth error (m) | Streamwise discharge error (m²/s) | Cross-channel discharge error (m²/s) |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.02467 | 0.05520 | 0.0004317 |
| 0.25 | 0.01785 | 0.03993 | 0.0001136 |
| 0.125 | 0.01199 | 0.02772 | 0.00004310 |

The table reports area-weighted mean absolute errors at 1.2 s. Observed depth orders were 0.47 and 0.57. The exact cross-channel discharge is zero. Nonzero computed cross-channel discharge measures numerical error inside the channel. Closed-wall flux is checked separately.

### Multilayer MPI and restart

The stock multilayer case uses three layers with nominal thicknesses 4, 5, and 7 m, partial bottom layers, an oblique horizontal cut, and an active linear equation of state. The overlay changes `rA`, `dyG`, and `dxG` and their reciprocals before `INI_CG2D`. Native `hFacC/W/S` still represent vertical fractions. `rStar` evolves the native layer and face thicknesses over the fixed wet mask. The native momentum correction, `exactConserv` continuity, and temperature and salinity transport run in the stock model. For the budget audit, each native layer volume is the effective cell area times the reference layer thickness times the `rStar` factor $(H+\eta)/H$. Mode 3 requires implicit surface pressure and divergence, staggered time stepping, and the Boussinesq hydrostatic Cartesian configuration. This experiment disables momentum advection, viscosity, forcing, and rotation.

For moving shorelines, each rank advances its own cells and exchanges depth, velocity, bed, and donor concentration with MITgcm's halo routines. The CFL step is the minimum across the ranks. At an MPI seam, both tiles evaluate the same oriented flux from the same two exchanged states. Each applies the transfer to its own cell, and the east or north tile writes the shared face record. Global geometry and checkpoint files require a shared run directory.

| Mode | Resume at iteration `N` |
| --- | --- |
| 1, stock single-layer pressure plus overlay tracer | Set `nIter0=N` and copy the native `pickup.N` data and metadata together with `cut_pickup.N.bin`. Use `pChkptFreq` and `ccPickupEvery` to schedule both files for the same iteration. |
| 2, depth-averaged wetting and drying | Set `nIter0=N` and copy `cut_pickup.N.bin` and the original geometry and standard MITgcm inputs. The checkpoint supplies the bed, volume, discharge and tracer content. The original `cut_initial.bin` is needed only for a cold start. |
| 3, stock multilayer `rStar` | Set `nIter0=N` and copy the native `pickup.N` data and metadata. Reuse the same bathymetry, cut geometry, grid metrics and timestep. |

Pickup iteration suffixes are ten digits, such as `0000000003`. In modes 1 and 2, set `ccPickupEvery` to a positive iteration interval to write numbered overlay pickups. The run also writes one at its final iteration. Rank zero gathers and writes the global checkpoint to a temporary stream and renames it after closing. The reader checks version, dimensions, mode, iteration, model time, timestep, gravity, and effective geometry. The format stores global arrays, so resuming can use a different MPI process grid. The stock native pickup handles the native state and timestep history in modes 1 and 3.

```sh
python validation/verify_mitgcm_parallel.py ../MITgcm validation/results/mitgcm_parallel --jobs 4
```

The parallel verifier requires GNU Fortran and MPI compiler wrappers and launcher. It compiles serial and 2×2 process-grid executables, runs uninterrupted and iteration-3 restart cases, and compares the MDS state at every common step. It also checks native volume, temperature and salinity content using the cut cell areas and `rStar` layer volumes, verifies transport across MPI seams, and rejects a mode-2 checkpoint when the geometry changes.

| Case | Serial vs. four ranks | Four-rank checkpoint resumed on one rank |
| --- | --- | --- |
| Stock three-layer `rStar`, 14 horizontal cut cells and 39 partial bottom cells | Maximum surface difference $1.35\times10^{-15}$ m. Relative volume drift $2.12\times10^{-16}$, zero measured temperature and salinity content drift. | Maximum surface difference $9.23\times10^{-16}$ m. Temperature and salinity snapshots matched exactly. |
| Stock single-layer pressure with overlay tracer | Maximum surface difference $1.18\times10^{-16}$ m and tracer concentration difference $1.11\times10^{-16}$. | Maximum surface difference $6.25\times10^{-17}$ m and tracer concentration difference $5.55\times10^{-17}$. |
| Nonlinear wetting and drying | State and face-transfer snapshots matched bit for bit. Sum of the absolute recorded seam transfers was 3.94 m³. | State and face transfers matched bit for bit. A changed cell fraction caused the pickup to be rejected. |

Stock multilayer coupling supports fixed horizontal cut geometry with moving `rStar` layers. Activation or deactivation of whole columns uses the depth-averaged shoreline solver in mode 2. A moving wet mask inside the stock multilayer `CG2D` solver, curved walls, and upstream packaging are outside the implemented configuration. [MITgcm nonlinear free-surface formulation](https://mitgcm.readthedocs.io/en/latest/algorithm/nonlinear-freesurf.html)
