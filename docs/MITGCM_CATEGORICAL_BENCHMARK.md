# Native MITgcm stepped-shelf benchmark

The comparison uses stock MITgcm
[`fef697a`](https://github.com/MITgcm/MITgcm/commit/fef697a599dd2ad9918d2876d17b68764d32f871)
(checkpoint69s) and the head
[`8924836`](https://github.com/dustincarrollocean/MITgcm/commit/892483669c0fdfe5ac1ff0d535c2ad2f2b1edf08)
of [MITgcm PR #1048](https://github.com/MITgcm/MITgcm/pull/1048).
It compiles the upstream Fortran and runs the PR executable both with and
without `pkg/wetdry`. Package-disabled parity controls for the different
checkout bases.

## Source audit and scope

| Method | Storage and shared face rule | Consequence for the comparison |
| --- | --- | --- |
| Stock nonlinear free surface, `rStar`, `exactConserv` | Partial bottom cells, vertically rescaled layer thickness, shared C-grid transports. `calc_r_star.F` guards against excessively small column thickness | An exact volume budget is compatible with a fixed horizontal wet domain and a thin-column stop |
| #1048 enabled | Columns retain a film. `wetdry_update_masks.F` applies a strict higher-surface minus higher-bed crest threshold. The mask affects `hFacW/S` and transport. `wetdry_limit_outflow.F` scales outgoing transport using available donor volume | Active faces can change while the ambient column graph remains fixed |
| #1048 floor backstop | `wetdry_apply_floor.F` can redistribute volume from eligible neighbors and reports uncompensated floor adjustment separately | Film, volume budget, and global mass drift need separate checks |
| This branch | Exact dual-chain, GCL, overlap, and common-atom witnesses, plus a separate pre-existing planar-bed pressure reference | The checker compares recorded quantities. It does not solve the native shelf dynamics |

The native experiment uses a closed rectangular basin with 30 water columns,
one level, 10 m cells, `select_rStar=2`, `nonlinFreeSurf=4`,
`exactConserv=.TRUE.`, 100 steps at 2 s, and initial volume 1185 m³.
The package film is 0.05 m and the strict opening threshold is 0.10 m.
Initial depths are already at or above the package film in all cases.
The guard case raises `hFacInf` to 0.80. The limiter stress case instead
starts at the nominal `hFacInf` of 0.20 and imposes a −3 m/s velocity on the
five faces west of the 0.12 m shelf. There is no forcing or tracer stepping.

## Results

Linux x86-64, GNU Fortran 13.2.0, IEEE build. The numbers are maxima over
the available snapshots or per-step native monitors.

| Case | Stock and PR disabled | PR enabled minimum depth | PR enabled max volume drift | PR enabled peak face flips / limited columns |
| --- | --- | ---: | ---: | ---: |
| Wettable shelf | Both finish. All 101 Eta snapshots are bitwise equal. Stock drift 4.55 × 10⁻¹³ m³ | 0.06 m | 2.27 × 10⁻¹³ m³ | 5 / 0 |
| Raised thickness guard | Both stop at the stock `CALC_R_STAR` guard | 0.06 m | 2.27 × 10⁻¹³ m³ | 5 / 0 |
| Limiter stress | Both stop at the stock `CALC_R_STAR` guard | 0.05 m | 2.27 × 10⁻¹³ m³ | 5 / 5 |

The nominal run starts from the same Eta for all three configurations.
Its largest enabled-versus-stock cell volume difference is 1.908 m³.
Using an exact chain-map quotient that groups the six columns into three
adjacent pairs, the largest paired-block difference is 10.885 m³. The
largest summed state difference is 1.22 × 10⁻¹⁴ m³ when summed directly
from the cell differences. Final mean shoal depths are 0.13880 m (stock)
and 0.13154 m (enabled). The sign and magnitude of these differences
describe the schemes on this shelf. The case has no analytic inundation
solution for judging their relative accuracy.

The paired-state projection uses the new dual-mesh implementation. The
native outputs used here are Eta, resting depth, grid area and the wet/dry monitor.
They do not expose the accepted face transports on both sides of each
interface atom. The exact gluing and repair tests therefore exercise
manufactured traces, separately from the native snapshot experiment.
This serial barotropic benchmark provides no evidence about tracer
uniformity, 3-D dynamics, MPI seams, restart, or field accuracy of #1048.

## Reproduce

Use clean checkouts at the two pins and GNU Fortran, GNU make, NumPy and
Matplotlib. The helper enables `NONLIN_FRSURF` in its local build options
and leaves the checkouts clean.

```sh
git clone https://github.com/MITgcm/MITgcm.git MITgcm-stock
git -C MITgcm-stock checkout --detach fef697a599dd2ad9918d2876d17b68764d32f871
git clone https://github.com/MITgcm/MITgcm.git MITgcm-pr1048
git -C MITgcm-pr1048 fetch origin refs/pull/1048/head
git -C MITgcm-pr1048 checkout --detach 892483669c0fdfe5ac1ff0d535c2ad2f2b1edf08
python validation/benchmark_wetdry.py MITgcm-stock MITgcm-pr1048 /tmp/mitgcm-shelf --jobs 4
```

The script writes `report.json` and preserves native logs, parameters
and MDS snapshots in the output directory. The previous
[static MITgcm benchmark](MITGCM_CUT_CELLS.md) remains the reference for
native partial bottom cells and the separate horizontal cut-cell prototype.
