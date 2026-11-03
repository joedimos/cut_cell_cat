# Native wet/dry transport audit at MITgcm tile seams

The audit locates the first inconsistent **iteration, face, layer and tile
pair** in the native free-surface continuity update. It targets the
[`pkg/wetdry` PR #1048](https://github.com/MITgcm/MITgcm/pull/1048)
head `892483669c0fdfe5ac1ff0d535c2ad2f2b1edf08`.

## What is measured

The read-only build override inserts a call inside
[`INTEGR_CONTINUITY`](https://github.com/dustincarrollocean/MITgcm/blob/892483669c0fdfe5ac1ff0d535c2ad2f2b1edf08/model/src/integr_continuity.F).
It observes the native `uTrans` and `vTrans` arrays after the wet/dry
limiter and immediately before their terms enter the layer's horizontal
divergence. Each tile reports its own copy independently. The auditor
also records the wet/dry face gate and multiplies the face term by the
adjacent cell's layer mask, as in the native divergence. The transport
is signed outward and integrated using `deltaTFreeSurf`. The hook runs
before the film-floor backstop.

For every face, layer and iteration, the two records must have opposite
outward volumes and equal face gates. A closed gate requires zero volume
on each side, even when two nonzero values cancel. Missing or duplicate
records make the audit fail and invalidate the aggregate comparison.
The first failure is ordered by iteration, axis, global face indices and
layer. Indices use MITgcm's one-based west/south C-grid convention.
Tile IDs are one-based and MPI ranks are zero-based. The versioned
records appear as `%WD_FACE_V1` lines in native stdout.

This is the [flux gluing condition](CATEGORICAL_CONSERVATION.md#4-cellular-flux-sheaf-and-open-system-semantics)
evaluated on native finite-precision witnesses. The pairwise tolerance
defaults to `1e-12 + 1e-12 * max(abs(q_left), abs(q_right))` m³. The
zero-flux checks use the absolute tolerance. Use the audit for interior
tile seams on a regular Cartesian, single-thread, height-coordinate
Boussinesq configuration using `exactConserv`, fully implicit divergence
and `pkg/wetdry`. Physical boundaries, periodic wraps, non-Cartesian
exchange, tracers and pickup/restart validation are outside the tested
scope.

## Native fault experiment

The harness builds the same stepped-shelf case three ways. It runs
unaltered #1048, #1048 with the read-only recorder, and a separate
test-only build. The test build modifies the **native** transport
array on the south copy of two closed Y faces, `(6, 5)` and `(7, 5)`.
It adds +0.0625 and −0.0625 m³ at iteration 1 and reverses those signs
at iteration 2. Both adjoining cells are wet in the layer. The
perturbations cancel across faces within each timestep and across
timesteps on each face.

| Check | One process, four tiles | Two MPI ranks, four tiles |
| --- | ---: | ---: |
| Clean recorder versus unaltered native Eta | Bitwise equal | Bitwise equal |
| Clean paired face/layer/iteration records | 800 passed | 800 passed |
| Faulty face/layer/iteration pairs flagged | 4 | 4 |
| Largest fault-run native `%WETDRY_MON` budget error | 2.27 × 10⁻¹³ m³ | 2.27 × 10⁻¹³ m³ |
| Largest signed seam sum within one step | 0 m³ | 0 m³ |
| Largest signed per-face, per-layer sum over the run | 0 m³ | 0 m³ |
| First failure | Iteration 1, Y(6, 5), level 1 | Iteration 1, Y(6, 5), level 1, ranks 0 and 1 |
| First failure's two outward volumes | 0.0625 and 0 m³ | 0.0625 and 0 m³ |

All three runs finish normally and stay above the 0.05 m film. The
first faulty cell loses 0.0625 m³ relative to the clean run, confirming
that the change reached the state update. The signed window sum is a
reference aggregation check, with the same loss of time resolution
as any register that retains only the final signed sum. The experiment
demonstrates a diagnostic blind spot under controlled injection.
It establishes no spontaneous MITgcm bug or priority over prior methods.

## Reproduce

Use a clean checkout at the pinned head, GNU Fortran, GNU make, NumPy
and Matplotlib. The two-rank command also requires MPI and its development
headers.

```sh
git clone https://github.com/MITgcm/MITgcm.git MITgcm-pr1048
git -C MITgcm-pr1048 fetch origin refs/pull/1048/head
git -C MITgcm-pr1048 checkout --detach 892483669c0fdfe5ac1ff0d535c2ad2f2b1edf08
python validation/audit_mitgcm_wetdry.py benchmark MITgcm-pr1048 /tmp/wd-audit-serial
python validation/audit_mitgcm_wetdry.py benchmark MITgcm-pr1048 /tmp/wd-audit-mpi --ranks 2
```

Each benchmark writes `report.json` and retains the native logs and MDS
snapshots. To audit the saved MPI logs independently, run the following
command. It returns status 1 and a JSON report for the faulty case.

```sh
python validation/audit_mitgcm_wetdry.py audit /tmp/wd-audit-mpi/fault/run/STDOUT.* \
  --grid 8 8 --tile 4 4 --levels 1 --steps 1 50
```

For a separate native run, use `instrument MITgcm-pr1048 /path/to/mods`
with the same script to write the read-only overrides into the directory
passed to MITgcm's `genmake2 -mods`. Keep the existing case options
and use one log per rank when running `audit`. The requested iteration
range must match the supplied records. Instrumentation emits a record
for each side of every covered seam, layer and continuity step, including
zero transport.
