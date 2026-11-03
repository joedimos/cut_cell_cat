"""Instrument or audit native MITgcm wet/dry face transports at tile seams."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from benchmark_wetdry import CRITICAL, DT, FLOOR, WETDRY_PIN
from compare_mitgcm import CASE, build_model, run_logged
from cutcell.research.wetdry_audit import audit

NATIVE = Path(__file__).resolve().parent / "mitgcm_wetdry_audit"
ANCHOR = "C-    Integrate vertically the Horizontal Divergence"
STEPS = 50
LAYOUT = {"nx": 8, "ny": 8, "snx": 4, "sny": 4, "levels": 1, "first": 1, "last": STEPS}
BUDGET_TOL = 1e-8
FAULT_VOLUME = .0625


def check_source(source):
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                     "--untracked-files=no"], text=True)
    if head != WETDRY_PIN or dirty:
        raise ValueError(f"expected a clean MITgcm #1048 checkout at {WETDRY_PIN}")


def instrument(source, mods, *, test_fault=False):
    """Write only native source overrides. Never edit the upstream checkout."""
    original = (source / "model/src/integr_continuity.F").read_text()
    if original.count(ANCHOR) != 1:
        raise ValueError("cannot uniquely locate the native continuity divergence")
    hook = """#ifdef ALLOW_WETDRY
        IF ( useWETDRY .AND. myIter.GT.nIter0 ) THEN
"""
    if test_fault:
        hook += """         CALL WD_AUDIT_TEST_FAULT( bi, bj, k, vTrans, myIter )
"""
    hook += """         CALL WD_AUDIT_SEAM( bi, bj, k, uTrans, vTrans,
     &                       myIter, myThid )
        ENDIF
#endif
"""
    mods.mkdir(parents=True, exist_ok=True)
    names = ["integr_continuity.F", "wd_audit_seam.F"]
    if test_fault:
        names.append("wd_audit_test_fault.F")
    existing = [p.name for p in mods.iterdir() if p.name in (*names, "wd_audit_test_fault.F")]
    if existing:
        raise ValueError(f"existing audit overrides in {mods}: {', '.join(sorted(existing))}")
    (mods / names[0]).write_text(original.replace(ANCHOR, hook + "\n" + ANCHOR))
    for name in names[1:]:
        shutil.copyfile(NATIVE / name, mods / name)


def _build(source, work, mode, jobs, ranks):
    mods = work / "mods"
    mods.mkdir(parents=True)
    size = (CASE / "SIZE.h").read_text().replace("sNx=8, sNy=7", "sNx=4, sNy=4")
    size = size.replace("nSx=1, nSy=1, nPx=1, nPy=1", f"nSx=2, nSy={2//ranks}, nPx=1, nPy={ranks}")
    (mods / "SIZE.h").write_text(size)
    cpp = (source / "model/inc/CPP_OPTIONS.h").read_text()
    if cpp.count("#undef NONLIN_FRSURF") != 1:
        raise ValueError("unexpected NONLIN_FRSURF default")
    (mods / "CPP_OPTIONS.h").write_text(cpp.replace("#undef NONLIN_FRSURF", "#define NONLIN_FRSURF"))
    (mods / "packages.conf").write_text("gfd\nwetdry\n")
    if mode != "baseline":
        instrument(source, mods, test_fault=mode == "fault")
    return build_model(source, work / "build", mods, jobs, mpi=ranks == 2)


def _case(directory):
    directory.mkdir(parents=True)
    data = (CASE / "data").read_text().replace(
        " saltStepping=.FALSE.,", """ saltStepping=.FALSE.,
 nonlinFreeSurf=4, select_rStar=2, hFacInf=0.20,
 implicitDiffusion=.TRUE., vectorInvariantMomentum=.TRUE.,
 useSingleCpuIO=.TRUE.,""")
    data = (data.replace("nTimeSteps=5", f"nTimeSteps={STEPS}")
            .replace("deltaT=15.", f"deltaT={DT}.")
            .replace("dumpFreq=15.", f"dumpFreq={DT}.")
            .replace("monitorFreq=15.", f"monitorFreq={DT}.")
            .replace("delX=90., 80., 100., 95., 110., 85., 120., 105.", "delX=8*10.,")
            .replace("delY=120., 100., 105., 85., 90., 110., 95.", "delY=8*10.,")
            .replace("delR=16.", "delR=1."))
    (directory / "data").write_text(data)
    (directory / "data.pkg").write_text(" &PACKAGES\n useWETDRY=.TRUE.,\n &\n")
    (directory / "data.wetdry").write_text(
        f" &WETDRY_PARM01\n wdMinDepth={FLOOR}, wdCritDepth={CRITICAL}, wdMonFreq={DT}.,\n &\n")
    shutil.copyfile(CASE / "eedata", directory / "eedata")
    bathy = np.zeros((8, 8))
    bathy[1:-1, 1:-1] = [-.60, -.60, -.50, -.25, -.12, -.08]
    eta = np.zeros_like(bathy)
    eta[1:-1, 1:-1] = [.12, .12, 0, 0, -.03, -.02]
    for name, array in (("bathy", bathy), ("eta", eta)):
        array.astype(">f8").tofile(directory / f"{name}.bin")


def _run(executable, directory, rdmds, ranks, launcher):
    _case(directory)
    command = [str(executable)]
    if ranks > 1:
        command = [launcher, "-n", str(ranks), *command]
    run_logged(command, directory, "run.log")
    logs = sorted(directory.glob("STDOUT.*")) or [directory / "run.log"]
    if len(logs) != ranks:
        raise RuntimeError(f"expected {ranks} native stdout logs in {directory}")
    master = logs[0].read_text(errors="replace")
    if "PROGRAM MAIN: Execution ended Normally" not in master:
        raise RuntimeError(f"MITgcm did not finish normally in {directory}")
    budget = re.findall(r"%WETDRY_MON: cumInflow,budgetErr,budgetErr/vol0=\s*(\S+)\s+(\S+)\s+(\S+)", master)
    flips = re.findall(r"nFlips=\s*(\d+)", master)
    if len(budget) != STEPS + 1 or len(flips) != STEPS + 1:
        raise RuntimeError(f"incomplete native wetdry monitor in {directory}")
    budget_error = np.array([float(row[1].replace("D", "E")) for row in budget])
    area = rdmds(str(directory / "RAC"))
    rest = rdmds(str(directory / "Depth"))
    eta = np.stack([rdmds(str(directory / "Eta"), i) for i in range(STEPS + 1)])
    wet = rest > 0
    depth = (eta + rest)[:, wet]
    volume = depth @ area[wet]
    if not (np.isfinite(budget_error).all() and np.isfinite(depth).all()):
        raise RuntimeError(f"nonfinite native outputs in {directory}")
    return logs, eta, {
        "native_max_budget_error_m3": float(np.max(np.abs(budget_error))),
        "snapshot_max_global_volume_drift_m3": float(np.max(np.abs(volume - volume[0]))),
        "min_depth_m": float(depth.min()), "peak_face_flips": max(map(int, flips)),
    }


def benchmark(source, work, *, jobs, ranks, launcher):
    check_source(source)
    work.mkdir(parents=True)
    sys.path.insert(0, str(source / "utils/python/MITgcmutils"))
    from MITgcmutils import rdmds

    modes = ("baseline", "clean", "fault")
    with ThreadPoolExecutor(max_workers=len(modes)) as pool:
        models = {m: pool.submit(_build, source, work / m, m, jobs, ranks) for m in modes}
    report = {"mitgcm_commit": WETDRY_PIN, "ranks": ranks, "layout": LAYOUT, "dt_s": DT,
              "aggregate_tolerance_m3": BUDGET_TOL, "runs": {}}
    eta = {}
    for mode, model in models.items():
        logs, eta[mode], metrics = _run(model.result(), work / mode / "run", rdmds, ranks, launcher)
        report["runs"][mode] = metrics
        if mode != "baseline":
            metrics["audit"] = audit(logs, **LAYOUT)
        if (max(metrics["native_max_budget_error_m3"], metrics["snapshot_max_global_volume_drift_m3"])
                > BUDGET_TOL or metrics["min_depth_m"] < FLOOR - 1e-12 or not metrics["peak_face_flips"]):
            raise AssertionError(f"expected native budget, film and changing faces in {mode}: {metrics}")
    np.testing.assert_array_equal(eta["baseline"].view(np.uint64), eta["clean"].view(np.uint64))
    report["clean_eta_bitwise_equal_to_baseline"] = True
    if not report["runs"]["clean"]["audit"]["ok"]:
        raise AssertionError(f"clean traces failed: {report['runs']['clean']['audit']}")
    fault = report["runs"]["fault"]["audit"]
    first = fault["first_issue"] or {}
    if (fault["ok"] or fault["bad_face_steps"] != 4
            or fault["counts"] != {"closed-face-transport": 4, "outward-transport-mismatch": 4}
            or (first.get("iteration"), first.get("axis"), first.get("i"), first.get("j"), first.get("level"))
            != (1, "Y", 6, 5, 1)
            or first["signed_defect_m3"] != FAULT_VOLUME
            or (ranks == 2 and first["negative_side"][0]["rank"] == first["positive_side"][0]["rank"])):
        raise AssertionError(f"failed to recover the native closed-face fault: {fault}")
    if max(fault["max_signed_step_sum_m3"], fault["max_signed_face_window_sum_m3"]) > BUDGET_TOL:
        raise AssertionError(f"fault is visible to the aggregate comparisons: {fault}")
    # Demonstrate that the test transport reached the model's state update.
    row, col = first["j"] - 2, first["i"] - 1
    delta = 100. * (eta["fault"][1, row, col] - eta["clean"][1, row, col])
    np.testing.assert_allclose(delta, -FAULT_VOLUME, atol=1e-12, rtol=0)
    report["first_fault_cell_volume_change_m3"] = float(delta)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    overlay = commands.add_parser("instrument", help="write read-only Fortran build overrides")
    overlay.add_argument("source", type=Path)
    overlay.add_argument("mods", type=Path)
    trace = commands.add_parser("audit", help="check existing native stdout logs, exit 1 on a defect")
    trace.add_argument("logs", type=Path, nargs="+")
    trace.add_argument("--grid", type=int, nargs=2, required=True, metavar=("NX", "NY"))
    trace.add_argument("--tile", type=int, nargs=2, required=True, metavar=("SNX", "SNY"))
    trace.add_argument("--levels", type=int, required=True)
    trace.add_argument("--steps", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    trace.add_argument("--atol", type=float, default=1e-12)
    trace.add_argument("--rtol", type=float, default=1e-12)
    bench = commands.add_parser("benchmark", help="build and run baseline, read-only, and fault cases")
    bench.add_argument("source", type=Path)
    bench.add_argument("workdir", type=Path)
    bench.add_argument("--jobs", type=int, default=2, help="make jobs per build")
    bench.add_argument("--ranks", type=int, choices=(1, 2), default=1)
    bench.add_argument("--launcher", default="mpiexec")
    args = parser.parse_args()
    if args.command == "instrument":
        check_source(args.source)
        instrument(args.source, args.mods)
        return 0
    if args.command == "audit":
        try:
            report = audit(args.logs, nx=args.grid[0], ny=args.grid[1], snx=args.tile[0], sny=args.tile[1],
                           levels=args.levels, first=args.steps[0], last=args.steps[1],
                           atol=args.atol, rtol=args.rtol)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    else:
        report = benchmark(args.source.resolve(), args.workdir.resolve(), jobs=args.jobs,
                           ranks=args.ranks, launcher=args.launcher)
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.command == "benchmark":
        (args.workdir / "report.json").write_text(result)
    print(result, end="")
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
