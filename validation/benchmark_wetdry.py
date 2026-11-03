"""Run one closed stepped shelf with stock MITgcm and wetdry PR #1048."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from compare_mitgcm import CASE, PIN, build_model
from cutcell.research.conservation import OrientedMesh

WETDRY_PIN = "892483669c0fdfe5ac1ff0d535c2ad2f2b1edf08"
STEPS = 100
DT = 2
FLOOR = .05
CRITICAL = .10


def build(upstream, work, wetdry, jobs):
    mods = work / "mods"
    mods.mkdir(parents=True)
    shutil.copyfile(CASE / "SIZE.h", mods / "SIZE.h")
    cpp = (upstream / "model/inc/CPP_OPTIONS.h").read_text()
    if cpp.count("#undef NONLIN_FRSURF") != 1:
        raise RuntimeError("unexpected NONLIN_FRSURF default")
    (mods / "CPP_OPTIONS.h").write_text(cpp.replace("#undef NONLIN_FRSURF", "#define NONLIN_FRSURF"))
    (mods / "packages.conf").write_text("gfd\n" + ("wetdry\n" if wetdry else ""))
    return build_model(upstream, work / "build", mods, jobs)


def run(executable, directory, rdmds, *, wetdry, hfac_inf, initial_speed):
    directory.mkdir(parents=True)
    data = (CASE / "data").read_text().replace(
        " saltStepping=.FALSE.,", f""" saltStepping=.FALSE.,
 nonlinFreeSurf=4,
 select_rStar=2,
 implicitDiffusion=.TRUE.,
 hFacInf={hfac_inf},
 vectorInvariantMomentum=.TRUE.,""")
    data = (data.replace("nTimeSteps=5", f"nTimeSteps={STEPS}")
            .replace("deltaT=15.", f"deltaT={DT}.")
            .replace("dumpFreq=15.", f"dumpFreq={DT}.")
            .replace("monitorFreq=15.", f"monitorFreq={DT}.")
            .replace("delX=90., 80., 100., 95., 110., 85., 120., 105.", "delX=8*10.,")
            .replace("delY=120., 100., 105., 85., 90., 110., 95.", "delY=7*10.,")
            .replace("delR=16.", "delR=1."))
    if initial_speed:
        data = data.replace(" pSurfInitFile='eta.bin',", " pSurfInitFile='eta.bin', uVelInitFile='u.bin',")
    (directory / "data").write_text(data)
    (directory / "data.pkg").write_text(" &PACKAGES\n useWETDRY=.TRUE.,\n &\n" if wetdry
                                        else (CASE / "data.pkg").read_text())
    if wetdry:
        (directory / "data.wetdry").write_text(
            f" &WETDRY_PARM01\n wdMinDepth={FLOOR}, wdCritDepth={CRITICAL}, wdMonFreq={DT}.,\n &\n")
    shutil.copyfile(CASE / "eedata", directory / "eedata")
    bathy = np.zeros((7, 8))
    bathy[1:-1, 1:7] = [-.60, -.60, -.50, -.25, -.12, -.08]
    eta = np.zeros_like(bathy)
    eta[1:-1, 1:7] = [.12, .12, 0, 0, 0, -.02]
    for name, array in (("bathy", bathy), ("eta", eta)):
        array.astype(">f8").tofile(directory / f"{name}.bin")
    if initial_speed:
        u = np.zeros_like(bathy)
        u[1:-1, 5] = initial_speed
        u.astype(">f8").tofile(directory / "u.bin")
    with (directory / "run.log").open("w") as log:
        result = subprocess.run([str(executable)], cwd=directory, stdout=log, stderr=subprocess.STDOUT)
    output = "\n".join(p.read_text(errors="replace") for p in sorted(directory.glob("STDOUT*")))
    errors = "\n".join(p.read_text(errors="replace") for p in sorted(directory.glob("STDERR*")))
    combined = (directory / "run.log").read_text(errors="replace") + "\n" + output + "\n" + errors
    normal = "PROGRAM MAIN: Execution ended Normally" in combined and result.returncode == 0
    if not normal:
        if "STOP in CALC_R_STAR : too SMALL" in combined and not wetdry:
            return {"status": "rstar-thickness-guard"}, None
        raise RuntimeError(f"unexpected MITgcm failure in {directory}; see run.log and STDERR")
    area = rdmds(str(directory / "RAC"))
    resting_depth = rdmds(str(directory / "Depth"))
    wet = resting_depth > 0
    np.testing.assert_array_equal(wet, bathy < 0)
    snapshots = np.stack([rdmds(str(directory / "Eta"), i) for i in range(STEPS + 1)])
    if snapshots.shape != (STEPS + 1, *bathy.shape) or not np.isfinite(snapshots).all():
        raise RuntimeError("unexpected or nonfinite Eta snapshots")
    depth = (snapshots + resting_depth)[:, wet]
    volumes = depth @ area[wet]
    metrics = {
        "status": "completed", "steps": STEPS, "min_depth_m": float(depth.min()),
        "initial_volume_m3": float(volumes[0]),
        "max_volume_drift_m3": float(np.max(np.abs(volumes - volumes[0]))),
        "final_shoal_mean_depth_m": float(np.mean((snapshots[-1] + resting_depth)[1:-1, 6])),
    }
    if wetdry:
        lines = re.findall(r"%WETDRY_MON: iter=\s*(\d+)\s+nDry=\s*(\d+)"
                           r"\s+nFlips=\s*(\d+)\s+nLimit=\s*(\d+)", combined)
        monitor = np.array(lines, dtype=int)
        if not lines or not np.array_equal(monitor[:, 0], np.arange(STEPS + 1)):
            raise RuntimeError("expected one native wetdry monitor record per iteration")
        metrics.update(peak_dry_columns=int(monitor[:, 1].max()),
                       peak_face_flips=int(monitor[:, 2].max()),
                       peak_limited_columns=int(monitor[:, 3].max()))
    return metrics, snapshots


def compare_paired_snapshots(stock, wetdry):
    """Project same-grid state differences onto three pairs of columns."""
    np.testing.assert_array_equal(stock[0], wetdry[0])
    rows, columns = stock[0, 1:-1, 1:-1].shape
    ids = np.arange(rows * columns).reshape(rows, columns)
    faces = [*zip(ids[:, :-1].flat, ids[:, 1:].flat), *zip(ids[:-1, :].flat, ids[1:, :].flat)]
    mesh = OrientedMesh(ids.size, faces)
    quotient = mesh.quotient(tuple(j // 2 for _ in range(rows) for j in range(columns)))
    delta = 100. * (wetdry - stock)[:, 1:-1, 1:-1].reshape(-1, ids.size)
    block_delta = delta @ quotient.cells.astype(float).T
    return {"max_cell_difference_m3": float(np.max(np.abs(delta))),
            "max_block_difference_m3": float(np.max(np.abs(block_delta))),
            "max_total_difference_m3": float(np.max(np.abs(delta.sum(axis=1))))}


def benchmark(stock, pr, work, jobs):
    work.mkdir(parents=True)
    sys.path.insert(0, str(stock / "utils/python/MITgcmutils"))
    from MITgcmutils import rdmds
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(build, stock, work / "stock", False, jobs)
        pr_model = build(pr, work / "pr", True, jobs)
        stock_model = pending.result()
    report = {"stock_pin": PIN, "wetdry_pin": WETDRY_PIN, "dt_s": DT, "film_m": FLOOR,
              "threshold_m": CRITICAL, "shelf": {}}
    cases = (("wettable", .20, 0.), ("near_guard", .80, 0.), ("limiter_stress", .20, -3.))
    for label, hfac_inf, speed in cases:
        case = work / label
        settings = {"hfac_inf": hfac_inf, "initial_speed": speed}
        stock_result, stock_eta = run(stock_model, case / "stock", rdmds, wetdry=False, **settings)
        off_result, off_eta = run(pr_model, case / "pr_disabled", rdmds, wetdry=False, **settings)
        wet_result, wet_eta = run(pr_model, case / "pr_enabled", rdmds, wetdry=True, **settings)
        if stock_result["status"] != off_result["status"]:
            raise AssertionError(f"disabled parity failed in {label}")
        if stock_eta is not None:
            np.testing.assert_array_equal(stock_eta.view(np.uint64), off_eta.view(np.uint64))
        if wet_result["min_depth_m"] < FLOOR - 1e-12 or wet_result["max_volume_drift_m3"] >= 1e-8:
            raise AssertionError(f"film or volume tolerance failed in {label}")
        report["shelf"][label] = {"hfac_inf": hfac_inf, "initial_face_velocity_m_s": speed,
                                  "stock": stock_result, "pr_disabled": off_result, "pr_enabled": wet_result,
                                  "disabled_eta_bitwise_equal": None if stock_eta is None else True}
        if stock_eta is not None:
            report["shelf"][label]["paired_states"] = compare_paired_snapshots(stock_eta, wet_eta)
    if report["shelf"]["wettable"]["stock"]["status"] != "completed":
        raise AssertionError("the nominal stock case must finish")
    if report["shelf"]["wettable"]["pr_enabled"]["peak_face_flips"] == 0:
        raise AssertionError("the nominal wetdry case must exercise changing face masks")
    if report["shelf"]["near_guard"]["stock"]["status"] != "rstar-thickness-guard":
        raise AssertionError("the guard case must reach the stock rStar thickness check")
    if report["shelf"]["limiter_stress"]["pr_enabled"]["peak_limited_columns"] == 0:
        raise AssertionError("the stress case must exercise the native limiter")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock", type=Path)
    parser.add_argument("pr1048", type=Path)
    parser.add_argument("workdir", type=Path, help="new directory for the native binaries and results")
    parser.add_argument("--jobs", type=int, default=4, help="build jobs per executable")
    args = parser.parse_args()
    for source, expected in ((args.stock, PIN), (args.pr1048, WETDRY_PIN)):
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            parser.error(f"{source} has HEAD {actual}; expected {expected}")
        changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                           "--untracked-files=no"], text=True)
        if changes:
            parser.error(f"{source} has changes to tracked files")
    report = benchmark(args.stock.resolve(), args.pr1048.resolve(), args.workdir.resolve(), args.jobs)
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.workdir / "report.json").write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
