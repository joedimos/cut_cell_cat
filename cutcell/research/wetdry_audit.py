"""Audit native MITgcm continuity transports on Cartesian tile seams."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math

MARKER = "%WD_FACE_V1"


@dataclass(frozen=True, order=True)
class FaceStep:
    iteration: int
    axis: str
    i: int
    j: int
    level: int


@dataclass(frozen=True)
class Trace:
    tile_x: int
    tile_y: int
    rank: int
    open: int
    active: int
    outward_m3: float

    def report(self):
        out = asdict(self)
        if not math.isfinite(self.outward_m3):
            out["outward_m3"] = str(self.outward_m3)
        return out


def _seams(nx, ny, snx, sny):
    """Map each seam face to the tile on its negative and positive sides."""
    faces = {}
    for i in range(snx + 1, nx + 1, snx):
        left = (i - 1) // snx
        for j in range(1, ny + 1):
            ty = (j - 1) // sny + 1
            faces["X", i, j] = (left, ty), (left + 1, ty)
    for j in range(sny + 1, ny + 1, sny):
        south = (j - 1) // sny
        for i in range(1, nx + 1):
            tx = (i - 1) // snx + 1
            faces["Y", i, j] = (tx, south), (tx, south + 1)
    return faces


def _read_record(line, source, line_number):
    fields = line.partition(MARKER)[2].split()
    try:
        iteration, k, axis, i, j, side, tx, ty, rank, gate, active = map(int, fields[:-1])
        if axis not in (1, 2) or side not in (-1, 1) or rank < 0:
            raise ValueError("invalid axis, side, or rank")
        q = float(fields[-1].replace("D", "E"))
    except (ValueError, IndexError) as exc:
        raise ValueError(f"invalid {MARKER} record in {source}:{line_number}") from exc
    return FaceStep(iteration, "X" if axis == 1 else "Y", i, j, k), side, Trace(tx, ty, rank, gate, active, q)


def audit(logs, *, nx, ny, snx, sny, levels, first, last, atol=1e-12, rtol=1e-12):
    """Verify both outward traces per face, layer and native continuity step.

    Logs are rank-local MITgcm stdout files, supplied once each. Indices are
    one-based except rank. Only interior Cartesian seams are in scope.
    """
    if (min(nx, ny, snx, sny, levels, first) < 1 or nx % snx or ny % sny or last < first
            or not all(math.isfinite(x) and x >= 0 for x in (atol, rtol))):
        raise ValueError("invalid Cartesian layout, iteration range or tolerance")
    seams = _seams(nx, ny, snx, sny)
    if not seams:
        raise ValueError("the layout must have at least one interior tile seam")
    traces = defaultdict(lambda: defaultdict(list))
    for log in logs:
        with log.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if MARKER not in line:
                    continue
                key, side, value = _read_record(line, log, line_number)
                face = (key.axis, key.i, key.j)
                if (not first <= key.iteration <= last or not 1 <= key.level <= levels or face not in seams):
                    raise ValueError(f"unexpected face/layer/iteration {key} in {log}:{line_number}")
                tile = seams[face][0 if side == 1 else 1]
                if (value.tile_x, value.tile_y) != tile:
                    raise ValueError(f"unexpected tile for {key}, side {side} in {log}:{line_number}")
                traces[key][side].append(value)

    n_bad = checked = 0
    first_issue = None
    counts = Counter()
    step_defects = defaultdict(list)
    window_defects = defaultdict(list)
    max_atom = 0.
    complete = True
    for iteration in range(first, last + 1):
        for axis, i, j in sorted(seams):
            for k in range(1, levels + 1):
                key = FaceStep(iteration, axis, i, j, k)
                rows = traces.get(key, {})
                minus = rows.get(1, [])
                plus = rows.get(-1, [])
                issues = []
                defect = None
                if not minus or not plus:
                    issues.append("missing-side")
                if len(minus) > 1 or len(plus) > 1:
                    issues.append("duplicate-side")
                if len(minus) != 1 or len(plus) != 1:
                    complete = False
                else:
                    a, b = minus[0], plus[0]
                    checked += 1
                    if any(x.open not in (0, 1) or x.active not in (0, 1) for x in (a, b)):
                        issues.append("invalid-mask")
                    if a.open != b.open:
                        issues.append("mask-disagreement")
                    if not all(math.isfinite(x.outward_m3) for x in (a, b)):
                        issues.append("nonfinite-transport")
                        complete = False
                    else:
                        if any(x.open == 0 and abs(x.outward_m3) > atol for x in (a, b)):
                            issues.append("closed-face-transport")
                        if any(x.active == 0 and abs(x.outward_m3) > atol for x in (a, b)):
                            issues.append("inactive-cell-transport")
                        defect = a.outward_m3 + b.outward_m3
                        tol = atol + rtol * max(abs(a.outward_m3), abs(b.outward_m3))
                        if abs(defect) > tol:
                            issues.append("outward-transport-mismatch")
                        max_atom = max(max_atom, abs(defect))
                        step_defects[iteration].append(defect)
                        window_defects[axis, i, j, k].append(defect)
                if issues:
                    n_bad += 1
                    counts.update(issues)
                    if first_issue is None:
                        first_issue = {**asdict(key), "issues": issues, "signed_defect_m3": defect,
                                       "negative_side": [x.report() for x in minus],
                                       "positive_side": [x.report() for x in plus]}
    return {
        "ok": n_bad == 0,
        "expected_face_steps": len(seams) * levels * (last - first + 1),
        "paired_face_steps": checked,
        "bad_face_steps": n_bad,
        "counts": dict(counts),
        "first_issue": first_issue,
        "max_face_step_defect_m3": max_atom if complete else None,
        "max_signed_step_sum_m3": max((abs(math.fsum(q)) for q in step_defects.values()), default=0.)
        if complete else None,
        "max_signed_face_window_sum_m3": max((abs(math.fsum(q)) for q in window_defects.values()), default=0.)
        if complete else None,
    }
