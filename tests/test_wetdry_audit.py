"""Regressions for native seam record orientation, coverage and fault locality."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cutcell.research.wetdry_audit import audit


class NativeFaceAuditTests(unittest.TestCase):
    layout = {"nx": 2, "ny": 2, "snx": 2, "sny": 1, "levels": 1, "first": 1, "last": 2}

    @staticmethod
    def trace(*, it, i, side, q, gate=1, active=1):
        ty = 1 if side == 1 else 2
        return f"(PID.TID) %WD_FACE_V1 {it} 1 2 {i} 2 {side} 1 {ty} {ty-1} {gate} {active} {q}\n"

    def read(self, records, **tolerance):
        with TemporaryDirectory() as directory:
            log = Path(directory) / "stdout"
            log.write_text("unrelated native log\n" + "".join(records))
            return audit([log], **self.layout, **tolerance)

    def records(self, fn):
        return [self.trace(it=it, i=i, side=side, **fn(it, i, side))
                for it in (1, 2) for i in (1, 2) for side in (1, -1)]

    def test_balanced_open_faces_and_explicit_closed_zeroes(self):
        result = self.read(self.records(lambda it, i, side: {"q": side if i == 1 else 0,
                                                             "gate": int(i == 1)}))
        self.assertTrue(result["ok"])
        self.assertEqual(result["paired_face_steps"], 4)
        self.assertIsNone(result["first_issue"])

    def test_signed_cancellation_in_space_and_time(self):
        def fault(it, i, side):
            return {"q": (3-it*2) * (3-i*2) * .0625 if side == 1 else 0, "gate": 0}

        result = self.read(list(reversed(self.records(fault))))
        self.assertFalse(result["ok"])
        self.assertEqual(result["counts"], {"closed-face-transport": 4, "outward-transport-mismatch": 4})
        self.assertEqual(result["max_face_step_defect_m3"], .0625)
        self.assertEqual(result["max_signed_step_sum_m3"], 0)
        self.assertEqual(result["max_signed_face_window_sum_m3"], 0)
        first = result["first_issue"]
        self.assertEqual((first["iteration"], first["axis"], first["i"], first["j"]), (1, "Y", 1, 2))
        self.assertEqual(first["negative_side"][0]["rank"], 0)
        self.assertEqual(first["positive_side"][0]["rank"], 1)

    def test_mask_disagreement_is_visible_with_zero_flux(self):
        result = self.read(self.records(lambda it, i, side: {"q": 0, "gate": int(side == 1)}))
        self.assertEqual(result["counts"], {"mask-disagreement": 4})

    def test_gate_leak_detected_even_when_opposing_fluxes_agree(self):
        result = self.read(self.records(lambda it, i, side: {"q": side, "gate": 0}))
        self.assertEqual(result["counts"], {"closed-face-transport": 4})
        self.assertEqual(result["max_face_step_defect_m3"], 0)

    def test_missing_and_duplicate_sides_make_aggregates_unavailable(self):
        lines = self.records(lambda *_: {"q": 0})
        result = self.read(lines[1:] + [lines[1]])
        self.assertEqual(result["counts"], {"missing-side": 1, "duplicate-side": 1})
        self.assertEqual(result["paired_face_steps"], 3)
        self.assertIsNone(result["max_signed_face_window_sum_m3"])

    def test_empty_trace_cannot_pass(self):
        result = self.read([])
        self.assertFalse(result["ok"])
        self.assertEqual(result["counts"], {"missing-side": 4})

    def test_nonfinite_and_invalid_mask_are_localized(self):
        lines = self.records(lambda *_: {"q": 0})
        lines[0] = self.trace(it=1, i=1, side=1, q="NaN", gate=-1)
        result = self.read(lines)
        self.assertEqual(result["counts"], {"invalid-mask": 1, "mask-disagreement": 1, "nonfinite-transport": 1})
        self.assertEqual(result["first_issue"]["negative_side"][0]["outward_m3"], "nan")
        self.assertIsNone(result["max_face_step_defect_m3"])

    def test_tolerances_allow_roundoff_but_can_be_tightened(self):
        lines = self.records(lambda it, i, side: {"q": 100 if side == 1 else -100.00000000001})
        self.assertTrue(self.read(lines)["ok"])
        self.assertEqual(self.read(lines, atol=0, rtol=0)["counts"], {"outward-transport-mismatch": 4})

    def test_wrong_tile_is_rejected(self):
        lines = self.records(lambda *_: {"q": 0})
        lines[0] = lines[0].replace("2 1 1 1 0", "2 1 2 1 0")
        with self.assertRaisesRegex(ValueError, "unexpected tile"):
            self.read(lines)

    def test_malformed_record_is_rejected_with_line_number(self):
        with self.assertRaisesRegex(ValueError, "record in .*stdout:2"):
            self.read(["%WD_FACE_V1 1 1 2\n"])
