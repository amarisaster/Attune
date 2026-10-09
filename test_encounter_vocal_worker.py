from __future__ import annotations

import unittest

from attune_encounter.vocal_worker import _compact


class VocalWorkerTests(unittest.TestCase):
    def test_compacts_attune_evidence_without_identity_or_full_note_stream(self) -> None:
        result = _compact(
            {
                "is_melodic": True,
                "voiced_fraction": 0.67,
                "pitch_range_semitones": 13,
                "notes": [
                    {"note_name": "C4", "dur_s": 0.2},
                    {"note_name": "G4", "dur_s": 1.4},
                ],
                "glides": [{"from_note": "C4", "to_note": "G4", "dur_s": 0.6, "start_s": 3.2}],
                "vibrato": [{"note_name": "G4", "rate_hz": 5.4, "extent_cents": 62}],
                "dynamics": {"dynamic_range_db": 18.2, "start_db": -31.0, "end_db": -24.0, "segments": []},
            },
            {"available": True, "label": "clear/tonal"},
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "separated_vocal_stem")
        self.assertEqual(result["held_note"], {"note": "G4", "duration_s": 1.4})
        self.assertNotIn("notes", result)
        self.assertNotIn("identity", result)
        self.assertIn("not singer identity", result["limits"])

    def test_no_melodic_line_is_explicitly_unavailable(self) -> None:
        result = _compact({"is_melodic": False}, {"available": False, "reason": "quiet"})
        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "separated_vocal_stem")


if __name__ == "__main__":
    unittest.main()
