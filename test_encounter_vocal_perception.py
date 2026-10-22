from __future__ import annotations

import unittest

import numpy as np

from attune_encounter.vocal_perception import analyze_vocal_perception


class VocalPerceptionTests(unittest.TestCase):
    def test_weak_nonmelodic_separation_residue_is_omitted(self) -> None:
        rate = 8_000
        time = np.arange(rate * 3) / rate
        samples = 0.001 * np.sin(2 * np.pi * 220 * time)
        result = analyze_vocal_perception(
            samples,
            rate,
            {"is_melodic": False, "notes": [], "glides": [], "vibrato": []},
            {
                "available": True,
                "label": "airy/noisy indicator",
                "periodicity_median": 0.2,
            },
        )

        self.assertFalse(result["available"])
        self.assertIn("weak non-melodic residue", result["reason"])
        self.assertIn("omitted", result["limits"])

    def test_reports_phrase_activity_attacks_timbre_and_grounded_summary(self) -> None:
        rate = 8_000
        samples = np.zeros(rate * 8, dtype=float)
        for start_s, end_s, hz in ((0.5, 2.5, 220), (3.0, 5.8, 330), (6.3, 7.5, 440)):
            start, end = round(start_s * rate), round(end_s * rate)
            time = np.arange(end - start) / rate
            envelope = np.minimum(1.0, time / 0.15)
            samples[start:end] = 0.25 * envelope * np.sin(2 * np.pi * hz * time)
        singing = {
            "is_melodic": True,
            "notes": [
                {"note_name": "A3", "midi": 57, "start_s": 0.5, "end_s": 2.5, "dur_s": 2.0},
                {"note_name": "E4", "midi": 64, "start_s": 3.0, "end_s": 5.8, "dur_s": 2.8},
                {"note_name": "A4", "midi": 69, "start_s": 6.3, "end_s": 7.5, "dur_s": 1.2},
            ],
            "glides": [],
            "vibrato": [],
        }
        texture = {"available": True, "label": "clear/tonal"}

        result = analyze_vocal_perception(samples, rate, singing, texture)

        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["vocal_activity"]["phrase_count"], 3)
        self.assertTrue(result["onset_attack"]["available"])
        self.assertTrue(result["timbre_by_phrase"]["available"])
        self.assertIn("no stable vibrato pattern detected", result["summary"])
        self.assertFalse(result["layered_vocals"]["available"])
        self.assertIn("not reliable", result["layered_vocals"]["reason"])

    def test_detects_rapid_note_run_without_calling_it_a_specific_technique(self) -> None:
        rate = 8_000
        time = np.arange(rate * 3) / rate
        samples = 0.2 * np.sin(2 * np.pi * 220 * time)
        notes = [
            {"note_name": name, "midi": midi, "start_s": i * 0.3, "end_s": i * 0.3 + 0.24, "dur_s": 0.24}
            for i, (name, midi) in enumerate((("C4", 60), ("D4", 62), ("E4", 64), ("G4", 67), ("A4", 69)))
        ]
        result = analyze_vocal_perception(
            samples,
            rate,
            {"is_melodic": True, "notes": notes, "glides": [], "vibrato": []},
            {"available": True, "label": "clear/tonal"},
        )

        runs = result["pitch_events"]["rapid_note_run_candidates"]
        self.assertEqual(1, len(runs))
        self.assertEqual("rapid note-run candidate", runs[0]["label"])
        self.assertIn("not culture- or technique-specific", result["pitch_events"]["claim_limit"])


if __name__ == "__main__":
    unittest.main()
