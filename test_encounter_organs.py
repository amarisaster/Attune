from __future__ import annotations

from array import array
import math
import unittest

from attune_encounter.organs import analyze_passage


class ListeningOrganTests(unittest.TestCase):
    def test_harmonic_color_finds_local_a_energy_without_making_key_claim(self) -> None:
        rate = 16_000
        samples = array("f", (0.3 * math.sin(2 * math.pi * 440 * index / rate) for index in range(rate * 3)))
        result = analyze_passage(samples, rate)
        spectral = result["organs"][0]
        self.assertEqual(spectral["harmonic_color"]["strongest_classes"][0], "A")
        self.assertIn("not a key", spectral["harmonic_color"]["claim_limit"])
        self.assertEqual(result["analysis_scope"], "current_passage_only")
        self.assertIn("source-separated instruments", result["unsupported"])

    def test_change_organ_reports_transient_activity(self) -> None:
        rate = 16_000
        samples = array("f", [0.0] * (rate * 4))
        for second in (1, 2, 3):
            for index in range(second * rate, second * rate + rate // 20):
                samples[index] = 0.8
        change = analyze_passage(samples, rate)["organs"][1]
        self.assertGreater(change["events_per_minute"], 0)
        self.assertTrue(change["strongest_offsets_s"])


if __name__ == "__main__":
    unittest.main()
