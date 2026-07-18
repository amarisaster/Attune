"""Plain-assert tests for energy_gate.should_skip_stt. Run directly:
    venv\\Scripts\\python.exe test_energy_gate.py
No pytest, no I/O, no ffmpeg -- exercises the pure RMS/voiced-fraction gate
used to skip STT on silent-or-near-silent audio (see energy_gate.py's
docstring for the hallucination-on-silence rationale).
"""

import sys

import numpy as np

from energy_gate import rms_dbfs, should_skip_stt, voiced_fraction

SR = 16000


def _silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * seconds))


def _tiny_noise(seconds: float, amplitude: float = 1e-4, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-amplitude, amplitude, int(sr * seconds))


def _tone(seconds: float, freq: float = 220.0, amplitude: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


def case_literal_silence_is_gated():
    print('--- case (a): a literally-zero array is gated ---')
    x = _silence(3.0)
    skip, reason = should_skip_stt(x, SR)
    print('  skip:', skip, 'reason:', reason)
    assert skip is True
    assert reason
    print('PASS case (a)')


def case_quiet_noise_floor_is_gated():
    print('--- case (b): quiet noise well under the default -50 dBFS gate is gated ---')
    x = _tiny_noise(3.0)
    print('  rms_dbfs:', rms_dbfs(x))
    skip, reason = should_skip_stt(x, SR)
    assert skip is True
    print('PASS case (b)')


def case_loud_tone_is_not_gated():
    print('--- case (c): a sustained loud tone clears both the RMS and voiced-fraction checks ---')
    x = _tone(3.0)
    print('  rms_dbfs:', rms_dbfs(x))
    skip, reason = should_skip_stt(x, SR)
    print('  skip:', skip, 'reason:', reason)
    assert skip is False
    assert reason == ''
    print('PASS case (c)')


def case_brief_transient_in_silence_is_gated():
    print('--- case (d): a short loud burst (a giggle) inside an otherwise silent 7.6s '
          'chunk is still gated -- mirrors the trailing giggle-silence chunk that '
          'hallucinated fluent Indonesian. The burst alone is loud enough that the '
          "clip's OVERALL RMS clears the -50 dBFS gate on its own (demonstrating why "
          'the overall-RMS check alone is not enough); the voiced-fraction check is '
          'what catches it. ---')
    burst = _tone(0.1, amplitude=0.3)
    tail = _silence(7.5)
    x = np.concatenate([burst, tail])
    overall = rms_dbfs(x)
    vfrac = voiced_fraction(x, SR, -50.0)
    print('  overall rms_dbfs:', overall, ' voiced_fraction:', vfrac)
    assert overall >= -50.0, 'test setup: burst should clear the overall-RMS gate alone'
    assert vfrac < 0.02, 'test setup: burst should still be a tiny fraction of frames'
    skip, reason = should_skip_stt(x, SR)
    print('  skip:', skip, 'reason:', reason)
    assert skip is True
    print('PASS case (d)')


def case_custom_threshold_is_honored():
    print('--- case (e): a quieter-but-real tone is gated at the default threshold but '
          'passes once ATTUNE_STT_SILENCE_DBFS is lowered (made more permissive) ---')
    x = _tone(3.0, amplitude=0.003)  # roughly -50 dBFS-ish quiet speech
    default_skip, _ = should_skip_stt(x, SR, threshold_dbfs=-50.0)
    permissive_skip, _ = should_skip_stt(x, SR, threshold_dbfs=-70.0)
    print('  default_skip:', default_skip, 'permissive_skip:', permissive_skip)
    assert permissive_skip is False
    print('PASS case (e)')


def case_empty_array_is_gated():
    print('--- case (f): an empty array is gated, not a crash ---')
    skip, reason = should_skip_stt(np.array([]), SR)
    assert skip is True
    print('PASS case (f)')


if __name__ == '__main__':
    case_literal_silence_is_gated()
    case_quiet_noise_floor_is_gated()
    case_loud_tone_is_not_gated()
    case_brief_transient_in_silence_is_gated()
    case_custom_threshold_is_honored()
    case_empty_array_is_gated()
    print('ALL PASS')
    sys.exit(0)
