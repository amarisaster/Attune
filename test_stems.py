"""Tests for stems.py and basic_pitch_onnx.py — model-dependent.

Run: python test_stems.py
Models are downloads (scripts/get-models.py); tests SKIP cleanly when absent
rather than hard-requiring them.
"""

import sys

import numpy as np

import basic_pitch_onnx
import stems


def _skip(reason):
    print(f'SKIP: {reason}')
    sys.exit(0)


def synth_voice(dur_s, sr):
    """Vibrato 'voice': 300 Hz with 5.5 Hz / 40-cent vibrato + harmonics."""
    t = np.arange(int(dur_s * sr)) / sr
    cents = 40 * np.sin(2 * np.pi * 5.5 * t)
    freq = 300.0 * 2 ** (cents / 1200)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    x = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    return x / np.max(np.abs(x))


def synth_drums(dur_s, sr, bpm=120.0, seed=1):
    rng = np.random.default_rng(seed)
    x = np.zeros(int(dur_s * sr))
    period = int(sr * 60.0 / bpm)
    burst = int(0.05 * sr)
    env = np.exp(-np.linspace(0, 6, burst))
    for i in range(0, len(x) - burst, period):
        x[i:i + burst] += env * rng.standard_normal(burst)
    return x / np.max(np.abs(x))


def band_energy(x_mono, sr, lo, hi):
    spec = np.abs(np.fft.rfft(x_mono))
    freqs = np.fft.rfftfreq(len(x_mono), 1 / sr)
    sel = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(spec[sel] ** 2))


def test_separation():
    sr = stems.SR
    dur = 8.0
    voice = synth_voice(dur, sr)
    drums = synth_drums(dur, sr)
    mix = 0.6 * voice + 0.4 * drums
    x = np.stack([mix, mix])  # stereo

    out = stems.separate(x, progress_cb=lambda f: None)
    assert out['vocals'].shape == x.shape, 'vocals length mismatch'
    assert out['instrumental'].shape == x.shape, 'instrumental length mismatch'

    # The voice lives at 300-1000 Hz (fundamental + harmonics). The vocals
    # stem should hold a larger share of that band than the instrumental
    # holds, relative to their broadband content.
    v = out['vocals'][0]
    i = out['instrumental'][0]
    v_ratio = band_energy(v, sr, 250, 1000) / (band_energy(v, sr, 20, sr / 2 - 1) + 1e-12)
    i_ratio = band_energy(i, sr, 250, 1000) / (band_energy(i, sr, 20, sr / 2 - 1) + 1e-12)
    assert v_ratio > i_ratio, f'voice-band share not higher in vocals stem: {v_ratio:.3f} vs {i_ratio:.3f}'
    print(f'PASS separation: voice-band share vocals {v_ratio:.2f} vs instrumental {i_ratio:.2f}')
    return out


def test_stem_wav_names(tmp_out):
    import re
    import tempfile
    from pathlib import Path
    drops_re = re.compile(r'^[A-Za-z0-9_-]+\.(webm|ogg|m4a|mp3|wav)$')
    with tempfile.TemporaryDirectory() as td:
        names = stems.write_stem_wavs(tmp_out, Path(td))
        for stem_name, fname in names.items():
            assert drops_re.match(fname), f'{fname} fails DROPS_NAME_RE'
            assert (Path(td) / fname).stat().st_size > 44, f'{fname} empty'
    print(f'PASS stem wavs: {list(names.values())}')


def test_melody_model():
    sr = 22050
    def tone(f, d):
        t = np.arange(int(d * sr)) / sr
        x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
        return 0.5 * x / np.max(np.abs(x))
    midis = [57, 60, 64, 69]
    hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)  # noqa: E731
    x = np.concatenate([tone(hz(m), 0.4) for m in midis * 2])
    r = basic_pitch_onnx.transcribe(x, sr)
    got = sorted(set(n['midi'] for n in r['notes']))
    for want in midis:
        assert any(abs(g - want) <= 1 for g in got), f'midi {want} not recovered in {got}'
    print(f'PASS melody: {midis} recovered as {got}')


if __name__ == '__main__':
    ok_bp, reason_bp = basic_pitch_onnx.is_available()
    ok_st, reason_st = stems.is_available()
    if not ok_bp and not ok_st:
        _skip(f'models not available ({reason_bp}; {reason_st})')

    if ok_bp:
        test_melody_model()
    else:
        print(f'SKIP melody: {reason_bp}')

    if ok_st:
        out = test_separation()
        test_stem_wav_names(out)
    else:
        print(f'SKIP separation: {reason_st}')

    print('\nALL STEM/MELODY TESTS PASSED (or skipped where models absent)')
    sys.exit(0)
