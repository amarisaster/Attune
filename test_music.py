"""Tests for music.py — pure DSP, no models, no server, no ffmpeg.

Run: python test_music.py
Pattern matches test_singing.py: numpy-synthesized ground truth, plain
asserts, PASS prints, statistical false-positive trials.
"""

import sys

import numpy as np

from music import (SR, analyze_music, chromagram, detect_chords, detect_sections,
                   energy_arc, estimate_key, estimate_tempo, format_compare_section,
                   format_music_section, spectral_flatness, stft_mags,
                   compare_voice_to_track)

# ── Synth helpers (adapted from test_singing.py) ─────────────────────────────


def synth_tone(freq_hz, dur_s, sr=SR, amp=0.5):
    """Tone with mild harmonics so it doesn't look like a pure sine spike."""
    t = np.arange(int(dur_s * sr)) / sr
    x = (np.sin(2 * np.pi * freq_hz * t)
         + 0.3 * np.sin(2 * np.pi * 2 * freq_hz * t)
         + 0.1 * np.sin(2 * np.pi * 3 * freq_hz * t))
    return (amp * x / np.max(np.abs(x))).astype(np.float64)


def synth_triad(root_hz, quality, dur_s, sr=SR, amp=0.5):
    """Root-position triad with harmonics per voice."""
    third = root_hz * (2 ** (4 / 12) if quality == 'maj' else 2 ** (3 / 12))
    fifth = root_hz * (2 ** (7 / 12))
    x = sum(synth_tone(f, dur_s, sr, amp=1.0) for f in (root_hz, third, fifth))
    return (amp * x / np.max(np.abs(x))).astype(np.float64)


def synth_click_track(bpm, dur_s, sr=SR, noise_amp=0.05, seed=0):
    """Short noise bursts on the beat over low background noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(dur_s * sr)) * noise_amp
    period = int(sr * 60.0 / bpm)
    burst = int(0.02 * sr)
    env = np.hanning(burst)
    for i in range(0, len(x) - burst, period):
        x[i:i + burst] += env * rng.standard_normal(burst) * 0.9
    return (x / np.max(np.abs(x))).astype(np.float64)


def note_hz(name_midi):
    return 440.0 * 2 ** ((name_midi - 69) / 12)


def chroma_of(x):
    mags, freqs, times = stft_mags(x)
    return chromagram(mags, freqs), times, mags


# ── Cases ────────────────────────────────────────────────────────────────────


def test_key_arpeggio():
    # A natural minor arpeggio/scale walk: A3 C4 E4 A4 E4 C4 B3 A3, looped.
    midis = [57, 60, 64, 69, 64, 60, 59, 57]
    x = np.concatenate([synth_tone(note_hz(m), 0.4) for m in midis * 4])
    chroma, _, _ = chroma_of(x)
    key = estimate_key(chroma)
    assert key['key'] == 'A minor', f"expected A minor, got {key}"
    print('PASS key: A-minor arpeggio ->', key['key'], key['confidence'])


def test_chords_progression():
    # C - F - G - C, 2 s each.
    seq = [(note_hz(48), 'maj'), (note_hz(53), 'maj'), (note_hz(55), 'maj'), (note_hz(48), 'maj')]
    labels = ['C', 'F', 'G', 'C']
    x = np.concatenate([synth_triad(f, q, 2.0) for f, q in seq])
    chroma, times, _ = chroma_of(x)
    res = detect_chords(chroma, times)
    got = [c['chord'] for c in res['chords']]
    # Require >=3 of the 4 target labels present in order.
    matched, gi = 0, 0
    for want in labels:
        while gi < len(got):
            if got[gi] == want:
                matched += 1
                gi += 1
                break
            gi += 1
    assert matched >= 3, f"expected >=3 of {labels} in order, got {got}"
    assert res['confidence'] in ('medium', 'high'), f"triads should not be low confidence: {res}"
    print(f"PASS chords: {labels} -> {got} [{res['confidence']}]")


def test_tempo_click():
    x = synth_click_track(120.0, 20.0)
    res = estimate_tempo(x)
    assert res['bpm'] is not None, 'no bpm on a click track'
    assert abs(res['bpm'] - 120.0) <= 2.0, f"expected ~120, got {res['bpm']}"
    assert res['confidence'] == 'high', f"clean click should be high, got {res['confidence']}"
    grid = np.array(res['beat_times'])
    period = 0.5
    errs = np.abs((grid / period) - np.round(grid / period)) * period
    assert float(np.mean(errs)) < 0.03, f"beat grid mean error {np.mean(errs):.3f}s"
    print(f"PASS tempo: click 120 -> {res['bpm']} BPM [{res['confidence']}], "
          f"grid err {np.mean(errs)*1000:.0f}ms")


def test_tempo_octave_guard():
    x = synth_click_track(60.0, 30.0, seed=3)
    res = estimate_tempo(x)
    assert res['bpm'] is not None
    ok_60 = abs(res['bpm'] - 60.0) <= 2.0
    reported_120_confident = abs(res['bpm'] - 120.0) <= 2.0 and res['confidence'] == 'high'
    assert not reported_120_confident, f"60 BPM click confidently doubled: {res}"
    print(f"PASS octave guard: 60 BPM click -> {res['bpm']} [{res['confidence']}]"
          + ('' if ok_60 else ' (octave ambiguity allowed, but not confident-wrong)'))


def test_noise_false_positives():
    rng = np.random.default_rng(42)
    confident = 0
    trials = 20
    for i in range(trials):
        # Mix of white and brown-ish noise clips, 6 s.
        w = rng.standard_normal(6 * SR)
        b = np.cumsum(rng.standard_normal(6 * SR))
        b = b / np.max(np.abs(b))
        x = (w / np.max(np.abs(w))) * 0.5 + b * 0.5 * (i % 2)
        chroma, times, _ = chroma_of(x)
        key = estimate_key(chroma)
        chords = detect_chords(chroma, times)
        tempo = estimate_tempo(x)
        if key['confidence'] == 'strong' or chords['confidence'] == 'high' \
                or tempo['confidence'] == 'high':
            confident += 1
    assert confident <= 2, f"{confident}/{trials} noise clips got confident claims"
    print(f"PASS noise FP: {confident}/{trials} confident claims on noise")


def test_energy_ramp():
    x = synth_tone(220.0, 12.0)
    ramp = np.linspace(0.05, 1.0, len(x))
    res = energy_arc(x * ramp)
    assert res['start_db'] < res['end_db'], f"ramp not detected: {res}"
    assert res['loudest_t'] > res['quietest_t'], f"extremes inverted: {res}"
    print(f"PASS energy: ramp {res['start_db']} -> {res['end_db']} dB")


def test_sections_change():
    # 30 s of A-minor arpeggio then 30 s of Db-major triads — a hard seam.
    a = np.concatenate([synth_tone(note_hz(m), 0.5) for m in [57, 60, 64, 69] * 15])
    b = np.concatenate([synth_triad(note_hz(49), 'maj', 2.0) for _ in range(15)])
    x = np.concatenate([a, b])
    chroma, times, _ = chroma_of(x)
    changes = detect_sections(chroma, x, SR, times)
    boundary = len(a) / SR
    assert any(abs(t - boundary) < 10.0 for t in changes), \
        f"no section change near {boundary:.0f}s in {changes}"
    print(f"PASS sections: seam at {boundary:.0f}s detected in {changes}")


def test_card_render():
    result = {
        'duration_s': 227.0,
        'key': {'key': 'A minor', 'correlation': 0.82, 'confidence': 'strong'},
        'chords': {'chords': [], 'progression': ['Am', 'F', 'C', 'G'],
                   'coverage': 0.82, 'confidence': 'medium'},
        'tempo': {'bpm': 128.0, 'confidence': 'high', 'beat_times': [0.5, 1.0]},
        'energy': {'start_db': -18.0, 'end_db': -12.0, 'loudest_t': 80.0,
                   'loudest_db': -9.0, 'quietest_t': 190.0, 'quietest_db': -30.0,
                   'sections': []},
        'section_changes': [52.0, 104.0, 156.0],
        'spectral_flatness': 0.21,
    }
    card = format_music_section(result, melody={'summary': 'line centers G4-E5, 140 note events'})
    assert 'KEY  : A minor [strong]' in card
    assert 'TEMPO: 128.0 BPM [high]' in card
    assert 'CHORDS: Am → F → C → G' in card
    assert '[basic-pitch]' in card
    for banned in ('happy', 'sad', 'melancholic', 'euphoric'):
        assert banned not in card.lower()
    cmp_card = format_compare_section({
        'in_key_fraction': 0.91, 'track_key': 'A minor', 'track_key_confidence': 'strong',
        'timing_median_ms': 40.0, 'timing_feel': 'behind', 'tempo_confidence': 'high',
        'voice_build_db': 6.0, 'track_build_db': 9.0,
    })
    assert 'PITCH : 91%' in cmp_card
    assert 'behind' in cmp_card
    print('PASS card render:\n' + card + '\n' + cmp_card)


def test_compare_voice_to_track():
    singing_result = {
        'notes': [{'midi': 69, 'start_s': 0.51, 'dur_s': 1.0},   # A, in A minor
                  {'midi': 72, 'start_s': 1.52, 'dur_s': 1.0},   # C
                  {'midi': 76, 'start_s': 2.55, 'dur_s': 1.0},   # E
                  {'midi': 70, 'start_s': 3.49, 'dur_s': 0.2}],  # Bb, out of key
        'dynamics': {'start_db': -20.0, 'end_db': -14.0},
    }
    music_result = {
        'key': {'key': 'A minor', 'confidence': 'strong'},
        'tempo': {'bpm': 120.0, 'confidence': 'high',
                  'beat_times': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]},
        'energy': {'start_db': -18.0, 'end_db': -9.0},
    }
    cmp = compare_voice_to_track(singing_result, music_result)
    assert cmp['in_key_fraction'] > 0.9, cmp
    assert cmp['timing_feel'] in ('on', 'behind'), cmp
    assert cmp['voice_build_db'] == 6.0 and cmp['track_build_db'] == 9.0
    print(f"PASS compare: in-key {cmp['in_key_fraction']}, "
          f"timing {cmp['timing_median_ms']}ms {cmp['timing_feel']}")


if __name__ == '__main__':
    test_key_arpeggio()
    test_chords_progression()
    test_tempo_click()
    test_tempo_octave_guard()
    test_noise_false_positives()
    test_energy_ramp()
    test_sections_change()
    test_card_render()
    test_compare_voice_to_track()
    print('\nALL MUSIC TESTS PASSED')
    sys.exit(0)
