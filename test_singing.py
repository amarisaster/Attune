"""Plain-assert smoke tests for singing.py. Run directly:
    venv\\Scripts\\python.exe test_singing.py
No pytest needed. Synthesizes audio with numpy, writes 16-bit PCM WAV with
stdlib `wave`, and runs it through analyze_singing(). Prints PASS per case
and exits 0, or raises/exits nonzero on failure.
"""

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from singing import analyze_singing, midi_to_note_name, hz_to_midi

SR = 22050


def write_wav(path, x, sr=SR):
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def synth_glide(freq_start, freq_end, dur_s, sr=SR, amp=0.5):
    """A continuous glissando: instantaneous frequency interpolated LINEARLY
    IN CENTS (i.e. exponentially in Hz) from freq_start to freq_end over
    dur_s -- the natural axis for a sung slide/scoop, and the same axis
    segment_notes' glide-slope detector measures on."""
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    cents_end = 1200.0 * np.log2(freq_end / freq_start)
    cents_t = cents_end * (t / dur_s)
    inst_freq = freq_start * (2.0 ** (cents_t / 1200.0))
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    sig = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    sig = sig / np.max(np.abs(sig))
    return amp * sig


def synth_tone(freq, dur_s, sr=SR, amp=0.5, vibrato_hz=0.0, vibrato_cents=0.0,
                amp_scale=1.0, vibrato_phase=0.0):
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    if vibrato_hz > 0:
        # cents modulation -> instantaneous frequency ratio 2**(cents/1200).
        # vibrato_phase shifts where in the sine cycle the note STARTS, so
        # tests can synthesize the same note beginning at a vibrato peak,
        # trough, or zero-crossing.
        mod = vibrato_cents * np.sin(2 * np.pi * vibrato_hz * t + vibrato_phase)
        inst_freq = freq * (2.0 ** (mod / 1200.0))
        phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    else:
        phase = 2 * np.pi * freq * t
    envelope = np.linspace(1.0, amp_scale, n)
    # a few harmonics so it doesn't look like a pure sine spike
    sig = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    sig = sig / np.max(np.abs(sig))
    return amp * envelope * sig


def case_a():
    print('--- case (a): 3-note melody with vibrato + rising dynamics ---')
    note1 = synth_tone(220.00, 0.5, amp=0.25, amp_scale=1.0)      # A3
    note2 = synth_tone(277.18, 0.5, amp=0.5, amp_scale=1.0)       # C#4
    note3 = synth_tone(329.63, 1.0, amp=0.75, vibrato_hz=5.5, vibrato_cents=30, amp_scale=1.6)  # E4
    # apply an overall amplitude ramp across the whole clip (2x rise)
    clip = np.concatenate([note1, note2, note3])
    ramp = np.linspace(1.0, 2.0, len(clip))
    clip = clip * ramp
    clip = clip / np.max(np.abs(clip)) * 0.9

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'a.wav'
        write_wav(p, clip)
        result = analyze_singing(str(p))

    assert result['is_melodic'] is True, f"expected is_melodic True, got {result.get('is_melodic')}"
    names = [n['note_name'] for n in result['notes']]
    print('detected notes:', names)
    # collapse consecutive duplicates for an order-only check
    collapsed = [names[0]] if names else []
    for n in names[1:]:
        if n != collapsed[-1]:
            collapsed.append(n)
    assert collapsed[:3] == ['A3', 'C#4', 'E4'], f'expected A3,C#4,E4 order, got {collapsed}'

    vibrato = result.get('vibrato', [])
    print('vibrato entries:', vibrato)
    e4_vib = [v for v in vibrato if v['note_name'] == 'E4' and 4.5 <= v['rate_hz'] <= 6.5]
    assert e4_vib, f'expected vibrato ~5.5Hz on E4, got {vibrato}'
    # note3 is 1.0s at 5.5Hz = 5.5 cycles, clearing the >=1.0s / >=4-cycle
    # gate. This is the "+/-30 cent" calibration case: true modulation is
    # +/-30 cents = 60 cents peak-to-peak. The raw (uncalibrated) reading
    # underestimates that because YIN's own 2048-sample analysis window
    # low-passes fast pitch modulation before extent is ever measured off
    # the f0 contour; extent_cents_calibrated divides by the modeled
    # attenuation (numpy.sinc(rate_hz * frame/sr)) to correct for it.
    # Empirically (see test run notes) raw lands ~35-42 and calibrated
    # lands ~50-70, close to the true 60 -- i.e. the calibration recovers
    # the singer's actual vibrato depth far better than the raw reading.
    assert 15.0 <= e4_vib[0]['extent_cents_raw'] <= 50.0, \
        f"expected extent_cents_raw in [15, 50], got {e4_vib[0]['extent_cents_raw']}"
    assert 45.0 <= e4_vib[0]['extent_cents'] <= 75.0, \
        f"expected calibrated extent_cents in [45, 75], got {e4_vib[0]['extent_cents']}"
    assert e4_vib[0]['extent_cents'] > e4_vib[0]['extent_cents_raw'], \
        'calibration should correct the raw reading UPWARD (YIN attenuates, never inflates)'
    assert 'rate_precision_hz' in e4_vib[0] and e4_vib[0]['rate_precision_hz'] > 0, \
        f"expected a positive rate_precision_hz, got {e4_vib[0]}"

    dyn = result.get('dynamics', {})
    print('dynamics:', dyn)
    assert dyn.get('end_db', -999) > dyn.get('start_db', 999), \
        f"expected rising dynamics arc, got start={dyn.get('start_db')} end={dyn.get('end_db')}"

    print('PASS case (a)')


def case_b():
    print('--- case (b): speech-like monotone wobble (not melodic) ---')
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    # <2 semitone wobble around 150Hz (2 semitones ~ ratio 1.122)
    wobble_cents = 60  # well under a semitone (100 cents) of range, so < 2 semitones total
    mod = wobble_cents * np.sin(2 * np.pi * 2.0 * t)
    inst_freq = 150.0 * (2.0 ** (mod / 1200.0))
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    sig = np.sin(phase) + 0.2 * np.sin(2 * phase)
    sig = 0.4 * sig / np.max(np.abs(sig))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'b.wav'
        write_wav(p, sig)
        result = analyze_singing(str(p))

    print('result:', {k: v for k, v in result.items() if k != 'notes'})
    assert result['is_melodic'] is False, f"expected is_melodic False, got {result}"
    print('PASS case (b)')


def case_c():
    print('--- case (c): white noise (no crash, not melodic) ---')
    rng = np.random.default_rng(42)
    sig = rng.normal(0, 0.3, int(1.5 * SR))
    sig = np.clip(sig, -1.0, 1.0)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'c.wav'
        write_wav(p, sig)
        result = analyze_singing(str(p))  # must not raise

    print('result:', {k: v for k, v in result.items() if k != 'notes'})
    assert result['is_melodic'] is False, f"expected is_melodic False, got {result}"
    print('PASS case (c)')


def case_d():
    print('--- case (d): held 1.5s note with wide (+/-60 cent) 5Hz vibrato stays ONE note ---')
    note = synth_tone(261.63, 1.5, amp=0.6, vibrato_hz=5.0, vibrato_cents=60, amp_scale=1.0)  # C4

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'd.wav'
        write_wav(p, note)
        result = analyze_singing(str(p))

    print('result:', {k: v for k, v in result.items() if k != 'notes'})
    # A single held note with wide vibrato has essentially no melodic range,
    # so is_melodic should be False -- but that must come from too little
    # range, NOT from the vibrato splitting into alternating semitone notes.
    names = [n['note_name'] for n in result.get('notes', [])]
    print('detected notes (pre is_melodic gate):', names)
    collapsed = [names[0]] if names else []
    for n in names[1:]:
        if n != collapsed[-1]:
            collapsed.append(n)
    assert len(collapsed) <= 1, \
        f'wide vibrato split the held note into {collapsed}, expected a single note'

    # Re-run the note/vibrato pipeline directly (bypassing the is_melodic
    # >4-semitone-range gate, which a single held note will never clear) to
    # confirm vibrato is still detected on that one note.
    from singing import yin_f0, segment_notes, detect_vibrato, load_wav
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'd.wav'
        write_wav(p, note)
        x, sr = load_wav(str(p))
    f0, conf = yin_f0(x, sr)
    notes = segment_notes(f0, conf, sr, 256, frame=2048)
    assert len(notes) == 1, f'expected exactly one segmented note, got {len(notes)}'
    n0 = notes[0]
    assert n0['note_name'] == 'C4', f"expected C4, got {n0['note_name']}"
    v = detect_vibrato(f0, conf, sr, 256, n0['_i0'], n0['_i1'])
    assert v is not None, 'expected vibrato to be detected on the held note'
    print('vibrato on held note:', v)
    # This is the "+/-60 cent" calibration case: true modulation is
    # +/-60 cents = 120 cents peak-to-peak. Empirically the calibrated
    # reading lands very close to that true value (the note is held long
    # enough, and clean enough, for the sinc-based YIN-window correction to
    # do its job well) -- tighter than the raw reading would allow.
    assert 90.0 <= v['extent_cents'] <= 150.0, \
        f"expected calibrated extent_cents in [90, 150] for +/-60c modulation, got {v['extent_cents']}"
    print('PASS case (d)')


def case_vibrato_phase_invariance():
    print('--- case (e): vibrato-phase invariance -- note stays ONE note regardless of vibrato start phase ---')
    from singing import yin_f0, segment_notes, detect_vibrato, load_wav

    def run_one(vibrato_cents, phase, label):
        note = synth_tone(261.63, 1.5, amp=0.6, vibrato_hz=5.0,
                           vibrato_cents=vibrato_cents, vibrato_phase=phase, amp_scale=1.0)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'e.wav'
            write_wav(p, note)
            x, sr = load_wav(str(p))
        f0, conf = yin_f0(x, sr)
        notes = segment_notes(f0, conf, sr, 256, frame=2048)
        names = [n['note_name'] for n in notes]
        print(f'  {label}: notes={names}')
        assert len(notes) == 1, f'{label}: expected exactly one note, got {len(notes)} ({names})'
        assert notes[0]['note_name'] == 'C4', f"{label}: expected C4, got {notes[0]['note_name']}"
        v = detect_vibrato(f0, conf, sr, 256, notes[0]['_i0'], notes[0]['_i1'])
        assert v is not None, f'{label}: expected vibrato to be detected'
        assert 4.0 <= v['rate_hz'] <= 6.0, f"{label}: expected rate ~5Hz, got {v['rate_hz']}"
        return v

    # +/-60 cents at four starting phases around the vibrato cycle: 0
    # (rising through zero), pi/2 (starting at the peak), pi (falling
    # through zero), 3pi/2 (starting at the trough). Before the rolling
    # midrange-center fix, starting near a peak/trough anchored the run's
    # "center" to that extreme and the rest of the note swinging back past
    # it tripped the persist-frame transition logic, splitting one note
    # into several short fringe fragments (see singing.py segment_notes
    # docstring).
    for phase, label in [(0.0, 'phase=0'), (np.pi / 2, 'phase=pi/2'),
                          (np.pi, 'phase=pi'), (3 * np.pi / 2, 'phase=3pi/2')]:
        run_one(60.0, phase, label)

    # Repeat once at a deeper +/-100 cent modulation -- large enough that a
    # naive running-median center (tried and rejected during this fix, see
    # singing.py docstring) still lags on the first monotonic quarter-cycle
    # and re-splits the note; the running-min/max midrange center does not.
    run_one(100.0, 0.0, 'vibrato_cents=100')

    print('PASS case (e): vibrato-phase invariance')


def case_yin_boundaries():
    print('--- case (f): YIN fmin/fmax strict boundary enforcement ---')
    from singing import yin_f0, _YIN_BOUNDARY_EPS_HZ

    def yin_result(freq, fmin, fmax):
        tone = synth_tone(freq, 1.0, amp=0.6)
        f0, conf = yin_f0(tone, SR, fmin=fmin, fmax=fmax)
        return f0, conf

    def voiced_hz(freq, fmin, fmax):
        f0, conf = yin_result(freq, fmin, fmax)
        return f0[f0 > 0]

    # _YIN_BOUNDARY_EPS_HZ (a couple Hz) is the code's own tolerance for
    # parabolic-interpolation grid quantization near the boundary -- not a
    # percentage slack like the old 0.9x/1.1x band. Tests use the same
    # constant rather than a magic number so they track the code's actual
    # contract instead of re-guessing it.
    eps = _YIN_BOUNDARY_EPS_HZ

    # 65 Hz tone with fmin=70 -- the true period doesn't fit inside
    # [tau_min, tau_max] for fmin=70 at all, so it must come back
    # unvoiced/no notes, never a frame reported below fmin.
    #
    # This must be a GENUINE rejection, not just "no reported pitch is below
    # fmin" -- that weaker check would also pass if YIN found a confident
    # octave-alias (e.g. 130Hz, the 2nd harmonic, which IS >= 70) and
    # reported the tone as voiced at the wrong pitch. Assert full rejection:
    # every frame's confidence is 0 (equivalently every f0 is 0/unvoiced).
    f0_65, conf_65 = yin_result(65.0, 70, 1000)
    v = f0_65[f0_65 > 0]
    print(f'  65Hz @ fmin=70: {len(v)} voiced frames, max conf={conf_65.max() if len(conf_65) else 0.0}',
          (v.min() if len(v) else None))
    assert conf_65.max() == 0.0 if len(conf_65) else True, \
        f'expected zero confidence on ALL frames for a 65Hz tone below fmin=70 (genuine rejection, ' \
        f'not just "no reported pitch below fmin"), got max conf={conf_65.max() if len(conf_65) else None}'
    assert np.all(f0_65 == 0.0), \
        f'expected every frame unvoiced (f0=0) for a 65Hz tone below fmin=70, got {v}'

    # 70 Hz tone with fmin=70 -- right at the boundary, must be accepted.
    v = voiced_hz(70.0, 70, 1000)
    print(f'  70Hz @ fmin=70: {len(v)} voiced frames', (v.min() if len(v) else None))
    assert len(v) > 0, 'expected 70Hz tone to be voiced at fmin=70'
    assert np.all(v >= 70.0 - eps), f'expected all voiced hz >= 70, got min={v.min()}'

    # 1000 Hz tone with fmax=1000 -- right at the boundary, must be accepted.
    v = voiced_hz(1000.0, 70, 1000)
    print(f'  1000Hz @ fmax=1000: {len(v)} voiced frames', (v.max() if len(v) else None))
    assert len(v) > 0, 'expected 1000Hz tone to be voiced at fmax=1000'
    assert np.all(v <= 1000.0 + eps), f'expected all voiced hz <= 1000+eps, got max={v.max()}'

    # 1050 Hz tone with fmax=1000 -- outside the range entirely (50Hz over,
    # far past the couple-Hz interpolation epsilon), must be rejected. Even
    # with the old 10% slack this would previously have squeaked through.
    #
    # Same genuine-rejection requirement as the fmin case above: an
    # octave-alias (e.g. 525Hz, half the true frequency, which IS <= 1000)
    # reported as a confident voiced pitch would pass a weaker "no reported
    # pitch exceeds fmax" check while still being a real detector bug.
    f0_1050, conf_1050 = yin_result(1050.0, 70, 1000)
    v = f0_1050[f0_1050 > 0]
    print(f'  1050Hz @ fmax=1000: {len(v)} voiced frames, max conf={conf_1050.max() if len(conf_1050) else 0.0}',
          (v.max() if len(v) else None))
    assert conf_1050.max() == 0.0 if len(conf_1050) else True, \
        f'expected zero confidence on ALL frames for a 1050Hz tone above fmax=1000 (genuine rejection, ' \
        f'not just "no reported pitch above fmax"), got max conf={conf_1050.max() if len(conf_1050) else None}'
    assert np.all(f0_1050 == 0.0), \
        f'expected every frame unvoiced (f0=0) for a 1050Hz tone above fmax=1000, got {v}'

    print('PASS case (f): YIN boundary enforcement')


def case_malformed_format_raises():
    print('--- case (g): format_singing_section raises on a malformed melodic dict ---')
    from singing import format_singing_section
    # is_melodic True with a notes list missing required keys (e.g. 'midi')
    # must raise -- this is what proves _run_singing_analysis's try/except
    # guard in server.py is load-bearing rather than decorative.
    bad = {'is_melodic': True, 'notes': [{'note_name': 'C4'}]}
    raised = False
    try:
        format_singing_section(bad)
    except Exception as e:
        raised = True
        print('  raised as expected:', repr(e))
    assert raised, 'expected format_singing_section to raise on a malformed dict (notes missing midi)'
    print('PASS case (g): malformed dict raises')


def case_glide_c4_to_c5():
    print('--- case (h): 2s C4->C5 glissando is reported as ONE glide, not a fake note ladder ---')
    from singing import yin_f0, segment_notes, load_wav

    sig = synth_glide(261.63, 523.25, 2.0, amp=0.6)  # C4 -> C5, 1200 cents / 2s = 600 cents/s
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'h.wav'
        write_wav(p, sig)
        x, sr = load_wav(str(p))
    f0, conf = yin_f0(x, sr)
    notes, glides = segment_notes(f0, conf, sr, 256, frame=2048, return_glides=True)
    print('  notes:', [(n['note_name'], n['dur_s']) for n in notes])
    print('  glides:', glides)
    # A smooth 2-second slide must NOT be chopped into a semitone staircase
    # (the bug this whole fix round exists for) -- there should be at most
    # a couple of short plateau fragments at the very ends, and definitely
    # not a full C#4..B4 ladder of "notes".
    assert len(notes) <= 2, f'expected the slide to stay out of the notes list, got {notes}'
    assert len(glides) == 1, f'expected exactly one glide event, got {glides}'
    g = glides[0]
    assert g['type'] == 'glide'
    assert g['from_note'] == 'C4', f"expected glide to start at C4, got {g['from_note']}"
    assert g['to_note'] == 'C5', f"expected glide to end at C5, got {g['to_note']}"
    assert g['dur_s'] >= 1.0, f"expected the glide to span most of the 2s clip, got {g['dur_s']}"
    print('PASS case (h): C4->C5 glide')


def case_glide_c4_to_e4():
    print('--- case (i): 1s C4->E4 glissando (400 cents/s) is reported as ONE glide ---')
    from singing import yin_f0, segment_notes, load_wav

    sig = synth_glide(261.63, 329.63, 1.0, amp=0.6)  # C4 -> E4, 400 cents / 1s = 400 cents/s
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'i.wav'
        write_wav(p, sig)
        x, sr = load_wav(str(p))
    f0, conf = yin_f0(x, sr)
    notes, glides = segment_notes(f0, conf, sr, 256, frame=2048, return_glides=True)
    print('  notes:', [(n['note_name'], n['dur_s']) for n in notes])
    print('  glides:', glides)
    assert len(notes) <= 2, f'expected the slide to stay out of the notes list, got {notes}'
    assert len(glides) == 1, f'expected exactly one glide event, got {glides}'
    g = glides[0]
    assert g['from_note'] == 'C4', f"expected glide to start at C4, got {g['from_note']}"
    assert g['to_note'] == 'E4', f"expected glide to end at E4, got {g['to_note']}"
    print('PASS case (i): C4->E4 glide')


def case_glide_does_not_fire_on_step_transitions():
    print('--- case (j): instant step transitions between held notes are NOT misclassified as glides ---')
    from singing import yin_f0, segment_notes, load_wav

    # Same synthesis as case (a): 3 held notes back-to-back with instant
    # (sample-boundary) transitions, no gliding. The local slope right at
    # each step edge is enormous over a short window, but the edge is only
    # a few ms wide -- far short of the ~250ms glide_min_dur_s gate -- so
    # it must NOT be reported as a glide.
    note1 = synth_tone(220.00, 0.5, amp=0.25, amp_scale=1.0)      # A3
    note2 = synth_tone(277.18, 0.5, amp=0.5, amp_scale=1.0)       # C#4
    note3 = synth_tone(329.63, 1.0, amp=0.75, vibrato_hz=5.5, vibrato_cents=30, amp_scale=1.6)  # E4
    clip = np.concatenate([note1, note2, note3])
    ramp = np.linspace(1.0, 2.0, len(clip))
    clip = clip * ramp
    clip = clip / np.max(np.abs(clip)) * 0.9

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'j.wav'
        write_wav(p, clip)
        x, sr = load_wav(str(p))
    f0, conf = yin_f0(x, sr)
    notes, glides = segment_notes(f0, conf, sr, 256, frame=2048, return_glides=True)
    print('  notes:', [(n['note_name'], n['dur_s']) for n in notes])
    print('  glides:', glides)
    assert glides == [], f'expected NO glides for instant step transitions, got {glides}'
    names = [n['note_name'] for n in notes]
    assert names == ['A3', 'C#4', 'E4'], f'expected A3,C#4,E4 as real notes, got {names}'
    print('PASS case (j): step transitions stay notes, not glides')


def case_vibrato_dual_duration():
    print('--- case (k): 5Hz +/-60c vibrato at 1.5s AND 2.0s, 2 phases each -- calibrated extent near 120c ---')
    from singing import yin_f0, segment_notes, detect_vibrato, load_wav

    # True peak-to-peak extent for +/-60 cent amplitude modulation is
    # 120 cents. Calibrated extent must land within +/-8% of that in all
    # four (duration x phase) combinations -- i.e. [110.4, 129.6].
    target = 120.0
    tol = 0.08 * target

    for dur_s in (1.5, 2.0):
        for phase, label in [(0.0, 'phase=0'), (np.pi / 2, 'phase=pi/2')]:
            note = synth_tone(261.63, dur_s, amp=0.6, vibrato_hz=5.0,
                               vibrato_cents=60, vibrato_phase=phase, amp_scale=1.0)
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / 'k.wav'
                write_wav(p, note)
                x, sr = load_wav(str(p))
            f0, conf = yin_f0(x, sr)
            notes = segment_notes(f0, conf, sr, 256, frame=2048)
            assert len(notes) == 1, f'dur={dur_s} {label}: expected one note, got {len(notes)}'
            v = detect_vibrato(f0, conf, sr, 256, notes[0]['_i0'], notes[0]['_i1'])
            assert v is not None, f'dur={dur_s} {label}: expected vibrato to be detected'
            print(f'  dur={dur_s} {label}: {v}')
            assert abs(v['extent_cents'] - target) <= tol, \
                f"dur={dur_s} {label}: expected calibrated extent within +/-8% of 120c, got {v['extent_cents']}"
    print('PASS case (k): dual-duration vibrato calibration')


def synth_scale(start_freq, semitone_steps, notes_per_sec, sr=SR, amp=0.5):
    """A fast run of DISCRETE held notes (a melisma), each `semitone_steps[i]`
    away from start_freq in equal-tempered semitones, held for
    1/notes_per_sec seconds with an instantaneous (sample-boundary) step
    transition between them -- no gliding, unlike synth_glide. This is the
    shape a fast scale/run actually has: constant pitch per note, not a
    continuous slide."""
    dur = 1.0 / notes_per_sec
    parts = []
    for steps in semitone_steps:
        freq = start_freq * (2.0 ** (steps / 12.0))
        parts.append(synth_tone(freq, dur, sr=sr, amp=amp, amp_scale=1.0))
    sig = np.concatenate(parts)
    return sig / max(np.max(np.abs(sig)), 1e-9) * amp


def case_melisma_semitone_scale():
    print('--- case (l): 6 notes/sec semitone scale (8 steps) -- discrete notes, zero glides ---')
    from singing import yin_f0, segment_notes, load_wav

    sig = synth_scale(261.63, list(range(8)), notes_per_sec=6.0, amp=0.6)  # C4..G#4, 8 semitone steps
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'l.wav'
        write_wav(p, sig)
        x, sr = load_wav(str(p))
    f0, conf = yin_f0(x, sr)
    notes, glides = segment_notes(f0, conf, sr, 256, frame=2048, return_glides=True)
    print('  notes:', [(n['note_name'], n['dur_s']) for n in notes])
    print('  glides:', glides)
    # The window used for the glide-slope test (200ms) is wider than a
    # single note at 6 notes/sec (~167ms), so the raw slope-threshold flag
    # stays elevated continuously across the whole run and, pre-fix, the
    # whole scale got merged into one fake glide, starving the note count.
    # The plateau check must recover each held note from that run.
    assert glides == [], f'expected zero glides for a fast discrete scale, got {glides}'
    assert len(notes) >= 6, f'expected at least 6 discrete notes, got {len(notes)}: {notes}'
    print('PASS case (l): 6 notes/sec semitone scale')


def case_melisma_wholetone_scale():
    print('--- case (m): 5 notes/sec whole-tone scale -- discrete notes, zero glides ---')
    from singing import yin_f0, segment_notes, load_wav

    sig = synth_scale(261.63, [0, 2, 4, 6, 8, 10], notes_per_sec=5.0, amp=0.6)  # 6 whole-tone steps
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'm.wav'
        write_wav(p, sig)
        x, sr = load_wav(str(p))
    f0, conf = yin_f0(x, sr)
    notes, glides = segment_notes(f0, conf, sr, 256, frame=2048, return_glides=True)
    print('  notes:', [(n['note_name'], n['dur_s']) for n in notes])
    print('  glides:', glides)
    assert glides == [], f'expected zero glides for a fast discrete whole-tone scale, got {glides}'
    assert len(notes) >= 5, f'expected at least 5 discrete notes, got {len(notes)}: {notes}'
    print('PASS case (m): 5 notes/sec whole-tone scale')


def synth_tone_jitter(freq, dur_s, jitter_cents, sr=SR, amp=0.5, seed=0):
    """A flat note with WHITE (uncorrelated update-to-update) cents jitter --
    no periodic structure at all, just noise on the instantaneous frequency.
    Jitter is generated at a coarse update rate (~100Hz) and held constant
    between updates (a zero-order hold) so it doesn't alias into ultrasonic
    junk, then applied to the phase accumulator sample-by-sample."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    update_hz = 100.0
    n_updates = max(2, int(dur_s * update_hz) + 1)
    jitter_updates = rng.normal(0.0, jitter_cents, n_updates)
    update_idx = np.linspace(0, n_updates - 1, n)
    jitter_cents_trace = np.interp(update_idx, np.arange(n_updates), jitter_updates)
    inst_freq = freq * (2.0 ** (jitter_cents_trace / 1200.0))
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    sig = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    sig = sig / np.max(np.abs(sig))
    return amp * sig


def synth_tone_wander(freq, dur_s, wander_cents, sr=SR, amp=0.5, seed=0):
    """A flat note with smooth APERIODIC low-frequency wander on pitch (a
    random walk, low-pass filtered by a moving average so it has no sharp
    steps) -- real slow pitch drift with no single stable rate, as opposed
    to a clean sinusoid."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    update_hz = 50.0
    n_updates = max(2, int(dur_s * update_hz) + 1)
    steps = rng.normal(0.0, 1.0, n_updates)
    walk = np.cumsum(steps)
    # smooth with a short moving average so the walk is aperiodic but not
    # jagged, then rescale to the requested cents amplitude
    kernel = np.ones(5) / 5.0
    walk = np.convolve(walk, kernel, mode='same')
    walk = walk - np.mean(walk)
    peak = np.max(np.abs(walk))
    if peak > 0:
        walk = walk / peak * wander_cents
    update_idx = np.linspace(0, n_updates - 1, n)
    wander_trace = np.interp(update_idx, np.arange(n_updates), walk)
    inst_freq = freq * (2.0 ** (wander_trace / 1200.0))
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    sig = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    sig = sig / np.max(np.abs(sig))
    return amp * sig


def case_vibrato_jitter_false_positives():
    print('--- case (n): 30 trials of 40-cent white jitter on a flat note -- <=2 false vibrato detections ---')
    from singing import yin_f0, segment_notes, detect_vibrato, load_wav

    n_trials = 30
    false_positives = 0
    for trial in range(n_trials):
        note = synth_tone_jitter(261.63, 1.5, jitter_cents=40.0, amp=0.6, seed=trial)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'n.wav'
            write_wav(p, note)
            x, sr = load_wav(str(p))
        f0, conf = yin_f0(x, sr)
        notes = segment_notes(f0, conf, sr, 256, frame=2048)
        if not notes:
            continue
        # use the longest segmented run in case jitter fragmented the note
        n0 = max(notes, key=lambda nn: nn['dur_s'])
        if n0['dur_s'] < 0.9:
            continue
        v = detect_vibrato(f0, conf, sr, 256, n0['_i0'], n0['_i1'])
        if v is not None:
            false_positives += 1
            print(f'  trial {trial}: FALSE POSITIVE {v}')
    print(f'  false positives: {false_positives}/{n_trials}')
    assert false_positives <= 2, f'expected <=2 false vibrato detections on white jitter, got {false_positives}'
    print('PASS case (n): white-jitter false-positive rate')


def case_vibrato_wander_false_positives():
    print('--- case (o): 30 trials of smooth aperiodic wander on a flat note -- <=2 false vibrato detections ---')
    from singing import yin_f0, segment_notes, detect_vibrato, load_wav

    n_trials = 30
    false_positives = 0
    for trial in range(n_trials):
        note = synth_tone_wander(261.63, 1.5, wander_cents=40.0, amp=0.6, seed=trial + 1000)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'o.wav'
            write_wav(p, note)
            x, sr = load_wav(str(p))
        f0, conf = yin_f0(x, sr)
        notes = segment_notes(f0, conf, sr, 256, frame=2048)
        if not notes:
            continue
        n0 = max(notes, key=lambda nn: nn['dur_s'])
        if n0['dur_s'] < 0.9:
            continue
        v = detect_vibrato(f0, conf, sr, 256, n0['_i0'], n0['_i1'])
        if v is not None:
            false_positives += 1
            print(f'  trial {trial}: FALSE POSITIVE {v}')
    print(f'  false positives: {false_positives}/{n_trials}')
    assert false_positives <= 2, f'expected <=2 false vibrato detections on smooth wander, got {false_positives}'
    print('PASS case (o): smooth-wander false-positive rate')


def sanity_checks():
    assert midi_to_note_name(69) == 'A4'
    assert midi_to_note_name(60) == 'C4'
    assert abs(hz_to_midi(440.0) - 69.0) < 1e-6
    print('PASS sanity checks')


if __name__ == '__main__':
    sanity_checks()
    case_a()
    case_b()
    case_c()
    case_d()
    case_vibrato_phase_invariance()
    case_yin_boundaries()
    case_malformed_format_raises()
    case_glide_c4_to_c5()
    case_glide_c4_to_e4()
    case_glide_does_not_fire_on_step_transitions()
    case_vibrato_dual_duration()
    case_melisma_semitone_scale()
    case_melisma_wholetone_scale()
    case_vibrato_jitter_false_positives()
    case_vibrato_wander_false_positives()
    print('ALL PASS')
    sys.exit(0)
