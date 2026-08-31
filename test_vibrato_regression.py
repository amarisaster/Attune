"""Independent vibrato regression fixture. Run directly:
    venv\\Scripts\\python.exe test_vibrato_regression.py
No pytest needed. Exits 0 on pass, raises on failure.

Provenance: written as a *control* before reporting a suspected detector
bug -- a real clip returned an empty vibrato row, and rather than filing
"detector may be broken" this renders a signal built to detect_vibrato's
own published spec and checks the detector against known ground truth.
The detector passed (5.00 Hz built, 4.99 Hz read), which flipped the
conclusion: the empty row on the real clip was a correct refusal, not a
miss. The fixture keeps that property pinned down for future changes.

What this adds beyond test_singing.py's vibrato cases, deliberately:

1. INDEPENDENT SYNTHESIS PATH. test_singing.py builds its clips with
   numpy (vectorized cumsum phase integration, shared synth_tone helper).
   This file is pure stdlib -- math/struct/wave, sample-by-sample phase
   accumulation, written without reference to the other test's helpers.
   If a bug ever lands in an assumption shared between the main tests'
   synthesis and the analysis code, an oracle that shares zero synthesis
   code with them is the one that catches it.

2. TIGHTER ACCURACY BAND. test_singing.py asserts rate in [4.5, 6.5] on
   a 1.0s note. This note is held 5.0s (25 full cycles at 5 Hz), which
   supports a much stronger claim: rate within +/-0.3 Hz of ground truth.
   Observed on first run: rate_hz 4.99, rate_precision_hz 0.198,
   extent_cents 100.1 against a true peak-to-peak of 100.

Keep the interpreting separate from the measuring; this file only measures.
"""

import math
import struct
import tempfile
import wave
from pathlib import Path

from singing import analyze_singing

SR = 22050
VIB_RATE_HZ = 5.0       # squarely inside the declared 3.5-8.0 Hz band
VIB_DEPTH_CENTS = 50.0  # +/-50 cents -> 100 cents true peak-to-peak

# (freq Hz, duration s, vibrato?) -- range C4..A4 = 9 semitones, 4 notes,
# so is_melodic passes and execution actually reaches the vibrato step.
PHRASE = [
    (261.63, 0.60, False),  # C4
    (329.63, 0.60, False),  # E4
    (392.00, 5.00, True),   # G4  <- the note under test
    (440.00, 0.60, False),  # A4
]
GAP_S = 0.12


def render(path):
    """Stdlib-only synthesis: per-sample phase accumulation, a few
    harmonics so YIN sees a proper periodic signal, short fades so note
    edges segment cleanly. Intentionally shares no code with the numpy
    helpers in test_singing.py -- see docstring point 1."""
    samples = []
    for freq, dur, vib in PHRASE:
        n = int(dur * SR)
        phase = 0.0
        for i in range(n):
            t = i / SR
            f = freq
            if vib:
                # sinusoidal pitch modulation in cents -> multiplicative in Hz
                cents = VIB_DEPTH_CENTS * math.sin(2 * math.pi * VIB_RATE_HZ * t)
                f = freq * (2.0 ** (cents / 1200.0))
            phase += 2 * math.pi * f / SR
            v = (math.sin(phase)
                 + 0.35 * math.sin(2 * phase)
                 + 0.15 * math.sin(3 * phase))
            env = min(1.0, i / (0.03 * SR), (n - i) / (0.03 * SR))
            samples.append(v * env * 0.30)
        samples.extend([0.0] * int(GAP_S * SR))

    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b''.join(
            struct.pack('<h', max(-32767, min(32767, int(s * 32767))))
            for s in samples
        ))


def main():
    print('--- vibrato regression: 5.00 Hz / +/-50 cents on a 5.0s G4 ---')
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'vib.wav'
        render(p)
        result = analyze_singing(str(p))

    assert result['is_melodic'] is True, \
        f"expected is_melodic True, got {result.get('is_melodic')}"

    vibrato = result.get('vibrato') or []
    print('vibrato entries:', vibrato)
    assert vibrato, 'detector did NOT fire on a signal built to its own spec'

    g4 = [v for v in vibrato if v['note_name'] == 'G4']
    assert g4, f'expected the vibrato entry on G4, got {vibrato}'
    v = g4[0]

    # 25 cycles of ground truth support a tight rate claim. Observed 4.99
    # on first run; +/-0.3 leaves headroom without letting drift through.
    assert 4.7 <= v['rate_hz'] <= 5.3, \
        f"expected rate_hz within 5.0 +/- 0.3, got {v['rate_hz']}"

    # True modulation is +/-50 cents = 100 cents peak-to-peak. YIN's
    # analysis window low-passes fast modulation, so raw underestimates
    # and the calibrated figure should recover it (observed: raw 68.3,
    # calibrated 100.1).
    assert 30.0 <= v['extent_cents_raw'] <= 100.0, \
        f"expected extent_cents_raw in [30, 100], got {v['extent_cents_raw']}"
    assert 75.0 <= v['extent_cents'] <= 125.0, \
        f"expected calibrated extent_cents in [75, 125], got {v['extent_cents']}"
    assert v['extent_cents'] > v['extent_cents_raw'], \
        'calibration should correct the raw reading UPWARD'

    assert 'rate_precision_hz' in v and v['rate_precision_hz'] > 0, \
        f"expected a positive rate_precision_hz, got {v}"

    print(f"PASS  rate_hz={v['rate_hz']}  extent_raw={v['extent_cents_raw']}"
          f"  extent_calibrated={v['extent_cents']}")


if __name__ == '__main__':
    main()
