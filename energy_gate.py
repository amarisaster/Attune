"""Attune — pure RMS energy gate for chunked/single-shot webhook STT.

Whisper-class STT hallucinates confidently on near-silent audio: fed a
7.6-second giggle-then-silence chunk, it returned fluent, entirely invented
Indonesian rather than an empty string (observed directly against this
server's webhook STT path). Silence carries no linguistic content for the
model to condition on, so it falls back to its training prior instead of
reporting "nothing was said" — and it reports that confabulation with the
same confidence as a real transcription, with no signal a caller can use to
tell the two apart.

The fix is to never ask STT about a chunk (or a whole short clip) that is
below a plausible speech-energy floor in the first place. Silence is not
missing data — it's the absence of speech — so a gated chunk contributes
NOTHING to the transcript: no text, no gap marker (a gap marker means real
content is unknown; a gated chunk's content IS known — there wasn't any).

Pure numpy, no I/O, no ffmpeg, no network — standalone importable (like
stt_stitch.py) so test_energy_gate.py can exercise it without booting the
rest of server.py. Callers decode audio to a float64 mono array in [-1, 1]
(e.g. via singing.load_wav) before calling should_skip_stt.
"""

import numpy as np

# Frame width for the voiced-fraction sub-check below. Short enough that a
# brief loud transient (a laugh, a cough) sitting in an otherwise-silent
# chunk shows up as a small nonzero fraction rather than dragging the whole
# clip's single overall-RMS reading above the gate.
_FRAME_MS = 20.0

# A chunk that clears the overall RMS gate only because of a brief loud
# transient, but is voiced (above the same threshold) for less than this
# fraction of its frames, is still treated as silence -- the transient alone
# isn't enough spoken content to trust STT's output over.
_MIN_VOICED_FRACTION = 0.02

_SILENCE_FLOOR_DBFS = -120.0  # reported for a literally-zero signal


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level of `x` (a float array in [-1, 1]) in dBFS. A literal-zero
    (or empty) signal reports _SILENCE_FLOOR_DBFS rather than -inf."""
    if x is None or len(x) == 0:
        return _SILENCE_FLOOR_DBFS
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    if rms <= 1e-9:
        return _SILENCE_FLOOR_DBFS
    return 20.0 * np.log10(rms)


def _frame_rms_dbfs(x: np.ndarray, sr: int, frame_ms: float = _FRAME_MS) -> np.ndarray:
    hop = max(1, int(sr * frame_ms / 1000.0))
    n = max(0, len(x) // hop)
    out = np.full(n, _SILENCE_FLOOR_DBFS)
    for i in range(n):
        out[i] = rms_dbfs(x[i * hop:(i + 1) * hop])
    return out


def voiced_fraction(x: np.ndarray, sr: int, threshold_dbfs: float,
                     frame_ms: float = _FRAME_MS) -> float:
    """Fraction of `frame_ms`-wide frames whose RMS clears `threshold_dbfs`.
    0.0 for an empty/too-short array -- treated as "no voiced content", the
    same conservative direction as every other empty-input case here."""
    frames = _frame_rms_dbfs(x, sr, frame_ms)
    if len(frames) == 0:
        return 0.0
    return float(np.mean(frames > threshold_dbfs))


def should_skip_stt(x: np.ndarray, sr: int, threshold_dbfs: float = -50.0,
                     min_voiced_fraction: float = _MIN_VOICED_FRACTION) -> tuple:
    """Gate decision for a decoded audio array. Returns (skip: bool,
    reason: str) -- reason is '' when skip is False.

    Skips (returns True) when EITHER:
      - the clip's overall RMS is below `threshold_dbfs` (ATTUNE_STT_SILENCE_DBFS,
        default -50.0 dBFS) -- the whole clip reads as silence/near-silence, or
      - the fraction of frames that individually clear that same threshold is
        below `min_voiced_fraction` -- the clip is silence with at most a
        brief transient in it (a giggle, a cough), not sustained speech.

    Never raises: callers own decode failures (this function assumes `x` is
    already a valid decoded array)."""
    overall = rms_dbfs(x)
    if overall < threshold_dbfs:
        return True, f'overall {overall:.1f} dBFS < gate {threshold_dbfs:.1f} dBFS'
    vfrac = voiced_fraction(x, sr, threshold_dbfs)
    if vfrac < min_voiced_fraction:
        return True, f'voiced fraction {vfrac:.3f} < {min_voiced_fraction} (threshold {threshold_dbfs:.1f} dBFS)'
    return False, ''
