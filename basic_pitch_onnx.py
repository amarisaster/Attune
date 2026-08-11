"""Melody transcription for Attune via Spotify's basic-pitch ONNX model.

Runs the `nmp` (ICASSP 2022) checkpoint through onnxruntime — the one
model-inference path Smart App Control permits on this machine (Microsoft-
signed DLLs; torch/librosa are blocked, see singing.py's module docstring).

The model is polyphonic: on a full mix it emits many simultaneous note
events. The honesty rule from the Attune README applies — downstream cards
get summarize_melody()'s compressed description, never the raw event dump,
and every card line derived from this module is tagged [basic-pitch].

Model facts (verified empirically against the pinned file, 2026-08-11):
  input  serving_default_input_2:0  [batch, 43844, 1]  — 22050 Hz mono window
  output StatefulPartitionedCall:1  [172, 88]  — note posteriorgram (sustained)
  output StatefulPartitionedCall:2  [172, 88]  — onset posteriorgram (impulsive)
  output StatefulPartitionedCall:0  [172, 264] — pitch contour (3 bins/semitone)
Window hop follows upstream basic_pitch: 30-frame overlap (7680 samples),
frames at 22050/256 ~= 86.1 fps, pitch index 0 = MIDI 21 (A0).

Standalone-importable; never imports server. Model download:
scripts/get-models.py (pinned URL + SHA256).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILENAME = 'basic-pitch-nmp.onnx'

SR = 22050
FFT_HOP = 256
AUDIO_N_SAMPLES = 43844          # 2 * SR - FFT_HOP
N_OVERLAP_FRAMES = 30
OVERLAP_SAMPLES = N_OVERLAP_FRAMES * FFT_HOP          # 7680
WINDOW_HOP = AUDIO_N_SAMPLES - OVERLAP_SAMPLES        # 36164
FRAMES_PER_WINDOW = 172
FPS = SR / FFT_HOP                                    # ~86.13 frames/second
MIDI_OFFSET = 21                                      # pitch index 0 = A0

INPUT_NAME = 'serving_default_input_2:0'
NOTE_OUTPUT = 'StatefulPartitionedCall:1'
ONSET_OUTPUT = 'StatefulPartitionedCall:2'

# Note-event extraction defaults, mirroring upstream basic_pitch inference.
ONSET_THRESH = 0.5
FRAME_THRESH = 0.3
MIN_NOTE_FRAMES = 11              # ~128 ms
ENERGY_GAP_FRAMES = 11            # tolerated sub-threshold gap inside a note

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

_session = None


def models_dir() -> Path:
    override = os.environ.get('ATTUNE_MODELS_DIR', '').strip()
    return Path(override) if override else BASE_DIR / 'models'


def model_path() -> Path:
    return models_dir() / MODEL_FILENAME


def is_available() -> tuple:
    """(usable, reason). Never raises — SAC may block onnxruntime at load."""
    if not model_path().exists():
        return False, f'model missing: {model_path()} (run scripts/get-models.py)'
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:
        return False, f'onnxruntime unavailable: {e}'
    return True, ''


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(str(model_path()),
                                        providers=['CPUExecutionProvider'])
    return _session


def _midi_name(midi: int) -> str:
    return f'{NOTE_NAMES[midi % 12]}{midi // 12 - 1}'


def transcribe(x: np.ndarray, sr: int) -> dict:
    """Note events from mono audio. Input any sample rate; resampled to 22050
    by linear interpolation (adequate for posteriorgram features).

    Returns {'notes': [{start_s, dur_s, midi, note_name, salience}], 'n_notes'}.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n_out),
                      np.arange(len(x)), x).astype(np.float32)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0:
        x = x / peak

    # Upstream pads the start with half the overlap so frame 0 is centered.
    x = np.concatenate([np.zeros(OVERLAP_SAMPLES // 2, dtype=np.float32), x])

    session = _get_session()
    note_rows, onset_rows = [], []
    half_ov = N_OVERLAP_FRAMES // 2
    pos = 0
    first = True
    while pos < len(x):
        window = x[pos:pos + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(window, (0, AUDIO_N_SAMPLES - len(window)))
        out = session.run([NOTE_OUTPUT, ONSET_OUTPUT],
                          {INPUT_NAME: window.reshape(1, -1, 1)})
        note_w, onset_w = out[0][0], out[1][0]
        lo = 0 if first else half_ov
        hi = FRAMES_PER_WINDOW - half_ov
        note_rows.append(note_w[lo:hi])
        onset_rows.append(onset_w[lo:hi])
        first = False
        pos += WINDOW_HOP
    notes_pg = np.concatenate(note_rows, axis=0)
    onsets_pg = np.concatenate(onset_rows, axis=0)

    events = _extract_events(notes_pg, onsets_pg)
    return {'notes': events, 'n_notes': len(events)}


def _extract_events(notes_pg: np.ndarray, onsets_pg: np.ndarray) -> list:
    """Onset-driven note tracking, a simplified port of upstream
    output_to_notes_polyphonic: for each onset local-max above ONSET_THRESH,
    follow the note posteriorgram forward while it stays above FRAME_THRESH
    (tolerating short gaps), require MIN_NOTE_FRAMES."""
    n_frames, n_pitches = notes_pg.shape
    claimed = np.zeros_like(notes_pg, dtype=bool)
    events = []
    # Onset local maxima per pitch column.
    for p in range(n_pitches):
        col = onsets_pg[:, p]
        for i in range(1, n_frames - 1):
            if col[i] < ONSET_THRESH or col[i] < col[i - 1] or col[i] < col[i + 1]:
                continue
            if claimed[i, p]:
                continue
            j, gap = i, 0
            while j < n_frames and gap <= ENERGY_GAP_FRAMES:
                if notes_pg[j, p] >= FRAME_THRESH:
                    gap = 0
                else:
                    gap += 1
                j += 1
            end = j - gap
            if end - i >= MIN_NOTE_FRAMES:
                claimed[i:end, p] = True
                midi = p + MIDI_OFFSET
                events.append({
                    'start_s': round(i / FPS, 3),
                    'dur_s': round((end - i) / FPS, 3),
                    'midi': midi,
                    'note_name': _midi_name(midi),
                    'salience': round(float(notes_pg[i:end, p].mean()), 3),
                })
    events.sort(key=lambda e: (e['start_s'], e['midi']))
    return events


def summarize_melody(notes: list, duration_s: float | None = None) -> dict:
    """Compress polyphonic note events into one honest card line.

    The dominant line is the highest-salience event per half-second slice —
    a proxy for 'the line an ear follows', claimed as such and no more.
    """
    if not notes:
        return {'summary': None, 'n_notes': 0}
    total = len(notes)
    end_t = max(n['start_s'] + n['dur_s'] for n in notes)
    dur = duration_s or end_t

    # Dominant line: best event per 0.5 s slice.
    line = []
    t = 0.0
    while t < end_t:
        active = [n for n in notes if n['start_s'] < t + 0.5 and n['start_s'] + n['dur_s'] > t]
        if active:
            line.append(max(active, key=lambda n: n['salience']))
        t += 0.5
    if not line:
        return {'summary': f'~{total} note events, no dominant line found', 'n_notes': total}

    midis = np.array([n['midi'] for n in line], dtype=np.float64)
    lo, hi = int(midis.min()), int(midis.max())
    center = int(round(float(np.median(midis))))

    # Contour: compare mean pitch across thirds of the dominant line.
    thirds = np.array_split(midis, 3) if len(midis) >= 3 else [midis]
    means = [float(t3.mean()) for t3 in thirds if len(t3)]
    contour = ''
    if len(means) == 3:
        if means[2] - means[0] > 2:
            contour = ', rising overall'
        elif means[0] - means[2] > 2:
            contour = ', falling overall'
        elif max(means) - min(means) > 3:
            contour = ', arcing through the middle'

    reg = (f'register {_midi_name(lo)}–{_midi_name(hi)}'
           if hi - lo > 2 else f'held near {_midi_name(center)}')
    summary = (f'line centers {_midi_name(center)}, {reg}{contour}, '
               f'~{total} note events over {int(round(dur))}s')
    return {'summary': summary, 'n_notes': total,
            'register': [_midi_name(lo), _midi_name(hi)],
            'center': _midi_name(center)}
