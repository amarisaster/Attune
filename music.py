"""Music perception for Attune — harmony, rhythm, energy, structure.

Pure numpy + stdlib, same constraint and for the same reason as singing.py:
Smart App Control on this machine blocks unsigned native DLLs, so no librosa,
no scipy, no essentia. ffmpeg is a subprocess (via singing.load_wav).

Design rule inherited from the Attune README: numbers, not diagnoses.
Everything reported is a measurement with a confidence label. No mood words —
mode, tempo, energy and register are the measurable proxies; interpretation
belongs to whoever reads the card.

Analysis floor to be honest about: chroma is folded from 4096-point STFT
frames at 22050 Hz (~5.4 Hz bin spacing), which resolves semitones reliably
from about C2 (65 Hz) up. Content below that contributes little to harmony
estimates, which is acceptable for key/chords (bass doubles the harmony) but
is stated here rather than hidden.

Standalone-importable: never imports server. Reuses singing.load_wav for
decode so every clip goes through the same ffmpeg path and 480 s cap.
"""

from __future__ import annotations

import math

import numpy as np

from singing import load_wav  # pure module; ffmpeg subprocess decode, 480 s cap

# ── Constants ────────────────────────────────────────────────────────────────

SR = 22050
FRAME = 4096
HOP = 1024

# Chroma fold range: C2..C7. Below C2 the 5.4 Hz bins smear semitones.
CHROMA_FMIN = 65.406  # C2
CHROMA_FMAX = 2093.0  # C7

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key profiles (same family singing.guess_key uses on
# note events; duplicated here because the input is chroma, not notes).
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Chord templates: root-position triads, binary with a light root emphasis.
# 24 chords (12 maj + 12 min) plus an explicit no-chord template so noise has
# somewhere honest to land.
_MAJ_TEMPLATE = np.array([1.2, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0])
_MIN_TEMPLATE = np.array([1.2, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0])

TEMPO_MIN_BPM = 60.0
TEMPO_MAX_BPM = 200.0

# ── Shared STFT ──────────────────────────────────────────────────────────────


def stft_mags(x: np.ndarray, sr: int = SR, frame: int = FRAME, hop: int = HOP):
    """Windowed rFFT magnitudes.

    Returns (mags [n_frames, n_bins], freqs [n_bins], times [n_frames]).
    """
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n_frames = 1 + (len(x) - frame) // hop
    window = np.hanning(frame).astype(np.float64)
    mags = np.empty((n_frames, frame // 2 + 1), dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + frame] * window
        mags[i] = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(frame, d=1.0 / sr)
    times = (np.arange(n_frames) * hop + frame / 2) / sr
    return mags, freqs, times


def spectral_flatness(mags: np.ndarray) -> float:
    """Geometric/arithmetic mean ratio of the average magnitude spectrum."""
    spec = np.mean(mags, axis=0) + 1e-12
    return float(np.exp(np.mean(np.log(spec))) / np.mean(spec))


# ── Chromagram ───────────────────────────────────────────────────────────────


def chromagram(mags: np.ndarray, freqs: np.ndarray,
               fmin: float = CHROMA_FMIN, fmax: float = CHROMA_FMAX) -> np.ndarray:
    """Fold STFT magnitudes into a 12 x n_frames pitch-class energy matrix.

    Each semitone from fmin..fmax gets a triangular weighting of the FFT bins
    within +/-0.5 semitone of its center; energy accumulates into the
    semitone's pitch class. Frames are L2-normalized (zero frames left zero).
    """
    n_semitones = int(round(12 * math.log2(fmax / fmin))) + 1
    chroma = np.zeros((12, mags.shape[0]), dtype=np.float64)
    for s in range(n_semitones):
        f_center = fmin * (2.0 ** (s / 12.0))
        if f_center > fmax:
            break
        f_lo = f_center * (2.0 ** (-0.5 / 12.0))
        f_hi = f_center * (2.0 ** (0.5 / 12.0))
        sel = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
        if len(sel) == 0:
            continue
        # Triangular weight peaking at f_center.
        w = 1.0 - np.abs(freqs[sel] - f_center) / max(f_center - f_lo, 1e-9)
        w = np.clip(w, 0.0, 1.0)
        pc = (s + int(round(12 * math.log2(fmin / 261.626)))) % 12  # C-aligned
        chroma[pc] += mags[:, sel] @ w
    norms = np.linalg.norm(chroma, axis=0)
    nz = norms > 1e-9
    chroma[:, nz] /= norms[nz]
    return chroma


# ── Key estimation ───────────────────────────────────────────────────────────


def _correlate_profile(hist: np.ndarray, profile: np.ndarray) -> float:
    h = hist - hist.mean()
    p = profile - profile.mean()
    denom = np.linalg.norm(h) * np.linalg.norm(p)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(h, p) / denom)


def estimate_key(chroma: np.ndarray) -> dict:
    """Krumhansl-Schmuckler over the time-summed chromagram.

    Same output shape and confidence labels as singing.guess_key:
    {key, correlation, confidence: 'strong'|'weak'}.
    """
    hist = chroma.sum(axis=1)
    if hist.sum() < 1e-9:
        return {'key': None, 'correlation': 0.0, 'confidence': 'weak'}
    scores = []
    for tonic in range(12):
        rolled = np.roll(hist, -tonic)
        scores.append((f'{NOTE_NAMES[tonic]} major', _correlate_profile(rolled, _KS_MAJOR)))
        scores.append((f'{NOTE_NAMES[tonic]} minor', _correlate_profile(rolled, _KS_MINOR)))
    scores.sort(key=lambda kv: kv[1], reverse=True)
    best_key, best_corr = scores[0]
    margin = best_corr - scores[1][1]
    confidence = 'strong' if (best_corr > 0.6 and margin > 0.1) else 'weak'
    return {'key': best_key, 'correlation': round(best_corr, 3), 'confidence': confidence}


# ── Chord detection ──────────────────────────────────────────────────────────


def _chord_label(index: int) -> str:
    root = NOTE_NAMES[index % 12]
    return root if index < 12 else root + 'm'


def detect_chords(chroma: np.ndarray, times: np.ndarray,
                  window_s: float = 0.5) -> dict:
    """Template-match maj/min triads over ~window_s chroma averages.

    Returns {'chords': [{start_s, end_s, chord, score}], 'progression': [...],
             'coverage': float, 'confidence': 'low'|'medium'|'high'}.

    Confidence is driven by the mean winning margin over the runner-up —
    calibrated against noise in test_music.py so random input stays 'low'.
    """
    n_frames = chroma.shape[1]
    if n_frames == 0:
        return {'chords': [], 'progression': [], 'coverage': 0.0, 'confidence': 'low'}
    frame_dt = times[1] - times[0] if len(times) > 1 else HOP / SR
    win = max(1, int(round(window_s / frame_dt)))

    # 25 templates: 12 maj, 12 min, 1 no-chord (uniform).
    templates = np.zeros((25, 12))
    for r in range(12):
        templates[r] = np.roll(_MAJ_TEMPLATE, r)
        templates[12 + r] = np.roll(_MIN_TEMPLATE, r)
    templates[24] = np.ones(12) * 0.5
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)

    labels, margins, starts = [], [], []
    for i in range(0, n_frames, win):
        seg = chroma[:, i:i + win].mean(axis=1)
        norm = np.linalg.norm(seg)
        if norm < 1e-9:
            labels.append(24)
            margins.append(0.0)
        else:
            sims = templates @ (seg / norm)
            order = np.argsort(sims)[::-1]
            labels.append(int(order[0]))
            margins.append(float(sims[order[0]] - sims[order[1]]))
        starts.append(times[i] if i < len(times) else i * frame_dt)

    # Median filter (width 3) to kill single-window flickers.
    smoothed = list(labels)
    for i in range(1, len(labels) - 1):
        trio = sorted([labels[i - 1], labels[i], labels[i + 1]])
        smoothed[i] = trio[1]

    # Merge runs.
    runs = []
    for i, lab in enumerate(smoothed):
        t0 = starts[i]
        t1 = starts[i + 1] if i + 1 < len(starts) else times[-1] if len(times) else t0 + window_s
        if runs and runs[-1]['_label'] == lab:
            runs[-1]['end_s'] = t1
        else:
            runs.append({'_label': lab, 'start_s': t0, 'end_s': t1})

    chords = []
    for r in runs:
        if r['_label'] == 24:
            continue
        chords.append({
            'start_s': round(r['start_s'], 2),
            'end_s': round(r['end_s'], 2),
            'chord': _chord_label(r['_label']),
        })

    chord_windows = sum(1 for lab in smoothed if lab != 24)
    coverage = chord_windows / len(smoothed) if smoothed else 0.0
    mean_margin = float(np.mean([m for lab, m in zip(smoothed, margins) if lab != 24])) \
        if chord_windows else 0.0

    if coverage > 0.5 and mean_margin > 0.10:
        confidence = 'high' if mean_margin > 0.18 else 'medium'
    else:
        confidence = 'low'

    # Progression: most frequent contiguous 4-run label cycle, else run order.
    run_labels = [c['chord'] for c in chords]
    progression: list = []
    if len(run_labels) >= 4:
        grams: dict = {}
        for i in range(len(run_labels) - 3):
            g = tuple(run_labels[i:i + 4])
            grams[g] = grams.get(g, 0) + 1
        best = max(grams.items(), key=lambda kv: kv[1])
        progression = list(best[0]) if best[1] > 1 else run_labels[:4]
    else:
        progression = run_labels

    return {'chords': chords, 'progression': progression,
            'coverage': round(coverage, 2), 'confidence': confidence}


# ── Tempo & beat grid ────────────────────────────────────────────────────────


def _onset_envelope(x: np.ndarray, sr: int, frame: int = 1024, hop: int = 512):
    """Positive spectral flux — same approach as singing.tempo_candidates."""
    if len(x) < frame:
        return np.zeros(1), hop / sr
    n_frames = 1 + (len(x) - frame) // hop
    window = np.hanning(frame)
    prev = None
    flux = np.zeros(n_frames)
    for i in range(n_frames):
        mag = np.abs(np.fft.rfft(x[i * hop: i * hop + frame] * window))
        if prev is not None:
            d = mag - prev
            flux[i] = np.sum(d[d > 0])
        prev = mag
    if flux.max() > 0:
        flux = flux / flux.max()
    return flux, hop / sr


def estimate_tempo(x: np.ndarray, sr: int = SR) -> dict:
    """Autocorrelation tempo with octave-error comb check and a beat grid.

    Unlike singing.tempo_candidates (always 'low' — correct for rubato a
    cappella), music with a steady onset pattern can earn 'medium'/'high':
    sharpness of the comb-boosted autocorrelation peak plus beat-grid
    consistency decide the tier.

    Returns {bpm, confidence, beat_times}.
    """
    flux, dt = _onset_envelope(x, sr)
    if len(flux) < 8:
        return {'bpm': None, 'confidence': 'low', 'beat_times': []}

    f = flux - flux.mean()
    ac = np.correlate(f, f, mode='full')[len(f) - 1:]
    if ac[0] > 0:
        ac = ac / ac[0]

    lag_min = max(1, int(round(60.0 / TEMPO_MAX_BPM / dt)))
    lag_max = min(len(ac) - 1, int(round(60.0 / TEMPO_MIN_BPM / dt)))
    if lag_max <= lag_min:
        return {'bpm': None, 'confidence': 'low', 'beat_times': []}

    # Comb score: the true period is the SHORTEST lag supported at its own
    # multiples. Support at 2x/3x the lag confirms a fundamental; a strong
    # HALF-lag means this lag is a doubled period (octave error) — penalize at
    # full weight, because quantization inflates the double period (a
    # fractional true period splits its energy across two bins while its
    # double lands on an integer and scores far higher). Neighborhood-max
    # lookups (+/-1 bin) keep multiples from missing fractional peaks.
    def _nb(i: int) -> float:
        lo, hi = max(0, i - 1), min(len(ac), i + 2)
        return float(np.max(ac[lo:hi])) if hi > lo else 0.0

    comb = np.full(len(ac), -np.inf)
    for lag in range(lag_min, lag_max + 1):
        s = _nb(lag)
        if 2 * lag < len(ac):
            s += 0.5 * _nb(2 * lag)
        if 3 * lag < len(ac):
            s += 0.33 * _nb(3 * lag)
        half = lag // 2
        if half >= 1:
            s -= 1.0 * max(0.0, _nb(half))
        comb[lag] = s
    best_lag = int(np.argmax(comb[lag_min:lag_max + 1])) + lag_min
    # Parabolic refinement on the raw autocorrelation: a fractional true
    # period splits energy across bins, so the integer lag alone misreads BPM.
    period_f = float(best_lag)
    if 1 <= best_lag < len(ac) - 1:
        y1, y2, y3 = ac[best_lag - 1], ac[best_lag], ac[best_lag + 1]
        denom = y1 - 2 * y2 + y3
        if abs(denom) > 1e-12:
            delta = 0.5 * (y1 - y3) / denom
            if -1.0 < delta < 1.0:
                period_f = best_lag + delta
    bpm = 60.0 / (period_f * dt)

    # Raw periodicity strength at the chosen lag (neighborhood max — the
    # fractional-period split means the exact bin understates it).
    peak_ac = _nb(best_lag)

    # Beat grid: choose the phase whose impulse train best matches the flux.
    # Built on the refined float period so the grid doesn't drift over long
    # clips the way an integer period would.
    period_i = max(1, best_lag)
    best_phase, best_energy = 0, -1.0
    for phase in range(period_i):
        idx = np.round(np.arange(phase, len(flux) - 0.5, period_f)).astype(int)
        idx = idx[idx < len(flux)]
        e = float(flux[idx].sum())
        if e > best_energy:
            best_energy, best_phase = e, phase
    grid = np.round(np.arange(best_phase, len(flux) - 0.5, period_f)).astype(int)
    grid = grid[grid < len(flux)]
    # Snap each grid point to the local flux max within +/-10% of the period.
    snap = max(1, int(period_f * 0.1))
    beat_frames = []
    for g in grid:
        lo, hi = max(0, g - snap), min(len(flux), g + snap + 1)
        beat_frames.append(lo + int(np.argmax(flux[lo:hi])))
    beat_times = [round(bf * dt, 3) for bf in beat_frames]

    # Consistency: how much of the onset energy sits on the grid.
    on_grid = float(flux[beat_frames].sum())
    total = float(flux.sum()) + 1e-9
    grid_share = on_grid / total
    expected_share = len(beat_frames) / len(flux)
    consistency = grid_share / (expected_share + 1e-9)

    # Tiers calibrated in test_music.py: synthetic clicks measure peak_ac
    # 0.5-0.96 with snapped-grid consistency well above 1.5; noise clips top
    # out near peak_ac 0.15 / consistency 1.1.
    if peak_ac >= 0.35 and consistency > 1.5:
        confidence = 'high'
    elif peak_ac >= 0.20 and consistency > 1.2:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {'bpm': round(bpm, 1), 'confidence': confidence, 'beat_times': beat_times}


# ── Energy arc ───────────────────────────────────────────────────────────────


def _rms_db_envelope(x: np.ndarray, sr: int, win_s: float = 0.05):
    win = max(1, int(sr * win_s))
    n = len(x) // win
    if n == 0:
        return np.array([-60.0]), win / sr
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    return np.maximum(db, -60.0), win / sr


def energy_arc(x: np.ndarray, sr: int = SR, section_s: float = 10.0) -> dict:
    """Loudness over time: start/end levels, extremes, ~10 s section means."""
    db, dt = _rms_db_envelope(x, sr)
    n_head = max(1, int(5.0 / dt))
    start_db = float(np.mean(db[:n_head]))
    end_db = float(np.mean(db[-n_head:]))
    # Smooth before locating extremes so a single transient isn't "loudest".
    k = max(1, int(0.5 / dt))
    kernel = np.ones(k) / k
    smooth = np.convolve(db, kernel, mode='same')
    loudest_i = int(np.argmax(smooth))
    quietest_i = int(np.argmin(smooth))
    sec = max(1, int(section_s / dt))
    sections = []
    for i in range(0, len(db), sec):
        seg = db[i:i + sec]
        sections.append({'t0': round(i * dt, 1),
                         't1': round(min((i + sec) * dt, len(db) * dt), 1),
                         'db': round(float(seg.mean()), 1)})
    return {
        'start_db': round(start_db, 1), 'end_db': round(end_db, 1),
        'loudest_t': round(loudest_i * dt, 1), 'loudest_db': round(float(smooth[loudest_i]), 1),
        'quietest_t': round(quietest_i * dt, 1), 'quietest_db': round(float(smooth[quietest_i]), 1),
        'sections': sections,
    }


# ── Structure ────────────────────────────────────────────────────────────────


def detect_sections(chroma: np.ndarray, x: np.ndarray, sr: int = SR,
                    times: np.ndarray | None = None,
                    max_sections: int = 6) -> list:
    """Novelty-based boundary candidates from chroma + loudness self-similarity.

    Labels are only 'section change ~t' — never 'chorus' (numbers, not
    diagnoses). Returns a list of boundary times in seconds, at most
    max_sections, each at least 20 s from its neighbours.
    """
    n_frames = chroma.shape[1]
    if n_frames < 8:
        return []
    frame_dt = (times[1] - times[0]) if times is not None and len(times) > 1 else HOP / SR
    # Downsample features to ~1 s hops.
    step = max(1, int(round(1.0 / frame_dt)))
    feats = []
    db, db_dt = _rms_db_envelope(x, sr)
    for i in range(0, n_frames, step):
        c = chroma[:, i:i + step].mean(axis=1)
        t = i * frame_dt
        j = min(len(db) - 1, int(t / db_dt))
        feats.append(np.concatenate([c, [(db[j] + 60.0) / 60.0]]))
    F = np.array(feats)
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    F = F / np.maximum(norms, 1e-9)
    S = F @ F.T  # cosine self-similarity

    # Checkerboard novelty along the diagonal, kernel ~16 s.
    k = min(16, len(F) // 2)
    if k < 2:
        return []
    kernel = np.ones((2 * k, 2 * k))
    kernel[:k, k:] = -1
    kernel[k:, :k] = -1
    novelty = np.zeros(len(F))
    for i in range(k, len(F) - k):
        novelty[i] = float(np.sum(S[i - k:i + k, i - k:i + k] * kernel))
    if novelty.max() > novelty.min():
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min())

    # Peak-pick: local maxima above 0.5, >=20 s apart, strongest first.
    candidates = []
    for i in range(1, len(novelty) - 1):
        if novelty[i] > 0.5 and novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]:
            candidates.append((novelty[i], i))
    candidates.sort(reverse=True)
    chosen: list = []
    for _, i in candidates:
        t = i * 1.0  # ~1 s hops
        if all(abs(t - c) >= 20.0 for c in chosen):
            chosen.append(t)
        if len(chosen) >= max_sections:
            break
    return sorted(round(t, 1) for t in chosen)


# ── Top-level analysis ───────────────────────────────────────────────────────


def analyze_music(path: str) -> dict:
    """Decode once, run harmony/rhythm/energy/structure. Free-form dict out."""
    x, sr = load_wav(path)
    x = x.astype(np.float64)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))
    duration_s = len(x) / sr

    mags, freqs, times = stft_mags(x, sr)
    chroma = chromagram(mags, freqs)

    return {
        'duration_s': round(duration_s, 2),
        'key': estimate_key(chroma),
        'chords': detect_chords(chroma, times),
        'tempo': estimate_tempo(x, sr),
        'energy': energy_arc(x, sr),
        'section_changes': detect_sections(chroma, x, sr, times),
        'spectral_flatness': round(spectral_flatness(mags), 3),
    }


# ── Voice vs track comparison ────────────────────────────────────────────────

_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


def compare_voice_to_track(singing_result: dict, music_result: dict) -> dict:
    """Compare a singing.analyze_singing result (on the vocal stem) against a
    music.analyze_music result (on the instrumental stem).

    Only measures what both sides actually provide; missing inputs produce
    missing keys, never guesses.
    """
    out: dict = {}

    key_info = music_result.get('key') or {}
    key_name = key_info.get('key')
    notes = singing_result.get('notes') or []
    if key_name and notes:
        tonic_name, quality = key_name.split(' ', 1)
        tonic = NOTE_NAMES.index(tonic_name)
        scale = _MAJOR_SCALE if quality == 'major' else _MINOR_SCALE
        total_dur = sum(n.get('dur_s', 0.0) for n in notes) or 1e-9
        in_key_dur = sum(n.get('dur_s', 0.0) for n in notes
                         if (n.get('midi', 0) - tonic) % 12 in scale)
        out['in_key_fraction'] = round(in_key_dur / total_dur, 2)
        out['track_key'] = key_name
        out['track_key_confidence'] = key_info.get('confidence', 'weak')

    beat_times = (music_result.get('tempo') or {}).get('beat_times') or []
    if beat_times and notes:
        grid = np.array(beat_times)
        offsets = []
        for n in notes:
            t = n.get('start_s')
            if t is None:
                continue
            j = int(np.argmin(np.abs(grid - t)))
            offsets.append((t - grid[j]) * 1000.0)
        if offsets:
            median_off = float(np.median(offsets))
            out['timing_median_ms'] = round(median_off, 0)
            out['timing_feel'] = ('on' if abs(median_off) <= 35
                                  else 'behind' if median_off > 0 else 'ahead')
            out['tempo_confidence'] = (music_result.get('tempo') or {}).get('confidence', 'low')

    v_dyn = singing_result.get('dynamics') or {}
    t_energy = music_result.get('energy') or {}
    if v_dyn and t_energy:
        v_delta = (v_dyn.get('end_db') or 0) - (v_dyn.get('start_db') or 0)
        t_delta = (t_energy.get('end_db') or 0) - (t_energy.get('start_db') or 0)
        out['voice_build_db'] = round(float(v_delta), 1)
        out['track_build_db'] = round(float(t_delta), 1)

    return out


# ── Card rendering ───────────────────────────────────────────────────────────


def _fmt_t(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f'{m}:{s:02d}'


def format_music_section(result: dict, melody: dict | None = None,
                         stems_urls: dict | None = None) -> str:
    """Render the MUSIC card section. Numbers, confidence labels, source tags.

    No mood vocabulary anywhere in this function, deliberately.
    """
    lines = [f"MUSIC: {_fmt_t(result.get('duration_s', 0))} analyzed"]

    key = result.get('key') or {}
    chords = result.get('chords') or {}
    if key.get('key'):
        agree = ''
        prog = chords.get('progression') or []
        if prog and chords.get('confidence') in ('medium', 'high'):
            tonic = key['key'].split(' ')[0]
            minor = key['key'].endswith('minor')
            tonic_chord = tonic + ('m' if minor else '')
            agree = (' (chord evidence agrees)' if tonic_chord in prog
                     else f" (chords lean {'/'.join(prog[:2])} — ambiguous)")
        lines.append(f"KEY  : {key['key']} [{key['confidence']}]{agree}")

    tempo = result.get('tempo') or {}
    if tempo.get('bpm'):
        grid_note = ' — steady grid' if tempo['confidence'] == 'high' else ''
        lines.append(f"TEMPO: {tempo['bpm']} BPM [{tempo['confidence']}]{grid_note}")

    prog = chords.get('progression') or []
    if prog:
        cov = chords.get('coverage', 0.0)
        lines.append(f"CHORDS: {' → '.join(prog)} ({int(cov * 100)}% coverage) "
                     f"[{chords.get('confidence', 'low')}]")

    if melody and melody.get('summary'):
        lines.append(f"MELODY: {melody['summary']} [basic-pitch]")

    energy = result.get('energy') or {}
    if energy:
        delta = energy.get('loudest_db', 0) - energy.get('start_db', 0)
        lines.append(
            f"ENERGY: starts {energy.get('start_db')} dB, "
            f"peaks {'+' if delta >= 0 else ''}{round(delta, 1)} dB at {_fmt_t(energy.get('loudest_t', 0))}, "
            f"ends {energy.get('end_db')} dB")

    changes = result.get('section_changes') or []
    if changes:
        lines.append('SECTIONS: changes near ' + ', '.join(_fmt_t(t) for t in changes))

    if stems_urls:
        pairs = '  '.join(f'{name}: {url}' for name, url in stems_urls.items())
        lines.append(f'STEMS : {pairs} (kept until ~200 newer drops arrive)')

    return '\n'.join(lines)


def format_compare_section(cmp: dict) -> str:
    """Render the COMPARE (voice vs track) card section."""
    if not cmp:
        return 'COMPARE: not enough shared evidence to compare (needs notes + track analysis)'
    lines = ['COMPARE (voice vs track):']
    if 'in_key_fraction' in cmp:
        lines.append(
            f"PITCH : {int(cmp['in_key_fraction'] * 100)}% of sung note time in "
            f"{cmp['track_key']} [{cmp['track_key_confidence']}]")
    if 'timing_median_ms' in cmp:
        off = cmp['timing_median_ms']
        feel = {'on': 'on the beat', 'behind': 'behind the beat',
                'ahead': 'ahead of the beat'}[cmp['timing_feel']]
        lines.append(
            f"TIMING: onsets median {abs(off):.0f}ms {cmp['timing_feel']} — {feel} "
            f"[tempo confidence: {cmp.get('tempo_confidence', 'low')}]")
    if 'voice_build_db' in cmp:
        lines.append(
            f"ARC   : voice moves {cmp['voice_build_db']:+.1f} dB start→end while "
            f"track moves {cmp['track_build_db']:+.1f} dB")
    return '\n'.join(lines)
