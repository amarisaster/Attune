"""Attune — pure-numpy singing analysis.

No native audio deps (no librosa/numba/soundfile/scipy) — Smart App Control
on this machine blocks unsigned DLLs. Decoding goes through ffmpeg as a
subprocess (mirroring vendor/seven-ears/seven_ears_card.py's convert_to_wav),
then the resulting PCM WAV is parsed with stdlib `wave` + numpy. Everything
downstream (YIN pitch tracking, note segmentation, vibrato, dynamics, key,
tempo candidates) is hand-rolled numpy/stdlib — see RESEARCH-singing-perception.md
steps 1-2.

Honesty rules baked in throughout: tempo is always low-confidence for
unaccompanied/rubato singing, key guesses carry a strong/weak label instead
of fake precision, and when the clip looks like it has backing music the
whole result is flagged analysis_mode='singing-with-music' so callers know
the pitch track is "most prominent pitch", not isolated vocals.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np


def _resolve_ffmpeg_dir() -> str:
    """Optional ffmpeg bin directory to prepend to PATH. This module stays
    usable standalone (test_singing.py imports it directly, without going
    through server.py's config section), so it resolves its own value with
    the same precedence: ATTUNE_FFMPEG_DIR env var first (server.py sets
    this from its own config on boot), then attune.config.json beside this
    file, then '' (rely on PATH already having ffmpeg)."""
    env_v = os.environ.get('ATTUNE_FFMPEG_DIR')
    if env_v is not None:
        # Set-but-empty means "rely on PATH" explicitly — do NOT fall through
        # to JSON (unified env-precedence rule, same as server.py).
        v = env_v.strip()
        return str(Path(v).expanduser()) if v else ''
    cfg_path = Path(__file__).resolve().parent / 'attune.config.json'
    try:
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        v = str(cfg.get('ffmpeg_dir') or '').strip()
        return str(Path(v).expanduser()) if v else ''
    except (OSError, ValueError):
        return ''


FFMPEG_DIR = _resolve_ffmpeg_dir()
TARGET_SR = 22050
MAX_ANALYSIS_S = 8 * 60  # cap so a 25-minute clip can't pin a worker thread

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# See yin_f0's boundary-check comment: a small flat Hz allowance for
# parabolic-interpolation grid quantization near fmax, not a percentage slack.
_YIN_BOUNDARY_EPS_HZ = 2.0


def _ffmpeg_env() -> dict:
    env = os.environ.copy()
    if FFMPEG_DIR:
        env['PATH'] = FFMPEG_DIR + os.pathsep + env.get('PATH', '')
    return env


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_wav(path: str, sr: int = TARGET_SR) -> tuple:
    """Decode ANY input via ffmpeg to 16-bit PCM mono WAV at `sr`, then parse
    with stdlib `wave` + numpy. Returns (float64 mono array in [-1, 1], sr).
    Always re-encodes (even if already a matching WAV) so callers never have
    to reason about the source container/codec."""
    ffmpeg = shutil.which('ffmpeg', path=_ffmpeg_env()['PATH']) or 'ffmpeg'
    with tempfile.TemporaryDirectory(prefix='attune_singing_') as td:
        dst = Path(td) / 'audio.wav'
        proc = subprocess.run(
            # -t caps the DECODE itself (placed after -i so it limits output
            # duration) so a multi-hour file never fully decodes -- the array
            # slice below is a backstop, not the primary cap.
            [ffmpeg, '-y', '-v', 'error', '-i', str(path), '-t', str(MAX_ANALYSIS_S),
             '-ar', str(sr), '-ac', '1', '-c:a', 'pcm_s16le', str(dst)],
            capture_output=True, text=True, env=_ffmpeg_env(), timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f'ffmpeg decode failed: {proc.stderr[-500:]}')
        with wave.open(str(dst), 'rb') as w:
            actual_sr = w.getframerate()
            raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return x, actual_sr


# ---------------------------------------------------------------------------
# YIN pitch tracking
# ---------------------------------------------------------------------------

def _yin_difference(sub: np.ndarray, W: int, tau_max: int) -> np.ndarray:
    """Classic YIN difference function d(tau) for tau in [0, tau_max], computed
    over a window of length W using a length-(W+tau_max) buffer `sub`. Cross
    term via FFT cross-correlation, energy terms via cumulative sums — avoids
    an O(W*tau_max) python double loop."""
    n = len(sub)
    N = 1
    while N < 2 * n:
        N *= 2
    w = sub[:W]
    s = sub[:n]
    # cross(tau) = sum_j w[j] * s[j+tau] = ifft(fft(s) * conj(fft(w)))[tau]
    cross = np.fft.ifft(np.fft.fft(s, N) * np.conj(np.fft.fft(w, N))).real[:tau_max + 1]
    sq = sub ** 2
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    term1 = csum[W]  # sum sub[0:W]^2, constant
    taus = np.arange(tau_max + 1)
    term2 = csum[W + taus] - csum[taus]  # sum sub[tau:tau+W]^2
    d = term1 + term2 - 2 * cross
    d = np.maximum(d, 0.0)
    return d


def _parabolic_interp(y: np.ndarray, i: int) -> float:
    """Parabolic interpolation around index i; returns fractional offset."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y0 - y2) / denom


def yin_f0(x: np.ndarray, sr: int, fmin: float = 70, fmax: float = 1000,
           frame: int = 2048, hop: int = 256, threshold: float = 0.15) -> tuple:
    """Pure-numpy YIN. Returns (f0_hz array, confidence array), one value per
    hop. Unvoiced/low-confidence frames are f0=0, confidence=0."""
    tau_min = max(1, int(sr / fmax))
    tau_max = min(frame // 2, int(sr / fmin) + 1)
    n_frames = max(0, 1 + (len(x) - frame - tau_max) // hop)
    f0 = np.zeros(n_frames)
    conf = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        sub = x[start:start + frame + tau_max]
        if len(sub) < frame + tau_max:
            break
        if np.max(np.abs(sub)) < 1e-6:
            continue  # silence
        d = _yin_difference(sub, frame, tau_max)
        # cumulative mean normalized difference
        d_prime = np.ones_like(d)
        running = 0.0
        for tau in range(1, len(d)):
            running += d[tau]
            d_prime[tau] = d[tau] * tau / running if running > 0 else 1.0
        search = d_prime[tau_min:tau_max + 1]
        if len(search) == 0:
            continue
        below = np.where(search < threshold)[0]
        if len(below) == 0:
            # No candidate crossed the absolute CMND threshold anywhere in
            # the search range -- this frame is unvoiced/unreliable. Do NOT
            # fall back to the global minimum; f0 stays 0, conf stays 0.
            continue
        # first local minimum under the absolute threshold
        local_idx = None
        for k in below:
            tau = tau_min + k
            if tau + 1 < len(d_prime) and d_prime[tau] <= d_prime[tau + 1]:
                local_idx = tau
                break
        if local_idx is None:
            # Every candidate that dipped below the absolute threshold was
            # still DESCENDING when the search range ran out -- the CMND
            # curve never turned a corner, so there's no evidence any of
            # them is an actual periodicity minimum rather than the
            # closest-available point on a curve still heading toward a true
            # (out-of-range) period. This shows up concretely for a tone
            # below fmin: its true tau lies past tau_max, so d_prime keeps
            # falling all the way to the tau_max edge and would otherwise be
            # reported as a confident but spurious pitch near fmax's low
            # boundary. Treat the frame as unvoiced rather than guess.
            continue
        best_tau = local_idx
        best_val = d_prime[best_tau]
        # Clamp the interpolation offset -- a noisy/pathological CMND curve
        # can otherwise push the "refined" tau arbitrarily far from the
        # sampled bin it's supposed to be interpolating around.
        frac = float(np.clip(_parabolic_interp(d_prime, best_tau), -1.0, 1.0))
        refined_tau = best_tau + frac
        if refined_tau <= 0:
            continue
        # The interpolation offset can still walk refined_tau outside the
        # search range entirely (best_tau at the tau_min/tau_max edge with a
        # frac pointing further out) -- reject rather than report a period
        # nobody actually searched.
        if refined_tau < tau_min or refined_tau > tau_max:
            continue
        hz = sr / refined_tau
        # Strict fmin/fmax: no percentage slack (the old 0.9x/1.1x band is
        # gone). _YIN_BOUNDARY_EPS_HZ is a small flat allowance, not a
        # design slack -- near fmax the tau grid is coarse (d(hz)/d(tau)
        # grows as sr/tau**2), so even a well-behaved 3-point parabolic fit
        # can land a bin or so short of the true continuous minimum. A tone
        # placed exactly at fmax needs this to be reported as in-range
        # rather than rejected by a few Hz of interpolation quantization.
        if fmin - _YIN_BOUNDARY_EPS_HZ <= hz <= fmax + _YIN_BOUNDARY_EPS_HZ:
            f0[i] = hz
            conf[i] = float(np.clip(1.0 - best_val, 0.0, 1.0))
    return f0, conf


def _median_smooth(f0: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Median-smooth the voiced contour; unvoiced (0) frames stay 0 and don't
    get pulled into neighboring medians."""
    out = f0.copy()
    half = kernel // 2
    n = len(f0)
    for i in range(n):
        if f0[i] == 0:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = f0[lo:hi]
        voiced = window[window > 0]
        if len(voiced):
            out[i] = float(np.median(voiced))
    return out


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def midi_to_note_name(midi: int) -> str:
    octave = midi // 12 - 1
    name = NOTE_NAMES[midi % 12]
    return f'{name}{octave}'


# ---------------------------------------------------------------------------
# Note segmentation
# ---------------------------------------------------------------------------

def _sliding_slope_cents_per_s(times: np.ndarray, cents: np.ndarray, voiced: np.ndarray,
                                window_s: float = 0.2, min_points: int = 5) -> tuple:
    """Local slope (cents/sec) of the voiced pitch contour, via a small
    least-squares linear fit over a `window_s`-wide sliding window centered
    on each voiced frame. Returns (slopes, valid) -- `valid[i]` is False
    when frame i is unvoiced, or when too few voiced neighbors fall inside
    its window to trust a fit (near clip edges / short voiced fragments).
    Uses a two-pointer sweep over the sorted voiced-frame timestamps --
    O(n_voiced) rather than an O(n_voiced^2) window search."""
    n = len(cents)
    slopes = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    voiced_idx = np.where(voiced)[0]
    m = len(voiced_idx)
    if m == 0:
        return slopes, valid
    vt = times[voiced_idx]
    vc = cents[voiced_idx]
    half = window_s / 2.0
    lo = 0
    hi = 0
    for k in range(m):
        t0 = vt[k]
        while lo < m and vt[lo] < t0 - half:
            lo += 1
        while hi < m and vt[hi] <= t0 + half:
            hi += 1
        if hi - lo >= min_points:
            slope = np.polyfit(vt[lo:hi], vc[lo:hi], 1)[0]
            i = voiced_idx[k]
            slopes[i] = float(slope)
            valid[i] = True
    return slopes, valid


def _find_glide_regions(is_glide_frame: np.ndarray, gap_frames: int) -> list:
    """Merge contiguous (with up to `gap_frames` tolerance for interior
    non-glide frames, e.g. a single unvoiced dropout mid-slide) glide-
    flagged frames into inclusive (start_i, end_i) regions."""
    n = len(is_glide_frame)
    regions = []
    i = 0
    while i < n:
        if not is_glide_frame[i]:
            i += 1
            continue
        start = i
        last = i
        j = i + 1
        while j < n:
            if is_glide_frame[j]:
                last = j
                j += 1
                continue
            if (j - last) <= gap_frames:
                j += 1
                continue
            break
        regions.append((start, last))
        i = last + 1
    return regions


def _monotonic_fraction(cents_seq: np.ndarray) -> float:
    """Fraction of consecutive (nonzero) frame-to-frame cents deltas within a
    glide candidate that agree in sign with the region's overall net
    direction. A true glissando moves essentially one way for its whole
    duration. This alone does NOT catch a monotonically-trending staircase
    (a fast ascending scale has ~100% net-direction agreement too) -- it
    exists to reject zigzag/alternating movement; the plateau check below
    is what separates a real glide from a run of discrete steps."""
    if len(cents_seq) < 2:
        return 1.0
    overall = cents_seq[-1] - cents_seq[0]
    if overall == 0:
        return 0.0
    direction = np.sign(overall)
    diffs = np.diff(cents_seq)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 1.0
    return float(np.mean(np.sign(nonzero) == direction))


def _find_plateaus(cents_seq: np.ndarray, times_seq: np.ndarray,
                    band_cents: float, min_dur_s: float) -> list:
    """Find interior spans of `cents_seq` (a glide candidate's actual PITCH
    trace, not its slope) that dwell within a `band_cents`-wide band for at
    least `min_dur_s`. A genuine glide's pitch keeps moving throughout its
    whole span; a fast-note run's pitch instead dwells on each discrete note
    for that note's duration even though the *slope* computed over a wide
    sliding window can stay elevated across the whole run (the window spans
    several step transitions at once, so is_glide_frame alone can't tell a
    fast scale from a real slide). Looking at the pitch trace directly
    catches the dwell that the slope-based flag misses.

    band_cents/min_dur_s are deliberately small (defaults 12 cents / 50ms,
    not a full semitone / a "real" note-length floor) so a genuine glide can
    never trip this by accident: even right at the glide-slope threshold
    (glide_slope_thresh_cents_s, default 300 cents/s), crossing a 12-cent
    band takes ~40ms -- safely under the 50ms floor -- while a truly held
    note's near-zero residual drift sits inside a 12-cent band for far
    longer than 50ms. A wider/looser pair (100 cents / 90ms) was tried first
    and rejected on two counts: (1) it let the C4->C5/C4->E4 true-glide test
    cases (400-600 cents/s) get chopped into fake plateaus, since crossing
    100 cents at that pace takes 167-250ms; and (2) at fast note-run
    tempos (5-6 notes/sec) each note's true flat interior is itself only
    ~70-90ms once YIN's own ~93ms analysis window (frame/sr) smears the
    step transition on either side, so a 90ms floor missed real plateaus
    too. The tighter pair catches the melisma-scale plateaus (their flat
    interior is comfortably >50ms) while keeping enough margin below the
    worst-case glide crossing time not to trip on a real slide.

    Returns a list of (start_pos, end_pos) index pairs into
    cents_seq/times_seq; positions covered by a returned span should be
    excluded from glide candidacy and left for normal note segmentation."""
    n = len(cents_seq)
    plateaus = []
    i = 0
    while i < n:
        run_min = cents_seq[i]
        run_max = cents_seq[i]
        j = i
        while j + 1 < n:
            c = cents_seq[j + 1]
            new_min = min(run_min, c)
            new_max = max(run_max, c)
            if new_max - new_min <= band_cents:
                run_min, run_max = new_min, new_max
                j += 1
            else:
                break
        if times_seq[j] - times_seq[i] >= min_dur_s:
            plateaus.append((i, j))
            i = j + 1
        else:
            i += 1
    return plateaus


def segment_notes(f0: np.ndarray, conf: np.ndarray, sr: int, hop: int,
                   frame: int = 2048, min_dur_s: float = 0.07,
                   note_merge_gap_s: float = 0.08,
                   conf_thresh: float = 0.5, gap_frames: int = 2,
                   hysteresis_cents: float = 70.0, persist_frames: int = 3,
                   glide_slope_thresh_cents_s: float = 300.0,
                   glide_min_dur_s: float = 0.25, glide_window_s: float = 0.2,
                   glide_monotonic_frac_min: float = 0.85,
                   glide_plateau_band_cents: float = 12.0,
                   glide_plateau_min_dur_s: float = 0.05,
                   return_glides: bool = False):
    """Stable-region note segmentation in CONTINUOUS cents space, quantized
    to semitones only after segmentation. A note stays open as long as voiced
    frames sit within `hysteresis_cents` of its center; a brief excursion
    beyond that (e.g. a vibrato swing) is absorbed into the note rather than
    splitting it unless the deviation actually persists for `persist_frames`
    consecutive voiced frames, in which case it's treated as a real note
    change. This keeps ±50+ cent vibrato from alternating between semitones
    or manufacturing fake melodic range.

    The run's center is a ROLLING ROBUST CENTER -- the midpoint of the
    running min/max of cents seen so far in the run -- not an anchor to the
    run's first voiced frame. Anchoring to frame one means a run that
    happens to start near a vibrato peak/trough sees the *rest* of the note
    swing >70 cents to the opposite side, tripping the persist-frame
    transition logic and splitting one sung note into several fringe
    fragments purely because of where in the vibrato cycle it happened to
    start. A running-median center was tried first and rejected: on a
    monotonic quarter-cycle swing (the common case right after a run opens)
    the median of a handful of same-direction samples lags behind and stays
    close to the run's start, reproducing the same false-split bug in a
    different shape. The running-min/max midpoint tracks a symmetric
    oscillation's true center almost immediately (it only needs to have
    seen the two extremes, which a vibrato swing reaches within roughly a
    quarter cycle), so it stays correct regardless of vibrato phase at
    onset, including at deeper ±100-cent modulation depths.

    Frame i is timestamped at its WINDOW CENTER (start + frame/2 samples),
    not its window start, so note boundaries line up with the audio instead
    of landing ~frame/(2*sr) seconds early.
    """
    frame_dur = hop / sr
    frame_offset_s = (frame / 2) / sr
    smoothed = _median_smooth(f0, kernel=5)
    voiced = (smoothed > 0) & (conf >= conf_thresh)
    n = len(smoothed)
    cents = np.zeros(n)
    cents[voiced] = 1200.0 * np.log2(smoothed[voiced] / 440.0)

    # -----------------------------------------------------------------
    # Glide (glissando) detection -- runs BEFORE the plateau/hysteresis
    # note-run logic below. That logic is built for STEPPED pitch (held
    # plateaus separated by brief transitions) and, fed a smooth sustained
    # pitch slide, fabricates a ladder of fake discrete notes purely as an
    # artifact of the hysteresis band being crossed repeatedly on the way
    # up/down (e.g. a 2-second scoop from C4 to C5 gets chopped into a
    # semitone staircase that the singer never actually held). Frames that
    # are part of a genuine sustained slide are pulled out of the voiced
    # set used for note-run building below, so segment_notes reports the
    # plateaus either side of the slide as real notes and the slide itself
    # as a separate 'glide' event instead of more fake notes.
    #
    # The duration gate (glide_min_dur_s, ~250ms) -- not the slope
    # threshold alone -- is what keeps an INSTANTANEOUS step transition
    # between two held notes from being misclassified as a glide: right at
    # a step edge the local slope over the sliding window can be enormous
    # (a whole semitone or more crammed into a couple of frames), but that
    # region is only as wide as the median-smoothing kernel (a few tens of
    # ms) -- far short of glide_min_dur_s -- so it never accumulates into a
    # long-enough run. A sustained slide keeps |slope| pinned above
    # threshold for its whole duration and comfortably clears the gate.
    times = np.arange(n) * frame_dur + frame_offset_s
    slopes, slope_valid = _sliding_slope_cents_per_s(times, cents, voiced, window_s=glide_window_s)
    is_glide_frame = slope_valid & (np.abs(slopes) >= glide_slope_thresh_cents_s)
    glides = []
    voiced_for_notes = voiced.copy()
    # A wide (glide_window_s ~200ms) sliding window used for the slope test
    # above means a run of fast discrete notes (e.g. 5-6 notes/sec) can flag
    # is_glide_frame continuously across MANY step transitions at once -- the
    # window spans several steps together, so its regression slope stays
    # elevated the whole time even though the singer never actually slid.
    # Left unchecked that produces one long "glide" region covering the
    # whole run, and the duration gate alone can't reject it (it's long).
    # Each raw region below is therefore split at any internal PLATEAU (a
    # span where the actual pitch trace, not its slope, dwells within about
    # a semitone -- i.e. a genuinely held note) before the duration and
    # monotonicity checks are applied to what's left. Plateau frames are
    # never removed from voiced_for_notes, so they flow straight back into
    # normal note segmentation below instead of being swallowed by a fake
    # glide. This also structurally protects the <3-notes non-melodic exit
    # in analyze_singing: a fast scale can no longer get misclassified as
    # one giant glide event that starves the note count.
    for start, last in _find_glide_regions(is_glide_frame, gap_frames):
        region_frames = np.arange(start, last + 1)
        region_voiced_frames = region_frames[voiced[region_frames]]
        if len(region_voiced_frames) < 2:
            continue
        region_cents = cents[region_voiced_frames]
        region_times = times[region_voiced_frames]

        plateaus = _find_plateaus(region_cents, region_times,
                                   band_cents=glide_plateau_band_cents,
                                   min_dur_s=glide_plateau_min_dur_s)
        plateau_positions = set()
        for p0, p1 in plateaus:
            plateau_positions.update(range(p0, p1 + 1))
        # A plateau found ANYWHERE in this raw region is direct evidence
        # it's a run of discrete held notes (a melisma), not a single smooth
        # glide or a brief non-glide artifact (e.g. a vibrato-driven slope
        # burst, or the couple-frame smear right at an isolated instant step
        # edge) -- those never dwell long enough near a fixed cents value to
        # register a plateau at all. Only in that discrete-run case are the
        # non-plateau (transitional) frames unconditionally pulled out of
        # voiced_for_notes below; without evidence of a real held-note
        # dwell, a short/non-monotonic sub-span is left untouched (voiced
        # for note segmentation) exactly like before this fix, so it can't
        # eat into genuine note boundaries the way an unconditional
        # exclusion would (this was tried and it clipped frames off the
        # start/end of real notes near ordinary transitions and vibrato
        # onset/offset, shortening them below min_dur_s/vibrato's min_dur_s
        # gates and causing false negatives on unrelated cases).
        region_has_plateau = len(plateaus) > 0

        # Non-plateau positions are the actual glide candidates -- split the
        # region at each plateau so a note held mid-run doesn't get folded
        # into a glide on either side of it.
        sub_spans = []
        cur = None
        for pos in range(len(region_voiced_frames)):
            if pos in plateau_positions:
                if cur is not None:
                    sub_spans.append(cur)
                    cur = None
            else:
                cur = [pos, pos] if cur is None else [cur[0], pos]
        if cur is not None:
            sub_spans.append(cur)

        for pos0, pos1 in sub_spans:
            f0_idx = int(region_voiced_frames[pos0])
            f1_idx = int(region_voiced_frames[pos1])
            if region_has_plateau:
                # Confirmed discrete-note run: this sub-span is transitional
                # smear between two real held notes (or between a held note
                # and the region edge), not reliable steady-pitch data for
                # the plateau/hysteresis note builder below. Exclude it from
                # voiced_for_notes unconditionally -- the note builder then
                # treats the gap like a brief unvoiced dropout (bridged if
                # short via gap_frames, or a clean run boundary if not),
                # which is exactly what correctly separates two adjacent
                # notes instead of letting their smeared transition get
                # silently absorbed into one merged hysteresis run.
                voiced_for_notes[f0_idx:f1_idx + 1] = False
            if pos1 - pos0 < 1:
                continue  # need at least two points to judge direction/duration
            start_s = f0_idx * frame_dur + frame_offset_s
            end_s = (f1_idx + 1) * frame_dur + frame_offset_s
            dur_s_glide = end_s - start_s
            # Duration gate applied per SUB-region (post plateau-split),
            # before any further merging -- a bundle of short inter-note
            # transition edges left over after plateau removal must each
            # individually clear glide_min_dur_s, not benefit from having
            # once been part of a longer raw region.
            if dur_s_glide < glide_min_dur_s:
                continue
            sub_cents = region_cents[pos0:pos1 + 1]
            # Monotonicity gate: a true glissando moves essentially one
            # direction throughout; reject alternating/zigzag movement that
            # slipped past the slope threshold.
            if _monotonic_fraction(sub_cents) < glide_monotonic_frac_min:
                continue
            # Endpoint pitches are read straight off the sub-region's own
            # boundary frames (both are voiced by construction), converted
            # to nearest note name. These are real pitches the singer
            # deliberately passed through, just not held long enough to
            # count as separate notes.
            from_hz = float(smoothed[f0_idx])
            to_hz = float(smoothed[f1_idx])
            if from_hz <= 0 or to_hz <= 0:
                continue
            from_midi = int(round(hz_to_midi(from_hz)))
            to_midi = int(round(hz_to_midi(to_hz)))
            glides.append({
                'type': 'glide',
                'from_note': midi_to_note_name(from_midi),
                'to_note': midi_to_note_name(to_midi),
                'from_midi': from_midi,
                'to_midi': to_midi,
                'start_s': round(start_s, 3),
                'end_s': round(end_s, 3),
                'dur_s': round(dur_s_glide, 3),
            })
            # Reported glide -- excluded here too for the (not
            # region_has_plateau) branch, where the unconditional exclusion
            # above didn't already run.
            voiced_for_notes[f0_idx:f1_idx + 1] = False

    # Pass 1: build runs in continuous-cents space with hysteresis, using a
    # rolling-median center (see docstring) instead of a first-frame anchor.
    # Uses voiced_for_notes (glide frames excluded) so a sustained slide
    # can't seed or extend a fake plateau run.
    runs = []  # (center_cents, i0, i1)
    i = 0
    while i < n:
        if not voiced_for_notes[i]:
            i += 1
            continue
        run_min = cents[i]
        run_max = cents[i]
        center = cents[i]
        last_voiced = i
        j = i + 1
        while j < n:
            if not voiced_for_notes[j]:
                if (j - last_voiced) <= gap_frames:
                    j += 1
                    continue
                break
            diff = cents[j] - center
            if abs(diff) <= hysteresis_cents:
                run_min = min(run_min, cents[j])
                run_max = max(run_max, cents[j])
                center = (run_min + run_max) / 2.0
                last_voiced = j
                j += 1
                continue
            # Candidate transition -- only end the run if the deviation
            # actually persists; a lone outlier or a vibrato peak gets
            # absorbed into the current note instead.
            k = j
            confirmed = 0
            scanned = 0
            while k < n and scanned < persist_frames + gap_frames:
                if voiced_for_notes[k]:
                    scanned += 1
                    if abs(cents[k] - center) > hysteresis_cents:
                        confirmed += 1
                        if confirmed >= persist_frames:
                            break
                    else:
                        confirmed = 0  # dipped back inside the band
                k += 1
            if confirmed >= persist_frames:
                break  # real transition -- close this run before j
            run_min = min(run_min, cents[j])
            run_max = max(run_max, cents[j])
            center = (run_min + run_max) / 2.0
            last_voiced = j
            j += 1
        runs.append((center, i, last_voiced))
        i = last_voiced + 1

    # Discard sub-minimum-duration runs BEFORE the same-semitone merge pass
    # below. A vibrato-phase fringe fragment (e.g. a handful of frames that
    # got kicked into their own run at a run boundary) is too short to ever
    # count as a note on its own, but left in place it sits BETWEEN two
    # longer runs of the true note and blocks them from merging on
    # `(i0 - quantized[-1][2]) <= gap_frames`. Dropping it first lets the
    # merge pass see the two real runs as adjacent.
    def _run_dur_s(i0, i1):
        start_s = i0 * frame_dur + frame_offset_s
        end_s = (i1 + 1) * frame_dur + frame_offset_s
        return end_s - start_s

    runs = [r for r in runs if _run_dur_s(r[1], r[2]) >= min_dur_s]

    # Pass 2: quantize each run's center to a semitone, then merge adjacent
    # runs that land on the same semitone (e.g. split by a longer unvoiced
    # gap than gap_frames tolerates). Uses a WIDER gap tolerance than
    # gap_frames/glide-frame gap handling -- note_merge_gap_frames, derived
    # from note_merge_gap_s (~80ms) rather than the ~23ms gap_frames used
    # for hysteresis/glide-region bridging. Two runs landing on the SAME
    # semitone are, by construction, never a false merge (there's no
    # different-pitch content being glued together), so it's safe to bridge
    # a bigger real-world gap between them -- e.g. a short same-note tail
    # fragment left over near a vibrato phase boundary (a handful of frames
    # a persist-frames confirm split off, separated from the main run by
    # more than gap_frames but still the same note) merges back into the
    # main note instead of surviving as its own too-short-to-filter
    # fragment. min_dur_s was lowered (0.07s, was 0.12s) to admit real but
    # brief plateaus recovered from fast melisma runs, which made exactly
    # this kind of same-note fringe fragment long enough to survive the
    # discard-before-merge filter above on its own; widening the merge gap
    # here reunites it with its note instead.
    note_merge_gap_frames = max(gap_frames, int(round(note_merge_gap_s / frame_dur)))
    quantized = []
    for center, i0, i1 in runs:
        note = int(round(69.0 + center / 100.0))
        if quantized and quantized[-1][0] == note and (i0 - quantized[-1][2]) <= note_merge_gap_frames:
            quantized[-1] = (note, quantized[-1][1], i1)
        else:
            quantized.append((note, i0, i1))

    notes = []
    for note, i0, i1 in quantized:
        start_s = i0 * frame_dur + frame_offset_s
        end_s = (i1 + 1) * frame_dur + frame_offset_s
        dur_s = end_s - start_s
        if dur_s < min_dur_s:
            continue
        seg_hz = smoothed[i0:i1 + 1]
        seg_conf = conf[i0:i1 + 1]
        seg_hz_voiced = seg_hz[seg_hz > 0]
        if len(seg_hz_voiced) == 0:
            continue
        notes.append({
            'note_name': midi_to_note_name(note),
            'midi': int(note),
            'start_s': round(start_s, 3),
            'end_s': round(end_s, 3),
            'dur_s': round(dur_s, 3),
            'mean_hz': round(float(np.mean(seg_hz_voiced)), 2),
            'confidence': round(float(np.mean(seg_conf[seg_conf > 0])) if np.any(seg_conf > 0) else 0.0, 3),
            '_i0': i0,
            '_i1': i1,
        })
    if return_glides:
        return notes, glides
    return notes


# ---------------------------------------------------------------------------
# Vibrato
# ---------------------------------------------------------------------------

# A true vibrato peak should tower over the rest of the 2-12Hz spectrum, not
# just edge out its immediate neighbors -- white jitter and smooth aperiodic
# wander both spread energy fairly evenly across that range instead of
# concentrating it in a narrow band, so a real modulation's peak/rest-median
# ratio comes out far above this even with margin to spare. Empirically
# measured (see test_singing.py debug notes): true-vibrato cases (±30c at
# 5.5/7.9Hz, ±60c at 5Hz across durations/phases, ±100c) land at
# ratio ~78-226; the 40-cent white-jitter and smooth-wander false-positive
# trials that slipped past the earlier ratio=3.0 threshold peaked at
# ratio ~7.6. 15.0 sits with wide margin on both sides.
_VIBRATO_PROMINENCE_RATIO = 15.0
# A genuine sinusoidal vibrato explains most of the detrended contour's
# variance once fit at its own rate; jitter/wander are not well-modeled by a
# single sinusoid and leave most of the variance unexplained even when they
# happen to have a peak somewhere in the 3.5-8Hz band. Empirically: true
# cases land at r2 ~0.995-0.999; jitter/wander false positives that slipped
# past r2=0.3 peaked at r2 ~0.44. 0.65 sits with wide margin on both sides.
_VIBRATO_MIN_R2 = 0.65


def detect_vibrato(f0: np.ndarray, conf: np.ndarray, sr: int, hop: int, i0: int, i1: int,
                    band=(3.5, 8.0), extent_min_cents: float = 15.0,
                    conf_thresh: float = 0.5, min_dur_s: float = 0.9,
                    min_cycles: float = 4.0, frame: int = 2048) -> Optional[dict]:
    frame_rate = sr / hop
    seg = f0[i0:i1 + 1]
    seg_conf = conf[i0:i1 + 1]
    # Mask by confidence, not just f0 > 0 -- a frame that squeaked past f0
    # acceptance but has low confidence shouldn't seed the vibrato trace.
    voiced_mask = (seg > 0) & (seg_conf >= conf_thresh)
    if voiced_mask.sum() < 8:
        return None
    dur_s = len(seg) / frame_rate
    if dur_s < min_dur_s:
        # Require ~1.0s of stable note before trusting a vibrato reading;
        # the 0.9s floor (not a hard 1.0) accounts for YIN's own frame+tau_max
        # lookahead trimming a few frames off a note that sits at the very
        # end of the analyzed audio -- an unavoidable windowing artifact, not
        # slack for short/unstable notes. Below this, omit rather than
        # report bin-quantized fake precision.
        return None
    # fill unvoiced/low-confidence gaps by linear interpolation so the FFT
    # sees a continuous trace instead of zeros
    idx = np.arange(len(seg))
    filled = np.interp(idx, idx[voiced_mask], seg[voiced_mask])
    cents = 1200.0 * np.log2(filled / np.median(filled[voiced_mask]))
    # detrend: remove linear glide so a scoop/fall isn't mistaken for vibrato
    A = np.vstack([idx, np.ones_like(idx)]).T
    slope, intercept = np.linalg.lstsq(A, cents, rcond=None)[0]
    detrended = cents - (slope * idx + intercept)
    n = len(detrended)
    if n < 6:
        return None
    window = np.hanning(n)
    windowed = detrended * window
    spec = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / frame_rate)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return None
    mags = np.abs(spec)
    band_idx = np.where(band_mask)[0]
    peak_i = band_idx[np.argmax(mags[band_idx])]
    # Spectral prominence gate: the declared-band peak must clearly dominate
    # the rest of a wider 2-12Hz window, not just be the tallest bin inside
    # an otherwise flat/noisy spectrum. White jitter and smooth aperiodic
    # wander spread energy across that range fairly evenly; a real vibrato
    # concentrates it at one rate. Excludes the peak's immediate neighbor
    # bins (spectral leakage) from the "rest" comparison.
    wide_mask = (freqs >= 2.0) & (freqs <= 12.0)
    wide_idx = np.where(wide_mask)[0]
    neighbor = {int(peak_i) - 1, int(peak_i), int(peak_i) + 1}
    rest_idx = np.array([i for i in wide_idx if i not in neighbor])
    if len(rest_idx) >= 3:
        rest_median = float(np.median(mags[rest_idx]))
        if mags[peak_i] < _VIBRATO_PROMINENCE_RATIO * max(rest_median, 1e-12):
            return None
    # Parabolic interpolation on the spectral peak -- the raw bin gives a
    # rate quantized to frame_rate/n Hz; this recovers a sub-bin estimate
    # the same way _parabolic_interp already does for YIN's tau bins.
    bin_spacing = frame_rate / n
    peak_frac = _parabolic_interp(mags, int(peak_i))
    rate_hz = float(freqs[peak_i] + peak_frac * bin_spacing)
    # Clamp to the declared vibrato band -- the sub-bin parabolic offset can
    # otherwise walk the interpolated rate just outside [band[0], band[1]]
    # even though the selected bin itself was inside it; a rate outside the
    # declared band isn't vibrato by this function's own contract.
    if rate_hz < band[0] or rate_hz > band[1]:
        return None
    # Require enough full cycles at the detected rate within this note --
    # otherwise the FFT bin is too coarse to trust the rate/extent reading.
    if dur_s * rate_hz < min_cycles:
        return None
    # Report the honest resolution of that rate estimate alongside it: the
    # FFT bin spacing for a segment this long, i.e. 1/duration.
    rate_precision_hz = 1.0 / dur_s
    # Amplitude via a direct sinusoidal least-squares fit at the already
    # interpolated rate_hz, NOT off the FFT bin magnitude. The rate is
    # sub-bin-accurate (parabolic interpolation above), but true_freq
    # essentially never lands exactly on an FFT bin center -- reading
    # amplitude off mags[peak_i] leaks/spreads energy into neighboring bins
    # (classic "scalloping loss"), systematically UNDERESTIMATING extent
    # whenever the true rate falls between bins. Fitting
    # a*sin(2*pi*f*t) + b*cos(2*pi*f*t) + c directly to the (unwindowed)
    # detrended cents contour at the known rate sidesteps bin discretization
    # entirely -- the window taper only mattered for locating the peak.
    t_s = idx / frame_rate
    design = np.column_stack([
        np.sin(2.0 * np.pi * rate_hz * t_s),
        np.cos(2.0 * np.pi * rate_hz * t_s),
        np.ones_like(t_s),
    ])
    (a_coef, b_coef, _c_coef), *_ = np.linalg.lstsq(design, detrended, rcond=None)
    amplitude = float(np.hypot(a_coef, b_coef))  # true sinusoid amplitude
    # Goodness-of-fit gate: a genuine sinusoidal vibrato explains most of the
    # detrended contour's variance once fit at its own rate. White jitter
    # (uncorrelated frame to frame) and smooth aperiodic wander (real
    # low-frequency energy, but not a single stable sinusoid) both leave the
    # bulk of the variance unexplained by ANY one rate's sin/cos pair, even
    # when the FFT happens to show a local peak inside the declared band.
    # This is the check that actually kills those false positives; the
    # prominence gate above narrows candidates but a fit-quality check is
    # needed to confirm the candidate rate is a real periodic component
    # rather than the tallest bin of an otherwise unstructured spectrum.
    fitted = a_coef * design[:, 0] + b_coef * design[:, 1] + _c_coef * design[:, 2]
    residual = detrended - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((detrended - np.mean(detrended)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    if r2 < _VIBRATO_MIN_R2:
        return None
    extent_cents_raw = float(2.0 * amplitude)  # peak-to-peak, as measured off the (already-smoothed) f0 contour
    if extent_cents_raw < extent_min_cents:
        return None
    # Calibrate for YIN's own analysis-window smoothing: YIN estimates one
    # f0 value per hop from a `frame`-sample window, which acts as a
    # low-pass filter on fast pitch modulation -- the f0 contour vibrato
    # sits on is itself an ATTENUATED version of the singer's true vibrato.
    # Modeled as the window's frequency response at the vibrato rate; for a
    # rectangular-ish `frame`-sample averaging window that's the normalized
    # sinc (numpy.sinc(x) = sin(pi x)/(pi x)) evaluated at rate*window_s.
    # Empirically verified against the synthetic ±30/±60-cent test cases in
    # test_singing.py (see its comments for the observed attenuation and
    # resulting calibrated values).
    yin_window_s = frame / sr
    attenuation = float(np.sinc(rate_hz * yin_window_s))
    # Floor the divisor so a near-zero/negative sinc lobe (very high rate or
    # long window) can't blow the calibrated value up to something absurd --
    # in that regime the calibration itself is unreliable, so this is a
    # safety clamp, not a claim of accuracy out there.
    safe_attenuation = max(abs(attenuation), 0.15)
    extent_cents_calibrated = extent_cents_raw / safe_attenuation
    return {
        'rate_hz': round(rate_hz, 2),
        'rate_precision_hz': round(rate_precision_hz, 3),
        'extent_cents_raw': round(extent_cents_raw, 1),
        'extent_cents': round(extent_cents_calibrated, 1),
    }


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def _rms_envelope(x: np.ndarray, sr: int, hop_ms: float = 50) -> tuple:
    hop = max(1, int(sr * hop_ms / 1000))
    n_frames = max(0, len(x) // hop)
    rms = np.zeros(n_frames)
    times = np.zeros(n_frames)
    for i in range(n_frames):
        seg = x[i * hop:(i + 1) * hop]
        rms[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
        times[i] = i * hop / sr
    return rms, times


def analyze_dynamics(x: np.ndarray, sr: int, floor_db: float = -60.0) -> dict:
    rms, times = _rms_envelope(x, sr)
    if len(rms) == 0:
        return {}
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    db = np.maximum(db, floor_db)
    loudest_i = int(np.argmax(db))
    quietest_i = int(np.argmin(db))
    dyn_range = round(float(db.max() - db.min()), 1)

    # sustained slope segments: sliding-window linear regression, keep runs
    # with a consistent trend and enough duration to call it a crescendo.
    segments = []
    win = max(3, int(0.3 * sr / (sr * 0.05)))  # ~0.3s worth of 50ms hops (~6 frames)
    i = 0
    n = len(db)
    while i < n - win:
        seg_t = times[i:i + win]
        seg_db = db[i:i + win]
        A = np.vstack([seg_t, np.ones_like(seg_t)]).T
        slope, _ = np.linalg.lstsq(A, seg_db, rcond=None)[0]
        if abs(slope) >= 6.0:  # dB/sec threshold for "sustained" motion
            j = i + win
            direction = np.sign(slope)
            while j < n - 1:
                local_slope = db[j + 1] - db[j]
                if np.sign(local_slope) != direction and abs(local_slope) > 0.5:
                    break
                j += 1
            delta = round(float(db[j] - db[i]), 1)
            if abs(delta) >= 3.0:
                segments.append({
                    'type': 'crescendo' if delta > 0 else 'decrescendo',
                    'start_s': round(float(times[i]), 2),
                    'end_s': round(float(times[j]), 2),
                    'delta_db': delta,
                })
            i = j
        else:
            i += 1

    return {
        'dynamic_range_db': dyn_range,
        'loudest_db': round(float(db[loudest_i]), 1),
        'loudest_t': round(float(times[loudest_i]), 2),
        'quietest_db': round(float(db[quietest_i]), 1),
        'quietest_t': round(float(times[quietest_i]), 2),
        'start_db': round(float(db[0]), 1),
        'end_db': round(float(db[-1]), 1),
        'segments': segments,
    }


# ---------------------------------------------------------------------------
# Key guess (Krumhansl-Schmuckler)
# ---------------------------------------------------------------------------

_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def guess_key(notes: list) -> Optional[dict]:
    if not notes:
        return None
    hist = np.zeros(12)
    for note in notes:
        pc = note['midi'] % 12
        hist[pc] += note['dur_s']
    if hist.sum() <= 0:
        return None
    best = None
    scores = []
    for mode, profile in (('major', _KK_MAJOR), ('minor', _KK_MINOR)):
        for root in range(12):
            rotated = np.roll(profile, root)
            if np.std(rotated) == 0 or np.std(hist) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(rotated, hist)[0, 1])
            scores.append((corr, root, mode))
    scores.sort(key=lambda t: t[0], reverse=True)
    best_corr, best_root, best_mode = scores[0]
    second_corr = scores[1][0] if len(scores) > 1 else -1.0
    margin = best_corr - second_corr
    strength = 'strong' if (best_corr > 0.6 and margin > 0.1) else 'weak'
    return {
        'key': f'{_PITCH_CLASSES[best_root]} {best_mode}',
        'correlation': round(best_corr, 3),
        'confidence': strength,
    }


# ---------------------------------------------------------------------------
# Tempo candidates (always low confidence for unaccompanied singing)
# ---------------------------------------------------------------------------

def tempo_candidates(x: np.ndarray, sr: int, frame: int = 1024, hop: int = 512) -> list:
    n_frames = max(0, 1 + (len(x) - frame) // hop)
    if n_frames < 4:
        return []
    window = np.hanning(frame)
    mags = np.zeros((n_frames, frame // 2 + 1))
    for i in range(n_frames):
        seg = x[i * hop:i * hop + frame] * window
        mags[i] = np.abs(np.fft.rfft(seg))
    flux = np.zeros(n_frames)
    for i in range(1, n_frames):
        diff = mags[i] - mags[i - 1]
        flux[i] = np.sum(np.maximum(diff, 0))
    if np.std(flux) == 0:
        return []
    flux = flux - flux.mean()
    frame_rate = sr / hop
    ac = np.correlate(flux, flux, mode='full')[len(flux) - 1:]
    min_lag = max(1, int(frame_rate * 60 / 180))  # 180 BPM upper bound
    max_lag = min(len(ac) - 1, int(frame_rate * 60 / 60))  # 60 BPM lower bound
    if max_lag <= min_lag:
        return []
    window_ac = ac[min_lag:max_lag + 1]
    if len(window_ac) == 0 or np.all(window_ac <= 0):
        return []
    # top 2 peaks
    order = np.argsort(window_ac)[::-1]
    cands = []
    seen_bpm = []
    for idx in order:
        lag = min_lag + idx
        bpm = 60.0 * frame_rate / lag
        if any(abs(bpm - b) < 5 for b in seen_bpm):
            continue
        seen_bpm.append(bpm)
        cands.append({'bpm': round(float(bpm), 1), 'confidence': 'low'})
        if len(cands) >= 2:
            break
    return cands


# ---------------------------------------------------------------------------
# Music-aware honesty heuristic
# ---------------------------------------------------------------------------

def _spectral_flatness(x: np.ndarray, sr: int, frame: int = 2048, hop: int = 1024) -> float:
    n_frames = max(0, 1 + (len(x) - frame) // hop)
    if n_frames == 0:
        return 0.0
    window = np.hanning(frame)
    flatnesses = []
    for i in range(n_frames):
        seg = x[i * hop:i * hop + frame] * window
        mag = np.abs(np.fft.rfft(seg)) + 1e-12
        gmean = np.exp(np.mean(np.log(mag)))
        amean = np.mean(mag)
        flatnesses.append(gmean / amean if amean > 0 else 0.0)
    return float(np.mean(flatnesses))


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

MAX_ANALYSIS_S = 8 * 60  # cap so a 25-minute clip can't pin a worker thread


def analyze_singing(path: str) -> dict:
    x, sr = load_wav(path)

    capped_extra = {}
    max_samples = int(MAX_ANALYSIS_S * sr)
    if len(x) > max_samples:
        x = x[:max_samples]
        capped_extra = {'analysis_window_s': round(len(x) / sr, 1)}

    if len(x) < sr * 0.2:
        return {'is_melodic': False, **capped_extra}

    f0, conf = yin_f0(x, sr)
    hop = 256
    frame = 2048

    voiced_frac = float(np.mean(f0 > 0)) if len(f0) else 0.0
    energy = float(np.sqrt(np.mean(x ** 2)))
    flatness = _spectral_flatness(x, sr)
    analysis_mode = 'singing'
    if flatness > 0.3 or (voiced_frac < 0.25 and energy > 0.02):
        analysis_mode = 'singing-with-music'

    notes, glides = segment_notes(f0, conf, sr, hop, frame=frame, return_glides=True)
    if len(notes) < 3:
        return {'is_melodic': False, 'analysis_mode': analysis_mode, **capped_extra}

    # Range/melodic-ness uses true notes PLUS glide endpoints -- a glide's
    # from/to pitches are real notes the singer deliberately passed through
    # (just not held long enough to count as a separate note), so a phrase
    # that glides from a low note up to a high one should still register
    # its full range even though the glide itself never produced a
    # standalone note entry.
    glide_midis = [g['from_midi'] for g in glides] + [g['to_midi'] for g in glides]
    midis = [n['midi'] for n in notes] + glide_midis
    pitch_range = max(midis) - min(midis) if midis else 0
    is_melodic = pitch_range > 4 and len(notes) >= 3

    if not is_melodic:
        return {'is_melodic': False, 'analysis_mode': analysis_mode, **capped_extra}

    vibrato_by_note = []
    for note in notes:
        if note['dur_s'] > 0.35:
            v = detect_vibrato(f0, conf, sr, hop, note['_i0'], note['_i1'])
            if v:
                vibrato_by_note.append({'note_index': notes.index(note), 'note_name': note['note_name'], **v})

    for note in notes:
        note.pop('_i0', None)
        note.pop('_i1', None)

    dynamics = analyze_dynamics(x, sr)
    # Same "notes PLUS glide endpoints" logic for the key histogram -- each
    # glide endpoint contributes half the glide's duration as weight (split
    # between its two endpoints) rather than the full note weight a held
    # note gets, since the singer didn't linger there the way they do on an
    # actual held note.
    notes_for_key = notes + [
        {'midi': g['from_midi'], 'dur_s': g['dur_s'] / 2.0} for g in glides
    ] + [
        {'midi': g['to_midi'], 'dur_s': g['dur_s'] / 2.0} for g in glides
    ]
    key = guess_key(notes_for_key)
    tempo = tempo_candidates(x, sr)

    return {
        'is_melodic': True,
        'analysis_mode': analysis_mode,
        'notes': notes,
        'glides': glides,
        'vibrato': vibrato_by_note,
        'dynamics': dynamics,
        'key': key,
        'tempo_candidates': tempo,
        'voiced_fraction': round(voiced_frac, 3),
        'pitch_range_semitones': int(pitch_range),
        **capped_extra,
    }


# ---------------------------------------------------------------------------
# Card formatting
# ---------------------------------------------------------------------------

def format_singing_section(result: dict) -> str:
    if not result or not result.get('is_melodic'):
        return ''
    lines = []
    notes = result.get('notes', [])
    glides = result.get('glides', [])
    if notes or glides:
        # Interleave real notes and glides by start time so a glide shows
        # up in the melody line where it actually happened (e.g.
        # "C4 → slide C4→C5 (2.0s) → C5") instead of being tacked on.
        entries = [(n['start_s'], n['midi'], n['note_name']) for n in notes]
        entries += [
            (g['start_s'], None, f"slide {g['from_note']}→{g['to_note']} ({g['dur_s']:.1f}s)")
            for g in glides
        ]
        entries.sort(key=lambda e: e[0])
        melody = ' → '.join(text for _, _, text in entries)
        # Range includes glide endpoints -- see analyze_singing's
        # notes_for_key/midis comment for why (real pitches the singer
        # visited, just not held as a standalone note).
        ranged_midis = [(n['midi'], n['note_name']) for n in notes]
        ranged_midis += [(g['from_midi'], g['from_note']) for g in glides]
        ranged_midis += [(g['to_midi'], g['to_note']) for g in glides]
        lo = min(ranged_midis, key=lambda t: t[0])[1]
        hi = max(ranged_midis, key=lambda t: t[0])[1]
        key = result.get('key')
        key_str = ''
        if key:
            key_str = f", ~key {key['key']} [{key['confidence']}]"
        lines.append(f'MELODY: {melody} ({len(notes)} notes, range {lo}–{hi}{key_str})')

    line_bits = []
    if len(notes) >= 2:
        if notes[-1]['midi'] > notes[0]['midi']:
            line_bits.append('rises through the phrase')
        elif notes[-1]['midi'] < notes[0]['midi']:
            line_bits.append('falls through the phrase')
        else:
            line_bits.append('returns to its starting pitch')
    longest = max(notes, key=lambda n: n['dur_s']) if notes else None
    vibrato = result.get('vibrato', [])
    if longest and longest['dur_s'] >= 0.5:
        held_bit = f"holds {longest['note_name']} {longest['dur_s']:.1f}s"
        matching_vib = next((v for v in vibrato if v['note_name'] == longest['note_name']), None)
        if matching_vib:
            held_bit += f" with vibrato ~{matching_vib['rate_hz']:.1f} Hz (±{matching_vib['extent_cents']/2:.0f} cents)"
        line_bits.append(held_bit)
    elif vibrato:
        v = vibrato[0]
        line_bits.append(f"vibrato ~{v['rate_hz']:.1f} Hz (±{v['extent_cents']/2:.0f} cents) on {v['note_name']}")
    if line_bits:
        lines.append('LINE : ' + ', '.join(line_bits))

    dyn = result.get('dynamics') or {}
    if dyn:
        start_db = dyn.get('start_db')
        end_db = dyn.get('end_db')
        delta = None
        if start_db is not None and end_db is not None:
            delta = round(end_db - start_db, 1)
        power_bits = []
        if start_db is not None:
            power_bits.append(f'starts at {start_db:.0f} dB')
        if delta is not None and abs(delta) >= 3:
            direction = 'building' if delta > 0 else 'falling'
            power_bits.append(f'{direction} {delta:+.0f} dB by the end')
        widest_t = dyn.get('loudest_t')
        if widest_t is not None:
            m, s = divmod(int(widest_t), 60)
            power_bits.append(f'widest at {m}:{s:02d}')
        if power_bits:
            lines.append('POWER: ' + ', '.join(power_bits))

    tempo = result.get('tempo_candidates') or []
    if tempo:
        top = tempo[0]
        lines.append(f"TEMPO: ~{top['bpm']:.0f} BPM candidate (low confidence — rubato likely)")

    if result.get('analysis_mode') == 'singing-with-music':
        lines.append('Style: backing music present — melody tracked as most-prominent pitch.')

    return '\n'.join(lines)
