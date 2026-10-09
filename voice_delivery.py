"""Evidence-bound spoken-voice texture and delivery summaries.

This layer translates Attune's existing measurements into language a listener
can use, and adds three small DSP indicators for vocal texture. It describes
the recording; it never assigns an emotion, intention, personality, or health
state to the speaker.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from singing import load_wav


def analyze_vocal_texture(path: str) -> dict[str, Any]:
    """Measure voice texture without inferring emotion, intent, or identity."""
    x, sr = load_wav(path)
    frame = max(256, int(sr * 0.04))
    hop = max(128, int(sr * 0.02))
    if len(x) < frame:
        return {"available": False, "reason": "not enough audio"}

    rows: list[tuple[float, np.ndarray]] = []
    for start in range(0, len(x) - frame + 1, hop):
        segment = x[start:start + frame]
        rms = float(np.sqrt(np.mean(segment * segment) + 1e-12))
        rows.append((rms, segment))
    if not rows:
        return {"available": False, "reason": "not enough audio"}

    levels = np.asarray([row[0] for row in rows])
    threshold = max(float(np.percentile(levels, 60)), float(levels.max()) * 0.018)
    active = [segment for rms, segment in rows if rms >= threshold]
    if len(active) < 4:
        return {"available": False, "reason": "not enough active voice"}

    periodicity: list[float] = []
    flatness: list[float] = []
    zcr: list[float] = []
    lag_min = max(1, int(sr / 450.0))
    lag_max = min(frame - 2, int(sr / 70.0))
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    band = (freqs >= 80.0) & (freqs <= min(8000.0, sr / 2.0))

    for raw in active:
        centered = raw - float(np.mean(raw))
        windowed = centered * window
        power = np.abs(np.fft.rfft(windowed)) ** 2 + 1e-12
        selected = power[band]
        if selected.size:
            flatness.append(float(np.exp(np.mean(np.log(selected))) / np.mean(selected)))
        zcr.append(float(np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1]))))
        corr = np.correlate(windowed, windowed, mode="full")[frame - 1:]
        base = float(corr[0])
        if base > 1e-10 and lag_max > lag_min:
            periodicity.append(float(np.max(corr[lag_min:lag_max + 1]) / base))

    if not periodicity or not flatness:
        return {"available": False, "reason": "texture could not be measured"}

    p = float(np.median(periodicity))
    f = float(np.median(flatness))
    z = float(np.median(zcr)) if zcr else 0.0
    if p >= 0.62 and f < 0.10:
        label = "clear/tonal"
    elif f >= 0.20:
        label = "airy/noisy indicator"
    elif p < 0.38:
        label = "rough/irregular indicator"
    else:
        label = "mixed clear and airy/noisy"
    return {
        "available": True,
        "label": label,
        "periodicity_median": round(p, 3),
        "spectral_flatness_median": round(f, 4),
        "zero_crossing_rate_median": round(z, 4),
        "active_frames": len(active),
    }


def _pace_label(data: dict) -> str | None:
    pace = data.get("pace") or {}
    reading = pace.get("word_span") or pace.get("speech_span") or pace.get("whole_clip") or {}
    wpm = reading.get("wpm")
    if not isinstance(wpm, (int, float)) or isinstance(wpm, bool):
        return None
    if wpm < 90:
        return "slow measured pace"
    if wpm <= 155:
        return "moderate measured pace"
    return "quick measured pace"


def _pitch_label(data: dict) -> tuple[str | None, float | None]:
    pitch = data.get("pitch") or {}
    low, high = pitch.get("p10_hz"), pitch.get("p90_hz")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in (low, high)):
        return None, None
    span = 12.0 * math.log2(float(high) / float(low))
    label = "narrow pitch movement" if span < 3 else "varied pitch movement" if span < 7 else "wide pitch movement"
    return label, round(span, 1)


def analyze_voice_delivery(path: str, measurements: dict) -> dict:
    """Return measurements plus a short grounded description of delivery."""
    texture = analyze_vocal_texture(path)
    descriptors: list[str] = []
    pace = _pace_label(measurements)
    if pace:
        descriptors.append(pace)
    pitch_label, pitch_span = _pitch_label(measurements)
    if pitch_label:
        descriptors.append(pitch_label)
    acoustic = measurements.get("acoustic") or {}
    brightness = acoustic.get("brightness_label")
    if isinstance(brightness, str) and brightness:
        descriptors.append(f"{brightness} measured timbre")
    dynamics = acoustic.get("dynamics_label")
    if isinstance(dynamics, str) and dynamics:
        descriptors.append(f"{dynamics} volume movement")
    held = (measurements.get("timeline") or {}).get("held_silences") or []
    if held:
        descriptors.append(f"{len(held)} held pause{'s' if len(held) != 1 else ''}")

    summary = ", ".join(descriptors) if descriptors else "not enough stable speech for a delivery summary"
    return {
        "summary": summary,
        "pitch_span_semitones": pitch_span,
        "texture": texture,
        "boundary": (
            "Recording and microphone conditions can change texture readings. "
            "These cues describe sound only; they do not prove emotion or intent."
        ),
    }


def format_voice_delivery_section(result: dict) -> str:
    lines = ["\nVOICE DELIVERY (observed, not diagnosed)", f"READ  : {result['summary']}"]
    texture = result.get("texture") or {}
    if texture.get("available"):
        lines.append(
            "VOICE : {label} · periodicity {p:.3f} · spectral flatness {f:.4f}".format(
                label=texture["label"],
                p=texture["periodicity_median"],
                f=texture["spectral_flatness_median"],
            )
        )
    else:
        lines.append(f"VOICE : unavailable ({texture.get('reason', 'insufficient evidence')})")
    lines.append(f"LIMIT : {result['boundary']}")
    return "\n".join(lines)
