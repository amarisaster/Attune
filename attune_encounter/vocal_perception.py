"""Evidence-bound vocal perception for a music passage or voice recording.

This module measures either a separated vocal stem or a direct voice recording
without claiming who is speaking or singing, what they feel, or which vocal
technique produced a signal. Where mono audio cannot support a claim (voice
count, harmony, secondary-speaker identity), the result says so instead of
manufacturing precision.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

import numpy as np


FRAME_SECONDS = 0.04
HOP_SECONDS = 0.02
DENSITY_WINDOW_SECONDS = 5.0


def _confidence(value: float) -> str:
    if value >= 0.72:
        return "high"
    if value >= 0.42:
        return "moderate"
    return "low"


def _segments(mask: np.ndarray, hop_seconds: float, *, bridge_s: float = 0.16) -> list[tuple[float, float]]:
    if mask.size == 0:
        return []
    filled = mask.astype(bool).copy()
    bridge = max(0, round(bridge_s / hop_seconds))
    if bridge:
        true_indices = np.flatnonzero(filled)
        for left, right in zip(true_indices, true_indices[1:]):
            if 0 < right - left - 1 <= bridge:
                filled[left:right + 1] = True
    result: list[tuple[float, float]] = []
    start = None
    for index, active in enumerate(np.append(filled, False)):
        if active and start is None:
            start = index
        elif not active and start is not None:
            end = index
            if (end - start) * hop_seconds >= 0.12:
                result.append((start * hop_seconds, end * hop_seconds + FRAME_SECONDS))
            start = None
    return result


def _frame_features(samples: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    frame = max(256, round(FRAME_SECONDS * sample_rate))
    hop = max(128, round(HOP_SECONDS * sample_rate))
    if samples.size < frame:
        return {name: np.asarray([], dtype=float) for name in ("time", "db", "centroid", "flatness", "zcr")}
    window = np.hanning(frame)
    frequencies = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    rows = {name: [] for name in ("time", "db", "centroid", "flatness", "zcr")}
    for start in range(0, samples.size - frame + 1, hop):
        raw = samples[start:start + frame]
        centered = raw - float(np.mean(raw))
        rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
        power = np.abs(np.fft.rfft(centered * window)) ** 2 + 1e-16
        total = float(np.sum(power))
        arithmetic = float(np.mean(power))
        geometric = float(np.exp(np.mean(np.log(power))))
        rows["time"].append(start / sample_rate)
        rows["db"].append(20.0 * math.log10(max(rms, 1e-6)))
        rows["centroid"].append(float(np.sum(frequencies * power) / max(total, 1e-16)))
        rows["flatness"].append(geometric / max(arithmetic, 1e-16))
        rows["zcr"].append(float(np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1]))))
    return {name: np.asarray(values, dtype=float) for name, values in rows.items()}


def _activity(
    features: dict[str, np.ndarray],
    duration_s: float,
    *,
    source: str,
) -> tuple[dict, np.ndarray, list[tuple[float, float]]]:
    levels = features["db"]
    if levels.size == 0:
        return ({"available": False, "reason": "not enough audio"}, np.asarray([], dtype=bool), [])
    floor = float(np.percentile(levels, 20))
    peak = float(np.percentile(levels, 95))
    contrast = max(0.0, peak - floor)
    threshold = min(peak - 3.0, floor + max(6.0, contrast * 0.35))
    active = levels >= threshold
    phrases = [
        (start, min(duration_s, end))
        for start, end in _segments(active, HOP_SECONDS)
    ]
    durations = [end - start for start, end in phrases]
    active_fraction = float(np.mean(active))
    if not phrases or active_fraction < 0.08:
        character = "brief/sparse vocal activity"
    elif max(durations) >= 4.0 or (statistics.median(durations) >= 2.0 and active_fraction >= 0.35):
        character = "sustained vocal activity"
    elif len(phrases) >= 4 and statistics.median(durations) < 1.5:
        character = "fragmented vocal activity"
    else:
        character = "mixed sustained and brief vocal activity"

    density = []
    window_frames = max(1, round(DENSITY_WINDOW_SECONDS / HOP_SECONDS))
    for start in range(0, active.size, window_frames):
        stop = min(active.size, start + window_frames)
        density.append({
            "start_s": round(start * HOP_SECONDS, 2),
            "end_s": round(min(duration_s, stop * HOP_SECONDS + FRAME_SECONDS), 2),
            "active_share": round(float(np.mean(active[start:stop])), 3),
        })
    support = min(1.0, contrast / 18.0) * min(1.0, duration_s / 8.0)
    result = {
        "available": True,
        "classification": character,
        "confidence": _confidence(support),
        "active_fraction": round(active_fraction, 3),
        "phrase_count": len(phrases),
        "phrase_duration_s": {
            "median": round(statistics.median(durations), 2) if durations else None,
            "longest": round(max(durations), 2) if durations else None,
        },
        "entrances_s": [round(start, 2) for start, _ in phrases[:16]],
        "exits_s": [round(end, 2) for _, end in phrases[:16]],
        "density_over_time": density,
        "threshold_dbfs": round(threshold, 1),
        "level_contrast_db": round(contrast, 1),
        "claim_limit": (
            "Activity is measured on the separated vocal stem; accompaniment bleed can create false activity."
            if source == "separated_vocal_stem"
            else "Activity is measured from the recording level; background sound can create false voice activity."
        ),
    }
    if source == "separated_vocal_stem":
        result["separation_contrast_db"] = result["level_contrast_db"]
    return result, active, phrases


def _onsets(features: dict[str, np.ndarray], phrases: Sequence[tuple[float, float]]) -> dict:
    levels = features["db"]
    attacks = []
    for start_s, _ in phrases[:16]:
        start = min(levels.size - 1, max(0, round(start_s / HOP_SECONDS)))
        stop = min(levels.size, start + max(2, round(0.6 / HOP_SECONDS)))
        if stop <= start:
            continue
        local = levels[start:stop]
        target = float(np.max(local)) - 3.0
        reached = np.flatnonzero(local >= target)
        rise_s = float(reached[0]) * HOP_SECONDS if reached.size else (stop - start) * HOP_SECONDS
        label = "sharp" if rise_s <= 0.08 else "measured" if rise_s <= 0.22 else "gradual"
        attacks.append({"offset_s": round(start_s, 2), "rise_s": round(rise_s, 2), "character": label})
    if not attacks:
        return {"available": False, "reason": "no stable vocal entrances were measured"}
    counts = {label: sum(item["character"] == label for item in attacks) for label in ("sharp", "measured", "gradual")}
    dominant = max(counts, key=counts.get)
    return {
        "available": True,
        "dominant_character": dominant,
        "events": attacks,
        "confidence": "moderate",
        "claim_limit": "Attack character is an amplitude-envelope measurement, not a claim about vocal technique or intent.",
    }


def _breath_like_events(
    features: dict[str, np.ndarray],
    phrases: Sequence[tuple[float, float]],
    noise_floor: float,
    *,
    source: str,
) -> dict:
    candidates = []
    levels = features["db"]
    for start_s, _ in phrases[1:16]:
        end = max(0, round(start_s / HOP_SECONDS))
        begin = max(0, end - round(0.5 / HOP_SECONDS))
        if end - begin < 2:
            continue
        local_flatness = features["flatness"][begin:end]
        local_centroid = features["centroid"][begin:end]
        local_levels = levels[begin:end]
        evidence = (
            float(np.median(local_flatness)) >= 0.18
            and float(np.median(local_centroid)) >= 1200.0
            and float(np.max(local_levels)) >= noise_floor + 3.0
        )
        if evidence:
            candidates.append({
                "offset_s": round(max(0.0, start_s - 0.5), 2),
                "before_entrance_s": round(start_s, 2),
                "spectral_flatness": round(float(np.median(local_flatness)), 3),
                "centroid_hz": round(float(np.median(local_centroid))),
            })
    return {
        "available": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "confidence": "low",
        "claim_limit": (
            "These are breath-like noise candidates before vocal entrances. "
            + (
                "Stem artifacts, consonants, or accompaniment bleed can look similar; they are not confirmed breaths."
                if source == "separated_vocal_stem"
                else "Consonants, handling noise, or background sound can look similar; they are not confirmed breaths."
            )
        ),
    }


def _timbre_by_phrase(features: dict[str, np.ndarray], phrases: Sequence[tuple[float, float]]) -> dict:
    rows = []
    for start_s, end_s in phrases[:12]:
        begin = max(0, round(start_s / HOP_SECONDS))
        end = min(features["centroid"].size, max(begin + 1, round(end_s / HOP_SECONDS)))
        if end <= begin:
            continue
        rows.append({
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "centroid_hz": round(float(np.median(features["centroid"][begin:end]))),
            "spectral_flatness": round(float(np.median(features["flatness"][begin:end])), 4),
        })
    changes = []
    for prior, current in zip(rows, rows[1:]):
        centroid_delta = current["centroid_hz"] - prior["centroid_hz"]
        flatness_delta = current["spectral_flatness"] - prior["spectral_flatness"]
        labels = []
        if abs(centroid_delta) >= 350:
            labels.append("brighter" if centroid_delta > 0 else "darker")
        if abs(flatness_delta) >= 0.06:
            labels.append("noisier" if flatness_delta > 0 else "more tonal")
        if labels:
            changes.append({
                "at_s": current["start_s"],
                "measured_change": " and ".join(labels),
                "centroid_delta_hz": centroid_delta,
                "flatness_delta": round(flatness_delta, 4),
            })
    return {
        "available": bool(rows),
        "phrases": rows,
        "changes": changes,
        "claim_limit": "Centroid and flatness describe recorded timbre; they do not identify a singer, register, mood, or technique.",
    }


def _pitch_events(singing: dict) -> dict:
    notes = sorted((item for item in (singing.get("notes") or []) if isinstance(item, dict)), key=lambda item: item.get("start_s", 0.0))
    glides = [item for item in (singing.get("glides") or []) if isinstance(item, dict)]
    transitions = []
    for prior, current in zip(notes, notes[1:]):
        if not all(isinstance(item.get("midi"), (int, float)) for item in (prior, current)):
            continue
        gap = float(current.get("start_s", 0.0)) - float(prior.get("end_s", prior.get("start_s", 0.0)))
        delta = float(current["midi"]) - float(prior["midi"])
        if gap <= 0.4 and abs(delta) >= 5:
            transitions.append({
                "at_s": current.get("start_s"),
                "direction": "upward" if delta > 0 else "downward",
                "interval_semitones": round(abs(delta), 1),
                "from": prior.get("note_name"),
                "to": current.get("note_name"),
            })

    runs = []
    index = 0
    while index < len(notes):
        group = [notes[index]]
        if float(group[0].get("dur_s", 99.0)) > 0.42:
            index += 1
            continue
        cursor = index + 1
        while cursor < len(notes):
            prior, current = group[-1], notes[cursor]
            gap = float(current.get("start_s", 0.0)) - float(prior.get("end_s", prior.get("start_s", 0.0)))
            if gap > 0.18 or float(current.get("dur_s", 99.0)) > 0.42:
                break
            group.append(current)
            cursor += 1
        midis = [item.get("midi") for item in group if isinstance(item.get("midi"), (int, float))]
        if len(group) >= 4 and midis and max(midis) - min(midis) >= 3:
            runs.append({
                "start_s": group[0].get("start_s"),
                "end_s": group[-1].get("end_s"),
                "note_count": len(group),
                "span_semitones": round(max(midis) - min(midis), 1),
                "label": "rapid note-run candidate",
            })
            index = cursor
        else:
            index += 1

    transitions = transitions[:12]
    return {
        "pitch_transition_candidates": transitions,
        "register_shift_candidates": transitions,
        "rapid_note_run_candidates": runs[:8],
        "glide_count": len(glides),
        "falling_glide_count": sum(
            isinstance(item.get("from_midi"), (int, float))
            and isinstance(item.get("to_midi"), (int, float))
            and item["to_midi"] < item["from_midi"]
            for item in glides
        ),
        "claim_limit": (
            "Large pitch jumps are pitch-transition candidates, not physiological register breaks. "
            "Rapid note runs are not culture- or technique-specific labels."
        ),
    }


def _summary(
    activity: dict,
    singing: dict,
    texture: dict,
    timbre: dict,
    pitch_events: dict,
    *,
    content_kind: str,
    source: str,
) -> str:
    parts = []
    classification = activity.get("classification")
    if classification:
        phrase = activity.get("phrase_duration_s") or {}
        median = phrase.get("median")
        text = classification
        if isinstance(median, (int, float)):
            text += f", with a median measured phrase of {median:.1f}s"
        parts.append(text)
    if singing.get("is_melodic"):
        held = max((item.get("dur_s", 0.0) for item in singing.get("notes") or []), default=0.0)
        if held >= 0.5:
            parts.append(f"a longest stable note of {held:.1f}s")
        glides = len(singing.get("glides") or [])
        if glides:
            parts.append(f"{glides} measured glide{'s' if glides != 1 else ''}")
        vibrato = singing.get("vibrato") or []
        if vibrato:
            rates = [item.get("rate_hz") for item in vibrato if isinstance(item.get("rate_hz"), (int, float))]
            parts.append(
                f"a stable vibrato pattern near {statistics.median(rates):.1f}Hz"
                if rates else "a stable vibrato pattern"
            )
        else:
            parts.append("no stable vibrato pattern detected")
    elif content_kind == "singing":
        parts.append("no sufficiently stable sustained melody for pitch-detail claims")
    if texture.get("available") and texture.get("label"):
        parts.append(f"a {texture['label']} texture reading")
    changes = timbre.get("changes") or []
    if changes:
        parts.append(f"{len(changes)} measured timbral shift{'s' if len(changes) != 1 else ''} between phrases")
    runs = pitch_events.get("rapid_note_run_candidates") or []
    if runs:
        parts.append(f"{len(runs)} rapid note-run candidate{'s' if len(runs) != 1 else ''}")
    if not parts:
        subject = "separated vocal stem" if source == "separated_vocal_stem" else "voice recording"
        return f"The {subject} did not support a stable vocal summary for this passage."
    return parts[0].capitalize() + "; " + "; ".join(parts[1:]) + "."


def analyze_vocal_perception(
    samples: Sequence[float],
    sample_rate: int,
    singing: dict[str, Any],
    texture: dict[str, Any],
    *,
    source: str = "separated_vocal_stem",
    content_kind: str = "singing",
    apply_residue_gate: bool = True,
) -> dict[str, Any]:
    """Build a passage-local perceptual packet from measured vocal evidence."""
    if source not in {"separated_vocal_stem", "direct_voice_recording"}:
        raise ValueError(f"unsupported vocal source: {source}")
    if content_kind not in {"singing", "spoken_voice"}:
        raise ValueError(f"unsupported vocal content kind: {content_kind}")
    x = np.asarray(samples, dtype=float)
    duration_s = x.size / max(sample_rate, 1)
    features = _frame_features(x, sample_rate)
    activity, _active_mask, phrases = _activity(features, duration_s, source=source)
    if not activity.get("available"):
        return {
            "available": False,
            "reason": activity.get("reason", "not enough vocal evidence"),
            "source": source,
            "texture": texture,
        }
    # A separator always emits *something*, even for an instrumental tone.
    # Do not turn low-contrast, non-melodic, non-periodic residue into a
    # confident story about a singer merely because it came out of the
    # tensor named "vocals".
    periodicity = texture.get("periodicity_median")
    weak_voice_specific_evidence = (
        not singing.get("is_melodic")
        and float(activity.get("level_contrast_db") or 0.0) < 6.0
        and (
            not texture.get("available")
            or not isinstance(periodicity, (int, float))
            or periodicity < 0.38
        )
    )
    if apply_residue_gate and weak_voice_specific_evidence:
        return {
            "available": False,
            "reason": "the separated stem contained only weak non-melodic residue, not enough voice-specific evidence",
            "source": source,
            "analysis_scope": "current_passage_only" if source == "separated_vocal_stem" else "whole_recording",
            "vocal_activity": activity,
            "texture": texture,
            "limits": "Low-level separation residue was omitted rather than described as a singer.",
        }
    noise_floor = float(np.percentile(features["db"], 20)) if features["db"].size else -120.0
    onset = _onsets(features, phrases)
    breaths = _breath_like_events(features, phrases, noise_floor, source=source)
    timbre = _timbre_by_phrase(features, phrases)
    pitch_events = _pitch_events(singing)
    return {
        "available": True,
        "source": source,
        "content_kind": content_kind,
        "analysis_scope": "current_passage_only" if source == "separated_vocal_stem" else "whole_recording",
        "measurement_version": 2,
        "summary": _summary(
            activity,
            singing,
            texture,
            timbre,
            pitch_events,
            content_kind=content_kind,
            source=source,
        ),
        "vocal_activity": activity,
        "onset_attack": onset,
        "breath_like_events": breaths,
        "timbre_by_phrase": timbre,
        "pitch_events": pitch_events,
        "texture": texture,
        "layered_vocals": {
            "available": False,
            "reason": (
                "simultaneous voice count and harmony are not reliable from the current separated stem without validated multi-pitch evidence"
                if source == "separated_vocal_stem"
                else "simultaneous voice count and harmony are not reliable from a mono voice recording without validated multi-pitch evidence"
            ),
        },
        "secondary_vocals": {
            "available": False,
            "reason": (
                "ad-libs and secondary-vocal entrances require voice separation or identity evidence this organ does not have"
                if source == "separated_vocal_stem"
                else "overlapping or secondary speakers require speaker-separation evidence this organ does not have"
            ),
        },
        "limits": (
            (
                "Voice isolation can retain accompaniment or separation artifacts. "
                if source == "separated_vocal_stem"
                else "Microphone, room, compression, handling, and background sound can affect these measurements. "
            )
            + "Measurements describe recorded sound, not singer identity, emotion, intent, culture-specific technique, or vocal health."
        ),
    }
