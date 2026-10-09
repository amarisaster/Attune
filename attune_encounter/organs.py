"""Independent, passage-local listening organs.

Every value is derived from only the supplied passage. Nothing is normalized
against the rest of the recording, so the encounter boundary remains causal.
The measurements describe sound; they do not prescribe an emotional reading.
"""

from __future__ import annotations

import cmath
import math
import statistics
from typing import Sequence


FFT_SIZE = 2048
SPECTRAL_HOP_SECONDS = 2.0
ONSET_FRAME_SECONDS = 0.1
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _fft(values: list[complex]) -> list[complex]:
    """In-place-style radix-2 FFT returned as a new list."""
    size = len(values)
    if size == 0 or size & (size - 1):
        raise ValueError("FFT input length must be a power of two")
    output = list(values)
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            output[i], output[j] = output[j], output[i]
    length = 2
    while length <= size:
        root = cmath.exp(-2j * math.pi / length)
        half = length // 2
        for start in range(0, size, length):
            weight = 1 + 0j
            for offset in range(half):
                even = output[start + offset]
                odd = weight * output[start + offset + half]
                output[start + offset] = even + odd
                output[start + offset + half] = even - odd
                weight *= root
        length <<= 1
    return output


def _spectral_frame(samples: Sequence[float], sample_rate: int) -> dict:
    frame = [0.0] * FFT_SIZE
    take = min(len(samples), FFT_SIZE)
    for index in range(take):
        # Hann window reduces boundary leakage without using any future frame.
        frame[index] = float(samples[index]) * (0.5 - 0.5 * math.cos(2 * math.pi * index / (FFT_SIZE - 1)))
    spectrum = _fft([complex(value, 0) for value in frame])[: FFT_SIZE // 2 + 1]
    powers = [value.real * value.real + value.imag * value.imag for value in spectrum]
    frequencies = [index * sample_rate / FFT_SIZE for index in range(len(powers))]
    total = sum(powers) or 1e-20
    centroid = sum(freq * power for freq, power in zip(frequencies, powers)) / total
    cumulative = 0.0
    rolloff = 0.0
    for freq, power in zip(frequencies, powers):
        cumulative += power
        if cumulative >= total * 0.85:
            rolloff = freq
            break
    bands = {"low_20_250": 0.0, "low_mid_250_1000": 0.0, "high_mid_1000_4000": 0.0, "high_4000_plus": 0.0}
    chroma = [0.0] * 12
    positive = []
    for freq, power in zip(frequencies[1:], powers[1:]):
        if freq < 20:
            continue
        positive.append(max(power, 1e-20))
        if freq < 250:
            bands["low_20_250"] += power
        elif freq < 1000:
            bands["low_mid_250_1000"] += power
        elif freq < 4000:
            bands["high_mid_1000_4000"] += power
        else:
            bands["high_4000_plus"] += power
        if 55 <= freq <= 2000:
            midi = 69 + 12 * math.log2(freq / 440.0)
            chroma[round(midi) % 12] += power
    band_total = sum(bands.values()) or 1e-20
    chroma_total = sum(chroma) or 1e-20
    arithmetic = sum(positive) / max(1, len(positive))
    geometric = math.exp(sum(math.log(value) for value in positive) / max(1, len(positive)))
    return {
        "signal_power": total if sum(powers) > 0 else 0.0,
        "centroid_hz": centroid,
        "rolloff_85_hz": rolloff,
        "flatness": geometric / max(arithmetic, 1e-20),
        "bands": {key: value / band_total for key, value in bands.items()},
        "chroma": [value / chroma_total for value in chroma],
    }


def _spectral_organ(samples: Sequence[float], sample_rate: int) -> dict:
    hop = max(FFT_SIZE, round(SPECTRAL_HOP_SECONDS * sample_rate))
    frames = [_spectral_frame(samples[start:start + FFT_SIZE], sample_rate) for start in range(0, len(samples), hop)]
    if not frames:
        frames = [_spectral_frame([], sample_rate)]
    mean_bands = {
        key: round(sum(frame["bands"][key] for frame in frames) / len(frames), 4)
        for key in frames[0]["bands"]
    }
    mean_chroma = [sum(frame["chroma"][index] for frame in frames) / len(frames) for index in range(12)]
    supported = any(frame["signal_power"] > 1e-12 for frame in frames)
    ranked = sorted(range(12), key=lambda index: mean_chroma[index], reverse=True)[:3] if supported else []
    return {
        "organ": "spectral_balance",
        "version": 1,
        "supported": supported,
        "frame_seconds": SPECTRAL_HOP_SECONDS,
        "centroid_hz_mean": round(sum(frame["centroid_hz"] for frame in frames) / len(frames), 1),
        "rolloff_85_hz_mean": round(sum(frame["rolloff_85_hz"] for frame in frames) / len(frames), 1),
        "spectral_flatness_mean": round(sum(frame["flatness"] for frame in frames) / len(frames), 4),
        "band_energy_share": mean_bands,
        "harmonic_color": {
            "kind": "local_chroma_like_energy",
            "pitch_class_share": {PITCH_CLASSES[index]: round(mean_chroma[index], 4) for index in range(12)},
            "strongest_classes": [PITCH_CLASSES[index] for index in ranked],
            "claim_limit": "This is pitch-class energy, not a key, chord, melody, or tuning claim.",
        },
    }


def _onset_organ(samples: Sequence[float], sample_rate: int) -> dict:
    frame_size = max(1, round(ONSET_FRAME_SECONDS * sample_rate))
    levels = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start:start + frame_size]
        if not frame:
            continue
        levels.append(math.sqrt(sum(float(value) ** 2 for value in frame) / len(frame)))
    rises = [max(0.0, levels[index] - levels[index - 1]) for index in range(1, len(levels))]
    if not rises:
        return {"organ": "change_activity", "version": 1, "events_per_minute": 0.0, "strongest_offsets_s": []}
    median = statistics.median(rises)
    deviations = [abs(value - median) for value in rises]
    threshold = median + 3 * (statistics.median(deviations) or 1e-9)
    events = [(index + 1, value) for index, value in enumerate(rises) if value > threshold and value > 1e-5]
    strongest = sorted(events, key=lambda item: item[1], reverse=True)[:8]
    duration_minutes = max(len(samples) / sample_rate / 60, 1e-9)

    # Pulse is inferred only from the local positive-change envelope.
    best_bpm = None
    best_score = 0.0
    total_energy = sum(value * value for value in rises) or 1e-20
    for lag in range(max(2, round(60 / 200 / ONSET_FRAME_SECONDS)), round(60 / 50 / ONSET_FRAME_SECONDS) + 1):
        if lag >= len(rises):
            break
        score = sum(rises[index] * rises[index - lag] for index in range(lag, len(rises))) / total_energy
        if score > best_score:
            best_score = score
            best_bpm = 60 / (lag * ONSET_FRAME_SECONDS)
    pulse = {
        "supported": best_bpm is not None and best_score >= 0.18 and len(events) >= 3,
        "bpm": round(best_bpm, 1) if best_bpm is not None and best_score >= 0.18 and len(events) >= 3 else None,
        "confidence": round(min(best_score, 1.0), 3),
        "claim_limit": "Local change-envelope periodicity; not a global tempo or beat-grid claim.",
    }
    return {
        "organ": "change_activity",
        "version": 1,
        "frame_seconds": ONSET_FRAME_SECONDS,
        "events_per_minute": round(len(events) / duration_minutes, 2),
        "strongest_offsets_s": [round(index * ONSET_FRAME_SECONDS, 2) for index, _ in strongest],
        "local_pulse": pulse,
    }


def analyze_passage(samples: Sequence[float], sample_rate: int) -> dict:
    """Return richer evidence derived only from one already-revealed passage."""
    return {
        "analysis_scope": "current_passage_only",
        "organs": [_spectral_organ(samples, sample_rate), _onset_organ(samples, sample_rate)],
        "unsupported": [
            "source-separated instruments",
            "speaker or singer identity",
            "verified melody transcription",
            "vocal technique labels",
            "global key, structure, or tempo",
            "emotion or intended meaning",
        ],
    }
