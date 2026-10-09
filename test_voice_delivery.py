"""Small deterministic tests for the spoken voice-delivery layer."""

import tempfile
import wave
from pathlib import Path

import numpy as np

from voice_delivery import analyze_voice_delivery, format_voice_delivery_section


def _write(path: Path, signal: np.ndarray, sr: int = 22050) -> None:
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sr)
        output.writeframes(pcm.tobytes())


def _base() -> dict:
    return {
        "pace": {"speech_span": {"wpm": 112}},
        "pitch": {"available": True, "p10_hz": 180.0, "p90_hz": 250.0},
        "acoustic": {"brightness_label": "warm", "dynamics_label": "dynamic"},
        "timeline": {"held_silences": [{"dur_s": 1.2}]},
    }


def test_tonal_signal_gets_grounded_summary() -> None:
    sr = 22050
    t = np.arange(sr * 2) / sr
    signal = 0.45 * (np.sin(2 * np.pi * 180 * t) + 0.25 * np.sin(2 * np.pi * 360 * t))
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "voice.wav"
        _write(path, signal, sr)
        result = analyze_voice_delivery(str(path), _base())
    assert result["texture"]["available"] is True
    assert result["texture"]["label"] == "clear/tonal"
    assert "moderate measured pace" in result["summary"]
    assert "varied pitch movement" in result["summary"]
    rendered = format_voice_delivery_section(result)
    assert "observed, not diagnosed" in rendered
    assert "do not prove emotion or intent" in rendered


def test_noise_is_not_mislabeled_as_emotion() -> None:
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 0.2, 22050 * 2)
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "noise.wav"
        _write(path, signal)
        result = analyze_voice_delivery(str(path), _base())
    assert result["texture"]["available"] is True
    assert "noisy" in result["texture"]["label"]
    rendered = format_voice_delivery_section(result).lower()
    for forbidden in ("sad", "happy", "angry", "anxious"):
        assert forbidden not in rendered


if __name__ == "__main__":
    test_tonal_signal_gets_grounded_summary()
    test_noise_is_not_mislabeled_as_emotion()
    print("PASS voice delivery")
