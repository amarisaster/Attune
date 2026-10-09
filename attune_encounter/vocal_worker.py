"""Isolated Attune vocal-stem worker for sequential encounters.

The service's small stdlib runtime launches this module with Attune's venv.
It separates a recording once, then measures each already-defined passage on
the vocal stem. JSON enters on stdin and leaves on stdout; song identity is
never supplied to this process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import wave


ATTUNE_DIR = Path(os.environ.get(
    "ATTUNE_ENCOUNTER_ROOT", str(Path(__file__).resolve().parent.parent),
)).resolve()


def _compact(result: dict, texture: dict) -> dict:
    if not result.get("is_melodic"):
        return {
            "available": False,
            "reason": "no sufficiently stable melodic vocal line was measured in this passage",
            "source": "separated_vocal_stem",
            "texture": texture,
        }
    notes = result.get("notes") or []
    glides = result.get("glides") or []
    vibrato = result.get("vibrato") or []
    longest = max(notes, key=lambda item: item.get("dur_s", 0.0), default=None)
    dynamics = result.get("dynamics") or {}
    dynamic_segments = sorted(
        dynamics.get("segments") or [],
        key=lambda item: abs(float(item.get("delta_db", 0.0))), reverse=True,
    )[:8]
    dynamic_segments.sort(key=lambda item: float(item.get("start_s", 0.0)))
    pitched = [item for item in notes if isinstance(item.get("midi"), (int, float))]
    pitch_bounds = None
    if pitched:
        low = min(pitched, key=lambda item: item["midi"])
        high = max(pitched, key=lambda item: item["midi"])
        pitch_bounds = {"low": low.get("note_name"), "high": high.get("note_name")}
    contour = None
    if len(pitched) >= 2:
        delta = pitched[-1]["midi"] - pitched[0]["midi"]
        contour = "rising" if delta > 1 else "falling" if delta < -1 else "returning/level"
    return {
        "available": True,
        "source": "separated_vocal_stem",
        "analysis_scope": "current_passage_only",
        "voiced_fraction": result.get("voiced_fraction"),
        "pitch_range_semitones": result.get("pitch_range_semitones"),
        "pitch_bounds": pitch_bounds,
        "phrase_contour": contour,
        "held_note": ({
            "note": longest.get("note_name"),
            "duration_s": longest.get("dur_s"),
        } if longest and longest.get("dur_s", 0.0) >= 0.5 else None),
        "glides": [{
            "from": item.get("from_note"), "to": item.get("to_note"),
            "duration_s": item.get("dur_s"), "start_s": item.get("start_s"),
        } for item in glides[:8]],
        "vibrato": [{
            "note": item.get("note_name"), "rate_hz": item.get("rate_hz"),
            "extent_cents": item.get("extent_cents"),
        } for item in vibrato[:8]],
        "dynamics": {
            "dynamic_range_db": dynamics.get("dynamic_range_db"),
            "start_db": dynamics.get("start_db"),
            "end_db": dynamics.get("end_db"),
            "strongest_movements": dynamic_segments,
        },
        "texture": texture,
        "limits": (
            "Voice isolation can retain accompaniment or separation artifacts. "
            "These measurements describe sound, not singer identity, emotion, intent, or vocal health."
        ),
    }


def main() -> None:
    request = json.load(sys.stdin)
    source = Path(str(request["audio_path"]))
    ranges = request["ranges"]
    sys.path.insert(0, str(ATTUNE_DIR))
    import numpy as np
    import stems
    from singing import analyze_singing
    from voice_delivery import analyze_vocal_texture

    mixed = stems.load_stereo_44k(str(source))
    vocals = stems.separate(mixed)["vocals"]
    results = []
    with tempfile.TemporaryDirectory(prefix="attune-encounter-vocals-") as temporary:
        root = Path(temporary)
        for index, item in enumerate(ranges):
            start = max(0, round(float(item[0]) * stems.SR))
            end = min(vocals.shape[1], round(float(item[1]) * stems.SR))
            clip = vocals[:, start:end]
            path = root / f"{index:04d}.wav"
            peak = float(np.max(np.abs(clip))) if clip.size else 0.0
            scaled = np.clip(clip / max(peak, 1.0), -1.0, 1.0)
            pcm = (scaled.T * 32767.0).astype(np.int16)
            with wave.open(str(path), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(stems.SR)
                output.writeframes(pcm.tobytes())
            results.append(_compact(analyze_singing(str(path)), analyze_vocal_texture(str(path))))
    json.dump({"passages": results}, sys.stdout, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
