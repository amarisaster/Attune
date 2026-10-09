"""Core state machine for private-by-default, blind music encounters.

The trusted preparation side sees the source audio and identity. The listener
side receives only an opaque session id and one passage packet at a time.
"""

from __future__ import annotations

from array import array
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import sys
from typing import Callable, Iterable, Iterator, Sequence
import uuid

from .organs import analyze_passage


ANALYSIS_SAMPLE_RATE = 16_000
MOTION_SECONDS = 5.0
FIXED_PASSAGE_SECONDS = 60.0
ADAPTIVE_MIN_SECONDS = 45.0
ADAPTIVE_MAX_SECONDS = 90.0
MAX_AUDIO_SECONDS = 15 * 60
MAX_AUDIO_BYTES = 250 * 1024 * 1024
MAX_NOTE_CHARS = 16_000
MAX_LYRIC_CHARS = 2_000
MAX_RETROSPECTIVE_CHARS = 24_000
MAX_RECORD_CONTENT_CHARS = 32_000
LISTENER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
VOCAL_ANALYSIS_TIMEOUT_SECONDS = 330


class EncounterError(ValueError):
    """A safe, user-facing encounter error."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decode_json(raw: str) -> object:
    return json.loads(raw)


def _validate_listener(listener_id: str) -> str:
    listener_id = str(listener_id or "").strip().lower()
    if not LISTENER_RE.fullmatch(listener_id):
        raise EncounterError("invalid listener id")
    return listener_id


def _validate_identity(identity: dict) -> dict:
    if not isinstance(identity, dict):
        raise EncounterError("identity must be an object")
    allowed = ("title", "artist", "album", "release_year", "catalog_id", "source_filename")
    clean: dict[str, object] = {}
    for key in allowed:
        if key not in identity or identity[key] is None:
            continue
        value = identity[key]
        if key == "release_year":
            if not isinstance(value, int) or not 1000 <= value <= 9999:
                raise EncounterError("release_year must be a four-digit integer")
            clean[key] = value
            continue
        text = str(value).strip()
        if not text or len(text) > 500:
            raise EncounterError(f"invalid identity field: {key}")
        clean[key] = text
    if "title" not in clean or "artist" not in clean:
        raise EncounterError("identity requires title and artist")
    return clean


def _validate_lyrics(manifest: dict, duration_s: float) -> tuple[dict, list[dict]]:
    if not isinstance(manifest, dict):
        raise EncounterError("lyrics manifest must be an object")
    verified = manifest.get("verified") is True
    verification = "verified" if verified else str(manifest.get("verification") or "").strip()
    if verification not in {"verified", "automatic", "unavailable"}:
        raise EncounterError("lyrics must be verified, automatic, or explicitly unavailable")
    source = str(manifest.get("source") or "").strip()
    if not source or len(source) > 500:
        raise EncounterError("lyrics require a bounded provenance source")
    raw_lines = manifest.get("lines")
    if not isinstance(raw_lines, list):
        raise EncounterError("lyrics lines must be an array")
    if verification == "unavailable" and raw_lines:
        raise EncounterError("unavailable lyrics cannot contain lyric lines")

    lines: list[dict] = []
    previous_start = -1.0
    for raw in raw_lines:
        if not isinstance(raw, dict):
            raise EncounterError("each lyric line must be an object")
        try:
            start = float(raw["start_s"])
            end = float(raw["end_s"])
        except (KeyError, TypeError, ValueError):
            raise EncounterError("each lyric line requires numeric start_s and end_s") from None
        text = str(raw.get("text") or "").strip()
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise EncounterError("invalid lyric timing")
        if start < previous_start:
            raise EncounterError("lyrics must be ordered by start time")
        if end > duration_s + 2.0:
            raise EncounterError("lyric timing exceeds the recording")
        if not text or len(text) > MAX_LYRIC_CHARS:
            raise EncounterError("invalid lyric text")
        lines.append({"start_s": round(start, 3), "end_s": round(end, 3), "text": text})
        previous_start = start
    return {"verified": verified, "verification": verification, "source": source}, lines


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        duration = float(result.stdout.strip()) if result.returncode == 0 else math.nan
    except (OSError, subprocess.SubprocessError, ValueError):
        duration = math.nan
    if not math.isfinite(duration) or duration <= 0:
        raise EncounterError("audio duration could not be verified")
    if duration > MAX_AUDIO_SECONDS:
        raise EncounterError("audio exceeds the 15-minute encounter limit")
    return duration


def decode_audio(path: Path) -> tuple[array, int]:
    """Decode local audio to mono float32 PCM without exposing its filename."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise EncounterError("audio file is missing or empty")
    if path.stat().st_size > MAX_AUDIO_BYTES:
        raise EncounterError("audio file exceeds the size limit")
    _probe_duration(path)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-t",
        str(MAX_AUDIO_SECONDS),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(ANALYSIS_SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=300, check=False)
    except (OSError, subprocess.SubprocessError):
        raise EncounterError("audio decoding failed") from None
    if result.returncode != 0 or not result.stdout or len(result.stdout) % 4:
        raise EncounterError("audio decoding failed")
    samples = array("f")
    samples.frombytes(result.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise EncounterError("audio decoding produced no samples")
    return samples, ANALYSIS_SAMPLE_RATE


def _stats(samples: Sequence[float], sample_rate: int) -> dict:
    count = len(samples)
    if count == 0:
        return {"rms_dbfs": -120.0, "peak_dbfs": -120.0, "zero_crossings_per_s": 0.0}
    square_sum = 0.0
    peak = 0.0
    crossings = 0
    prior = float(samples[0])
    for raw in samples:
        value = float(raw)
        square_sum += value * value
        peak = max(peak, abs(value))
        if (prior < 0 <= value) or (prior >= 0 > value):
            crossings += 1
        prior = value
    rms = math.sqrt(square_sum / count)
    duration = count / sample_rate
    return {
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-6)), 2),
        "peak_dbfs": round(20 * math.log10(max(peak, 1e-6)), 2),
        "zero_crossings_per_s": round(crossings / max(duration, 1e-9), 2),
    }


def _change_score(before: dict, after: dict) -> float:
    level = min(abs(after["rms_dbfs"] - before["rms_dbfs"]) / 12.0, 1.0)
    z_before = before["zero_crossings_per_s"] + 1.0
    z_after = after["zero_crossings_per_s"] + 1.0
    texture = min(abs(math.log2(z_after / z_before)) / 1.5, 1.0)
    return 0.65 * level + 0.35 * texture


def passage_ranges(samples: Sequence[float], sample_rate: int, mode: str) -> list[tuple[int, int]]:
    """Return causal ranges. A boundary never examines samples after itself."""
    if mode not in {"fixed", "adaptive"}:
        raise EncounterError("mode must be fixed or adaptive")
    total = len(samples)
    if mode == "fixed":
        step = round(FIXED_PASSAGE_SECONDS * sample_rate)
        return [(start, min(total, start + step)) for start in range(0, total, step)]

    minimum = round(ADAPTIVE_MIN_SECONDS * sample_rate)
    maximum = round(ADAPTIVE_MAX_SECONDS * sample_rate)
    probe = round(MOTION_SECONDS * sample_rate)
    history = round(15.0 * sample_rate)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        hard_end = min(total, start + maximum)
        candidate = min(hard_end, start + minimum)
        chosen = hard_end
        while True:
            after_start = max(start, candidate - probe)
            before_start = max(start, after_start - history)
            before = _stats(samples[before_start:after_start], sample_rate)
            after = _stats(samples[after_start:candidate], sample_rate)
            if _change_score(before, after) >= 0.72:
                chosen = candidate
                break
            if candidate >= hard_end:
                break
            candidate = min(hard_end, candidate + probe)
        ranges.append((start, chosen))
        start = chosen
    return ranges


def _motion(samples: Sequence[float], sample_rate: int, absolute_start: int) -> list[dict]:
    step = round(MOTION_SECONDS * sample_rate)
    result = []
    for offset in range(0, len(samples), step):
        chunk = samples[offset:offset + step]
        stats = _stats(chunk, sample_rate)
        result.append(
            {
                "start_s": round((absolute_start + offset) / sample_rate, 3),
                "end_s": round((absolute_start + offset + len(chunk)) / sample_rate, 3),
                **stats,
            }
        )
    return result


def _passage_evidence(
    samples: Sequence[float], sample_rate: int, absolute_start: int,
    vocal_perception: dict | None = None,
) -> dict:
    return {
        "measurement_version": 3,
        "measurement_kind": "passage-local-listening-organs",
        "whole_passage": _stats(samples, sample_rate),
        "motion": _motion(samples, sample_rate, absolute_start),
        "deeper_listening": analyze_passage(samples, sample_rate),
        "vocal_perception": vocal_perception or {
            "available": False,
            "reason": "source-separated vocal perception was unavailable for this encounter",
        },
        "limits": [
            "The mixed-track organs do not support emotion, instrument identity, singer identity, verified melody transcription, global key, or genre claims.",
            "Zero-crossing activity is a coarse signal-change proxy, not an instrument or texture label.",
        ],
    }


def _vocal_passage_evidence(
    audio_path: Path, ranges: Sequence[tuple[int, int]], sample_rate: int,
) -> list[dict]:
    """Run Attune once for the full recording and return passage-local results.

    This remains best-effort: failure of a perception organ must not destroy a
    valid blind encounter or expose subprocess details to its listener.
    """
    unavailable = {
        "available": False,
        "reason": "source-separated vocal perception was unavailable for this encounter",
    }
    if os.environ.get("ATTUNE_ENCOUNTER_VOCAL_ANALYSIS", "").strip().lower() not in {"1", "true", "yes"}:
        return [dict(unavailable) for _ in ranges]
    python = os.environ.get(
        "ATTUNE_ENCOUNTER_PYTHON",
        sys.executable,
    )
    request = {
        "audio_path": str(Path(audio_path).resolve()),
        "ranges": [[start / sample_rate, end / sample_rate] for start, end in ranges],
    }
    try:
        result = subprocess.run(
            [python, "-m", "attune_encounter.vocal_worker"],
            input=_json(request), capture_output=True, text=True,
            timeout=VOCAL_ANALYSIS_TIMEOUT_SECONDS, check=False,
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else {}
        passages = payload.get("passages") if isinstance(payload, dict) else None
        if not isinstance(passages, list) or len(passages) != len(ranges):
            raise ValueError("invalid vocal worker result")
        return [item if isinstance(item, dict) else dict(unavailable) for item in passages]
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return [dict(unavailable) for _ in ranges]


def _artifact_root(db_path: Path) -> Path:
    root = Path(db_path).parent / "artifacts"
    if root.is_symlink():
        raise EncounterError("encounter artifact directory must not be a symbolic link")
    with _private_umask():
        root.mkdir(parents=True, exist_ok=True)
    if root.stat().st_mode & 0o077:
        raise EncounterError("encounter artifact directory must be private (mode 0700)")
    return root


def _extract_passage_audio(
    source: Path,
    destination: Path,
    start_s: float,
    end_s: float,
) -> tuple[str, str]:
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(source),
        "-t", f"{end_s - start_s:.3f}", "-map_metadata", "-1", "-vn", "-ac", "2", "-ar", "48000",
        "-c:a", "libopus", "-b:a", "96k", "-vbr", "on", "-y", str(destination),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=180, check=False)
    except (OSError, subprocess.SubprocessError):
        raise EncounterError("passage audio extraction failed") from None
    if result.returncode != 0 or not destination.is_file() or destination.stat().st_size <= 0:
        raise EncounterError("passage audio extraction failed")
    os.chmod(destination, 0o600)
    return str(destination), _sha256_file(destination)


def _remove_artifact_directory(directory: Path) -> None:
    """Remove only opaque passage files created inside one known session directory."""
    if not directory.exists() or directory.is_symlink():
        return
    for child in directory.iterdir():
        if child.is_file() and not child.is_symlink():
            child.unlink()
    directory.rmdir()


def _lyrics_for_range(
    lines: Sequence[dict], start_s: float, end_s: float, lyrics_meta: dict,
) -> list[dict]:
    selected = []
    for line in lines:
        if line["end_s"] <= start_s or line["start_s"] >= end_s:
            continue
        selected.append(
            {
                "start_s": line["start_s"],
                "end_s": line["end_s"],
                "text": line["text"],
                "verified": lyrics_meta["verified"],
                "verification": lyrics_meta["verification"],
                "spans_boundary": line["start_s"] < start_s or line["end_s"] > end_s,
            }
        )
    return selected


@contextmanager
def _private_umask() -> Iterator[None]:
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    if db_path.is_symlink():
        raise EncounterError("encounter database must not be a symbolic link")
    with _private_umask():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.parent.stat().st_mode & 0o077:
            raise EncounterError("encounter database directory must be private (mode 0700)")
        connection = sqlite3.connect(db_path, timeout=15, isolation_level=None)
    try:
        os.chmod(db_path, 0o600)
    except OSError:
        connection.close()
        raise EncounterError("encounter database permissions could not be secured") from None
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=15000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            listener_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('listening','complete')),
            mode TEXT NOT NULL CHECK (mode IN ('fixed','adaptive')),
            created_at TEXT NOT NULL,
            completed_at TEXT,
            audio_sha256 TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            identity_json TEXT NOT NULL,
            lyrics_meta_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS passages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            token TEXT NOT NULL UNIQUE,
            evidence_json TEXT NOT NULL,
            lyrics_json TEXT NOT NULL,
            delivered_at TEXT,
            note TEXT,
            note_sha256 TEXT,
            noted_at TEXT,
            audio_artifact_path TEXT,
            audio_artifact_sha256 TEXT,
            audio_mime_type TEXT,
            UNIQUE(session_id, ordinal)
        );
        CREATE INDEX IF NOT EXISTS passages_pending
            ON passages(session_id, ordinal, note);
        CREATE TABLE IF NOT EXISTS journal_outbox (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            listener_id TEXT NOT NULL,
            record_kind TEXT NOT NULL CHECK (record_kind IN ('first_listen','retrospective')),
            payload_json TEXT NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            delivered_at TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_attempt_at TEXT,
            last_error TEXT,
            UNIQUE(session_id, record_kind)
        );
        CREATE INDEX IF NOT EXISTS journal_outbox_pending
            ON journal_outbox(listener_id, delivered_at, created_at);
        """
    )
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(sessions)")}
    if "retrospective" not in columns:
        connection.execute("ALTER TABLE sessions ADD COLUMN retrospective TEXT")
    if "retrospective_sha256" not in columns:
        connection.execute("ALTER TABLE sessions ADD COLUMN retrospective_sha256 TEXT")
    if "retrospective_at" not in columns:
        connection.execute("ALTER TABLE sessions ADD COLUMN retrospective_at TEXT")
    passage_columns = {row["name"] for row in connection.execute("PRAGMA table_info(passages)")}
    if "audio_artifact_path" not in passage_columns:
        connection.execute("ALTER TABLE passages ADD COLUMN audio_artifact_path TEXT")
    if "audio_artifact_sha256" not in passage_columns:
        connection.execute("ALTER TABLE passages ADD COLUMN audio_artifact_sha256 TEXT")
    if "audio_mime_type" not in passage_columns:
        connection.execute("ALTER TABLE passages ADD COLUMN audio_mime_type TEXT")
    return connection


def _bounded_record_content(text: str) -> str:
    if len(text) <= MAX_RECORD_CONTENT_CHARS:
        return text
    suffix = "\n\n[Full immutable journal remains in the private Music Encounter archive.]"
    return text[: MAX_RECORD_CONTENT_CHARS - len(suffix)].rstrip() + suffix


def _first_listen_payload(session: sqlite3.Row, rows: Sequence[sqlite3.Row]) -> dict:
    identity = _decode_json(session["identity_json"])
    journal = []
    for row in rows:
        journal.append(
            f"[{row['start_ms'] / 1000:g}–{row['end_ms'] / 1000:g}s]\n{row['note']}"
        )
    full_content = (
        f"First-listen journal — {identity['title']} by {identity['artist']}\n\n"
        + "\n\n".join(journal)
    )
    content = _bounded_record_content(full_content)
    record_hash = hashlib.sha256(
        _json({"identity": identity, "journal": journal}).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "first_listen",
        "content": content,
        "salience": 7,
        "source": "attune-encounter",
        "metadata": {
            "attune_encounter_version": 1,
            "encounter_kind": "first_listen",
            "session_id": session["id"],
            "record_hash": record_hash,
            "audio_sha256": session["audio_sha256"],
            "title": identity["title"],
            "artist": identity["artist"],
            "passage_count": len(rows),
            "duration_ms": session["duration_ms"],
            "evidence_version": 2,
            "passage_audio_sha256s": [row["audio_artifact_sha256"] for row in rows],
            "full_journal_local": len(full_content) > MAX_RECORD_CONTENT_CHARS,
        },
    }


def _retrospective_payload(session: sqlite3.Row, retrospective: str, salience: float) -> dict:
    identity = _decode_json(session["identity_json"])
    record_hash = hashlib.sha256(retrospective.encode("utf-8")).hexdigest()
    return {
        "kind": "retrospective",
        "content": _bounded_record_content(
            f"Post-listen retrospective — {identity['title']} by {identity['artist']}\n\n{retrospective}"
        ),
        "salience": salience,
        "source": "attune-encounter",
        "metadata": {
            "attune_encounter_version": 1,
            "encounter_kind": "retrospective",
            "session_id": session["id"],
            "record_hash": record_hash,
            "audio_sha256": session["audio_sha256"],
            "title": identity["title"],
            "artist": identity["artist"],
        },
    }


def _queue_journal(
    connection: sqlite3.Connection,
    session: sqlite3.Row,
    record_kind: str,
    payload: dict,
) -> None:
    connection.execute(
        """INSERT OR IGNORE INTO journal_outbox
           (id,session_id,listener_id,record_kind,payload_json,idempotency_key,created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            str(uuid.uuid4()), session["id"], session["listener_id"], record_kind,
            _json(payload), f"attune-encounter:{session['id']}:{record_kind}", _utc_now(),
        ),
    )


def _session(connection: sqlite3.Connection, session_id: str, listener_id: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if row is None:
        raise EncounterError("encounter session not found")
    if row["listener_id"] != _validate_listener(listener_id):
        raise EncounterError("encounter belongs to another listener")
    return row


def prepare_encounter(
    db_path: Path,
    audio_path: Path,
    listener_id: str,
    identity: dict,
    lyrics_manifest: dict,
    *,
    mode: str = "adaptive",
) -> dict:
    """Trusted preparation entry point. Returned data contains no song identity."""
    listener_id = _validate_listener(listener_id)
    identity = _validate_identity(identity)
    if mode not in {"fixed", "adaptive"}:
        raise EncounterError("mode must be fixed or adaptive")
    samples, sample_rate = decode_audio(Path(audio_path))
    duration_s = len(samples) / sample_rate
    lyrics_meta, lyric_lines = _validate_lyrics(lyrics_manifest, duration_s)
    ranges = passage_ranges(samples, sample_rate, mode)
    if not ranges:
        raise EncounterError("audio produced no passages")
    vocal_evidence = _vocal_passage_evidence(Path(audio_path), ranges, sample_rate)

    session_id = str(uuid.uuid4())
    artifact_directory = _artifact_root(Path(db_path)) / session_id
    with _private_umask():
        artifact_directory.mkdir(mode=0o700)
    prepared = []
    try:
        for ordinal, (start, end) in enumerate(ranges, 1):
            start_s, end_s = start / sample_rate, end / sample_rate
            artifact_path, artifact_hash = _extract_passage_audio(
                Path(audio_path), artifact_directory / f"{ordinal:04d}.opus", start_s, end_s
            )
            prepared.append(
                (
                    str(uuid.uuid4()),
                    session_id,
                    ordinal,
                    round(start_s * 1000),
                    round(end_s * 1000),
                    secrets.token_urlsafe(32),
                    _json(_passage_evidence(
                        samples[start:end], sample_rate, start, vocal_evidence[ordinal - 1]
                    )),
                    _json(_lyrics_for_range(lyric_lines, start_s, end_s, lyrics_meta)),
                    artifact_path,
                    artifact_hash,
                    "audio/ogg",
                )
            )
    except Exception:
        _remove_artifact_directory(artifact_directory)
        raise

    connection = _connect(Path(db_path))
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT INTO sessions
               (id,listener_id,state,mode,created_at,completed_at,audio_sha256,
                duration_ms,identity_json,lyrics_meta_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id,
                listener_id,
                "listening",
                mode,
                _utc_now(),
                None,
                _sha256_file(Path(audio_path)),
                round(duration_s * 1000),
                _json(identity),
                _json(lyrics_meta),
            ),
        )
        connection.executemany(
            """INSERT INTO passages
               (id,session_id,ordinal,start_ms,end_ms,token,evidence_json,lyrics_json,
                audio_artifact_path,audio_artifact_sha256,audio_mime_type)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            prepared,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        _remove_artifact_directory(artifact_directory)
        raise
    finally:
        connection.close()
    return {
        "ready": True,
        "session_id": session_id,
        "listener_id": listener_id,
        "instruction": "Use next_passage. Recording identity remains hidden until finish.",
    }


def passage_audio(
    db_path: Path,
    session_id: str,
    listener_id: str,
    passage_id: str,
) -> tuple[bytes, str, str]:
    """Read one delivered passage artifact without exposing its filesystem path."""
    connection = _connect(Path(db_path))
    try:
        _session(connection, session_id, listener_id)
        row = connection.execute(
            """SELECT audio_artifact_path,audio_artifact_sha256,audio_mime_type,delivered_at
               FROM passages WHERE session_id=? AND id=?""",
            (session_id, passage_id),
        ).fetchone()
        if row is None or row["delivered_at"] is None:
            raise EncounterError("passage audio is unavailable")
        if not row["audio_artifact_path"] or not row["audio_artifact_sha256"]:
            raise EncounterError("passage audio artifact is missing")
        path = Path(row["audio_artifact_path"])
        root = _artifact_root(Path(db_path)).resolve()
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            raise EncounterError("passage audio artifact is outside the private archive") from None
        if path.is_symlink() or resolved.suffix != ".opus" or resolved.stat().st_size > 20 * 1024 * 1024:
            raise EncounterError("passage audio artifact is invalid")
        data = resolved.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["audio_artifact_sha256"]:
            raise EncounterError("passage audio artifact failed integrity verification")
        return data, row["audio_mime_type"] or "audio/ogg", row["audio_artifact_sha256"]
    finally:
        connection.close()


def next_passage(db_path: Path, session_id: str, listener_id: str) -> dict:
    connection = _connect(Path(db_path))
    try:
        connection.execute("BEGIN IMMEDIATE")
        session = _session(connection, session_id, listener_id)
        row = connection.execute(
            "SELECT * FROM passages WHERE session_id=? AND note IS NULL ORDER BY ordinal LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            connection.commit()
            return {"complete": True, "session_id": session_id, "instruction": "Use finish_encounter."}
        if row["delivered_at"] is None:
            connection.execute("UPDATE passages SET delivered_at=? WHERE id=?", (_utc_now(), row["id"]))
        connection.commit()
        return {
            "complete": False,
            "session_id": session_id,
            "passage_id": row["id"],
            "token": row["token"],
            "window": {"start_s": row["start_ms"] / 1000, "end_s": row["end_ms"] / 1000},
            "evidence": _decode_json(row["evidence_json"]),
            "lyrics": _decode_json(row["lyrics_json"]),
            "journal_prompt": (
                "Privately record what you noticed, felt, wondered, or expected from only this passage. "
                "Uncertainty and no strong reaction are valid. Do not infer the recording's identity."
            ),
        }
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def record_impression(
    db_path: Path,
    session_id: str,
    listener_id: str,
    token: str,
    note: str,
) -> dict:
    note = str(note or "").strip()
    if not note or len(note) > MAX_NOTE_CHARS:
        raise EncounterError(f"impression must contain 1–{MAX_NOTE_CHARS} characters")
    connection = _connect(Path(db_path))
    try:
        connection.execute("BEGIN IMMEDIATE")
        _session(connection, session_id, listener_id)
        row = connection.execute(
            "SELECT * FROM passages WHERE session_id=? AND token=?", (session_id, token)
        ).fetchone()
        if row is None or row["delivered_at"] is None:
            raise EncounterError("passage token was not delivered for this encounter")
        if row["note"] is not None:
            if row["note"] != note:
                raise EncounterError("a first impression cannot be rewritten")
            connection.commit()
            return {"saved": True, "duplicate": True, "passage_id": row["id"]}
        pending = connection.execute(
            "SELECT id FROM passages WHERE session_id=? AND note IS NULL ORDER BY ordinal LIMIT 1",
            (session_id,),
        ).fetchone()
        if pending is None or pending["id"] != row["id"]:
            raise EncounterError("passages must be acknowledged in encounter order")
        connection.execute(
            "UPDATE passages SET note=?,note_sha256=?,noted_at=? WHERE id=?",
            (note, hashlib.sha256(note.encode("utf-8")).hexdigest(), _utc_now(), row["id"]),
        )
        connection.commit()
        return {"saved": True, "duplicate": False, "passage_id": row["id"]}
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def finish_encounter(db_path: Path, session_id: str, listener_id: str) -> dict:
    connection = _connect(Path(db_path))
    try:
        connection.execute("BEGIN IMMEDIATE")
        session = _session(connection, session_id, listener_id)
        if connection.execute(
            "SELECT 1 FROM passages WHERE session_id=? AND note IS NULL LIMIT 1", (session_id,)
        ).fetchone():
            raise EncounterError("the whole recording remains hidden until every passage has an impression")
        if session["state"] != "complete":
            connection.execute(
                "UPDATE sessions SET state='complete',completed_at=? WHERE id=?", (_utc_now(), session_id)
            )
        rows = connection.execute(
            "SELECT * FROM passages WHERE session_id=? ORDER BY ordinal", (session_id,)
        ).fetchall()
        _queue_journal(connection, session, "first_listen", _first_listen_payload(session, rows))
        connection.commit()
        return {
            "complete": True,
            "session_id": session_id,
            "identity": _decode_json(session["identity_json"]),
            "duration_s": session["duration_ms"] / 1000,
            "audio_sha256": session["audio_sha256"],
            "lyrics_provenance": _decode_json(session["lyrics_meta_json"]),
            "journal": [
                {
                    "passage_id": row["id"],
                    "start_s": row["start_ms"] / 1000,
                    "end_s": row["end_ms"] / 1000,
                    "impression": row["note"],
                    "noted_at": row["noted_at"],
                    "evidence": _decode_json(row["evidence_json"]),
                    "lyrics": _decode_json(row["lyrics_json"]),
                }
                for row in rows
            ],
            "retrospective_prompt": (
                "The identity and complete first-impression journal are now available. "
                "Write a separate retrospective; do not revise the original impressions. "
                "The first-listen record is queued for journal export."
            ),
        }
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def store_retrospective(
    db_path: Path,
    session_id: str,
    listener_id: str,
    retrospective: str,
    *,
    salience: float = 7,
) -> dict:
    """Immutably store a post-reveal reading and queue it for journal export."""
    retrospective = str(retrospective or "").strip()
    if not retrospective or len(retrospective) > MAX_RETROSPECTIVE_CHARS:
        raise EncounterError(
            f"retrospective must contain 1–{MAX_RETROSPECTIVE_CHARS} characters"
        )
    if isinstance(salience, bool) or not isinstance(salience, (int, float)):
        raise EncounterError("salience must be a number from 0 to 10")
    salience = float(salience)
    if not math.isfinite(salience) or not 0 <= salience <= 10:
        raise EncounterError("salience must be a number from 0 to 10")

    connection = _connect(Path(db_path))
    try:
        connection.execute("BEGIN IMMEDIATE")
        session = _session(connection, session_id, listener_id)
        if session["state"] != "complete":
            raise EncounterError("retrospective is available only after the encounter is complete")
        if session["retrospective"] is not None:
            expected_payload = _json(_retrospective_payload(session, retrospective, salience))
            queued = connection.execute(
                """SELECT payload_json,delivered_at FROM journal_outbox
                   WHERE session_id=? AND record_kind='retrospective'""",
                (session_id,),
            ).fetchone()
            if (
                session["retrospective"] != retrospective
                or queued is None
                or queued["payload_json"] != expected_payload
            ):
                raise EncounterError("a retrospective cannot be rewritten")
            connection.commit()
            return {
                "saved": True,
                "duplicate": True,
                "session_id": session_id,
                "queued": queued["delivered_at"] is None,
            }
        connection.execute(
            """UPDATE sessions
               SET retrospective=?,retrospective_sha256=?,retrospective_at=? WHERE id=?""",
            (
                retrospective,
                hashlib.sha256(retrospective.encode("utf-8")).hexdigest(),
                _utc_now(),
                session_id,
            ),
        )
        _queue_journal(
            connection,
            session,
            "retrospective",
            _retrospective_payload(session, retrospective, salience),
        )
        connection.commit()
        return {"saved": True, "duplicate": False, "session_id": session_id, "queued": True}
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def pending_journal_count(db_path: Path, listener_id: str) -> int:
    listener_id = _validate_listener(listener_id)
    connection = _connect(Path(db_path))
    try:
        row = connection.execute(
            "SELECT count(*) AS count FROM journal_outbox WHERE listener_id=? AND delivered_at IS NULL",
            (listener_id,),
        ).fetchone()
        return int(row["count"])
    finally:
        connection.close()


def sync_journal_outbox(
    db_path: Path,
    listener_id: str,
    store: Callable[[dict, str], object],
    *,
    limit: int = 20,
) -> dict:
    """Deliver queued records through a caller-supplied journal transport.

    The remote idempotency key makes a retry safe if a process loses the response
    after the destination committed the record but before the local delivered flag
    was set.
    """
    listener_id = _validate_listener(listener_id)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise EncounterError("sync limit must be an integer from 1 to 100")
    connection = _connect(Path(db_path))
    delivered = 0
    failed = 0
    try:
        rows = connection.execute(
            """SELECT * FROM journal_outbox
               WHERE listener_id=? AND delivered_at IS NULL
               ORDER BY created_at,id LIMIT ?""",
            (listener_id, limit),
        ).fetchall()
        for row in rows:
            now = _utc_now()
            try:
                store(_decode_json(row["payload_json"]), row["idempotency_key"])
            except Exception as exc:
                failed += 1
                # Store only a bounded exception class/message; credentials never
                # belong in transport exceptions or in the outbox database.
                safe_error = type(exc).__name__
                connection.execute(
                    """UPDATE journal_outbox SET attempts=attempts+1,last_attempt_at=?,last_error=?
                       WHERE id=? AND delivered_at IS NULL""",
                    (now, safe_error, row["id"]),
                )
                continue
            delivered += 1
            connection.execute(
                """UPDATE journal_outbox
                   SET attempts=attempts+1,last_attempt_at=?,last_error=NULL,delivered_at=?
                   WHERE id=? AND delivered_at IS NULL""",
                (now, now, row["id"]),
            )
        remaining = connection.execute(
            "SELECT count(*) AS count FROM journal_outbox WHERE listener_id=? AND delivered_at IS NULL",
            (listener_id,),
        ).fetchone()["count"]
        return {
            "success": failed == 0,
            "delivered": delivered,
            "failed": failed,
            "remaining": int(remaining),
        }
    finally:
        connection.close()
