"""Trusted YouTube ingestion for blind Music Encounter preparation."""

from __future__ import annotations

import json
import html
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from .core import EncounterError, MAX_AUDIO_BYTES, MAX_AUDIO_SECONDS, prepare_encounter


YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
LRC_TIMESTAMP_RE = re.compile(r"\[(\d{1,3}):(\d{2}(?:\.\d{1,3})?)\]")
VTT_TIMESTAMP_RE = re.compile(
    r"(?:(\d{1,3}):)?(\d{2}):(\d{2}(?:\.\d{1,3})?)\s+-->\s+"
    r"(?:(\d{1,3}):)?(\d{2}):(\d{2}(?:\.\d{1,3})?)"
)
CAPTION_TAG_RE = re.compile(r"<[^>]+>")
ALLOWED_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be",
}


def validate_youtube_url(value: str) -> str:
    """Accept one ordinary YouTube video URL, never playlists or arbitrary hosts."""
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        raise EncounterError("invalid YouTube URL") from None
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in ALLOWED_HOSTS or parsed.username or parsed.password:
        raise EncounterError("only HTTPS YouTube video URLs are accepted")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    elif parsed.path.startswith("/shorts/") or parsed.path.startswith("/live/"):
        video_id = parsed.path.split("/")[2]
    else:
        raise EncounterError("YouTube URL must identify one video")
    if not YOUTUBE_ID_RE.fullmatch(video_id):
        raise EncounterError("YouTube URL has an invalid video id")
    return f"https://www.youtube.com/watch?v={video_id}"


def _run_json(command: list[str], timeout: int) -> dict:
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        raise EncounterError("yt-dlp is not installed") from None
    except subprocess.TimeoutExpired:
        raise EncounterError("YouTube import timed out") from None
    if result.returncode != 0:
        raise EncounterError("YouTube metadata could not be retrieved")
    try:
        value = json.loads(result.stdout)
    except (UnicodeError, json.JSONDecodeError):
        raise EncounterError("YouTube returned invalid metadata") from None
    if not isinstance(value, dict):
        raise EncounterError("YouTube returned invalid metadata")
    return value


def inspect_youtube(url: str) -> tuple[str, dict]:
    canonical = validate_youtube_url(url)
    metadata = _run_json([
        "yt-dlp", "--no-playlist", "--skip-download", "--no-warnings",
        "--dump-single-json", canonical,
    ], 90)
    video_id = str(metadata.get("id") or "")
    duration = metadata.get("duration")
    if not YOUTUBE_ID_RE.fullmatch(video_id):
        raise EncounterError("YouTube metadata did not identify one video")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
        raise EncounterError("YouTube video duration is unavailable")
    if duration <= 0 or duration > MAX_AUDIO_SECONDS:
        raise EncounterError(f"YouTube audio must be at most {MAX_AUDIO_SECONDS // 60} minutes")
    return canonical, metadata


def identity_from_youtube(metadata: dict) -> dict:
    title = str(metadata.get("track") or metadata.get("title") or "").strip()
    artist = str(
        metadata.get("artist") or metadata.get("creator") or metadata.get("uploader") or ""
    ).strip()
    if not title or not artist:
        raise EncounterError("YouTube metadata requires a usable title and artist")
    identity: dict[str, object] = {
        "title": title,
        "artist": artist,
        "catalog_id": f"youtube:{metadata['id']}",
    }
    album = str(metadata.get("album") or "").strip()
    if album:
        identity["album"] = album
    date = str(metadata.get("release_date") or metadata.get("upload_date") or "")
    if len(date) >= 4 and date[:4].isdigit():
        identity["release_year"] = int(date[:4])
    return identity


def download_youtube_audio(canonical_url: str, directory: Path) -> Path:
    output = directory / "source.%(ext)s"
    try:
        result = subprocess.run([
            "yt-dlp", "--no-playlist", "--no-warnings", "--restrict-filenames",
            "--max-filesize", str(MAX_AUDIO_BYTES), "-f", "bestaudio/best",
            "-o", str(output), "--print", "after_move:filepath", canonical_url,
        ], check=False, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        raise EncounterError("yt-dlp is not installed") from None
    except subprocess.TimeoutExpired:
        raise EncounterError("YouTube audio download timed out") from None
    if result.returncode != 0:
        raise EncounterError("YouTube audio could not be downloaded")
    paths = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not paths:
        paths = sorted(directory.glob("source.*"))
    if len(paths) != 1:
        raise EncounterError("YouTube import produced an unexpected file set")
    path = paths[0].resolve()
    if path.parent != directory.resolve() or not path.is_file() or path.is_symlink():
        raise EncounterError("YouTube audio artifact is invalid")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_AUDIO_BYTES:
        raise EncounterError("YouTube audio exceeds the file-size limit")
    return path


def _parse_lrc(value: str, duration_s: float) -> list[dict]:
    timed: list[tuple[float, str]] = []
    for raw in value.splitlines():
        text = LRC_TIMESTAMP_RE.sub("", raw).strip()
        if not text:
            continue
        for minute, second in LRC_TIMESTAMP_RE.findall(raw):
            start = int(minute) * 60 + float(second)
            if 0 <= start < duration_s:
                timed.append((start, text))
    timed.sort(key=lambda item: item[0])
    deduplicated: list[tuple[float, str]] = []
    for item in timed:
        if not deduplicated or item != deduplicated[-1]:
            deduplicated.append(item)
    lines = []
    for index, (start, text) in enumerate(deduplicated):
        following = deduplicated[index + 1][0] if index + 1 < len(deduplicated) else duration_s
        end = min(duration_s, max(start + 0.25, following))
        lines.append({"start_s": round(start, 3), "end_s": round(end, 3), "text": text})
    return lines


def lyrics_from_lrclib(
    identity: dict, duration_s: float,
    opener: Callable[..., object] = urlopen,
) -> dict:
    query = urlencode({
        "track_name": identity["title"], "artist_name": identity["artist"],
        "duration": round(duration_s),
    })
    request = Request(
        f"https://lrclib.net/api/get?{query}",
        headers={"User-Agent": "Attune/1.0 (sequential listening importer)"},
    )
    try:
        response = opener(request, timeout=20)
        raw = response.read(2 * 1024 * 1024 + 1)
    except (HTTPError, URLError, TimeoutError):
        raise EncounterError("verified timed lyrics were not found; provide --lyrics-file") from None
    if len(raw) > 2 * 1024 * 1024:
        raise EncounterError("lyrics response is too large")
    try:
        record = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        raise EncounterError("lyrics service returned invalid data") from None
    if not isinstance(record, dict):
        raise EncounterError("lyrics service returned invalid data")
    record_id = record.get("id")
    if record.get("instrumental") is True:
        lines: list[dict] = []
    else:
        synced = record.get("syncedLyrics")
        if not isinstance(synced, str) or not synced.strip():
            raise EncounterError("verified timed lyrics were not found; provide --lyrics-file")
        lines = _parse_lrc(synced, duration_s)
        if not lines:
            raise EncounterError("verified timed lyrics were not found; provide --lyrics-file")
    return {
        "verified": True,
        "verification": "verified",
        "source": f"LRCLIB record {record_id}; source-grounded timed lyrics matched by title, artist, and duration",
        "lines": lines,
    }


def _caption_seconds(hour: str | None, minute: str, second: str) -> float:
    return (int(hour or 0) * 3600) + (int(minute) * 60) + float(second)


def _clean_caption_text(value: str) -> str:
    text = html.unescape(CAPTION_TAG_RE.sub("", value))
    return " ".join(text.replace("\u200b", "").split()).strip()


def _normalise_caption_lines(rows: list[tuple[float, float, str]], duration_s: float) -> list[dict]:
    lines: list[dict] = []
    for start, end, raw_text in sorted(rows, key=lambda item: (item[0], item[1])):
        text = _clean_caption_text(raw_text)
        if not text or not math.isfinite(start) or not math.isfinite(end) or start < 0:
            continue
        start = min(start, duration_s)
        end = min(duration_s, max(start + 0.25, end))
        if start >= duration_s or end <= start:
            continue
        if lines and lines[-1]["text"] == text and start <= lines[-1]["end_s"] + 0.25:
            lines[-1]["end_s"] = round(max(lines[-1]["end_s"], end), 3)
            continue
        lines.append({"start_s": round(start, 3), "end_s": round(end, 3), "text": text[:2000]})
    return lines


def _parse_youtube_json3(value: str, duration_s: float) -> list[dict]:
    try:
        payload = json.loads(value)
    except (UnicodeError, json.JSONDecodeError):
        return []
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return []
    rows: list[tuple[float, float, str]] = []
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("segs"), list):
            continue
        try:
            start = float(event.get("tStartMs", 0)) / 1000
            duration = float(event.get("dDurationMs", 0)) / 1000
        except (TypeError, ValueError):
            continue
        text = "".join(str(segment.get("utf8") or "") for segment in event["segs"] if isinstance(segment, dict))
        rows.append((start, start + max(duration, 0.25), text))
    return _normalise_caption_lines(rows, duration_s)


def _parse_youtube_vtt(value: str, duration_s: float) -> list[dict]:
    rows: list[tuple[float, float, str]] = []
    blocks = re.split(r"\r?\n\s*\r?\n", value)
    for block in blocks:
        block_lines = block.splitlines()
        timing_index = next((index for index, line in enumerate(block_lines) if "-->" in line), None)
        if timing_index is None:
            continue
        match = VTT_TIMESTAMP_RE.search(block_lines[timing_index])
        if not match:
            continue
        start = _caption_seconds(match.group(1), match.group(2), match.group(3))
        end = _caption_seconds(match.group(4), match.group(5), match.group(6))
        rows.append((start, end, " ".join(block_lines[timing_index + 1:])))
    return _normalise_caption_lines(rows, duration_s)


def _caption_language(tracks: dict, metadata: dict) -> str | None:
    usable = [str(key) for key, value in tracks.items() if key != "live_chat" and isinstance(value, list) and value]
    if not usable:
        return None
    preferences = [metadata.get("language"), metadata.get("original_language"), "en"]
    for preference in preferences:
        if isinstance(preference, str) and preference in usable:
            return preference
    english = next((key for key in usable if key.lower().startswith("en")), None)
    return english or sorted(usable)[0]


def captions_from_youtube(
    canonical_url: str, metadata: dict, duration_s: float, directory: Path,
) -> dict:
    """Return creator, automatic, or explicitly unavailable caption provenance."""
    manual = metadata.get("subtitles") if isinstance(metadata.get("subtitles"), dict) else {}
    automatic = metadata.get("automatic_captions") if isinstance(metadata.get("automatic_captions"), dict) else {}
    language = _caption_language(manual, metadata)
    is_automatic = False
    tracks = manual
    if language is None:
        tracks = automatic
        language = _caption_language(automatic, metadata)
        is_automatic = language is not None
    video_id = str(metadata.get("id") or "unknown")
    if language is None:
        return {
            "verified": False,
            "verification": "unavailable",
            "source": f"No LRCLIB match or YouTube captions were available for video {video_id}",
            "lines": [],
        }

    output = directory / "captions.%(ext)s"
    command = [
        "yt-dlp", "--no-playlist", "--skip-download", "--no-warnings",
        "--sub-langs", language, "--sub-format", "json3/vtt/best",
        "--write-auto-subs" if is_automatic else "--write-subs",
        "-o", str(output), canonical_url,
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result = None
    candidates = sorted(directory.glob("captions*.json3")) + sorted(directory.glob("captions*.vtt"))
    lines: list[dict] = []
    if result is not None and result.returncode == 0 and candidates:
        path = candidates[0].resolve()
        if path.parent == directory.resolve() and path.is_file() and not path.is_symlink() and path.stat().st_size <= 5 * 1024 * 1024:
            raw = path.read_text(encoding="utf-8", errors="replace")
            lines = _parse_youtube_json3(raw, duration_s) if path.suffix == ".json3" else _parse_youtube_vtt(raw, duration_s)
    if not lines:
        return {
            "verified": False,
            "verification": "unavailable",
            "source": f"YouTube captions for video {video_id} could not be converted into timed lines",
            "lines": [],
        }
    if is_automatic:
        return {
            "verified": False,
            "verification": "automatic",
            "source": f"YouTube automatic captions ({language}) for video {video_id}; may contain transcription errors",
            "lines": lines,
        }
    return {
        "verified": True,
        "verification": "verified",
        "source": f"YouTube creator-provided captions ({language}) for video {video_id}",
        "lines": lines,
    }


def prepare_youtube_encounter(
    db_path: Path,
    youtube_url: str,
    listener_id: str,
    *,
    identity_override: dict | None = None,
    lyrics_manifest: dict | None = None,
    mode: str = "adaptive",
) -> dict:
    """Download one trusted source transiently, then retain only blind passages."""
    canonical, metadata = inspect_youtube(youtube_url)
    identity = identity_from_youtube(metadata)
    if identity_override:
        if not isinstance(identity_override, dict):
            raise EncounterError("identity override must be an object")
        identity.update(identity_override)
    duration = float(metadata["duration"])
    with tempfile.TemporaryDirectory(prefix="attune-encounter-youtube-") as temporary:
        directory = Path(temporary)
        if lyrics_manifest:
            lyrics = lyrics_manifest
        else:
            try:
                lyrics = lyrics_from_lrclib(identity, duration)
            except EncounterError:
                lyrics = captions_from_youtube(canonical, metadata, duration, directory)
        audio = download_youtube_audio(canonical, directory)
        return prepare_encounter(
            Path(db_path), audio, listener_id, identity, lyrics, mode=mode,
        )
