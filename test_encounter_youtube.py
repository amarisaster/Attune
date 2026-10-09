from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from attune_encounter.core import EncounterError
from attune_encounter.youtube import (
    _parse_youtube_json3,
    _parse_youtube_vtt,
    captions_from_youtube,
    identity_from_youtube,
    lyrics_from_lrclib,
    prepare_youtube_encounter,
    validate_youtube_url,
)


class FakeResponse:
    def __init__(self, value: dict):
        self.value = json.dumps(value).encode()

    def read(self, limit: int) -> bytes:
        return self.value


class YouTubeImportTests(unittest.TestCase):
    def test_url_validation_accepts_video_forms_and_discards_playlist_context(self) -> None:
        expected = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(validate_youtube_url("https://youtu.be/dQw4w9WgXcQ?t=42"), expected)
        self.assertEqual(
            validate_youtube_url("https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=private"),
            expected,
        )

    def test_url_validation_rejects_non_youtube_and_credentials(self) -> None:
        for value in (
            "http://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com.evil.invalid/watch?v=dQw4w9WgXcQ",
            "https://user:pass@youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/playlist?list=dQw4w9WgXcQ",
        ):
            with self.subTest(value=value), self.assertRaises(EncounterError):
                validate_youtube_url(value)

    def test_identity_prefers_music_metadata_and_keeps_catalog_provenance(self) -> None:
        identity = identity_from_youtube({
            "id": "dQw4w9WgXcQ", "title": "Video title", "uploader": "Channel",
            "track": "Exact track", "artist": "Exact artist", "album": "Album",
            "release_date": "20260102",
        })
        self.assertEqual(identity["title"], "Exact track")
        self.assertEqual(identity["artist"], "Exact artist")
        self.assertEqual(identity["catalog_id"], "youtube:dQw4w9WgXcQ")
        self.assertEqual(identity["release_year"], 2026)

    def test_lrclib_lines_are_timed_and_source_grounded(self) -> None:
        record = {
            "id": 42,
            "instrumental": False,
            "syncedLyrics": "[00:02.00]First line\n[00:07.50]Second line",
        }
        lyrics = lyrics_from_lrclib(
            {"title": "Track", "artist": "Artist"}, 12,
            opener=lambda request, timeout: FakeResponse(record),
        )
        self.assertTrue(lyrics["verified"])
        self.assertIn("LRCLIB record 42", lyrics["source"])
        self.assertEqual(lyrics["lines"], [
            {"start_s": 2.0, "end_s": 7.5, "text": "First line"},
            {"start_s": 7.5, "end_s": 12, "text": "Second line"},
        ])

    def test_youtube_caption_parsers_preserve_timing_and_clean_markup(self) -> None:
        json3 = json.dumps({"events": [
            {"tStartMs": 2000, "dDurationMs": 3000, "segs": [{"utf8": "First "}, {"utf8": "line"}]},
            {"tStartMs": 5000, "dDurationMs": 2500, "segs": [{"utf8": "Second line"}]},
        ]})
        self.assertEqual(_parse_youtube_json3(json3, 10), [
            {"start_s": 2.0, "end_s": 5.0, "text": "First line"},
            {"start_s": 5.0, "end_s": 7.5, "text": "Second line"},
        ])
        vtt = """WEBVTT

00:00:01.000 --> 00:00:03.500
<c.colorE5E5E5>Hello &amp; goodbye</c>
"""
        self.assertEqual(_parse_youtube_vtt(vtt, 10), [
            {"start_s": 1.0, "end_s": 3.5, "text": "Hello & goodbye"},
        ])

    def test_caption_fallback_labels_manual_automatic_and_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)

            def fake_run(*args, **kwargs):
                (directory / "captions.en.json3").write_text(json.dumps({"events": [
                    {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "A line"}]},
                ]}))
                return type("Result", (), {"returncode": 0})()

            with patch("attune_encounter.youtube.subprocess.run", side_effect=fake_run):
                manual = captions_from_youtube(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    {"id": "dQw4w9WgXcQ", "subtitles": {"en": [{}]}}, 10, directory,
                )
            self.assertTrue(manual["verified"])
            self.assertEqual(manual["verification"], "verified")
            self.assertIn("creator-provided", manual["source"])

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            def fake_auto_run(*args, **kwargs):
                (directory / "captions.en.json3").write_text(json.dumps({"events": [
                    {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "Auto line"}]},
                ]}))
                return type("Result", (), {"returncode": 0})()

            with patch("attune_encounter.youtube.subprocess.run", side_effect=fake_auto_run):
                automatic = captions_from_youtube(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    {"id": "dQw4w9WgXcQ", "automatic_captions": {"en": [{}]}}, 10, directory,
                )
            self.assertFalse(automatic["verified"])
            self.assertEqual(automatic["verification"], "automatic")

        unavailable = captions_from_youtube(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            {"id": "dQw4w9WgXcQ"}, 10, Path("/unused"),
        )
        self.assertEqual(unavailable["verification"], "unavailable")
        self.assertEqual(unavailable["lines"], [])

    def test_import_uses_transient_audio_and_returns_only_blind_result(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            observed: dict = {}

            def fake_download(url: str, directory: Path) -> Path:
                path = directory / "source.webm"
                path.write_bytes(b"temporary")
                observed["path"] = path
                return path

            def fake_prepare(db, audio, listener, identity, lyrics, *, mode):
                self.assertTrue(Path(audio).exists())
                observed.update({"identity": identity, "lyrics": lyrics, "mode": mode})
                return {"ready": True, "session_id": "opaque", "listener_id": listener}

            with (
                patch("attune_encounter.youtube.inspect_youtube", return_value=(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    {"id": "dQw4w9WgXcQ", "duration": 120, "track": "Track", "artist": "Artist"},
                )),
                patch("attune_encounter.youtube.download_youtube_audio", side_effect=fake_download),
                patch("attune_encounter.youtube.prepare_encounter", side_effect=fake_prepare),
            ):
                result = prepare_youtube_encounter(
                    Path(root) / "encounters.sqlite", "https://youtu.be/dQw4w9WgXcQ", "listener-a",
                    lyrics_manifest={"verified": True, "source": "checked", "lines": []},
                )
            self.assertEqual(result["session_id"], "opaque")
            self.assertFalse(observed["path"].exists())
            self.assertEqual(observed["identity"]["catalog_id"], "youtube:dQw4w9WgXcQ")


if __name__ == "__main__":
    unittest.main()
