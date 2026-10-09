from __future__ import annotations

from array import array
import concurrent.futures
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
import wave

from attune_encounter.core import (
    ANALYSIS_SAMPLE_RATE,
    EncounterError,
    finish_encounter,
    next_passage,
    passage_audio,
    passage_ranges,
    prepare_encounter,
    pending_journal_count,
    record_impression,
    store_retrospective,
    sync_journal_outbox,
)


def tone(seconds: float, *, hz: float = 220.0, amplitude: float = 0.2, rate: int = ANALYSIS_SAMPLE_RATE) -> array:
    return array("f", (amplitude * math.sin(2 * math.pi * hz * i / rate) for i in range(round(seconds * rate))))


def joined(*parts: array) -> array:
    result = array("f")
    for part in parts:
        result.extend(part)
    return result


def write_wav(path: Path, samples: array, rate: int = ANALYSIS_SAMPLE_RATE) -> None:
    ints = [max(-32768, min(32767, round(float(value) * 32767))) for value in samples]
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(struct.pack(f"<{len(ints)}h", *ints))


class EncounterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "private" / "encounters.sqlite"
        self.audio = self.root / "obvious-song-name.wav"
        write_wav(self.audio, tone(125))
        self.identity = {
            "title": "The Hidden Song",
            "artist": "Secret Artist",
            "album": "Surprise",
            "source_filename": self.audio.name,
        }
        self.lyrics = {
            "verified": True,
            "source": "human checked",
            "lines": [
                {"start_s": 2, "end_s": 7, "text": "opening words"},
                {"start_s": 58, "end_s": 63, "text": "crossing words"},
                {"start_s": 90, "end_s": 94, "text": "future words"},
            ],
        }

    def prepare(self, *, listener: str = "listener-a", mode: str = "fixed") -> dict:
        return prepare_encounter(
            self.db, self.audio, listener, self.identity, self.lyrics, mode=mode
        )

    def complete(self, *, listener: str = "listener-a") -> dict:
        ready = self.prepare(listener=listener)
        while True:
            packet = next_passage(self.db, ready["session_id"], listener)
            if packet["complete"]:
                break
            record_impression(
                self.db, ready["session_id"], listener, packet["token"],
                f"Private reading at {packet['window']['start_s']}.",
            )
        finish_encounter(self.db, ready["session_id"], listener)
        return ready

    def test_fixed_blind_delivery_and_causal_lyrics(self) -> None:
        ready = self.prepare()
        encoded_ready = json.dumps(ready)
        self.assertNotIn("Hidden", encoded_ready)
        self.assertNotIn("Secret", encoded_ready)
        self.assertNotIn("125", encoded_ready)
        packet = next_passage(self.db, ready["session_id"], "listener-a")
        encoded = json.dumps(packet)
        self.assertNotIn("The Hidden Song", encoded)
        self.assertNotIn("Secret Artist", encoded)
        self.assertNotIn("future words", encoded)
        self.assertNotIn("human checked", encoded)
        self.assertEqual(packet["window"], {"start_s": 0.0, "end_s": 60.0})
        self.assertEqual([line["text"] for line in packet["lyrics"]], ["opening words", "crossing words"])
        self.assertTrue(packet["lyrics"][-1]["spans_boundary"])
        self.assertNotIn("passage_count", packet)
        self.assertNotIn("duration_s", packet)
        with sqlite3.connect(self.db) as connection:
            future_id = connection.execute(
                "SELECT id FROM passages WHERE session_id=? AND ordinal=2",
                (ready["session_id"],),
            ).fetchone()[0]
        with self.assertRaisesRegex(EncounterError, "unavailable"):
            passage_audio(self.db, ready["session_id"], "listener-a", future_id)
        audio, mime_type, digest = passage_audio(
            self.db, ready["session_id"], "listener-a", packet["passage_id"]
        )
        self.assertEqual(mime_type, "audio/ogg")
        self.assertGreater(len(audio), 100)
        self.assertEqual(len(digest), 64)
        with sqlite3.connect(self.db) as connection:
            artifact_path = Path(connection.execute(
                "SELECT audio_artifact_path FROM passages WHERE id=?", (packet["passage_id"],)
            ).fetchone()[0])
        artifact_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(EncounterError, "integrity"):
            passage_audio(self.db, ready["session_id"], "listener-a", packet["passage_id"])

    def test_pending_packet_replays_after_reopen(self) -> None:
        ready = self.prepare()
        first = next_passage(self.db, ready["session_id"], "listener-a")
        replay = next_passage(self.db, ready["session_id"], "listener-a")
        self.assertEqual(first, replay)

    def test_note_retry_is_idempotent_and_immutable(self) -> None:
        ready = self.prepare()
        packet = next_passage(self.db, ready["session_id"], "listener-a")
        saved = record_impression(self.db, ready["session_id"], "listener-a", packet["token"], "I expected a lift.")
        self.assertFalse(saved["duplicate"])
        duplicate = record_impression(self.db, ready["session_id"], "listener-a", packet["token"], "I expected a lift.")
        self.assertTrue(duplicate["duplicate"])
        with self.assertRaisesRegex(EncounterError, "cannot be rewritten"):
            record_impression(self.db, ready["session_id"], "listener-a", packet["token"], "I knew the ending.")

    def test_concurrent_exact_note_retries_write_once(self) -> None:
        ready = self.prepare()
        packet = next_passage(self.db, ready["session_id"], "listener-a")

        def save(_: int) -> dict:
            return record_impression(
                self.db, ready["session_id"], "listener-a", packet["token"], "The same private reading."
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(save, range(4)))
        self.assertEqual(sum(not result["duplicate"] for result in results), 1)
        self.assertEqual(sum(result["duplicate"] for result in results), 3)

    def test_listener_isolation_fails_closed(self) -> None:
        ready = self.prepare(listener="listener-c")
        with self.assertRaisesRegex(EncounterError, "another listener"):
            next_passage(self.db, ready["session_id"], "listener-d")

    def test_finish_is_gated_then_reveals_identity(self) -> None:
        ready = self.prepare()
        with self.assertRaisesRegex(EncounterError, "remains hidden"):
            finish_encounter(self.db, ready["session_id"], "listener-a")
        while True:
            packet = next_passage(self.db, ready["session_id"], "listener-a")
            if packet["complete"]:
                break
            record_impression(
                self.db,
                ready["session_id"],
                "listener-a",
                packet["token"],
                f"Private reading at {packet['window']['start_s']}.",
            )
        result = finish_encounter(self.db, ready["session_id"], "listener-a")
        self.assertEqual(result["identity"]["title"], "The Hidden Song")
        self.assertEqual(result["lyrics_provenance"]["source"], "human checked")
        self.assertEqual(len(result["journal"]), 3)
        self.assertIn("separate retrospective", result["retrospective_prompt"])

    def test_private_database_permissions(self) -> None:
        self.prepare()
        self.assertEqual(os.stat(self.db).st_mode & 0o777, 0o600)

    def test_public_database_directory_is_rejected(self) -> None:
        public = self.root / "public"
        public.mkdir(mode=0o755)
        os.chmod(public, 0o755)
        with self.assertRaisesRegex(EncounterError, "directory must be private"):
            prepare_encounter(
                public / "encounters.sqlite",
                self.audio,
                "listener-a",
                self.identity,
                self.lyrics,
                mode="fixed",
            )

    def test_lyrics_require_verification_and_order(self) -> None:
        bad = dict(self.lyrics)
        bad["verified"] = False
        with self.assertRaisesRegex(EncounterError, "verified, automatic, or explicitly unavailable"):
            prepare_encounter(self.db, self.audio, "listener-a", self.identity, bad, mode="fixed")
        bad = dict(self.lyrics)
        bad["lines"] = list(reversed(self.lyrics["lines"]))
        with self.assertRaisesRegex(EncounterError, "ordered"):
            prepare_encounter(self.db, self.audio, "listener-a", self.identity, bad, mode="fixed")

    def test_automatic_and_unavailable_lyrics_keep_truthful_provenance(self) -> None:
        automatic = {
            "verified": False,
            "verification": "automatic",
            "source": "YouTube automatic captions; may contain errors",
            "lines": [{"start_s": 2, "end_s": 7, "text": "machine transcript"}],
        }
        ready = prepare_encounter(
            self.db, self.audio, "listener-a", self.identity, automatic, mode="fixed",
        )
        packet = next_passage(self.db, ready["session_id"], "listener-a")
        self.assertFalse(packet["lyrics"][0]["verified"])
        self.assertEqual(packet["lyrics"][0]["verification"], "automatic")

        unavailable_db = self.root / "unavailable" / "encounters.sqlite"
        unavailable = {
            "verified": False,
            "verification": "unavailable",
            "source": "No timed lyrics available",
            "lines": [],
        }
        ready = prepare_encounter(
            unavailable_db, self.audio, "listener-b", self.identity, unavailable, mode="fixed",
        )
        self.assertEqual(next_passage(unavailable_db, ready["session_id"], "listener-b")["lyrics"], [])

    def test_adaptive_range_and_causal_prefix(self) -> None:
        prefix = joined(tone(55, hz=180, amplitude=0.03), tone(5, hz=900, amplitude=0.8))
        left = joined(prefix, tone(80, hz=220, amplitude=0.1))
        right = joined(prefix, tone(80, hz=1600, amplitude=0.7))
        left_ranges = passage_ranges(left, ANALYSIS_SAMPLE_RATE, "adaptive")
        right_ranges = passage_ranges(right, ANALYSIS_SAMPLE_RATE, "adaptive")
        self.assertEqual(left_ranges[0], right_ranges[0])
        first_seconds = left_ranges[0][1] / ANALYSIS_SAMPLE_RATE
        self.assertGreaterEqual(first_seconds, 45)
        self.assertLessEqual(first_seconds, 90)
        for start, end in left_ranges[:-1]:
            duration = (end - start) / ANALYSIS_SAMPLE_RATE
            self.assertGreaterEqual(duration, 45)
            self.assertLessEqual(duration, 90)

    def test_fixed_passages_are_sixty_seconds(self) -> None:
        ranges = passage_ranges(tone(125), ANALYSIS_SAMPLE_RATE, "fixed")
        self.assertEqual([(a / ANALYSIS_SAMPLE_RATE, b / ANALYSIS_SAMPLE_RATE) for a, b in ranges],
                         [(0.0, 60.0), (60.0, 120.0), (120.0, 125.0)])

    def test_finish_queues_generic_journal_record(self) -> None:
        ready = self.complete(listener="listener-c")
        self.assertEqual(pending_journal_count(self.db, "listener-c"), 1)
        seen = []

        def store(payload: dict, key: str) -> None:
            seen.append((payload, key))

        result = sync_journal_outbox(self.db, "listener-c", store)
        self.assertEqual(result, {"success": True, "delivered": 1, "failed": 0, "remaining": 0})
        payload, key = seen[0]
        self.assertEqual(payload["kind"], "first_listen")
        self.assertEqual(payload["source"], "attune-encounter")
        self.assertNotIn("drawer", payload)
        self.assertEqual(payload["metadata"]["encounter_kind"], "first_listen")
        self.assertEqual(payload["metadata"]["session_id"], ready["session_id"])
        self.assertEqual(key, f"attune-encounter:{ready['session_id']}:first_listen")

    def test_retrospective_is_separate_immutable_music_record(self) -> None:
        ready = self.complete()
        saved = store_retrospective(
            self.db, ready["session_id"], "listener-a", "Now that I know the song, its restraint reads differently.",
            salience=8,
        )
        self.assertFalse(saved["duplicate"])
        duplicate = store_retrospective(
            self.db, ready["session_id"], "listener-a", "Now that I know the song, its restraint reads differently.",
            salience=8,
        )
        self.assertTrue(duplicate["duplicate"])
        with self.assertRaisesRegex(EncounterError, "cannot be rewritten"):
            store_retrospective(self.db, ready["session_id"], "listener-a", "A rewritten reading.")
        with self.assertRaisesRegex(EncounterError, "cannot be rewritten"):
            store_retrospective(
                self.db, ready["session_id"], "listener-a",
                "Now that I know the song, its restraint reads differently.", salience=2,
            )
        self.assertEqual(pending_journal_count(self.db, "listener-a"), 2)

        seen = []
        sync_journal_outbox(self.db, "listener-a", lambda payload, key: seen.append((payload, key)))
        self.assertEqual(
            [payload["metadata"]["encounter_kind"] for payload, _ in seen],
            ["first_listen", "retrospective"],
        )
        self.assertEqual(seen[1][0]["salience"], 8.0)

    def test_outbox_retries_without_losing_or_duplicating_queue_rows(self) -> None:
        ready = self.complete(listener="listener-d")
        calls = []

        def unavailable(payload: dict, key: str) -> None:
            calls.append(key)
            raise RuntimeError("temporary outage")

        failed = sync_journal_outbox(self.db, "listener-d", unavailable)
        self.assertEqual(failed, {"success": False, "delivered": 0, "failed": 1, "remaining": 1})
        self.assertEqual(pending_journal_count(self.db, "listener-d"), 1)
        delivered = sync_journal_outbox(self.db, "listener-d", lambda payload, key: calls.append(key))
        self.assertEqual(delivered["delivered"], 1)
        self.assertEqual(calls[0], calls[1])
        with sqlite3.connect(self.db) as connection:
            attempts, rows = connection.execute(
                "SELECT attempts,count(*) FROM journal_outbox WHERE session_id=?",
                (ready["session_id"],),
            ).fetchone()
        self.assertEqual((attempts, rows), (2, 1))


if __name__ == "__main__":
    unittest.main()
