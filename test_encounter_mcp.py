from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import server


class EncounterMCPTests(unittest.TestCase):
    def test_public_tool_set_is_complete_and_listener_is_not_caller_controlled(self) -> None:
        tools = {tool["name"]: tool for tool in server._mcp_encounter_tool_defs()}
        self.assertEqual(
            set(tools),
            {
                "music_prepare_encounter",
                "music_next_passage",
                "music_record_impression",
                "music_finish_encounter",
                "music_store_retrospective",
            },
        )
        for tool in tools.values():
            self.assertNotIn("listener_id", tool["inputSchema"].get("properties", {}))

    def test_next_passage_attaches_audio_without_exposing_another_namespace(self) -> None:
        packet = {"complete": False, "session_id": "session", "passage_id": "passage"}
        with patch.dict(server.CONFIG, {"encounter_listener_id": "server-listener"}), \
                patch("attune_encounter.next_passage", return_value=packet.copy()) as next_mock, \
                patch(
                    "attune_encounter.passage_audio",
                    return_value=(b"audio-bytes", "audio/ogg", "a" * 64),
                ) as audio_mock:
            result, audio = asyncio.run(server._mcp_encounter_call(
                "music_next_passage", {"session_id": "session", "listener_id": "attacker"}
            ))

        self.assertEqual(audio, (b"audio-bytes", "audio/ogg"))
        self.assertTrue(result["audio"]["attached"])
        next_mock.assert_called_once_with(
            server.CONFIG["encounter_db"], "session", "server-listener"
        )
        audio_mock.assert_called_once_with(
            server.CONFIG["encounter_db"], "session", "server-listener", "passage"
        )


if __name__ == "__main__":
    unittest.main()
