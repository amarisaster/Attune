"""Listener-scoped sequential music encounters for Attune."""

from .core import (
    EncounterError,
    finish_encounter,
    next_passage,
    passage_audio,
    pending_journal_count,
    prepare_encounter,
    record_impression,
    store_retrospective,
    sync_journal_outbox,
)
from .youtube import prepare_youtube_encounter

__all__ = [
    "EncounterError",
    "finish_encounter",
    "next_passage",
    "passage_audio",
    "pending_journal_count",
    "prepare_encounter",
    "prepare_youtube_encounter",
    "record_impression",
    "store_retrospective",
    "sync_journal_outbox",
]
