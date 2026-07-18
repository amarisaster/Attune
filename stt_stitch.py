"""Attune — deterministic overlap-dedup stitching for chunked webhook STT.

Pure, no I/O, no ffmpeg, no network: given the transcripts of sequential
overlapping audio chunks (see server.py's _split_chunks/_webhook_stt_chunked),
concatenates them in order, stripping the duplicated overlap off the START of
each transcript after the first. This is a HEURISTIC, not real alignment: it
compares the tail of the already-stitched result against the head of the next
chunk's transcript, looks for the longest matching suffix/prefix run
(case/punctuation-insensitive), and drops that many words off the front of
the next chunk before joining. Kept intentionally simple and deterministic —
no fuzzy/edit-distance matching, no external deps. Standalone importable
(like singing.py) so test_stitch.py can exercise it without booting the rest
of server.py.

CONSERVATIVE STITCHING POLICY: a match is only trusted enough to dedup when
it is BOTH long enough to be a genuine overlap AND short enough to plausibly
fit in the actual overlap window between two chunks:
  - minimum match: _MIN_WORD_MATCH words (whitespace scripts) /
    _MIN_CJK_CHAR_MATCH chars (CJK fallback) -- below this, a match is too
    likely to be coincidental (e.g. "the chorus" repeating in a song) to
    justify deleting anything.
  - maximum match: derived from the caller-supplied `overlap_seconds` (the
    actual overlap between adjacent chunks) at a generous assumed ceiling of
    ~4 words/sec (~7 chars/sec for CJK) of spoken audio -- a match longer
    than what could plausibly have been spoken in the overlap window is more
    likely two genuinely different (but textually similar) passages than a
    real duplicate, e.g. a repeated chorus line.
When a candidate match falls outside [minimum, maximum], nothing is deleted
-- the chunks are concatenated in full. For singing/lyrics in particular, a
duplicated half-line is far less harmful than a deleted chorus, so this
module always errs toward keeping text over dropping it.

For scripts where whitespace doesn't delimit words (CJK etc — a word-split
comparison there would almost always find zero overlap even when the chunks
plainly repeat the same text), this falls back to a raw string
suffix/prefix comparison instead.

GAP_MARKER: server.py inserts this exact string in place of a chunk whose
transcription persistently failed (see _webhook_stt_chunked). Stitching
NEVER dedups across a gap boundary -- a gap means real audio content is
unknown, and heuristically matching across it could hide or corrupt genuine
words on either side. Boundaries adjacent to a gap marker are always
concatenated as-is.
"""

from typing import List

GAP_MARKER = '[…]'

_MIN_WORD_MATCH = 3
_MIN_CJK_CHAR_MATCH = 6
_WORDS_PER_SEC_CAP = 4.0
_CHARS_PER_SEC_CAP = 7.0
# Matches server.py's CONFIG default for stt_chunk_overlap_seconds -- used
# only when a caller doesn't pass overlap_seconds explicitly.
_DEFAULT_OVERLAP_SECONDS = 2.0


def _normalize_word(w: str) -> str:
    """Casefold and strip punctuation for comparison only -- the original
    (un-normalized) words are what actually get kept/joined."""
    return ''.join(ch for ch in w.lower() if ch.isalnum())


def _looks_wordless(text: str) -> bool:
    """True when the text has content but no whitespace at all -- the signal
    that a whitespace word-split won't isolate meaningful units (CJK and
    similar unspaced scripts), so the caller should use the raw
    character-suffix/prefix fallback instead."""
    stripped = text.strip()
    return bool(stripped) and not any(c.isspace() for c in stripped)


def _word_overlap_len(prev_words: List[str], next_words: List[str]) -> int:
    """Longest run (checked from the full overlap of both lists down to 1) of
    the tail of prev_words matching the head of next_words, normalized.
    Returns 0 if no run of any length matches. Not capped here -- the cap is
    applied by the caller so an over-long match can be distinguished from "no
    match" and handled as "don't dedup" rather than silently truncated."""
    max_k = min(len(prev_words), len(next_words))
    for k in range(max_k, 0, -1):
        prev_tail = [_normalize_word(w) for w in prev_words[-k:]]
        next_head = [_normalize_word(w) for w in next_words[:k]]
        if all(prev_tail) and prev_tail == next_head:
            return k
    return 0


def _char_overlap_len(prev_text: str, next_text: str) -> int:
    """Longest run of prev_text's tail exactly matching next_text's head --
    the fallback for wordless scripts. Not capped here, same reasoning as
    _word_overlap_len."""
    max_k = min(len(prev_text), len(next_text))
    for k in range(max_k, 0, -1):
        if prev_text[-k:] == next_text[:k]:
            return k
    return 0


def stitch_transcripts(chunk_transcripts: List[str], overlap_seconds: float = _DEFAULT_OVERLAP_SECONDS) -> str:
    """Concatenate sequential overlapping-chunk transcripts into one string,
    deduplicating each chunk boundary's overlap when (and only when) the
    matched span is plausibly a real duplicate -- see module docstring for
    the exact min/max policy and the GAP_MARKER no-dedup rule. Empty/
    whitespace-only chunks (other than GAP_MARKER itself) are dropped
    entirely -- they contribute nothing and don't interfere with the dedup
    comparison between their real neighbors.

    `overlap_seconds` is the actual overlap window between adjacent chunks
    (ATTUNE_STT_CHUNK_OVERLAP_SECONDS in server.py) and bounds how long a
    match is allowed to be before it's treated as coincidental rather than
    genuine overlap. Deterministic heuristic; no fuzzy/edit-distance
    matching, no external deps."""
    parts = [t if t == GAP_MARKER else t.strip() for t in chunk_transcripts if t and t.strip()]
    if not parts:
        return ''
    if len(parts) == 1:
        return parts[0]

    overlap_seconds = max(0.0, overlap_seconds)
    # If the configured overlap is so small that even the minimum-confidence
    # match couldn't plausibly come from overlapping audio, disable dedup
    # entirely (cap below min => the [min, cap] window is empty). Never
    # force the cap UP to the minimum — that would let a 3-word match delete
    # text at zero overlap, exactly the false-dedup this guards against.
    word_cap = int(_WORDS_PER_SEC_CAP * overlap_seconds)
    char_cap = int(_CHARS_PER_SEC_CAP * overlap_seconds)

    result = parts[0]
    prev_was_gap = (parts[0] == GAP_MARKER)
    for next_text in parts[1:]:
        next_is_gap = (next_text == GAP_MARKER)

        if prev_was_gap or next_is_gap:
            # Hard boundary: never dedup across a gap in either direction --
            # the gap means real content is unknown, so any apparent overlap
            # match here is coincidental, not evidence of duplication.
            sep = '' if not result or result.endswith((' ', '\n')) or next_text.startswith((' ', '\n')) else ' '
            result = result + sep + next_text
            prev_was_gap = next_is_gap
            continue

        prev_text = result
        wordless = _looks_wordless(prev_text) or _looks_wordless(next_text)
        if wordless:
            k = _char_overlap_len(prev_text, next_text)
            if k < _MIN_CJK_CHAR_MATCH or k > char_cap:
                k = 0  # too short to trust, or too long to plausibly fit the overlap window
            remainder = next_text[k:].lstrip()
        else:
            prev_words = prev_text.split()
            next_words = next_text.split()
            k = _word_overlap_len(prev_words, next_words)
            if k < _MIN_WORD_MATCH or k > word_cap:
                k = 0
            remainder = ' '.join(next_words[k:])
        if not remainder:
            prev_was_gap = False
            continue
        if wordless:
            result = result + remainder
        else:
            result = result + (' ' if result and not result.endswith((' ', '\n')) else '') + remainder
        prev_was_gap = False
    return result
