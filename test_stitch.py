"""Plain-assert tests for stt_stitch.stitch_transcripts. Run directly:
    venv\\Scripts\\python.exe test_stitch.py
No pytest, no I/O, no ffmpeg -- exercises the pure stitching heuristic used
to dedupe overlap between chunked webhook-STT transcripts.
"""

import sys

from stt_stitch import GAP_MARKER, stitch_transcripts


def case_overlap_dedup_english():
    print('--- case (a): english overlap (>= min, <= cap) is deduplicated ---')
    chunks = [
        'the quick brown fox jumps over the lazy dog',
        'Jumps over the lazy dog, and runs away fast',
    ]
    out = stitch_transcripts(chunks)
    print('  result:', out)
    assert out == 'the quick brown fox jumps over the lazy dog and runs away fast', out
    print('PASS case (a)')


def case_cjk_fallback():
    print('--- case (b): CJK (no whitespace) falls back to char suffix/prefix ---')
    chunks = [
        '今日は天気がとても良いですね',
        'とても良いですねので散歩に行きました',
    ]
    out = stitch_transcripts(chunks)
    print('  result:', out)
    assert out == '今日は天気がとても良いですねので散歩に行きました', out
    print('PASS case (b)')


def case_no_overlap_concat():
    print('--- case (c): chunks with no shared words just concatenate with a space ---')
    chunks = ['hello there', 'completely different words']
    out = stitch_transcripts(chunks)
    print('  result:', out)
    assert out == 'hello there completely different words', out
    print('PASS case (c)')


def case_empty_chunk_handling():
    print('--- case (d): empty/whitespace-only chunks are dropped, not joined as gaps ---')
    assert stitch_transcripts([]) == ''
    assert stitch_transcripts(['', '  ', '\n']) == ''
    assert stitch_transcripts(['only one']) == 'only one'
    assert stitch_transcripts(['', 'leading empty chunk']) == 'leading empty chunk'
    assert stitch_transcripts(['trailing empty chunk', '   ']) == 'trailing empty chunk'
    # 3-word overlap ("chunk here now") clears the new minimum, so this still
    # dedups exactly as before -- empty chunks in between are still dropped
    # cleanly and don't interfere with the real neighbors' overlap match.
    out = stitch_transcripts(['first chunk here now', '', '   ', 'chunk here now continues on'])
    print('  result:', out)
    assert out == 'first chunk here now continues on', out
    print('PASS case (d)')


def case_repeated_chorus_below_minimum():
    print('--- case (e): repeated chorus -- genuine overlap is only 2 words (< 3-word '
          'minimum), so NOTHING is deleted ---')
    chunks = [
        'we sang the chorus',
        'the chorus begins again and fades',
    ]
    out = stitch_transcripts(chunks)
    print('  result:', out)
    # "the chorus" (2 words) matches but is below _MIN_WORD_MATCH=3, so the
    # policy refuses to dedup -- both copies of "the chorus" survive. A
    # duplicated half-line beats a deleted chorus.
    assert out == 'we sang the chorus the chorus begins again and fades', out
    print('PASS case (e)')


def case_match_exceeds_overlap_window_cap():
    print('--- case (f): a 9-word match exceeds the ~8-word cap for a 2s overlap window '
          '-- too long to be a real duplicate, so nothing is deleted ---')
    chunks = [
        'start w1 w2 w3 w4 w5 w6 w7 w8 w9',
        'w1 w2 w3 w4 w5 w6 w7 w8 w9 end',
    ]
    out = stitch_transcripts(chunks, overlap_seconds=2)
    print('  result:', out)
    assert out == 'start w1 w2 w3 w4 w5 w6 w7 w8 w9 w1 w2 w3 w4 w5 w6 w7 w8 w9 end', out
    print('PASS case (f)')

    print('  ...but the same 9-word match DOES dedup when overlap_seconds is wide '
          'enough to raise the cap above it')
    out2 = stitch_transcripts(chunks, overlap_seconds=3)  # cap = max(3, 4*3) = 12
    print('  result:', out2)
    assert out2 == 'start w1 w2 w3 w4 w5 w6 w7 w8 w9 end', out2
    print('PASS case (f) cap-widened variant')


def case_gap_marker_never_dedups():
    print('--- case (g): GAP_MARKER boundaries never dedup, even with a genuine overlap '
          'on either side ---')
    chunks = [
        'we sang the whole song together',
        GAP_MARKER,
        'the whole song together was great',
    ]
    out = stitch_transcripts(chunks)
    print('  result:', out)
    assert GAP_MARKER in out, out
    # Both "the whole song together" runs survive in full -- the stitcher
    # never reaches into a chunk adjacent to a gap for overlap dedup.
    assert out == f'we sang the whole song together {GAP_MARKER} the whole song together was great', out
    print('PASS case (g)')


if __name__ == '__main__':
    case_overlap_dedup_english()
    case_cjk_fallback()
    case_no_overlap_concat()
    case_empty_chunk_handling()
    case_repeated_chorus_below_minimum()
    case_match_exceeds_overlap_window_cap()
    case_gap_marker_never_dedups()
    print('ALL PASS')
    sys.exit(0)
