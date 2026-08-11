"""Tests for lyrics.py — LRC parsing and card formatting, fully offline.

Run: python test_lyrics.py
Network lookup (fetch_lyrics) is deliberately untested here: tests must not
depend on lrclib.net being reachable. Parsing and rendering carry the logic.
"""

import sys

from lyrics import format_lyrics_section, parse_lrc


def test_parse_lrc():
    lrc = (
        '[00:07.58] Maluma, baby\n'
        '[00:09.30]Apenas sale el sol\n'
        '[01:02.5] Second minute line\n'
        'not a timestamp line\n'
        '[00:55] No-millis line\n'
        '[00:58.123]   \n'  # blank text -> dropped
    )
    lines = parse_lrc(lrc)
    assert [ln['text'] for ln in lines] == [
        'Maluma, baby', 'Apenas sale el sol', 'No-millis line', 'Second minute line']
    assert lines[0]['time'] == 7.58
    assert lines[2]['time'] == 55.0
    assert abs(lines[3]['time'] - 62.5) < 0.01
    print('PASS parse: ordering, optional millis, blank/garbage dropped')


def test_parse_empty():
    assert parse_lrc('') == []
    assert parse_lrc(None) == []
    print('PASS parse: empty input')


def test_format_synced():
    lyr = {
        'track': 'Felices los 4', 'artist': 'Maluma',
        'instrumental': False, 'synced': True,
        'lines': [
            {'time': 7.58, 'text': 'Maluma, baby'},
            {'time': 55.62, 'text': 'Felices los 4'},
            {'time': 187.0, 'text': 'Last chorus'},
        ],
    }
    card = format_lyrics_section(lyr, peak_t=186.0, section_changes=[36.0])
    assert 'LYRICS: Felices los 4 — Maluma [lrclib]' in card
    assert '0:08  Maluma, baby' in card
    assert '— section change ~0:36 —' in card
    assert '3:07  Last chorus   ← energy peak' in card
    print('PASS format synced:\n' + card)


def test_format_instrumental_and_plain():
    inst = format_lyrics_section({'track': 'T', 'artist': 'A', 'instrumental': True,
                                  'synced': False, 'lines': []})
    assert 'marked instrumental' in inst
    plain = format_lyrics_section({'track': 'T', 'artist': 'A', 'instrumental': False,
                                   'synced': False,
                                   'lines': [{'time': None, 'text': 'only words'}]})
    assert 'only words' in plain and 'no timestamps' in plain
    empty = format_lyrics_section(None)
    assert empty == ''
    print('PASS format: instrumental, plain, empty')


if __name__ == '__main__':
    test_parse_lrc()
    test_parse_empty()
    test_format_synced()
    test_format_instrumental_and_plain()
    print('\nALL LYRICS TESTS PASSED')
    sys.exit(0)
