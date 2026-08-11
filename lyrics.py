"""Lyrics layer for Attune's music perception — LRCLIB lookup + LRC parsing.

LRCLIB (https://lrclib.net) is a public, keyless synced-lyrics database.
This module talks to that one fixed host and nothing else — the same
locked-outbound posture as the STT webhook. It never fetches audio, never
follows a caller-supplied URL, and every failure degrades to "no lyrics
section" rather than failing the analysis (fail-soft, like singing/music).

Attune cannot know a song's title from an audio URL, so lyrics only appear
when the caller passes track + artist to analyze_music. Words come from the
database, not from STT — running speech-to-text over a full mix hallucinates
(the exact failure energy_gate.py exists to prevent on voice notes), and a
lyrics database is simply the more honest source for released songs.

Stdlib + nothing. Standalone-importable; never imports server.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

LRCLIB_BASE = 'https://lrclib.net/api'
TIMEOUT_S = 10
USER_AGENT = 'Attune/1.0 (https://github.com/amarisaster/Attune)'

# "[mm:ss.xx] text" — LRC timestamp lines. Milliseconds part optional.
_LRC_LINE = re.compile(r'^\[(\d+):(\d{2})(?:\.(\d{1,3}))?\]\s?(.*)$')


def _http_get_json(url: str):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode('utf-8'))


def parse_lrc(synced: str) -> list:
    """LRC text -> [{'time': seconds, 'text': str}], sorted, blanks dropped."""
    out = []
    for raw in (synced or '').splitlines():
        m = _LRC_LINE.match(raw.strip())
        if not m:
            continue
        minutes, seconds, frac, text = m.groups()
        t = int(minutes) * 60 + int(seconds)
        if frac:
            t += int(frac.ljust(3, '0')) / 1000.0
        text = text.strip()
        if text:
            out.append({'time': round(t, 2), 'text': text})
    out.sort(key=lambda ln: ln['time'])
    return out


def fetch_lyrics(track: str, artist: str, duration_s: float | None = None):
    """Best-effort LRCLIB lookup. Returns
    {'track', 'artist', 'instrumental', 'synced': bool, 'lines': [...]} or
    None when nothing matched. Never raises on network trouble — callers
    treat None and exceptions identically (no lyrics section).
    """
    track = (track or '').strip()
    artist = (artist or '').strip()
    if not track or not artist:
        return None

    params = {'track_name': track, 'artist_name': artist}
    if duration_s:
        params['duration'] = str(int(round(duration_s)))
    try:
        data = _http_get_json(f'{LRCLIB_BASE}/get?{urllib.parse.urlencode(params)}')
    except Exception:
        data = None
    if not data:
        # Fuzzy fallback: /api/search, take the first hit with any lyrics.
        try:
            q = urllib.parse.urlencode({'q': f'{track} {artist}'})
            results = _http_get_json(f'{LRCLIB_BASE}/search?{q}')
        except Exception:
            return None
        data = next((r for r in (results or [])
                     if r.get('syncedLyrics') or r.get('plainLyrics')
                     or r.get('instrumental')), None)
        if not data:
            return None

    if data.get('instrumental'):
        return {'track': data.get('trackName', track), 'artist': data.get('artistName', artist),
                'instrumental': True, 'synced': False, 'lines': []}

    synced = data.get('syncedLyrics') or ''
    lines = parse_lrc(synced)
    if lines:
        return {'track': data.get('trackName', track), 'artist': data.get('artistName', artist),
                'instrumental': False, 'synced': True, 'lines': lines}
    plain = (data.get('plainLyrics') or '').strip()
    if plain:
        lines = [{'time': None, 'text': ln} for ln in plain.splitlines() if ln.strip()]
        return {'track': data.get('trackName', track), 'artist': data.get('artistName', artist),
                'instrumental': False, 'synced': False, 'lines': lines}
    return None


def _fmt_t(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f'{m}:{s:02d}'


def format_lyrics_section(lyr: dict, peak_t: float | None = None,
                          section_changes: list | None = None) -> str:
    """Render the LYRICS card section. Synced lines carry their timestamp on
    the shared clock; the energy peak and section changes are annotated
    inline so words and measurements read as one timeline. Source-tagged —
    the words come from a database, not from listening."""
    if not lyr:
        return ''
    head = f"LYRICS: {lyr['track']} — {lyr['artist']} [lrclib]"
    if lyr.get('instrumental'):
        return head + '\n  (marked instrumental — no words)'
    lines_out = [head]
    changes = sorted(section_changes or [])
    peak_done = peak_t is None
    for ln in lyr['lines']:
        t = ln['time']
        if t is None:
            lines_out.append(f'  {ln["text"]}')
            continue
        while changes and t >= changes[0]:
            lines_out.append(f'  — section change ~{_fmt_t(changes[0])} —')
            changes.pop(0)
        marker = ''
        if not peak_done and t >= peak_t:
            marker = '   ← energy peak'
            peak_done = True
        lines_out.append(f'  {_fmt_t(t)}  {ln["text"]}{marker}')
    if not lyr.get('synced'):
        lines_out.append('  (plain lyrics — no timestamps available)')
    return '\n'.join(lines_out)
