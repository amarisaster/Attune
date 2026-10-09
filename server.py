"""Attune — voice-note acoustic + singing analysis.

Thin FastAPI wrapper around the vendored seven-ears engine (Seven Verity +
Sunny, MIT, pinned d33e7c1). Acoustics always run locally. Speech-to-text is
pluggable — local faster-whisper, a webhook to your own STT service, or none
at all (acoustics-only) — see the CONFIGURATION section below and README.md.
"""

import asyncio
import base64
import contextlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
SEVEN_EARS_SCRIPT = BASE_DIR / 'vendor' / 'seven-ears' / 'seven_ears_card.py'
# The server always runs inside the venv it was installed to, so
# sys.executable IS that venv's interpreter -- portable across platforms
# (no more hardcoded venv/Scripts/python.exe, which only existed on Windows).
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# CONFIGURATION — the one place config is resolved. Precedence: environment
# variable, then attune.config.json beside this file (gitignored — see
# attune.config.example.json for the template), then the default below.
# Nothing here is deployment-specific; every value that used to be hardcoded
# lives here now. Validated at the bottom of this section; the resolved
# profile (never secrets) is logged to stderr on boot.
# ---------------------------------------------------------------------------


def _load_json_config() -> dict:
    path = BASE_DIR / 'attune.config.json'
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        print(f'[attune] WARNING: could not parse {path}: {e} -- ignoring', file=sys.stderr, flush=True)
        return {}


_JSON_CONFIG = _load_json_config()


def _cfg_str(env_name: str, json_key: str, default: str = '') -> str:
    # An env var that IS set (even to '' / whitespace) is an explicit
    # override that short-circuits JSON -- same rule _cfg_list already uses
    # for an explicit empty env var meaning "no items", not "fall through".
    # For a scalar there's no "no items" state, so explicit-empty here means
    # "use the default", still without consulting JSON.
    v = os.environ.get(env_name)
    if v is not None:
        v = v.strip()
        return v if v != '' else default
    v = _JSON_CONFIG.get(json_key)
    if isinstance(v, str) and v.strip() != '':
        return v.strip()
    return default


def _cfg_int(env_name: str, json_key: str, default: int) -> int:
    # Same explicit-env-wins-and-skips-JSON rule as _cfg_str above. An
    # invalid int also warns AND uses the default directly (matching the
    # warning text) rather than silently falling through to JSON.
    v = os.environ.get(env_name)
    if v is not None:
        v = v.strip()
        if v == '':
            return default
        try:
            return int(v)
        except ValueError:
            print(f'[attune] WARNING: {env_name}={v!r} is not an integer, using default {default}',
                  file=sys.stderr, flush=True)
            return default
    v = _JSON_CONFIG.get(json_key)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    return default


def _cfg_float(env_name: str, json_key: str, default: float) -> float:
    # Same explicit-env-wins-and-skips-JSON rule as _cfg_int above.
    v = os.environ.get(env_name)
    if v is not None:
        v = v.strip()
        if v == '':
            return default
        try:
            return float(v)
        except ValueError:
            print(f'[attune] WARNING: {env_name}={v!r} is not a number, using default {default}',
                  file=sys.stderr, flush=True)
            return default
    v = _JSON_CONFIG.get(json_key)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return default


def _cfg_list(env_name: str, json_key: str, default=()) -> list:
    v = os.environ.get(env_name)
    if v is not None:
        items = [p.strip() for p in v.split(',') if p.strip()]
        if items:
            return items
        return []  # explicit empty env var means "no prefixes", not "fall through"
    v = _JSON_CONFIG.get(json_key)
    if isinstance(v, list) and v:
        return [str(p).strip() for p in v if str(p).strip()]
    return list(default)


CONFIG = {
    'port': _cfg_int('ATTUNE_PORT', 'port', 8452),
    'token_file': Path(_cfg_str('ATTUNE_TOKEN_FILE', 'token_file', str(Path.home() / '.attune-token'))).expanduser(),
    # Optional. Empty means "rely on PATH" -- /health reports discoverability.
    # expanduser'd the same way token_file is, so a '~/...' value works here too.
    'ffmpeg_dir': (lambda v: str(Path(v).expanduser()) if v else v)(
        _cfg_str('ATTUNE_FFMPEG_DIR', 'ffmpeg_dir', '')),
    # REQUIRED for the MCP url tool. Empty list disables analyze_voice_note
    # with a clear message rather than fetching from anywhere on request --
    # never ship an open fetch-anything endpoint.
    'allowed_audio_prefixes': _cfg_list('ATTUNE_ALLOWED_AUDIO_PREFIXES', 'allowed_audio_prefixes', []),
    'stt_mode': _cfg_str('ATTUNE_STT_MODE', 'stt_mode', 'local').lower(),
    'whisper_model': _cfg_str('ATTUNE_WHISPER_MODEL', 'whisper_model', 'base.en'),
    'stt_url': _cfg_str('ATTUNE_STT_URL', 'stt_url', ''),
    'stt_token_file': _cfg_str('ATTUNE_STT_TOKEN_FILE', 'stt_token_file', ''),
    'stt_max_bytes': _cfg_int('ATTUNE_STT_MAX_BYTES', 'stt_max_bytes', 6 * 1024 * 1024),
    # Chunked webhook STT (used only in stt_mode=webhook, only for audio
    # longer than _STT_CHUNK_TRIGGER_S -- see _resolve_stt/_webhook_stt_chunked).
    'stt_chunk_seconds': _cfg_int('ATTUNE_STT_CHUNK_SECONDS', 'stt_chunk_seconds', 28),
    'stt_chunk_overlap_seconds': _cfg_int('ATTUNE_STT_CHUNK_OVERLAP_SECONDS', 'stt_chunk_overlap_seconds', 2),
    'stt_max_chunks': _cfg_int('ATTUNE_STT_MAX_CHUNKS', 'stt_max_chunks', 20),
    # Below this RMS level (dBFS), webhook STT is skipped for that audio --
    # near-silent audio has no linguistic content to condition on, and
    # whisper-class STT hallucinates fluent-sounding invented text rather
    # than reporting silence (see energy_gate.py's docstring). Applies to
    # both single-shot short clips and, per-chunk, the chunked path.
    'stt_silence_dbfs': _cfg_float('ATTUNE_STT_SILENCE_DBFS', 'stt_silence_dbfs', -50.0),
    'max_upload_bytes': _cfg_int('ATTUNE_MAX_UPLOAD_BYTES', 'max_upload_bytes', 50 * 1024 * 1024),
    # transcript rides argv; Windows CreateProcess caps ~32k
    'max_transcript_chars': _cfg_int('ATTUNE_MAX_TRANSCRIPT_CHARS', 'max_transcript_chars', 16 * 1024),
    # Voice drop page (GET /drop, POST /api/drop, GET /drops/<name>) -- see
    # README's "Voice drop page" section.
    'drops_dir': _cfg_str('ATTUNE_DROPS_DIR', 'drops_dir', './drops'),
    'drops_keep': _cfg_int('ATTUNE_DROPS_KEEP', 'drops_keep', 200),
    # Empty means "derive from the request's Host header" -- see
    # _drop_public_url. Set this explicitly when Attune sits behind a
    # reverse proxy / tunnel that terminates TLS or rewrites Host, so
    # returned drop links match the externally reachable origin.
    'public_base_url': _cfg_str('ATTUNE_PUBLIC_BASE_URL', 'public_base_url', ''),
    # Sequential first-listen storage. The MCP uses one server-configured
    # namespace because Attune authentication is one shared bearer token, not
    # an account/identity system.
    'encounter_db': Path(_cfg_str(
        'ATTUNE_ENCOUNTER_DB', 'encounter_db',
        str(Path.home() / '.attune' / 'encounters.sqlite'),
    )).expanduser(),
    'encounter_listener_id': _cfg_str(
        'ATTUNE_ENCOUNTER_LISTENER_ID', 'encounter_listener_id', 'default-listener'
    ),
}

if CONFIG['stt_mode'] not in ('local', 'webhook', 'none'):
    print(f"[attune] FATAL: ATTUNE_STT_MODE={CONFIG['stt_mode']!r} must be one of "
          "'local', 'webhook', 'none'", file=sys.stderr, flush=True)
    raise SystemExit(1)

if CONFIG['stt_mode'] == 'webhook' and not CONFIG['stt_url']:
    print('[attune] WARNING: ATTUNE_STT_MODE=webhook but ATTUNE_STT_URL is unset -- '
          'webhook transcription will silently no-op', file=sys.stderr, flush=True)

print(
    '[attune] config: port={port} token_file={token_file} ffmpeg_dir={ffmpeg_dir} '
    'stt_mode={stt_mode} whisper_model={whisper_model} stt_url={stt_url} '
    'stt_silence_dbfs={stt_silence_dbfs} '
    'allowed_audio_prefixes={n_prefixes} configured max_upload_bytes={max_upload_bytes} '
    'drops_dir={drops_dir} drops_keep={drops_keep} public_base_url={public_base_url}'.format(
        port=CONFIG['port'],
        token_file=CONFIG['token_file'],
        ffmpeg_dir=CONFIG['ffmpeg_dir'] or '(relying on PATH)',
        stt_mode=CONFIG['stt_mode'],
        whisper_model=CONFIG['whisper_model'] if CONFIG['stt_mode'] == 'local' else '-',
        stt_url=CONFIG['stt_url'] or '(unset)',
        stt_silence_dbfs=CONFIG['stt_silence_dbfs'],
        n_prefixes=len(CONFIG['allowed_audio_prefixes']),
        max_upload_bytes=CONFIG['max_upload_bytes'],
        drops_dir=CONFIG['drops_dir'],
        drops_keep=CONFIG['drops_keep'],
        public_base_url=CONFIG['public_base_url'] or '(derived from Host header)',
    ),
    file=sys.stderr, flush=True,
)

# singing.py resolves its own ffmpeg dir (env var, then its own
# attune.config.json read) since it must stay usable standalone by
# test_singing.py without importing this file. Propagate our resolution into
# the environment so both modules agree even when the value only came from
# the JSON file.
if CONFIG['ffmpeg_dir']:
    os.environ.setdefault('ATTUNE_FFMPEG_DIR', CONFIG['ffmpeg_dir'])

from singing import analyze_singing, format_singing_section, load_wav  # noqa: E402  (after config)
from stt_stitch import GAP_MARKER, stitch_transcripts  # noqa: E402  (after config)
from energy_gate import should_skip_stt  # noqa: E402  (after config)
from voice_delivery import analyze_voice_delivery, format_voice_delivery_section  # noqa: E402
import music  # noqa: E402  (after config; pure numpy — safe at boot, no model deps)
import lyrics as lyrics_mod  # noqa: E402  (stdlib-only; one fixed outbound host, lrclib.net)

# ---------------------------------------------------------------------------
# END CONFIGURATION
# ---------------------------------------------------------------------------

PORT = CONFIG['port']
TOKEN_FILE = CONFIG['token_file']
MAX_UPLOAD_BYTES = CONFIG['max_upload_bytes']
MAX_TRANSCRIPT_CHARS = CONFIG['max_transcript_chars']
BODY_READ_IDLE_TIMEOUT_S = 30  # per-chunk gap allowed while receiving the upload
STT_MAX_BYTES = CONFIG['stt_max_bytes']
STT_CHUNK_SECONDS = max(1, CONFIG['stt_chunk_seconds'])
STT_CHUNK_OVERLAP_SECONDS = max(0, CONFIG['stt_chunk_overlap_seconds'])
STT_MAX_CHUNKS = max(1, CONFIG['stt_max_chunks'])
STT_SILENCE_DBFS = CONFIG['stt_silence_dbfs']
# Audio at/under this duration uses the original single-shot webhook path,
# byte-identical to pre-chunking behavior. Longer audio is chunked instead
# of being skipped -- see _resolve_stt/_webhook_stt_chunked.
_STT_CHUNK_TRIGGER_S = 30.0

# Semaphore(2): UP TO TWO analyses may run concurrently, not one-at-a-time --
# a second request acquires the free slot immediately. The locked() pre-check
# below only 429s a request when BOTH slots are already taken, i.e. on the
# THIRD concurrent request. Each slot's work is bounded, not unbounded, so
# worst case with both slots busy is roughly bounded rather than open-ended:
# the seven-ears subprocess.run has a hard 300s timeout, and singing.py's own
# ffmpeg decode is separately capped (MAX_ANALYSIS_S=480s via `-t`, plus an
# array-slice backstop) so a multi-hour upload can't pin a slot indefinitely.
# Subprocesses run in worker threads so the event loop (health checks, auth)
# never blocks on ffmpeg/analyzer work.
_ANALYSIS_SLOTS = asyncio.Semaphore(2)

# STT is pluggable (see CONFIGURATION above): 'local' asks the seven-ears CLI
# to run faster-whisper itself (--stt whisper); 'webhook' POSTs the audio to
# your own STT service at ATTUNE_STT_URL; 'none' skips transcription
# entirely (acoustics-only). If a caller already supplies a transcript (e.g.
# a frontend that runs its own STT before uploading), that transcript is
# always used as-is and STT_MODE is never consulted. Note for Windows users:
# Smart App Control can block faster-whisper's unsigned DLLs -- see README's
# Platform notes for the 'webhook'/'none' workaround.


def _write_token_file(token: str) -> None:
    """Create/overwrite TOKEN_FILE with restrictive permissions where the
    platform allows (0o600 -- owner read/write only). os.open with an
    explicit mode applies on POSIX; on Windows the mode bits are mostly
    ignored by the OS but harmless to pass."""
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(TOKEN_FILE), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(token)
    # O_CREAT's mode only applies to NEW files; tighten pre-existing ones too.
    with contextlib.suppress(OSError):
        os.chmod(str(TOKEN_FILE), 0o600)


def _load_token() -> str:
    # An empty/whitespace-only token file is treated as missing, never as a
    # valid (empty) API_TOKEN -- an empty API_TOKEN would make `Bearer `
    # match every blank Authorization header via compare_digest.
    existing = ''
    if TOKEN_FILE.exists():
        existing = TOKEN_FILE.read_text(encoding='utf-8').strip()
    if existing:
        return existing
    if TOKEN_FILE.exists():
        print(f'[attune] WARNING: token file {TOKEN_FILE} is empty/whitespace -- '
              'regenerating', file=sys.stderr, flush=True)
    token = secrets.token_urlsafe(32)
    _write_token_file(token)
    print(f'[attune] generated new token at {TOKEN_FILE}', file=sys.stderr, flush=True)
    return token


API_TOKEN = _load_token()

# Voice drop storage. Relative ATTUNE_DROPS_DIR (the default, './drops') is
# resolved against BASE_DIR, not the process's cwd, so it lands next to
# server.py regardless of how/where the process was launched. Created on
# boot -- a missing drops dir must never surface as a 500 on first upload.
DROPS_DIR = Path(CONFIG['drops_dir']).expanduser()
if not DROPS_DIR.is_absolute():
    DROPS_DIR = BASE_DIR / DROPS_DIR
DROPS_DIR = DROPS_DIR.resolve()
DROPS_DIR.mkdir(parents=True, exist_ok=True)
DROPS_KEEP = max(0, CONFIG['drops_keep'])

# Read once at startup -- no template rendering, the page pulls its own `k`
# from location.search client-side (see drop.html).
DROP_PAGE_HTML = (BASE_DIR / 'drop.html').read_text(encoding='utf-8')

_DROP_UNAUTHORIZED_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Attune - Voice Drop</title>
<style>
  body { background:#0f1115; color:#e8e8ec; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         display:flex; align-items:center; justify-content:center; min-height:100vh; margin:0; padding:24px; text-align:center; }
  div { max-width: 420px; }
  h1 { font-size:1.2rem; color:#d64550; }
</style></head>
<body><div><h1>Unauthorized</h1><p>This link is missing or has an invalid access token.</p></div></body></html>"""


def _subprocess_env() -> dict:
    env = os.environ.copy()
    if CONFIG['ffmpeg_dir']:
        env['PATH'] = CONFIG['ffmpeg_dir'] + os.pathsep + env.get('PATH', '')
    # Windows child pythons default to cp1252; the card opens with an emoji.
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


def _check_auth(authorization: Optional[str]) -> None:
    expected = f'Bearer {API_TOKEN}'
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(401, 'Unauthorized')


app = FastAPI(title='Attune')


def _local_whisper_available() -> bool:
    """Whether faster-whisper is importable in this environment (the analyzer
    runs on sys.executable, i.e. this same env). find_spec is cheap — no
    heavy import happens here."""
    import importlib.util
    try:
        return importlib.util.find_spec('faster_whisper') is not None
    except Exception:
        return False


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _music_models_present() -> dict:
    import basic_pitch_onnx
    md = basic_pitch_onnx.models_dir()
    return {
        'basic_pitch': basic_pitch_onnx.model_path().exists(),
        'mdx_vocals': (md / 'UVR-MDX-NET-Voc_FT.onnx').exists(),
    }


@app.get('/health')
async def health():
    body = {
        'status': 'ok',
        'ffmpeg': shutil.which('ffmpeg', path=_subprocess_env()['PATH']) is not None,
        'engine': f"acoustics-local/stt-{CONFIG['stt_mode']}",
        # Honest presence check for the vendored engine itself (separate
        # from ffmpeg discoverability above) -- False before
        # scripts/get-seven-ears.py has been run.
        'engine_present': SEVEN_EARS_SCRIPT.exists(),
        # Music perception surface: pure-DSP analysis always works; the
        # melody/stems models need onnxruntime + scripts/get-models.py.
        'onnxruntime_available': _onnxruntime_available(),
        'music_models_present': _music_models_present(),
    }
    if CONFIG['stt_mode'] == 'local':
        # The #1 "why no transcripts?" cause: Install step 3 (the vendor
        # requirements that pull in faster-whisper) was skipped. Surface it
        # where a single GET can find it (first community deployment hit
        # exactly this, 2026-07-18).
        body['local_whisper_available'] = _local_whisper_available()
    return body


_SUFFIX_BY_CONTENT_TYPE = {
    'audio/webm': '.webm',
    'audio/mp4': '.m4a',
    'audio/x-m4a': '.m4a',
    'audio/ogg': '.ogg',
    'audio/wav': '.wav',
    'audio/x-wav': '.wav',
}

FORMAT_SNIPPET = (
    "import sys, json, importlib.util; "
    "spec = importlib.util.spec_from_file_location('seven_ears_card', sys.argv[1]); "
    "mod = importlib.util.module_from_spec(spec); "
    "spec.loader.exec_module(mod); "
    "print(mod.format_card(json.load(sys.stdin)))"
)


async def _run_singing_analysis(tmp_path: str) -> tuple:
    """Best-effort singing analysis (YIN pitch/notes/vibrato/dynamics/key/
    tempo) AND its card-text formatting. Never fails the request — any
    exception, including one raised by format_singing_section, is logged and
    swallowed, same non-fatal posture as _render_card. A formatting bug must
    never 500 an otherwise-good analysis. Returns (result_or_None, section)
    where section is '' unless a melodic result formatted cleanly."""
    try:
        result = await asyncio.to_thread(analyze_singing, tmp_path)
        section = ''
        if result and result.get('is_melodic'):
            section = format_singing_section(result) or ''
        return result, section
    except Exception as e:
        print(f'[attune] singing analysis failed: {e}', file=sys.stderr, flush=True)
        return None, ''


async def _run_voice_delivery(tmp_path: str, data: dict) -> tuple:
    """Best-effort spoken delivery/texture description for voice notes."""
    try:
        result = await asyncio.to_thread(analyze_voice_delivery, tmp_path, data)
        return result, format_voice_delivery_section(result)
    except Exception as e:
        print(f'[attune] voice delivery analysis failed: {e}', file=sys.stderr, flush=True)
        return None, ''


def _melody_analysis(tmp_path: str) -> Optional[dict]:
    """basic-pitch melody summary, or None when the model/runtime is absent.
    Lazy import — server boot must never depend on onnxruntime loading."""
    import basic_pitch_onnx
    ok, reason = basic_pitch_onnx.is_available()
    if not ok:
        print(f'[attune] melody skipped: {reason}', file=sys.stderr, flush=True)
        return None
    x, sr = load_wav(tmp_path)
    tr = basic_pitch_onnx.transcribe(x, sr)
    summary = basic_pitch_onnx.summarize_melody(tr['notes'], len(x) / sr)
    # measurements carry the compact summary, never the raw polyphonic dump.
    return summary


async def _run_music_analysis(tmp_path: str) -> tuple:
    """Best-effort music analysis (key/chords/tempo/energy/sections + melody
    when the basic-pitch model is present) AND its card-section formatting.
    Same non-fatal posture as _run_singing_analysis: any exception is logged
    and swallowed — a music-analysis bug must never fail an otherwise-good
    request. Returns (result_or_None, section_str)."""
    try:
        result = await asyncio.to_thread(music.analyze_music, tmp_path)
    except Exception as e:
        print(f'[attune] music analysis failed: {e}', file=sys.stderr, flush=True)
        return None, ''
    melody = None
    try:
        melody = await asyncio.to_thread(_melody_analysis, tmp_path)
    except Exception as e:
        print(f'[attune] melody analysis failed: {e}', file=sys.stderr, flush=True)
    try:
        if result and melody and melody.get('summary'):
            result['melody'] = melody
        section = music.format_music_section(result, melody=melody) if result else ''
        return result, section
    except Exception as e:
        print(f'[attune] music card formatting failed: {e}', file=sys.stderr, flush=True)
        return result, ''


def _lyrics_section(track: str, artist: str, music_result: Optional[dict]) -> str:
    """Best-effort LRCLIB lyrics section, annotated on the analysis timeline.
    Blocking (network) — run via asyncio.to_thread. '' on any failure."""
    try:
        duration = (music_result or {}).get('duration_s')
        lyr = lyrics_mod.fetch_lyrics(track, artist, duration)
        if not lyr:
            return ''
        energy = (music_result or {}).get('energy') or {}
        return lyrics_mod.format_lyrics_section(
            lyr,
            peak_t=energy.get('loudest_t'),
            section_changes=(music_result or {}).get('section_changes'),
        )
    except Exception as e:
        print(f'[attune] lyrics lookup failed: {e}', file=sys.stderr, flush=True)
        return ''


def _render_card(data: dict, script: Optional[Path] = None, timeout: int = 60) -> str:
    """Render the human-readable acoustic card in a second subprocess that
    imports only format_card from seven_ears_card. Non-fatal: any failure
    (nonzero return or timeout) yields '' rather than failing the request."""
    args = [PYTHON, '-c', FORMAT_SNIPPET, str(script or SEVEN_EARS_SCRIPT)]
    try:
        proc = subprocess.run(
            args,
            input=json.dumps(data),
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=_subprocess_env(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print('[attune] _render_card timed out', file=sys.stderr, flush=True)
        return ''
    if proc.returncode != 0:
        print(f'[attune] _render_card failed: {proc.stderr[-800:]}', file=sys.stderr, flush=True)
        return ''
    # Our wrapper, our masthead. The engine keeps full credit in README.md,
    # vendor/, and the measurements' engine field.
    return proc.stdout.replace('SEVEN EARS CARD', 'ATTUNE')


def _webhook_token() -> str:
    token_file = CONFIG['stt_token_file']
    if not token_file:
        return ''
    try:
        return Path(token_file).expanduser().read_text(encoding='utf-8').strip()
    except OSError:
        return ''


def _gate_check(path: str) -> tuple:
    """Decode `path` (already a small re-encoded audio file, same as
    singing.py's own decode path) and check the RMS energy gate. Returns
    (skip: bool, reason: str) -- see energy_gate.should_skip_stt.

    Fails OPEN: a decode failure (corrupt chunk, ffmpeg unavailable, etc.)
    returns (False, '') -- an undeterminable energy level must never
    silently suppress a legitimate transcription attempt. Runs synchronously
    (ffmpeg subprocess + numpy) -- callers already run this from a worker
    thread (the chunked path) or wrap it in asyncio.to_thread themselves
    (the single-shot path in _resolve_stt)."""
    try:
        x, sr = load_wav(path)
    except Exception as e:
        print(f'[attune] energy gate: decode failed, not skipping: {e}', file=sys.stderr, flush=True)
        return False, ''
    return should_skip_stt(x, sr, STT_SILENCE_DBFS)


def _webhook_stt(path: str, language: str = '') -> str:
    """Transcribe via the configured ATTUNE_STT_URL webhook. Best-effort.
    Reads the audio file from disk here, inside the worker thread, rather
    than the caller holding a large blob on the event loop. Never loads a
    file bigger than STT_MAX_BYTES into memory -- this is the last line of
    defense even though callers are expected to skip calling this at all
    once they know the file exceeds that size (or, for chunked audio, to
    have already split it into chunks that individually fit).

    `language` is an optional caller-supplied hint (e.g. 'en', 'ja'),
    forwarded as a `?language=` query param on the webhook URL when set --
    omitted entirely when empty, so a caller that never passes it gets the
    exact same request as before this parameter existed."""
    import urllib.parse
    import urllib.request
    token = _webhook_token()
    url = CONFIG['stt_url']
    if not token or not url:
        return ''
    try:
        if os.path.getsize(path) > STT_MAX_BYTES:
            return ''
    except OSError:
        return ''
    if language:
        sep = '&' if '?' in url else '?'
        url = f'{url}{sep}language={urllib.parse.quote(language)}'
    with open(path, 'rb') as f:
        blob = f.read()
    req = urllib.request.Request(
        url, data=blob, method='POST',
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/octet-stream',
            # Some edges bot-fight the default Python-urllib agent.
            'User-Agent': 'attune-mcp',
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            out = json.loads(resp.read().decode('utf-8'))
            return str(out.get('transcript', '') or '')
    except Exception as e:
        print(f'[attune] webhook stt failed: {e}', file=sys.stderr, flush=True)
        return ''


def _probe_duration_s(path: str) -> Optional[float]:
    """Best-effort audio duration in seconds via ffprobe, falling back to
    parsing `ffmpeg -i`'s stderr banner if ffprobe isn't discoverable.
    Returns None on any failure -- callers treat that the same as "audio is
    short enough for single-shot" (see _resolve_stt), never blocking
    transcription just because duration couldn't be determined."""
    env = _subprocess_env()
    ffprobe = shutil.which('ffprobe', path=env['PATH'])
    if ffprobe:
        try:
            proc = subprocess.run(
                [ffprobe, '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', path],
                capture_output=True, text=True, env=env, timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return float(proc.stdout.strip())
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
    ffmpeg = shutil.which('ffmpeg', path=env['PATH']) or 'ffmpeg'
    try:
        proc = subprocess.run(
            [ffmpeg, '-i', path], capture_output=True, text=True, env=env, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', proc.stderr or '')
    if not m:
        return None
    h, mi, s = m.groups()
    try:
        return int(h) * 3600 + int(mi) * 60 + float(s)
    except ValueError:
        return None


def _split_chunks(path: str, duration_s: float) -> list:
    """Split `path` into overlapping segments (ATTUNE_STT_CHUNK_SECONDS wide,
    ATTUNE_STT_CHUNK_OVERLAP_SECONDS overlap between consecutive segments),
    capped at ATTUNE_STT_MAX_CHUNKS. Every segment is RE-ENCODED -- there is
    deliberately no `-c copy` stream-copy fast path. A stream copy with `-ss`
    snaps to the source container's nearest keyframe/cluster boundary rather
    than the exact requested timestamp (documented ffmpeg behavior), which
    silently shifts where a chunk actually starts and makes the overlap-word
    math in stt_stitch.py lie about how much audio two chunks actually share.
    `-ss` placed BEFORE `-i` (input seeking) combined with re-encoding is
    sample-accurate in modern ffmpeg, so that's what every chunk gets.

    Encoded to opus/webm, mono, ~32kbps -- more than adequate for
    whisper-class STT and comfortably small, but still checked against
    STT_MAX_BYTES same as before. Returns a list of (start_seconds, path)
    tuples in chronological order -- the caller owns deleting the paths.
    Runs entirely in the calling worker thread (subprocess.run, not
    asyncio.to_thread here -- callers wrap the whole chunking+upload
    sequence in one to_thread call)."""
    starts = []
    step = max(1, STT_CHUNK_SECONDS - STT_CHUNK_OVERLAP_SECONDS)
    t = 0.0
    while t < duration_s and len(starts) < STT_MAX_CHUNKS:
        starts.append(t)
        t += step

    env = _subprocess_env()
    ffmpeg = shutil.which('ffmpeg', path=env['PATH']) or 'ffmpeg'
    chunks = []
    for start in starts:
        enc_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
        enc_ok = False
        try:
            proc = subprocess.run(
                [ffmpeg, '-y', '-v', 'error', '-ss', str(start), '-i', path, '-t', str(STT_CHUNK_SECONDS),
                 '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'libopus', '-b:a', '32k', enc_path],
                capture_output=True, text=True, env=env, timeout=60,
            )
            enc_ok = (proc.returncode == 0 and os.path.exists(enc_path)
                      and 0 < os.path.getsize(enc_path) <= STT_MAX_BYTES)
        except (OSError, subprocess.TimeoutExpired):
            enc_ok = False
        if enc_ok:
            chunks.append((start, enc_path))
        else:
            with contextlib.suppress(OSError):
                os.unlink(enc_path)
            # A failed SPLIT must surface exactly like a failed POST: a gap
            # marker in the assembled transcript, never a silent omission the
            # stitcher would then dedup across.
            chunks.append((start, None))
            print(f'[attune] chunk at {start:.0f}s failed to split -- gap marker inserted', file=sys.stderr, flush=True)
    return chunks


def _webhook_stt_chunked(path: str, language: str, duration_s: float) -> tuple:
    """Webhook STT for audio longer than _STT_CHUNK_TRIGGER_S: split into
    overlapping chunks, POST them to the webhook SEQUENTIALLY (never
    concurrently -- this keeps the whole operation, and its ffmpeg/network
    work, inside one worker thread with a bounded, easy-to-reason-about
    request pattern), then stitch the resulting per-chunk transcripts back
    together with the deterministic overlap-dedup heuristic in stt_stitch.py.

    Per chunk: one retry (a fresh POST, not a resend of the same request
    object) on failure. If both attempts fail, that chunk's position gets an
    explicit stt_stitch.GAP_MARKER placeholder instead of being silently
    dropped -- the stitcher never dedups across a gap boundary -- and the
    missing time range is logged. Returns (stitched_transcript, partial)
    where `partial` is True iff at least one chunk was persistently lost;
    same '' contract as _webhook_stt when every chunk is lost or none split."""
    chunk_pairs = _split_chunks(path, duration_s)
    if not chunk_pairs:
        print('[attune] chunking produced no usable segments', file=sys.stderr, flush=True)
        return '', False
    step = max(1, STT_CHUNK_SECONDS - STT_CHUNK_OVERLAP_SECONDS)
    if len(chunk_pairs) >= STT_MAX_CHUNKS and duration_s > STT_MAX_CHUNKS * step + STT_CHUNK_OVERLAP_SECONDS:
        print(f'[attune] audio ({duration_s:.0f}s) exceeds ATTUNE_STT_MAX_CHUNKS={STT_MAX_CHUNKS} -- '
              'transcribing only the first portion', file=sys.stderr, flush=True)

    transcripts = []
    partial = False
    try:
        for i, (start, chunk_path) in enumerate(chunk_pairs):
            if chunk_path is None:
                # Split itself failed for this range (see _split_chunks) —
                # same honest treatment as a failed POST.
                end = start + STT_CHUNK_SECONDS
                print(f'[attune] chunk {i + 1}/{len(chunk_pairs)} ({start:.0f}s-{end:.0f}s) '
                      'was never split -- inserting gap marker', file=sys.stderr, flush=True)
                transcripts.append(GAP_MARKER)
                partial = True
                continue
            gate_skip, gate_reason = _gate_check(chunk_path)
            if gate_skip:
                # Silence is not missing data -- it's the absence of speech.
                # Contribute NOTHING to the transcript: no text, no
                # GAP_MARKER (a gap marker means real content is unknown;
                # here the content IS known -- there wasn't any).
                end = start + STT_CHUNK_SECONDS
                print(f'[attune] chunk {i + 1}/{len(chunk_pairs)} ({start:.0f}s-{end:.0f}s) '
                      f'skipped: below energy gate ({gate_reason})', file=sys.stderr, flush=True)
                continue
            t = _webhook_stt(chunk_path, language) or _webhook_stt(chunk_path, language)
            if t:
                transcripts.append(t)
            else:
                end = start + STT_CHUNK_SECONDS
                print(f'[attune] chunk {i + 1}/{len(chunk_pairs)} ({start:.0f}s-{end:.0f}s) '
                      'failed to transcribe after 1 retry -- inserting gap marker',
                      file=sys.stderr, flush=True)
                transcripts.append(GAP_MARKER)
                partial = True
    finally:
        for _, chunk_path in chunk_pairs:
            if chunk_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(chunk_path)

    return stitch_transcripts(transcripts, overlap_seconds=STT_CHUNK_OVERLAP_SECONDS), partial


async def _resolve_stt(tmp_path: str, transcript: str, language: str = '') -> tuple:
    """Given a temp audio file and any already-known transcript, return
    (transcript, extra_cli_args, stt_skipped, stt_partial) honoring
    ATTUNE_STT_MODE. A caller-supplied transcript always wins and
    short-circuits STT_MODE entirely -- this is what keeps a frontend that
    already runs its own STT (like this deployment's) behaving identically
    regardless of STT_MODE.

    `stt_partial` is True only when the chunked webhook path lost at least
    one chunk after its retry (see _webhook_stt_chunked) -- callers surface
    this as measurements['stt_partial']=True (REST) or an explicit "partial"
    note (MCP) rather than silently returning a transcript with an
    undisclosed gap in it.

    `language` is an optional hint forwarded to the webhook (see
    _webhook_stt); per-chunk auto language detection is still the default
    when it's empty, deliberately -- a multilingual singer isn't forced into
    one language just because chunking kicked in."""
    if transcript:
        return transcript, ['--transcript', transcript, '--transcript-source', 'stt_api'], False, False

    mode = CONFIG['stt_mode']
    if mode == 'local':
        return '', ['--stt', 'whisper', '--whisper-model', CONFIG['whisper_model']], False, False
    if mode == 'webhook':
        stt_skipped = False
        stt_partial = False
        # Duration decides single-shot vs. chunked -- NOT the byte-size cap
        # anymore, which used to skip transcription outright for anything
        # over STT_MAX_BYTES. Oversized-but-short-enough audio still takes
        # the byte-cap-gated single-shot path below (unchanged); anything
        # over _STT_CHUNK_TRIGGER_S seconds gets chunked instead of skipped.
        duration_s = await asyncio.to_thread(_probe_duration_s, tmp_path)
        if duration_s is not None and duration_s > _STT_CHUNK_TRIGGER_S:
            transcript, stt_partial = await asyncio.to_thread(_webhook_stt_chunked, tmp_path, language, duration_s)
            transcript = transcript.strip()
        else:
            # Short audio, or duration unknown (ffprobe/ffmpeg unavailable or
            # failed to parse) -- single-shot path, byte-identical to the
            # pre-chunking behavior.
            try:
                size = os.path.getsize(tmp_path)
            except OSError:
                size = 0
            if size > STT_MAX_BYTES:
                stt_skipped = True
            else:
                gate_skip, gate_reason = await asyncio.to_thread(_gate_check, tmp_path)
                if gate_skip:
                    # Same "silence contributes nothing" posture as the
                    # chunked path above -- empty transcript, no webhook
                    # call, not treated as stt_skipped (that flag means
                    # "STT was owed but couldn't be attempted", not "there
                    # was nothing to transcribe").
                    print(f'[attune] audio skipped: below energy gate ({gate_reason})',
                          file=sys.stderr, flush=True)
                else:
                    transcript = (await asyncio.to_thread(_webhook_stt, tmp_path, language)).strip()
        args = ['--transcript', transcript, '--transcript-source', 'stt_api'] if transcript else []
        return transcript, args, stt_skipped, stt_partial
    # mode == 'none': acoustics only
    return '', [], False, False


# Client-visible note when local STT was requested but faster-whisper isn't
# installed. A silent empty transcript looks like a broken deploy (the first
# community deployment reported exactly this); the cause and the fix must be
# discoverable from the RESPONSE, not just server stderr.
STT_UNAVAILABLE_NOTE = (
    'transcript unavailable: ATTUNE_STT_MODE=local but faster-whisper is not '
    'installed — run Install step 3 (pip install -r vendor/seven-ears/requirements.txt) '
    'or set ATTUNE_STT_MODE to webhook/none; see /health local_whisper_available'
)


async def _spawn_analyzer(args: list, filename: str, log_tag: str):
    """Run the seven-ears analyzer subprocess. Returns (proc, stt_fell_back):
    the completed process on success or None after logging on any failure,
    plus whether the local-STT fallback below fired (so callers can surface
    STT_UNAVAILABLE_NOTE instead of a silent empty transcript).

    Local-STT resilience: if ATTUNE_STT_MODE=local but faster-whisper isn't
    installed, seven-ears exits nonzero when whisper is explicitly requested.
    Rather than failing the whole analysis over a missing OPTIONAL dependency,
    retry once without the --stt args — the caller gets acoustics (and any
    singing analysis) with an empty transcript, as documented in the README.
    """
    attempt_args = args
    stt_fell_back = False
    for attempt in (1, 2):
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                attempt_args,
                capture_output=True,
                text=True,
                encoding='utf-8',
                env=_subprocess_env(),
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            print(f'{log_tag} analysis timed out for {filename}', file=sys.stderr, flush=True)
            return None, stt_fell_back
        except OSError as e:
            print(f'{log_tag} analysis spawn failed: {e}', file=sys.stderr, flush=True)
            return None, stt_fell_back
        if proc.returncode == 0:
            return proc, stt_fell_back
        stderr_tail = proc.stderr[-800:] if proc.stderr else ''
        if attempt == 1 and '--stt' in attempt_args and 'faster_whisper' in stderr_tail:
            print(f'{log_tag} local STT unavailable (faster-whisper not importable); retrying acoustics-only', file=sys.stderr, flush=True)
            stt_fell_back = True
            i = attempt_args.index('--stt')
            attempt_args = attempt_args[:i] + attempt_args[i + 4:]  # drop --stt whisper --whisper-model <m>
            continue
        print(f'{log_tag} analysis failed: {stderr_tail}', file=sys.stderr, flush=True)
        return None, stt_fell_back
    return None, stt_fell_back


@app.post('/api/analyze')
async def analyze(
    request: Request,
    audio: UploadFile = File(...),
    transcript: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

    # Defense-in-depth only: FastAPI parses multipart before the handler
    # runs, so the real memory bound is the AnalyzeGuardMiddleware ASGI
    # guard (registered below) plus the streamed read-loop cap just after
    # this check.
    cl = request.headers.get('content-length')
    if cl is not None and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'audio exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MiB limit')

    filename = audio.filename or 'voice-note'
    suffix = Path(filename).suffix
    if not suffix:
        suffix = _SUFFIX_BY_CONTENT_TYPE.get(audio.content_type or '', '.webm')

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        total = 0
        while chunk := await audio.read(1 << 20):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f'audio exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MiB limit')
            # Chunks are capped at 1 MiB and a 50 MiB upload is at most ~50
            # thread hops -- worth it to keep the write off the event loop.
            await asyncio.to_thread(tmp.write, chunk)
        tmp.close()

        t = (transcript or '').strip()
        if len(t.encode('utf-8')) > MAX_TRANSCRIPT_CHARS:
            raise HTTPException(413, 'transcript too long')
        lang = (language or '').strip()

        # Semaphore acquired BEFORE the STT/probe/chunk pipeline (not just
        # around the analyzer subprocess) -- _resolve_stt can itself spawn
        # ffprobe/ffmpeg (chunking) and a sequence of webhook POSTs, and that
        # work needs to be bounded by the same two concurrent-analysis slots
        # as everything else, or a third+ caller could pile up unbounded
        # ffmpeg/webhook work while only two analyzer subprocesses run.
        if _ANALYSIS_SLOTS.locked():
            raise HTTPException(429, 'analysis busy, retry shortly')
        async with _ANALYSIS_SLOTS:
            t, extra_args, _stt_skipped, stt_partial = await _resolve_stt(tmp_path, t, lang)

            args = [PYTHON, str(SEVEN_EARS_SCRIPT), tmp_path, '--json']
            args[3:3] = extra_args

            print(f'[attune] spawning analysis for {filename}', file=sys.stderr, flush=True)
            proc, stt_fell_back = await _spawn_analyzer(args, filename, '[attune]')
            if proc is None:
                # Details stay in the server log; clients get a fixed message.
                raise HTTPException(500, 'analysis failed')

            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                raise HTTPException(500, 'analysis failed')
            if stt_fell_back:
                data['stt_note'] = STT_UNAVAILABLE_NOTE

            # Overwrite with the friendly filename before rendering the card —
            # format_card requires data['file'] and this prevents the OS temp
            # path from leaking into the rendered card text.
            data['file'] = filename
            if stt_partial:
                # At least one chunk was lost after its retry -- the
                # transcript has an explicit GAP_MARKER gap in it. Surface
                # this so callers don't silently trust a partial transcript.
                data['stt_partial'] = True
            card_text = await asyncio.to_thread(_render_card, data)

            voice_delivery, voice_section = await _run_voice_delivery(tmp_path, data)
            if voice_delivery:
                data['voice_delivery'] = voice_delivery
            if voice_section:
                card_text = f'{card_text}\n{voice_section}' if card_text else voice_section

            singing, section = await _run_singing_analysis(tmp_path)
            if singing and singing.get('is_melodic'):
                data['singing'] = singing
            if section:
                card_text = f'{card_text}\n{section}' if card_text else section

            # Backing music detected behind the voice? Enrich with the music
            # card (key/chords/tempo/energy) — best-effort, same posture.
            if singing and singing.get('analysis_mode') == 'singing-with-music':
                music_result, music_section = await _run_music_analysis(tmp_path)
                if music_result:
                    data['music'] = music_result
                if music_section:
                    card_text = f'{card_text}\n{music_section}' if card_text else music_section

        return {
            'transcript': data.get('transcript', ''),
            'card_text': card_text,
            'measurements': data,
        }
    finally:
        with contextlib.suppress(OSError):
            tmp.close()
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


@app.post('/api/transcribe')
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
):
    """Transcribe a voice note without running the full acoustic analyzer."""
    _check_auth(authorization)

    cl = request.headers.get('content-length')
    if cl is not None and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'audio exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MiB limit')

    filename = audio.filename or 'voice-note'
    suffix = Path(filename).suffix
    if not suffix:
        suffix = _SUFFIX_BY_CONTENT_TYPE.get(audio.content_type or '', '.webm')

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        total = 0
        while chunk := await audio.read(1 << 20):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f'audio exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MiB limit')
            await asyncio.to_thread(tmp.write, chunk)
        tmp.close()

        if _ANALYSIS_SLOTS.locked():
            raise HTTPException(429, 'transcription busy, retry shortly')
        async with _ANALYSIS_SLOTS:
            text, _extra_args, skipped, partial = await _resolve_stt(
                tmp_path, '', (language or '').strip(),
            )
        return {
            'transcript': text,
            'stt_skipped': skipped,
            'stt_partial': partial,
        }
    finally:
        with contextlib.suppress(OSError):
            tmp.close()
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Voice drop page — GET /drop, POST /api/drop, GET /drops/<name>
# ---------------------------------------------------------------------------
# A tiny self-contained browser page for recording a voice note and getting
# back a shareable link (e.g. to paste into a claude.ai chat that has the
# Attune MCP connector as a custom connector). GET /drop and POST /api/drop
# both require the same ?k=<token> the MCP endpoint uses — claude.ai's
# custom connectors already require this pattern, so the drop page rides
# the same auth model rather than inventing a second one. GET /drops/<name>
# itself is deliberately NOT gated: the link IS the capability, exactly
# like any public storage bucket link (see README's Security notes). See
# ATTUNE_DROPS_DIR / ATTUNE_DROPS_KEEP / ATTUNE_PUBLIC_BASE_URL above.
#
# Design note: GET /drop authenticates IN-ROUTE (query k before any content
# is served) rather than in AnalyzeGuardMiddleware. The middleware's job is
# auth-BEFORE-BODY-PARSING for endpoints that receive large bodies; a GET
# has no request body to protect, so in-route auth is security-equivalent
# and keeps the middleware's scope honest.

DROPS_NAME_RE = re.compile(r'^[A-Za-z0-9_-]+\.(webm|ogg|m4a|mp3|wav)$')

_CONTENT_TYPE_BY_DROP_SUFFIX = {
    '.webm': 'audio/webm',
    '.ogg': 'audio/ogg',
    '.m4a': 'audio/mp4',
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
}


@app.get('/drop')
async def drop_page(k: str = ''):
    if not secrets.compare_digest(k, API_TOKEN):
        return Response(_DROP_UNAUTHORIZED_HTML, status_code=401, media_type='text/html')
    return Response(DROP_PAGE_HTML, media_type='text/html')


def _drop_public_url(request: Request, name: str) -> str:
    base = CONFIG['public_base_url']
    if not base:
        # Derived from the request's Host header -- fine for direct access.
        # If Attune sits behind a reverse proxy / tunnel that terminates
        # TLS and/or rewrites Host, set ATTUNE_PUBLIC_BASE_URL explicitly
        # so the returned link matches the externally reachable origin
        # rather than whatever internal host/port the proxy forwarded to.
        host = request.headers.get('host') or request.url.netloc
        base = f'{request.url.scheme}://{host}'
    return f"{base.rstrip('/')}/drops/{name}"


def _prune_drops() -> None:
    """After a new drop, delete the oldest files beyond ATTUNE_DROPS_KEEP
    (simple mtime sort). Best-effort: a stat/unlink failure on one file is
    logged and skipped rather than aborting the whole prune. Runs in a
    worker thread (see caller) since iterdir/stat/unlink are blocking."""
    try:
        # Only prune files WE created (regex-valid drop names). A user who
        # points ATTUNE_DROPS_DIR at a shared folder must never have
        # unrelated files deleted by our retention policy.
        entries = [p for p in DROPS_DIR.iterdir() if p.is_file() and DROPS_NAME_RE.match(p.name)]
    except OSError as e:
        print(f'[attune] drops prune: could not list {DROPS_DIR}: {e}', file=sys.stderr, flush=True)
        return
    if len(entries) <= DROPS_KEEP:
        return

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    entries.sort(key=_mtime, reverse=True)
    for stale in entries[DROPS_KEEP:]:
        try:
            stale.unlink()
            print(f'[attune] drops prune: deleted {stale.name} (over ATTUNE_DROPS_KEEP={DROPS_KEEP})',
                  file=sys.stderr, flush=True)
        except OSError as e:
            print(f'[attune] drops prune: failed to delete {stale.name}: {e}', file=sys.stderr, flush=True)


@app.post('/api/drop')
async def api_drop(request: Request):
    # Defense-in-depth only, same posture as /api/analyze's own re-check:
    # the real auth-before-body and size-cap guarantees live in
    # AnalyzeGuardMiddleware (registered below, now covering this path too).
    k = request.query_params.get('k', '')
    if not secrets.compare_digest(k, API_TOKEN):
        raise HTTPException(401, 'Unauthorized')

    cl = request.headers.get('content-length')
    if cl is not None and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'audio exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MiB limit')

    # Raw-body upload (documented in README): the request body IS the audio
    # bytes, not multipart -- simpler for a small self-contained page's
    # fetch(..., {body: blob}) than constructing multipart/form-data.
    content_type = (request.headers.get('content-type') or '').split(';', 1)[0].strip().lower()
    suffix = _SUFFIX_BY_CONTENT_TYPE.get(content_type, '.webm')
    name = secrets.token_urlsafe(16) + suffix
    dest_path = DROPS_DIR / name

    total = 0
    f = open(dest_path, 'wb')
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f'audio exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MiB limit')
            await asyncio.to_thread(f.write, chunk)
    except BaseException:
        f.close()
        with contextlib.suppress(OSError):
            os.unlink(dest_path)
        raise
    f.close()

    if total == 0:
        with contextlib.suppress(OSError):
            os.unlink(dest_path)
        raise HTTPException(400, 'empty upload')

    await asyncio.to_thread(_prune_drops)

    return {'url': _drop_public_url(request, name)}


@app.get('/drops/{name}')
async def get_drop(name: str):
    # Strict allowlist-by-regex: no traversal segment, no path separator, no
    # percent-encoding, no extension outside the fixed set can ever match --
    # `name` here is already the decoded path segment Starlette routed on,
    # and the {name} converter itself cannot contain '/'. The containment
    # check below is belt-and-suspenders on top of that, not load-bearing.
    if not DROPS_NAME_RE.fullmatch(name):
        raise HTTPException(404, 'Not found')
    path = (DROPS_DIR / name).resolve()
    if path.parent != DROPS_DIR or not path.is_file():
        raise HTTPException(404, 'Not found')
    media_type = _CONTENT_TYPE_BY_DROP_SUFFIX.get(path.suffix.lower(), 'application/octet-stream')
    return FileResponse(
        path,
        media_type=media_type,
        headers={'Cache-Control': 'public, max-age=31536000, immutable'},
    )


# ---------------------------------------------------------------------------
# MCP endpoint — claude.ai custom connector
# ---------------------------------------------------------------------------
# Minimal stateless streamable-HTTP MCP: one tool, analyze_voice_note, taking
# the PUBLIC URL of a voice note already at an allowed storage origin. Auth
# rides the connector URL as ?k=<token> (claude.ai custom connectors can't
# send bearer headers without full OAuth). Fetches are pinned to
# ATTUNE_ALLOWED_AUDIO_PREFIXES: this is NEVER a general analyze-anything
# endpoint. An empty prefix list disables the tool outright rather than
# defaulting open.


import urllib.error
import urllib.parse
import urllib.request


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Disables redirect-following entirely. Returning None from
    redirect_request tells urllib this handler declines to handle the 3xx,
    so no other installed handler follows it either -- the request falls
    through to HTTPErrorProcessor, which raises urllib.error.HTTPError with
    the original 3xx status for us to catch and refuse explicitly. A
    same-origin, same-prefix redirect could otherwise be used to bounce a
    validated URL to somewhere the allowlist would never have approved."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler)


def _path_no_traversal(path: str) -> Optional[str]:
    """Percent-decode a URL path and reject any '..' segment, whether it
    arrived literally or as (possibly mixed-case) %2e%2e encoding. Returns
    the decoded path, or None if traversal was detected."""
    decoded = urllib.parse.unquote(path)
    normalized = decoded.replace('\\', '/')
    if any(seg == '..' for seg in normalized.split('/')):
        return None
    return normalized


def _validate_audio_url(url: str, prefixes: list) -> Optional[str]:
    """Hardened allowlist check for the MCP audio_url. Returns None if the
    URL is safe to fetch, otherwise a short refusal message. Requirements,
    all enforced against the PARSED url (never a raw string.startswith):
    https only; exact hostname match against a prefix's hostname; matching
    port (default 443 on both sides if unspecified); no userinfo/credentials
    unless the matching prefix itself specifies the identical ones; and a
    normalized, traversal-free path that is a genuine path-segment prefix of
    the configured prefix's path (not just a leading-substring — so
    '/bucket-evil/' can never match a configured '/bucket/')."""
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return 'Refused: could not parse audio_url.'
    if parsed.scheme != 'https':
        return 'Refused: audio_url must use https.'
    if not parsed.hostname:
        return 'Refused: audio_url has no host.'

    norm_path = _path_no_traversal(parsed.path)
    if norm_path is None:
        return 'Refused: audio_url path contains path traversal.'

    for prefix in prefixes:
        try:
            p = urllib.parse.urlsplit(prefix)
        except ValueError:
            continue
        if p.scheme != 'https' or not p.hostname:
            continue
        if parsed.hostname.lower() != p.hostname.lower():
            continue
        try:
            # .port raises ValueError lazily on malformed/out-of-range ports;
            # a weird URL must be a refusal, never a 500.
            if (parsed.port or 443) != (p.port or 443):
                continue
        except ValueError:
            return 'Refused: audio_url has an invalid port.'
        if parsed.username != p.username or parsed.password != p.password:
            continue
        prefix_path = _path_no_traversal(p.path)
        if prefix_path is None:
            continue
        if not norm_path.startswith(prefix_path):
            continue
        # Segment-boundary check: '/bucket-evil/x' must not match a
        # configured prefix path of '/bucket'. Safe when the prefix path
        # already ends in '/' (the common/documented case).
        if not prefix_path.endswith('/') and len(norm_path) > len(prefix_path) \
                and norm_path[len(prefix_path)] != '/':
            continue
        return None
    return 'Refused: audio_url must start with one of this server\'s allowed storage prefixes.'


def _fetch_allowed_audio(url: str, dest_path: str) -> None:
    """Download from an allowed prefix only, streaming chunked straight to
    dest_path with the size cap enforced as bytes arrive — never buffers the
    full download in memory. Runs in a thread.

    Redirects are refused by design (see _NoRedirectHandler): the caller's
    URL was validated against the allowlist, and a 3xx response could
    otherwise be used to reach a host/path the allowlist never approved.
    A 3xx response raises urllib.error.HTTPError, translated here into a
    ValueError with a clear message for the caller to surface."""
    req = urllib.request.Request(url, headers={'User-Agent': 'attune-mcp'})
    try:
        resp = _NO_REDIRECT_OPENER.open(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            raise ValueError(
                f'server returned a redirect ({e.code}) — Attune refuses to follow '
                'redirects on MCP audio fetches by design'
            ) from e
        raise
    with resp, open(dest_path, 'wb') as f:
        total = 0
        while chunk := resp.read(1 << 20):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise ValueError(f'audio exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MiB limit')
            f.write(chunk)


def _mcp_tool_def() -> dict:
    prefixes = CONFIG['allowed_audio_prefixes']
    if prefixes:
        description = (
            'Acoustic analysis of a voice note from an allowed storage origin. '
            'Give it the public audio URL (and optionally the transcript) and it returns '
            'the Attune acoustic card: timing, pauses, pace, pitch, dynamics, and an '
            'evidence-bound plain-language description of vocal delivery and texture. '
            'Numbers, not diagnoses — cues support interpretation in context.'
        )
        url_description = 'Public URL starting with one of: ' + ', '.join(prefixes)
    else:
        description = (
            'DISABLED — no ATTUNE_ALLOWED_AUDIO_PREFIXES configured on this server, so '
            'analyze_voice_note refuses every URL. This is intentional: Attune never ships '
            'an open fetch-anything endpoint. Set ATTUNE_ALLOWED_AUDIO_PREFIXES to a '
            'comma-separated list of trusted storage URL prefixes to enable it.'
        )
        url_description = 'Not usable until ATTUNE_ALLOWED_AUDIO_PREFIXES is configured on the server.'
    return {
        'name': 'analyze_voice_note',
        'description': description,
        'inputSchema': {
            'type': 'object',
            'properties': {
                'audio_url': {'type': 'string', 'description': url_description},
                'transcript': {'type': 'string', 'description': 'Optional transcript of the note, if known'},
                'language': {
                    'type': 'string',
                    'description': (
                        "Optional STT language hint (e.g. 'en', 'ja'), forwarded to the STT webhook "
                        "when ATTUNE_STT_MODE=webhook. Leave unset for per-clip auto-detection."
                    ),
                },
            },
            'required': ['audio_url'],
        },
    }


MUSIC_MODES = ('auto', 'song', 'instrumental', 'sing_vs_track', 'stems')
# sing_vs_track and stems run source separation (~minutes on this CPU) —
# they go through the job store below, never a synchronous MCP response.
_MUSIC_HEAVY_MODES = ('sing_vs_track', 'stems')

# ── Music job store ─────────────────────────────────────────────────────────
# Minimal and in-process by design: jobs do NOT survive a server restart
# (documented in the tool description). One heavy job runs at a time
# (_HEAVY_SLOT, separate from _ANALYSIS_SLOTS so separation never starves
# voice notes); one more may queue behind it; a third is refused.
_HEAVY_SLOT = asyncio.Semaphore(1)
_ENCOUNTER_SLOT = asyncio.Semaphore(1)
_JOBS: dict = {}
_JOBS_MAX = 20


def _jobs_evict() -> None:
    while len(_JOBS) > _JOBS_MAX:
        done = [jid for jid, j in _JOBS.items() if j['status'] in ('done', 'error')]
        victim = done[0] if done else next(iter(_JOBS))
        _JOBS.pop(victim, None)


def _run_music_heavy(tmp_path: str, mode: str, filename: str, job: dict,
                     track: str = '', artist: str = '') -> str:
    """Blocking heavy pipeline — runs inside asyncio.to_thread under
    _HEAVY_SLOT. Returns the finished card text."""
    import stems as stems_mod

    def prog(frac: float) -> None:
        # Separation is ~90% of the wall clock; scale it to 0..0.9.
        job['progress'] = round(0.9 * frac, 3)

    x = stems_mod.load_stereo_44k(tmp_path)
    separated = stems_mod.separate(x, progress_cb=prog)
    names = stems_mod.write_stem_wavs(separated, DROPS_DIR)
    _prune_drops()
    base_url = (CONFIG['public_base_url'] or '').rstrip('/')
    if base_url:
        urls = {k: f'{base_url}/drops/{v}' for k, v in names.items()}
    else:
        urls = {k: f'/drops/{v} (no public_base_url configured — relative path)'
                for k, v in names.items()}

    job['progress'] = 0.92
    vocal_path = str(DROPS_DIR / names['vocals'])
    inst_path = str(DROPS_DIR / names['instrumental'])

    parts = [f'{filename} — mode {mode}']
    if mode == 'stems':
        # Free value: quick key/tempo of the mix since it's already decoded.
        try:
            quick = music.analyze_music(tmp_path)
            parts.append(music.format_music_section(quick, stems_urls=urls))
        except Exception as e:
            print(f'[attune-music-job] quick analysis failed: {e}', file=sys.stderr, flush=True)
            pairs = '  '.join(f'{k}: {u}' for k, u in urls.items())
            parts.append(f'STEMS : {pairs} (kept until ~200 newer drops arrive)')
        return '\n\n'.join(parts)

    # sing_vs_track: her voice on the vocal stem, the music on the rest.
    singing_result = None
    try:
        singing_result = analyze_singing(vocal_path)
        if singing_result and singing_result.get('is_melodic'):
            section = format_singing_section(singing_result) or ''
            if section:
                parts.append('VOICE (from separated vocal stem):\n' + section)
        else:
            parts.append('VOICE: vocal stem held no clear melodic line to analyze')
    except Exception as e:
        print(f'[attune-music-job] vocal analysis failed: {e}', file=sys.stderr, flush=True)
        parts.append('VOICE: vocal stem analysis failed')
    job['progress'] = 0.96

    music_result = None
    try:
        music_result = music.analyze_music(inst_path)
        melody = None
        try:
            melody = _melody_analysis(inst_path)
        except Exception:
            pass
        parts.append(music.format_music_section(music_result, melody=melody, stems_urls=urls))
    except Exception as e:
        print(f'[attune-music-job] instrumental analysis failed: {e}', file=sys.stderr, flush=True)
        parts.append('TRACK: instrumental analysis failed')

    if singing_result and music_result:
        try:
            cmp = music.compare_voice_to_track(singing_result, music_result)
            parts.append(music.format_compare_section(cmp))
        except Exception as e:
            print(f'[attune-music-job] compare failed: {e}', file=sys.stderr, flush=True)
    if track and artist:
        lyr_section = _lyrics_section(track, artist, music_result)
        if lyr_section:
            parts.append(lyr_section)
    return '\n\n'.join(parts)


async def _music_job_runner(job_id: str, tmp_path: str, mode: str, filename: str,
                            track: str = '', artist: str = '') -> None:
    job = _JOBS[job_id]
    try:
        async with _HEAVY_SLOT:
            job['status'] = 'running'
            card = await asyncio.to_thread(_run_music_heavy, tmp_path, mode, filename, job,
                                           track, artist)
            job['card'] = card
            job['progress'] = 1.0
            job['status'] = 'done'
            print(f'[attune-music-job] {job_id} done', file=sys.stderr, flush=True)
    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        print(f'[attune-music-job] {job_id} failed: {e}', file=sys.stderr, flush=True)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _start_music_job(tmp_path: str, mode: str, filename: str,
                     track: str = '', artist: str = '') -> str:
    job_id = secrets.token_urlsafe(8)
    _JOBS[job_id] = {'id': job_id, 'mode': mode, 'status': 'queued',
                     'progress': 0.0, 'created_at': time.time(),
                     'card': None, 'error': None, 'file': filename}
    _jobs_evict()
    asyncio.get_running_loop().create_task(
        _music_job_runner(job_id, tmp_path, mode, filename, track, artist))
    return job_id


def _mcp_music_tool_defs() -> list:
    prefixes = CONFIG['allowed_audio_prefixes']
    if prefixes:
        url_description = 'Public URL starting with one of: ' + ', '.join(prefixes)
        music_description = (
            'Music perception: key, tempo + beat confidence, chord progression, '
            'energy arc and section changes of a track from an allowed storage '
            'origin. Modes: auto (default), song, instrumental return the card '
            'immediately; sing_vs_track and stems separate vocals from the music '
            'first and return a job id to poll with music_job_status (jobs take '
            'minutes). Numbers with confidence labels, not mood diagnoses.'
        )
    else:
        url_description = 'Not usable until ATTUNE_ALLOWED_AUDIO_PREFIXES is configured on the server.'
        music_description = (
            'DISABLED — no ATTUNE_ALLOWED_AUDIO_PREFIXES configured on this server. '
            'Attune never ships an open fetch-anything endpoint.'
        )
    return [
        {
            'name': 'analyze_music',
            'description': music_description,
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'audio_url': {'type': 'string', 'description': url_description},
                    'youtube_url': {
                        'type': 'string',
                        'description': ('A youtube.com, music.youtube.com, or youtu.be URL. '
                                        'Attune downloads it locally before analysis.'),
                    },
                    'mode': {
                        'type': 'string',
                        'enum': list(MUSIC_MODES),
                        'description': (
                            "auto/song/instrumental: immediate card. sing_vs_track: separate "
                            "her voice from the backing track and analyze both (job). "
                            "stems: split vocals/instrumental into files (job)."
                        ),
                    },
                    'track': {
                        'type': 'string',
                        'description': ('Optional song title. With artist, adds a time-synced '
                                        'LYRICS section from the lrclib database (words are '
                                        'looked up, never transcribed from the mix).'),
                    },
                    'artist': {
                        'type': 'string',
                        'description': 'Optional artist name, used with track for the lyrics lookup.',
                    },
                },
            },
        },
        {
            'name': 'music_job_status',
            'description': (
                'Check a music analysis job started by analyze_music (sing_vs_track '
                'or stems mode). Returns progress, or the finished card. Poll no '
                'more than about once per 30 seconds. Jobs do not survive a server '
                'restart — an unknown id means resubmit.'
            ),
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'job_id': {'type': 'string', 'description': 'Job id returned by analyze_music'},
                },
                'required': ['job_id'],
            },
        },
    ]


async def _mcp_analyze_music(tool_args: dict) -> str:
    audio_url = str(tool_args.get('audio_url', ''))
    youtube_url = str(tool_args.get('youtube_url', '')).strip()
    mode = str(tool_args.get('mode', '') or 'auto').strip().lower()
    track = str(tool_args.get('track', '') or '').strip()[:200]
    artist = str(tool_args.get('artist', '') or '').strip()[:200]
    if mode not in MUSIC_MODES:
        return f"Refused: unknown mode '{mode}'. Modes: {', '.join(MUSIC_MODES)}."

    if bool(audio_url) == bool(youtube_url):
        return 'Refused: provide exactly one of audio_url or youtube_url.'

    if youtube_url:
        try:
            parsed = urllib.parse.urlsplit(youtube_url)
            host = (parsed.hostname or '').lower().rstrip('.')
            allowed_hosts = {'youtube.com', 'www.youtube.com', 'music.youtube.com', 'youtu.be'}
            if (parsed.scheme != 'https' or host not in allowed_hosts or parsed.username
                    or parsed.password or parsed.port not in (None, 443)):
                return 'Refused: youtube_url must be an HTTPS YouTube watch URL.'
        except (ValueError, UnicodeError):
            return 'Refused: invalid youtube_url.'
    else:
        prefixes = CONFIG['allowed_audio_prefixes']
        if not prefixes:
            return ('Refused: this server has no ATTUNE_ALLOWED_AUDIO_PREFIXES configured, so '
                    'music-by-URL analysis is disabled. Attune never ships an open '
                    'fetch-anything endpoint.')
        url_err = _validate_audio_url(audio_url, prefixes)
        if url_err:
            return url_err

    if mode in _MUSIC_HEAVY_MODES:
        import stems as stems_mod
        ok, reason = stems_mod.is_available()
        if not ok:
            return (f"Mode '{mode}' needs source separation, unavailable here: {reason}. "
                    'auto/song/instrumental modes are live.')
        queued = sum(1 for j in _JOBS.values() if j['status'] in ('queued', 'running'))
        if queued >= 2:
            return ('A music job is running and one is already queued — try again '
                    'when the current job finishes (music_job_status shows progress).')

    filename = 'youtube-track.mp3' if youtube_url else (audio_url.rsplit('/', 1)[-1] or 'track')
    suffix = '.mp3' if youtube_url else (Path(filename).suffix or '.mp3')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    owned_by_job = False
    try:
        try:
            if youtube_url:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                command = [
                    sys.executable, '-m', 'yt_dlp', '--no-playlist',
                    '--js-runtimes', 'node:/usr/bin/node',
                    '--max-filesize', '50M', '-x', '--audio-format', 'mp3',
                    '--audio-quality', '160K', '--force-overwrites',
                    '-o', tmp_path, youtube_url,
                ]
                completed = await asyncio.to_thread(
                    subprocess.run, command, capture_output=True, text=True, timeout=180,
                )
                if completed.returncode != 0 or not os.path.exists(tmp_path):
                    detail = (completed.stderr or '')[-500:]
                    raise RuntimeError(f'yt-dlp failed: {detail}')
            else:
                await asyncio.to_thread(_fetch_allowed_audio, audio_url, tmp_path)
        except ValueError as e:
            print(f'[attune-mcp] music fetch refused: {e}', file=sys.stderr, flush=True)
            return f'Refused: {e}'
        except Exception as e:
            print(f'[attune-mcp] music fetch failed: {e}', file=sys.stderr, flush=True)
            return 'Could not fetch that audio URL.'

        if mode in _MUSIC_HEAVY_MODES:
            # The job runner owns (and deletes) the temp file from here on.
            job_id = _start_music_job(tmp_path, mode, filename, track, artist)
            owned_by_job = True
            return (f'Job started: {job_id} ({mode}, {filename}). Separation takes '
                    'minutes on this hardware — poll music_job_status no more than '
                    'about once per 30 seconds. Jobs do not survive a server restart.')

        if _ANALYSIS_SLOTS.locked():
            return 'Attune is busy analyzing — try again in a moment.'
        async with _ANALYSIS_SLOTS:
            print(f'[attune-mcp] music analysis for {filename} (mode={mode})',
                  file=sys.stderr, flush=True)
            result, section = await _run_music_analysis(tmp_path)
        if not result:
            return 'Music analysis failed.'
        if track and artist:
            lyr_section = await asyncio.to_thread(_lyrics_section, track, artist, result)
            if lyr_section:
                section = f'{section}\n{lyr_section}' if section else lyr_section
        header = f'{filename} — mode {mode}'
        return f'{header}\n{section}' if section else f'{header}\n(no analyzable content found)'
    finally:
        if not owned_by_job:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


async def _mcp_music_job_status(tool_args: dict) -> str:
    job_id = str(tool_args.get('job_id', '')).strip()
    job = _JOBS.get(job_id)
    if not job:
        return (f"Unknown job '{job_id}' — jobs do not survive a server restart. "
                'Resubmit with analyze_music.')
    if job['status'] == 'queued':
        return f"Job {job_id} ({job['mode']}, {job['file']}): queued behind the current job."
    if job['status'] == 'running':
        return (f"Job {job_id} ({job['mode']}, {job['file']}): running, "
                f"{int(job['progress'] * 100)}% done.")
    if job['status'] == 'error':
        return f"Job {job_id} failed: {job['error']}"
    return job['card'] or f'Job {job_id} finished with no card.'


def _mcp_encounter_tool_defs() -> list:
    return [
        {
            'name': 'music_prepare_encounter',
            'description': (
                'Prepare one blind sequential first-listen encounter from an HTTPS '
                'YouTube or YouTube Music video. Identity, total duration, passage '
                'count, and future evidence stay hidden until listening is complete.'
            ),
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'youtube_url': {'type': 'string', 'format': 'uri'},
                    'mode': {'type': 'string', 'enum': ['adaptive', 'fixed'], 'default': 'adaptive'},
                },
                'required': ['youtube_url'],
                'additionalProperties': False,
            },
        },
        {
            'name': 'music_next_passage',
            'description': (
                'Receive or replay the current unlocked audio passage with only its '
                'causal measurements and provenance-labelled timed lyrics.'
            ),
            'inputSchema': {
                'type': 'object',
                'properties': {'session_id': {'type': 'string'}},
                'required': ['session_id'],
                'additionalProperties': False,
            },
        },
        {
            'name': 'music_record_impression',
            'description': 'Save the immutable first impression for the current passage.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'session_id': {'type': 'string'},
                    'passage_token': {'type': 'string'},
                    'impression': {'type': 'string'},
                },
                'required': ['session_id', 'passage_token', 'impression'],
                'additionalProperties': False,
            },
        },
        {
            'name': 'music_finish_encounter',
            'description': (
                'After every passage has an impression, reveal the recording identity '
                'and complete immutable first-listen journal.'
            ),
            'inputSchema': {
                'type': 'object',
                'properties': {'session_id': {'type': 'string'}},
                'required': ['session_id'],
                'additionalProperties': False,
            },
        },
        {
            'name': 'music_store_retrospective',
            'description': 'Store one separate immutable post-reveal interpretation.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'session_id': {'type': 'string'},
                    'retrospective': {'type': 'string'},
                    'salience': {'type': 'number', 'minimum': 0, 'maximum': 10, 'default': 7},
                },
                'required': ['session_id', 'retrospective'],
                'additionalProperties': False,
            },
        },
    ]


async def _mcp_encounter_call(tool_name: str, tool_args: dict):
    from attune_encounter import (
        EncounterError,
        finish_encounter,
        next_passage,
        passage_audio,
        prepare_youtube_encounter,
        record_impression,
        store_retrospective,
    )

    db_path = CONFIG['encounter_db']
    listener_id = CONFIG['encounter_listener_id']
    try:
        if tool_name == 'music_prepare_encounter':
            if _ENCOUNTER_SLOT.locked():
                return {'error': 'encounter preparation is busy; retry shortly'}, None
            async with _ENCOUNTER_SLOT:
                result = await asyncio.to_thread(
                    prepare_youtube_encounter,
                    db_path,
                    str(tool_args.get('youtube_url') or ''),
                    listener_id,
                    mode=str(tool_args.get('mode') or 'adaptive'),
                )
            return result, None
        session_id = str(tool_args.get('session_id') or '')
        if tool_name == 'music_next_passage':
            packet = await asyncio.to_thread(next_passage, db_path, session_id, listener_id)
            if packet.get('complete'):
                return packet, None
            data, mime_type, digest = await asyncio.to_thread(
                passage_audio, db_path, session_id, listener_id, packet['passage_id']
            )
            packet['audio'] = {'mime_type': mime_type, 'sha256': digest, 'attached': True}
            return packet, (data, mime_type)
        if tool_name == 'music_record_impression':
            return await asyncio.to_thread(
                record_impression, db_path, session_id, listener_id,
                str(tool_args.get('passage_token') or ''),
                str(tool_args.get('impression') or ''),
            ), None
        if tool_name == 'music_finish_encounter':
            return await asyncio.to_thread(
                finish_encounter, db_path, session_id, listener_id
            ), None
        if tool_name == 'music_store_retrospective':
            return await asyncio.to_thread(
                store_retrospective, db_path, session_id, listener_id,
                str(tool_args.get('retrospective') or ''),
                salience=tool_args.get('salience', 7),
            ), None
    except EncounterError as exc:
        return {'error': str(exc)}, None
    except Exception as exc:
        print(f'[attune-encounter] operation failed: {type(exc).__name__}', file=sys.stderr, flush=True)
        return {'error': 'operation failed'}, None
    return {'error': 'unknown encounter tool'}, None


async def _mcp_analyze(audio_url: str, transcript: str, language: str = '') -> str:
    prefixes = CONFIG['allowed_audio_prefixes']
    if not prefixes:
        return ('Refused: this server has no ATTUNE_ALLOWED_AUDIO_PREFIXES configured, so '
                'voice-note-by-URL analysis is disabled. Attune never ships an open '
                'fetch-anything endpoint.')
    url_err = _validate_audio_url(audio_url, prefixes)
    if url_err:
        return url_err
    if len(transcript.encode('utf-8')) > MAX_TRANSCRIPT_CHARS:
        return 'Refused: transcript too long.'

    filename = audio_url.rsplit('/', 1)[-1] or 'voice-note'
    suffix = Path(filename).suffix or '.webm'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    try:
        # Stream the download directly to tmp_path in the worker thread — no
        # large blob ever lives on the event loop.
        try:
            await asyncio.to_thread(_fetch_allowed_audio, audio_url, tmp_path)
        except ValueError as e:
            # Redirect refusal and the size-cap ValueError both land here
            # with an already-clear message -- surface it as-is.
            print(f'[attune-mcp] fetch refused: {e}', file=sys.stderr, flush=True)
            return f'Refused: {e}'
        except Exception as e:
            print(f'[attune-mcp] fetch failed: {e}', file=sys.stderr, flush=True)
            return 'Could not fetch that audio URL.'

        # Semaphore acquired BEFORE the STT/probe/chunk pipeline (not just
        # around the analyzer subprocess) -- same reasoning as /api/analyze:
        # _resolve_stt's own ffprobe/ffmpeg/webhook work must be bounded by
        # the two concurrent-analysis slots too.
        if _ANALYSIS_SLOTS.locked():
            return 'Attune is busy analyzing another note — try again in a moment.'
        async with _ANALYSIS_SLOTS:
            # No transcript supplied? Resolve one per ATTUNE_STT_MODE so
            # claude.ai callers can receive words + acoustics in one call.
            transcript, extra_args, stt_skipped, stt_partial = await _resolve_stt(tmp_path, transcript, language)

            args = [PYTHON, str(SEVEN_EARS_SCRIPT), tmp_path, '--json']
            args[3:3] = extra_args
            print(f'[attune-mcp] spawning analysis for {filename}', file=sys.stderr, flush=True)
            proc, stt_fell_back = await _spawn_analyzer(args, filename, '[attune-mcp]')
            if proc is None:
                return 'Analysis failed.'
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                return 'Analysis failed.'
            data['file'] = filename
            card = await asyncio.to_thread(_render_card, data)

            voice_delivery, voice_section = await _run_voice_delivery(tmp_path, data)
            if voice_delivery:
                data['voice_delivery'] = voice_delivery
            if voice_section:
                card = f'{card}\n{voice_section}' if card else voice_section

            singing, mcp_section = await _run_singing_analysis(tmp_path)
            if singing and singing.get('is_melodic'):
                data['singing'] = singing
            if mcp_section:
                card = f'{card}\n{mcp_section}' if card else mcp_section
        found_transcript = data.get('transcript', '')
        parts = [card or 'Card rendering unavailable — raw timing data was measured.']
        if found_transcript and not transcript:
            parts.append(f'Transcript: {found_transcript}')
        if stt_skipped:
            parts.append('Transcript omitted: note exceeds the transcription size limit.')
        if stt_partial:
            parts.append('Transcript is partial — some segments failed transcription.')
        if stt_fell_back:
            parts.append(STT_UNAVAILABLE_NOTE[0].upper() + STT_UNAVAILABLE_NOTE[1:])
        return '\n\n'.join(parts)
    finally:
        with contextlib.suppress(OSError):
            tmp.close()
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _rpc_result(req_id, result):
    return {'jsonrpc': '2.0', 'id': req_id, 'result': result}


def _rpc_error(req_id, code, message):
    return {'jsonrpc': '2.0', 'id': req_id, 'error': {'code': code, 'message': message}}


@app.get('/mcp')
async def mcp_get():
    # No server-initiated streams; streamable-HTTP allows a plain 405 here.
    raise HTTPException(405, 'POST only')


@app.post('/mcp')
async def mcp_post(request: Request):
    k = request.query_params.get('k', '')
    if not secrets.compare_digest(k, API_TOKEN):
        raise HTTPException(401, 'Unauthorized')
    try:
        msg = await request.json()
    except Exception:
        raise HTTPException(400, 'bad JSON')

    method = msg.get('method', '')
    req_id = msg.get('id')

    if method == 'initialize':
        client_ver = (msg.get('params') or {}).get('protocolVersion', '2025-03-26')
        return _rpc_result(req_id, {
            'protocolVersion': client_ver,
            'capabilities': {'tools': {}},
            'serverInfo': {'name': 'attune', 'version': '1.1.0'},
        })
    if method.startswith('notifications/'):
        return Response(status_code=202)
    if method == 'ping':
        return _rpc_result(req_id, {})
    if method == 'tools/list':
        return _rpc_result(req_id, {
            'tools': [_mcp_tool_def()] + _mcp_music_tool_defs() + _mcp_encounter_tool_defs()
        })
    if method == 'tools/call':
        params = msg.get('params') or {}
        tool_name = params.get('name')
        tool_args = params.get('arguments') or {}
        audio = None
        if tool_name == 'analyze_voice_note':
            text = await _mcp_analyze(
                str(tool_args.get('audio_url', '')),
                str(tool_args.get('transcript', '') or '').strip(),
                str(tool_args.get('language', '') or '').strip(),
            )
        elif tool_name == 'analyze_music':
            text = await _mcp_analyze_music(tool_args)
        elif tool_name == 'music_job_status':
            text = await _mcp_music_job_status(tool_args)
        elif tool_name in {tool['name'] for tool in _mcp_encounter_tool_defs()}:
            result, audio = await _mcp_encounter_call(tool_name, tool_args)
            text = json.dumps(result, ensure_ascii=False, allow_nan=False)
        else:
            return _rpc_error(req_id, -32602, 'unknown tool')
        content = [{'type': 'text', 'text': text}]
        if audio is not None:
            data, mime_type = audio
            content.insert(0, {
                'type': 'audio',
                'data': base64.b64encode(data).decode('ascii'),
                'mimeType': mime_type,
            })
        return _rpc_result(req_id, {'content': content})
    return _rpc_error(req_id, -32601, f'method not supported: {method}')


# ---------------------------------------------------------------------------
# ASGI guard middleware — RULING 1
# ---------------------------------------------------------------------------
# Runs OUTERMOST (registered last — Starlette builds the middleware stack
# last-added-outermost), so it sees every byte of POST /api/analyze and
# POST /api/drop before routing, multipart/body parsing, or CORSMiddleware.
# Auth-before-body: the credential is checked BEFORE any request body is
# read. The streamed cap wraps `receive` so an oversized body is rejected
# mid-stream, never buffered in full. Matching only method=='POST' and path
# in {'/api/analyze', '/api/drop'} is the explicit OPTIONS exemption — CORS
# preflights fall through untouched to CORSMiddleware.
#
# The two paths authenticate differently, same as their route handlers do:
# /api/analyze checks the bearer Authorization header; /api/drop checks the
# ?k= query param (the same credential the /drop page and /mcp both use —
# claude.ai's custom connectors, and this page's own JS, can't set custom
# headers without more machinery than a token-in-the-URL needs).

class _BodyTooLarge(Exception):
    pass


class _BodyStalled(Exception):
    pass


async def _send_json(send, status: int, obj: dict) -> None:
    body = json.dumps(obj).encode('utf-8')
    await send({
        'type': 'http.response.start',
        'status': status,
        'headers': [
            (b'content-type', b'application/json'),
            (b'content-length', str(len(body)).encode()),
            # Load-bearing: guard responses never reach CORSMiddleware, so
            # without this header the browser cannot read the 401/413 body.
            (b'access-control-allow-origin', b'*'),
        ],
    })
    await send({'type': 'http.response.body', 'body': body})


class AnalyzeGuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        if not (
            scope['type'] == 'http'
            and scope['method'] == 'POST'
            and path in ('/api/analyze', '/api/drop')
        ):
            await self.app(scope, receive, send)
            return

        ok = False
        # Decoded for BOTH paths: auth (analyze) and the content-length check
        # below (both). Initializing inside the analyze branch only caused an
        # UnboundLocalError crash on authenticated /api/drop requests.
        headers = {k.decode('latin-1').lower(): v.decode('latin-1') for k, v in scope['headers']}
        try:
            if path == '/api/analyze':
                auth = headers.get('authorization')
                expected = f'Bearer {API_TOKEN}'
                ok = auth is not None and secrets.compare_digest(auth, expected)
            else:  # '/api/drop' — auth rides ?k=, like /mcp and /drop
                query_string = scope.get('query_string', b'').decode('latin-1')
                qs = urllib.parse.parse_qs(query_string)
                k = (qs.get('k') or [''])[0]
                ok = secrets.compare_digest(k, API_TOKEN)
        except TypeError:
            # Malformed/non-ASCII header or query value — treat as unauth,
            # never let compare_digest's TypeError surface as a 500.
            ok = False
        if not ok:
            await _send_json(send, 401, {'detail': 'Unauthorized'})
            return

        content_length = headers.get('content-length')
        if content_length is not None and content_length.isdigit() and int(content_length) > MAX_UPLOAD_BYTES:
            await _send_json(send, 413, {'detail': f'audio exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MiB limit'})
            return

        total = 0
        started = False

        async def guarded_receive():
            nonlocal total
            # Idle deadline per chunk: an authenticated client dripping bytes
            # forever would otherwise hold parser/connection state open.
            try:
                message = await asyncio.wait_for(receive(), timeout=BODY_READ_IDLE_TIMEOUT_S)
            except asyncio.TimeoutError:
                raise _BodyStalled()
            if message['type'] == 'http.request':
                total += len(message.get('body', b''))
                if total > MAX_UPLOAD_BYTES:
                    raise _BodyTooLarge()
            return message

        async def guarded_send(message):
            nonlocal started
            if message['type'] == 'http.response.start':
                started = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, guarded_send)
        except _BodyTooLarge:
            if not started:
                await _send_json(send, 413, {'detail': f'audio exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MiB limit'})
            # else: response already started — connection aborts.
        except _BodyStalled:
            if not started:
                await _send_json(send, 408, {'detail': 'upload timed out'})


# Registration order matters: CORSMiddleware first, then the guard LAST so
# it is outermost in the resulting stack.
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
app.add_middleware(AnalyzeGuardMiddleware)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=PORT)
