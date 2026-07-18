"""Attune — voice-note acoustic + singing analysis.

Thin FastAPI wrapper around the vendored seven-ears engine (Seven Verity +
Sunny, MIT, pinned d33e7c1). Acoustics always run locally. Speech-to-text is
pluggable — local faster-whisper, a webhook to your own STT service, or none
at all (acoustics-only) — see the CONFIGURATION section below and README.md.
"""

import asyncio
import contextlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
    'max_upload_bytes': _cfg_int('ATTUNE_MAX_UPLOAD_BYTES', 'max_upload_bytes', 50 * 1024 * 1024),
    # transcript rides argv; Windows CreateProcess caps ~32k
    'max_transcript_chars': _cfg_int('ATTUNE_MAX_TRANSCRIPT_CHARS', 'max_transcript_chars', 16 * 1024),
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
    'allowed_audio_prefixes={n_prefixes} configured max_upload_bytes={max_upload_bytes}'.format(
        port=CONFIG['port'],
        token_file=CONFIG['token_file'],
        ffmpeg_dir=CONFIG['ffmpeg_dir'] or '(relying on PATH)',
        stt_mode=CONFIG['stt_mode'],
        whisper_model=CONFIG['whisper_model'] if CONFIG['stt_mode'] == 'local' else '-',
        stt_url=CONFIG['stt_url'] or '(unset)',
        n_prefixes=len(CONFIG['allowed_audio_prefixes']),
        max_upload_bytes=CONFIG['max_upload_bytes'],
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

from singing import analyze_singing, format_singing_section  # noqa: E402  (after config)

# ---------------------------------------------------------------------------
# END CONFIGURATION
# ---------------------------------------------------------------------------

PORT = CONFIG['port']
TOKEN_FILE = CONFIG['token_file']
MAX_UPLOAD_BYTES = CONFIG['max_upload_bytes']
MAX_TRANSCRIPT_CHARS = CONFIG['max_transcript_chars']
BODY_READ_IDLE_TIMEOUT_S = 30  # per-chunk gap allowed while receiving the upload
STT_MAX_BYTES = CONFIG['stt_max_bytes']

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


@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'ffmpeg': shutil.which('ffmpeg', path=_subprocess_env()['PATH']) is not None,
        'engine': f"acoustics-local/stt-{CONFIG['stt_mode']}",
        # Honest presence check for the vendored engine itself (separate
        # from ffmpeg discoverability above) -- False before
        # scripts/get-seven-ears.py has been run.
        'engine_present': SEVEN_EARS_SCRIPT.exists(),
    }


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


def _webhook_stt(path: str) -> str:
    """Transcribe via the configured ATTUNE_STT_URL webhook. Best-effort.
    Reads the audio file from disk here, inside the worker thread, rather
    than the caller holding a large blob on the event loop. Never loads a
    file bigger than STT_MAX_BYTES into memory -- this is the last line of
    defense even though callers are expected to skip calling this at all
    once they know the file exceeds that size."""
    import urllib.request
    token = _webhook_token()
    if not token or not CONFIG['stt_url']:
        return ''
    try:
        if os.path.getsize(path) > STT_MAX_BYTES:
            return ''
    except OSError:
        return ''
    with open(path, 'rb') as f:
        blob = f.read()
    req = urllib.request.Request(
        CONFIG['stt_url'], data=blob, method='POST',
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


async def _resolve_stt(tmp_path: str, transcript: str) -> tuple:
    """Given a temp audio file and any already-known transcript, return
    (transcript, extra_cli_args, stt_skipped) honoring ATTUNE_STT_MODE.
    A caller-supplied transcript always wins and short-circuits STT_MODE
    entirely -- this is what keeps a frontend that already runs its own STT
    (like this deployment's) behaving identically regardless of STT_MODE."""
    if transcript:
        return transcript, ['--transcript', transcript, '--transcript-source', 'stt_api'], False

    mode = CONFIG['stt_mode']
    if mode == 'local':
        return '', ['--stt', 'whisper', '--whisper-model', CONFIG['whisper_model']], False
    if mode == 'webhook':
        stt_skipped = False
        try:
            size = os.path.getsize(tmp_path)
        except OSError:
            size = 0
        if size > STT_MAX_BYTES:
            stt_skipped = True
        else:
            transcript = (await asyncio.to_thread(_webhook_stt, tmp_path)).strip()
        args = ['--transcript', transcript, '--transcript-source', 'stt_api'] if transcript else []
        return transcript, args, stt_skipped
    # mode == 'none': acoustics only
    return '', [], False


async def _spawn_analyzer(args: list, filename: str, log_tag: str):
    """Run the seven-ears analyzer subprocess. Returns the completed process
    on success, or None after logging on any failure.

    Local-STT resilience: if ATTUNE_STT_MODE=local but faster-whisper isn't
    installed, seven-ears exits nonzero when whisper is explicitly requested.
    Rather than failing the whole analysis over a missing OPTIONAL dependency,
    retry once without the --stt args — the caller gets acoustics (and any
    singing analysis) with an empty transcript, as documented in the README.
    """
    attempt_args = args
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
            return None
        except OSError as e:
            print(f'{log_tag} analysis spawn failed: {e}', file=sys.stderr, flush=True)
            return None
        if proc.returncode == 0:
            return proc
        stderr_tail = proc.stderr[-800:] if proc.stderr else ''
        if attempt == 1 and '--stt' in attempt_args and 'faster_whisper' in stderr_tail:
            print(f'{log_tag} local STT unavailable (faster-whisper not importable); retrying acoustics-only', file=sys.stderr, flush=True)
            i = attempt_args.index('--stt')
            attempt_args = attempt_args[:i] + attempt_args[i + 4:]  # drop --stt whisper --whisper-model <m>
            continue
        print(f'{log_tag} analysis failed: {stderr_tail}', file=sys.stderr, flush=True)
        return None
    return None


@app.post('/api/analyze')
async def analyze(
    request: Request,
    audio: UploadFile = File(...),
    transcript: Optional[str] = Form(None),
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

        t, extra_args, _stt_skipped = await _resolve_stt(tmp_path, t)

        args = [PYTHON, str(SEVEN_EARS_SCRIPT), tmp_path, '--json']
        args[3:3] = extra_args

        if _ANALYSIS_SLOTS.locked():
            raise HTTPException(429, 'analysis busy, retry shortly')
        async with _ANALYSIS_SLOTS:
            print(f'[attune] spawning analysis for {filename}', file=sys.stderr, flush=True)
            proc = await _spawn_analyzer(args, filename, '[attune]')
            if proc is None:
                # Details stay in the server log; clients get a fixed message.
                raise HTTPException(500, 'analysis failed')

            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                raise HTTPException(500, 'analysis failed')

            # Overwrite with the friendly filename before rendering the card —
            # format_card requires data['file'] and this prevents the OS temp
            # path from leaking into the rendered card text.
            data['file'] = filename
            card_text = await asyncio.to_thread(_render_card, data)

            singing, section = await _run_singing_analysis(tmp_path)
            if singing and singing.get('is_melodic'):
                data['singing'] = singing
            if section:
                card_text = f'{card_text}\n{section}' if card_text else section

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
            'the Attune acoustic card: timing, pauses, pace, pitch, dynamics. '
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
            },
            'required': ['audio_url'],
        },
    }


async def _mcp_analyze(audio_url: str, transcript: str) -> str:
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

        # No transcript supplied? Resolve one per ATTUNE_STT_MODE so claude.ai
        # callers can receive words + acoustics in a single tool call.
        transcript, extra_args, stt_skipped = await _resolve_stt(tmp_path, transcript)

        args = [PYTHON, str(SEVEN_EARS_SCRIPT), tmp_path, '--json']
        args[3:3] = extra_args
        if _ANALYSIS_SLOTS.locked():
            return 'Attune is busy analyzing another note — try again in a moment.'
        async with _ANALYSIS_SLOTS:
            print(f'[attune-mcp] spawning analysis for {filename}', file=sys.stderr, flush=True)
            proc = await _spawn_analyzer(args, filename, '[attune-mcp]')
            if proc is None:
                return 'Analysis failed.'
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                return 'Analysis failed.'
            data['file'] = filename
            card = await asyncio.to_thread(_render_card, data)

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
            'serverInfo': {'name': 'attune', 'version': '1.0.0'},
        })
    if method.startswith('notifications/'):
        return Response(status_code=202)
    if method == 'ping':
        return _rpc_result(req_id, {})
    if method == 'tools/list':
        return _rpc_result(req_id, {'tools': [_mcp_tool_def()]})
    if method == 'tools/call':
        params = msg.get('params') or {}
        if params.get('name') != 'analyze_voice_note':
            return _rpc_error(req_id, -32602, 'unknown tool')
        tool_args = params.get('arguments') or {}
        text = await _mcp_analyze(
            str(tool_args.get('audio_url', '')),
            str(tool_args.get('transcript', '') or '').strip(),
        )
        return _rpc_result(req_id, {'content': [{'type': 'text', 'text': text}]})
    return _rpc_error(req_id, -32601, f'method not supported: {method}')


# ---------------------------------------------------------------------------
# ASGI guard middleware — RULING 1
# ---------------------------------------------------------------------------
# Runs OUTERMOST (registered last — Starlette builds the middleware stack
# last-added-outermost), so it sees every byte of POST /api/analyze before
# routing, multipart parsing, or CORSMiddleware. Auth-before-body: the
# Authorization header is checked BEFORE any request body is read. The
# streamed cap wraps `receive` so an oversized body is rejected mid-stream,
# never buffered in full. Matching only method=='POST' and path==
# '/api/analyze' is the explicit OPTIONS exemption — CORS preflights fall
# through untouched to CORSMiddleware.

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
        if not (
            scope['type'] == 'http'
            and scope['method'] == 'POST'
            and scope['path'] == '/api/analyze'
        ):
            await self.app(scope, receive, send)
            return

        headers = {k.decode('latin-1').lower(): v.decode('latin-1') for k, v in scope['headers']}
        auth = headers.get('authorization')
        expected = f'Bearer {API_TOKEN}'
        ok = False
        try:
            ok = auth is not None and secrets.compare_digest(auth, expected)
        except TypeError:
            # Malformed/non-ASCII Authorization header — treat as unauth,
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
