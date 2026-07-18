# Attune

A voice-note analysis service. Receives a recorded voice note and returns:

- **Acoustic measurements** — timing, pauses, pace, pitch, dynamics (via the
  vendored [seven-ears](https://github.com/meatwife/seven-ears) engine)
- **Singing analysis** — melody notes, glides/slides, vibrato, a dynamics
  arc, and a key guess, when the clip is melodic (via `singing.py`, a
  pure-numpy addition on top of seven-ears)
- Optionally a **transcript**, depending on how you configure speech-to-text

**Numbers, not diagnoses.** Everything Attune reports is a measurement —
timing, Hz, dB, cents. It never labels a mood or makes a clinical claim.
Acoustic cues can support a human's own interpretation of their own voice in
context; they don't prove an internal state, and nothing here should be
treated as a diagnostic signal.

## Credit

The analysis engine is [seven-ears](https://github.com/meatwife/seven-ears)
by **Seven Verity** (AI companion) and **Sunny** (MIT license), vendored
unmodified at `vendor/seven-ears/` and pinned to commit `d33e7c1`. Attune is
only the thin HTTP wrapper around it, plus the singing-analysis addition.
Thank you both — this tool's ethics are as good as its measurements.

## Install

Requires Python 3.11+ and [ffmpeg](https://ffmpeg.org/download.html) (either
on your `PATH`, or point `ATTUNE_FFMPEG_DIR` at its `bin/` folder — see
Configuration below).

```bash
# 1. Vendor the pinned analysis engine
python scripts/get-seven-ears.py

# 2. Install Attune's own dependencies
python -m venv venv
venv/bin/pip install -r requirements.txt      # venv\Scripts\pip on Windows

# 3. (Optional, only for ATTUNE_STT_MODE=local) install seven-ears' own deps,
#    which include faster-whisper
venv/bin/pip install -r vendor/seven-ears/requirements.txt
```

## Configuration

Attune reads config from, in order of precedence: **environment variable**,
then **`attune.config.json`** (a file you create next to `server.py` —
gitignored, see `attune.config.example.json` for a template), then the
default below. The resolved profile (never secrets) is logged to stderr on
boot.

An environment variable that is *set but empty/whitespace* counts as an
explicit override, not "unset" — it resolves straight to the default and
never falls through to `attune.config.json`, for every key (list-valued
keys like `ATTUNE_ALLOWED_AUDIO_PREFIXES` and scalar keys alike). Only a
genuinely unset environment variable falls through to the JSON file. The
same applies if `ATTUNE_PORT`/`ATTUNE_MAX_UPLOAD_BYTES`/etc. is set to
something that isn't a valid integer: Attune warns to stderr and uses the
default, without consulting the JSON file either.

| Env var | JSON key | Default | Meaning |
|---|---|---|---|
| `ATTUNE_PORT` | `port` | `8452` | HTTP port |
| `ATTUNE_TOKEN_FILE` | `token_file` | `~/.attune-token` | Bearer token file; auto-generated on first boot if missing |
| `ATTUNE_FFMPEG_DIR` | `ffmpeg_dir` | *(unset)* | ffmpeg's `bin/` directory, if not already on `PATH` |
| `ATTUNE_ALLOWED_AUDIO_PREFIXES` | `allowed_audio_prefixes` | *(empty)* | Comma-separated (env) or JSON array of URL prefixes the MCP tool is allowed to fetch from. **Required** to enable `analyze_voice_note` — see Security notes |
| `ATTUNE_STT_MODE` | `stt_mode` | `local` | `local` \| `webhook` \| `none` — see below |
| `ATTUNE_WHISPER_MODEL` | `whisper_model` | `base.en` | faster-whisper model name, used when `stt_mode=local` |
| `ATTUNE_STT_URL` | `stt_url` | *(unset)* | Your own STT endpoint, used when `stt_mode=webhook` |
| `ATTUNE_STT_TOKEN_FILE` | `stt_token_file` | *(unset)* | File containing the bearer token for `ATTUNE_STT_URL` |
| `ATTUNE_STT_MAX_BYTES` | `stt_max_bytes` | `6291456` (6 MiB) | Size cap before STT is skipped |
| `ATTUNE_MAX_UPLOAD_BYTES` | `max_upload_bytes` | `52428800` (50 MiB) | Upload size cap, enforced streaming |
| `ATTUNE_MAX_TRANSCRIPT_CHARS` | `max_transcript_chars` | `16384` | Caller-supplied transcript length cap, measured in **UTF-8 bytes** (not characters). The key name says "chars" for backward compatibility with existing configs, but the enforcement has always been byte-based. |

**STT modes**, when the caller doesn't already supply a transcript:

- `local` (the default) — Attune asks the vendored seven-ears CLI to
  transcribe locally with faster-whisper (`--stt whisper --whisper-model
  <model>`). Most machines can run this fine, but it **requires step 3 of
  Install** (`venv/bin/pip install -r vendor/seven-ears/requirements.txt`,
  which pulls in faster-whisper). If that step was skipped, or
  faster-whisper otherwise fails to import, Attune detects the failed
  transcription attempt and automatically retries the analysis
  acoustics-only — you get a response with acoustics (and singing
  analysis) but an empty `transcript` field, rather than an error.
- `webhook` — Attune POSTs the raw audio to `ATTUNE_STT_URL` with
  `Authorization: Bearer <contents of ATTUNE_STT_TOKEN_FILE>`, and expects
  back `{"transcript": "..."}`. Use this if you already run STT elsewhere
  (a cloud function, another service), or if local transcription isn't an
  option on this machine (see Platform notes).
- `none` — no transcription at all; acoustics (and singing analysis) only.

If a caller already supplies a transcript with the request, it's used as-is
and `ATTUNE_STT_MODE` is never consulted.

## Run

```bash
venv/bin/python server.py      # venv\Scripts\python.exe server.py on Windows
```

On first boot, a bearer token is generated and written to
`ATTUNE_TOKEN_FILE` (default `~/.attune-token`). Every request below except
`/health` requires it.

### Endpoints

- `GET /health` → `{status, ffmpeg, engine, engine_present}` — no auth
  required. `ffmpeg` reports whether ffmpeg is discoverable on `PATH` (or
  `ATTUNE_FFMPEG_DIR`); `engine_present` reports whether the vendored engine
  itself (`vendor/seven-ears/seven_ears_card.py`) exists — `false` until
  `scripts/get-seven-ears.py` has been run (see Install step 1).
- `POST /api/analyze` — multipart form, field `audio` (the recording, required)
  and optional `transcript`; header `Authorization: Bearer <token>`. Returns
  `{transcript, card_text, measurements}`.
- `POST /mcp?k=<token>` — a minimal stateless streamable-HTTP MCP endpoint
  exposing one tool, `analyze_voice_note`, for
  [claude.ai custom connectors](https://support.claude.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp).
  The tool takes an `audio_url` (must start with a configured
  `ATTUNE_ALLOWED_AUDIO_PREFIXES` entry) and an optional `transcript`.

  To add it as a custom connector: in claude.ai, go to **Settings → Connectors
  → Add custom connector**, and enter your server's `/mcp` URL with the token
  query param, e.g. `https://your-host/mcp?k=<your-token>`. claude.ai's
  custom connectors can't send bearer headers without full OAuth, which is
  why auth here rides the URL instead.

## Security notes

- **Bearer gating.** `/api/analyze` is gated by an ASGI middleware that
  checks the Authorization header *before* reading any request body — an
  unauthenticated request never gets its bytes parsed. `/mcp` checks the `k`
  query param the same way, before touching the JSON-RPC body.
- **Allowed-origins design, not an open fetch.** The MCP tool only ever
  downloads from URL prefixes you explicitly configure via
  `ATTUNE_ALLOWED_AUDIO_PREFIXES`. With that unset (the default), the tool
  is present but refuses every URL with a clear message. This is
  deliberate: a "fetch any URL and analyze it" tool is a server-side-request-
  forgery surface, and Attune will never ship with that open by default.
  The check itself is hardened, not a raw `startswith`: `audio_url` is
  parsed (`urllib.parse`) and must be `https`, its hostname must exactly
  match a configured prefix's hostname, the port must match (default 443
  on both sides), any userinfo/credentials must match the prefix's own, and
  the request path is percent-decoded and checked for `..` traversal
  (including `%2e%2e`-style encoding) before being matched as a genuine
  path-segment prefix — a path like `/bucket-evil/x` can never match a
  configured `/bucket/` prefix.
- **Redirects are refused by design.** The MCP audio fetch never follows a
  3xx response — a redirect could otherwise be used to bounce an
  already-allowlisted URL to a host or path the allowlist was never meant
  to approve. A redirect response is surfaced to the caller as a clear
  refusal instead of being followed transparently.
- **Size caps everywhere on audio payloads.** Uploads and MCP-fetched
  audio are capped and enforced as bytes stream in — nothing is buffered
  in full before the cap is checked. The webhook-STT cap is checked
  against the already-on-disk file's size before it is read. (JSON-RPC request
  bodies on `/mcp`, e.g. the `tools/call` envelope, are small and parsed
  normally — this streaming-cap guarantee is specifically about the audio
  bytes, not every request body Attune ever reads.)
- **Run behind your own tunnel if exposing this to the internet.** Attune
  itself only does bearer-token auth; if you're exposing it beyond
  localhost, put it behind something you already trust for TLS + access
  control (a reverse proxy, Cloudflare Tunnel, Tailscale Funnel, etc.) rather
  than binding it directly to a public IP.

## Platform notes

- **Windows with Smart App Control (or similar DLL-signing enforcement)**
  can block faster-whisper's unsigned native DLLs, making `ATTUNE_STT_MODE=local`
  fail outright. If local transcription won't run on your machine, switch to
  `webhook` (point it at any STT service you control) or `none` (acoustics
  and singing analysis only, no transcript). The acoustic and singing
  pipelines have no native audio dependencies (no librosa/numba/soundfile/
  scipy — see `singing.py`'s module docstring) and are unaffected either way.
- **Linux / macOS** — `local` mode (faster-whisper) works normally; no
  special handling needed.

## Example card output

A synthetic example, not a real recording:

```
ATTUNE
File: example-note.webm
Duration: 8.4s | Words: 22 | Pace: 157 wpm

TIMING: 3 pauses (0.4s, 0.6s, 0.3s) | longest gap at 4.1s
PACE  : starts slow (128 wpm), builds through the middle (180 wpm), settles by the end (150 wpm)
PITCH : range 92-210 Hz, median 138 Hz | rises on emphasis at 2.1s, 5.8s
DYNAMICS: starts at -22 dB, builds +6 dB by 3.0s, quietest at 6.2s (-28 dB)

MELODY: C4 → D4 → slide D4→F4 (0.6s) → F4 (5 notes, range C4–F4, ~key C major [weak])
LINE  : rises through the phrase, holds F4 0.7s with vibrato ~5.4 Hz (±38 cents)
POWER : starts at -20 dB, building +5 dB by the end, widest at 0:04
TEMPO : ~92 BPM candidate (low confidence — rubato likely)
```

## Layout

- `server.py` — FastAPI wrapper (port defaults to 8452; see Configuration)
- `singing.py` — pure-numpy singing analysis (melody, vibrato, key, dynamics)
- `vendor/seven-ears/` — pinned upstream engine (fetched by `scripts/get-seven-ears.py`)
- `scripts/get-seven-ears.py` — vendoring/setup script, stdlib only
- `test_singing.py` — smoke tests for `singing.py`, run directly with no pytest needed
- `attune.config.example.json` — configuration template
