# Security Policy

## Reporting a vulnerability

Please report security issues **privately** via GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository (Security tab → "Report a vulnerability"), rather than
opening a public issue. You'll get a response as soon as we're able —
this is a small community project, not a company with an on-call rotation.

## Scope notes for deployers

Attune is designed to run on a machine you control, behind transport you
trust. Things worth knowing before exposing it anywhere:

- **Auth is a single bearer token** (`ATTUNE_TOKEN_FILE`) shared by every
  caller. There are no accounts, roles, or rate limits per identity. If the
  token leaks, rotate it by deleting the file and restarting.
- **The `/mcp` endpoint carries the token in the URL** (`?k=`), because
  claude.ai custom connectors can't send bearer headers without full OAuth.
  URLs get logged by proxies — treat any URL containing the token as a
  credential.
- **URL fetching is allowlist-only and refuses redirects** by design. Keep
  `ATTUNE_ALLOWED_AUDIO_PREFIXES` as narrow as possible; every prefix you
  add is a place the server will fetch from on request.
- **Analysis is CPU-bounded but real work.** Size caps, duration caps, a
  concurrency limit, and timeouts are built in, but a determined
  token-holder can still keep your CPU busy. Don't share the token with
  anyone you wouldn't let run code-adjacent workloads on the box.
- If you expose Attune to the internet, put it behind TLS + access control
  you already trust (reverse proxy, Cloudflare Tunnel, Tailscale Funnel).

## Supported versions

The latest commit on `main`. There are no maintained release branches.
