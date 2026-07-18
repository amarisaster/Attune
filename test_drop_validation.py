"""Plain-assert tests for server.DROPS_NAME_RE, the strict allowlist regex
guarding GET /drops/<name>. Run directly:
    venv\\Scripts\\python.exe test_drop_validation.py
Importing server.py does the same things it does on a normal boot (loads
attune.config.json if present, generates/reads the token file, creates the
drops dir) -- that's exercised deliberately here rather than mocked, since
the whole point is to test the actual regex the running server uses, not a
copy of it.
"""

import sys

from server import DROPS_NAME_RE


def _assert_valid(name):
    assert DROPS_NAME_RE.fullmatch(name), f'expected valid, rejected: {name!r}'


def _assert_invalid(name):
    assert not DROPS_NAME_RE.fullmatch(name), f'expected invalid, accepted: {name!r}'


def case_valid_names():
    print('--- case (a): well-formed drop names are accepted ---')
    for name in [
        'aB3-_9.webm',
        'note123.ogg',
        'x.m4a',
        'y.mp3',
        'z.wav',
        '0123456789.webm',
        'a-b_c-d_e.webm',
        ('a' * 64) + '.webm',  # secrets.token_urlsafe(16)-length names are far shorter; long names still fine
    ]:
        _assert_valid(name)
    print('PASS case (a)')


def case_traversal_rejected():
    print('--- case (b): path traversal in any form is rejected ---')
    for name in [
        '../evil.webm',
        '..\\evil.webm',
        '../../etc/passwd.webm',
        '..%2Fevil.webm',
        '%2e%2e%2Fevil.webm',
        '%2e%2e%5Cevil.webm',
        'a/../../b.webm',
        '....//evil.webm',
        'a/b.webm',
        'a\\b.webm',
    ]:
        _assert_invalid(name)
    print('PASS case (b)')


def case_unc_and_absolute_paths_rejected():
    print('--- case (c): UNC paths and absolute/drive paths are rejected ---')
    for name in [
        '\\\\server\\share\\evil.webm',
        '//server/share/evil.webm',
        'C:\\evil.webm',
        '/etc/passwd.webm',
        '\\evil.webm',
    ]:
        _assert_invalid(name)
    print('PASS case (c)')


def case_bad_extension_or_shape_rejected():
    print('--- case (d): disallowed extensions / malformed shapes are rejected ---')
    for name in [
        'evil.exe',
        'evil.php',
        'evil.webm.exe',
        'noext',
        '.webm',           # empty basename
        '',                # empty string entirely
        'a b.webm',         # whitespace
        'a.WEBM',           # uppercase extension not in the allowed set
        'a.webm ',          # trailing whitespace
        ' a.webm',          # leading whitespace
        'a.webm\x00.exe',   # embedded NUL
        'a.tar.gz',
    ]:
        _assert_invalid(name)
    print('PASS case (d)')


if __name__ == '__main__':
    case_valid_names()
    case_traversal_rejected()
    case_unc_and_absolute_paths_rejected()
    case_bad_extension_or_shape_rejected()
    print('ALL PASS')
    sys.exit(0)
