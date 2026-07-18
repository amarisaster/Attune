#!/usr/bin/env python3
"""Fetch the vendored seven-ears engine at its pinned commit.

Clones https://github.com/meatwife/seven-ears into vendor/seven-ears and
checks out the commit Attune is built against. Stdlib only (subprocess +
pathlib) -- no third-party dependency needed just to run setup.

Idempotent: safe to re-run. If vendor/seven-ears already exists and is
already at the pinned commit, this is a no-op. If it exists but is at a
different commit, it fetches and checks out the pinned commit in place
(no data loss for a plain clone -- if you've made local edits to the
vendored copy, stash or back them up first).
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = 'https://github.com/meatwife/seven-ears'
PINNED_COMMIT = 'd33e7c1929237cfbb0c77e01c55811ee9f2360e5'

BASE_DIR = Path(__file__).resolve().parent.parent
VENDOR_DIR = BASE_DIR / 'vendor' / 'seven-ears'


def _run(args, cwd=None, check=True):
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def main() -> int:
    if shutil.which('git') is None:
        print('ERROR: git is not on PATH. Install git and re-run this script.', file=sys.stderr)
        print('  https://git-scm.com/downloads', file=sys.stderr)
        return 1

    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)

    if (VENDOR_DIR / '.git').exists():
        # Check HEAD FIRST, before touching the network at all -- the
        # common case (already vendored at the pin) should be a pure local
        # check with zero fetch/clone traffic.
        try:
            current = _run(['git', 'rev-parse', 'HEAD'], cwd=VENDOR_DIR).stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f'ERROR: git rev-parse failed:\n{e.stderr}', file=sys.stderr)
            return 1
        if current == PINNED_COMMIT:
            print(f'Already at pinned commit {PINNED_COMMIT[:12]} -- nothing to do.')
            return 0
        print(f'{VENDOR_DIR} exists at {current[:12]} (not the pin) -- fetching latest refs')
        try:
            _run(['git', 'fetch', '--all'], cwd=VENDOR_DIR)
        except subprocess.CalledProcessError as e:
            print(f'ERROR: git fetch failed:\n{e.stderr}', file=sys.stderr)
            return 1
    else:
        if VENDOR_DIR.exists() and any(VENDOR_DIR.iterdir()):
            print(f'ERROR: {VENDOR_DIR} already exists and is not a git checkout.', file=sys.stderr)
            print('Remove it (or move it aside) and re-run this script.', file=sys.stderr)
            return 1
        print(f'Cloning {REPO_URL} -> {VENDOR_DIR}')
        try:
            _run(['git', 'clone', REPO_URL, str(VENDOR_DIR)])
        except subprocess.CalledProcessError as e:
            print(f'ERROR: git clone failed:\n{e.stderr}', file=sys.stderr)
            return 1
        current = _run(['git', 'rev-parse', 'HEAD'], cwd=VENDOR_DIR).stdout.strip()
        if current == PINNED_COMMIT:
            print(f'Cloned default branch is already at the pinned commit {PINNED_COMMIT[:12]}.')
            return 0

    print(f'Checking out pinned commit {PINNED_COMMIT[:12]}')
    try:
        _run(['git', 'checkout', PINNED_COMMIT], cwd=VENDOR_DIR)
    except subprocess.CalledProcessError as e:
        print(f'ERROR: git checkout failed:\n{e.stderr}', file=sys.stderr)
        print('If this repo was cloned shallow and the commit is unreachable, try:', file=sys.stderr)
        print(f'  git -C "{VENDOR_DIR}" fetch --unshallow', file=sys.stderr)
        return 1

    print(f'vendor/seven-ears is now at {PINNED_COMMIT[:12]}. Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
