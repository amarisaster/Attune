"""Fetch the pinned ONNX models Attune's music perception uses.

Mirrors scripts/get-seven-ears.py: stdlib only, idempotent, pinned by URL and
SHA256 — an existing file with a matching hash is a no-op; a hash mismatch is
an error and never a silent overwrite.

Run:  python scripts/get-models.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

MODELS = [
    {
        'name': 'basic-pitch-nmp.onnx',
        'url': ('https://raw.githubusercontent.com/spotify/basic-pitch/v0.4.0/'
                'basic_pitch/saved_models/icassp_2022/nmp.onnx'),
        'sha256': '2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec',
        'bytes': 230444,
        'license': 'Apache-2.0 (Spotify basic-pitch, ICASSP 2022 nmp checkpoint)',
    },
    {
        'name': 'UVR-MDX-NET-Voc_FT.onnx',
        'url': ('https://github.com/TRvlvr/model_repo/releases/download/'
                'all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx'),
        'sha256': '534b2070fcc7df514b13ef660dc8cbb328679c2374d04354a5c42bb14ecce111',
        'bytes': 66762490,
        'license': 'MIT (Ultimate Vocal Remover model release)',
    },
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(model: dict) -> None:
    dest = MODELS_DIR / model['name']
    if dest.exists():
        got = sha256_of(dest)
        if got == model['sha256']:
            print(f"ok       {model['name']} (already present, hash verified)")
            return
        print(f"ERROR    {model['name']} exists but hash mismatch:")
        print(f"  expected {model['sha256']}")
        print(f"  got      {got}")
        print('  Refusing to overwrite. Move the file aside and rerun.')
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    print(f"fetching {model['name']} ({model['bytes'] / 1e6:.1f} MB) ...")
    urllib.request.urlretrieve(model['url'], part)  # noqa: S310 — pinned https URL
    got = sha256_of(part)
    if got != model['sha256']:
        part.unlink(missing_ok=True)
        print(f"ERROR    {model['name']} download hash mismatch — not installed.")
        print(f"  expected {model['sha256']}")
        print(f"  got      {got}")
        sys.exit(1)
    part.rename(dest)
    print(f"ok       {model['name']} installed, hash verified")


def write_readme() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ['Models fetched by scripts/get-models.py — pinned by SHA256.\n']
    for m in MODELS:
        lines.append(f"{m['name']}\n  source : {m['url']}\n  sha256 : {m['sha256']}\n"
                     f"  license: {m['license']}\n")
    (MODELS_DIR / 'README.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    for m in MODELS:
        fetch(m)
    write_readme()
    print('done')
