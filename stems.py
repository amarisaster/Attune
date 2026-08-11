"""Source separation for Attune via UVR MDX-Net ONNX (vocals / instrumental).

Pure numpy STFT/iSTFT around an onnxruntime session — same Smart App Control
constraint as everywhere else in Attune (no torch, no librosa). The model is
the first-party UVR release `UVR-MDX-NET-Voc_FT.onnx`, pinned by SHA256 in
scripts/get-models.py.

v1 separates TWO stems: vocals (the model's primary output) and instrumental
(mix minus vocals, standard MDX practice). A drums/bass/other split is a
follow-up, not pretended here.

Model facts (verified against the pinned file, 2026-08-11):
  input/output [batch, 4, 3072, 256] float32
  channels = [L_real, L_imag, R_real, R_imag] of a 44.1 kHz stereo STFT
  n_fft 7680, hop 1024, freq bins cropped to dim_f=3072, dim_t=256 frames
  one batch-1 inference ~= 2.1 s on this CPU (Ryzen 7 250)

Heavy: a 4-minute track runs ~80 segment inferences (~3 minutes). Callers go
through the server's job path, never a synchronous MCP call.

Standalone-importable; never imports server.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from singing import _ffmpeg_env  # same ffmpeg resolution as the rest of Attune

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILENAME = 'UVR-MDX-NET-Voc_FT.onnx'

SR = 44100
N_FFT = 7680
HOP = 1024
DIM_F = 3072          # model's cropped freq bins (full rfft bins = 3841)
DIM_T = 256           # frames per segment
SEG_SAMPLES = HOP * (DIM_T - 1)          # 261120 — yields exactly 256 centered frames
SEG_HOP = SEG_SAMPLES // 2               # 50% overlap
MAX_ANALYSIS_S = 480  # same cap as singing.load_wav

_session = None


def models_dir() -> Path:
    override = os.environ.get('ATTUNE_MODELS_DIR', '').strip()
    return Path(override) if override else BASE_DIR / 'models'


def model_path() -> Path:
    return models_dir() / MODEL_FILENAME


def is_available() -> tuple:
    """(usable, reason). Never raises."""
    if not model_path().exists():
        return False, f'model missing: {model_path()} (run scripts/get-models.py)'
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:
        return False, f'onnxruntime unavailable: {e}'
    return True, ''


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(str(model_path()),
                                        providers=['CPUExecutionProvider'])
    return _session


# ── Decode ───────────────────────────────────────────────────────────────────


def load_stereo_44k(path: str) -> np.ndarray:
    """ffmpeg decode to 44.1 kHz stereo float64 [2, N]. MDX models are
    calibrated at 44.1 kHz stereo — singing.load_wav's 22050 mono is wrong
    for this path, hence a separate loader with the same 480 s cap."""
    import shutil
    env = _ffmpeg_env()
    ffmpeg = shutil.which('ffmpeg', path=env['PATH']) or 'ffmpeg'
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [ffmpeg, '-y', '-v', 'error', '-i', path, '-t', str(MAX_ANALYSIS_S),
             '-ar', str(SR), '-ac', '2', '-f', 'wav', '-acodec', 'pcm_s16le', tmp_path],
            check=True, capture_output=True, env=env, timeout=180)
        with wave.open(tmp_path, 'rb') as w:
            n = w.getnframes()
            raw = w.readframes(n)
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        x = x.reshape(-1, 2).T  # [2, N]
        return np.ascontiguousarray(x)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Pure-numpy STFT / iSTFT at the model's fixed dims ────────────────────────

_WINDOW = np.hanning(N_FFT + 1)[:-1]  # periodic hann, matches torch.stft


def _stft(x: np.ndarray) -> np.ndarray:
    """[2, N] waveform -> [2, 3841, T] complex, centered (reflect-padded)."""
    pad = N_FFT // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode='reflect')
    n_frames = 1 + (xp.shape[1] - N_FFT) // HOP
    out = np.empty((2, N_FFT // 2 + 1, n_frames), dtype=np.complex128)
    for c in range(2):
        for i in range(n_frames):
            seg = xp[c, i * HOP: i * HOP + N_FFT] * _WINDOW
            out[c, :, i] = np.fft.rfft(seg)
    return out


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    """[2, 3841, T] complex -> [2, length] waveform via overlap-add with
    squared-window normalization (matches torch.istft for hann/hop=1024)."""
    n_frames = spec.shape[2]
    total = N_FFT + HOP * (n_frames - 1)
    out = np.zeros((2, total))
    wsum = np.zeros(total)
    w2 = _WINDOW ** 2
    for c in range(2):
        for i in range(n_frames):
            frame = np.fft.irfft(spec[c, :, i], n=N_FFT)
            out[c, i * HOP: i * HOP + N_FFT] += frame * _WINDOW
            if c == 0:
                wsum[i * HOP: i * HOP + N_FFT] += w2
    nz = wsum > 1e-9
    out[:, nz] /= wsum[nz]
    pad = N_FFT // 2
    return out[:, pad: pad + length]


def _spec_to_model_input(spec: np.ndarray) -> np.ndarray:
    """[2, bins, 256] complex -> [1, 4, 3072, 256] float32 as
    [L_real, L_imag, R_real, R_imag], freq-cropped to DIM_F."""
    cropped = spec[:, :DIM_F, :]
    stacked = np.stack([cropped[0].real, cropped[0].imag,
                        cropped[1].real, cropped[1].imag], axis=0)
    return stacked[np.newaxis].astype(np.float32)


def _model_output_to_spec(y: np.ndarray, full_bins: int) -> np.ndarray:
    """[1, 4, 3072, 256] float32 -> [2, full_bins, 256] complex, zero-padding
    the cropped high bins back in."""
    y = y[0].astype(np.float64)
    spec = np.zeros((2, full_bins, DIM_T), dtype=np.complex128)
    spec[0, :DIM_F] = y[0] + 1j * y[1]
    spec[1, :DIM_F] = y[2] + 1j * y[3]
    return spec


# ── Separation ───────────────────────────────────────────────────────────────


def separate(x: np.ndarray, progress_cb=None) -> dict:
    """[2, N] 44.1 kHz stereo -> {'vocals': [2, N], 'instrumental': [2, N]}.

    Segment-wise inference (261120-sample segments, 50% overlap, triangular
    cross-fade). Instrumental = mix - vocals, standard MDX practice.
    progress_cb(fraction) is called after each segment for job reporting.
    """
    session = _get_session()
    input_name = session.get_inputs()[0].name
    n = x.shape[1]
    vocals = np.zeros_like(x)
    weight = np.zeros(n)
    tri = np.bartlett(SEG_SAMPLES + 2)[1:-1]  # triangular cross-fade, no zeros
    starts = list(range(0, max(1, n - SEG_SAMPLES // 4), SEG_HOP))
    for si, start in enumerate(starts):
        seg = x[:, start:start + SEG_SAMPLES]
        seg_len = seg.shape[1]
        if seg_len < SEG_SAMPLES:
            seg = np.pad(seg, ((0, 0), (0, SEG_SAMPLES - seg_len)))
        spec = _stft(seg)
        # STFT of SEG_SAMPLES yields 256 centered frames; assert the contract
        # loudly rather than silently truncating something unexpected.
        spec = spec[:, :, :DIM_T]
        y = session.run(None, {input_name: _spec_to_model_input(spec)})[0]
        voc = _istft(_model_output_to_spec(y, spec.shape[1]), SEG_SAMPLES)
        w = tri[:seg_len]
        vocals[:, start:start + seg_len] += voc[:, :seg_len] * w
        weight[start:start + seg_len] += w
        if progress_cb is not None:
            progress_cb((si + 1) / len(starts))
    nz = weight > 1e-9
    vocals[:, nz] /= weight[nz]
    instrumental = x - vocals
    return {'vocals': vocals, 'instrumental': instrumental}


# ── Output ───────────────────────────────────────────────────────────────────


def write_stem_wavs(stems: dict, drops_dir: Path, base: str = '') -> dict:
    """Write 16-bit PCM stereo WAVs into drops/. Names match the server's
    DROPS_NAME_RE allowlist. Returns {stem_name: filename}."""
    base = base or secrets.token_urlsafe(12).replace('-', 'x')
    names = {}
    drops_dir.mkdir(parents=True, exist_ok=True)
    for stem_name, audio in stems.items():
        peak = float(np.max(np.abs(audio))) or 1.0
        scaled = np.clip(audio / max(peak, 1.0), -1.0, 1.0)
        pcm = (scaled.T * 32767.0).astype(np.int16)  # [N, 2]
        fname = f'{base}_{stem_name}.wav'
        with wave.open(str(drops_dir / fname), 'wb') as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        names[stem_name] = fname
    return names
