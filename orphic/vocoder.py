"""Her voice, the vocoder way (how Daft Punk's robots get words across).

A vocoder borrows articulation from real speech and puts it on any sound: the speech's
spectrum is measured band by band over time, and the carrier (a synth playing the notes)
is shaped by it. The speech here is eSpeak NG's, a speech synthesizer built from rules and
formants, with no one's recorded voice in it. Its own buzzy timbre is thrown away; only
how its spectrum moves is kept. Deterministic: eSpeak, the stretches and the filters are
all fixed, and the noise is seeded.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from .voice import SR, Note, syllables

BANDS = 24


def speak(word: str, voice: str = "en-us", speed: int = 150) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "w.wav"
        subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), "-w", str(p), word], check=True,
                       capture_output=True)
        y, sr = sf.read(p)
    y = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=SR)
    y, _ = librosa.effects.trim(y, top_db=35)
    return y


def modulator(text: str, notes: Sequence[Note], voice: str = "en-us") -> np.ndarray:
    """eSpeak's words laid on the notes: each word stretched over the notes its syllables take."""
    words = [w for w in text.replace(",", " ").replace("?", " ").replace("!", " ").split() if w.strip()]
    total = max(n.start + n.dur for n in notes) + 0.5
    out = np.zeros(int(total * SR))
    i = 0
    for w in words:
        k = max(1, len(syllables(w)))
        span = notes[i:i + k]
        if not span:
            break
        i += k
        t0, t1 = span[0].start, span[-1].start + span[-1].dur
        y = speak(w, voice)
        if len(y) < 64:
            continue
        want = max(0.05, (t1 - t0) * 0.97)
        rate = (len(y) / SR) / want
        y = librosa.effects.time_stretch(y, rate=rate) if abs(rate - 1) > 0.02 else y
        a = int(t0 * SR)
        b = min(len(out), a + len(y))
        out[a:b] += y[: b - a]
    return out


def carrier(notes: Sequence[Note], n: int, kind: str = "saw", noise: float = 0.12, seed: int = 7) -> np.ndarray:
    f0 = np.zeros(n)
    for nt in notes:
        a, b = int(nt.start * SR), min(n, int((nt.start + nt.dur) * SR))
        f0[a:b] = 440.0 * 2 ** ((nt.midi - 69) / 12)
    idx = np.where(f0 > 0)[0]
    if len(idx):
        f0 = np.interp(np.arange(n), idx, f0[idx])
    ph = np.cumsum(f0) / SR
    if kind == "square":
        tone = np.sign(np.sin(2 * np.pi * ph))
    elif kind == "fm":
        tone = np.sin(2 * np.pi * ph + 2.5 * np.sin(2 * np.pi * ph))
    else:
        tone = 2 * (ph - np.floor(ph + 0.5))
    return tone + noise * np.random.default_rng(seed).standard_normal(n)


def vocode(mod: np.ndarray, car: np.ndarray, bands: int = BANDS, lo: float = 90.0, hi: float = 9000.0,
           env_hz: float = 40.0) -> np.ndarray:
    n = min(len(mod), len(car))
    mod, car = mod[:n], car[:n]
    edges = np.geomspace(lo, hi, bands + 1)
    env_sos = butter(2, env_hz / (SR / 2), output="sos")
    out = np.zeros(n)
    for a, b in zip(edges[:-1], edges[1:]):
        sos = butter(2, [a / (SR / 2), b / (SR / 2)], btype="band", output="sos")
        m = sosfilt(sos, mod)
        env = sosfilt(env_sos, np.abs(m))
        out += sosfilt(sos, car) * env
    peak = np.abs(out).max()
    return out / peak * 0.8 if peak > 0 else out


def sing(text: str, notes: Sequence[Note], kind: str = "saw", voice: str = "en-us", carrier_audio=None):
    mod = modulator(text, notes, voice)
    car = carrier_audio if carrier_audio is not None else carrier(notes, len(mod), kind)
    return vocode(mod, car), mod


def clarity(mod: np.ndarray, sung: np.ndarray) -> float:
    from pystoi import stoi
    m = min(len(mod), len(sung))
    return float(stoi(mod[:m], sung[:m], SR, extended=False))
