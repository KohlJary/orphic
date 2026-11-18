"""Speech to singing, the Vocaloid way, from a speech voice (Piper).

The voice speaks each word (deterministically: Piper's noise off). WORLD splits the speech
into pitch, spectral envelope (the formants: the words and the voice) and aperiodicity
(breath). Each syllable's frames are laid on its note: the onset consonant at its own
speed, the rest stretched to fill the note. The pitch is replaced with the melody (a
little vibrato on held notes) and the voice is resynthesized: the same voice, singing.
"""
from __future__ import annotations

from typing import Sequence

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf

from .voice import Note, syllables

FP = 5.0  # WORLD frame period, ms


_VOICES = {}


def piper(text: str, model: str) -> tuple:
    """Piper speaking `text`, deterministically (its noise turned off)."""
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    v = _VOICES.get(model) or _VOICES.setdefault(model, PiperVoice.load(model))
    cfg = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0, length_scale=1.0)
    chunks = list(v.synthesize(text, syn_config=cfg))
    y = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float64)
    sr = chunks[0].sample_rate
    y, _ = librosa.effects.trim(y, top_db=35)
    return y, sr


def sing(text: str, notes: Sequence[Note], model: str, vibrato: float = 0.006, formant_shift: float = 1.0,
         speech_shift: float = 0.0):
    notes = list(notes)
    total = int((max(n.start + n.dur for n in notes) + 0.5) * 1000 / FP)
    F0 = np.zeros(total)
    SP = AP = None
    sr = None
    i = 0
    for w in text.replace(",", " ").replace("?", " ").split():
        k = max(1, len(syllables(w)))
        span = notes[i:i + k]
        i += k
        if not span:
            break
        y, sr = piper(w, model)
        f0, t = pw.harvest(y, sr, frame_period=FP)
        sp = pw.cheaptrick(y, f0, t, sr)
        ap = pw.d4c(y, f0, t, sr)
        if formant_shift != 1.0:   # move the formants (a different-sized throat), keep the words
            bins = np.arange(sp.shape[1])
            src = np.clip(bins / formant_shift, 0, sp.shape[1] - 1)
            sp = np.array([np.interp(src, bins, row) for row in sp])
        if SP is None:
            SP = np.full((total, sp.shape[1]), 1e-12)
            AP = np.ones((total, ap.shape[1]))
        bounds = np.linspace(0, len(f0), k + 1).astype(int)
        for j, nt in enumerate(span):
            a, b = bounds[j], bounds[j + 1]
            d0 = int(nt.start * 1000 / FP)
            d1 = min(total, int((nt.start + nt.dur * 0.95) * 1000 / FP))
            keep = min(10, max(0, (b - a) // 3))
            n_dst = max(1, d1 - d0)
            src_idx = np.concatenate([np.arange(a, a + keep),
                                      np.linspace(a + keep, max(a + keep, b - 1), max(1, n_dst - keep))]).astype(int)[:n_dst]
            dst = np.arange(d0, d0 + len(src_idx))
            ok = dst < total
            dst, src_idx = dst[ok], src_idx[ok]
            SP[dst], AP[dst] = sp[src_idx], ap[src_idx]
            if nt.midi is None or nt.midi < 0:
                # rapped (spoken) syllable: keep the voice's own intonation, moved by `speech_shift` semitones
                F0[dst] = f0[src_idx] * 2 ** (speech_shift / 12)
                continue
            hz = 440.0 * 2 ** ((nt.midi - 69) / 12)
            ramp = np.clip((np.arange(len(dst)) - 40) / 40, 0, 1)
            vib = 1 + vibrato * ramp * np.sin(2 * np.pi * 5.5 * np.arange(len(dst)) * FP / 1000)
            F0[dst] = np.where(f0[src_idx] > 0, hz * vib, 0)
    out = pw.synthesize(F0, SP, AP, sr, frame_period=FP)
    return out / (np.abs(out).max() + 1e-12) * 0.8, sr
