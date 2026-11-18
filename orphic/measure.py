"""What a render sounds like, in a sound designer's terms, measured."""
from __future__ import annotations

from typing import Dict, Optional

import librosa
import numpy as np

from .host import SR

LABELS = {
    "level_db": "level (dB)", "f0_hz": "pitch (Hz)", "pitch_wobble_cents": "pitch wobble (cents)",
    "brightness_hz": "brightness (Hz)", "noise": "noise vs tone (0 tone, 1 noise)",
    "harmonicity": "share of energy on the harmonic series", "odd_even": "odd vs even harmonics (dB)",
    "attack_ms": "attack (ms)", "sustain": "sustain level (share of peak)", "release_ms": "release (ms)",
    "brightness_motion": "brightness change over the note (octaves)", "am_rate_hz": "tremolo/beating rate (Hz)",
    "am_depth": "tremolo/beating depth", "width": "stereo width (0 mono)", "peak_hz": "resonant peak (Hz)",
    "peak_db": "resonant peak height (dB)",
}
SCALE = {  # what counts as a clear change, per measure
    "level_db": 3.0, "f0_hz": None, "pitch_wobble_cents": 15.0, "brightness_hz": None, "noise": 0.05,
    "harmonicity": 0.08, "odd_even": 3.0, "attack_ms": None, "sustain": 0.1, "release_ms": None,
    "brightness_motion": 0.3, "am_rate_hz": 1.0, "am_depth": 0.1, "width": 0.08, "peak_hz": None, "peak_db": 3.0,
}


def _env(y, hop=128):
    return librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0], hop / SR


def measure(a: np.ndarray, hold: float) -> Dict[str, Optional[float]]:
    y = a.mean(axis=0)
    out: Dict[str, Optional[float]] = {k: None for k in LABELS}
    peak = float(np.abs(y).max())
    if peak < 1e-5:
        out["level_db"] = -120.0
        return out
    if 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12) < -60:   # too quiet to say anything else
        out["level_db"] = round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)), 2)
        return out
    env, dt = _env(y)
    t_peak = int(np.argmax(env))
    out["level_db"] = round(float(20 * np.log10(np.sqrt(np.mean(y[: int(hold * SR)] ** 2)) + 1e-12)), 2)
    above = np.where(env >= 0.1 * env[t_peak])[0]
    t10 = above[0] if len(above) else 0
    t90 = t10 + int(np.argmax(env[t10:t_peak + 1] >= 0.9 * env[t_peak])) if t_peak >= t10 else t10
    out["attack_ms"] = round(float((t90 - t10) * dt * 1000), 1)
    i_hold = min(len(env) - 1, int(hold * 0.9 / dt))
    out["sustain"] = round(float(env[i_hold] / (env[t_peak] + 1e-12)), 3)
    i_off = int(hold / dt)
    if i_off < len(env):
        start = env[i_off]
        gone = np.where(env[i_off:] < 0.01 * max(start, 1e-9))[0]
        out["release_ms"] = round(float((gone[0] if len(gone) else len(env) - i_off) * dt * 1000), 1)
    seg = y[int(0.05 * SR): int(hold * SR)]
    if len(seg) > 4096:
        S = np.abs(librosa.stft(seg, n_fft=4096, hop_length=1024))
        f = librosa.fft_frequencies(sr=SR, n_fft=4096)
        cen = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
        out["brightness_hz"] = round(float(np.median(cen)), 1)
        if len(cen) > 4:
            out["brightness_motion"] = round(float(np.log2((cen[-2] + 1) / (cen[1] + 1))), 3)
        out["noise"] = round(float(np.median(librosa.feature.spectral_flatness(S=S)[0])), 4)
        spec = S.mean(axis=1)
        f0s = librosa.yin(seg, fmin=30, fmax=2000, sr=SR, frame_length=4096, hop_length=2048)
        # keep frames whose period really repeats (normalised autocorrelation at the YIN lag)
        keep = []
        for i, f0 in enumerate(f0s):
            a0 = i * 2048
            fr = seg[a0:a0 + 4096]
            lag = int(round(SR / f0))
            if len(fr) > lag + 256:
                x, z = fr[:-lag], fr[lag:]
                r = float(np.dot(x, z) / (np.sqrt(np.dot(x, x) * np.dot(z, z)) + 1e-12))
                if r > 0.8:
                    keep.append(f0)
        f0s = np.array(keep)
        cen_med = float(np.median(cen)) if len(cen) else 0.0
        if len(f0s) and np.median(f0s) > 4 * max(cen_med, 1.0):
            f0s = np.array([])   # a "pitch" far above where the energy is: noise, not a note
        if len(f0s) >= 3:
            f0 = float(np.median(f0s))
            out["f0_hz"] = round(f0, 2)
            out["pitch_wobble_cents"] = round(float(np.std(1200 * np.log2(f0s / f0))), 1)
            harm = np.zeros_like(spec, dtype=bool)
            odd, even = 0.0, 0.0
            for h in range(1, int(min(40, (SR / 2) // f0))):
                band = (f > h * f0 * 0.97) & (f < h * f0 * 1.03)
                harm |= band
                e = spec[band].sum() ** 2
                if h > 1:
                    if h % 2:
                        odd += e
                    else:
                        even += e
            out["harmonicity"] = round(float((spec[harm] ** 2).sum() / ((spec ** 2).sum() + 1e-12)), 3)
            out["odd_even"] = round(float(10 * np.log10((odd + 1e-12) / (even + 1e-12))), 2)
        # resonant peak: the strongest bump of the smoothed spectrum above its trend
        lg = 20 * np.log10(spec + 1e-9)
        k = 31
        smooth = np.convolve(lg, np.ones(k) / k, mode="same")
        trend = np.convolve(lg, np.ones(301) / 301, mode="same")
        bump = smooth - trend
        sel = (f > 80) & (f < 12000)
        i = int(np.argmax(np.where(sel, bump, -1e9)))
        out["peak_hz"], out["peak_db"] = round(float(f[i]), 1), round(float(bump[i]), 2)
        # amplitude modulation (tremolo, detune beating) over the held part
        e2, dt2 = _env(seg, hop=256)
        e2 = e2 - e2.mean()
        if len(e2) > 32:
            spec_e = np.abs(np.fft.rfft(e2 * np.hanning(len(e2))))
            fr = np.fft.rfftfreq(len(e2), dt2)
            band = (fr > 0.5) & (fr < 25)
            if band.any():
                j = int(np.argmax(np.where(band, spec_e, 0)))
                out["am_rate_hz"] = round(float(fr[j]), 2)
                out["am_depth"] = round(float(2 * spec_e[j] / (np.sum(np.hanning(len(e2))) + 1e-9) /
                                              (np.mean(_env(seg, hop=256)[0]) + 1e-9)), 3)
    m, s = (a[0] + a[1]) / 2, (a[0] - a[1]) / 2
    out["width"] = round(float((s ** 2).sum() / ((m ** 2).sum() + (s ** 2).sum() + 1e-12)), 3)
    return out


def change(a: dict, b: dict) -> Dict[str, float]:
    """How far b is from a, per measure, in units of 'a clear change' (>= 1 is clear)."""
    out = {}
    for k in LABELS:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            if (x is None) != (y is None) and k in ("f0_hz",):
                out[k] = 1.0
            continue
        sc = SCALE[k]
        if sc is None:  # ratio measures: an eighth of an octave (pitch: a quarter semitone)
            if x <= 0 or y <= 0:
                continue
            octs = abs(np.log2(y / x))
            out[k] = octs / ((1 / 48) if k == "f0_hz" else 0.25)
        else:
            out[k] = abs(y - x) / sc
    return out
