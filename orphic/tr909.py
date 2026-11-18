"""A TR-909, modelled: each voice built the way its circuit makes sound, with the panel's knobs.

Deterministic DSP, no samples. The analog voices follow their circuits' structure:
- bass drum: a triangle-ish oscillator whose pitch falls fast from a high start (the
  pitch envelope), through a soft waveshaper; a click and a short noise burst for the
  attack. Knobs: tune (how far the pitch falls / its tail pitch), attack (click), decay.
- snare: two tuned oscillators (the "body", about 180 and 330 Hz) plus noise through a
  filter; tone balances the oscillators against the noise's brightness, snappy sets the
  noise level. Knobs: tune, tone, snappy.
- toms (low, mid, high): an oscillator with a small pitch drop and a little noise.
  Knobs: tune, decay.
- rimshot: two short, ringing band-passed bursts. clap: three or four noise bursts a
  few ms apart, then a reverberant noise tail, band-passed.
The hats, crash and ride were 6-bit samples in the real machine; here they're an
approximation (six detuned square oscillators through band-pass filters, as metal
cymbals are often modelled), named honestly as such. Knobs: decay (and tune for cymbals).
All knobs run 0 to 1, 0.5 being the panel's middle.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

SR = 44100
VOICES = ("kick", "snare", "low_tom", "mid_tom", "high_tom", "rim", "clap", "hat", "open", "crash", "ride")
KNOBS = {"kick": ("tune", "attack", "decay"), "snare": ("tune", "tone", "snappy"),
         "low_tom": ("tune", "decay"), "mid_tom": ("tune", "decay"), "high_tom": ("tune", "decay"),
         "rim": (), "clap": (), "hat": ("decay",), "open": ("decay",), "crash": ("tune",), "ride": ("tune",)}


def _rng(seed: int):
    return np.random.default_rng(seed)


def _bp(x, lo, hi):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, [lo / (SR / 2), min(hi, SR / 2 - 100) / (SR / 2)], btype="band", output="sos"), x)


def _hp(x, f):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, f / (SR / 2), btype="high", output="sos"), x)


def _lp(x, f):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, min(f, SR / 2 - 100) / (SR / 2), btype="low", output="sos"), x)


def _osc(freq_curve):
    ph = 2 * np.pi * np.cumsum(freq_curve) / SR
    return ph


def kick(tune=0.5, attack=0.5, decay=0.5, length=1.2):
    t = np.arange(int(length * SR)) / SR
    base = 45 + 20 * tune                      # the tail's pitch
    f = base + (220 + 150 * tune) * np.exp(-t / 0.012) + 60 * np.exp(-t / 0.05)
    ph = _osc(f)
    tri = 2 / np.pi * np.arcsin(np.sin(ph))    # triangle-ish core
    body = np.tanh(2.2 * tri) / np.tanh(2.2)   # the waveshaper rounds it toward a sine with a little edge
    env = np.exp(-t / (0.15 + 0.75 * decay))
    click = (np.exp(-t / 0.0015) * _rng(9).standard_normal(len(t)) * 0.6 * attack
             + np.exp(-t / 0.0006) * 1.2 * attack)
    noise = _lp(_rng(1).standard_normal(len(t)) * np.exp(-t / 0.008) * 0.25, 3000)
    return body * env + _hp(click, 1500) + noise


def snare(tune=0.5, tone=0.5, snappy=0.5, length=0.6):
    t = np.arange(int(length * SR)) / SR
    k = 1 + 0.25 * (tune - 0.5)
    o1 = np.sin(_osc((180 * k) + 60 * np.exp(-t / 0.01))) * np.exp(-t / 0.09)
    o2 = np.sin(_osc((330 * k) + 80 * np.exp(-t / 0.01))) * np.exp(-t / 0.06)
    body = (0.7 * o1 + 0.45 * o2) * (1.2 - 0.6 * tone)
    nz = _rng(2).standard_normal(len(t))
    nz = _hp(_lp(nz, 6000 + 6000 * tone), 1200) * np.exp(-t / (0.08 + 0.12 * snappy))
    return body + nz * (0.3 + 1.1 * snappy)


def tom(which="mid_tom", tune=0.5, decay=0.5, length=1.0):
    t = np.arange(int(length * SR)) / SR
    base = {"low_tom": 95, "mid_tom": 135, "high_tom": 185}[which] * (1 + 0.35 * (tune - 0.5))
    f = base + base * 0.45 * np.exp(-t / 0.03)
    body = np.sin(_osc(f)) * np.exp(-t / (0.12 + 0.45 * decay))
    nz = _lp(_rng(3).standard_normal(len(t)), 4000) * np.exp(-t / 0.02) * 0.15
    return body + nz


def rim(length=0.15):
    t = np.arange(int(length * SR)) / SR
    x = _rng(4).standard_normal(len(t)) * np.exp(-t / 0.002)
    return 2.5 * (_bp(x, 1600, 1800) + 0.6 * _bp(x, 450, 520)) * np.exp(-t / 0.03)


def clap(length=0.6):
    t = np.arange(int(length * SR)) / SR
    nz = _rng(5).standard_normal(len(t))
    env = np.zeros(len(t))
    for d in (0.0, 0.011, 0.022, 0.033):       # the hand claps, a few ms apart
        i = int(d * SR)
        env[i:] += np.exp(-(t[i:] - d) / 0.004)
    tail = np.exp(-np.maximum(t - 0.033, 0) / 0.14) * (t >= 0.033) * 0.45
    return 1.6 * _bp(nz * (env + tail), 900, 2600)


def _metal(length, decay_s, lo, hi, tune=0.5, seed=6):
    t = np.arange(int(length * SR)) / SR
    freqs = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * (1 + 0.3 * (tune - 0.5))
    x = sum(np.sign(np.sin(2 * np.pi * f * t + p)) for f, p in zip(freqs, _rng(seed).uniform(0, 6, 6)))
    x = x + 0.4 * _rng(seed + 1).standard_normal(len(t))
    return _hp(_bp(x, lo, hi), lo) * np.exp(-t / decay_s) * 0.35


def hat(decay=0.5):
    return _metal(0.4, 0.02 + 0.06 * decay, 7000, 16000)


def open_hat(decay=0.5):
    return _metal(1.2, 0.12 + 0.5 * decay, 6500, 16000, seed=7)


def crash(tune=0.5):
    return _metal(2.5, 0.9, 4000, 16000, tune, seed=8) * 1.2


def ride(tune=0.5):
    return _metal(2.5, 1.1, 3000, 12000, tune, seed=9) * 0.8 + _metal(2.5, 1.5, 6000, 9000, tune, seed=10) * 0.3


def voice(name: str, knobs: Dict[str, float]) -> np.ndarray:
    k = {kk: float(v) for kk, v in (knobs or {}).items()}
    if name == "kick":
        y = kick(k.get("tune", .5), k.get("attack", .5), k.get("decay", .5))
    elif name == "snare":
        y = snare(k.get("tune", .5), k.get("tone", .5), k.get("snappy", .5))
    elif name in ("low_tom", "mid_tom", "high_tom"):
        y = tom(name, k.get("tune", .5), k.get("decay", .5))
    elif name == "rim":
        y = rim()
    elif name == "clap":
        y = clap()
    elif name == "hat":
        y = hat(k.get("decay", .5))
    elif name == "open":
        y = open_hat(k.get("decay", .5))
    elif name == "crash":
        y = crash(k.get("tune", .5))
    elif name == "ride":
        y = ride(k.get("tune", .5))
    else:
        raise ValueError(name)
    return y / (np.abs(y).max() + 1e-12) * 0.9
