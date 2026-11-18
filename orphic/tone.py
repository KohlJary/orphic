"""Tone: what a guitar rig does to the sound, measured, so tone can be obsessed over.

Every rig hears the same probe, so two tries compare fairly:
  0.0-2.0 s  a held single note (E3)          -> harmonics: odd vs even, how fast they fall
  2.2-4.6 s  a held power chord (E2 B2 E3)    -> how clean a power chord stays
  4.8-7.2 s  a held full chord (E2 B2 E3 G#3 B3) -> intermodulation: whether it turns to mud
  7.4-9.0 s  low chugs (E2 sixteenths)        -> tightness: how fast the low end stops
  9.0-10.0 s silence                          -> the noise floor
Measures (all deterministic): band balance (sub, low, mud, scoop, mid, presence, fizz,
air, in dB against the whole), odd/even harmonics and their slope, intermodulation (energy
off the harmonic series in a chord), sustain (how long the held note stays within 20 dB),
compression (crest factor), tightness (low-end decay after a chug), noise floor, width;
then a few plain words from those numbers. No model.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .song import SR, Event, Track, apply_fx, render_gm, render_synth, resolve_chain

TEMPO = 140.0
PROBE = {
    "note": (0.0, 2.0, [52]),
    "power": (2.2, 4.6, [40, 47, 52]),
    "chord": (4.8, 7.2, [40, 47, 52, 56, 59]),
}
CHUGS = (7.4, 9.0, 40)
TOTAL = 10.0
BANDS = {"sub": (20, 80), "low": (80, 200), "mud": (200, 500), "scoop": (500, 800), "mid": (800, 2000),
         "presence": (2000, 4500), "fizz": (4500, 10000), "air": (10000, 18000)}


def probe_events() -> List[Event]:
    ev = []
    for _, (a, b, notes) in PROBE.items():
        ev += [Event(a, b - a, n, 100) for n in notes]
    a, b, n = CHUGS
    step = 60.0 / TEMPO / 4
    t = a
    while t < b - step / 2:
        ev.append(Event(t, step * 0.6, n, 110))
        t += step
    return ev


def render_probe(instrument: Dict, chain=None) -> np.ndarray:
    tr = Track("guitar", dict(instrument), events=probe_events())
    if "gm" in instrument:
        a = render_gm(tr, TOTAL)
    else:
        a = render_synth(tr, TOTAL)
    return apply_fx(a, chain if chain is not None else resolve_chain(instrument))


def _seg(y, a, b):
    return y[int(a * SR): int(b * SR)]


def _spec(y):
    w = np.hanning(len(y))
    S = np.abs(np.fft.rfft(y * w)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return f, S


def _db(x):
    return float(10 * np.log10(x + 1e-20))


def _harmonics(y, f0, n=15):
    f, S = _spec(y)
    out = []
    for h in range(1, n + 1):
        band = (f > h * f0 * 0.97) & (f < h * f0 * 1.03)
        out.append(S[band].sum() if band.any() else 0.0)
    return np.array(out), f, S


def measure_tone(a: np.ndarray) -> Dict[str, Optional[float]]:
    y = a.mean(axis=0)
    peak = float(np.abs(y).max()) + 1e-12
    out: Dict[str, Optional[float]] = {}
    body = y[: int(CHUGS[1] * SR)]
    f, S = _spec(body)
    tot = S[(f >= 20) & (f < 18000)].sum() + 1e-20
    for k, (lo, hi) in BANDS.items():
        out[f"band_{k}_db"] = round(_db(S[(f >= lo) & (f < hi)].sum() / tot), 1)
    # harmonics of the held E3
    f0 = 440.0 * 2 ** ((52 - 69) / 12)
    note = _seg(y, 0.15, 1.9)
    H, _, _ = _harmonics(note, f0)
    odd, even = H[2::2].sum(), H[1::2].sum()   # 3rd, 5th, ... vs 2nd, 4th, ...
    out["odd_even_db"] = round(_db(odd / (even + 1e-20)), 1)
    hd = 10 * np.log10(H + 1e-20) - 10 * np.log10(H[0] + 1e-20)
    out["harmonic_slope_db"] = round(float(np.polyfit(np.arange(1, 16), hd, 1)[0]), 2)
    out["harmonics_above_minus40"] = int(np.sum(hd[1:] > -40))
    # intermodulation: energy off the harmonic series of the chord's notes
    def offseries(seg, midis):
        fs, Ss = _spec(seg)
        on = np.zeros_like(fs, dtype=bool)
        for m in midis:
            fm = 440.0 * 2 ** ((m - 69) / 12)
            for h in range(1, 60):
                on |= (fs > h * fm * 0.985) & (fs < h * fm * 1.015)
        sel = (fs > 60) & (fs < 6000)
        return float(Ss[sel & ~on].sum() / (Ss[sel].sum() + 1e-20))
    out["imd_power_pct"] = round(100 * offseries(_seg(y, 2.35, 4.4), PROBE["power"][2]), 1)
    out["imd_chord_pct"] = round(100 * offseries(_seg(y, 4.95, 7.0), PROBE["chord"][2]), 1)
    # sustain: how long the held note stays within 20 dB of its peak
    # from when it first reaches half its peak to the last moment within 20 dB of it
    env = np.sqrt(np.convolve(_seg(y, 0, 2.0) ** 2, np.ones(441) / 441, mode="same"))
    top = env.max() + 1e-12
    up = np.where(env >= 0.5 * top)[0]
    held = np.where(env >= 0.1 * top)[0]
    out["sustain_s"] = round(float((held[-1] - up[0]) / SR), 2) if len(up) and len(held) else 0.0
    riff = y[: int(CHUGS[1] * SR)]
    out["crest_db"] = round(20 * np.log10(np.abs(riff).max() / (np.sqrt(np.mean(riff ** 2)) + 1e-12) + 1e-12), 1)
    # tightness: after each chug, how fast the low end (< 200 Hz) falls 20 dB
    from scipy.signal import butter, sosfilt
    low = sosfilt(butter(4, 200 / (SR / 2), output="sos"), _seg(y, CHUGS[0], CHUGS[1]))
    le = np.sqrt(np.convolve(low ** 2, np.ones(88) / 88, mode="same"))
    step = int(60.0 / TEMPO / 4 * SR)
    decays = []
    for i in range(0, len(le) - step, step):
        w = le[i:i + step]
        p = int(np.argmax(w))
        d = np.where(w[p:] < w[p] * 0.1)[0]
        decays.append((d[0] if len(d) else len(w) - p) / SR * 1000)
    out["chug_decay_ms"] = round(float(np.median(decays)), 1) if decays else None
    tail = _seg(y, 9.3, 9.9)
    out["noise_floor_db"] = round(20 * np.log10(np.sqrt(np.mean(tail ** 2)) / peak + 1e-12), 1)
    m, s = (a[0] + a[1]) / 2, (a[0] - a[1]) / 2
    out["width"] = round(float((s ** 2).sum() / ((m ** 2).sum() + (s ** 2).sum() + 1e-20)), 3)
    return out


def words(t: Dict) -> List[str]:
    w = []
    # distortion shows as intermodulation (energy off the harmonic series); a plucked
    # string alone is rich in harmonics but has almost none
    imd = max(t["imd_power_pct"], t["imd_chord_pct"])
    if imd < 1.0:
        w.append("clean")
    elif imd < 5.0 and t["crest_db"] > 14:
        w.append("lightly driven")
    elif t["crest_db"] < 6:
        w.append("crushed, near square-wave saturation")
    else:
        w.append("driven, saturated")
    w.append("odd-heavy (harder, more square, fuzz-like)" if t["odd_even_db"] > 4 else
             "even-rich (warmer, tube-like)" if t["odd_even_db"] < -2 else "balanced odd/even")
    if t["band_fizz_db"] > -12:
        w.append("fizzy on top")
    elif t["band_fizz_db"] < -24:
        w.append("dark, little top end")
    if t["band_scoop_db"] < t["band_mud_db"] - 6 and t["band_scoop_db"] < t["band_presence_db"] - 6:
        w.append("scooped mids")
    if t["band_mud_db"] > -2.5:
        w.append("muddy low mids")
    if t["imd_chord_pct"] > 2 * max(t["imd_power_pct"], 1.0) and t["imd_chord_pct"] > 15:
        w.append("full chords turn to mush; power chords hold")
    if t["chug_decay_ms"] is not None:
        w.append("tight low end" if t["chug_decay_ms"] < 60 else "loose, blooming low end" if t["chug_decay_ms"] > 110 else "")
    if t["crest_db"] < 10:
        w.append("compressed, little dynamic range")
    if t["noise_floor_db"] > -55:
        w.append("audible hiss/hum between notes")
    if t["width"] > 0.1:
        w.append("wide")
    return [x for x in w if x]


LABELS = {
    "band_sub_db": "sub (20-80 Hz)", "band_low_db": "low (80-200)", "band_mud_db": "mud (200-500)",
    "band_scoop_db": "scoop zone (500-800)", "band_mid_db": "mids (800-2k)", "band_presence_db": "presence (2-4.5k)",
    "band_fizz_db": "fizz (4.5-10k)", "band_air_db": "air (10k+)", "odd_even_db": "odd vs even harmonics (dB)",
    "harmonic_slope_db": "harmonic fall-off (dB per harmonic)", "harmonics_above_minus40": "harmonics within 40 dB",
    "imd_power_pct": "intermod on a power chord (%)", "imd_chord_pct": "intermod on a full chord (%)",
    "sustain_s": "sustain (s)", "crest_db": "crest (dB, lower = more compressed)",
    "chug_decay_ms": "chug low-end decay (ms)", "noise_floor_db": "noise floor (dB)", "width": "stereo width",
}
CLEAR = {"odd_even_db": 2.0, "harmonic_slope_db": 0.3, "harmonics_above_minus40": 2, "imd_power_pct": 3.0,
         "imd_chord_pct": 3.0, "sustain_s": 0.15, "crest_db": 1.5, "chug_decay_ms": 15.0, "noise_floor_db": 4.0,
         "width": 0.05}


def differences(before: Dict, after: Dict) -> List[str]:
    out = []
    for k, lab in LABELS.items():
        a, b = before.get(k), after.get(k)
        if a is None or b is None:
            continue
        clear = CLEAR.get(k, 2.0)   # bands: 2 dB
        if abs(b - a) >= clear:
            out.append(f"{lab}: {a} -> {b}")
    return out or ["no clear change in the tone"]


def tone(instrument: Dict, chain=None) -> Dict:
    a = render_probe(instrument, chain)
    t = measure_tone(a)
    return {"tone": t, "in_words": words(t), "labels": LABELS}, a
