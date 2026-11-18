"""A voice that isn't anyone's: words carried by formants, sung through any sound.

Articulation and timbre are kept apart, the way a vocoder or a talkbox does it.
- Articulation: lyrics -> phonemes (CMU pronouncing dictionary, a lookup) -> per-5 ms
  targets: formant positions (moving smoothly between sounds, which is most of what
  makes words understandable), voicing, frication bands, stop bursts, nasal murmur.
  Syllables are laid on the notes they're sung to.
- The throat: what the formants shape. A built-in FM source, a saw, or the audio of any
  synth patch playing the same notes. Inhuman on purpose; no one's recorded voice.
Clarity is measured with STOI against the same words through a plain source.
Deterministic: noise comes from a seeded generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

SR = 44100
FRAME = 0.005
HOP = int(SR * FRAME)

# Vowels (ARPAbet, stress digits stripped): F1, F2, F3 (Hz), adult averages.
VOWELS = {
    "AA": (730, 1090, 2440), "AE": (660, 1720, 2410), "AH": (640, 1190, 2390), "AO": (570, 840, 2410),
    "EH": (530, 1840, 2480), "ER": (490, 1350, 1690), "IH": (390, 1990, 2550), "IY": (270, 2290, 3010),
    "UH": (440, 1020, 2240), "UW": (300, 870, 2240),
}
DIPHTHONGS = {"AW": ("AA", "UH"), "AY": ("AA", "IY"), "EY": ("EH", "IY"), "OW": ("AO", "UH"), "OY": ("AO", "IY")}

# Consonants: kind, formants (where the tract points), voiced, noise band, loudness.
#   kind: stop, fric, nasal, approx, aspirate, affric
CONS: Dict[str, Tuple] = {
    "P": ("stop", (300, 900, 2300), False, (500, 1800), 0.35), "B": ("stop", (300, 900, 2300), True, (500, 1800), 0.3),
    "T": ("stop", (300, 1800, 2700), False, (3000, 7000), 0.5), "D": ("stop", (300, 1800, 2700), True, (3000, 6500), 0.4),
    "K": ("stop", (300, 2200, 2600), False, (1500, 3500), 0.5), "G": ("stop", (300, 2200, 2600), True, (1500, 3200), 0.4),
    "S": ("fric", (400, 1800, 2700), False, (4500, 9000), 0.55), "Z": ("fric", (400, 1800, 2700), True, (4500, 9000), 0.4),
    "SH": ("fric", (400, 1900, 2400), False, (2000, 5500), 0.55), "ZH": ("fric", (400, 1900, 2400), True, (2000, 5500), 0.4),
    "F": ("fric", (400, 1000, 2300), False, (1500, 8000), 0.22), "V": ("fric", (400, 1000, 2300), True, (1500, 8000), 0.18),
    "TH": ("fric", (400, 1400, 2600), False, (1500, 8000), 0.18), "DH": ("fric", (400, 1400, 2600), True, (1500, 8000), 0.15),
    "HH": ("aspirate", None, False, None, 0.3),
    "CH": ("affric", (400, 1900, 2400), False, (2000, 5500), 0.55), "JH": ("affric", (400, 1900, 2400), True, (2000, 5500), 0.4),
    "M": ("nasal", (250, 1100, 2300), True, None, 0.45), "N": ("nasal", (250, 1700, 2500), True, None, 0.45),
    "NG": ("nasal", (250, 2100, 2600), True, None, 0.45),
    "L": ("approx", (360, 1300, 2700), True, None, 0.7), "R": ("approx", (420, 1300, 1600), True, None, 0.7),
    "W": ("approx", (300, 700, 2200), True, None, 0.65), "Y": ("approx", (280, 2250, 3000), True, None, 0.7),
}

_DICT = None


def phonemes(word: str) -> List[str]:
    global _DICT
    if _DICT is None:
        import cmudict
        _DICT = cmudict.dict()
    w = "".join(c for c in word.lower() if c.isalpha() or c == "'")
    got = _DICT.get(w)
    if got:
        return got[0]
    # unknown word: spell it with crude letter rules (better than silence)
    rough = {"a": "AE1", "e": "EH1", "i": "IH1", "o": "AA1", "u": "AH1", "y": "IY1"}
    out = []
    for c in w:
        if c in rough:
            out.append(rough[c])
        elif c.upper() in CONS:
            out.append(c.upper())
        elif c == "c":
            out.append("K")
        elif c == "j":
            out.append("JH")
        elif c == "q":
            out.append("K")
        elif c == "x":
            out += ["K", "S"]
    return out


def is_vowel(p: str) -> bool:
    return p[-1].isdigit()


def syllables(text: str) -> List[List[str]]:
    """Words -> syllables, each with one vowel; consonants between vowels go to the next
    syllable's onset, except one stays as the coda when there are several."""
    out: List[List[str]] = []
    for word in text.split():
        ph = phonemes(word)
        idx = [i for i, p in enumerate(ph) if is_vowel(p)]
        if not idx:
            if out:
                out[-1] += ph
            continue
        cuts = [0]
        for a, b in zip(idx, idx[1:]):
            between = b - a - 1
            cuts.append(a + 1 + (1 if between >= 2 else 0))
        cuts.append(len(ph))
        for s, e in zip(cuts, cuts[1:]):
            out.append(ph[s:e])
    return out


@dataclass
class Note:
    midi: float
    start: float      # seconds
    dur: float        # seconds


def _targets(syls: List[List[str]], notes: List[Note], total: float):
    """Per-frame control tracks from syllables laid on notes."""
    n = int(total / FRAME) + 1
    F = np.zeros((n, 3))
    have = np.zeros(n, bool)
    voice = np.zeros(n)
    noise = np.zeros(n)
    nlo = np.full(n, 3000.0)
    nhi = np.full(n, 6000.0)
    asp = np.zeros(n)
    f0 = np.zeros(n)
    fi = lambda t: int(round(t / FRAME))

    def put(a, b, form=None, v=0.0, nz=0.0, band=None, h=0.0):
        a, b = max(0, fi(a)), min(n, max(fi(a) + 1, fi(b)))
        if form is not None:
            F[a:b] = form
            have[a:b] = True
        voice[a:b] = v
        noise[a:b] = nz
        if band:
            nlo[a:b], nhi[a:b] = band
        asp[a:b] = h

    if len(syls) > len(notes):    # more syllables than notes: share the last note
        extra = len(syls) - len(notes) + 1
        last = notes[-1]
        notes = notes[:-1] + [Note(last.midi, last.start + i * last.dur / extra, last.dur / extra) for i in range(extra)]
    for i, note in enumerate(notes):
        a = fi(note.start)
        b = fi(note.start + note.dur)
        f0[a:b] = 440.0 * 2 ** ((note.midi - 69) / 12)
        if i >= len(syls):         # more notes than syllables: hold the last vowel
            continue
        syl = syls[i]
        v_at = next(k for k, p in enumerate(syl) if is_vowel(p))
        onset, nucleus, coda = syl[:v_at], syl[v_at][:-1], syl[v_at + 1:]
        t = note.start
        for c in onset:
            t = _consonant(put, c, t)
        tc = note.start + note.dur - 0.07 * len(coda)
        t_end_vowel = max(t + 0.06, tc)
        if nucleus in DIPHTHONGS:
            v1, v2 = DIPHTHONGS[nucleus]
            mid = t + 0.55 * (t_end_vowel - t)
            put(t, mid, VOWELS[v1], v=1.0)
            put(mid, t_end_vowel, VOWELS[v2], v=1.0)
        else:
            put(t, t_end_vowel, VOWELS.get(nucleus, VOWELS["AH"]), v=1.0)
        t = t_end_vowel
        for c in coda:
            t = _consonant(put, c, t)
    # formants: fill gaps from neighbours, then glide between targets (~35 ms)
    idx = np.where(have)[0]
    if len(idx):
        for k in range(3):
            F[:, k] = np.interp(np.arange(n), idx, F[idx, k])
        w = 7
        ker = np.ones(w) / w
        for k in range(3):
            F[:, k] = np.convolve(np.pad(F[:, k], (w // 2, w // 2), mode="edge"), ker, mode="valid")
    # f0 gaps (consonants between notes) held, with a short glide
    nz = np.where(f0 > 0)[0]
    if len(nz):
        f0 = np.interp(np.arange(n), nz, f0[nz])
        f0 = np.convolve(np.pad(f0, (4, 4), mode="edge"), np.ones(9) / 9, mode="valid")
    sm = lambda x, w=3: np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")
    return {"F": F, "voice": sm(voice), "noise": sm(noise), "nlo": nlo, "nhi": nhi, "asp": sm(asp), "f0": f0}


def _consonant(put, c: str, t: float) -> float:
    kind, form, voiced, band, loud = CONS.get(c, ("approx", None, True, None, 0.5))
    vb = 0.25 if voiced else 0.0
    if kind == "stop":
        put(t, t + 0.045, form, v=vb)                               # closure (a voice bar if voiced)
        put(t + 0.045, t + 0.06, form, v=vb, nz=loud, band=band)    # burst
        if not voiced:
            put(t + 0.06, t + 0.095, form, h=0.5 * loud)            # aspiration
            return t + 0.095
        return t + 0.065
    if kind == "affric":
        put(t, t + 0.04, form, v=vb)
        put(t + 0.04, t + 0.12, form, v=vb, nz=loud, band=band)
        return t + 0.12
    if kind == "fric":
        put(t, t + 0.10, form, v=0.4 if voiced else 0.0, nz=loud, band=band)
        return t + 0.10
    if kind == "aspirate":
        put(t, t + 0.07, None, h=loud)
        return t + 0.07
    if kind == "nasal":
        put(t, t + 0.075, form, v=loud)
        return t + 0.075
    put(t, t + 0.06, form, v=loud)                                  # approximants
    return t + 0.06


# --- the throat ----------------------------------------------------------------

def throat_fm(f0_frames, n, ratio=1.0, index=3.0, index_motion=0.0):
    f0 = np.repeat(f0_frames, HOP)[:n]
    f0 = np.pad(f0, (0, max(0, n - len(f0))), mode="edge")
    ph_c = 2 * np.pi * np.cumsum(f0) / SR
    ph_m = 2 * np.pi * np.cumsum(f0 * ratio) / SR
    idx = index * (1 + index_motion * np.sin(2 * np.pi * 0.3 * np.arange(n) / SR))
    return np.sin(ph_c + idx * np.sin(ph_m))


def throat_saw(f0_frames, n):
    f0 = np.repeat(f0_frames, HOP)[:n]
    f0 = np.pad(f0, (0, max(0, n - len(f0))), mode="edge")
    ph = np.cumsum(f0) / SR
    return 2 * (ph - np.floor(ph + 0.5))


def _reson(f, bw):
    r = np.exp(-np.pi * bw / SR)
    th = 2 * np.pi * f / SR
    return np.array([1 - r]), np.array([1, -2 * r * np.cos(th), r * r])


BW = (90.0, 110.0, 160.0, 250.0)
GAIN = (1.0, 0.7, 0.4, 0.2)


def render(text: str, notes: Sequence[Note], throat: str = "fm", carrier: Optional[np.ndarray] = None,
           fm: Optional[Dict] = None, tail: float = 0.3, seed: int = 7) -> np.ndarray:
    notes = list(notes)
    total = max(n.start + n.dur for n in notes) + tail
    tr = _targets(syllables(text), notes, total)
    nfr = len(tr["f0"])
    n = nfr * HOP
    if carrier is not None:
        src = np.zeros(n)
        src[:min(n, len(carrier))] = carrier[:n]
    elif throat == "saw":
        src = throat_saw(tr["f0"], n)
    else:
        src = throat_fm(tr["f0"], n, **(fm or {}))
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    out = np.zeros(n)
    zi_f = [None] * 4
    zi_n, zi_a = None, [None] * 4
    nb = None
    for i in range(nfr):
        a, b = i * HOP, (i + 1) * HOP
        Fs = list(tr["F"][i]) + [3500.0]
        seg_v = src[a:b] * tr["voice"][i]
        seg_h = white[a:b] * tr["asp"][i] * 0.3
        y = np.zeros(b - a)
        for k in range(4):
            bb, aa = _reson(Fs[k], BW[k])
            if zi_f[k] is None:
                zi_f[k] = lfilter_zi(bb, aa) * 0
                zi_a[k] = lfilter_zi(bb, aa) * 0
            yk, zi_f[k] = lfilter(bb, aa, seg_v, zi=zi_f[k])
            hk, zi_a[k] = lfilter(bb, aa, seg_h, zi=zi_a[k])
            y += GAIN[k] * (yk + hk)
        if tr["noise"][i] > 0:
            lo, hi = tr["nlo"][i], min(tr["nhi"][i], SR / 2 - 100)
            nb = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band")
            if zi_n is None or len(zi_n) != max(len(nb[0]), len(nb[1])) - 1:
                zi_n = lfilter_zi(*nb) * 0
            fr, zi_n = lfilter(nb[0], nb[1], white[a:b], zi=zi_n)
            y += fr * tr["noise"][i] * 0.6
        out[a:b] = y
    peak = np.abs(out).max()
    return out / peak * 0.8 if peak > 0 else out


def clarity(text: str, notes: Sequence[Note], **styled) -> float:
    """STOI of the styled voice against the same words through a plain saw (0-1)."""
    from pystoi import stoi
    ref = render(text, notes, throat="saw")
    sty = render(text, notes, **styled)
    m = min(len(ref), len(sty))
    return float(stoi(ref[:m], sty[:m], SR, extended=False))


def line(text: str, midis: Sequence[float], bpm: float = 100, beats: Optional[Sequence[float]] = None) -> List[Note]:
    """One note per syllable, back to back (beats per note default 1)."""
    beats = beats or [1.0] * len(midis)
    spb = 60.0 / bpm
    t, out = 0.1, []
    for m, b in zip(midis, beats):
        out.append(Note(m, t, b * spb))
        t += b * spb
    return out
