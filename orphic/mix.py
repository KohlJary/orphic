"""Hearing a mix: what's covering what, measured from the song's own stems.

A render has every part on its own track, so this is exact about who plays what and where;
it's a rough model of hearing. What it reports, in plain findings with where they happen:

- covered: how much of a part sits under other parts, in the same critical bands (the ear's
  frequency bands, about a third of an octave wide) at the same moments, by enough that the
  ear hears the louder one instead. Who covers it, in which frequencies, worst where.
- collisions: two parts' attacks landing on each other above 5 kHz (sibilants on hats,
  picks on shakers), so the top end of each smears into the other.
- balance by section: a part that drops well under its own level somewhere, or under the mix.
- the lead's presence band (2-5 kHz, where the ear finds words and melody): who crowds it.

The masking model is simple on purpose: a part's band is covered when the others in that band
(and, more faintly, the band just under it, since masking spreads upward) are 6 dB or more
louder, in both the left and right channel. So panning two parts apart does help, the way
it does in a real mix. It isn't a model of a listener's attention, and it can't say whether
a part is meant to sit under the others: a pad is often supposed to.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

SR = 44100
N_FFT = 2048
HOP = 1024
# Zwicker's critical band edges, Hz
EDGES = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150,
         3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]
NB = len(EDGES) - 1
COVER_DB = 6.0          # others this much louder in the band: the part is covered there
SPREAD_BELOW_DB = 10.0  # a masker one band down still covers, this much weaker
FLOOR_DB = 45.0         # a part's cell counts as sounding within this many dB of the part's own loudest cell
REPORT_COVERED = 0.25   # report a part when at least this share of it is covered
LEADS = ("vocal", "vox", "voice", "lead", "melody", "sing")

_freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
_band_of = np.searchsorted(EDGES, _freqs, side="right") - 1
_M = np.zeros((len(_freqs), NB), dtype=np.float32)       # bin -> band, summed by one matmul
for _i, _b in enumerate(_band_of):
    if 0 <= _b < NB:
        _M[_i, _b] = 1.0


def bands(a: np.ndarray) -> np.ndarray:
    """Energy per channel, frame and critical band: shape (2, frames, bands)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = np.vstack([a, a])
    n = 1 + max(0, (a.shape[1] - N_FFT) // HOP)
    win = np.hanning(N_FFT).astype(np.float32)
    out = np.zeros((2, n, NB))
    for ch in range(2):
        x = a[ch]
        for s in range(0, n, 512):                    # in blocks, to keep memory small on long songs
            k = np.arange(s, min(n, s + 512))
            frames = x[(k[:, None] * HOP) + np.arange(N_FFT)[None, :]] * win
            p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
            out[ch, k] = p @ _M
    return out


def _db(e):
    return 10 * np.log10(e + 1e-20)


def _hz(lo: int, hi: int) -> str:
    def f(v):
        return f"{v / 1000:g} kHz" if v >= 1000 else f"{v} Hz"
    return f"{f(EDGES[lo])} to {f(EDGES[hi + 1])}"


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


class _Where:
    """Frame -> section (with which time round it is), for saying where something happens."""

    def __init__(self, marks: List[Tuple[str, float]], length: float):
        self.spans = []
        seen: Dict[str, int] = {}
        for i, (sec, t0) in enumerate(marks):
            t1 = marks[i + 1][1] if i + 1 < len(marks) else length
            seen[sec] = seen.get(sec, 0) + 1
            self.spans.append((sec, seen[sec], t0, t1))

    def of(self, frame: int) -> int:
        t = frame * HOP / SR
        for i, (_, _, t0, t1) in enumerate(self.spans):
            if t0 <= t < t1:
                return i
        return len(self.spans) - 1

    def say(self, i: int) -> str:
        sec, n, t0, t1 = self.spans[i]
        nth = {1: "", 2: ", the second time", 3: ", the third time"}.get(n, f", time {n}")
        return f"{sec}{nth} ({_clock(t0)}-{_clock(t1)})"


def pick_lead(parts, lead: Optional[str]) -> Optional[str]:
    if lead:
        return lead if lead in parts else None
    for n in parts:
        if any(w in n.lower() for w in LEADS):
            return n
    return None


def hear(parts: Dict[str, np.ndarray], marks: List[Tuple[str, float]], length: float,
         lead: Optional[str] = None) -> Dict:
    """parts: name -> bands() of that part's stem (after its gain and pan). Plain findings."""
    names = [n for n in parts if parts[n].sum() > 0]
    if len(names) < 2 or not marks:
        return {"findings": [], "note": "one part alone: nothing to cover it"}
    n = min(parts[k].shape[1] for k in names)
    E = {k: parts[k][:, :n] for k in names}
    total = sum(E.values())
    where = _Where(marks, length)
    sec_of = np.array([where.of(f) for f in range(n)])
    findings: List[str] = []
    covered_share: Dict[str, float] = {}
    detail: Dict[str, Dict] = {}

    for p in names:
        e = E[p]
        others = total - e
        spread = np.concatenate([np.zeros((2, n, 1)), others[:, :, :-1]], axis=2) * 10 ** (-SPREAD_BELOW_DB / 10)
        masker = others + spread
        mono = e.sum(axis=0)
        sounding = _db(mono) > _db(mono.max()) - FLOOR_DB
        cov = (_db(masker) - _db(e) >= COVER_DB).all(axis=0) & sounding
        w = mono * sounding
        if w.sum() == 0:
            continue
        share = float((mono * cov).sum() / w.sum())
        covered_share[p] = round(share, 2)
        if share < REPORT_COVERED:
            continue
        # who: the other part with the most energy in the covered cells
        by = {q: float((E[q].sum(axis=0) * cov).sum()) for q in names if q != p}
        who = sorted(by, key=by.get, reverse=True)
        main = who[0]
        second = who[1] if len(who) > 1 and by[who[1]] > 0.5 * by[main] else None
        # which frequencies: the bands holding most of the covered energy
        per_band = (mono * cov).sum(axis=0)
        order = np.argsort(per_band)[::-1]
        keep, acc = [], 0.0
        for b in order:
            keep.append(int(b))
            acc += per_band[b]
            if acc >= 0.7 * per_band.sum():
                break
        lo, hi = min(keep), max(keep)
        # where: the section where the most of it is covered
        per_sec = np.bincount(sec_of, weights=(mono * cov).sum(axis=1), minlength=len(where.spans))
        sec_all = np.bincount(sec_of, weights=w.sum(axis=1), minlength=len(where.spans))
        frac = np.divide(per_sec, sec_all, out=np.zeros_like(per_sec), where=sec_all > 0)
        frac[sec_all < 0.1 * sec_all.max()] = 0        # only sections where it really plays, not its tail
        worst = int(np.argmax(frac))
        under = float(np.median((_db(masker.sum(axis=0)) - _db(mono))[cov]))
        by_txt = main + (f" and {second}" if second else "")
        findings.append(f"{p}: {round(share * 100)}% of what it plays is covered, mostly by {by_txt}, "
                        f"in {_hz(lo, hi)}; worst in {where.say(worst)}, where {round(frac[worst] * 100)}% "
                        f"of it is under. Where it's covered, the others are about {under:.0f} dB louder in those bands.")
        detail[p] = {"covered": round(share, 2), "by": who[:3], "hz": [EDGES[lo], EDGES[hi + 1]],
                     "worst": where.say(worst)}

    findings += _collisions(E, names, where)
    findings += _balance(E, names, where, sec_of)
    lead = pick_lead(names, lead)
    if lead:
        f = _presence(E, names, lead, where, sec_of)
        if f:
            findings.append(f)
    return {"findings": findings, "covered": covered_share, "detail": detail, "lead": lead,
            "how": "measured from the stems: critical bands, others 6 dB louder in both channels counts as "
                   "covered. It can't tell a part that's buried from one meant to sit under."}


HI = EDGES.index(5300)


def _hi(e: np.ndarray) -> np.ndarray:
    return e.sum(axis=0)[:, HI:].sum(axis=1)


def _onsets(e: np.ndarray, mix_hi: np.ndarray) -> np.ndarray:
    """Attack frames above 5.3 kHz: a jump in that band's energy, where the part is loud enough up
    there to be heard (within 10 dB of everything else there)."""
    hi = _hi(e)
    if hi.max() <= 0:
        return np.array([], dtype=int)
    d = _db(hi)
    jump = np.diff(d, prepend=d[0])
    heard = d > _db(mix_hi) - 10
    return np.where((jump > 9) & heard)[0]


def _collisions(E, names, where) -> List[str]:
    out = []
    mix_hi = _hi(sum(E.values()))
    ons = {p: _onsets(E[p], mix_hi) for p in names}
    for i, p in enumerate(names):
        for q in names[i + 1:]:
            a, b = ons[p], ons[q]
            if len(a) < 4 or len(b) < 4:
                continue
            hit = a[np.isin(a, np.concatenate([b - 1, b, b + 1]))]
            if len(hit) < 4 or len(hit) < 0.3 * min(len(a), len(b)):
                continue
            secs = np.bincount([where.of(int(f)) for f in hit], minlength=len(where.spans))
            worst = int(np.argmax(secs))
            out.append(f"{p} and {q}: their attacks land together above 5.3 kHz {len(hit)} times "
                       f"({round(100 * len(hit) / min(len(a), len(b)))}% of {p if len(a) <= len(b) else q}'s), "
                       f"most in {where.say(worst)}; the top of each smears into the other there.")
    return out


def _balance(E, names, where, sec_of) -> List[str]:
    out = []
    mixe = sum(E.values()).sum(axis=(0, 2))
    for p in names:
        pe = E[p].sum(axis=(0, 2))
        rel = []
        for i in range(len(where.spans)):
            s_ = sec_of == i
            m = s_ & (pe > pe.max() * 1e-3)                 # playing, within 30 dB of its loudest
            rel.append(float(_db(pe[m].mean()) - _db(mixe[m].mean())) if m.sum() > max(4, 0.25 * s_.sum()) else None)
        have = [r for r in rel if r is not None]
        if len(have) < 2:
            continue
        typical = float(np.median(have))
        for i, r in enumerate(rel):
            if r is not None and r < typical - 8:
                out.append(f"{p} drops to {r:.0f} dB under the mix in {where.say(i)}, from about "
                           f"{typical:.0f} dB elsewhere.")
    return out


def _presence(E, names, lead, where, sec_of) -> Optional[str]:
    lo, hi = EDGES.index(2000), EDGES.index(5300)
    le = E[lead].sum(axis=0)[:, lo:hi].sum(axis=1)
    on = le > le.max() * 10 ** (-40 / 10)
    if on.sum() < 5:
        return None
    share = {q: float(E[q].sum(axis=0)[:, lo:hi].sum(axis=1)[on].sum()) for q in names}
    tot = sum(share.values())
    mine = share.pop(lead) / tot if tot else 0
    if mine >= 0.5:
        return f"{lead} owns its presence band (2 to 5.3 kHz): {round(mine * 100)}% of what's there while it plays."
    crowd = sorted(share, key=share.get, reverse=True)
    names_txt = ", ".join(f"{q} {round(100 * share[q] / tot)}%" for q in crowd[:3] if share[q] / tot >= 0.05)
    return (f"{lead}'s presence band (2 to 5.3 kHz, where the ear finds a melody or words) is crowded: "
            f"{lead} is only {round(mine * 100)}% of what's there while it plays; the rest is {names_txt}.")
