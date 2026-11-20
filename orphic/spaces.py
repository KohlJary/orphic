"""Spaces: a named room instead of reverb knobs.

An instrument (or the whole song) says `space: high-ceilings`, and plays in that room. Each
space is a few numbers with a plain line saying what it is; the numbers are there to read,
and to copy when she writes her own. Her own spaces live on the gun belt
(belt/spaces/<name>.json, written with save_space), in her words, and win over the built-ins.

What a space is made of:
- size: how big the room is, 0 to 1 (pedalboard's room_size)
- damping: how much the walls soak up the highs as it rings, 0 to 1
- walls_ms: how far away the walls are: the gap before the room answers (pre-delay)
- amount: how much of the room you hear against the instrument itself, 0 to 1
- tone_hz: the room's brightness: the ring is rolled off above this
- width: how wide the room is in the stereo field, 0 to 1

The reverb underneath (Freeverb, through pedalboard) can't make a room that rings much under
half a second; a closet sounds close because you hear little of it, not because it's short.
Reaper sessions don't carry spaces yet: a session plays the parts dry.

`ring_s` is measured, not set: an impulse goes through the room and we time how long it
takes the ring to fall 60 dB. It's what the numbers actually did.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np

SR = 44100
FIELDS = {"size": (0.0, 1.0), "damping": (0.0, 1.0), "walls_ms": (0.0, 500.0), "amount": (0.0, 1.0),
          "tone_hz": (500.0, 20000.0), "width": (0.0, 1.0)}

BUILT_IN: Dict[str, Dict] = {
    "closet": {"says": "a small room full of coats: close, dead, a little dark",
               "size": 0.12, "damping": 0.85, "walls_ms": 2, "amount": 0.18, "tone_hz": 5000, "width": 0.4},
    "room": {"says": "a living room: you can hear the walls, but they're near",
             "size": 0.35, "damping": 0.55, "walls_ms": 8, "amount": 0.2, "tone_hz": 9000, "width": 0.7},
    "live-room": {"says": "a studio's live room: wood, tall enough, a bright short bloom",
                  "size": 0.5, "damping": 0.35, "walls_ms": 14, "amount": 0.22, "tone_hz": 12000, "width": 0.9},
    "high-ceilings": {"says": "a tall empty room: the sound goes up before it comes back, and hangs there",
                      "size": 0.82, "damping": 0.25, "walls_ms": 45, "amount": 0.3, "tone_hz": 11000, "width": 1.0},
    "hall": {"says": "a concert hall: big, warm, a long even ring",
             "size": 0.88, "damping": 0.45, "walls_ms": 30, "amount": 0.3, "tone_hz": 8000, "width": 1.0},
    "cathedral": {"says": "stone and height: far walls, a ring that outlasts the note",
                  "size": 0.97, "damping": 0.2, "walls_ms": 80, "amount": 0.38, "tone_hz": 7000, "width": 1.0},
}


class SpaceError(ValueError):
    pass


def _belt() -> Path:
    return Path(os.environ.get("ORPHIC_BELT", Path.home() / "orphic" / "belt")) / "spaces"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", str(name).lower()).strip("-")


def _check(sp: Dict, name: str) -> Dict:
    out = {}
    for k, (lo, hi) in FIELDS.items():
        if k not in sp:
            raise SpaceError(f"space {name!r} needs {k} (a space has {', '.join(FIELDS)})")
        try:
            v = float(sp[k])
        except (TypeError, ValueError):
            raise SpaceError(f"{k} in space {name!r} is a number")
        if not lo <= v <= hi:
            raise SpaceError(f"{k} in space {name!r} goes from {lo:g} to {hi:g}; it's {v:g}")
        out[k] = v
    return out


def get(spec) -> Dict:
    """A space by name (hers first, then the built-ins), or a name with changes:
    {name: hall, amount: 0.15}, or a whole space written inline."""
    if isinstance(spec, str):
        spec = {"name": spec}
    if not isinstance(spec, dict):
        raise SpaceError("space is a name (space: hall) or {name: hall, amount: 0.2}")
    base: Dict = {}
    name = spec.get("name")
    if name:
        n = slug(name)
        p = _belt() / f"{n}.json"
        if p.exists():
            base = json.loads(p.read_text())
        elif n in BUILT_IN:
            base = BUILT_IN[n]
        else:
            raise SpaceError(f"no space {name!r}; there's {', '.join(sorted(set(BUILT_IN) | set(mine())))}, "
                             "or write your own")
    merged = {**base, **{k: v for k, v in spec.items() if k != "name"}}
    return _check(merged, name or "inline")


def mine() -> Dict[str, Dict]:
    d = _belt()
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def apply(a: np.ndarray, spec) -> np.ndarray:
    """The sound in the room: the instrument itself, plus the room's answer."""
    import pedalboard as pb
    sp = get(spec)
    x = np.ascontiguousarray(a.astype(np.float32))
    pre = int(SR * sp["walls_ms"] / 1000)
    if pre:
        x = np.pad(x, ((0, 0), (pre, 0)))[:, : a.shape[1]]
    room = pb.Pedalboard([pb.Reverb(room_size=sp["size"], damping=sp["damping"], wet_level=1.0, dry_level=0.0,
                                    width=sp["width"]),
                          pb.LowpassFilter(cutoff_frequency_hz=sp["tone_hz"])])
    wet = room(x, SR, reset=True).astype(np.float64)
    if wet.shape[0] == 1:
        wet = np.vstack([wet, wet])
    return a + sp["amount"] * wet[:2, : a.shape[1]]


@lru_cache(maxsize=64)
def _ring(size: float, damping: float, walls_ms: float, tone_hz: float, width: float) -> float:
    imp = np.zeros((2, SR * 12))
    imp[:, 0] = 1.0
    sp = {"size": size, "damping": damping, "walls_ms": walls_ms, "amount": 1.0, "tone_hz": tone_hz, "width": width}
    wet = apply(imp, sp) - imp
    e = (wet ** 2).sum(axis=0)
    edc = np.cumsum(e[::-1])[::-1]               # Schroeder's backward integration
    edc_db = 10 * np.log10(edc / (edc[0] + 1e-30) + 1e-30)
    t = np.arange(len(edc_db)) / SR
    m = (edc_db <= -5) & (edc_db >= -25)          # T20: fit the fall from -5 to -25 dB, times three
    if m.sum() < 10:
        return 0.0
    slope = np.polyfit(t[m], edc_db[m], 1)[0]
    return round(float(-60.0 / slope), 2) if slope < 0 else 0.0


def ring_s(spec) -> float:
    """How long the room rings (seconds to fall 60 dB), measured with an impulse."""
    sp = get(spec)
    return _ring(sp["size"], sp["damping"], sp["walls_ms"], sp["tone_hz"], sp["width"])


def listing() -> Dict[str, Dict]:
    out = {}
    for n, sp in {**BUILT_IN, **mine()}.items():
        out[n] = {"says": sp.get("says", ""), **{k: sp[k] for k in FIELDS if k in sp},
                  "ring_s": ring_s(n), "mine": n in mine()}
    return out


def save_space(name: str, says: str, numbers: Dict) -> Dict:
    """Her own space on the gun belt: what it is in her words, and its numbers."""
    n = slug(name)
    if not n:
        raise SpaceError("a space needs a name")
    if not (says or "").strip():
        raise SpaceError("a space needs a line in your words saying what room it is")
    sp = _check({**BUILT_IN.get(n, {}), **(numbers or {})}, n)
    entry = {"name": n, "kind": "space", "says": says.strip(), **sp}
    _belt().mkdir(parents=True, exist_ok=True)
    (_belt() / f"{n}.json").write_text(json.dumps(entry, indent=1))
    _ring.cache_clear()
    return {**entry, "ring_s": ring_s(n)}
