"""Monochord: learn a synth by sweeping its knobs.

Pythagoras's monochord was one string over a movable bridge: slide the bridge, measure
what happens. Here: from the synth's default patch, each parameter is screened at its
two extremes (most do nothing from a default patch: effects that are off, unused
modulators); each one that changes the sound is then swept through its range, and
every render is measured. The map says, per parameter, what it does to the sound and
how (steadily, in steps, only at one end), and which parameters matter most.

What a parameter does depends on the patch it's turned in: this map is from the
default patch. Other starting patches are a later pass.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .host import HELD, LOW, Synth
from .measure import LABELS, change, measure

SKIP_WORDS = ("bypass", "midi", "program", "preset", "bpm", "beats_per_minute", "tempo", "sync", "volume_master",
              "polyphony", "voice_count", "mpe", "oversampl", "pitch_bend_range", "legato", "portamento_type")


def _effect(base: Dict, other: Dict) -> Dict[str, float]:
    return {k: v for k, v in change(base, other).items() if v >= 1.0}


def screen(s: Synth, probe=HELD, log=print) -> Dict[str, Dict]:
    base = measure(s.render(probe), probe.hold)
    moved = {}
    names = [n for n in s.params if not any(w in n.lower() for w in SKIP_WORDS)]
    for i, name in enumerate(names):
        eff = {}
        for v in (0.0, 1.0):
            if abs(v - s.defaults[name]) < 1e-6:
                continue
            m = measure(s.render(probe, {name: v}), probe.hold)
            for k, x in _effect(base, m).items():
                eff[k] = max(eff.get(k, 0), x)
        if eff:
            moved[name] = eff
        if (i + 1) % 100 == 0:
            log(f"  screened {i + 1}/{len(names)}; {len(moved)} change the sound")
    return {"base": base, "moved": moved, "screened": len(names)}


def sweep(s: Synth, name: str, steps: int = 7, probe=HELD) -> List[Dict]:
    rows = []
    p = s.params[name]
    for v in np.linspace(0, 1, steps):
        m = measure(s.render(probe, {name: float(v)}), probe.hold)
        try:
            p.raw_value = float(v)
            s._dirty.add(name)
            shown = p.string_value
        except Exception:
            shown = None
        rows.append({"raw": round(float(v), 3), "shown": shown, **m})
    s.restore()
    return rows


def _shape(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 3:
        return ""
    d = np.diff(xs)
    if np.all(d >= -1e-9) or np.all(d <= 1e-9):
        big = np.abs(d).max() / (np.abs(d).sum() + 1e-12)
        return "jumps once" if big > 0.7 else ("steadily" if big < 0.4 else "unevenly")
    return "up and down"


def _fmt(k, v):
    if v is None:
        return "-"
    if k in ("brightness_hz", "f0_hz", "peak_hz"):
        return f"{v:.0f}"
    return f"{v:g}"


def describe(name: str, rows: List[Dict], effects: Dict[str, float]) -> str:
    bits = []
    for k in sorted(effects, key=lambda k: -effects[k])[:4]:
        vals = [r.get(k) for r in rows]
        if all(v is None for v in vals):
            continue
        a, b = vals[0], vals[-1]
        bits.append(f"{LABELS[k]} {_fmt(k, a)} → {_fmt(k, b)} ({_shape(vals)})")
    ends = (rows[0].get("shown"), rows[-1].get("shown"))
    rng = f" [{ends[0]} … {ends[1]}]" if all(ends) else ""
    shown = [r.get("shown") for r in rows]
    if all(shown) and not _numeric(shown) and len(set(shown)) < len(shown):
        # a switch (waveform, filter type): what each position measures, not just the ends
        k = sorted(effects, key=lambda k: -effects[k])[0] if effects else None
        pos, seen = [], set()
        for r in rows:
            if r["shown"] in seen:
                continue
            seen.add(r["shown"])
            pos.append(f"{r['shown']}: " + ", ".join(f"{LABELS[x].split(' (')[0]} {_fmt(x, r.get(x))}"
                                                      for x in sorted(effects, key=lambda x: -effects[x])[:3]))
        return f"- **{name}** (switch): " + "; ".join(pos)
    return f"- **{name}**{rng}: " + "; ".join(bits)


def _numeric(vals):
    try:
        [float(str(v).split()[0]) for v in vals]
        return True
    except ValueError:
        return False


def map_synth(path: str, out_dir: Path, steps: int = 7, top: int = 60, log=print) -> Dict:
    t0 = time.time()
    s = Synth(path)
    log(f"{Path(path).name}: {len(s.params)} parameters")
    sc = screen(s, log=log)
    ranked = sorted(sc["moved"].items(), key=lambda kv: -sum(min(v, 10) for v in kv[1].values()))
    log(f"  {len(ranked)} of {sc['screened']} change the sound from the default patch; sweeping the top {min(top, len(ranked))}")
    swept = {}
    for name, eff in ranked[:top]:
        swept[name] = {"effects": eff, "held": sweep(s, name, steps), "low": None}
    # the strongest few, on a short low note too (plucks, envelopes)
    for name, _ in ranked[: min(12, len(ranked))]:
        swept[name]["low"] = sweep(s, name, steps, probe=LOW)
    result = {"synth": Path(path).name, "path": path, "parameters": len(s.params), "screened": sc["screened"],
              "base": sc["base"], "changes_sound": len(ranked), "swept": swept,
              "still_from_default": [n for n in s.params if n not in sc["moved"]][:2000],
              "probes": {"held": HELD.name, "low": LOW.name}, "seconds": round(time.time() - t0, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "map.json").write_text(json.dumps(result, indent=1, default=float))
    (out_dir / "map.md").write_text(render(result))
    log(f"  done in {result['seconds']}s -> {out_dir}")
    return result


def render(r: Dict) -> str:
    b = r["base"]
    L = [f"# {r['synth']}: how it works", "",
         f"Monochord map (orphic). {r['parameters']} parameters; from the default patch, {r['changes_sound']} of the "
         f"{r['screened']} screened change the sound (the rest belong to parts that are off, or do their work only "
         "in other patches). Each that does was swept end to end on a held C4; the strongest also on a short C2.", "",
         "## The default patch", "",
         f"Pitch {_fmt('f0_hz', b.get('f0_hz'))} Hz, brightness {_fmt('brightness_hz', b.get('brightness_hz'))} Hz, "
         f"noise {b.get('noise')}, harmonicity {b.get('harmonicity')}, attack {b.get('attack_ms')} ms, sustain "
         f"{b.get('sustain')}, release {b.get('release_ms')} ms, width {b.get('width')}.", "",
         "## What matters, strongest first", "",
         "Each line: the measures it moves most, from its lowest setting to its highest, and how.", ""]
    for name, d in r["swept"].items():
        L.append(describe(name, d["held"], d["effects"]))
        if d.get("low"):
            low = d["low"]
            eff = {k: v for k, v in change(low[0], low[-1]).items() if v >= 1}
            if eff:
                L.append("  on a short C2: " + describe(name, low, eff).split(": ", 1)[-1])
    L += ["", f"## Swept in {r['seconds']} s. Measures", ""] + [f"- {k}: {v}" for k, v in LABELS.items()]
    return "\n".join(L) + "\n"
