"""Audition: one section heard in its place in the song, without rendering the song.

Name a section, and which time it comes around (the 3rd chorus), and it renders just that
stretch: the section before, the section itself (as many passes as you ask, to hear it the
5th time round), and the start of the section after. One part can be swapped for new notes
just for this audition, so a change is heard in context before it goes in the song file.
Nothing is written to the song.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import yaml

from .song import SR, SongError, render

NEXT_BARS = 2   # how much of the next section to keep: enough to hear the landing


SOURCES = {"synth", "gm", "sfz", "kit", "samples"}


def window(song: Dict, section: str, time: int = 1, repeat: int = 1,
           swap: Optional[Dict] = None) -> Tuple[Dict, Optional[str]]:
    """The audition as a song of its own: before, the section (repeated), after; and the
    name of the section after, if there is one."""
    from .song import normalise
    normalise(song)
    sections = song.get("sections") or {}
    arrangement = list(song.get("arrangement") or sections)
    if section not in sections:
        raise SongError(f"no section {section!r}; the song has {', '.join(sections)}")
    where = [i for i, n in enumerate(arrangement) if n == section]
    if not where:
        raise SongError(f"{section!r} is written but the arrangement never plays it")
    time = int(time or 1)
    if not 1 <= time <= len(where):
        raise SongError(f"{section!r} comes around {len(where)} time{'s' if len(where) != 1 else ''}, "
                        f"so time goes from 1 to {len(where)}")
    repeat = int(repeat or 1)
    if not 1 <= repeat <= 8:
        raise SongError("repeat goes from 1 to 8")
    i = where[time - 1]
    trying = copy.deepcopy(sections[section])
    from .song import normalise
    instruments = normalise(song).get("instruments") or {}
    sound = {}
    for part, content in (swap or {}).items():
        if part in ("bars", "hold"):
            trying[part] = content
            continue
        if part == "instruments":
            if not isinstance(content, dict):
                raise SongError("swap instruments is parts by name with what to change, like "
                                "{\"instruments\": {\"pad\": {\"gain\": 0.4}}}")
            for n, change in content.items():
                if n not in instruments:
                    raise SongError(f"swap instruments names {n!r}, which isn't an instrument ({', '.join(instruments)})")
                if not isinstance(change, dict):
                    raise SongError(f"swap instruments {n!r} is what to change, like {{\"gain\": 0.4}}")
                sound[n] = change
            continue
        if part not in instruments:
            raise SongError(f"swap names {part!r}, which isn't an instrument ({', '.join(instruments)})")
        trying[part] = content
    label = f"{section} (trying)" if swap else section
    out = copy.deepcopy(song)
    for n, change in sound.items():           # how a part sounds: for the whole audition, before and after too
        spec = dict(out["instruments"][n] or {})
        if SOURCES & set(change):              # a new source replaces the old one
            spec = {k: v for k, v in spec.items() if k not in SOURCES | {"patch", "params"}}
        spec.update(change)
        out["instruments"][n] = spec
    out["sections"] = dict(sections)
    out["sections"][label] = trying
    arr = ([arrangement[i - 1]] if i > 0 else []) + [label] * repeat
    after = arrangement[i + 1] if i + 1 < len(arrangement) else None
    if after:
        arr.append(after)
    out["arrangement"] = arr
    out["title"] = f"{song.get('title', 'song')}: {section}, time {time}"
    return out, after


def audition(song_path: Path, out_dir: Path, section: str, time: int = 1, repeat: int = 1,
             swap: Optional[Dict] = None, log=print) -> Dict:
    song_path = Path(song_path)
    song = yaml.safe_load(song_path.read_text())
    w, after = window(song, section, time, repeat, swap)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audition.yaml").write_text(yaml.safe_dump(w, sort_keys=False))
    cut = None
    if after:
        # keep the landing: the first bars of the next section, and let them ring out a little
        from .song import normalise, parse
        _, _, marks = parse(normalise(copy.deepcopy(w)), song_path.parent)
        meter = int(song.get("meter", 4))
        bar = meter * 60.0 / float(song["tempo"])
        bars_next = int((song["sections"][after] or {}).get("bars", 1))
        cut = marks[-1][1] + min(NEXT_BARS, bars_next) * bar
    # the_mix hears what she hears: the crop, not the whole of the next section
    res = render(out_dir / "audition.yaml", out_dir, tail=2.0, log=log, base=song_path.parent, hear_until=cut)
    if after:
        y, sr = sf.read(out_dir / "render.wav", always_2d=True)
        end = min(len(y), int((cut + 1.0) * SR))
        y = y[:end]
        fade = min(int(0.25 * SR), end)
        y[-fade:] *= np.linspace(1, 0, fade)[:, None]
        sf.write(out_dir / "render.wav", y, sr, subtype="PCM_24")
        res["length_s"] = round(end / SR, 2)
        if res.get("by_section"):
            res["by_section"][-1]["bars"] = min(NEXT_BARS, bars_next)
            res["by_section"][-1]["note"] = f"only its first {min(NEXT_BARS, bars_next)} bars"
    res["window"] = w["arrangement"]
    res["swapped"] = sorted((swap or {}).keys())
    (out_dir / "render.json").write_text(json.dumps(res, indent=1))
    return res
