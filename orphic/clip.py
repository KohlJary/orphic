"""Clips: a part as MIDI, written and read directly.

A clip is a stretch of notes in beats from its own start: pitch, start, length, velocity
and whether the note is palm-muted. It's saved as a standard MIDI file (960 ticks a beat;
palm-muted notes on MIDI channel 2, everything else on channel 1), so a clip opens in any
DAW, and a record's transcription can become one.

She writes a clip either in the song notation ("[E2 B2 E3]/0.25pm ... | ...") or as notes:

    {"bar": 1, "beat": 1,   "pitch": ["E2", "B2", "E3"], "len": 0.25, "pm": true}
    {"bar": 1, "beat": 1.25, "pitch": "E2", "len": 0.25, "vel": 110}

and reads it back as a piano roll, a bar at a time: one row per pitch, one column per
sixteenth, so what's there and what's silent are both plain to see.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

TPB = 960
PM_CHANNEL = 1   # 0-based: MIDI channel 2


class ClipError(ValueError):
    pass


@dataclass
class Note:
    start: float    # beats from the clip's start
    length: float   # beats
    midi: int
    vel: int = 96
    pm: bool = False


def _name(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"


def from_notation(text: str, meter: int = 4) -> List[Note]:
    from .song import parse_melodic, split_bars, SongError
    bars = split_bars(text)
    try:
        ev = parse_melodic(text, meter, 1.0, 0.0, len(bars))
    except SongError as e:
        raise ClipError(str(e))
    return [Note(e.start, round(e.dur / 0.98, 6), e.midi, e.vel, e.pm) for e in ev]


def from_notes(rows: List[Dict], meter: int = 4) -> List[Note]:
    from .song import note_midi, SongError
    out = []
    for i, r in enumerate(rows or []):
        try:
            bar, beat = int(r.get("bar", 1)), float(r.get("beat", 1))
            length = float(r.get("len", r.get("length", 1)))
        except (TypeError, ValueError):
            raise ClipError(f"note {i + 1}: bar, beat and len are numbers")
        if bar < 1 or not (1 <= beat < meter + 1) or length <= 0:
            raise ClipError(f"note {i + 1}: bar from 1, beat from 1 to {meter + 1} (not included), len above 0")
        pitches = r.get("pitch")
        pitches = pitches if isinstance(pitches, list) else [pitches]
        try:
            mids = [int(p) if isinstance(p, int) else note_midi(str(p)) for p in pitches]
        except SongError as e:
            raise ClipError(f"note {i + 1}: {e}")
        vel = max(1, min(127, int(r.get("vel", 96))))
        start = (bar - 1) * meter + (beat - 1)
        out += [Note(start, length, m, vel, bool(r.get("pm"))) for m in mids]
    return sorted(out, key=lambda n: (n.start, n.midi))


def save(notes: List[Note], path: Path, meter: int = 4) -> Path:
    import mido
    mid = mido.MidiFile(ticks_per_beat=TPB)
    trk = mido.MidiTrack()
    mid.tracks.append(trk)
    trk.append(mido.MetaMessage("time_signature", numerator=meter, denominator=4, time=0))
    evs = []
    for n in notes:
        ch = PM_CHANNEL if n.pm else 0
        a, b = int(round(n.start * TPB)), int(round((n.start + n.length) * TPB))
        evs.append((a, 1, mido.Message("note_on", channel=ch, note=n.midi, velocity=n.vel)))
        evs.append((b, 0, mido.Message("note_off", channel=ch, note=n.midi, velocity=0)))
    evs.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for t, _, m in evs:
        m.time = t - last
        last = t
        trk.append(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(path)
    return path


def load(path: Path, track: Optional[str] = None) -> List[Note]:
    """A MIDI file's notes (all tracks, or the one named), in beats."""
    import mido
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    out = []
    for t in mid.tracks:
        if track and t.name.lower() != track.lower():
            continue
        now, on = 0, {}
        for m in t:
            now += m.time
            if m.type == "note_on" and m.velocity > 0:
                on[(m.channel, m.note)] = (now, m.velocity)
            elif m.type in ("note_off", "note_on") and (m.channel, m.note) in on:
                a, v = on.pop((m.channel, m.note))
                out.append(Note(a / tpb, max(1, now - a) / tpb, m.note, v, m.channel == PM_CHANNEL))
    return sorted(out, key=lambda n: (n.start, n.midi))


def bars_of(notes: List[Note], meter: int = 4) -> int:
    end = max((n.start + n.length for n in notes), default=0.0)
    return max(1, int(-(-round(end, 6) // meter)))


def roll(notes: List[Note], meter: int = 4, first: int = 1, last: Optional[int] = None,
         steps_per_beat: int = 4) -> str:
    """A piano roll, a bar at a time: rows are pitches (high at the top), columns
    sixteenths. X starts a note, m starts a palm-muted one, = holds, . is silence."""
    nb = bars_of(notes, meter)
    last = min(last or nb, nb)
    step = 1.0 / steps_per_beat
    out = []
    head = "".join(str(b + 1) + "   " for b in range(meter))[: meter * steps_per_beat]
    for bar in range(first, last + 1):
        a, b = (bar - 1) * meter, bar * meter
        inbar = [n for n in notes if n.start < b - 1e-9 and n.start + n.length > a + 1e-9]
        sounding = [False] * (meter * steps_per_beat)
        lines = []
        for m in sorted({n.midi for n in inbar}, reverse=True):
            cells = ["."] * (meter * steps_per_beat)
            for n in (x for x in inbar if x.midi == m):
                s = int(round((n.start - a) / step))
                e = int(round((n.start + n.length - a) / step))
                for c in range(max(0, s), min(len(cells), max(e, s + 1))):
                    cells[c] = "=" if c != s else ("m" if n.pm else "X")
                    sounding[c] = True
            lines.append(f"  {_name(m):>4} |{''.join(cells)}|")
        filled = sum(sounding) / steps_per_beat
        note = "" if filled >= meter else f"  (sounds {filled:g} of {meter} beats)"
        out.append(f"bar {bar}{note}\n         {head}\n" + ("\n".join(lines) if lines else "  (empty)"))
    return "\n".join(out)


def to_notation(notes: List[Note], meter: int = 4) -> str:
    """Back to the song notation, where it fits (chords at one start and length become [..])."""
    groups: Dict[tuple, List[Note]] = {}
    for n in notes:
        groups.setdefault((round(n.start, 4), round(n.length, 4), n.pm), []).append(n)
    bars = []
    for bar in range(bars_of(notes, meter)):
        a, toks, pos = bar * meter, [], bar * meter
        for (s, L, pm) in sorted(k for k in groups if bar * meter <= k[0] < (bar + 1) * meter):
            if s > pos + 1e-6:
                toks.append(f"r/{s - pos:g}")
            ms = sorted(n.midi for n in groups[(s, L, pm)])
            L2 = min(L, a + meter - s)
            what = _name(ms[0]) if len(ms) == 1 else "[" + " ".join(_name(m) for m in ms) + "]"
            toks.append(f"{what}/{L2:g}" + ("pm" if pm else ""))
            pos = max(pos, s + L2)
        if pos < a + meter - 1e-6:
            toks.append(f"r/{a + meter - pos:g}")
        bars.append(" ".join(toks))
    return " | ".join(bars)
