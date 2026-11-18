"""A song as a text file, rendered: the studio, tier 1.

She writes the song in YAML: tempo, meter, instruments, sections and an arrangement. Each
part is written bar by bar ("|" between bars). Drums use a 16-step grid per bar, the same
x/. notation her ears show her. Melodic parts are NOTE/beats tokens, chords by name or as
[notes], and rests (r/beats).

    title: First light
    tempo: 92
    meter: 4
    instruments:
      drums: {kit: "808"}
      bass:  {synth: vitalium, gain: 0.8}
      keys:  {gm: 4, gain: 0.5, pan: -0.3}
    sections:
      verse:
        bars: 4
        drums:
          kick:  "x.......x.x....."
          snare: "....x.......x..."
          hat:   "x.x.x.x.x.x.x.x."
        bass: "E2/2 r/1 E2/1 | G2/2 A2/2 | C3/3 B2/1 | E2/4"
        keys: "Em/4 | G/4 | C/4 | B7/4"
    arrangement: [verse, verse]

Instruments: `synth` (a VST3 synth by name or path: vitalium, nekobi, surge, ...,
with `params` by name, or `patch`, a saved patch from her gun belt), `gm` (a General
MIDI program, played by FluidSynth), or `kit: "808"` (a real TR-808, sampled at known
settings; `kick: BD2575` and so on choose the samples). Every instrument has `gain` and `pan`.
Rendering is deterministic: the same file gives the same audio.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import yaml

SR = 44100
PLUGIN_DIRS = ["/usr/lib/vst3", "/usr/local/lib/vst3", str(Path.home() / ".vst3")]
SYNTHS = {"vitalium": "vitalium.vst3", "nekobi": "Nekobi.vst3", "surge": "Surge.vst3", "sfizz": "sfizz.vst3",
          "kars": "Kars.vst3"}
SOUNDFONT = "/usr/share/soundfonts/FluidR3_GM.sf2"
KIT_DIR = Path(os.environ.get("ORPHIC_808", Path.home() / "acousmatic-data" / "808" / "fischer"))
KIT_DEFAULT = {"kick": "BD/BD2575", "snare": "SD/SD5025", "clap": "CP/CP", "hat": "CH/CH", "open": "OH/OH50",
               "rim": "RS/RS", "cowbell": "CB/CB", "low_tom": "LT/LT50", "mid_tom": "MT/MT50", "high_tom": "HT/HT50",
               "cymbal": "CY/CY5050", "clave": "CL/CL", "maracas": "MA/MA"}
BELT = Path(os.environ.get("ORPHIC_BELT", Path.home() / "orphic" / "belt"))

NOTE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")
PCS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
QUAL = {"": (0, 4, 7), "m": (0, 3, 7), "5": (0, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
        "dim": (0, 3, 6), "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7), "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9),
        "9": (0, 4, 7, 10, 14), "add9": (0, 4, 7, 14), "m9": (0, 3, 7, 10, 14)}


class SongError(ValueError):
    pass


def note_midi(tok: str) -> int:
    m = NOTE.match(tok)
    if not m:
        raise SongError(f"not a note: {tok!r} (like E2, C#4, Bb3)")
    pc = PCS[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    return 12 * (int(m.group(3)) + 1) + pc


def chord_midis(name: str, octave: int = 3) -> List[int]:
    m = re.match(r"^([A-G])([#b]?)(.*)$", name)
    if not m or m.group(3) not in QUAL:
        raise SongError(f"not a chord: {name!r} (like Em, C, G7, Bbmaj7, F#m7, Dsus4)")
    root = 12 * (octave + 1) + PCS[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    return [root + i for i in QUAL[m.group(3)]]


@dataclass
class Event:
    start: float   # seconds
    dur: float
    midi: int
    vel: int = 96


@dataclass
class Track:
    name: str
    spec: Dict
    events: List[Event] = field(default_factory=list)
    hits: List[Tuple[float, str, int]] = field(default_factory=list)   # drums: (time, voice, velocity)


def parse_melodic(text: str, beats_per_bar: int, spb: float, t0: float, bars: int) -> List[Event]:
    bar_texts = [b.strip() for b in str(text).split("|")]
    out: List[Event] = []
    for bi in range(bars):
        bt = bar_texts[bi % len(bar_texts)]
        beat = 0.0
        for tok in re.findall(r"\[[^\]]*\]/\S+|\S+", bt):
            accent = tok.endswith("^")
            tok = tok.rstrip("^")
            if "/" not in tok:
                raise SongError(f"{tok!r} needs a length in beats, like {tok}/1")
            what, length = tok.rsplit("/", 1)
            try:
                L = float(length)
            except ValueError:
                raise SongError(f"{tok!r}: the length after / is a number of beats")
            start = t0 + (bi * beats_per_bar + beat) * spb
            if what.lower() != "r":
                if what.startswith("["):
                    mids = [note_midi(x) for x in what.strip("[]").split()] if " " in what else \
                        [note_midi(x) for x in what.strip("[]").split(",")]
                elif NOTE.match(what):
                    mids = [note_midi(what)]
                else:
                    mids = chord_midis(what)
                for mm in mids:
                    out.append(Event(start, L * spb * 0.98, mm, 120 if accent else 96))
            beat += L
        if beat > beats_per_bar + 1e-6:
            raise SongError(f"bar {bi + 1} '{bt}' has {beat:g} beats, more than {beats_per_bar}")
    return out


def parse_drums(grid: Dict[str, str], beats_per_bar: int, spb: float, t0: float, bars: int):
    steps = beats_per_bar * 4
    hits = []
    for voice, text in (grid or {}).items():
        if voice not in KIT_DEFAULT:
            raise SongError(f"no drum called {voice!r}; the kit has {', '.join(KIT_DEFAULT)}")
        bar_texts = [b.replace(" ", "") for b in str(text).split("|")]
        for bi in range(bars):
            bt = bar_texts[bi % len(bar_texts)]
            if len(bt) != steps:
                raise SongError(f"{voice} bar '{bt}' has {len(bt)} steps; a bar of {beats_per_bar} beats has {steps}")
            for k, c in enumerate(bt):
                if c in "xX":
                    hits.append((t0 + (bi * beats_per_bar + k / 4) * spb, voice, 120 if c == "X" else 96))
    return hits


def parse(song: Dict) -> Tuple[Dict[str, Track], float, List[Tuple[str, float]]]:
    try:
        tempo = float(song["tempo"])
    except (KeyError, TypeError, ValueError):
        raise SongError("a song needs a tempo (beats per minute)")
    meter = int(song.get("meter", 4))
    spb = 60.0 / tempo
    instruments = song.get("instruments") or {}
    sections = song.get("sections") or {}
    arrangement = song.get("arrangement") or list(sections)
    tracks = {n: Track(n, spec or {}) for n, spec in instruments.items()}
    t = 0.0
    marks = []
    for name in arrangement:
        sec = sections.get(name)
        if sec is None:
            raise SongError(f"the arrangement names {name!r}, but there's no such section")
        bars = int(sec.get("bars", 1))
        marks.append((name, round(t, 3)))
        for part, content in sec.items():
            if part == "bars":
                continue
            if part not in tracks:
                raise SongError(f"section {name!r} has a part {part!r} that isn't an instrument")
            tr = tracks[part]
            if tr.spec.get("kit"):
                tr.hits += parse_drums(content, meter, spb, t, bars)
            else:
                tr.events += parse_melodic(content, meter, spb, t, bars)
        t += bars * meter * spb
    return tracks, t, marks


# --- instruments ------------------------------------------------------------------

def _plugin_path(name: str) -> str:
    if os.path.exists(name):
        return name
    file = SYNTHS.get(name.lower(), name)
    for d in PLUGIN_DIRS:
        p = Path(d) / file
        if p.exists():
            return str(p)
    raise SongError(f"no synth {name!r} (known: {', '.join(SYNTHS)})")


def _set_params(plugin, params: Dict):
    for k, v in (params or {}).items():
        if k not in plugin.parameters:
            raise SongError(f"the synth has no parameter {k!r}")
        try:
            setattr(plugin, k, v)
        except Exception:
            plugin.parameters[k].raw_value = float(v)


def render_synth(tr: Track, total: float) -> np.ndarray:
    from mido import Message
    from pedalboard import load_plugin
    spec = dict(tr.spec)
    if spec.get("patch"):
        p = BELT / f"{spec['patch']}.json"
        if not p.exists():
            raise SongError(f"no patch {spec['patch']!r} on the gun belt")
        saved = json.loads(p.read_text())
        spec.setdefault("synth", saved["synth"])
        spec["params"] = {**saved.get("params", {}), **(spec.get("params") or {})}
    plugin = load_plugin(_plugin_path(spec["synth"]))
    _set_params(plugin, spec.get("params"))
    msgs = []
    for e in tr.events:
        msgs.append(Message("note_on", note=e.midi, velocity=e.vel, time=e.start))
        msgs.append(Message("note_off", note=e.midi, velocity=0, time=e.start + e.dur))
    msgs.sort(key=lambda m: (m.time, m.type == "note_on"))
    a = None
    for ch in (2, 1):
        try:
            a = plugin(msgs, duration=total, sample_rate=SR, num_channels=ch, reset=True)
            break
        except ValueError:
            continue
    if a is None:
        raise SongError(f"the synth {spec['synth']!r} gave no 1- or 2-channel output")
    a = np.asarray(a, dtype=np.float64)
    return np.vstack([a, a]) if a.shape[0] == 1 else a[:2]


def render_gm(tr: Track, total: float) -> np.ndarray:
    import mido
    prog = int(tr.spec["gm"])
    mid = mido.MidiFile(ticks_per_beat=960)
    trk = mido.MidiTrack()
    mid.tracks.append(trk)
    tempo_us = 500000   # 120 bpm: ticks are seconds * 1920
    trk.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    trk.append(mido.Message("program_change", program=max(0, min(127, prog - 1)), time=0))
    evs = []
    for e in tr.events:
        evs.append((e.start, 1, mido.Message("note_on", note=e.midi, velocity=e.vel)))
        evs.append((e.start + e.dur, 0, mido.Message("note_off", note=e.midi, velocity=0)))
    evs.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for t, _, msg in evs:
        tick = int(round(t * 1920))
        msg.time = tick - last
        last = tick
        trk.append(msg)
    with tempfile.TemporaryDirectory() as d:
        mp, wp = Path(d) / "t.mid", Path(d) / "t.wav"
        mid.save(mp)
        subprocess.run(["fluidsynth", "-ni", "-R", "0", "-C", "0", "-g", "0.8", "-F", str(wp), "-r", str(SR),
                        SOUNDFONT, str(mp)], check=True, capture_output=True)
        a, _ = sf.read(wp, always_2d=True)
    a = a.T.astype(np.float64)
    n = int(total * SR)
    return a[:, :n] if a.shape[1] >= n else np.pad(a, ((0, 0), (0, n - a.shape[1])))


_SAMPLES: Dict[str, np.ndarray] = {}


def _sample(rel: str) -> np.ndarray:
    if rel not in _SAMPLES:
        p = KIT_DIR / f"{rel}.WAV"
        if not p.exists():
            raise SongError(f"no 808 sample {rel!r}")
        y, sr = sf.read(p)
        if sr != SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        _SAMPLES[rel] = y.astype(np.float64)
    return _SAMPLES[rel]


def render_kit(tr: Track, total: float) -> np.ndarray:
    choice = dict(KIT_DEFAULT)
    for v, s in (tr.spec.get("samples") or {}).items():
        choice[v] = s if "/" in s else f"{s[:2]}/{s}"
    n = int(total * SR)
    out = np.zeros(n)
    for t, voice, vel in tr.hits:
        y = _sample(choice[voice]) * (vel / 127.0)
        a = int(round(t * SR))
        b = min(n, a + len(y))
        if a < n:
            out[a:b] += y[: b - a]
    return np.vstack([out, out])


def _pan(st: np.ndarray, pan: float) -> np.ndarray:
    th = (float(pan) + 1) * np.pi / 4
    return np.vstack([st[0] * np.cos(th) * np.sqrt(2), st[1] * np.sin(th) * np.sqrt(2)])


def render(song_path: Path, out_dir: Path, tail: float = 2.0, log=print) -> Dict:
    song = yaml.safe_load(Path(song_path).read_text())
    tracks, length, marks = parse(song)
    total = length + tail
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stems").mkdir(exist_ok=True)
    mix = np.zeros((2, int(total * SR)))
    info = {}
    for name, tr in tracks.items():
        if tr.spec.get("kit"):
            a = render_kit(tr, total)
            kind = "808 kit"
        elif "gm" in tr.spec:
            a = render_gm(tr, total)
            kind = f"General MIDI {tr.spec['gm']}"
        elif tr.spec.get("synth") or tr.spec.get("patch"):
            a = render_synth(tr, total)
            kind = tr.spec.get("patch") or tr.spec.get("synth")
        else:
            raise SongError(f"instrument {name!r} needs synth, patch, gm or kit")
        n = mix.shape[1]
        a = a[:, :n] if a.shape[1] >= n else np.pad(a, ((0, 0), (0, n - a.shape[1])))
        a = _pan(a * float(tr.spec.get("gain", 0.8)), tr.spec.get("pan", 0.0))
        sf.write(out_dir / "stems" / f"{name}.wav", a.T, SR, subtype="PCM_24")
        mix += a
        info[name] = {"plays": kind, "notes": len(tr.events), "hits": len(tr.hits),
                      "peak_db": round(float(20 * np.log10(np.abs(a).max() + 1e-12)), 1)}
        log(f"  {name}: {kind}, {len(tr.events) or len(tr.hits)} {'notes' if tr.events else 'hits'}")
    peak = np.abs(mix).max()
    clipped = bool(peak > 1.0)
    if peak > 0:
        mix = mix / peak * 0.89   # about -1 dBFS
    sf.write(out_dir / "render.wav", mix.T, SR, subtype="PCM_24")
    result = {"title": song.get("title", Path(song_path).stem), "tempo": song["tempo"], "length_s": round(length, 2),
              "sections": marks, "tracks": info, "mix_peak_before_normalising_db": round(float(20 * np.log10(peak + 1e-12)), 1),
              "would_have_clipped": clipped}
    (out_dir / "render.json").write_text(json.dumps(result, indent=1))
    return result


def save_patch(name: str, synth: str, params: Dict, notes: str = "") -> Path:
    """Put a patch on the gun belt: a synth and its settings, under a name she can call."""
    BELT.mkdir(parents=True, exist_ok=True)
    p = BELT / f"{name}.json"
    p.write_text(json.dumps({"name": name, "synth": synth, "params": params, "notes": notes}, indent=1))
    return p
