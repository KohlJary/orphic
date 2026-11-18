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

Lengths are in beats. In 4/4: 4 is the whole bar, 2 a half note, 1 a quarter, 0.5 an
eighth, 0.25 a sixteenth. Each bar should add up to the meter: whatever a bar doesn't fill
is silence (write it as r/beats when that's what you mean). E3/1 rings for one beat, not
until the next note; E3/4 holds the whole bar.

Chords: a name (Em/4, G/2) or the notes in brackets. A power chord is root, fifth and
octave: [E2 B2 E3]/1. Palm-mute with pm after the length ([E2 B2]/0.25pm), accent with ^.
A chugging bar: "[E2 B2]/0.25pm [E2 B2]/0.25pm [E2 B2]/0.5pm [G2 D3]/1 [A2 E3]/2".

Instruments: `synth` (a VST3 synth by name or path: vitalium, nekobi, surge, ...,
with `params` by name, or `patch`, a saved patch from her gun belt), `gm` (a General
MIDI program, played by FluidSynth), or `kit: "808"` (a real TR-808, sampled at known
settings; `kick: BD2575` and so on choose the samples), or `sfz` (a sampled instrument: fsbs-direct, a
real electric guitar). `fx` is an effects chain and `rig` a saved one from the gun belt.
Every instrument has `gain` and `pan`.
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
EFFECTS = {"swankyamp": "SwankyAmp.vst3", "zamtube": "ZamTube.vst3"}
BUILTIN_FX = ("distortion", "compressor", "reverb", "chorus", "delay", "gain", "highpass", "lowpass", "noisegate",
              "convolution", "phaser", "bitcrush", "ladderfilter")
SOUNDFONT = "/usr/share/soundfonts/FluidR3_GM.sf2"
SFZ_DIR = Path(os.environ.get("ORPHIC_SFZ_DIR", Path.home() / "acousmatic-data" / "guitar"))
# sampled instruments: name -> folder under SFZ_DIR holding one .sfz (with its license)
SFZ_LIBS = {"fsbs-direct": "freepats-direct"}   # Freepats FSBS electric guitar, direct (CC0): run it into an amp
PM_SECONDS = float(os.environ.get("ORPHIC_PM_SECONDS", 0.08))      # how quickly a palm-muted note is damped
PM_LOWPASS_HZ = float(os.environ.get("ORPHIC_PM_LOWPASS", 700.0))  # the palm takes the highs off the string
PM_GAIN = float(os.environ.get("ORPHIC_PM_GAIN", 1.4))             # chugs are picked hard
PM_CLICK_S = 0.006                                                  # the pick's click stays bright
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
    pm: bool = False   # palm-muted


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
            pm = tok.endswith("pm") or tok.endswith("pm^")
            tok = tok.replace("pm", "") if pm else tok
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
                    out.append(Event(start, L * spb * 0.98, mm, 120 if accent else 96, pm))
            beat += L
        if beat > beats_per_bar + 1e-6:
            raise SongError(f"bar {bi + 1} '{bt}' has {beat:g} beats, more than {beats_per_bar}")
    return out


def bar_fill(text: str, beats_per_bar: int) -> List[str]:
    """What each written bar holds, and any bar that leaves beats silent."""
    out = []
    for bi, bt in enumerate(b.strip() for b in str(text).split("|")):
        beats, toks = 0.0, []
        for tok in re.findall(r"\[[^\]]*\]/\S+|\S+", bt):
            what, _, length = tok.replace("pm", "").rstrip("^").rpartition("/")
            try:
                L = float(length)
            except ValueError:
                continue
            beats += L
            toks.append(f"{'rest' if what.lower() == 'r' else what} {L:g}" + (" muted" if "pm" in tok else ""))
        line = f"bar {bi + 1}: " + ", ".join(toks) + f" = {beats:g} of {beats_per_bar} beats"
        if beats < beats_per_bar - 1e-6:
            line += f" (the last {beats_per_bar - beats:g} beats are silent)"
        out.append(line)
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


def _midi_file(events, path: Path, program: Optional[int] = None):
    import mido
    mid = mido.MidiFile(ticks_per_beat=960)
    trk = mido.MidiTrack()
    mid.tracks.append(trk)
    trk.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))   # 120 bpm: seconds * 1920 ticks
    if program is not None:
        trk.append(mido.Message("program_change", program=max(0, min(127, program - 1)), time=0))
    evs = []
    for e in events:
        evs.append((e.start, 1, mido.Message("note_on", note=e.midi, velocity=e.vel)))
        evs.append((e.start + e.dur, 0, mido.Message("note_off", note=e.midi, velocity=0)))
    evs.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for t, _, msg in evs:
        tick = int(round(t * 1920))
        msg.time = tick - last
        last = tick
        trk.append(msg)
    mid.save(path)


def _fit(a: np.ndarray, total: float) -> np.ndarray:
    n = int(total * SR)
    return a[:, :n] if a.shape[1] >= n else np.pad(a, ((0, 0), (0, n - a.shape[1])))


def render_sfz(tr: Track, total: float) -> np.ndarray:
    name = str(tr.spec["sfz"])
    folder = SFZ_DIR / SFZ_LIBS.get(name, name)
    sfz = next(folder.rglob("*.sfz"), None) if folder.exists() else None
    if sfz is None:
        raise SongError(f"no sampled instrument {name!r} (known: {', '.join(SFZ_LIBS)})")
    with tempfile.TemporaryDirectory() as d:
        mp, wp = Path(d) / "t.mid", Path(d) / "t.wav"
        _midi_file(tr.events, mp)
        subprocess.run(["sfizz_render", "--sfz", str(sfz), "--midi", str(mp), "--wav", str(wp), "-s", str(SR)],
                       check=True, capture_output=True, timeout=600)
        a, _ = sf.read(wp, always_2d=True)
    return _fit(a.T.astype(np.float64), total)


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


def resolve_chain(spec: Dict):
    """An instrument's effects: its own fx, after a rig from the gun belt if it names one."""
    chain = []
    if spec.get("rig"):
        p = BELT / f"{spec['rig']}.json"
        if not p.exists():
            raise SongError(f"no rig {spec['rig']!r} on the gun belt")
        chain += json.loads(p.read_text()).get("chain", [])
    return chain + list(spec.get("fx") or [])


def save_rig(name: str, chain, notes: str = "") -> Path:
    """A pedalboard on the gun belt: an effects chain under a name she can call (rig: name)."""
    BELT.mkdir(parents=True, exist_ok=True)
    p = BELT / f"{name}.json"
    p.write_text(json.dumps({"name": name, "kind": "rig", "chain": chain, "notes": notes}, indent=1))
    return p


def render_source(tr: Track, total: float):
    """A melodic instrument's audio, palm mutes included: palm-muted notes are played in a
    pass of their own, cut short and damped (fast decay, highs rolled off), then joined."""
    def one(t: Track):
        if "gm" in t.spec:
            return render_gm(t, total), f"General MIDI {t.spec['gm']}"
        if t.spec.get("sfz"):
            return render_sfz(t, total), f"sampled {t.spec['sfz']}"
        if t.spec.get("synth") or t.spec.get("patch"):
            return render_synth(t, total), t.spec.get("patch") or t.spec.get("synth")
        raise SongError(f"instrument {t.name!r} needs synth, patch, gm, sfz or kit")
    open_ev = [e for e in tr.events if not e.pm]
    pm_ev = [Event(e.start, min(e.dur, PM_SECONDS), e.midi, e.vel, True) for e in tr.events if e.pm]
    if not pm_ev:
        return one(tr)
    a, kind = one(Track(tr.name, tr.spec, events=open_ev)) if open_ev else (np.zeros((2, int(total * SR))), None)
    b, kind2 = one(Track(tr.name, tr.spec, events=pm_ev))
    from scipy.signal import butter, sosfilt
    dark = sosfilt(butter(2, PM_LOWPASS_HZ / (SR / 2), output="sos"), b, axis=1)
    click = np.zeros(b.shape[1])     # the first few ms of each mute keep their highs
    for e in pm_ev:
        i = int(e.start * SR)
        click[i:i + int(PM_CLICK_S * SR)] = 1.0
    b = (dark * (1 - click) + b * click) * PM_GAIN
    # the damping: each palm-muted note dies away over about PM_SECONDS
    # each mute's damping runs only until the next palm-muted note starts (at speed, a
    # sixteenth is under 100 ms: a longer envelope would flatten the next attack)
    env = np.ones(b.shape[1])
    starts = sorted({int(e.start * SR) for e in pm_ev})
    for k, i in enumerate(starts):
        j = min(b.shape[1], i + int(0.35 * SR), starts[k + 1] if k + 1 < len(starts) else b.shape[1])
        t = np.arange(j - i) / SR
        env[i:j] = np.exp(-t / (PM_SECONDS / 2.5)) * 0.9 + 0.1 * (t < 0.01)
    b = b * env
    return _fit(a, total) + _fit(b, total), (kind or kind2) + " (with palm mutes)"


def apply_fx(a: np.ndarray, chain) -> np.ndarray:
    """An instrument's effects, in order: a VST3 effect by name (swankyamp, zamtube, or a path)
    with params, or one of pedalboard's built-ins (distortion, compressor, reverb, delay,
    chorus, gain, highpass, lowpass, noisegate, convolution for a cabinet IR, ...)."""
    if not chain:
        return a
    import pedalboard as pb
    built = {"distortion": pb.Distortion, "compressor": pb.Compressor, "reverb": pb.Reverb, "chorus": pb.Chorus,
             "delay": pb.Delay, "gain": pb.Gain, "highpass": pb.HighpassFilter, "lowpass": pb.LowpassFilter,
             "noisegate": pb.NoiseGate, "convolution": pb.Convolution, "phaser": pb.Phaser, "bitcrush": pb.Bitcrush,
             "ladderfilter": pb.LadderFilter}
    out = np.ascontiguousarray(a.astype(np.float32))
    for fx in chain:
        fx = dict(fx)
        if "builtin" in fx:
            kind = str(fx.pop("builtin")).lower()
            if kind not in built:
                raise SongError(f"no built-in effect {kind!r} (one of {', '.join(BUILTIN_FX)})")
            try:
                eff = built[kind](**fx)
            except TypeError as e:
                raise SongError(f"{kind}: {e}")
        else:
            name = str(fx.get("plugin") or "")
            path = name if os.path.exists(name) else None
            for d in PLUGIN_DIRS:
                cand = Path(d) / EFFECTS.get(name.lower(), name)
                if path is None and cand.exists():
                    path = str(cand)
            if not path:
                raise SongError(f"no effect {name!r} (known: {', '.join(EFFECTS)}, or a built-in)")
            from pedalboard import load_plugin
            eff = load_plugin(path)
            _set_params(eff, fx.get("params"))
        out = eff(out, SR, reset=True)
        if out.shape[0] == 1:
            out = np.vstack([out, out])
    return out.astype(np.float64)[:2]


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
        else:
            a, kind = render_source(tr, total)
        a = apply_fx(a, resolve_chain(tr.spec))
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
