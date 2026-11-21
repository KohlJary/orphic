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

A part can also be a MIDI clip from the project's clips/ folder, looped to fill the section:
`guitar: {clip: chug}` plays clips/chug.mid (palm-muted notes are on MIDI channel 2).

Instruments: `synth` (a VST3 synth by name or path: vitalium, nekobi, surge, ...,
with `params` by name, or `patch`, a saved patch from her gun belt), `gm` (a General
MIDI program, played by FluidSynth), `kit:` an acoustic kit (black-pearl, red-zeppelin,
blonde-bop, blonde-bop-hr, buskmans-holiday; AVL Drumkits, with kick, snare, rim, hat,
hat_pedal, open, low_tom, mid_tom, high_tom, crash, ride, ride_bell, splash, crash2),
`kit: "909"` (a TR-909 modelled from its circuits: kick, snare, low/mid/high_tom, rim,
clap, hat, open, crash, ride; knobs per voice, 0 to 1, like knobs: {kick: {tune: 0.4,
attack: 0.7, decay: 0.6}, snare: {tone: 0.6, snappy: 0.8}}; its hats and cymbals are an
approximation, the real ones were samples), or `kit: "808"` (a real TR-808, sampled at known
settings; `kick: BD2575` and so on choose the samples), or `sfz` (a sampled instrument: fsbs-direct, a
real electric guitar). `fx` is an effects chain and `rig` a saved one from the gun belt.
Every instrument has `gain` and `pan`.

Room: `space: high-ceilings` on an instrument puts it in a named room instead of reverb
knobs (closet, room, live-room, high-ceilings, hall, cathedral, or one of your own from the
gun belt); `space: {name: hall, amount: 0.15}` changes one thing about it. `space:` at the
top of the song puts the whole band in one room. `orphic spaces list` shows each room's
numbers and how long it rings (measured).

Holding: a section can say `hold: 3` (beats). Whatever is still sounding when the section
ends keeps ringing that long, and nothing new starts until it's over, so the next section
arrives when you've decided it should.

    sections:
      verse: {bars: 8, hold: 2, keys: "..."}

Hearing a change in place: `orphic audition song.yaml chorus --time 3 --repeat 2` plays the
section before the 3rd chorus, the chorus twice, and the first bars of what follows, without
rendering the song; `--swap '{"bass": "E2/2 G2/2 | ..."}'` tries different notes for a part
(or a different hold) just this once. Nothing is written to the song.

Hearing the mix: every render and audition says what's covering what (`the_mix`), measured
from the song's own stems: a part sitting under others in the same frequencies (who covers
it, where, and how much louder they are), bright attacks landing on each other, a part
that drops away in one section, and who crowds the lead's presence band (2 to 5 kHz).
`lead: melody` at the top of the song names the lead; otherwise a part called vocal, lead
or melody is taken as it. It can't tell a part that's buried from one meant to sit under:
that's yours to say.

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
SFZ_LIBS = {"fsbs-direct": "freepats-direct"}
SFZ_LEVEL_DB = {"fsbs-direct": 8.0}   # a DI guitar sample set comes out ~8 dB under a synth at the same gain
INSTRUMENT_KEYS = {"synth", "patch", "params", "gm", "sfz", "kit", "samples", "knobs", "gain", "volume", "pan", "fx", "rig", "space"}   # Freepats FSBS electric guitar, direct (CC0): run it into an amp
PM_SECONDS = float(os.environ.get("ORPHIC_PM_SECONDS", 0.08))      # how quickly a palm-muted note is damped
PM_LOWPASS_HZ = float(os.environ.get("ORPHIC_PM_LOWPASS", 700.0))  # the palm takes the highs off the string
PM_GAIN = float(os.environ.get("ORPHIC_PM_GAIN", 2.2))             # chugs are picked hard
PM_CLICK_S = 0.006                                                  # the pick's click stays bright
KIT_DIR = Path(os.environ.get("ORPHIC_808", Path.home() / "acousmatic-data" / "808" / "fischer"))
KIT_DEFAULT = {"kick": "BD/BD2575", "snare": "SD/SD5025", "clap": "CP/CP", "hat": "CH/CH", "open": "OH/OH50",
               "rim": "RS/RS", "cowbell": "CB/CB", "low_tom": "LT/LT50", "mid_tom": "MT/MT50", "high_tom": "HT/HT50",
               "cymbal": "CY/CY5050", "clave": "CL/CL", "maracas": "MA/MA"}
# acoustic kits: x42 AVLDrums' SoundFonts (Glen MacArthur's AVL Drumkits), General MIDI drum notes
AVL_DIR = Path("/usr/lib/lv2/avldrums.lv2")
ACOUSTIC_KITS = {"black-pearl": "Black_Pearl_4_LV2.sf2", "red-zeppelin": "Red_Zeppelin_4_LV2.sf2",
                 "blonde-bop": "Blonde_Bop_LV2.sf2", "blonde-bop-hr": "Blonde_Bop_HR_LV2.sf2",
                 "buskmans-holiday": "Buskmans_Holiday_LV2.sf2"}
ACOUSTIC_NOTES = {"kick": 36, "rim": 37, "snare": 38, "low_tom": 41, "hat": 42, "hat_pedal": 44, "mid_tom": 45,
                  "open": 46, "high_tom": 48, "crash": 49, "cymbal": 49, "ride": 51, "ride_bell": 53, "splash": 55,
                  "crash2": 57}
DRUM_NOTES = {"kick": 36, "rim": 37, "snare": 38, "clap": 39, "low_tom": 41, "hat": 42, "hat_pedal": 44,
              "mid_tom": 45, "open": 46, "high_tom": 48, "crash": 49, "cymbal": 49, "ride": 51, "ride_bell": 53,
              "splash": 55, "cowbell": 56, "crash2": 57, "maracas": 70, "clave": 75}
NOTE_DRUM = {}
for _v, _n in DRUM_NOTES.items():
    NOTE_DRUM.setdefault(_n, _v)
MISSING_CLIPS: List[str] = []   # clips a song names that aren't written yet (they play as silence)
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


def split_bars(text: str) -> List[str]:
    """Bars of a part; a "|" at the very end (or start) doesn't make an empty bar."""
    bars = [b.strip() for b in str(text).split("|")]
    while len(bars) > 1 and not bars[-1]:
        bars.pop()
    while len(bars) > 1 and not bars[0]:
        bars.pop(0)
    return bars


def parse_melodic(text: str, beats_per_bar: int, spb: float, t0: float, bars: int) -> List[Event]:
    bar_texts = split_bars(text)
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
    for bi, bt in enumerate(split_bars(text)):
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


def parse_drums(grid: Dict[str, str], beats_per_bar: int, spb: float, t0: float, bars: int, kit: str = "808"):
    steps = beats_per_bar * 4
    hits = []
    from .tr909 import VOICES as V909
    voices = ACOUSTIC_NOTES if str(kit).lower() in ACOUSTIC_KITS else \
        {v: 0 for v in V909} if str(kit) == "909" else KIT_DEFAULT
    for voice, text in (grid or {}).items():
        if voice not in voices:
            raise SongError(f"no drum called {voice!r}; the {kit} kit has {', '.join(voices)}")
        bar_texts = [b.replace(" ", "") for b in str(text).split("|")]
        for bi in range(bars):
            bt = bar_texts[bi % len(bar_texts)]
            if len(bt) != steps:
                raise SongError(f"{voice} bar '{bt}' has {len(bt)} steps; a bar of {beats_per_bar} beats has {steps}")
            for k, c in enumerate(bt):
                if c in "xX":
                    hits.append((t0 + (bi * beats_per_bar + k / 4) * spb, voice, 120 if c == "X" else 96))
    return hits


def _clip_events(content, base: Optional[Path], meter: int, spb: float, t0: float, bars: int) -> List[Event]:
    """A part given as a clip ({clip: name}, from the song's clips/ folder), looped over the section."""
    from .clip import bars_of, load
    name = str(content["clip"])
    p = (base or Path(".")) / "clips" / (name if name.endswith(".mid") else f"{name}.mid")
    if not p.exists():
        if name not in MISSING_CLIPS:
            MISSING_CLIPS.append(name)
        return []   # not written yet: silence in this section, and the render says so
    notes = load(p)
    span = bars_of(notes, meter) * meter
    out = []
    loop = 0
    while loop * span < bars * meter - 1e-9:
        for n in notes:
            at = loop * span + n.start
            if at < bars * meter - 1e-9:
                out.append(Event(t0 + at * spb, min(n.length, bars * meter - at) * spb * 0.98, n.midi, n.vel, n.pm))
        loop += 1
    return out


def parse(song: Dict, base: Optional[Path] = None) -> Tuple[Dict[str, Track], float, List[Tuple[str, float]]]:
    MISSING_CLIPS.clear()
    try:
        tempo = float(song["tempo"])
    except (KeyError, TypeError, ValueError):
        raise SongError("a song needs a tempo (beats per minute)")
    meter = int(song.get("meter", 4))
    spb = 60.0 / tempo
    instruments = song.get("instruments") or {}
    sections = song.get("sections") or {}
    arrangement = song.get("arrangement") or list(sections)
    tracks = {}
    for n, spec in instruments.items():
        spec = dict(spec or {})
        unknown = set(spec) - INSTRUMENT_KEYS
        if unknown:
            raise SongError(f"instrument {n!r} has {', '.join(sorted(unknown))}, which the studio doesn't know; "
                            f"an instrument takes {', '.join(sorted(INSTRUMENT_KEYS - {'volume'}))}")
        if "volume" in spec and "gain" not in spec:
            spec["gain"] = spec.pop("volume")   # volume reads as gain
        tracks[n] = Track(n, spec)
    t = 0.0
    marks = []
    for name in arrangement:
        sec = sections.get(name)
        if sec is None:
            raise SongError(f"the arrangement names {name!r}, but there's no such section")
        bars = int(sec.get("bars", 1))
        try:
            hold = float(sec.get("hold") or 0)
        except (TypeError, ValueError):
            raise SongError(f"hold in {name!r} is a number of beats, like hold: 3")
        if hold < 0:
            raise SongError(f"hold in {name!r} can't be negative")
        marks.append((name, round(t, 3)))
        before = {n: len(tr.events) for n, tr in tracks.items()}
        for part, content in sec.items():
            if part in ("bars", "hold"):
                continue
            if part not in tracks:
                raise SongError(f"section {name!r} has a part {part!r} that isn't an instrument")
            tr = tracks[part]
            if isinstance(content, dict) and set(content) == {"notes"}:
                content = content["notes"]   # {notes: "..."} reads the same as the notation itself
            if tr.spec.get("kit") and isinstance(content, dict) and content.get("clip"):
                # a drum clip: its notes are General MIDI drum notes
                for e in _clip_events(content, base, meter, spb, t, bars):
                    v = NOTE_DRUM.get(e.midi)
                    if v:
                        tr.hits.append((e.start, v, e.vel))
            elif tr.spec.get("kit"):
                if not isinstance(content, dict):
                    raise SongError(f"{part} is a drum kit: write a 16-step grid per drum, one bar each, like "
                                    f"{part}: {{kick: \"x.......x.x.....\", snare: \"....x.......x...\", "
                                    f"hat: \"x.x.x.x.x.x.x.x.\"}} (\"|\" between bars)")
                tr.hits += parse_drums(content, meter, spb, t, bars, str(tr.spec.get("kit")))
            elif isinstance(content, dict) and content.get("clip"):
                tr.events += _clip_events(content, base, meter, spb, t, bars)
            elif isinstance(content, dict):
                raise SongError(f"{part} in {name!r}: a part is notation (\"E2/1 G2/1 ...\") or {{clip: name}}, "
                                f"not {sorted(content)}")
            else:
                tr.events += parse_melodic(content, meter, spb, t, bars)
        end = t + bars * meter * spb
        if hold:
            # what's still sounding when the section ends keeps ringing for the hold,
            # and nothing new starts until it's over
            for n, tr in tracks.items():
                for e in tr.events[before[n]:]:
                    if e.start + e.dur >= end - spb * 0.25:   # notes are played a hair short of their length
                        e.dur = end + hold * spb - e.start
        t = end + hold * spb
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


PARAM_CACHE = Path(os.environ.get("ORPHIC_CACHE", Path.home() / ".cache" / "orphic")) / "param-values.json"
_PARAMS: Dict[str, float] = {}


def _param_cache(f: str):
    """Knob values already turned into the plugin's raw 0-1 values, by plugin file (its size
    and date, so an updated plugin is looked up afresh), knob and value. Finding the raw
    value is a search over the knob's whole range; the answer never changes."""
    try:
        st = os.stat(f)
        key0 = f"{f}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        key0 = f"{f}|?"
    if not _PARAMS and PARAM_CACHE.exists():
        try:
            _PARAMS.update(json.loads(PARAM_CACHE.read_text()))
        except ValueError:
            pass
    return _PARAMS, key0


def _param_cache_save(cache: Dict[str, float]) -> None:
    try:
        PARAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PARAM_CACHE.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(PARAM_CACHE)
    except OSError:
        pass


def _quick_load():
    """pedalboard looks at every knob of a plugin as it loads it, even with no knobs to set
    (Nekobi has 2090, mostly MIDI CCs: ten seconds). With nothing to set that look changes
    nothing, so it's skipped; _set_params looks only at the knobs a song names."""
    try:
        import pedalboard._pedalboard as P
    except ImportError:
        return
    M = P._PythonExternalPluginMixin
    if getattr(M, "_orphic_quick", False):
        return
    full = M.__set_initial_parameter_values__

    def quick(self, parameter_values):
        if parameter_values:
            return full(self, parameter_values)
    M.__set_initial_parameter_values__ = quick
    M._orphic_quick = True


def _set_params(plugin, params: Dict, path: str = ""):
    """Set a synth's knobs by name. Only the knobs named are looked at: pedalboard's own
    lookup builds every parameter first (each one searched over 1000 steps), which took a
    synth like Vitalium about 15 seconds before a note played."""
    if not params:
        return
    try:
        from pedalboard._pedalboard import AudioProcessorParameter, to_python_parameter_name
        names = {}
        for cp in plugin._parameters:
            n = to_python_parameter_name(cp)
            if n:
                names.setdefault(n, cp.name)
    except Exception:   # a pedalboard without these internals: the slow way below
        names = None
    if names is not None and all(k in names for k in params):
        cache, key0 = _param_cache(str(path or ""))
        cpp = {cp.name: cp for cp in plugin._parameters}
        for k, v in params.items():
            key = f"{key0}|{names[k]}|{json.dumps(v)}"
            if key not in cache:
                p = AudioProcessorParameter(plugin, names[k])
                try:
                    cache[key] = float(p.get_raw_value_for(v))
                except Exception:
                    cache[key] = float(v)
                _param_cache_save(cache)
            cpp[names[k]].raw_value = cache[key]
        return
    for k, v in (params or {}).items():
        if k not in plugin.parameters:
            import difflib
            has = sorted((x for x in plugin.parameters if k.lower() in x.lower()),
                         key=lambda x: (not x.startswith(("filter", "osc", "env", "amp")), len(x)))
            close = has[:4] or difflib.get_close_matches(k, list(plugin.parameters), n=4, cutoff=0.4)
            raise SongError(f"the synth has no parameter {k!r}" + (f"; close: {', '.join(close)}" if close else "")
                            + " (studio map lists them all)")
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
    _quick_load()
    path = _plugin_path(spec["synth"])
    plugin = load_plugin(path)
    _set_params(plugin, spec.get("params"), path)
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


def tool(name: str) -> str:
    """A system program, found even when the caller's PATH is thin (a systemd service)."""
    import shutil
    return shutil.which(name) or next((f"{d}/{name}" for d in ("/usr/bin", "/usr/local/bin", "/bin")
                                       if os.path.exists(f"{d}/{name}")), name)


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
        subprocess.run([tool("sfizz_render"), "--sfz", str(sfz), "--midi", str(mp), "--wav", str(wp), "-s", str(SR)],
                       check=True, capture_output=True, timeout=600)
        a, _ = sf.read(wp, always_2d=True)
    return _fit(a.T.astype(np.float64), total) * 10 ** (SFZ_LEVEL_DB.get(name, 0.0) / 20)


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
        subprocess.run([tool("fluidsynth"), "-ni", "-R", "0", "-C", "0", "-g", "0.8", "-F", str(wp), "-r", str(SR),
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
    kit = str(tr.spec.get("kit")).lower()
    if kit in ACOUSTIC_KITS:
        return render_acoustic(tr, total, AVL_DIR / ACOUSTIC_KITS[kit])
    if kit == "909":
        from .tr909 import voice
        n = int(total * SR)
        out = np.zeros(n)
        cache = {}
        for t, v, vel in tr.hits:
            if v not in cache:
                cache[v] = voice(v, (tr.spec.get("knobs") or {}).get(v))
            y = cache[v] * (vel / 127.0)
            a = int(round(t * SR))
            b = min(n, a + len(y))
            if a < n:
                out[a:b] += y[: b - a]
        return np.vstack([out, out])
    if kit != "808":
        raise SongError(f"no kit {tr.spec.get('kit')!r}; kits: 808, 909, {', '.join(ACOUSTIC_KITS)}")
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


def render_acoustic(tr: Track, total: float, sf2: Path) -> np.ndarray:
    """An acoustic kit: the hits as MIDI on its General MIDI notes, played by FluidSynth."""
    import mido
    if not sf2.exists():
        raise SongError(f"the kit's samples aren't installed ({sf2.name})")
    mid = mido.MidiFile(ticks_per_beat=960)
    trk = mido.MidiTrack()
    mid.tracks.append(trk)
    trk.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    evs = []
    for t, voice, vel in tr.hits:
        a = int(round(t * 1920))
        n = ACOUSTIC_NOTES[voice]
        evs += [(a, 1, mido.Message("note_on", channel=0, note=n, velocity=vel)),
                (a + 240, 0, mido.Message("note_off", channel=0, note=n, velocity=0))]
    evs.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for at, _, m in evs:
        m.time = at - last
        last = at
        trk.append(m)
    with tempfile.TemporaryDirectory() as d:
        mp, wp = Path(d) / "k.mid", Path(d) / "k.wav"
        mid.save(mp)
        subprocess.run([tool("fluidsynth"), "-ni", "-R", "0", "-C", "0", "-g", "0.8", "-F", str(wp), "-r", str(SR),
                        str(sf2), str(mp)], check=True, capture_output=True, timeout=600)
        a, _ = sf.read(wp, always_2d=True)
    return _fit(a.T.astype(np.float64), total)


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


def check_chain(chain) -> None:
    """Every effect in a chain exists and every setting names a real knob (SongError says which)."""
    if not isinstance(chain, list) or not chain:
        raise SongError("a chain is a list of effects")
    import inspect
    import pedalboard as pb
    built = {"distortion": pb.Distortion, "compressor": pb.Compressor, "reverb": pb.Reverb, "chorus": pb.Chorus,
             "delay": pb.Delay, "gain": pb.Gain, "highpass": pb.HighpassFilter, "lowpass": pb.LowpassFilter,
             "noisegate": pb.NoiseGate, "convolution": pb.Convolution, "phaser": pb.Phaser, "bitcrush": pb.Bitcrush,
             "ladderfilter": pb.LadderFilter}
    for i, fx in enumerate(chain, 1):
        if not isinstance(fx, dict):
            raise SongError(f"effect {i} isn't an object")
        if "builtin" in fx:
            kind = str(fx["builtin"]).lower()
            if kind not in built:
                raise SongError(f"effect {i}: no built-in {kind!r}; built-ins: {', '.join(built)} "
                                "(an EQ is highpass/lowpass, or the amp's own tone knobs)")
            try:
                ok = set(inspect.signature(built[kind].__init__).parameters) - {"self"}
            except (TypeError, ValueError):
                ok = None
            bad = set(fx) - {"builtin"} - (ok or set(fx))
            if bad:
                raise SongError(f"effect {i} ({kind}) has no {', '.join(sorted(bad))}; it takes {', '.join(sorted(ok))}")
        elif "plugin" in fx:
            name = str(fx["plugin"])
            path = name if os.path.exists(name) else next(
                (str(Path(d) / EFFECTS.get(name.lower(), name)) for d in PLUGIN_DIRS
                 if (Path(d) / EFFECTS.get(name.lower(), name)).exists()), None)
            if not path:
                raise SongError(f"effect {i}: no plugin {name!r} (known: {', '.join(EFFECTS)}, or a built-in)")
            from pedalboard import load_plugin
            try:
                _quick_load()
                _set_params(load_plugin(path), fx.get("params"), path)
            except SongError as e:
                raise SongError(f"effect {i} ({name}): {e}".replace("the synth", "it"))
        else:
            raise SongError(f"effect {i} needs plugin (a VST3 like swankyamp) or builtin")


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
            _quick_load()
            eff = load_plugin(path)
            _set_params(eff, fx.get("params"), path)
        out = eff(out, SR, reset=True)
        if out.shape[0] == 1:
            out = np.vstack([out, out])
    return out.astype(np.float64)[:2]


def _pan(st: np.ndarray, pan: float) -> np.ndarray:
    th = (float(pan) + 1) * np.pi / 4
    return np.vstack([st[0] * np.cos(th) * np.sqrt(2), st[1] * np.sin(th) * np.sqrt(2)])


def _rooms(song: Dict) -> List:
    """Every space a song plays in: the whole song's, and each instrument's."""
    out = [song["space"]] if song.get("space") else []
    return out + [spec["space"] for spec in (song.get("instruments") or {}).values() if (spec or {}).get("space")]


def render(song_path: Path, out_dir: Path, tail: float = 2.0, log=print, base: Optional[Path] = None) -> Dict:
    from . import spaces
    from . import mix as mix_ears
    song = yaml.safe_load(Path(song_path).read_text())
    tracks, length, marks = parse(song, Path(base or Path(song_path).parent))
    try:
        rings = [spaces.ring_s(r) for r in _rooms(song)]
    except spaces.SpaceError as e:
        raise SongError(str(e))
    tail = max([tail] + [min(r, 8.0) for r in rings])   # let the longest room finish ringing
    total = length + tail
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stems").mkdir(exist_ok=True)
    mix = np.zeros((2, int(total * SR)))
    info = {}
    stems_mono = {}
    heard = {}
    for name, tr in tracks.items():
        if tr.spec.get("kit"):
            a = render_kit(tr, total)
            kind = f"{tr.spec.get('kit')} kit"
        else:
            a, kind = render_source(tr, total)
        a = apply_fx(a, resolve_chain(tr.spec))
        if tr.spec.get("space"):
            a = spaces.apply(a, tr.spec["space"])
            kind += f", in {tr.spec['space'] if isinstance(tr.spec['space'], str) else 'its own space'}"
        n = mix.shape[1]
        a = a[:, :n] if a.shape[1] >= n else np.pad(a, ((0, 0), (0, n - a.shape[1])))
        a = _pan(a * float(tr.spec.get("gain", 0.8)), tr.spec.get("pan", 0.0))
        sf.write(out_dir / "stems" / f"{name}.wav", a.T, SR, subtype="PCM_24")
        stems_mono[name] = a.mean(axis=0)
        heard[name] = mix_ears.bands(a)
        mix += a
        mono = a.mean(axis=0)
        sounding = mono[np.abs(mono) > 1e-4]
        info[name] = {"plays": kind, "notes": len(tr.events), "hits": len(tr.hits),
                      "peak_db": round(float(20 * np.log10(np.abs(a).max() + 1e-12)), 1),
                      "loudness_db": round(float(20 * np.log10(np.sqrt(np.mean(sounding ** 2)) + 1e-12)), 1)
                      if len(sounding) else None}
        log(f"  {name}: {kind}, {len(tr.events) or len(tr.hits)} {'notes' if tr.events else 'hits'}")
    if song.get("space"):
        mix = spaces.apply(mix, song["space"])   # the whole band in one room (the stems are dry of it)
    peak = np.abs(mix).max()
    clipped = bool(peak > 1.0)
    if peak > 0:
        mix = mix / peak * 0.89   # about -1 dBFS
    sf.write(out_dir / "render.wav", mix.T, SR, subtype="PCM_24")
    loud = {n: i["loudness_db"] for n, i in info.items() if i.get("loudness_db") is not None}
    top = max(loud.values()) if loud else 0
    balance = [f"{n}: {v - top:+.1f} dB" for n, v in sorted(loud.items(), key=lambda kv: -kv[1])]
    # the song by section: what plays in each, and how loud each section is against the loudest
    def _db(x):
        return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)) if len(x) else -120.0
    spb_ = 60.0 / float(song["tempo"])
    bar_s = int(song.get("meter", 4)) * spb_
    table = []
    for i, (sec, t0) in enumerate(marks):
        t1 = marks[i + 1][1] if i + 1 < len(marks) else length
        a0, a1 = int(t0 * SR), int(t1 * SR)
        plays = [n for n, m in stems_mono.items() if _db(m[a0:a1]) > -60]
        table.append({"section": sec, "bars": int(round((t1 - t0) / bar_s)), "plays": plays,
                      "loudness_db": round(_db(mix.mean(axis=0)[a0:a1]), 1)})
    if table:
        top_s = max(r["loudness_db"] for r in table)
        for r in table:
            r["vs_loudest_db"] = round(r["loudness_db"] - top_s, 1)
    result = {"title": song.get("title", Path(song_path).stem), "tempo": song["tempo"], "length_s": round(length, 2),
              "balance": balance, "by_section": table,
              "the_mix": mix_ears.hear(heard, marks, length, song.get("lead")),
              **({"clips_not_written_yet": list(MISSING_CLIPS)} if MISSING_CLIPS else {}),
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
