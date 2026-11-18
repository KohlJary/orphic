"""Noodling: pick up an instrument and play a riff, quickly, with nothing committed.

A riff is a bar or a few, in the song notation (or a drum grid for the kit), looped.
It's rendered through the same instruments as a song, and what was heard comes back
fast: the notes (pYIN, for a melodic riff) and the sound in the vocabulary of a synth
map (brightness, noise, attack, sustain, width...). Turning a knob between tries shows
up as measured differences.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import soundfile as sf
import yaml

from .measure import LABELS, measure
from .song import SR, render


def noodle(instrument: Dict, phrase, tempo: float = 100.0, loops: int = 2, meter: int = 4,
           out_dir: Optional[Path] = None) -> Dict:
    bars = len(str(phrase).split("|")) if not isinstance(phrase, dict) else \
        max(len(str(v).split("|")) for v in phrase.values())
    song = {"title": "noodle", "tempo": tempo, "meter": meter, "instruments": {"riff": instrument},
            "sections": {"riff": {"bars": bars * max(1, int(loops)), "riff": phrase}}, "arrangement": ["riff"]}
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="noodle-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "noodle.yaml").write_text(yaml.safe_dump(song, sort_keys=False))
    res = render(out_dir / "noodle.yaml", out_dir, tail=1.0, log=lambda *_: None)
    y, _ = sf.read(out_dir / "render.wav", always_2d=True)
    a = y.T
    one_bar = meter * 60.0 / tempo
    sound = measure(a, hold=min(res["length_s"], one_bar * bars))
    if not instrument.get("kit"):
        # a riff has many notes: measures of one pitch (pitch, wobble, harmonic share) don't apply
        for k in ("f0_hz", "pitch_wobble_cents", "harmonicity", "odd_even"):
            sound[k] = None
    heard = {LABELS[k].split(" (")[0]: v for k, v in sound.items() if v is not None}
    notes = []
    if not instrument.get("kit"):
        mono = librosa.resample(a.mean(axis=0), orig_sr=SR, target_sr=22050)[: int(22050 * one_bar * bars)]
        f0, voiced, _ = librosa.pyin(mono, fmin=40, fmax=1500, sr=22050, frame_length=2048, hop_length=512)
        t = librosa.times_like(f0, sr=22050, hop_length=512)
        runs = []   # (midi, start, end): a pitch held over consecutive voiced frames
        for ti, f, v in zip(t, f0, voiced):
            if not v or np.isnan(f):
                runs.append(None)
                continue
            m = int(round(librosa.hz_to_midi(f)))
            if runs and runs[-1] and runs[-1][0] == m:
                runs[-1][2] = ti
            else:
                runs.append([m, ti, ti])
        hop_t = 512 / 22050
        last = None
        for r in runs:
            # a glide between notes passes through pitches for a frame or two: not notes
            if r is None or r[2] - r[1] + hop_t < 0.06:
                if r is None:
                    last = None
                continue
            if r[0] != last:
                notes.append(f"{librosa.midi_to_note(r[0], unicode=False)}@{r[1]:.2f}s")
            last = r[0]
    return {"audio": str(out_dir / "render.wav"), "length_s": res["length_s"], "sound": heard,
            "sound_raw": sound, "notes_heard": notes[:48],
            "would_have_clipped": res["would_have_clipped"]}


def differences(before: Dict, after: Dict) -> list:
    """What a knob turn did: the measures that moved clearly between two tries."""
    from .measure import change
    ch = change(before, after)
    out = []
    for k, v in sorted(ch.items(), key=lambda kv: -kv[1]):
        if v >= 1.0:
            out.append(f"{LABELS[k]}: {before.get(k)} -> {after.get(k)}")
    return out or ["no clear change in the sound"]
