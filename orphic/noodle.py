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
        # a riff has many notes and gaps between them: measures of one held note (pitch,
        # attack, sustain, release, tremolo...) don't apply, and spectral ones taken over the
        # whole loop measured the silence (Cass, 2026-11-18: "brightness at 3.6 Hz is
        # subsonic"). Brightness, noise, width and the peak come from the sounding parts only.
        sound.update(_riff_sound(a[:, : int(SR * one_bar * bars)]))
    names = {**LABELS, "sounding_share": "share of the loop that sounds", "rings_ms": "notes ring for (ms, median)"}
    heard = {names[k].split(" (")[0] if k not in ("sounding_share", "rings_ms") else names[k]: v
             for k, v in sound.items() if v is not None}
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
    from .song import bar_fill
    played = bar_fill(phrase, meter) if not isinstance(phrase, dict) else None
    return {"audio": str(out_dir / "render.wav"), "length_s": res["length_s"], "sound": heard,
            "sound_raw": sound, "played": played, "notes_heard": notes[:48],
            "notes_heard_is": ("a pitch tracker's guess at the audio, for checking by ear; it misses or misreads "
                               "short, palm-muted and distorted notes. 'played' is what was actually played."),
            "would_have_clipped": res["would_have_clipped"]}


def _riff_sound(a: np.ndarray) -> Dict:
    y = a.mean(axis=0)
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    out = {k: None for k in ("f0_hz", "pitch_wobble_cents", "harmonicity", "odd_even", "attack_ms", "sustain",
                             "release_ms", "brightness_motion", "am_rate_hz", "am_depth", "noise", "peak_db")}
    if rms.max() < 1e-5:
        return out
    on = rms > rms.max() * 10 ** (-35 / 20)       # sounding: within 35 dB of the loudest moment
    out["sounding_share"] = round(float(on.mean()), 3)
    runs, n = [], 0
    for v in list(on) + [False]:
        if v:
            n += 1
        elif n:
            runs.append(n * hop / SR * 1000)
            n = 0
    out["rings_ms"] = round(float(np.median(runs)), 1) if runs else None
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))[:, : len(on)]
    Sa = S[:, on[: S.shape[1]]]
    if Sa.shape[1]:
        out["brightness_hz"] = round(float(np.median(librosa.feature.spectral_centroid(S=Sa, sr=SR)[0])), 1)
        f = librosa.fft_frequencies(sr=SR, n_fft=2048)
        spec = Sa.mean(axis=1)
        sel = f > 60
        i = int(np.argmax(spec[sel]))
        out["peak_hz"] = round(float(f[sel][i]), 1)
        # (no peak height or noise share for a riff: a rig's low-pass and a chord's many
        # partials make both read as extremes that mean nothing)
    m, s_ = (a[0] + a[1]) / 2, (a[0] - a[1]) / 2
    out["width"] = round(float((s_ ** 2).sum() / ((m ** 2).sum() + (s_ ** 2).sum() + 1e-20)), 3)
    return out


def differences(before: Dict, after: Dict) -> list:
    """What a knob turn did: the measures that moved clearly between two tries."""
    from .measure import change
    ch = change(before, after)
    out = []
    for k, v in sorted(ch.items(), key=lambda kv: -kv[1]):
        if v >= 1.0:
            out.append(f"{LABELS[k]}: {before.get(k)} -> {after.get(k)}")
    return out or ["no clear change in the sound"]
