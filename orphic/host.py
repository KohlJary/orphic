"""Hosting a plugin synth without a DAW (pedalboard, VST3), with its defaults kept so
every render starts from the same patch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from mido import Message

SR = 44100


@dataclass
class Probe:
    name: str
    notes: List[int]
    hold: float
    tail: float
    velocity: int = 100

    def messages(self):
        on = [Message("note_on", note=n, velocity=self.velocity, time=0.0) for n in self.notes]
        off = [Message("note_off", note=n, velocity=0, time=self.hold) for n in self.notes]
        return on + off

    @property
    def duration(self):
        return self.hold + self.tail


HELD = Probe("held C4", [60], hold=1.2, tail=0.8)
LOW = Probe("staccato C2", [36], hold=0.15, tail=0.6)


class Synth:
    def __init__(self, path: str):
        from pedalboard import load_plugin
        self.path = path
        self.plugin = load_plugin(path)
        if not self.plugin.is_instrument:
            raise ValueError(f"{path} isn't an instrument")
        self.params = dict(self.plugin.parameters)
        self.defaults: Dict[str, float] = {k: p.raw_value for k, p in self.params.items()}
        self._dirty = set()
        self.channels = self._channels()

    def _channels(self):
        for ch in (2, 1):
            try:
                self.plugin(HELD.messages(), duration=0.2, sample_rate=SR, num_channels=ch)
                return ch
            except ValueError:
                continue
        raise ValueError("no 1- or 2-channel output")

    def restore(self):
        """Back to the default patch: only what was changed (a synth can expose thousands
        of parameters, MIDI CC slots among them)."""
        for k in list(self._dirty):
            try:
                self.params[k].raw_value = self.defaults[k]
            except Exception:
                pass
        self._dirty.clear()

    def render(self, probe: Probe, settings: Dict[str, float] = None) -> np.ndarray:
        """(2, n) audio for a probe, from the default patch plus `settings` (raw 0-1)."""
        self.restore()
        for k, v in (settings or {}).items():
            self.params[k].raw_value = float(v)
            self._dirty.add(k)
        try:
            self.plugin.reset()
        except Exception:
            pass
        a = self.plugin(probe.messages(), duration=probe.duration, sample_rate=SR, num_channels=self.channels,
                        reset=True)
        a = np.asarray(a, dtype=np.float64)
        return np.vstack([a, a]) if a.shape[0] == 1 else a[:2]
