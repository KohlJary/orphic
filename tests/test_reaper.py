"""Reaper sessions: spaces carried over, and synth settings found without the slow knob scan."""
import shutil
import time

import numpy as np
import pytest
import soundfile as sf
import yaml

from orphic import reaper, spaces
from orphic.song import PLUGIN_DIRS
from pathlib import Path

HAS_REAPER = bool(shutil.which("reaper") and shutil.which("xvfb-run"))
NEKOBI = next((str(Path(d) / "Nekobi.vst3") for d in PLUGIN_DIRS if (Path(d) / "Nekobi.vst3").exists()), None)


def test_space_fx_sliders_are_the_jsfx_sliders_in_order():
    sliders = [l.split("=")[0].split(":")[1] for l in (reaper.HERE / "jsfx" / "orphic_space").read_text().splitlines()
               if l.startswith("slider")]
    assert sliders == reaper.SPACE_SLIDERS
    fx = reaper.space_fx({"name": "hall", "amount": 0.1})
    assert fx["name"] == "JS:orphic/orphic_space"
    hall = spaces.BUILT_IN["hall"]
    assert fx["values"] == [0.1 if k == "amount" else float(hall[k]) for k in reaper.SPACE_SLIDERS]


def _lead(tmp_path, space):
    s = {"title": "t", "tempo": 100, "meter": 4,
         "instruments": {"lead": {"synth": "nekobi", **({"space": space} if space else {})}},
         "sections": {"a": {"bars": 2, "lead": "C4/1 E4/1 G4/2 | A4/4"}}, "arrangement": ["a"]}
    p = tmp_path / f"{space or 'dry'}.yaml"
    p.write_text(yaml.safe_dump(s))
    r = reaper.build(p, tmp_path / f"out-{space or 'dry'}", log=lambda *_: None)
    return sf.read(r["render"], always_2d=True)[0]


@pytest.mark.skipif(not (HAS_REAPER and NEKOBI), reason="needs Reaper and Nekobi")
def test_a_live_track_plays_its_room_in_reaper(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHIC_REAPER_CFG", str(tmp_path / "cfg"))
    wet, dry = _lead(tmp_path, "hall"), _lead(tmp_path, None)
    n = min(len(wet), len(dry))
    room = wet[:n] - dry[:n]
    want = spaces.apply(dry[:n].T, "hall").T - dry[:n]          # the room Orphic makes of the same take
    got_db = 10 * np.log10((room ** 2).sum() / (dry ** 2).sum())
    want_db = 10 * np.log10((want ** 2).sum() / (dry ** 2).sum())
    assert abs(got_db - want_db) < 1.0


@pytest.mark.skipif(not NEKOBI, reason="needs Nekobi")
def test_reaper_sets_knobs_without_scanning_every_one():
    reaper._normalized(NEKOBI, {"cutoff": 40.0})                 # warm the plugin and the cache
    t = time.time()
    vals, name = reaper._normalized(NEKOBI, {"cutoff": 40.0})
    assert time.time() - t < 3.0                                 # the full scan of Nekobi's 2090 knobs is ~10 s
    assert name == "Nekobi" and vals["cutoff"] == pytest.approx(0.4, abs=1e-3)
