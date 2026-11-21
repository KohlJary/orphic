"""Hearing a mix (plan item edff9c3c): what covers what, measured from the stems."""
import numpy as np
import pytest

from orphic import mix

SR = mix.SR


def tone(hz, secs=4.0, amp=1.0, start=0.0):
    t = np.arange(int(SR * secs)) / SR
    x = amp * np.sin(2 * np.pi * hz * t) * (t >= start)
    return np.vstack([x, x])


def pan(a, left, right):
    return np.vstack([a[0] * left, a[1] * right])


MARKS = [("verse", 0.0), ("chorus", 2.0)]


def heard(parts, lead=None):
    return mix.hear({n: mix.bands(a) for n, a in parts.items()}, MARKS, 4.0, lead)


def test_a_quiet_part_under_a_loud_one_in_the_same_band_is_covered():
    r = heard({"melody": tone(220, amp=0.05), "bass": tone(230, amp=1.0)})
    assert r["covered"]["melody"] > 0.8
    assert r["detail"]["melody"]["by"][0] == "bass"
    lo, hi = r["detail"]["melody"]["hz"]
    assert lo <= 220 <= hi
    assert any(f.startswith("melody:") and "bass" in f for f in r["findings"])


def test_the_same_part_an_octave_away_isnt():
    r = heard({"melody": tone(1760, amp=0.05), "bass": tone(110, amp=1.0)})
    assert r["covered"]["melody"] < 0.1
    assert not any(f.startswith("melody:") for f in r["findings"])


def test_turning_it_up_uncovers_it():
    r = heard({"melody": tone(220, amp=1.0), "bass": tone(230, amp=0.5)})
    assert r["covered"]["melody"] < 0.1


def test_panning_apart_helps_like_it_does_in_a_room():
    r = heard({"melody": pan(tone(220, amp=0.05), 1.0, 0.0), "bass": pan(tone(230), 0.0, 1.0)})
    assert r["covered"]["melody"] < 0.1


def test_it_says_where_it_is_worst():
    bass = tone(230)
    bass[:, : 2 * SR] = 0                       # bass only comes in for the chorus
    r = heard({"melody": tone(220, amp=0.05), "bass": bass})
    assert r["detail"]["melody"]["worst"].startswith("chorus")


def test_the_lead_is_found_by_name_and_its_presence_band_judged():
    r = heard({"lead vox": tone(3000, amp=0.05), "hats": tone(3300, amp=1.0)})
    assert r["lead"] == "lead vox"
    assert any("presence band" in f and "crowded" in f and "hats" in f for f in r["findings"])


def test_the_song_can_name_its_lead():
    r = heard({"keys": tone(3000), "guitar": tone(500)}, lead="keys")
    assert r["lead"] == "keys"
    assert any("keys owns its presence band" in f for f in r["findings"])


def test_a_part_that_drops_out_of_the_mix_somewhere_is_named():
    keys = tone(1000, amp=1.0)
    keys[:, 2 * SR:] *= 0.05                      # 26 dB down in the chorus
    r = heard({"keys": keys, "bass": tone(80)})
    assert any(f.startswith("keys drops") and "chorus" in f for f in r["findings"])


def test_one_part_alone_has_nothing_to_cover_it():
    assert heard({"keys": tone(440)})["findings"] == []


def test_render_carries_the_mix(tmp_path, monkeypatch):
    import yaml
    from orphic.song import render
    monkeypatch.setenv("ORPHIC_BELT", str(tmp_path / "belt"))
    song = {"title": "t", "tempo": 120, "meter": 4,
            "instruments": {"melody": {"gm": 1, "gain": 0.05}, "bass": {"gm": 34, "gain": 1.0}},
            "sections": {"a": {"bars": 2, "melody": "C3/4 | E3/4", "bass": "C3/4 | E3/4"}},
            "arrangement": ["a"]}
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(song))
    res = render(p, tmp_path / "out", log=lambda *_: None)
    assert res["the_mix"]["covered"]["melody"] > 0.5


def clicks(times, secs=4.0, amp=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(int(SR * secs))
    for t in times:
        i = int(t * SR)
        x[i:i + 2000] = amp * rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300)
    return np.vstack([x, x])


def test_bright_attacks_landing_together_are_collisions():
    beats = [0.25 + 0.5 * i for i in range(7)]
    r = heard({"vocal": clicks(beats, seed=1), "hats": clicks(beats, seed=2)})
    assert any(f.startswith("vocal and hats") and "attacks land together" in f for f in r["findings"])


def test_attacks_off_each_other_arent():
    r = heard({"vocal": clicks([0.25 + 0.5 * i for i in range(7)], seed=1),
               "hats": clicks([0.5 + 0.5 * i for i in range(7)], seed=2)})
    assert not any("attacks land together" in f for f in r["findings"])
