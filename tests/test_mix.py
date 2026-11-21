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


# --- instruments written as a list (Cass's first try, 2026-11-21) ---------------------------

def test_instruments_as_a_list_with_names_play_and_lead_true_names_the_lead():
    from orphic.song import parse
    s = {"tempo": 120, "instruments": [{"name": "melody", "gm": 1, "lead": True}, {"name": "bass", "gm": 34}],
         "sections": {"a": {"bars": 1, "melody": "C4/4", "bass": "C2/4"}}}
    tracks, _, _ = parse(s)
    assert set(tracks) == {"melody", "bass"} and s["lead"] == "melody"


@pytest.mark.parametrize("ins,says", [([{"gm": 1}], "needs a name"), ("piano", "written by name"),
                                      ([{"name": "a", "gm": 1}, {"name": "a", "gm": 2}], "two instruments")])
def test_badly_written_instruments_say_how_to_write_them(ins, says):
    from orphic.song import SongError, parse
    with pytest.raises(SongError, match=says):
        parse({"tempo": 120, "instruments": ins, "sections": {}})


def test_sections_as_a_list_play_in_that_order():
    from orphic.song import parse
    s = {"tempo": 120, "instruments": [{"name": "keys", "gm": 1}],
         "sections": [{"name": "verse", "bars": 1, "keys": "C/4"}, {"name": "chorus", "bars": 1, "keys": "F/4"}]}
    _, _, marks = parse(s)
    assert [m[0] for m in marks] == ["verse", "chorus"]


@pytest.mark.parametrize("song,says", [
    ({"sections": [{"bars": 1}]}, "needs a name"),
    ({"sections": "verse"}, "written by name"),
    ({"sections": {"verse": "C/4"}}, "set of keys"),
    ({"sections": {"verse": {"bars": 1}}, "arrangement": "verse"}, "list of section names"),
])
def test_badly_written_sections_say_how(song, says):
    from orphic.song import SongError, parse
    with pytest.raises(SongError, match=says):
        parse({"tempo": 120, "instruments": {"keys": {"gm": 1}}, **song})


def test_audition_reads_sections_as_a_list():
    from orphic.audition import window
    s = {"tempo": 120, "instruments": {"keys": {"gm": 1}},
         "sections": [{"name": "a", "bars": 1, "keys": "C/4"}, {"name": "b", "bars": 1, "keys": "F/4"}]}
    w, after = window(s, "a")
    assert w["arrangement"] == ["a", "b"]


@pytest.mark.parametrize("path,val", [
    ("instruments.keys", "piano"), ("instruments.keys.gm", "one"), ("instruments.keys.fx", "reverb"),
    ("instruments.keys.gain", "loud"), ("instruments.keys.pan", "left"), ("sections.a.bars", "eight"),
    ("sections.a.bars", None), ("sections.a.bars", -2), ("arrangement", [["a"]]), ("meter", "four"), ("meter", 0),
])
def test_a_badly_written_song_is_a_plain_song_error_never_a_python_one(path, val):
    from orphic.song import SongError, parse
    s = {"tempo": 120, "instruments": {"keys": {"gm": 1}}, "sections": {"a": {"bars": 1, "keys": "C/4"}},
         "arrangement": ["a"]}
    d, ks = s, path.split(".")
    for k in ks[:-1]:
        d = d[k]
    d[ks[-1]] = val
    with pytest.raises(SongError):
        parse(s)


def test_note_values_written_where_beats_go_get_a_hint():
    from orphic.song import SongError, parse
    s = {"tempo": 120, "instruments": {"keys": {"gm": 1}}, "sections": {"a": {"bars": 1, "keys": "A4/4 G4/4 A4/2"}}}
    with pytest.raises(SongError, match=r"A4/1 G4/1 A4/2"):
        parse(s)
    s["sections"]["a"]["keys"] = "A4/3 G4/3"            # just too long: no hint
    with pytest.raises(SongError) as e:
        parse(s)
    assert "note value" not in str(e.value)


@pytest.mark.parametrize("rest", ["r/1", "./1", "-/1"])
def test_rests_written_any_of_the_usual_ways(rest):
    from orphic.song import parse
    tracks, _, _ = parse({"tempo": 120, "instruments": {"k": {"gm": 1}},
                          "sections": {"a": {"bars": 1, "k": f"{rest} A4/1 G4/2"}}})
    assert [round(e.start, 2) for e in tracks["k"].events] == [0.5, 1.0]


def test_a_rest_without_a_length_says_how():
    from orphic.song import SongError, parse
    with pytest.raises(SongError, match="r/1 is a beat of silence"):
        parse({"tempo": 120, "instruments": {"k": {"gm": 1}}, "sections": {"a": {"bars": 1, "k": ". A4/1 G4/2"}}})


def test_a_plugin_that_crashes_when_it_plays_is_a_plain_error(tmp_path, monkeypatch):
    from orphic import song as S
    f = tmp_path / "Crashy.vst3"
    f.write_text("x")
    monkeypatch.setattr(S, "SAFE_CACHE", tmp_path / "safe.json")
    monkeypatch.setattr(S, "_PROBE", "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)")
    with pytest.raises(S.SongError, match="crashes when it plays"):
        S._plugin_path(str(f))
    monkeypatch.setattr(S, "_PROBE", "pass")          # remembered: not tried again
    with pytest.raises(S.SongError, match="crashes when it plays"):
        S._plugin_path(str(f))
    f.write_text("xy")                                # a reinstalled plugin is tried afresh
    assert S._plugin_path(str(f)) == str(f)
