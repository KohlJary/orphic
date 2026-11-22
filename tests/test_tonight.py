"""hold, spaces and audition (Cass's asks, 2026-11-20), and the quick plugin load."""
import json

import numpy as np
import pytest
import yaml

from orphic import spaces
from orphic.audition import window
from orphic.song import KIT_DIR, SongError, parse

needs_808 = pytest.mark.skipif(not KIT_DIR.is_dir(), reason="needs the 808 samples ($ORPHIC_808)")

SONG = {
    "title": "t", "tempo": 120, "meter": 4,
    "instruments": {"keys": {"gm": 1}, "drums": {"kit": "808"}},
    "sections": {
        "a": {"bars": 1, "keys": "C/2 G/2", "drums": {"kick": "x...x...x...x..."}},
        "b": {"bars": 1, "keys": "Am/4"},
        "c": {"bars": 2, "keys": "F/4 | G/4"},
    },
    "arrangement": ["a", "b", "a", "b", "c"],
}


def _song(**changes):
    s = json.loads(json.dumps(SONG))
    for sec, extra in changes.items():
        s["sections"][sec].update(extra)
    return s


# --- hold ---------------------------------------------------------------------

def test_hold_delays_the_next_section_by_its_beats():
    _, length, marks = parse(_song())
    _, held, held_marks = parse(_song(a={"hold": 3}))
    assert held - length == pytest.approx(2 * 3 * 0.5)     # a plays twice, 3 beats at 120 bpm each
    assert dict(held_marks)  # sections still marked
    assert held_marks[1][1] - marks[1][1] == pytest.approx(1.5)


def test_hold_rings_the_notes_sounding_at_the_end():
    tracks, _, _ = parse(_song(a={"hold": 2}))
    first_a = [e for e in tracks["keys"].events if e.start < 2.0]
    last = [e for e in first_a if e.start >= 1.0]            # the G/2 that ends the bar
    assert last and all(e.start + e.dur == pytest.approx(2.0 + 1.0) for e in last)
    early = [e for e in first_a if e.start < 1.0]            # the C/2 ends mid-bar: untouched
    assert all(e.start + e.dur < 1.0 for e in early)


def test_hold_starts_nothing_new_while_ringing():
    tracks, _, marks = parse(_song(a={"hold": 2}))
    b_start = marks[1][1]
    assert b_start == pytest.approx(2.0 + 1.0)
    assert not [h for h in tracks["drums"].hits if 2.0 <= h[0] < b_start]
    assert not [e for e in tracks["keys"].events if 2.0 <= e.start < b_start]


@pytest.mark.parametrize("bad", [-1, "long"])
def test_hold_must_be_a_number_of_beats(bad):
    with pytest.raises(SongError):
        parse(_song(a={"hold": bad}))


# --- audition -----------------------------------------------------------------

def test_audition_window_is_before_section_after():
    w, after = window(_song(), "b", time=2)
    assert w["arrangement"] == ["a", "b", "c"]
    assert after == "c"


def test_audition_first_section_has_nothing_before_and_last_nothing_after():
    w, after = window(_song(), "a", time=1)
    assert w["arrangement"] == ["a", "b"]
    w, after = window(_song(), "c")
    assert w["arrangement"] == ["b", "c"] and after is None


def test_audition_repeat_plays_it_over():
    w, _ = window(_song(), "b", time=1, repeat=5)
    assert w["arrangement"] == ["a"] + ["b"] * 5 + ["a"]


def test_audition_swap_changes_only_this_audition():
    s = _song()
    w, _ = window(s, "b", time=1, swap={"keys": "C/4"})
    assert w["arrangement"] == ["a", "b (trying)", "a"]
    assert w["sections"]["b (trying)"]["keys"] == "C/4"
    assert w["sections"]["b"]["keys"] == "Am/4"
    assert s["sections"]["b"]["keys"] == "Am/4"           # the song itself is untouched
    parse(w)                                               # and the window is a song that plays


def test_audition_swap_can_try_a_hold():
    w, _ = window(_song(), "b", swap={"hold": 2})
    assert w["sections"]["b (trying)"]["hold"] == 2


@pytest.mark.parametrize("kw,says", [
    ({"section": "x"}, "no section"),
    ({"section": "b", "time": 3}, "comes around 2 times"),
    ({"section": "b", "repeat": 9}, "repeat"),
    ({"section": "b", "swap": {"tuba": "C/4"}}, "isn't an instrument"),
])
def test_audition_errors_say_what_to_do(kw, says):
    with pytest.raises(SongError, match=says):
        window(_song(), **kw)


@needs_808
def test_audition_renders_only_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHIC_BELT", str(tmp_path / "belt"))
    from orphic.audition import audition
    p = tmp_path / "song.yaml"
    p.write_text(yaml.safe_dump(_song()))
    res = audition(p, tmp_path / "out", "b", time=1, log=lambda *_: None)
    assert res["window"] == ["a", "b", "a"]
    # a (2 s) + b (2 s) + the first bar of the next a (2 s) + a second of ring
    assert res["length_s"] == pytest.approx(7.0, abs=0.05)
    assert (tmp_path / "out" / "render.wav").exists()
    assert yaml.safe_load(p.read_text()) == _song()        # nothing written to the song


# --- spaces -------------------------------------------------------------------

def test_built_in_spaces_ring_in_order_of_size():
    rings = [spaces.ring_s(n) for n in ("closet", "room", "high-ceilings", "cathedral")]
    assert rings == sorted(rings) and rings[0] < 1.0 < rings[-1]


def test_a_space_with_changes():
    sp = spaces.get({"name": "hall", "amount": 0.1})
    assert sp["amount"] == 0.1 and sp["size"] == spaces.BUILT_IN["hall"]["size"]


def test_unknown_space_lists_the_rooms():
    with pytest.raises(spaces.SpaceError, match="cathedral"):
        spaces.get("ballroom")


def test_out_of_range_says_the_range():
    with pytest.raises(spaces.SpaceError, match="from 0 to 1"):
        spaces.get({"name": "hall", "size": 3})


def test_her_own_space_in_her_words(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHIC_BELT", str(tmp_path))
    saved = spaces.save_space("Attic", "under the roof, wood and dust",
                              {"size": 0.4, "damping": 0.6, "walls_ms": 10, "amount": 0.2, "tone_hz": 6000,
                               "width": 0.6})
    assert saved["name"] == "attic" and saved["ring_s"] > 0
    assert spaces.get("attic")["size"] == 0.4
    assert spaces.listing()["attic"]["mine"] is True


def test_her_space_needs_her_words(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHIC_BELT", str(tmp_path))
    with pytest.raises(spaces.SpaceError, match="your words"):
        spaces.save_space("attic", "", {})


def test_her_space_can_start_from_a_built_in(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHIC_BELT", str(tmp_path))
    saved = spaces.save_space("cathedral", "my cathedral, closer", {"walls_ms": 20})
    assert saved["walls_ms"] == 20 and saved["size"] == spaces.BUILT_IN["cathedral"]["size"]


def test_a_space_adds_the_room_and_keeps_the_sound():
    a = np.zeros((2, 44100))
    a[:, 100] = 1.0
    out = spaces.apply(a, "hall")
    assert out[0, 100] == pytest.approx(1.0, abs=0.05)     # the dry sound is still there
    assert np.abs(out[:, 44100 // 2:]).sum() > 0            # and the room is ringing half a second on


def test_unknown_space_in_a_song_is_a_song_error(tmp_path):
    from orphic.song import render
    s = _song()
    s["instruments"]["keys"]["space"] = "ballroom"
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(s))
    with pytest.raises(SongError, match="no space"):
        render(p, tmp_path / "out", log=lambda *_: None)


# --- the quick plugin load ----------------------------------------------------

def test_param_cache_is_keyed_by_plugin_file(tmp_path, monkeypatch):
    from orphic import song as S
    f = tmp_path / "a.vst3"
    f.write_text("x")
    _, k1 = S._param_cache(str(f))
    f.write_text("xy")
    _, k2 = S._param_cache(str(f))
    assert k1 != k2 and str(f) in k1


def test_her_space_fills_what_she_leaves_out_from_room_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHIC_BELT", str(tmp_path))
    saved = spaces.save_space("attic", "under the roof", {"size": 0.4})
    assert saved["size"] == 0.4 and saved["damping"] == spaces.BUILT_IN["room"]["damping"]
    assert "damping" in saved["from_room"] and "size" not in saved["from_room"]


def test_audition_swap_can_change_how_a_part_sounds():
    s = _song()
    w, _ = window(s, "b", time=1, swap={"instruments": {"keys": {"gain": 0.3}}})
    assert w["instruments"]["keys"] == {"gm": 1, "gain": 0.3}
    assert s["instruments"]["keys"] == {"gm": 1}                      # the song is untouched
    w, _ = window(s, "b", swap={"instruments": {"keys": {"synth": "vitalium"}}})
    assert w["instruments"]["keys"] == {"synth": "vitalium"}           # a new source replaces gm
    with pytest.raises(SongError, match="isn't an instrument"):
        window(s, "b", swap={"instruments": {"tuba": {"gain": 1}}})


@needs_808
def test_audition_mix_hears_only_the_crop(tmp_path, monkeypatch):
    """the_mix covers what the audition plays: not the rest of the next section after the cut."""
    monkeypatch.setenv("ORPHIC_BELT", str(tmp_path / "belt"))
    from orphic.audition import audition
    s = _song()
    s["instruments"]["pad"] = {"gm": 89}
    s["sections"]["c"] = {"bars": 4, "keys": "F/4 | G/4 | F/4 | G/4", "pad": "r/4 | r/4 | F/4 | G/4"}
    p = tmp_path / "song.yaml"
    p.write_text(yaml.safe_dump(s))
    res = audition(p, tmp_path / "out", "b", time=2, log=lambda *_: None)   # window: a, b, c; c cut after 2 bars
    assert res["window"] == ["a", "b", "c"]
    assert "pad" not in res["the_mix"]["covered"]                # the pad comes in after the cut
    assert "keys" in res["the_mix"]["covered"]


@pytest.mark.parametrize("two", [{"gm": 1, "synth": "vitalium"}, {"gm": 1, "sfz": "x"}, {"sfz": "x", "patch": "p"}])
def test_two_sound_sources_on_one_instrument_say_so(two):
    s = _song()
    s["instruments"]["keys"] = two
    with pytest.raises(SongError, match="has both"):
        parse(s)


def test_a_patch_on_its_synth_is_one_source():
    s = _song()
    s["instruments"]["keys"] = {"synth": "vitalium", "patch": "warm"}
    parse(s)
