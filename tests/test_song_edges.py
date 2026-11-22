"""Badly written songs get a plain SongError, never a Python one, and nothing is dropped without a word."""
import pytest

from orphic import song as S
from orphic.song import SongError, parse


def one(part="E2/1 G2/1 A2/1 B2/1", **top):
    s = {"tempo": 100, "instruments": {"bass": {"gm": 34}}, "sections": {"a": {"bars": 1, "bass": part}}}
    s.update(top)
    return s


@pytest.mark.parametrize("tempo", [0, -120, 100000, float("inf")])
def test_tempo_must_be_positive_and_playable(tempo):
    with pytest.raises(SongError, match="tempo"):
        parse(one(tempo=tempo))


@pytest.mark.parametrize("part", ["B9/4", "C-2/4"])
def test_note_outside_midi_range_is_a_song_error(part):
    with pytest.raises(SongError):
        parse(one(part))


@pytest.mark.parametrize("part", ["C4/-2", "C4/0"])
def test_negative_or_zero_length_is_a_song_error(part):
    with pytest.raises(SongError, match="more than 0 beats"):
        parse(one(part))


def test_flat_and_sharp_signs_read_as_b_and_hash():
    tracks, _, _ = parse(one("E♭4/2 F♯4/2"))
    assert [e.midi for e in tracks["bass"].events] == [63, 66]


def test_more_bars_written_than_the_section_has_says_so():
    parse(one("E2/4 | G2/4 | A2/4 | B2/4", sections={"a": {"bars": 2, "bass": "E2/4 | G2/4 | A2/4 | B2/4"}}))
    assert S.UNPLAYED_BARS and "bars 3-4 don't play" in S.UNPLAYED_BARS[0]
    parse(one())
    assert S.UNPLAYED_BARS == []


def test_a_fractional_meter_and_an_empty_song_are_song_errors():
    with pytest.raises(SongError, match="meter"):
        parse(one(meter=2.5))
    with pytest.raises(SongError, match="at least one section"):
        parse({"tempo": 100, "instruments": {}, "sections": {}})
