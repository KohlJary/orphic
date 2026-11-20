"""A song as a Reaper session: tracks, her instruments and settings, her MIDI, section markers.

Reaper builds the project itself from a Lua script (so plugins carry their real state),
headless on a virtual display, with Orphic's own Reaper settings (orphic/reaper-cfg), then
saves session.rpp and renders it. Kohl can open the .rpp on his desktop and play it.

Per track:
- live: a synth (vitalium, nekobi, surge, ...) whose effects are all VST3 plays in Reaper
  from her MIDI, with her settings (patch or params), set as the same normalized values.
- printed: anything Reaper can't play the same way yet (the sampled guitar and its palm
  mutes, General MIDI, the 808 kit, pedalboard's built-in effects) goes in as the audio
  Orphic renders for it.
Every part also gets its notes as MIDI on a muted "notes" lane under the track, so what
was played is there to see and edit.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import yaml

HERE = Path(__file__).resolve().parent
CFG_TEMPLATE = HERE.parent / "reaper-cfg"
VST_PATHS = os.environ.get("ORPHIC_VST_PATHS", f"{Path.home()}/.vst;{Path.home()}/.vst3;/usr/lib/vst3;/usr/lib/vst")
DRUM_NOTES = {"kick": 36, "rim": 37, "snare": 38, "clap": 39, "low_tom": 41, "hat": 42, "mid_tom": 45,
              "open": 46, "high_tom": 48, "cymbal": 49, "cowbell": 56, "maracas": 70, "clave": 75}


class ReaperError(RuntimeError):
    pass


# sfizz 1.2's saved state with no file loaded (as Reaper keeps it: a length, the
# component state, then the controller state). The SFZ path goes in as a length-prefixed
# string at byte 12 of the component state.
SFIZZ_EMPTY = ("WgAAAAEAAAAFAAAAAAAAAAEAAAAAAAAAAEAAAAAAAAAAACAAAAEAAAAAPAAAAAAA3EMAAAAAAgAAAAEAAAAKAAAAAwAAAAAA"
               "/////wMAAAAHACaTST8KAAAAAD8LAAAAgD8AAAAAAAAAAA==")
# pedalboard built-ins played in Reaper by Orphic's JSFX (orphic/jsfx), sliders in this order
JSFX = {"gain": ["gain_db"], "highpass": ["cutoff_frequency_hz"], "lowpass": ["cutoff_frequency_hz"],
        "noisegate": ["threshold_db", "ratio", "attack_ms", "release_ms"],
        "compressor": ["threshold_db", "ratio", "attack_ms", "release_ms"], "distortion": ["drive_db"],
        "delay": ["delay_seconds", "feedback", "mix"],
        "chorus": ["rate_hz", "depth", "centre_delay_ms", "feedback", "mix"]}
JSFX_DEFAULTS = {"gain": [1.0], "highpass": [50.0], "lowpass": [50.0], "noisegate": [-100.0, 10.0, 1.0, 100.0],
                 "compressor": [0.0, 1.0, 1.0, 100.0], "distortion": [25.0], "delay": [0.5, 0.0, 0.5],
                 "chorus": [1.0, 0.25, 7.0, 0.0, 0.5]}
PM_SFZ = "ampeg_decay=0.08 ampeg_sustain=0 ampeg_release=0.02 fil_type=lpf_2p cutoff=700 volume=15"


def sfizz_state(sfz: Path) -> str:
    import base64
    import struct
    b = base64.b64decode(SFIZZ_EMPTY)
    n = struct.unpack("<i", b[:4])[0]
    comp, rest = b[4:4 + n], b[4 + n:]
    path = str(sfz).encode() + b"\0"
    comp = comp[:12] + struct.pack("<I", len(path)) + path + comp[17:]
    return base64.b64encode(struct.pack("<i", len(comp)) + comp + rest).decode()


def guitar_sfz(orig: Path, out: Path, muted: bool = False) -> Path:
    """The sampled instrument, open or palm-muted (damped fast, highs rolled off, as Orphic's pm).
    sfizz's VST3 ignores MIDI channels, so the two are separate instruments on separate tracks."""
    # the instrument twice, once per channel; settings go on its own <global> line, since a
    # <global> header resets anything set above it (an #include under <master> lost them)
    text = orig.read_text(errors="replace")
    if "<global>" not in text:
        text = "<global>\n" + text
    out.write_text(f"<control> default_path={orig.parent}/\n"
                   + (text.replace("<global>", f"<global> {PM_SFZ}") if muted else text) + "\n")
    return out


def kit_sfz(spec: Dict, out: Path) -> Path:
    """Her 808 choices as an SFZ: each drum on its General MIDI note, one-shot."""
    from .song import KIT_DEFAULT, KIT_DIR
    choice = dict(KIT_DEFAULT)
    for v, smp in (spec.get("samples") or {}).items():
        choice[v] = smp if "/" in smp else f"{smp[:2]}/{smp}"
    lines = [f"<control> default_path={KIT_DIR}/", "<group> loop_mode=one_shot"]
    for voice, rel in choice.items():
        lines.append(f"<region> key={DRUM_NOTES.get(voice, 36)} sample={rel}.WAV amp_velcurve_127=1 amp_velcurve_1=0.2")
    out.write_text("\n".join(lines) + "\n")
    return out


def _cfg_dir() -> Path:
    """Orphic's own Reaper settings (never Kohl's): created once, plugin folders set."""
    d = Path(os.environ.get("ORPHIC_REAPER_CFG", Path.home() / "orphic" / "reaper-cfg"))
    ini = d / "reaper.ini"
    if not ini.exists():
        d.mkdir(parents=True, exist_ok=True)
        ini.write_text(f"[reaper]\nvstpath={VST_PATHS}\n")
    else:
        s = ini.read_text()
        if "vstpath=" not in s:
            ini.write_text(s.replace("[reaper]\n", f"[reaper]\nvstpath={VST_PATHS}\n", 1))
    eff = d / "Effects" / "orphic"
    eff.mkdir(parents=True, exist_ok=True)
    for f in (HERE / "jsfx").iterdir():
        if f.is_file() and (not (eff / f.name).exists() or (eff / f.name).read_text() != f.read_text()):
            shutil.copy(f, eff / f.name)
    return d


def _lua_str(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _lua(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(float(v))
    if isinstance(v, str):
        return _lua_str(v)
    if isinstance(v, dict):
        return "{" + ", ".join(f"[{_lua_str(k)}]={_lua(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "{" + ", ".join(_lua(x) for x in v) + "}"
    return "nil"


def _write_midi(events, path: Path, tempo: float, meter: int, drums=None):
    import mido
    tpb = 960
    mid = mido.MidiFile(ticks_per_beat=tpb)
    trk = mido.MidiTrack()
    mid.tracks.append(trk)
    trk.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    trk.append(mido.MetaMessage("time_signature", numerator=meter, denominator=4, time=0))
    spb = 60.0 / tempo
    evs = []
    if drums is not None:
        for t, voice, vel in drums:
            a = int(round(t / spb * tpb))
            from .song import ACOUSTIC_NOTES
            n = DRUM_NOTES.get(voice, ACOUSTIC_NOTES.get(voice, 36)) if not str(voice).isdigit() else int(voice)
            evs += [(a, 1, mido.Message("note_on", channel=9, note=n, velocity=vel)),
                    (a + tpb // 8, 0, mido.Message("note_off", channel=9, note=n, velocity=0))]
    for e in events:
        a, b = int(round(e.start / spb * tpb)), int(round((e.start + e.dur) / spb * tpb))
        ch = 1 if e.pm else 0
        evs += [(a, 1, mido.Message("note_on", channel=ch, note=e.midi, velocity=e.vel)),
                (max(b, a + 1), 0, mido.Message("note_off", channel=ch, note=e.midi, velocity=0))]
    evs.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for t, _, m in evs:
        m.time = t - last
        last = t
        trk.append(m)
    mid.save(path)


def _normalized(plugin_path: str, params: Dict) -> Dict[str, float]:
    """Her settings as the plugin's own normalized values (what Reaper takes)."""
    from pedalboard import load_plugin
    from .song import _quick_load, _set_params
    _quick_load()
    p = load_plugin(plugin_path)
    _set_params(p, params, plugin_path)
    return {k: float(p.parameters[k].raw_value) for k in (params or {})}, p.name


def _live_chain(chain) -> List[Dict]:
    """Each effect as Reaper will load it: a VST3 with normalized params, or an Orphic JSFX with values."""
    from .song import EFFECTS, PLUGIN_DIRS
    out = []
    for fx in chain:
        if "plugin" in fx:
            p = next((str(Path(d) / EFFECTS.get(fx["plugin"].lower(), fx["plugin"])) for d in PLUGIN_DIRS
                      if (Path(d) / EFFECTS.get(fx["plugin"].lower(), fx["plugin"])).exists()), fx["plugin"])
            n2, shown2 = _normalized(p, fx.get("params"))
            out.append({"name": "VST3:" + shown2, "params": n2})
        else:
            kind = str(fx["builtin"]).lower()
            vals = list(JSFX_DEFAULTS[kind])
            extra = set(fx) - {"builtin"} - set(JSFX[kind])
            if extra:
                raise ValueError(f"{kind} has no {', '.join(sorted(extra))} in Reaper")
            for i, k in enumerate(JSFX[kind]):
                if k in fx:
                    vals[i] = float(fx[k])
            out.append({"name": f"JS:orphic/orphic_{kind}", "values": vals})
    return out


def build(song_path: Path, out_dir: Path, log=print, timeout: float = 300) -> Dict:
    from .song import (BELT, BUILTIN_FX, EFFECTS, PLUGIN_DIRS, SongError, Track, _plugin_path, apply_fx, parse,
                       render_kit, render_source, resolve_chain)
    song = yaml.safe_load(Path(song_path).read_text())
    tracks, length, marks = parse(song, Path(song_path).parent)
    tempo, meter = float(song["tempo"]), int(song.get("meter", 4))
    total = length + 2.0
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "media").mkdir(parents=True)
    plan = []
    for name, tr in tracks.items():
        spec = dict(tr.spec)
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name)
        midi = out_dir / "media" / f"{safe}-notes.mid"   # the same notes as a file, for any DAW
        _write_midi(tr.events, midi, tempo, meter, drums=tr.hits if spec.get("kit") else None)
        spb = 60.0 / tempo
        if spec.get("kit"):
            from .song import ACOUSTIC_NOTES
            notes_of = ACOUSTIC_NOTES if str(spec["kit"]).lower() != "808" else DRUM_NOTES
            qn = [[t / spb, t / spb + 0.125, 9, notes_of.get(v, 36), vel] for t, v, vel in tr.hits]
        else:
            qn = [[e.start / spb, (e.start + e.dur) / spb, 1 if e.pm else 0, e.midi, e.vel] for e in tr.events]
        entry = {"name": name, "qn": qn, "end_qn": total / spb, "vol": float(spec.get("gain", 0.8)),
                 "pan": float(spec.get("pan", 0.0)), "live": False}
        chain = resolve_chain(spec)
        synthy = (spec.get("synth") or spec.get("patch")) and not spec.get("kit")
        livable = all("plugin" in fx or str(fx.get("builtin", "")).lower() in JSFX for fx in chain)
        fxs = []
        if livable:
            try:
                fxs = _live_chain(chain)
            except Exception as e:
                log(f"  {name}: its effects can't all run in Reaper ({e}); printing its audio")
                livable = False
        if livable and spec.get("sfz"):
            from .song import SFZ_DIR, SFZ_LIBS
            folder = SFZ_DIR / SFZ_LIBS.get(str(spec["sfz"]), str(spec["sfz"]))
            orig = next(folder.rglob("*.sfz"), None) if folder.exists() else None
            if orig is not None:
                layers = []
                for muted, label in ((False, "open"), (True, "palm-muted")):
                    notes = [q for q in qn if bool(q[2]) == muted]
                    if notes or not muted:
                        wrap = guitar_sfz(orig, out_dir / "media" / f"{safe}-{label}.sfz", muted)
                        layers.append({"name": f"{name} ({label})", "state": sfizz_state(wrap),
                                       "qn": [[a, b, 0, n, v] for a, b, _, n, v in notes]})
                entry.update({"live": True, "synth": "sfizz (SFZTools)", "layers": layers, "fx": fxs,
                              "shows": f"sampled {spec['sfz']} in sfizz" + (", palm mutes on their own track"
                                                                              if len(layers) > 1 else "")})
        elif livable and spec.get("kit") and str(spec["kit"]).lower() == "808":
            wrap = kit_sfz(spec, out_dir / "media" / f"{safe}.sfz")
            entry.update({"live": True, "synth": "sfizz (SFZTools)", "state": sfizz_state(wrap), "params": {},
                          "fx": fxs, "shows": "808 kit in sfizz"})
        elif synthy and livable:
            try:
                if spec.get("patch"):
                    saved = json.loads((BELT / f"{spec['patch']}.json").read_text())
                    spec.setdefault("synth", saved["synth"])
                    spec["params"] = {**saved.get("params", {}), **(spec.get("params") or {})}
                norm, shown = _normalized(_plugin_path(spec["synth"]), spec.get("params"))
                entry.update({"live": True, "synth": shown, "params": norm, "fx": fxs, "shows": shown})
            except Exception as e:
                log(f"  {name}: can't play live in Reaper ({e}); printing its audio")
        if not entry["live"]:
            if spec.get("kit"):
                a = render_kit(tr, total)
            else:
                a, _ = render_source(tr, total)
            a = apply_fx(a, chain)
            wav = out_dir / "media" / f"{safe}.wav"
            sf.write(wav, a.T, 44100, subtype="PCM_24")
            entry["audio"] = str(wav)
        plan.append(entry)
        log(f"  {name}: {'live: ' + entry.get('shows', '') if entry['live'] else 'printed audio'}")
    rpp = out_dir / "session.rpp"
    done = out_dir / ".done"
    lua = out_dir / ".build.lua"
    lua.write_text(LUA % {
        "plan": _lua(plan), "tempo": tempo, "meter": meter, "marks": _lua([[m, t] for m, t in marks]),
        "rpp": _lua_str(str(rpp)), "dir": _lua_str(str(out_dir)), "done": _lua_str(str(done)),
        "log": _lua_str(str(out_dir / "reaper-log.txt"))})
    cfg = _cfg_dir()
    t0 = time.time()
    shown = os.environ.get("ORPHIC_REAPER_DISPLAY")   # debugging: a display to watch it on
    from .song import tool
    cmd = [tool("reaper"), "-cfgfile", str(cfg / "reaper.ini"), "-nosplash", "-newinst", str(lua)]
    env = {**os.environ, "PATH": os.environ.get("PATH", "") + ":/usr/local/bin:/usr/bin:/bin"}  # xvfb-run calls Xvfb, xauth
    if shown:
        env["DISPLAY"] = shown
    proc = subprocess.Popen(cmd if shown else [tool("xvfb-run"), "-a"] + cmd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        while not done.exists():
            if proc.poll() is not None:
                raise ReaperError("Reaper quit before finishing the session")
            if time.time() - t0 > timeout:
                raise ReaperError("Reaper took too long")
            time.sleep(0.5)
        time.sleep(0.5)
    finally:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
    lua.unlink(missing_ok=True)
    # open at the start with the whole song in view (Reaper saves the view it was left at)
    text = rpp.read_text()
    text = re.sub(r"(?m)^  CURSOR .*$", "  CURSOR 0", text)
    text = re.sub(r"(?m)^  ZOOM .*$", f"  ZOOM {max(20, min(200, int(640 / max(total, 1))))} 0 0", text)
    rpp.write_text(text)
    rendered = next(iter(sorted((out_dir / "render").glob("*.wav"))), None)
    if rendered is None:
        raise ReaperError("Reaper saved the session but didn't render it")
    final = out_dir / "reaper-render.wav"
    shutil.move(str(rendered), final)
    shutil.rmtree(out_dir / "render", ignore_errors=True)
    rlog = (out_dir / "reaper-log.txt").read_text() if (out_dir / "reaper-log.txt").exists() else ""
    y, sr = sf.read(final, always_2d=True)
    return {"session": str(rpp), "render": str(final), "seconds": round(time.time() - t0, 1),
            "tracks": [{"name": p["name"], "plays": ("live in Reaper: " + p.get("shows", p["synth"])) if p["live"] else "printed audio"}
                       for p in plan],
            "render_peak_db": round(float(20 * np.log10(np.abs(y).max() + 1e-12)), 1),
            "reaper_notes": [l for l in rlog.splitlines() if l.strip()]}


LUA = r'''
local plan = %(plan)s
local marks = %(marks)s
local logf = io.open(%(log)s, "w")
local function note(s) logf:write(s .. "\n") end
local function norm(s) return (s or ""):lower():gsub("[^a-z0-9]", "") end
local function set_params(tr, fx, params)
  local n = reaper.TrackFX_GetNumParams(tr, fx)
  for want, v in pairs(params) do
    local hit = false
    for i = 0, n - 1 do
      local _, pname = reaper.TrackFX_GetParamName(tr, fx, i, "")
      if norm(pname) == norm(want) then reaper.TrackFX_SetParamNormalized(tr, fx, i, v); hit = true; break end
    end
    if not hit then note("no parameter " .. want) end
  end
end
local function notes(tr, p)
  local item = reaper.CreateNewMIDIItemInProj(tr, 0, reaper.TimeMap2_QNToTime(0, p.end_qn), false)
  local take = reaper.GetActiveTake(item)
  for _, n in ipairs(p.qn) do
    local a = reaper.MIDI_GetPPQPosFromProjQN(take, n[1])
    local b = reaper.MIDI_GetPPQPosFromProjQN(take, n[2])
    reaper.MIDI_InsertNote(take, false, false, a, b, math.floor(n[3]), math.floor(n[4]), math.floor(n[5]), true)
  end
  reaper.MIDI_Sort(take)
  reaper.GetSetMediaItemTakeInfo_String(take, "P_NAME", p.name .. " notes", true)
end
local function place(tr, file)
  reaper.SetOnlyTrackSelected(tr)
  reaper.SetEditCurPos(0, false, false)
  reaper.InsertMedia(file, 0)
end
reaper.Main_OnCommand(40859, 0)
reaper.SetTempoTimeSigMarker(0, -1, 0, -1, -1, %(tempo)s, %(meter)s, 4, false)
local idx = 0
for _, p in ipairs(plan) do
  reaper.InsertTrackAtIndex(idx, true)
  local tr = reaper.GetTrack(0, idx); idx = idx + 1
  reaper.GetSetMediaTrackInfo_String(tr, "P_NAME", p.name, true)
  reaper.SetMediaTrackInfo_Value(tr, "D_VOL", p.vol)
  reaper.SetMediaTrackInfo_Value(tr, "D_PAN", p.pan)
  if p.live and p.layers then
    -- a folder track with the rig on it, and an instrument track per layer inside it
    for _, e in ipairs(p.fx or {}) do
      local f = reaper.TrackFX_AddByName(tr, e.name, false, -1)
      if f < 0 then note(p.name .. ": couldn't load " .. e.name)
      elseif e.values then for i, v in ipairs(e.values) do reaper.TrackFX_SetParam(tr, f, i - 1, v) end
      else set_params(tr, f, e.params or {}) end
    end
    reaper.SetMediaTrackInfo_Value(tr, "I_FOLDERDEPTH", 1)
    for li, l in ipairs(p.layers) do
      reaper.InsertTrackAtIndex(idx, true)
      local lt = reaper.GetTrack(0, idx); idx = idx + 1
      reaper.GetSetMediaTrackInfo_String(lt, "P_NAME", l.name, true)
      local fx = reaper.TrackFX_AddByName(lt, "VST3:" .. p.synth, false, -1)
      if fx < 0 then note(l.name .. ": couldn't load " .. p.synth)
      elseif not reaper.TrackFX_SetNamedConfigParm(lt, fx, "vst_chunk", l.state) then note(l.name .. ": state not taken") end
      notes(lt, {name = l.name, qn = l.qn, end_qn = p.end_qn})
      if li == #p.layers then reaper.SetMediaTrackInfo_Value(lt, "I_FOLDERDEPTH", -1) end
    end
  elseif p.live then
    local fx = reaper.TrackFX_AddByName(tr, "VST3:" .. p.synth, false, -1)
    if fx < 0 then note(p.name .. ": couldn't load " .. p.synth)
    else
      if p.state then
        if not reaper.TrackFX_SetNamedConfigParm(tr, fx, "vst_chunk", p.state) then note(p.name .. ": state not taken") end
      end
      set_params(tr, fx, p.params)
    end
    for _, e in ipairs(p.fx or {}) do
      local f = reaper.TrackFX_AddByName(tr, e.name, false, -1)
      if f < 0 then note(p.name .. ": couldn't load " .. e.name)
      elseif e.values then for i, v in ipairs(e.values) do reaper.TrackFX_SetParam(tr, f, i - 1, v) end
      else set_params(tr, f, e.params or {}) end
    end
    notes(tr, p)
  else
    place(tr, p.audio)
    reaper.SetMediaTrackInfo_Value(tr, "I_FOLDERDEPTH", 1)
    reaper.InsertTrackAtIndex(idx, true)
    local lane = reaper.GetTrack(0, idx); idx = idx + 1
    reaper.GetSetMediaTrackInfo_String(lane, "P_NAME", p.name .. " (notes)", true)
    notes(lane, p)
    reaper.SetMediaTrackInfo_Value(lane, "B_MUTE", 1)
    reaper.SetMediaTrackInfo_Value(lane, "I_FOLDERDEPTH", -1)
  end
end
for i, m in ipairs(marks) do reaper.AddProjectMarker(0, false, m[2], 0, m[1], i) end
reaper.GetSetProjectInfo_String(0, "RENDER_FILE", %(dir)s .. "/render", true)
reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", "mix", true)
reaper.GetSetProjectInfo(0, "RENDER_BOUNDSFLAG", 1, true)
reaper.GetSetProjectInfo(0, "RENDER_TAILFLAG", 0, true)
reaper.Main_SaveProjectEx(0, %(rpp)s, 0)
reaper.Main_OnCommand(42230, 0)
logf:close()
local d = io.open(%(done)s, "w"); d:write("ok"); d:close()
'''
