"""orphic monochord <plugin.vst3> [-o DIR] [--top N] [--steps N]"""
import argparse
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orphic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("monochord", help="map a synth by sweeping its knobs")
    m.add_argument("plugin")
    m.add_argument("-o", "--out")
    m.add_argument("--top", type=int, default=60)
    m.add_argument("--steps", type=int, default=7)
    r = sub.add_parser("render", help="render a song file, and (with --listen) hear it back")
    r.add_argument("song")
    r.add_argument("-o", "--out")
    r.add_argument("--listen", action="store_true", help="listen back to the render with Acousmatic")
    n = sub.add_parser("noodle", help="play a riff quickly and hear it back")
    n.add_argument("--instrument", required=True, help='JSON, e.g. {"synth": "nekobi", "params": {"cutoff": 40}}')
    n.add_argument("--phrase", required=True, help='bars in song notation, or JSON drum grid for a kit')
    n.add_argument("--tempo", type=float, default=100)
    n.add_argument("--loops", type=int, default=2)
    n.add_argument("-o", "--out")
    t = sub.add_parser("tone", help="put a guitar source and rig through the tone probe, measured")
    t.add_argument("--instrument", default='{"synth": "kars"}')
    t.add_argument("--chain", default=None, help="JSON effects chain (else the instrument's fx/rig)")
    t.add_argument("-o", "--out")
    t.add_argument("--previous", help="JSON file with an earlier tone, to say what changed")
    me = sub.add_parser("map-effect", help="map an effect's knobs with a guitar going in")
    me.add_argument("effect")
    me.add_argument("--source", default='{"synth": "kars"}')
    me.add_argument("-o", "--out")
    c = sub.add_parser("clip", help="write, read or import a MIDI clip in a project's clips/ folder")
    c.add_argument("op", choices=["write", "read", "list", "import"])
    c.add_argument("name", nargs="?")
    c.add_argument("--dir", required=True, help="the project folder (clips live in DIR/clips)")
    c.add_argument("--riff", help="song notation")
    c.add_argument("--notes", help="JSON list of notes: bar, beat, pitch, len, vel, pm")
    c.add_argument("--meter", type=int, default=4)
    c.add_argument("--bars", help="read: a bar or range, like 3 or 1-4")
    c.add_argument("--file", help="import: a .mid file")
    c.add_argument("--track", help="import: one named track of it")
    rp = sub.add_parser("reaper", help="build a song as a Reaper session (session.rpp) and render it")
    rp.add_argument("song")
    rp.add_argument("-o", "--out", required=True)
    cc = sub.add_parser("check-chain", help="check an effects chain before saving it as a rig")
    cc.add_argument("--chain", required=True)
    au = sub.add_parser("audition", help="hear one section in its place in the song, without rendering the song")
    au.add_argument("song")
    au.add_argument("section")
    au.add_argument("--time", type=int, default=1, help="which time the section comes around (default the 1st)")
    au.add_argument("--repeat", type=int, default=1, help="play it this many times over, to hear it worn in")
    au.add_argument("--swap", help='JSON: parts to play differently just this once, {"bass": "E2/1 ...", "hold": 2}')
    au.add_argument("-o", "--out", required=True)
    au.add_argument("--listen", action="store_true")
    sp = sub.add_parser("spaces", help="the rooms a part can play in: list them, or save one of your own")
    sp.add_argument("op", choices=["list", "save"])
    sp.add_argument("name", nargs="?")
    sp.add_argument("--says", help="save: what room it is, in your words")
    sp.add_argument("--numbers", help='save: JSON, like {"size": 0.8, "walls_ms": 40}; anything left out comes from '
                                      'the built-in of the same name')
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)
    if a.cmd == "check-chain":
        import json
        from .song import SongError, check_chain
        try:
            check_chain(json.loads(a.chain))
            print(json.dumps({"ok": True}))
        except (SongError, ValueError) as e:
            print(json.dumps({"error": str(e)})); sys.exit(2)
        return
    if a.cmd == "spaces":
        import json
        from . import spaces as S
        try:
            if a.op == "list":
                print(json.dumps({"spaces": S.listing()}, indent=1))
            else:
                print(json.dumps(S.save_space(a.name or "", a.says or "", json.loads(a.numbers or "{}")), indent=1))
        except (S.SpaceError, ValueError) as e:
            print(json.dumps({"error": str(e)})); sys.exit(2)
        return
    if a.cmd == "audition":
        import json
        from .audition import audition
        from .song import SongError
        out = Path(a.out)
        try:
            res = audition(Path(a.song), out, a.section, a.time, a.repeat, json.loads(a.swap) if a.swap else None, log=log)
        except SongError as e:
            print(json.dumps({"error": str(e)})); sys.exit(2)
        except Exception as e:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"})); sys.exit(2)
        if a.listen:
            from acousmatic.listen import listen
            listen(out / "render.wav", out / "listening", engrave=False, offline=True, log=log)
            res["listening_record"] = str(out / "listening" / "record.md")
        print(json.dumps(res, indent=1, default=float))
        return
    if a.cmd == "reaper":
        import json
        from .reaper import ReaperError, build
        from .song import SongError
        try:
            print(json.dumps(build(Path(a.song), Path(a.out), log=log), default=float))
        except (SongError, ReaperError) as e:
            print(json.dumps({"error": str(e)})); sys.exit(2)
        except Exception as e:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"})); sys.exit(2)
        return
    if a.cmd == "clip":
        import json
        import re as _re
        from . import clip as C
        folder = Path(a.dir) / "clips"
        try:
            if a.op == "list":
                out = []
                for p in sorted(folder.glob("*.mid")):
                    n = C.load(p)
                    out.append({"clip": p.stem, "bars": C.bars_of(n, a.meter), "notes": len(n)})
                print(json.dumps({"clips": out})); return
            name = _re.sub(r"[^a-z0-9_-]+", "-", (a.name or "").lower()).strip("-")
            if not name:
                raise C.ClipError("a clip needs a name")
            path = folder / f"{name}.mid"
            if a.op == "write":
                if a.riff and a.riff.strip().startswith("{"):
                    notes = C.from_grid(json.loads(a.riff), a.meter)   # a drum grid
                elif a.riff:
                    notes = C.from_notation(a.riff, a.meter)
                elif a.notes:
                    notes = C.from_notes(json.loads(a.notes), a.meter)
                else:
                    raise C.ClipError("write needs a riff (notation) or notes")
                if not notes:
                    raise C.ClipError("that clip has no notes")
                C.save(notes, path, a.meter)
            elif a.op == "import":
                if not a.file:
                    raise C.ClipError("import needs a .mid file")
                notes = C.load(Path(a.file), a.track)
                if not notes:
                    raise C.ClipError("no notes in that file" + (f" on track {a.track!r}" if a.track else ""))
                C.save(notes, path, a.meter)
            elif not path.exists():
                raise C.ClipError(f"no clip {name!r}")
            notes = C.load(path)
            nb = C.bars_of(notes, a.meter)
            first, last = 1, nb
            if a.bars:
                m = _re.match(r"^(\d+)(?:-(\d+))?$", a.bars.strip())
                if m:
                    first = int(m.group(1)); last = int(m.group(2) or m.group(1))
            if a.op != "read":
                first, last = 1, min(nb, 4)
            print(json.dumps({"clip": name, "file": str(path), "bars": nb, "notes": len(notes),
                              "roll": C.roll(notes, a.meter, first, last),
                              "as_notation": C.to_notation(notes, a.meter),
                              "use_it": f"in a song section: <instrument>: {{clip: {name}}}"}))
        except (C.ClipError, ValueError) as e:
            print(json.dumps({"error": str(e)})); sys.exit(2)
        return
    if a.cmd == "tone":
        import json
        import soundfile as sf
        from .song import SongError
        from .tone import tone
        try:
            rep, audio = tone(json.loads(a.instrument), json.loads(a.chain) if a.chain else None)
        except SongError as e:
            print(json.dumps({"error": str(e)})); sys.exit(2)
        except Exception as e:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"})); sys.exit(2)
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            sf.write(Path(a.out) / "render.wav", audio.T, 44100, subtype="PCM_24")
            rep["audio"] = str(Path(a.out) / "render.wav")
        if a.previous and Path(a.previous).exists():
            from .tone import differences
            rep["since_last_try"] = differences(json.loads(Path(a.previous).read_text()), rep["tone"])
        print(json.dumps(rep, default=float))
        return
    if a.cmd == "map-effect":
        import json
        from .monochord import map_effect
        out = Path(a.out) if a.out else Path("maps") / Path(a.effect).stem
        map_effect(a.effect, json.loads(a.source), out, log=log)
        return
    if a.cmd == "noodle":
        import json
        from .noodle import noodle
        from .song import SongError
        try:
            phrase = json.loads(a.phrase) if a.phrase.strip().startswith("{") else a.phrase
            res = noodle(json.loads(a.instrument), phrase, a.tempo, a.loops, out_dir=Path(a.out) if a.out else None)
        except SongError as e:
            print(json.dumps({"error": str(e)})); sys.exit(2)
        except Exception as e:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"})); sys.exit(2)
        print(json.dumps(res, default=float))
        return
    if a.cmd == "monochord":
        from .monochord import map_synth
        out = Path(a.out) if a.out else Path("maps") / Path(a.plugin).stem
        map_synth(a.plugin, out, steps=a.steps, top=a.top, log=log)
    elif a.cmd == "render":
        import json
        from .song import SongError, render
        out = Path(a.out) if a.out else Path("renders") / Path(a.song).stem
        try:
            res = render(Path(a.song), out, log=log)
        except SongError as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(2)
        except Exception as e:   # a YAML mistake, a missing file
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
            sys.exit(2)
        if a.listen:
            from acousmatic.listen import listen
            listen(out / "render.wav", out / "listening", engrave=False, offline=True, log=log)
            res["listening_record"] = str(out / "listening" / "record.md")
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
