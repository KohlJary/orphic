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
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)
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
