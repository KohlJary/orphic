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
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)
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
