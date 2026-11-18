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
    a = ap.parse_args(argv)
    from .monochord import map_synth
    out = Path(a.out) if a.out else Path("maps") / Path(a.plugin).stem
    map_synth(a.plugin, out, steps=a.steps, top=a.top, log=lambda s: print(s, file=sys.stderr, flush=True))


if __name__ == "__main__":
    main()
