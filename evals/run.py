#!/usr/bin/env python3
"""Measure routing accuracy on evals/cases.json against live Jev (needs JEV_TOKEN).

  python3 evals/run.py            Jev + rules, as the hook runs
  python3 evals/run.py --no-jev   rules only (the offline fallback)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import route  # noqa: E402


def main(argv):
    use_jev = "--no-jev" not in argv
    cases = json.loads((ROOT / "evals" / "cases.json").read_text())
    rules = route.load_rules()

    ok = 0
    print(f"{'want':12}{'got':14}{'source':15}{'conf':>5}  prompt")
    for c in cases["tiers"]:
        d = route.classify("", c["prompt"], rules, use_jev=use_jev)
        good = d["tier"] == c["want"]
        ok += good
        conf = (d.get("jev") or {}).get("confidence")
        print(f"{c['want']:12}{d['tier'] + ('' if good else ' ✗'):14}{d['source']:15}"
              f"{conf if conf is not None else float('nan'):5.2f}  {c['prompt'][:70]}")
    print(f"\ntiers: {ok}/{len(cases['tiers'])}\n")

    nok = 0
    for c in cases["nudge"]:
        out, rec = route.nudge({"prompt": c["prompt"]}, rules, use_jev=use_jev)
        good = (out is not None) == c["want"]
        nok += good
        print(f"{'nudge' if c['want'] else 'quiet':6}{'' if good else '✗':2} "
              f"{rec.get('jev_noul', float('nan')):5.2f}  {c['prompt'][:70]}")
    print(f"\nnudge: {nok}/{len(cases['nudge'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
