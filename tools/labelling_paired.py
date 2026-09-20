#!/usr/bin/env python
"""Recompute the option-labelling comparison, split by whether the cell is at ceiling.

Experiment 5 asks the same 500 problems twice at each of three constraint
counts: once with options named in words that describe them, once with the same
options named by bitstrings. The experiment reports one accuracy per arm pooled
over all three counts, 0.953 against 0.935.

That pooled figure understates the effect, because one of the three cells is at
the ceiling this evaluation defines -- a task the model already answers almost
perfectly, where a decline cannot be measured because no accuracy is left to
lose. With two constraints both arms score above 0.99, so that cell contributes
a gap near zero for reasons that have nothing to do with labelling, and drags
the average down. The report's own glossary names the problem; the experiment
averaged across it anyway.

This splits the cells and runs the paired test on each, because the two arms are
the same problems at the same index and McNemar's test is what a paired
comparison calls for. The exact binomial on the discordant pairs is used rather
than the chi-square approximation, since the counts are small.

The logs are 1.3 GB and are not in the repository, so the result is written to
`runs/<run>/labelling_paired.json`, which is. That file is what the prompting
guide cites and what `tools/verify_citations.py` checks the guide against.

    python tools/labelling_paired.py --run full-20260919
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The experiment's names for the two arms, and the ones a reader wants.
ARMS = {"legible": "described", "bitstring": "bitstring"}


def mcnemar_exact(a: dict, b: dict) -> tuple[int, int, float]:
    """Two-sided exact binomial on the pairs where the two arms disagree."""
    only_a = sum(1 for i in a if a[i] and not b[i])
    only_b = sum(1 for i in a if b[i] and not a[i])
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n
    return only_a, only_b, min(1.0, 2 * tail)


def score(log: Path) -> dict[tuple[str, str], dict[int, bool]]:
    """Per arm and constraint count, whether each problem's answer was consistent."""
    out: dict[tuple[str, str], dict[int, bool]] = defaultdict(dict)
    with log.open() as fh:
        for line in fh:
            if '"enrolled-joint/k' not in line:
                continue
            rec = json.loads(line)
            condition = rec.get("condition", "")
            _, count, arm = condition.split("/")
            if arm not in ARMS or rec.get("outcome") != "ok":
                continue
            meta = rec["meta"]
            chosen = ((rec["response"].get("answers") or {}).get("joint") or {}).get("choice")
            # option_ids maps the canonical bitstring to the label as presented,
            # so the two arms' answers become comparable by inverting it.
            presented = meta.get("option_ids") or {}
            canonical = {v: k for k, v in presented.items()}.get(chosen, chosen)
            out[(count, arm)][meta["index"]] = canonical in set(meta["consistent_cells"])
    return out


def compare(scored: dict, counts: list[str]) -> dict:
    a: dict = {}
    b: dict = {}
    for count in counts:
        for i, v in scored[(count, "legible")].items():
            a[(count, i)] = v
        for i, v in scored[(count, "bitstring")].items():
            b[(count, i)] = v
    shared = [i for i in a if i in b]
    a = {i: a[i] for i in shared}
    b = {i: b[i] for i in shared}
    only_a, only_b, p = mcnemar_exact(a, b)
    described = sum(a.values()) / len(a)
    bitstring = sum(b.values()) / len(b)
    return {
        "constraint_counts": counts,
        "n_pairs": len(a),
        "described": described,
        "bitstring": bitstring,
        "gap": described - bitstring,
        "only_described_right": only_a,
        "only_bitstring_right": only_b,
        "mcnemar_p": p,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="full-20260919")
    ap.add_argument("--ceiling", type=float, default=0.99,
                    help="an arm scoring above this leaves no room for a gap (default 0.99)")
    args = ap.parse_args(argv)

    run = ROOT / "runs" / args.run
    log = run / "e5.jsonl"
    if not log.exists():
        print(f"no log at {log}; re-run experiment 5 to regenerate it", file=sys.stderr)
        return 2

    scored = score(log)
    counts = sorted({c for c, _ in scored})
    per_count = {c: compare(scored, [c]) for c in counts}
    at_ceiling = [c for c, r in per_count.items()
                  if r["described"] > args.ceiling and r["bitstring"] > args.ceiling]
    below = [c for c in counts if c not in at_ceiling]

    result = {
        "source": f"runs/{args.run}/e5.jsonl, enrolled-joint conditions",
        "definition": (
            "the same problems asked with options named in words and with the same "
            "options named by bitstrings, paired by index, scored as whether the "
            "chosen cell is consistent with the state"
        ),
        "ceiling_threshold": args.ceiling,
        "at_ceiling": at_ceiling,
        "by_constraint_count": per_count,
        "below_ceiling": compare(scored, below) if below else None,
        "all_counts": compare(scored, counts),
    }
    out = run / "labelling_paired.json"
    out.write_text(json.dumps(result, indent=1) + "\n")

    print(f"{'cells':<22} {'pairs':>6} {'described':>10} {'bitstring':>10} {'gap':>7} {'p':>8}")
    for c in counts:
        r = per_count[c]
        mark = "  at ceiling" if c in at_ceiling else ""
        print(f"{c:<22} {r['n_pairs']:>6} {r['described']:>10.3f} {r['bitstring']:>10.3f} "
              f"{r['gap']:>+7.3f} {r['mcnemar_p']:>8.4f}{mark}")
    for label, key in (("below the ceiling", "below_ceiling"), ("all cells pooled", "all_counts")):
        r = result[key]
        if r:
            print(f"{label:<22} {r['n_pairs']:>6} {r['described']:>10.3f} {r['bitstring']:>10.3f} "
                  f"{r['gap']:>+7.3f} {r['mcnemar_p']:>8.4f}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
