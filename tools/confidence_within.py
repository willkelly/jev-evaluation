#!/usr/bin/env python
"""Ask whether confidence predicts correctness within a difficulty, or only across them.

The prompting guide tells readers that confidence predicts whether an answer is
right, citing an AUROC of 0.878 pooled over every answer in the run that carried
one. That pooled figure has a hole in it. Confidence also tracks how hard a
problem looks, and a pool that mixes easy conditions with hard ones will show
confidence ranking correct answers above wrong ones even if, within any one
condition, it separates nothing: the easy cells supply most of the correct
answers at high confidence and the hard cells most of the wrong ones at low
confidence, and the ranking falls out of the mixture rather than the signal.

The test is to compute the same statistic inside each cell -- one condition at
one difficulty, so that everything a pool could be mixing is held fixed -- and
weight those by how many correct-wrong pairs each cell contributes. If the within-condition figure matches the pooled one,
confidence is doing what the guide says. If it collapses toward 0.5, the pooled
figure was measuring difficulty.

A cell in which every answer is right, or every answer is wrong, has no pairs to
rank and is excluded from the within-condition figure -- while still counting
toward the pooled one, which is part of how the two come apart. Those cells are
reported, because a condition the model gets wrong every time is a result in
itself.

Only `choice` answers are used. A `noul` carries no confidence at all, and a
`score` answer's truth is a rubric id rather than a label, so scoring it here
would reproduce a join bug this harness has already had once.

The logs are 1.3 GB and are not in the repository, so the result is written to
`runs/<run>/confidence_within.json`, which is.

    python tools/confidence_within.py --run full-20260919
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney U with tie-aware ranks. Ties matter: confidence is quantised."""
    if not pos or not neg:
        return None
    merged = sorted(pos + neg)
    ranks: dict[float, float] = {}
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1] == merged[i]:
            j += 1
        ranks[merged[i]] = (i + j) / 2 + 1
        i = j + 1
    total = sum(ranks[v] for v in pos)
    return (total - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def summarise(cells: dict[tuple, dict]) -> dict:
    right = [v for c in cells.values() for v in c["right"]]
    wrong = [v for c in cells.values() for v in c["wrong"]]
    n = len(right) + len(wrong)
    if not n:
        return {"n": 0}
    numer = denom = 0.0
    usable = all_right = all_wrong = 0
    for c in cells.values():
        a = auroc(c["right"], c["wrong"])
        if a is None:
            all_right += not c["wrong"]
            all_wrong += not c["right"]
            continue
        weight = len(c["right"]) * len(c["wrong"])
        numer += a * weight
        denom += weight
        usable += 1
    pooled = auroc(right, wrong)
    within = numer / denom if denom else None
    return {
        "n": n,
        "accuracy": len(right) / n,
        "mean_confidence_right": sum(right) / len(right) if right else None,
        "mean_confidence_wrong": sum(wrong) / len(wrong) if wrong else None,
        "auroc_pooled": pooled,
        "auroc_within_cell": within,
        "inflation": (pooled - within) if (pooled is not None and within is not None) else None,
        "cells": len(cells),
        "cells_usable": usable,
        "cells_all_right": all_right,
        "cells_all_wrong": all_wrong,
    }


def scan(run: Path, domains: dict[str, str]) -> dict:
    by_domain: dict[str, dict[tuple, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"right": [], "wrong": []}))
    per_cell: dict[str, dict] = {}

    for log in sorted(run.glob("e*.jsonl")):
        with log.open() as fh:
            for raw in fh:
                if '"confidence"' not in raw:
                    continue
                rec = json.loads(raw)
                if rec.get("outcome") != "ok":
                    continue
                meta = rec.get("meta") or {}
                truth = meta.get("truth") or {}
                # Only E1 and E9 record a generator. Elsewhere the condition's
                # first segment names the subject, which is what a reader wants.
                generator = meta.get("generator") or meta.get("family") or ""
                if not generator:
                    generator = str(rec.get("condition") or "").split("/")[0]
                domain = f"{log.stem.upper()} {domains.get(generator, generator)}".strip()
                for key, ans in (rec["response"].get("answers") or {}).items():
                    if not isinstance(ans, dict) or ans.get("type") != "choice":
                        continue
                    if "confidence" not in ans:
                        continue
                    want = truth.get(key)
                    # Truth must name one of the options actually offered, or the
                    # join is wrong rather than the answer.
                    if want is None or want not in (ans.get("probabilities") or {}):
                        continue
                    # The cell must hold difficulty fixed, not just the condition.
                    # E1 sweeps clause ratio inside one condition, so keying on
                    # the condition alone would leave the very mixture this is
                    # meant to remove.
                    cell = (log.stem, rec.get("condition"), key,
                            json.dumps(meta.get("difficulty"), sort_keys=True))
                    bucket = by_domain[domain][cell]
                    bucket["right" if ans["choice"] == want else "wrong"].append(ans["confidence"])

    out = {"by_domain": {}, "cells": per_cell}
    for domain, cells in by_domain.items():
        out["by_domain"][domain] = summarise(cells)
        for cell, c in cells.items():
            n = len(c["right"]) + len(c["wrong"])
            if n < 50:
                continue
            per_cell["/".join(str(x) for x in cell)] = {
                "domain": domain, "n": n, "accuracy": len(c["right"]) / n,
                "auroc": auroc(c["right"], c["wrong"]),
                "mean_confidence_right": sum(c["right"]) / len(c["right"]) if c["right"] else None,
                "mean_confidence_wrong": sum(c["wrong"]) / len(c["wrong"]) if c["wrong"] else None,
            }
    everything = {k: v for cells in by_domain.values() for k, v in cells.items()}
    out["all_domains"] = summarise(everything)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="full-20260919")
    args = ap.parse_args(argv)

    run = ROOT / "runs" / args.run
    if not list(run.glob("e*.jsonl")):
        print(f"no logs in {run}; re-run the experiments to regenerate them", file=sys.stderr)
        return 2

    # The generator names are the experiments' own; these are a reader's.
    domains = {
        "sat3": "3-SAT", "sudoku": "Sudoku", "dfa": "DFA successors",
        "semantic": "support tickets", "world": "propositions", "dyck": "Dyck words",
        "graph": "graph reachability", "program": "program reachability",
        "taxonomy": "taxonomy", "ordering": "pairwise ordering",
        "enrolled-joint": "proposition combinations", "numeric": "numeric scenes",
        "adversarial": "adversarial tickets", "positive-control": "support tickets",
    }
    result = scan(run, domains)
    result["source"] = f"runs/{args.run}/e*.jsonl, choice answers only"
    out = run / "confidence_within.json"
    out.write_text(json.dumps(result, indent=1) + "\n")

    print(f"{'subject':<36} {'n':>7} {'acc':>6} {'pooled':>7} {'within':>7} {'cells':>9}")
    rows = sorted(result["by_domain"].items(), key=lambda kv: -(kv[1].get("n") or 0))
    for domain, s in rows:
        if not s.get("n"):
            continue
        f = lambda v: f"{v:.3f}" if v is not None else "    --"
        cells = f"{s['cells_usable']}/{s['cells']}"
        print(f"{domain:<36} {s['n']:>7,} {s['accuracy']:>6.3f} {f(s['auroc_pooled']):>7} "
              f"{f(s['auroc_within_cell']):>7} {cells:>9}")
    a = result["all_domains"]
    f = lambda v: f"{v:.3f}" if v is not None else "    --"
    print(f"\n{'everything':<36} {a['n']:>7,} {a['accuracy']:>6.3f} {f(a['auroc_pooled']):>7} "
          f"{f(a['auroc_within_cell']):>7} {a['cells_usable']}/{a['cells']}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
