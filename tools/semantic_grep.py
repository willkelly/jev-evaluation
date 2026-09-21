#!/usr/bin/env python
"""Where does a yes/no predicate over many lines put its probabilities?

This began as a check on whether one request could answer a plain-English
predicate about each line of a document -- semantic grep. It can. The more
useful result was about where the answers land, which is why the file is kept.

A document of twenty-one lines goes into the state and the model is asked, of
each line, whether it is about elephants. Lines are written so the predicate
cannot be answered by looking for the word: no matching line contains
"elephant", half of them name nothing elephant-specific at all, and the
non-matching lines include rhinoceroses, hippopotamuses, whales and giraffes.

Four ways of asking:

    A  one request per line, the line alone in the state
    B  the whole document in state, one noul per line, named by number
    C  the whole document in state, one noul per line, quoting the line
    D  the whole document in state, one choice per group of seven lines, over
       the 128 options naming which lines in that group match

Scored at the conventional 0.5, arm A looks like the worst of them, missing
half the matching lines. It is not. Non-matching lines come back at a median
probability of 0.01 and matching lines spread from 0.10 to 0.90, so a cut at
0.5 runs through the middle of the true positives. Ranked instead of
thresholded, A and C separate the two populations perfectly. The signal was
never in question; the cut was in the wrong place, which is the point this file
is cited for.

    python tools/semantic_grep.py --score-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUP = 7
CONVENTIONAL, MEASURED = 0.5, 0.05


def auroc(pos: list[float], neg: list[float]) -> float | None:
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
    return (sum(ranks[v] for v in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def counts(pos: list[float], neg: list[float], thr: float) -> dict:
    tp = sum(1 for p in pos if p >= thr)
    fp = sum(1 for p in neg if p >= thr)
    fn, tn = len(pos) - tp, len(neg) - fp
    n = tp + fp + fn + tn
    return {"threshold": thr, "accuracy": (tp + tn) / n,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None, "errors": fp + fn}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--score-only", action="store_true", default=True,
                    help="the log is in the repository; this only ever rescores it")
    ap.add_argument("--dir", default=str(ROOT / "runs" / "guide-demos"))
    args = ap.parse_args(argv)

    base = Path(args.dir)
    docs = json.loads((base / "semantic-grep-docs.json").read_text())
    log = base / "semantic-grep.jsonl"
    if not log.exists():
        print(f"no log at {log}", file=sys.stderr)
        return 2

    arms = defaultdict(lambda: {"pos": [], "neg": [], "calls": 0, "itok": 0,
                                "tp": 0, "fp": 0, "tn": 0, "fn": 0, "exact": 0, "groups": 0})
    with log.open() as fh:
        for raw in fh:
            rec = json.loads(raw)
            if rec.get("outcome") != "ok":
                continue
            cond, meta = rec["condition"], rec["meta"]
            ans = rec["response"]["answers"]
            a = arms[cond]
            a["calls"] += 1
            a["itok"] += rec.get("input_tokens") or 0
            doc = docs[str(meta["doc"])]
            if cond.startswith("D"):
                want, got = meta["truth"]["which"], ans["which"]["choice"]
                a["groups"] += 1
                a["exact"] += got == want
                lo = meta["group"] * GROUP
                for j in range(GROUP):
                    t, g = want[j] == "1", got[j] == "1"
                    a["tp" if (t and g) else "fn" if t else "fp" if g else "tn"] += 1
            else:
                pairs = ([(ans["hit"]["noul"], doc["truth"][meta["line"]])] if cond.startswith("A")
                         else [(v["noul"], doc["truth"][int(k[1:]) - 1]) for k, v in ans.items()])
                for p, t in pairs:
                    (a["pos"] if t else a["neg"]).append(p)

    out = {"source": "runs/guide-demos/semantic-grep.jsonl",
           "predicate": "the line is about elephants",
           "design": {"documents": len(docs), "lines_per_document": 21, "group_size": GROUP,
                      "note": "no matching line contains the word 'elephant'"},
           "arms": {}}
    for cond in sorted(arms):
        a = arms[cond]
        e = {"requests": a["calls"], "input_tokens": a["itok"]}
        if a["pos"]:
            s_pos, s_neg = sorted(a["pos"]), sorted(a["neg"])
            q = lambda s, p: s[min(len(s) - 1, int(p * len(s)))]
            e["matching"] = {"n": len(s_pos), "p10": q(s_pos, .10), "median": q(s_pos, .5),
                             "p90": q(s_pos, .90)}
            e["non_matching"] = {"n": len(s_neg), "median": q(s_neg, .5), "p90": q(s_neg, .90)}
            e["auroc"] = auroc(a["pos"], a["neg"])
            e["at_conventional_threshold"] = counts(a["pos"], a["neg"], CONVENTIONAL)
            e["at_measured_threshold"] = counts(a["pos"], a["neg"], MEASURED)
        else:
            n = a["tp"] + a["fp"] + a["tn"] + a["fn"]
            e["no_threshold_to_set"] = "a choice returns one discrete subset"
            e["accuracy"] = (a["tp"] + a["tn"]) / n
            e["groups_exactly_right"] = a["exact"]
            e["groups"] = a["groups"]
        out["arms"][cond] = e

    path = base / "semantic-grep.json"
    path.write_text(json.dumps(out, indent=1) + "\n")

    print(f"{'arm':<24} {'AUROC':>6} | {'at 0.5':>22} | {'at 0.05':>22}")
    for cond in sorted(out["arms"]):
        e = out["arms"][cond]
        if "auroc" not in e:
            print(f"{cond:<24} {'  --':>6} | accuracy {e['accuracy']:.3f}, "
                  f"{e['groups_exactly_right']}/{e['groups']} groups exactly right")
            continue
        c, m = e["at_conventional_threshold"], e["at_measured_threshold"]
        f = lambda x: f"acc {x['accuracy']:.3f} rec {x['recall']:.3f}"
        print(f"{cond:<24} {e['auroc']:>6.3f} | {f(c):>22} | {f(m):>22}")
    a = out["arms"]["A-one-request-per-line"]
    print(f"\nnon-matching lines sit at a median of {a['non_matching']['median']:.2f}; "
          f"matching lines run {a['matching']['p10']:.2f} to {a['matching']['p90']:.2f}")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
