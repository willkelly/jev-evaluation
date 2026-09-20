#!/usr/bin/env python
"""Measure confidence as a detector of a state that cannot answer the question.

Experiment 9 reports this as a mean separation: average confidence on answerable
states minus average confidence on unanswerable ones. That number says the model
barely reacts to fluent nonsense -- a separation of 0.009 on a choice question --
and the report drew the obvious conclusion, that confidence misses this case.

The conclusion is wrong, and the mean is why. Confidence on an answerable state
is not spread over the range: it is exactly 1.00 on most of them. A state that
cannot answer the question pulls it off that ceiling by a little or a lot, and a
little is enough to rank every such state below almost every answerable one
while moving the mean by almost nothing. A mean cannot see that. A rank
statistic can, and so can a gate placed where the mass actually is.

So this reports, for each kind of unanswerable state and each question type:
the AUROC of confidence as a ranking of answerable above unanswerable, and what
two gates actually catch -- the conventional one at 0.8, and one that flags any
answer short of certainty.

Three kinds of unanswerable state, which behave differently enough that a single
figure over them is not worth having:

    underspecified  the state omits the fact the question needs
    contradictory   the state asserts both of two incompatible things
    nonsense        fluent, grammatical text that means nothing

A noul returns no confidence, so the plan's stand-in, 2*|p - 0.5|, is used for
that arm and labelled as a stand-in wherever it appears.

The gate at 1.00 has a condition on it that this experiment cannot test, and it
matters more than anything else here: it works only where the model is otherwise
certain. Its false-alarm rate is the fraction of answerable states that are not
at 1.00, which is 14% for the ticket routing used here and would be near 100% on
a subject the model finds hard -- it returns a mean confidence of 0.57 on
satisfiability. Measure that fraction on your own traffic before using the gate.

    python tools/abstention_gates.py --run full-20260919
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KINDS = ("underspecified", "contradictory", "nonsense")
GATES = (0.8, 1.0)


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """P(a random answerable scores above a random unanswerable), ties counted half."""
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


def quantiles(v: list[float]) -> dict:
    s = sorted(v)
    n = len(s)
    at = lambda p: s[min(n - 1, int(p * n))]
    return {"n": n, "min": s[0], "p05": at(.05), "p25": at(.25), "median": at(.5),
            "mean": sum(s) / n, "at_1_00": sum(1 for x in s if x >= 1.0) / n}


def collect(log: Path) -> dict[tuple[str, str], list[float]]:
    vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    with log.open() as fh:
        for raw in fh:
            if '"abstention/' not in raw:
                continue
            rec = json.loads(raw)
            if rec.get("outcome") != "ok":
                continue
            meta = rec["meta"]
            qtype, kind = meta.get("question_type"), meta.get("kind")
            for ans in (rec["response"].get("answers") or {}).values():
                if not isinstance(ans, dict):
                    continue
                if ans.get("type") == "noul":
                    value = 2 * abs(ans["noul"] - 0.5)
                elif "confidence" in ans:
                    value = ans["confidence"]
                else:
                    continue
                vals[(qtype, kind)].append(value)
    return vals


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="full-20260919")
    args = ap.parse_args(argv)

    log = ROOT / "runs" / args.run / "e9.jsonl"
    if not log.exists():
        print(f"no log at {log}; re-run experiment 9 to regenerate it", file=sys.stderr)
        return 2

    vals = collect(log)
    out: dict = {
        "source": f"runs/{args.run}/e9.jsonl, abstention conditions",
        "noul_note": "a noul returns no confidence; 2*|p-0.5| is used as a stand-in",
        "by_question_type": {},
    }
    for qtype in ("choice", "score", "noul"):
        answerable = vals.get((qtype, "answerable")) or []
        if not answerable:
            continue
        arm = {"answerable": quantiles(answerable),
               "false_alarm": {f"below_{g}": sum(1 for x in answerable if x < g) / len(answerable)
                               for g in GATES},
               "kinds": {}}
        for kind in KINDS:
            u = vals.get((qtype, kind)) or []
            if not u:
                continue
            arm["kinds"][kind] = {
                **quantiles(u),
                "separation": sum(answerable) / len(answerable) - sum(u) / len(u),
                "auroc": auroc(answerable, u),
                "caught": {f"below_{g}": sum(1 for x in u if x < g) / len(u) for g in GATES},
            }
        out["by_question_type"][qtype] = arm

    path = ROOT / "runs" / args.run / "abstention_gates.json"
    path.write_text(json.dumps(out, indent=1) + "\n")

    for qtype, arm in out["by_question_type"].items():
        tag = "  (stand-in, not a real confidence)" if qtype == "noul" else ""
        a = arm["answerable"]
        print(f"{qtype}{tag}: answerable n={a['n']}, mean {a['mean']:.3f}, "
              f"{a['at_1_00']:.0%} at exactly 1.00")
        print(f"   {'kind':<16} {'mean':>6} {'sep':>7} {'AUROC':>6} "
              f"{'<0.8 catch':>11} {'<1.00 catch':>12}")
        for kind, k in arm["kinds"].items():
            print(f"   {kind:<16} {k['mean']:>6.3f} {k['separation']:>+7.3f} {k['auroc']:>6.3f} "
                  f"{k['caught']['below_0.8']:>10.1%} {k['caught']['below_1.0']:>11.1%}")
        print(f"   {'false alarms':<16} {'':>6} {'':>7} {'':>6} "
              f"{arm['false_alarm']['below_0.8']:>10.1%} {arm['false_alarm']['below_1.0']:>11.1%}\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
