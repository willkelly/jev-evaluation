#!/usr/bin/env python
"""Measure whether the model favours the first option it is shown.

The prompting guide tells readers not to spend effort shuffling a choice's
options, because shuffling changed no answer. That advice needs its own
qualification: no effect on the answer is not the same as no effect at all, and
a small preference over the first position would be worth knowing about even if
it never changes a decision.

This measures it. For every `choice` answer in the run's logs it reads the
option order as it went out on the wire -- the insertion order of the question's
`criteria` object -- and records the index of the option that came back. A model
indifferent to position chooses index 0 at a rate of 1/k on a k-option question,
so the expected rate over a mixed population is the mean of 1/k, weighted by
how many answers each question contributed.

Only repeats of one instance whose option order varied between those repeats
are counted. That restriction is the whole of the method, and it is not
optional. Where a question sends its options in a fixed order, the rate at
which the first slot is chosen measures how often the model's preferred answer
was placed there, which is a property of whoever built the request. Pooling
across instances does not fix it: E1 asks a two-option satisfiability question
in an order fixed per formula, and because the model answers "satisfiable"
almost always, the first slot comes back 0.600 of the time -- a statement about
where the generator put the word, not about the model. Shuffling the same
instance between repeats breaks that tie, because the option then lands in each
slot at random and identity cannot carry the signal. On E1's shuffled arm the
same question gives 0.510 against a 0.500 expectation.

The logs are 1.3 GB and are not in the repository, so this writes its result to
`runs/<run>/position_bias.json`, which is. That file is what the guide cites and
what `tools/verify_citations.py` checks the guide against.

    python tools/position_bias.py --run full-20260919
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def scan(paths: list[Path]) -> dict:
    # group -> option-set -> set of orders seen, and the chosen-index counts
    orders: dict[tuple, set] = defaultdict(set)
    tally: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])  # n, first, sum(1/k)

    for path in paths:
        with path.open() as fh:
            for line in fh:
                # Cheap reject before paying for a 10 KB JSON parse.
                if '"type": "choice"' not in line:
                    continue
                rec = json.loads(line)
                if rec.get("outcome") != "ok" or rec.get("http_status") != 200:
                    continue
                questions = (rec.get("request") or {}).get("questions") or {}
                answers = (rec.get("response") or {}).get("answers") or {}
                for key, ans in answers.items():
                    if not isinstance(ans, dict) or ans.get("type") != "choice":
                        continue
                    criteria = (questions.get(key) or {}).get("criteria")
                    if not isinstance(criteria, dict) or len(criteria) < 2:
                        continue
                    options = tuple(criteria)
                    chosen = ans.get("choice")
                    if chosen not in options:
                        continue
                    group = (rec.get("experiment"), rec.get("condition"),
                             rec.get("instance_id"), key)
                    orders[group].add(options)
                    t = tally[group]
                    t[0] += 1
                    t[1] += options.index(chosen) == 0
                    t[2] += 1 / len(options)

    shuffled = {g for g, seen in orders.items() if len(seen) > 1}
    out = {"shuffled": [0, 0, 0.0], "fixed": [0, 0, 0.0], "by_group": {}}
    for group, (n, first, expected) in sorted(tally.items(), key=lambda kv: -kv[1][0]):
        bucket = "shuffled" if group in shuffled else "fixed"
        acc = out[bucket]
        acc[0] += n
        acc[1] += first
        acc[2] += expected
        if bucket == "shuffled":
            arm = "/".join(str(g) for g in (group[0], group[1], group[3]))
            roll = out["by_group"].setdefault(arm, {"n": 0, "first": 0, "_exp": 0.0})
            roll["n"] += n
            roll["first"] += first
            roll["_exp"] += expected

    for arm, roll in out["by_group"].items():
        roll["first_rate"] = roll["first"] / roll["n"]
        roll["uniform_rate"] = roll.pop("_exp") / roll["n"]
    out["by_group"] = dict(sorted(out["by_group"].items(), key=lambda kv: -kv[1]["n"]))

    for bucket in ("shuffled", "fixed"):
        n, first, expected = out[bucket]
        lo, hi = wilson(first, n)
        out[bucket] = {
            "n": n, "first": first,
            "first_rate": first / n if n else None,
            "uniform_rate": expected / n if n else None,
            "wilson_95": [lo, hi],
            "excess": (first / n - expected / n) if n else None,
        }
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="full-20260919")
    args = ap.parse_args(argv)

    run = ROOT / "runs" / args.run
    logs = sorted(run.glob("e*.jsonl"))
    if not logs:
        print(f"no logs in {run}; re-run the experiments to regenerate them", file=sys.stderr)
        return 2

    result = scan(logs)
    result["source"] = f"runs/{args.run}/" + ", ".join(p.name for p in logs)
    result["definition"] = (
        "index of the chosen option within the question's criteria as sent, pooled "
        "over repeats of one instance whose option order varied between those repeats; "
        "instances asked in a fixed order are reported separately and are confounded "
        "with option identity"
    )
    out = run / "position_bias.json"
    out.write_text(json.dumps(result, indent=1) + "\n")

    s = result["shuffled"]
    print(f"shuffled within instance: {s['first']:,} of {s['n']:,} answers chose the first option")
    print(f"  rate {s['first_rate']:.4f}  uniform {s['uniform_rate']:.4f}  "
          f"95% CI [{s['wilson_95'][0]:.4f}, {s['wilson_95'][1]:.4f}]")
    f = result["fixed"]
    print(f"fixed order (confounded with option identity, not used): {f['n']:,} answers, rate "
          f"{f['first_rate']:.4f} against uniform {f['uniform_rate']:.4f}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
