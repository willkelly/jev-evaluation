"""Independent rescoring of a run from its call log.

The plan's first non-negotiable is that every metric be recomputable from the
log without re-spending calls. The data satisfies that -- ground truth rides in
`meta.truth` on every record and `logstore.answers_of` re-parses the wire
response -- but only E2 implements a `score(run_dir)` of its own, so for the
other eight a redefined metric costs a re-run.

This module closes most of that gap generically rather than by writing eight
bespoke rescorers. It reads any experiment's log, joins each answer to the
ground truth stored beside it, and recomputes accuracy and calibration per
condition. That is not identical to what each experiment computes -- an
experiment's headline is often a derived quantity such as a drift, a forbidden
cell mass or a decay slope, which only it knows how to assemble. But it is
enough for the thing offline rescoring is actually for: checking that the
numbers in the report follow from the calls that were made, and recomputing a
redefined accuracy or ECE on data already paid for.

It is deliberately independent of the experiment modules. It imports none of
them, so a scoring bug in an experiment cannot reproduce itself here -- which is
what makes agreement between the two worth anything.

Usage:

    python -m jeveval.rescore runs/full-20260919
    python -m jeveval.rescore runs/full-20260919 --experiment E1 --verify
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import logstore, metrics


@dataclass
class Observation:
    """One answered question, joined to its ground truth."""

    experiment: str
    condition: str
    instance_id: str
    repetition: int
    question_key: str
    qtype: str
    truth: Any
    predicted: Any
    correct: bool
    # P(the answer that ground truth calls correct), where that is defined:
    # for a noul it is p or 1 - p; for a choice or score it is the probability
    # mass the model put on the true option. This is what calibration needs, and
    # it is not the same as the probability of the chosen answer.
    p_true: float | None
    p_raw: float | None
    confidence: float | None
    latency_s: float
    input_tokens: int
    model_version: str | None


def observations(run_dir: str | Path, experiment: str | None = None) -> Iterator[Observation]:
    """Every scorable answer in a run, joined to truth.

    Records whose truth cannot be joined are skipped and counted by
    `coverage`, not silently dropped: several experiments renumber question keys
    on the wire (E3 renames every key to a uniform format so the target's key
    cannot mark which question is scored), so the wire key and the truth key do
    not always agree, and pretending otherwise would score those as wrong.
    """
    run_dir = Path(run_dir)
    for log in sorted(run_dir.glob("*.jsonl")):
        for record in logstore.successful(logstore.read(log)):
            if experiment and record.get("experiment") != experiment:
                continue
            truth_map = ((record.get("meta") or {}).get("truth")) or {}
            if not isinstance(truth_map, dict):
                continue
            key_map = _key_map(record)
            answers = logstore.answers_of(record)
            for key, answer in answers.items():
                truth_key = key_map.get(key, key)
                if truth_key not in truth_map:
                    continue
                yield _observation(record, key, answer, truth_map[truth_key])


def _key_map(record: dict) -> dict[str, str]:
    """Wire question key -> the key ground truth is stored under.

    An experiment that renames keys on the wire has to record the mapping, or
    its answers cannot be rescored at all. E1 does exactly that: its
    key-renaming condition exists to check the vendor's claim that question ids
    are not used in inference, and one of its schemes ROTATES keys among the
    questions, so a wire key still matches a truth key while belonging to a
    different question. Joining on the key alone there does not fail loudly -- it
    joins every answer to the wrong truth and reports an accuracy of 0.000,
    which reads as a catastrophic model failure rather than as a harness bug.
    Found exactly that way while building this module.
    """
    pres = ((record.get("meta") or {}).get("presentation")) or {}
    km = pres.get("key_map")
    return {str(k): str(v) for k, v in km.items()} if isinstance(km, dict) else {}


def _observation(record: dict, key: str, answer: dict, truth: Any) -> Observation:
    qtype = answer.get("type", "")
    p_raw: float | None = None
    p_true: float | None = None

    if qtype == "noul":
        p_raw = answer.get("p")
        predicted: Any = answer.get("predicted")
        if isinstance(truth, bool) and isinstance(p_raw, (int, float)):
            p_true = float(p_raw) if truth else 1.0 - float(p_raw)
    else:
        predicted = answer.get("chosen")
        probs = answer.get("probabilities") or {}
        p_raw = answer.get("max_probability")
        if truth is not None and str(truth) in probs:
            p_true = float(probs[str(truth)])

    return Observation(
        experiment=record.get("experiment", ""),
        condition=record.get("condition", ""),
        instance_id=record.get("instance_id", ""),
        repetition=int(record.get("repetition") or 0),
        question_key=key,
        qtype=qtype,
        truth=truth,
        predicted=predicted,
        correct=bool(predicted == truth),
        p_true=p_true,
        p_raw=p_raw if isinstance(p_raw, (int, float)) else None,
        confidence=answer.get("confidence"),
        latency_s=float(record.get("latency_s") or 0.0),
        input_tokens=int(record.get("input_tokens") or 0),
        model_version=record.get("model_version"),
    )


def coverage(run_dir: str | Path, experiment: str | None = None) -> dict:
    """How much of the log could be joined to truth, and what could not.

    Reported rather than assumed, because a rescore covering half the answers
    and not saying so is worse than no rescore at all.
    """
    run_dir = Path(run_dir)
    answered = joined = records = 0
    unjoined_keys: dict[str, int] = defaultdict(int)
    for log in sorted(run_dir.glob("*.jsonl")):
        for record in logstore.successful(logstore.read(log)):
            if experiment and record.get("experiment") != experiment:
                continue
            records += 1
            truth_map = ((record.get("meta") or {}).get("truth")) or {}
            answers = logstore.answers_of(record)
            answered += len(answers)
            if not isinstance(truth_map, dict):
                continue
            key_map = _key_map(record)
            for key in answers:
                if key_map.get(key, key) in truth_map:
                    joined += 1
                else:
                    unjoined_keys[f"{record.get('experiment')}:{record.get('condition')}"] += 1
    return {
        "records": records,
        "answers": answered,
        "joined_to_truth": joined,
        "joined_fraction": (joined / answered) if answered else None,
        "unjoined_by_condition": dict(unjoined_keys),
    }


def summarize(obs: list[Observation]) -> dict:
    """Accuracy and calibration for one set of observations.

    Calibration uses `p_true` -- the mass on the answer ground truth calls
    correct -- against an outcome of 1, which is the form ECE is defined for
    here. Using the probability of the *chosen* answer instead would score a
    confidently wrong prediction as well calibrated.
    """
    if not obs:
        return {"n": 0}
    truths = [o.truth for o in obs]
    correct = [o.correct for o in obs]
    acc = sum(correct) / len(correct)
    lo, hi = metrics.wilson_interval(sum(correct), len(correct))

    out: dict[str, Any] = {
        "n": len(obs),
        "accuracy": acc,
        "accuracy_95": [lo, hi],
        "majority_baseline": metrics.majority_baseline(truths),
        "distinct_truths": len({str(t) for t in truths}),
        "question_types": sorted({o.qtype for o in obs}),
        "model_versions": sorted({o.model_version for o in obs if o.model_version}),
    }

    cal = [(o.p_true, 1) for o in obs if o.p_true is not None]
    if len(cal) >= 10:
        ps = [p for p, _ in cal]
        # The outcome for `p_true` is 1 by construction, so ECE against it
        # measures how far the mass on the true answer sits from certainty. For a
        # noul, the equivalent and more familiar form is p against the boolean
        # outcome, which is what this uses when every observation is a noul.
        if all(o.qtype == "noul" for o in obs):
            probs = [float(o.p_raw) for o in obs if o.p_raw is not None and isinstance(o.truth, bool)]
            ys = [1 if o.truth else 0 for o in obs if o.p_raw is not None and isinstance(o.truth, bool)]
            if len(probs) >= 10:
                out["ece"] = metrics.ece(probs, ys)
                out["brier"] = metrics.brier(probs, ys)
                out["auroc"] = metrics.auroc(probs, ys)
                out["reliability_monotone"] = metrics.is_monotone(
                    metrics.reliability_bins(probs, ys)
                )
        out["mean_p_on_true_answer"] = sum(ps) / len(ps)

    lat = sorted(o.latency_s for o in obs)
    out["latency"] = metrics.percentiles(lat) if lat else None
    out["mean_input_tokens"] = sum(o.input_tokens for o in obs) / len(obs)
    return out


def by_condition(run_dir: str | Path, experiment: str | None = None) -> dict[str, dict]:
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for o in observations(run_dir, experiment):
        grouped[f"{o.experiment}/{o.condition}"].append(o)
    return {k: summarize(v) for k, v in sorted(grouped.items())}


def verify(run_dir: str | Path, experiment: str) -> dict:
    """Compare a rescore against what the experiment reported.

    Only checks what is comparable. An experiment's headline is often a derived
    quantity this module cannot reconstruct -- a drift, a decay slope, a
    forbidden-cell mass -- and claiming to have verified one of those would be
    worse than saying nothing. So this reports agreement where the quantities
    are the same and says explicitly where they are not.
    """
    run_dir = Path(run_dir)
    result_path = next(run_dir.glob(f"{experiment.lower()}_result.json"), None)
    reported = json.loads(result_path.read_text()) if result_path else None
    conditions = by_condition(run_dir, experiment)
    cov = coverage(run_dir, experiment)

    checks: list[dict] = []
    if reported:
        h = reported.get("headline") or {}
        metric = str(h.get("metric", ""))
        pooled = summarize(list(observations(run_dir, experiment)))
        if "accuracy" in metric.lower() and pooled.get("accuracy") is not None:
            claimed = h.get("value")
            if isinstance(claimed, (int, float)):
                delta = abs(float(claimed) - pooled["accuracy"])
                checks.append({
                    "quantity": "pooled accuracy vs headline",
                    "reported": claimed,
                    "rescored": pooled["accuracy"],
                    "abs_difference": delta,
                    # A pooled rescore need not equal a headline computed over
                    # one condition, so this is informational unless the
                    # experiment's headline really is the pooled figure.
                    "note": "informational: the headline may be scoped to one condition",
                })
        else:
            checks.append({
                "quantity": metric or "headline",
                "note": "not reconstructible here; it is a derived quantity the "
                        "experiment assembles from its own conditions",
            })

    return {
        "experiment": experiment,
        "coverage": cov,
        "by_condition": conditions,
        "checks": checks,
        "reported_found": reported is not None,
    }


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    if args.verify:
        if not args.experiment:
            ap.error("--verify needs --experiment")
        print(json.dumps(verify(args.run_dir, args.experiment), indent=2, default=str))
        return 0

    cov = coverage(args.run_dir, args.experiment)
    frac = cov["joined_fraction"]
    print(
        f"coverage: {cov['joined_to_truth']:,} of {cov['answers']:,} answers joined to "
        f"truth ({'n/a' if frac is None else f'{frac:.1%}'}) over {cov['records']:,} calls"
    )
    if cov["unjoined_by_condition"]:
        print("  unjoined (wire keys renamed, so truth cannot be matched by key):")
        for k, v in sorted(cov["unjoined_by_condition"].items(), key=lambda x: -x[1])[:8]:
            print(f"    {k}: {v:,}")
    print()
    rows = by_condition(args.run_dir, args.experiment)
    print(f"{'condition':<44} {'n':>7} {'acc':>7} {'major':>7} {'ece':>7} {'auroc':>7}")
    for name, s in rows.items():
        if not s.get("n"):
            continue
        cells = [f"{name[:44]:<44}", f"{s['n']:>7,}"]
        for field in ("accuracy", "majority_baseline", "ece", "auroc"):
            v = s.get(field)
            cells.append(f"{v:>7.3f}" if isinstance(v, (int, float)) else f"{'—':>7}")
        print(" ".join(cells))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
