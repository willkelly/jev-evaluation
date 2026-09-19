"""Phase 0 -- the smoke test and the first gate.

The plan puts this before everything else and is explicit about why: "Confirm
auth, record the versioned model string, confirm the three question types
round-trip, confirm the semantic control is above chance. If the control fails
here, the harness is broken; fix before proceeding."

The last clause is the point. Every formal domain in this plan -- 3SAT,
reachability, Sudoku -- is off-distribution for this model, so a bad result there
is ambiguous between "hard domain", "broken harness" and "weak model". The
semantic control is the only thing that separates those, so if the control fails
nothing downstream is interpretable and the run stops here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config, metrics
from .client import Call, CallResult, JevClient
from .generators import semantic
from .instances import Instance


def _control_instances(count: int) -> list[Instance]:
    """Easy, unambiguous routing tickets, as a choice over departments."""
    sweep = semantic.difficulty_sweep()
    difficulty = dict(sweep[0])
    difficulty["question_type"] = "choice"
    return semantic.generate(
        difficulty=difficulty, seed=config.seed_for("smoke", "control"), count=count
    )


def _type_roundtrip_instances() -> list[Instance]:
    """One instance of each question type, to prove all three round-trip."""
    sweep = semantic.difficulty_sweep()
    out: list[Instance] = []
    for qtype in ("noul", "choice", "score"):
        difficulty = dict(sweep[0])
        difficulty["question_type"] = qtype
        out.extend(
            semantic.generate(
                difficulty=difficulty,
                seed=config.seed_for("smoke", f"type-{qtype}"),
                count=2,
            )
        )
    return out


def _to_calls(instances: list[Instance], condition: str) -> list[Call]:
    return [
        Call(
            experiment="E0",
            condition=condition,
            instance_id=inst.instance_id,
            state=inst.state,
            questions=inst.questions,
            # Ground truth goes into meta so the JSONL log alone is enough to
            # rescore this offline.
            meta={"truth": inst.truth, "difficulty": inst.difficulty, **inst.meta},
        )
        for inst in instances
    ]


def _score(results: list[CallResult], instances: list[Instance]) -> dict:
    by_id = {i.instance_id: i for i in instances}
    correct = total = 0
    failures = 0
    per_type: dict[str, dict] = {}
    for r in results:
        if not r.ok:
            failures += 1
            continue
        inst = by_id.get(r.call.instance_id)
        if inst is None:
            continue
        for key, answer in r.answers.items():
            truth = inst.truth.get(key)
            qtype = answer["type"]
            got = _predicted_value(answer, qtype)
            hit = got == truth
            total += 1
            correct += int(hit)
            slot = per_type.setdefault(qtype, {"correct": 0, "total": 0})
            slot["correct"] += int(hit)
            slot["total"] += 1
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "failures": failures,
        "per_type": per_type,
    }


def _predicted_value(answer: dict, qtype: str) -> Any:
    if qtype == "noul":
        return answer["predicted"]
    return answer.get("chosen")


def _baselines(instances: list[Instance]) -> dict:
    """Random, majority-class and cheap-heuristic baselines for a condition.

    Derived from the questions and ground truth themselves rather than from a
    meta key, because a generator that names its baseline differently would
    otherwise yield a baseline of zero -- which any result clears, turning the
    gate into a formality. The plan requires accuracy to be reported against the
    majority-class baseline in particular, since a class-imbalanced set makes
    accuracy meaningless on its own.
    """
    per_question_random: list[float] = []
    truths: list[Any] = []
    heuristic_hits = heuristic_total = 0

    for inst in instances:
        for key, q in inst.questions.items():
            if q["type"] == "choice":
                per_question_random.append(1.0 / max(1, len(q.get("options") or [])))
            elif q["type"] == "score":
                per_question_random.append(1.0 / max(1, len(q.get("rubric") or [])))
            else:
                per_question_random.append(0.5)
            truths.append(inst.truth.get(key))
        # The "obvious cheap heuristic" the plan asks for wherever one exists.
        guess = inst.meta.get("baseline_keyword_prediction")
        if guess is not None and len(inst.truth) == 1:
            heuristic_total += 1
            heuristic_hits += int(guess == next(iter(inst.truth.values())))

    random_baseline = (
        sum(per_question_random) / len(per_question_random) if per_question_random else 0.0
    )
    majority = metrics.majority_baseline(truths) if truths else 0.0
    return {
        "random": random_baseline,
        "majority_class": majority,
        "cheap_heuristic": (heuristic_hits / heuristic_total) if heuristic_total else None,
        "cheap_heuristic_n": heuristic_total,
        # "Above chance" means above random or majority-class, whichever is
        # higher. The cheap heuristic is reported beside the result, as the plan
        # requires, but it is deliberately not the gate threshold: on the easy
        # control level keyword matching is perfect, and gating against a perfect
        # baseline would make Phase 0 impossible to pass rather than making it
        # strict.
        "chance": max(random_baseline, majority),
    }


def run(run_dir: Path, *, control_n: int = 20) -> dict:
    """Run Phase 0. Returns a report dict; `passed` is the gate."""
    control = _control_instances(control_n)
    types = _type_roundtrip_instances()

    findings: list[str] = []
    notes: list[str] = []
    with JevClient(run_dir=run_dir, log_name="smoke.jsonl") as client:
        type_results = client.run(
            _to_calls(types, "type-roundtrip"), progress_every=0, label="E0/types"
        )
        control_results = client.run(
            _to_calls(control, "semantic-control"), progress_every=10, label="E0/control"
        )
        summary = client.summary()

    # -- auth and version ------------------------------------------------
    versions = summary["model_versions"]
    if not versions:
        findings.append("No successful call returned a model version string.")
    elif len(versions) > 1:
        findings.append(
            f"More than one model version answered during Phase 0: {versions}. "
            "Results across versions are not comparable."
        )

    # -- three types round-trip ------------------------------------------
    type_score = _score(type_results, types)
    seen_types = set(type_score["per_type"])
    missing = {"noul", "choice", "score"} - seen_types
    if missing:
        findings.append(f"These question types did not round-trip: {sorted(missing)}")

    # -- control above chance --------------------------------------------
    control_score = _score(control_results, control)
    bases = _baselines(control)
    baseline = bases["chance"]
    lo, hi = metrics.wilson_interval(control_score["correct"], control_score["total"])
    above_chance = lo > baseline
    if not above_chance:
        findings.append(
            f"Semantic control accuracy {control_score['accuracy']:.3f} "
            f"(95% CI {lo:.3f}-{hi:.3f}) is not above the {baseline:.3f} chance "
            "baseline. Per the plan, the harness is broken; fix before proceeding."
        )

    # Not a gate failure, but it constrains how the control may be reused.
    heuristic = bases.get("cheap_heuristic")
    if heuristic is not None and heuristic >= 0.95:
        notes.append(
            f"At this difficulty the control set is solvable by keyword matching "
            f"alone (heuristic accuracy {heuristic:.3f}). That is fine for a "
            "tripwire, whose job is to detect a broken harness rather than to be "
            "hard. But experiments that reuse the semantic control as a positive "
            "control should draw from the 'hard' level, where the same heuristic "
            "scores about 0.37 and beating it means something."
        )

    failed_calls = summary["failed_calls"]
    if failed_calls:
        findings.append(f"{failed_calls} call(s) failed outright during Phase 0.")

    report = {
        "phase": "E0 smoke",
        "passed": not findings,
        "model_versions": versions,
        "control": {
            **control_score,
            "baselines": bases,
            "random_baseline": baseline,
            "wilson_95": [lo, hi],
            "above_chance": above_chance,
        },
        "type_roundtrip": type_score,
        "run": summary,
        "findings": findings,
        "notes": notes,
    }
    (Path(run_dir) / "smoke_report.json").write_text(json.dumps(report, indent=2))
    return report


def format_report(report: dict) -> str:
    lines = []
    ok = "PASS" if report["passed"] else "FAIL"
    lines.append(f"Phase 0 smoke test: {ok}")
    lines.append(f"  model version(s):  {', '.join(report['model_versions']) or 'none'}")
    c = report["control"]
    b = c.get("baselines", {})
    heur = b.get("cheap_heuristic")
    lines.append(
        f"  semantic control:  {c['correct']}/{c['total']} = {c['accuracy']:.3f} "
        f"(95% CI {c['wilson_95'][0]:.3f}-{c['wilson_95'][1]:.3f})"
    )
    lines.append(
        f"    baselines:       random {b.get('random', 0):.3f}, "
        f"majority-class {b.get('majority_class', 0):.3f}"
        + (f", keyword heuristic {heur:.3f}" if heur is not None else "")
        + f"  -> gated against chance {c['random_baseline']:.3f}"
    )
    for qtype, s in sorted(report["type_roundtrip"]["per_type"].items()):
        lines.append(f"  {qtype:<6} round-trip: {s['correct']}/{s['total']} correct")
    r = report["run"]
    lines.append(
        f"  calls: {r['total_calls']}  failed: {r['failed_calls']}  "
        f"retries: {r['total_retries']}  "
        f"{r['sustained_req_per_s']:.1f} req/s at concurrency {r['final_concurrency']}"
    )
    lines.append(f"  cost so far: ${r['cost_usd']:.6f}  log: {r['log_path']}")
    for n in report.get("notes", []):
        lines.append(f"  note: {n}")
    for f in report["findings"]:
        lines.append(f"  ! {f}")
    return "\n".join(lines)
