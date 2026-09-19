"""E3 -- batching and parallelism.

The plan calls this the load-bearing experiment and it is. Every architecture
that makes this model economically interesting assumes question 200 of a batch
is answered as well as question 3, and that questions asked together do not
contaminate each other. Five conditions, each isolating one way that assumption
could fail.

1. **Positional decay.** One target question with a known label is embedded at
   position 1, 5, 20, 50, 100, 200 and 255 inside a batch that is always 255
   questions long. The fixed batch size is what makes this a position
   measurement: if the batch grew with the position, "question 200 is answered
   worse" and "a 200-question batch is answered worse" would be the same
   observation and neither could be attributed. Position 255 is swept because
   the rubric's Perfect rung reads `decay_255`; leaving it out would let Perfect
   be awarded on a rule whose main criterion was never measured.

2. **Contamination.** Ten questions about one state, asked together and then one
   at a time against the byte-identical state, so the batch is the only thing
   that changes. Four of the ten are about the support ticket and mutually
   constrain each other -- exactly one department is correct, so a coherent
   answer set is a real constraint. Six are about the maintenance log attached
   to the same state and share no referent with the ticket questions. That is
   the operationalisation of the plan's "related" and "unrelated": related
   questions are ones whose answers constrain each other, unrelated ones merely
   share a call.

3. **Interference.** Nine easy questions with and without one very hard 3SAT
   question at the phase transition, against an identical state in both arms.
   Only the question set differs, so a difference in easy-question accuracy is
   attributable to the hard question's presence and nothing else.

4. **Order versus count.** Two arms, because shuffling alone does not separate
   the two things the plan wants separated. Arm one holds the batch at 255 and
   shuffles the whole question list, so the target lands at a random position;
   comparing that against condition 1's curve tells you whether the effect is
   ordinal position or something about a designated slot. The comparison is made
   position by position rather than on pooled accuracy: a shuffle lands uniformly
   over 1..255 while the designated sweep spends five of its seven positions
   below 100, so pooled accuracy differs between the two arms under pure
   positional decay and no placement effect at all. Arm two is condition 5
   read for accuracy: the target sits at position 1 while the question count
   varies, which is the count axis with position held fixed. Together they
   separate "where the question sits" from "how many questions exist".

5. **Latency and cost.** p50/p95 wall clock, input tokens and cost per decision
   against question count from 1 to 255, with the state held byte-identical
   across every count so that the only thing varying is the questions. This is
   the empirical test of the marginal-cost-of-question-200-is-zero claim, and
   the slope is fitted rather than eyeballed.

Conditions 4 and 5 share their calls: the count sweep answers both, and running
it once rather than twice saves 3,000 calls for no loss of information.

Three things worth stating because they shape the numbers.

*The positional-decay target is a Dyck balance question, chosen by measurement
rather than by argument.* A target the model answers perfectly has no headroom
for decay to appear in, so a flat curve would be evidence of a ceiling rather
than of parallelism -- and this experiment's whole job is to decide whether the
architecture is genuinely parallel. The semantic control was the obvious base
task and turns out to be unusable: at its hard level this model scores 0.958 on
48 held-out instances, and E8 independently measured 40/40 with AUROC 1.00 on
the same task. Four points of room cannot show a 5% decay.

Candidates were measured against this model before choosing:

    semantic/hard choice     accuracy 0.958   headroom 0.04   rejected
    dyck length 24 depth 4   accuracy 0.833   headroom 0.17   chosen
    graphreach 3 hops        accuracy 0.688   headroom 0.31

Being hard is not sufficient on its own. E2 measured that this model answers
"satisfiable" for 100% of 3SAT instances at every ratio, and a model that
returns a constant shows zero decay no matter where the question sits, so a
3SAT target would produce the same false confirmation of P7 by the opposite
route. The chosen target is both below ceiling and demonstrably sensitive to
its state.

The contamination condition still uses the semantic control, because it
measures probability drift between batched and unbatched answers rather than
accuracy, and drift is measurable at a ceiling.

*Drift is measured as total-variation distance between the batched and
unbatched answer distributions.* A noul answer carries no confidence field --
only choice and score do -- so there is no single "confidence" quantity that
spans the three question types. Total variation is the one that does: for a
noul it is exactly |p_batched - p_unbatched|, which is what the rubric's `drift`
glossary asks for, and for a choice or score it is the natural generalisation
over the returned distribution. Both the probability drift and the rate at which
the discrete answer changes are reported, and P8 is scored against the
probability drift with that stated in its outcome.

*Latency is measured under our own concurrent load.* The client keeps several
requests in flight, so absolute p50 and p95 include queuing behind this
harness's other calls. Every count is measured under the same load and the
counts are interleaved in submission order, so the slope against question count
-- the quantity P9 is about -- is not biased by it, but the absolute numbers
should not be read as single-request latency. The mean in-flight concurrency
during the condition is reported beside them.

The position sweep is the one place in this module where question order is not
randomised, because question order is the independent variable. Everywhere else
the question order is shuffled per instance, and option order inside every
choice is shuffled by the generator that built it.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .. import config, metrics, plots, predictions, tiers
from ..client import Call, CallResult, JevClient
from ..generators import dyck, filler, sat3, semantic
from ..instances import Instance, noul, rng_for

EXPERIMENT = "E3"

# The full batch every position-sweep and count-sweep call is drawn from. 255 is
# the documented cap on choice options rather than on questions, but it is also
# the largest count the plan asks the latency curve to reach, so it is the
# natural top of the range and keeps the two conditions on one axis.
BATCH_QUESTIONS = 255

# The plan's positions, plus 255 so the rubric's Perfect rung has its criterion.
POSITIONS: tuple[int, ...] = (1, 5, 20, 50, 100, 200, 255)

# Question counts for the latency, cost and accuracy-against-count sweep.
COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 200, 255)

# Contamination: four questions about the ticket, six about the maintenance log.
# Four rather than five so the two nouls are exactly balanced within every single
# instance -- one true, one false -- which makes the per-instance majority-class
# baseline on the noul half exactly 0.5 rather than 2/3.
RELATED_QUESTIONS = 4
UNRELATED_QUESTIONS = 6
CONTAMINATION_QUESTIONS = RELATED_QUESTIONS + UNRELATED_QUESTIONS

# Interference: one hard question against nine easy ones.
EASY_QUESTIONS = 9

# 3SAT at the phase transition. 4.25 rather than 4.26 because the generator's
# ratio ladder is built from exact quarters, and an inexact ratio would change
# which instances the difficulty dict draws.
SAT_RATIO = 4.25
SAT_VARS = 20

# The semantic control's hard level, still used by the contamination condition,
# which measures probability drift rather than accuracy and so is not blocked by
# a ceiling.
TARGET_LEVEL = "hard"

# The positional-decay target is a Dyck balance question, not a routing ticket.
# Measured on 48 fresh instances of each candidate against this model:
#
#   semantic/hard choice     accuracy 0.958   headroom 0.04
#   dyck length 24 depth 4   accuracy 0.833   headroom 0.17
#   graphreach 3 hops        accuracy 0.688   headroom 0.31
#
# Routing leaves 4 points of room, so a 5% decay is inside the measurement and
# P7 ("no positional decay") would score right whether or not decay exists --
# on the plan's own load-bearing experiment.
#
# 3SAT is unusable here despite being hard for a different reason: E2 measured
# that this model answers "satisfiable" for 100% of instances at every ratio,
# and a constant answerer shows zero decay by construction. A usable target has
# to be both below ceiling and actually sensitive to its state, which is what
# the "const" column of that measurement checks.
TARGET_GENERATOR = dyck
TARGET_DIFFICULTY = {"length": 24, "max_depth": 4}
TARGET_HEADROOM_NOTE = (
    "Positional-decay target is dyck(length=24, max_depth=4), measured at "
    "accuracy 0.833 against a 0.500 majority baseline on 48 held-out instances. "
    "The semantic control's hard level was rejected as the target: it measured "
    "0.958, leaving too little room for a decay of the size the plan cares about "
    "to be visible."
)

# Filler accuracy is also read as a function of each filler's own position,
# which gives a 255-point positional curve for free from the same calls. Binned
# because a single position holds a different question in every instance.
FILLER_POSITION_BINS = 10

# The plan's Human rung, "<5% decay to 50", is the bar the knee is measured
# against: the knee is where decay first exceeds it by more than sampling noise.
KNEE_DECAY = 0.05

# The plan's Phase 2 gate: "if positional decay exceeds 10% by question 50, cap
# batch size at the measured knee and re-run E5 and E6 within that cap."
GATE_DECAY_50 = 0.10


# --------------------------------------------------------------------------
# Small numerical helpers
# --------------------------------------------------------------------------


def _ols(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """(slope, intercept) of an ordinary least-squares fit, or None."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def _loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Slope of log(y) against log(x). 0 is flat, 1 is linear.

    This is the quantity the E3 rubric grades latency on, and it is fitted
    rather than read off the plot because "roughly flat" and "scales linearly"
    are two ends of one number.
    """
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    fit = _ols([p[0] for p in pts], [p[1] for p in pts])
    return None if fit is None else fit[0]


def _diff_ci(k1: int, n1: int, k2: int, n2: int, conf: float = 0.95) -> tuple[float, float] | None:
    """Newcombe's hybrid-score interval for p1 - p2, built from two Wilsons.

    Used for decay, which is a difference of two accuracies. The two positions
    are measured on the same targets, so treating them as independent is
    conservative -- the paired interval would be narrower -- and a conservative
    interval is the right way round for deciding whether a decay is real.
    """
    if n1 <= 0 or n2 <= 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = metrics.wilson_interval(k1, n1, conf)
    l2, u2 = metrics.wilson_interval(k2, n2, conf)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def _rate(k: int, n: int) -> float | None:
    return (k / n) if n else None


def _mean(xs: Sequence[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


# --------------------------------------------------------------------------
# Answer handling
# --------------------------------------------------------------------------


def _predicted(answer: dict) -> Any:
    """The discrete answer, comparable to ground truth for all three types."""
    if answer["type"] == "noul":
        return answer["predicted"]
    return answer.get("chosen")


def _distribution(answer: dict) -> dict[str, float]:
    """The answer as a probability vector keyed by outcome id.

    A noul is rendered as a two-outcome distribution so total-variation distance
    means the same thing for it as for a choice, and reduces to |delta p|.
    """
    if answer["type"] == "noul":
        p = float(answer["p"])
        return {"true": p, "false": 1.0 - p}
    return {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}


def _total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _p_correct(answer: dict, truth: Any) -> float | None:
    """Probability mass the answer put on the correct outcome.

    The signed version of drift. A batched answer that moves mass toward the
    truth is the coherence feature the plan says to exploit; one that moves mass
    away is contamination. Accuracy alone cannot tell those apart when neither
    side crosses the decision threshold.
    """
    dist = _distribution(answer)
    if answer["type"] == "noul":
        return dist["true"] if truth else dist["false"]
    return dist.get(str(truth))


# --------------------------------------------------------------------------
# Material: targets, filler blocks, batches
# --------------------------------------------------------------------------


def _target_instances(n: int) -> list[Instance]:
    """`n` target instances with one question each, chosen for headroom.

    See TARGET_HEADROOM_NOTE. The instance carries exactly one question, so
    callers take its key rather than naming a generator-specific constant --
    which is what lets the target task be swapped without touching the three
    conditions that consume it.
    """
    return TARGET_GENERATOR.generate(
        difficulty=dict(TARGET_DIFFICULTY),
        seed=config.seed_for(EXPERIMENT, "targets"),
        count=n,
    )


def _target_key(inst: Instance) -> str:
    """The instance's single question key."""
    keys = list(inst.questions)
    if len(keys) != 1:
        raise ValueError(
            f"target instance {inst.instance_id} has {len(keys)} questions; "
            "the positional-decay conditions assume exactly one"
        )
    return keys[0]


def _block(index: int, count: int, tag: str) -> filler.FillerBlock:
    return filler.filler_block(
        seed=config.seed_for(EXPERIMENT, f"filler-{tag}"), count=count, index=index
    )


def _trim(block: filler.FillerBlock, count: int) -> filler.FillerBlock:
    """The first `count` questions of a block, over the block's whole log.

    The text and records are kept whole on purpose. Asking fewer questions must
    not also shrink the state, or the count sweep would be measuring state size
    and question count at once and the latency slope would be attributable to
    neither.
    """
    keys = list(block.questions)[:count]
    return filler.FillerBlock(
        text=block.text,
        records=block.records,
        questions={k: block.questions[k] for k in keys},
        truth={k: block.truth[k] for k in keys},
        meta=block.meta,
    )


def _assemble(
    items: Sequence[tuple[dict, Any]], key_format: str = "q{:03d}"
) -> tuple[dict[str, dict], dict[str, Any], list[str]]:
    """Number an ordered list of (question, truth) pairs into a question map.

    Keys are assigned after ordering, so a key never records where a question
    started out. The plan forbids ids from encoding the answer; a key that
    encoded the pre-shuffle slot would go further and mark which question is
    being scored.
    """
    questions: dict[str, dict] = {}
    truth: dict[str, Any] = {}
    keys: list[str] = []
    for i, (q, t) in enumerate(items):
        key = key_format.format(i)
        questions[key] = q
        truth[key] = t
        keys.append(key)
    return questions, truth, keys


def _question_types(questions: dict[str, dict]) -> dict[str, str]:
    return {k: q["type"] for k, q in questions.items()}


def _rubric_ids(questions: dict[str, dict]) -> dict[str, list[str]]:
    """Ordered rubric ids for each score question, keyed as the question is sent.

    Truth for a score question is a rubric *id*, but the wire form carries only
    the rubric's labels and the response identifies its answer by index, so the
    id never reaches the log on its own. `logstore.answers_of` therefore returns
    a label offline where the live path returned an id, and every score question
    rescores as wrong -- a fifth of the filler in this experiment. Recording the
    ordered ids restores the missing half of the mapping: offline, the winning
    key of `index_probabilities` indexes into this list and the result compares
    against `meta["truth"]`.

    Only score questions appear. Noul truth is a bool and choice truth is the
    option id the response already returns, so both rescore without help.
    """
    return {
        key: [str(point["id"]) for point in q["rubric"]]
        for key, q in questions.items()
        if q["type"] == "score"
    }


def _random_rate(q: dict) -> float:
    """Accuracy of guessing uniformly on one question."""
    if q["type"] == "choice":
        return metrics.random_baseline(max(1, len(q.get("options") or [])))
    if q["type"] == "score":
        return metrics.random_baseline(max(1, len(q.get("rubric") or [])))
    return 0.5


def _chance_baseline(questions: dict[str, dict], truths: Sequence[Any]) -> float:
    """Chance for a question set: the better of random guessing and majority class.

    Random is averaged over the questions because a mixed batch has a different
    option count per question. Majority class comes from the labels actually
    drawn, which is what makes a class-imbalanced condition legible.
    """
    rand = _mean([_random_rate(q) for q in questions.values()]) or 0.0
    majority = metrics.majority_baseline(list(truths)) if truths else 0.0
    return max(rand, majority)


# --------------------------------------------------------------------------
# Condition 1: positional decay
# --------------------------------------------------------------------------


def _position_calls(
    targets: Sequence[Instance],
    blocks: Sequence[filler.FillerBlock],
    states: Sequence[Any],
) -> list[Call]:
    calls: list[Call] = []
    for i, (inst, block, state) in enumerate(zip(targets, blocks, states)):
        key = _target_key(inst)
        target_q = inst.questions[key]
        target_truth = inst.truth[key]
        for position in POSITIONS:
            batch = filler.batch_with_target(
                block=block, target=target_q, target_truth=target_truth, position=position
            )
            calls.append(
                Call(
                    experiment=EXPERIMENT,
                    condition="position",
                    instance_id=inst.instance_id,
                    state=state,
                    # Not shuffled: question order is the independent variable.
                    questions=batch.questions,
                    meta={
                        "truth": batch.truth,
                        "types": _question_types(batch.questions),
                        "rubric_ids": _rubric_ids(batch.questions),
                        "target_key": batch.target_key,
                        "target_truth": target_truth,
                        "position": position,
                        "n_questions": len(batch.questions),
                        "target_index": i,
                        "difficulty": {"position": position},
                        "department": inst.meta.get("department"),
                        "baseline_keyword_prediction": inst.meta.get(
                            "baseline_keyword_prediction"
                        ),
                    },
                )
            )
    return calls


def _analyse_positions(results: Sequence[CallResult]) -> dict:
    per: dict[int, dict] = defaultdict(
        lambda: {
            "correct": 0,
            "n": 0,
            "truths": [],
            "heuristic_correct": 0,
            "filler_correct": 0,
            "filler_n": 0,
        }
    )
    filler_bins: dict[int, dict] = defaultdict(lambda: {"correct": 0, "n": 0, "positions": []})
    # The filler chance baseline is accumulated as running totals rather than by
    # keeping every question: a full sweep asks 2,100 x 254 filler questions and
    # the baseline only needs each one's option count and its label.
    filler_random_total = 0.0
    filler_random_n = 0
    filler_label_counts: Counter = Counter()
    target_questions: dict[str, dict] = {}

    for r in results:
        if not r.ok:
            continue
        meta = r.call.meta
        position = int(meta["position"])
        target_key = meta["target_key"]
        truth = meta["truth"]
        slot = per[position]

        answer = r.answers.get(target_key)
        if answer is not None:
            hit = _predicted(answer) == meta["target_truth"]
            slot["correct"] += int(hit)
            slot["n"] += 1
            slot["truths"].append(meta["target_truth"])
            slot["heuristic_correct"] += int(
                meta.get("baseline_keyword_prediction") == meta["target_truth"]
            )
            target_questions[target_key] = r.call.questions[target_key]

        for j, (key, ans) in enumerate(r.answers.items(), start=1):
            if key == target_key:
                continue
            hit = _predicted(ans) == truth.get(key)
            slot["filler_correct"] += int(hit)
            slot["filler_n"] += 1
            filler_random_total += _random_rate(r.call.questions[key])
            filler_random_n += 1
            filler_label_counts[truth.get(key)] += 1
            b = min(FILLER_POSITION_BINS - 1, (j - 1) * FILLER_POSITION_BINS // BATCH_QUESTIONS)
            filler_bins[b]["correct"] += int(hit)
            filler_bins[b]["n"] += 1
            filler_bins[b]["positions"].append(j)

    positions = [p for p in POSITIONS if per[p]["n"]]
    accuracy = {p: per[p]["correct"] / per[p]["n"] for p in positions}
    wilson = {p: metrics.wilson_interval(per[p]["correct"], per[p]["n"]) for p in positions}

    all_truths = [t for p in positions for t in per[p]["truths"]]
    chance = (
        _chance_baseline(target_questions, all_truths)
        if target_questions and all_truths
        else None
    )
    heuristic = _rate(
        sum(per[p]["heuristic_correct"] for p in positions),
        sum(per[p]["n"] for p in positions),
    )

    decay: dict[int, float] = {}
    decay_ci: dict[int, tuple[float, float] | None] = {}
    if 1 in accuracy:
        base_k, base_n = per[1]["correct"], per[1]["n"]
        for p in positions:
            decay[p] = accuracy[1] - accuracy[p]
            decay_ci[p] = _diff_ci(base_k, base_n, per[p]["correct"], per[p]["n"])

    filler_curve = []
    for b in sorted(filler_bins):
        slot = filler_bins[b]
        if not slot["n"]:
            continue
        filler_curve.append(
            {
                "bin": b,
                "mean_position": sum(slot["positions"]) / len(slot["positions"]),
                "accuracy": slot["correct"] / slot["n"],
                "n": slot["n"],
                "wilson": list(metrics.wilson_interval(slot["correct"], slot["n"])),
            }
        )

    filler_n = sum(per[p]["filler_n"] for p in positions)
    filler_chance = (
        max(
            filler_random_total / filler_random_n,
            max(filler_label_counts.values()) / sum(filler_label_counts.values()),
        )
        if filler_random_n
        else None
    )

    return {
        "positions": positions,
        "accuracy": accuracy,
        "correct": {p: per[p]["correct"] for p in positions},
        "wilson": {p: list(wilson[p]) for p in positions},
        "n": {p: per[p]["n"] for p in positions},
        "decay": decay,
        "decay_ci": {p: (list(c) if c else None) for p, c in decay_ci.items()},
        "chance_baseline": chance,
        "heuristic_baseline": heuristic,
        "filler_accuracy": _rate(sum(per[p]["filler_correct"] for p in positions), filler_n),
        "filler_n": filler_n,
        "filler_chance_baseline": filler_chance,
        "filler_accuracy_by_own_position": filler_curve,
    }


def _repeatability(position: dict, counts: dict) -> dict:
    """Position 1 against count 255: the same request, sent twice.

    Both conditions put the target first in a 255-question batch over the same
    state with the same fillers in the same order, so any difference between
    them is the service answering an identical request differently. E1 measures
    that properly; this is a cross-check that costs nothing and catches the case
    where a positional result is really run-to-run variance.
    """
    a = position.get("accuracy", {}).get(1)
    row = next((r for r in (counts.get("rows") or []) if r["questions"] == BATCH_QUESTIONS), None)
    b = row["target_accuracy"] if row else None
    if a is None or b is None:
        return {"comparable": False, "reason": "one of the two conditions produced no answers"}
    return {
        "comparable": True,
        "position_sweep_at_1": a,
        "count_sweep_at_255": b,
        "difference": a - b,
        "n": [position["n"].get(1), row["target_n"]],
        "note": "byte-identical requests in the two conditions; a non-zero difference is "
        "run-to-run variance, not a batching effect",
    }


def _knee(analysis: dict) -> dict:
    """Where accuracy first falls more than the Human bar below position 1.

    Two knees are reported. The nominal one is the first position whose decay
    exceeds 5%. The resolved one additionally requires the decay's confidence
    interval to exclude zero, so a 6% drop measured on 40 targets -- which is
    inside the noise -- is not reported as a batch-size cap that downstream
    experiments then have to respect.
    """
    decay = analysis.get("decay") or {}
    positions = [p for p in analysis.get("positions", []) if p != 1]
    nominal = next((p for p in positions if decay.get(p, 0.0) > KNEE_DECAY), None)
    resolved = next(
        (
            p
            for p in positions
            if decay.get(p, 0.0) > KNEE_DECAY
            and (analysis["decay_ci"].get(p) or [None, None])[0] is not None
            and analysis["decay_ci"][p][0] > 0.0
        ),
        None,
    )
    swept = analysis.get("positions") or []
    below = [p for p in swept if nominal is None or p < nominal]
    safe = max(below) if below else None
    # `resolved` is the first position that clears both tests, which is not
    # necessarily the nominal knee: a sweep can put its first >5% drop at
    # position 20 on an interval that includes zero and only resolve one at 100.
    # `safe_batch_size` is derived from the nominal knee, so a cap built from it
    # rests on that unresolved drop, and saying so is the difference between a
    # cap E5 and E6 have to respect and a number to sharpen first.
    return {
        "knee_position": nominal,
        "knee_position_resolved": resolved,
        "nominal_knee_resolved": (resolved == nominal) if nominal is not None else None,
        "safe_batch_size": safe,
        # The sweep is coarse, so the knee is known only to lie between the last
        # clean position and the first dirty one. Reporting the bracket keeps a
        # cap of "1" from reading as a measurement when what was measured is
        # "somewhere between 1 and 5".
        "knee_bracket": [safe, nominal] if nominal is not None else None,
        "knee_criterion": (
            f"first swept position whose accuracy is more than {KNEE_DECAY:.0%} below "
            "position 1; 'resolved' additionally requires the 95% interval on that "
            "decay to exclude zero"
        ),
        "max_position_swept": max(swept) if swept else None,
    }


# --------------------------------------------------------------------------
# Condition 2: contamination
# --------------------------------------------------------------------------


def _contamination_material(n: int) -> list[dict]:
    """One ticket per instance, four questions about it and six about its log."""
    seed = config.seed_for(EXPERIMENT, "contamination")
    instances = semantic.generate(
        difficulty={"level": TARGET_LEVEL, "question_type": "all"}, seed=seed, count=n
    )
    out = []
    for i, inst in enumerate(instances):
        rng = rng_for(EXPERIMENT, {"condition": "contamination"}, seed, i)
        block = _block(i, UNRELATED_QUESTIONS, "contamination")
        state = filler.attach(inst.state, block)

        asked = inst.meta["asked_department"]
        truth_dept = inst.meta["department"]
        # The generator's noul alternates polarity by index. The second noul
        # takes the opposite polarity so every instance carries exactly one true
        # and one false yes/no about its own routing, which is also what makes
        # the four related questions mutually constraining: at most one
        # department can be the right one.
        if asked == truth_dept:
            pool = [d for d in semantic.DEPARTMENTS if d != truth_dept]
            second = pool[rng.randrange(len(pool))]
        else:
            second = truth_dept
        second_q = noul(semantic.NOUL_QUESTION.format(label=semantic.DEPARTMENT_LABELS[second]))

        related: list[tuple[dict, Any, str]] = [
            (inst.questions[semantic.KEY_CHOICE], inst.truth[semantic.KEY_CHOICE], "choice"),
            (inst.questions[semantic.KEY_SCORE], inst.truth[semantic.KEY_SCORE], "score"),
            (inst.questions[semantic.KEY_NOUL], inst.truth[semantic.KEY_NOUL], "noul"),
            (second_q, second == truth_dept, "noul"),
        ]
        unrelated = [(block.questions[k], block.truth[k], block.questions[k]["type"]) for k in block.questions]

        items = [(q, t) for q, t, _ in related] + [(q, t) for q, t, _ in unrelated]
        groups = ["related"] * len(related) + ["unrelated"] * len(unrelated)
        order = list(range(len(items)))
        rng.shuffle(order)
        questions, truth, keys = _assemble([items[j] for j in order])
        group_map = {keys[pos]: groups[j] for pos, j in enumerate(order)}

        out.append(
            {
                "instance_id": inst.instance_id,
                "state": state,
                "questions": questions,
                "truth": truth,
                "groups": group_map,
            }
        )
    return out


def _contamination_calls(material: Sequence[dict]) -> list[Call]:
    calls: list[Call] = []
    for item in material:
        common = {
            "truth": item["truth"],
            "groups": item["groups"],
            "types": _question_types(item["questions"]),
            "rubric_ids": _rubric_ids(item["questions"]),
        }
        calls.append(
            Call(
                experiment=EXPERIMENT,
                condition="contamination-batched",
                instance_id=item["instance_id"],
                state=item["state"],
                questions=item["questions"],
                meta={**common, "n_questions": len(item["questions"])},
            )
        )
        for key, q in item["questions"].items():
            calls.append(
                Call(
                    experiment=EXPERIMENT,
                    condition="contamination-individual",
                    instance_id=item["instance_id"],
                    state=item["state"],
                    questions={key: q},
                    meta={
                        "truth": {key: item["truth"][key]},
                        "groups": {key: item["groups"][key]},
                        "types": {key: q["type"]},
                        "rubric_ids": _rubric_ids({key: q}),
                        "key": key,
                        "n_questions": 1,
                    },
                )
            )
    return calls


def _analyse_contamination(results: Sequence[CallResult]) -> dict:
    batched: dict[tuple[str, str], tuple[dict, Any, str]] = {}
    individual: dict[tuple[str, str], tuple[dict, Any, str]] = {}
    for r in results:
        if not r.ok:
            continue
        meta = r.call.meta
        if r.call.condition == "contamination-batched":
            for key, ans in r.answers.items():
                batched[(r.call.instance_id, key)] = (
                    ans,
                    meta["truth"].get(key),
                    meta["groups"].get(key),
                )
        else:
            key = meta["key"]
            ans = r.answers.get(key)
            if ans is not None:
                individual[(r.call.instance_id, key)] = (
                    ans,
                    meta["truth"].get(key),
                    meta["groups"].get(key),
                )

    buckets: dict[str, dict] = defaultdict(
        lambda: {
            "tv": [],
            "delta_p_correct": [],
            "answer_changed": 0,
            "n": 0,
            "batched_correct": [],
            "individual_correct": [],
        }
    )
    paired = 0
    for pair_key, (b_ans, truth, group) in batched.items():
        if pair_key not in individual:
            continue
        i_ans, _, _ = individual[pair_key]
        paired += 1
        tv = _total_variation(_distribution(b_ans), _distribution(i_ans))
        bp, ip = _p_correct(b_ans, truth), _p_correct(i_ans, truth)
        changed = _predicted(b_ans) != _predicted(i_ans)
        for name in ("all", group or "unknown", f"{b_ans['type']}"):
            slot = buckets[name]
            slot["tv"].append(tv)
            if bp is not None and ip is not None:
                slot["delta_p_correct"].append(bp - ip)
            slot["answer_changed"] += int(changed)
            slot["n"] += 1
            slot["batched_correct"].append(_predicted(b_ans) == truth)
            slot["individual_correct"].append(_predicted(i_ans) == truth)

    out: dict[str, Any] = {"paired_questions": paired, "by_group": {}}
    for name, slot in buckets.items():
        n = slot["n"]
        if not n:
            continue
        acc_b = metrics.accuracy(slot["batched_correct"])
        acc_i = metrics.accuracy(slot["individual_correct"])
        entry = {
            "n": n,
            "probability_drift": _mean(slot["tv"]),
            "answer_drift": slot["answer_changed"] / n,
            "mean_delta_p_correct": _mean(slot["delta_p_correct"]),
            "accuracy_batched": acc_b,
            "accuracy_individual": acc_i,
            "accuracy_delta": acc_b - acc_i,
            "mcnemar_p": metrics.mcnemar(slot["batched_correct"], slot["individual_correct"]),
            "direction": (
                "batched better" if acc_b > acc_i else "batched worse" if acc_b < acc_i else "no change"
            ),
        }
        out["by_group"][name] = entry
    overall = out["by_group"].get("all", {})
    out["probability_drift"] = overall.get("probability_drift")
    out["answer_drift"] = overall.get("answer_drift")
    out["unbatched_accuracy"] = overall.get("accuracy_individual")
    out["batched_accuracy"] = overall.get("accuracy_batched")
    return out


# --------------------------------------------------------------------------
# Condition 3: interference
# --------------------------------------------------------------------------


def _interference_material(n: int) -> list[dict]:
    seed = config.seed_for(EXPERIMENT, "interference")
    hard = sat3.generate(
        difficulty={"n": SAT_VARS, "ratio": SAT_RATIO}, seed=seed, count=n, encoding="clauses"
    )
    out = []
    for i, inst in enumerate(hard):
        rng = rng_for(EXPERIMENT, {"condition": "interference"}, seed, i)
        block = _block(i, EASY_QUESTIONS, "interference")
        # The state carries the formula and the log in both arms. Only the
        # question set differs, so nothing but the hard question's presence can
        # explain a difference in easy-question accuracy.
        state = filler.attach(inst.state, block)

        easy = [(block.questions[k], block.truth[k]) for k in block.questions]
        order = list(range(len(easy)))
        rng.shuffle(order)
        easy = [easy[j] for j in order]

        hard_q = inst.questions[sat3.QUESTION_KEY]
        hard_truth = inst.truth[sat3.QUESTION_KEY]
        slot = rng.randrange(len(easy) + 1)
        with_items = list(easy)
        with_items.insert(slot, (hard_q, hard_truth))

        with_q, with_t, with_keys = _assemble(with_items)
        without_q, without_t, without_keys = _assemble(easy)
        out.append(
            {
                "instance_id": inst.instance_id,
                "state": state,
                "with": {
                    "questions": with_q,
                    "truth": with_t,
                    "hard_key": with_keys[slot],
                    "easy_slots": {k: j for j, k in enumerate(k2 for k2 in with_keys if k2 != with_keys[slot])},
                },
                "without": {
                    "questions": without_q,
                    "truth": without_t,
                    "easy_slots": {k: j for j, k in enumerate(without_keys)},
                },
                "hard_truth": hard_truth,
                "hard_position": slot + 1,
                "density_prediction": inst.meta.get("baseline_density_pred"),
            }
        )
    return out


def _interference_calls(material: Sequence[dict]) -> list[Call]:
    calls: list[Call] = []
    for item in material:
        for arm in ("with", "without"):
            side = item[arm]
            meta = {
                "truth": side["truth"],
                "types": _question_types(side["questions"]),
                "rubric_ids": _rubric_ids(side["questions"]),
                "easy_slots": side["easy_slots"],
                "arm": arm,
                "n_questions": len(side["questions"]),
                "hard_truth": item["hard_truth"],
                "density_prediction": item["density_prediction"],
            }
            if arm == "with":
                meta["hard_key"] = side["hard_key"]
                meta["hard_position"] = item["hard_position"]
            calls.append(
                Call(
                    experiment=EXPERIMENT,
                    condition=f"interference-{arm}",
                    instance_id=item["instance_id"],
                    state=item["state"],
                    questions=side["questions"],
                    meta=meta,
                )
            )
    return calls


def _analyse_interference(results: Sequence[CallResult]) -> dict:
    easy: dict[str, dict[tuple[str, int], bool]] = {"with": {}, "without": {}}
    hard_correct: list[bool] = []
    hard_truths: list[Any] = []
    hard_density_correct = 0
    hard_questions: dict[str, dict] = {}

    for r in results:
        if not r.ok:
            continue
        meta = r.call.meta
        arm = meta["arm"]
        slots = meta["easy_slots"]
        for key, ans in r.answers.items():
            if arm == "with" and key == meta.get("hard_key"):
                hit = _predicted(ans) == meta["hard_truth"]
                hard_correct.append(hit)
                hard_truths.append(meta["hard_truth"])
                hard_density_correct += int(meta.get("density_prediction") == meta["hard_truth"])
                hard_questions[key] = r.call.questions[key]
                continue
            slot = slots.get(key)
            if slot is None:
                continue
            easy[arm][(r.call.instance_id, slot)] = _predicted(ans) == meta["truth"].get(key)

    shared = sorted(set(easy["with"]) & set(easy["without"]))
    a = [easy["with"][k] for k in shared]
    b = [easy["without"][k] for k in shared]

    out: dict[str, Any] = {
        "paired_easy_questions": len(shared),
        "easy_accuracy_with_hard": metrics.accuracy(a) if a else None,
        "easy_accuracy_without_hard": metrics.accuracy(b) if b else None,
        "mcnemar_p": metrics.mcnemar(a, b) if a else None,
        "hard_n": len(hard_correct),
        "hard_accuracy": metrics.accuracy(hard_correct) if hard_correct else None,
        "hard_majority_baseline": (
            metrics.majority_baseline(hard_truths) if hard_truths else None
        ),
        "hard_density_baseline": _rate(hard_density_correct, len(hard_correct)),
    }
    if out["easy_accuracy_with_hard"] is not None:
        out["easy_accuracy_delta"] = (
            out["easy_accuracy_with_hard"] - out["easy_accuracy_without_hard"]
        )
        out["interference_ci"] = (
            list(c)
            if (
                c := _diff_ci(sum(a), len(a), sum(b), len(b))
            )
            else None
        )
    return out


# --------------------------------------------------------------------------
# Conditions 4 and 5: order, count, latency and cost
# --------------------------------------------------------------------------


def _count_calls(
    targets: Sequence[Instance],
    blocks: Sequence[filler.FillerBlock],
    states: Sequence[Any],
) -> list[Call]:
    """Target at position 1, question count swept, state held identical."""
    calls: list[Call] = []
    for i, (inst, block, state) in enumerate(zip(targets, blocks, states)):
        key = _target_key(inst)
        target_q = inst.questions[key]
        target_truth = inst.truth[key]
        for count in COUNTS:
            batch = filler.batch_with_target(
                block=_trim(block, count - 1),
                target=target_q,
                target_truth=target_truth,
                position=1,
            )
            calls.append(
                Call(
                    experiment=EXPERIMENT,
                    condition="count",
                    instance_id=inst.instance_id,
                    state=state,
                    # Position is held at 1 on purpose: this condition is the
                    # count axis, so the position must not move with it.
                    questions=batch.questions,
                    meta={
                        "truth": batch.truth,
                        "types": _question_types(batch.questions),
                        "rubric_ids": _rubric_ids(batch.questions),
                        "target_key": batch.target_key,
                        "target_truth": target_truth,
                        "position": 1,
                        "n_questions": count,
                        "target_index": i,
                        "difficulty": {"questions": count},
                        "baseline_keyword_prediction": inst.meta.get(
                            "baseline_keyword_prediction"
                        ),
                    },
                )
            )
    return calls


def _order_calls(
    targets: Sequence[Instance],
    blocks: Sequence[filler.FillerBlock],
    states: Sequence[Any],
) -> list[Call]:
    """Full 255-question batch with the whole question list shuffled."""
    seed = config.seed_for(EXPERIMENT, "order")
    calls: list[Call] = []
    for i, (inst, block, state) in enumerate(zip(targets, blocks, states)):
        rng = rng_for(EXPERIMENT, {"condition": "order"}, seed, i)
        key = _target_key(inst)
        items = [(inst.questions[key], inst.truth[key])] + [
            (block.questions[k], block.truth[k]) for k in block.questions
        ]
        order = list(range(len(items)))
        rng.shuffle(order)
        realized = order.index(0) + 1
        questions, truth, keys = _assemble([items[j] for j in order])
        calls.append(
            Call(
                experiment=EXPERIMENT,
                condition="order",
                instance_id=inst.instance_id,
                state=state,
                questions=questions,
                meta={
                    "truth": truth,
                    "types": _question_types(questions),
                    "rubric_ids": _rubric_ids(questions),
                    "target_key": keys[realized - 1],
                    "target_truth": inst.truth[key],
                    "position": realized,
                    "n_questions": len(questions),
                    "target_index": i,
                    "baseline_keyword_prediction": inst.meta.get("baseline_keyword_prediction"),
                },
            )
        )
    return calls


def _analyse_counts(results: Sequence[CallResult]) -> dict:
    per: dict[int, dict] = defaultdict(
        lambda: {
            "correct": 0,
            "n": 0,
            "truths": [],
            "heuristic_correct": 0,
            "filler_correct": 0,
            "filler_n": 0,
            "latency": [],
            "input_tokens": [],
        }
    )
    target_questions: dict[str, dict] = {}
    for r in results:
        if not r.ok:
            continue
        meta = r.call.meta
        count = int(meta["n_questions"])
        slot = per[count]
        slot["latency"].append(r.latency_s)
        slot["input_tokens"].append(r.input_tokens)
        target_key = meta["target_key"]
        answer = r.answers.get(target_key)
        if answer is not None:
            slot["correct"] += int(_predicted(answer) == meta["target_truth"])
            slot["n"] += 1
            slot["truths"].append(meta["target_truth"])
            slot["heuristic_correct"] += int(
                meta.get("baseline_keyword_prediction") == meta["target_truth"]
            )
            target_questions[target_key] = r.call.questions[target_key]
        for key, ans in r.answers.items():
            if key == target_key:
                continue
            slot["filler_correct"] += int(_predicted(ans) == meta["truth"].get(key))
            slot["filler_n"] += 1

    counts = [c for c in COUNTS if per[c]["latency"]]
    rows = []
    for c in counts:
        slot = per[c]
        lat = [x for x in slot["latency"] if math.isfinite(x) and x > 0]
        tokens = slot["input_tokens"]
        pct = metrics.percentiles(lat, (50, 95)) if lat else {}
        mean_tokens = _mean(tokens)
        rows.append(
            {
                "questions": c,
                "calls": len(slot["latency"]),
                "p50_latency_s": pct.get("p50"),
                "p95_latency_s": pct.get("p95"),
                "mean_input_tokens": mean_tokens,
                "cost_per_call_usd": (
                    mean_tokens * config.USD_PER_INPUT_TOKEN if mean_tokens is not None else None
                ),
                "cost_per_decision_usd": (
                    metrics.cost_per_decision(mean_tokens, c) if mean_tokens is not None else None
                ),
                "target_accuracy": _rate(slot["correct"], slot["n"]),
                "target_n": slot["n"],
                "target_wilson": list(metrics.wilson_interval(slot["correct"], slot["n"])),
                "filler_accuracy": _rate(slot["filler_correct"], slot["filler_n"]),
                "filler_n": slot["filler_n"],
            }
        )

    lat_rows = [r for r in rows if r["p50_latency_s"]]
    slope = _loglog_slope([r["questions"] for r in lat_rows], [r["p50_latency_s"] for r in lat_rows])
    # A question cannot be added without adding its own tokens, so question count
    # and input size move together by construction and this condition cannot
    # fully separate them. Fitting latency against input tokens as well at least
    # says which of the two the timing tracks more closely, and E4's dilution
    # sweep -- which grows the state at a fixed question count -- is the
    # complementary measurement.
    token_slope = _loglog_slope(
        [r["mean_input_tokens"] for r in lat_rows if r["mean_input_tokens"]],
        [r["p50_latency_s"] for r in lat_rows if r["mean_input_tokens"]],
    )
    tok_rows = [r for r in rows if r["mean_input_tokens"] is not None]
    token_fit = _ols(
        [float(r["questions"]) for r in tok_rows], [float(r["mean_input_tokens"]) for r in tok_rows]
    )

    all_truths = [t for c in counts for t in per[c]["truths"]]
    chance = (
        _chance_baseline(target_questions, all_truths) if target_questions and all_truths else None
    )
    first, last = (lat_rows[0], lat_rows[-1]) if lat_rows else (None, None)
    return {
        "rows": rows,
        "latency_slope_loglog": slope,
        "latency_slope_vs_input_tokens_loglog": token_slope,
        "input_token_growth_factor": (
            lat_rows[-1]["mean_input_tokens"] / lat_rows[0]["mean_input_tokens"]
            if lat_rows and lat_rows[0].get("mean_input_tokens")
            else None
        ),
        "latency_ratio_across_range": (
            last["p50_latency_s"] / first["p50_latency_s"]
            if first and last and first["p50_latency_s"]
            else None
        ),
        "latency_span": (
            {"from_questions": first["questions"], "to_questions": last["questions"]}
            if first and last
            else None
        ),
        "marginal_input_tokens_per_question": token_fit[0] if token_fit else None,
        "fixed_input_tokens": token_fit[1] if token_fit else None,
        "marginal_cost_per_question_usd": (
            token_fit[0] * config.USD_PER_INPUT_TOKEN if token_fit else None
        ),
        "chance_baseline": chance,
        "heuristic_baseline": _rate(
            sum(per[c]["heuristic_correct"] for c in counts), sum(per[c]["n"] for c in counts)
        ),
        "unbatched_accuracy": next(
            (r["target_accuracy"] for r in rows if r["questions"] == 1), None
        ),
    }


def _analyse_order(results: Sequence[CallResult], designated: dict) -> dict:
    hits: list[tuple[int, bool]] = []
    for r in results:
        if not r.ok:
            continue
        meta = r.call.meta
        answer = r.answers.get(meta["target_key"])
        if answer is None:
            continue
        hits.append((int(meta["position"]), _predicted(answer) == meta["target_truth"]))

    quartiles: dict[int, dict] = defaultdict(lambda: {"correct": 0, "n": 0, "positions": []})
    for position, hit in hits:
        q = min(3, (position - 1) * 4 // BATCH_QUESTIONS)
        quartiles[q]["correct"] += int(hit)
        quartiles[q]["n"] += 1
        quartiles[q]["positions"].append(position)

    by_quartile = []
    for q in sorted(quartiles):
        slot = quartiles[q]
        by_quartile.append(
            {
                "quartile": q + 1,
                "position_range": [min(slot["positions"]), max(slot["positions"])],
                "accuracy": slot["correct"] / slot["n"],
                "n": slot["n"],
                "wilson": list(metrics.wilson_interval(slot["correct"], slot["n"])),
            }
        )

    correct = sum(1 for _, hit in hits if hit)
    n = len(hits)
    # The designated sweep pooled over its positions. Same batch size, same
    # targets, same fillers -- but NOT the same distribution of positions: the
    # designated sweep spends five of its seven positions below 100 while a
    # shuffle lands uniformly over 1..255. Under pure positional decay and no
    # placement effect at all, the pooled difference is therefore large and
    # negative, and reading it as a placement effect is a mistake the data
    # cannot support. It is kept because it is the raw like-for-like accuracy,
    # and the position-matched expectation below is what any claim about
    # placement has to be made against.
    d_positions = designated.get("positions", [])
    d_correct = sum(designated["correct"][p] for p in d_positions)
    d_n = sum(designated["n"][p] for p in d_positions)
    expected = _expected_from_curve(designated, [p for p, _ in hits])
    return {
        "n": n,
        "accuracy": _rate(correct, n),
        "wilson": list(metrics.wilson_interval(correct, n)),
        "by_realized_position_quartile": by_quartile,
        "designated_pooled_accuracy": _rate(d_correct, d_n),
        "designated_pooled_n": d_n,
        "shuffled_minus_designated": (
            (correct / n) - (d_correct / d_n) if n and d_n else None
        ),
        "difference_ci": (list(c) if (c := _diff_ci(correct, n, d_correct, d_n)) else None),
        "position_matched_expected_accuracy": expected,
        "shuffled_minus_position_matched": (
            (correct / n) - expected if n and expected is not None else None
        ),
        "comparison_note": (
            "`shuffled_minus_designated` compares two different position distributions and "
            "so absorbs positional decay; `shuffled_minus_position_matched` is the residual "
            "after the designated curve is evaluated at each shuffled trial's realized "
            "position, which is the part attributable to placement itself."
        ),
    }


def _expected_from_curve(designated: dict, realized: Sequence[int]) -> float | None:
    """Mean accuracy the designated sweep predicts at these realized positions.

    The designated curve is measured at seven positions; a shuffled target can
    land at any of 255. Each realized position is read off the curve by linear
    interpolation in log position -- log because the sweep is geometric, so
    position 150 sits between 100 and 200 the way the sweep spaced them -- and
    clamped to the end values outside the swept range. Averaging those gives
    what the shuffled arm should have scored if ordinal position were the whole
    story, which is the only thing a placement claim can be measured against.
    """
    curve = sorted(
        (p, designated["accuracy"][p]) for p in designated.get("positions", [])
    )
    if not curve or not realized:
        return None
    if len(curve) == 1:
        return curve[0][1]
    xs = [math.log(p) for p, _ in curve]
    total = 0.0
    for r in realized:
        x = math.log(max(1, r))
        if x <= xs[0]:
            total += curve[0][1]
            continue
        if x >= xs[-1]:
            total += curve[-1][1]
            continue
        for i in range(len(curve) - 1):
            x0, x1 = xs[i], xs[i + 1]
            if x0 <= x <= x1:
                a0, a1 = curve[i][1], curve[i + 1][1]
                total += a0 if x1 <= x0 else a0 + (a1 - a0) * (x - x0) / (x1 - x0)
                break
    return total / len(realized)


# --------------------------------------------------------------------------
# Response scanning: the plan's "unexpected behaviors" watch list
# --------------------------------------------------------------------------

_KNOWN_TOP_LEVEL = {"model", "answers", "usage"}
_KNOWN_ANSWER_FIELDS = {
    "noul": {"type", "noul"},
    "choice": {"type", "choice", "confidence", "probabilities"},
    "score": {"type", "score", "confidence", "probabilities", "legend"},
}


class _Scan:
    """Facts about the responses themselves, independent of what was asked.

    Everything here is on the plan's watch list and none of it is what E3 was
    designed to measure, so it is collected over every call E3 makes rather than
    per condition. It ingests each condition's results as they arrive and keeps
    only running totals: a full position sweep is 2,100 calls of 255 answers
    each, and holding all of them so they can be scanned at the end would cost
    hundreds of megabytes to compute a dozen scalars.
    """

    def __init__(self) -> None:
        self.probabilities = 0
        self.quantized = 0
        # Only the decision-carrying probability goes into the histogram: the
        # noul's p, and the winning option's probability on a choice or score.
        # Counting all 8 options of every choice would make near-zero 74% of the
        # sample and report "clustering" on every run, which is a property of
        # asking 8-way questions rather than of the model.
        self.histogram: Counter = Counter()
        self.decisions = 0
        self.conf_comparisons = 0
        self.conf_abs_total = 0.0
        self.conf_signed_total = 0.0
        self.conf_diverged = 0
        self.score_answers = 0
        self.legend_seen = 0
        self.extra_top: Counter = Counter()
        self.extra_answer: Counter = Counter()
        self.attempted = 0
        self.failed = 0
        self.reasons: Counter = Counter()

    def ingest(self, results: Sequence[CallResult]) -> None:
        for r in results:
            self.attempted += 1
            if not r.ok:
                self.failed += 1
                if r.http_status:
                    self.reasons[f"HTTP {r.http_status}"] += 1
                else:
                    self.reasons[(r.error or "unknown").split(":")[0][:60]] += 1
                continue
            raw = r.raw_response or {}
            for key in raw:
                if key not in _KNOWN_TOP_LEVEL:
                    self.extra_top[key] += 1
            for wire_answer in (raw.get("answers") or {}).values():
                atype = wire_answer.get("type")
                known = _KNOWN_ANSWER_FIELDS.get(atype, set())
                for key in wire_answer:
                    if key not in known:
                        self.extra_answer[f"{atype}.{key}"] += 1
            for ans in r.answers.values():
                # A noul contributes only p: 1-p carries no independent
                # information and would double every count here.
                values = (
                    [ans["p"]] if ans["type"] == "noul" else list(_distribution(ans).values())
                )
                for v in values:
                    self.probabilities += 1
                    if abs(round(v, 2) - v) < 1e-9:
                        self.quantized += 1
                decision = ans["p"] if ans["type"] == "noul" else ans.get("max_probability")
                if decision is not None:
                    self.histogram[round(decision, 2)] += 1
                    self.decisions += 1
                if ans["type"] == "score":
                    self.score_answers += 1
                    if ans.get("legend"):
                        self.legend_seen += 1
                conf, top = ans.get("confidence"), ans.get("max_probability")
                if conf is not None and top is not None:
                    self.conf_comparisons += 1
                    self.conf_abs_total += abs(conf - top)
                    self.conf_signed_total += conf - top
                    self.conf_diverged += int(abs(conf - top) > 1e-9)

    def responses(self) -> dict:
        n = self.probabilities
        c = self.conf_comparisons
        return {
            "probabilities_seen": n,
            "two_decimal_fraction": _rate(self.quantized, n),
            "decisions_seen": self.decisions,
            "most_common_decision_probabilities": [
                {"value": v, "count": k} for v, k in self.histogram.most_common(8)
            ],
            "confidence_comparisons": c,
            "confidence_vs_max_probability_mean_abs": (self.conf_abs_total / c) if c else None,
            "confidence_vs_max_probability_mean_signed": (
                (self.conf_signed_total / c) if c else None
            ),
            "confidence_diverged_fraction": _rate(self.conf_diverged, c),
            "score_answers": self.score_answers,
            "score_answers_with_legend": self.legend_seen,
            "undocumented_top_level_fields": dict(self.extra_top),
            "undocumented_answer_fields": dict(self.extra_answer),
        }

    def failures(self) -> dict:
        return {
            "calls": self.failed,
            "excluded": self.failed,
            "reasons": dict(self.reasons),
            "attempted": self.attempted,
        }


# --------------------------------------------------------------------------
# Tiering
# --------------------------------------------------------------------------


def _by_difficulty(position: dict, drift: float | None, slope: float | None, counts: dict) -> list:
    """Grade every swept position.

    Each position is graded on the decay measured *at that position*, which is
    why the same number is supplied under `decay_50`, `decay_200` and
    `decay_255`. The rubric's rungs read "<5% decay to 50", "<1% decay to 200"
    and "flat to 255"; a sweep earns a rung only if every position within that
    rung's reach clears its bar, so grading each position against all three and
    taking the worst is the same statement made one position at a time. It is
    also what makes `boundary_crossings` able to name the position at which each
    boundary falls, which is the thing the plan asks for.

    `drift` and `latency_slope` are properties of the whole run rather than of a
    position, so the same value goes into every row. That is faithful to the
    rubric -- E3's rungs are a joint claim about decay and drift -- but it means
    a large drift pins every position to one tier and the position axis stops
    saying anything. Passing both as None grades the sweep on decay alone, which
    is how the decay-only view beside the main table is produced.
    """
    unbatched = counts.get("unbatched_accuracy")
    rows = []
    for p in position.get("positions", []):
        decay = position["decay"].get(p)
        m: dict[str, Any] = {
            "accuracy": position["accuracy"][p],
            "decay": decay,
            "drift": drift,
            "latency_slope": slope,
            "decay_50": decay,
            "decay_200": decay,
            "decay_255": decay,
            "unbatched_accuracy": unbatched,
            "chance_baseline": position.get("chance_baseline"),
            "heuristic_baseline": position.get("heuristic_baseline"),
            "n": position["n"][p],
        }
        collapse = _collapse_at(position, p)
        if collapse is not None:
            m["collapse_batch_size"] = collapse
        rows.append(tiers.assign(EXPERIMENT, m, difficulty={"position": p}))
    return rows


def _collapse_at(position: dict, p: int) -> int | None:
    """The position at which accuracy first collapsed to chance, if it did.

    Supplied only when a collapse really happened: the rubric's disqualifier
    fires on the key being present at all, so reporting "no collapse" as a value
    would grade every condition as Doesn't work.
    """
    chance = position.get("chance_baseline")
    if chance is None:
        return None
    for q in position.get("positions", []):
        if q > p:
            break
        hi = position["wilson"][q][1]
        if hi <= chance:
            return q
    return None


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


def _score_predictions(
    position: dict, contamination: dict, counts: dict, knee: dict
) -> list[dict]:
    claims = {p.id: p for p in predictions.for_experiment(EXPERIMENT)}
    out: list[dict] = []

    # P7 -- no positional decay, falsified by >5% decay by position 200.
    decay_200 = position.get("decay", {}).get(200)
    ci_200 = position.get("decay_ci", {}).get(200)
    if decay_200 is None:
        out.append(
            {
                "id": "P7",
                "claim": claims["P7"].claim,
                "outcome": "No usable answers came back for the target at position 200, "
                "so decay at 200 was not measured.",
                "verdict": "untestable",
                "evidence": {"decay_200": None},
            }
        )
    else:
        wrong = decay_200 > 0.05
        out.append(
            {
                "id": "P7",
                "claim": claims["P7"].claim,
                "outcome": (
                    f"Accuracy fell {decay_200:+.3f} from position 1 to position 200 "
                    f"(95% CI {ci_200[0]:+.3f} to {ci_200[1]:+.3f})"
                    if ci_200
                    else f"Accuracy fell {decay_200:+.3f} from position 1 to position 200"
                )
                + (
                    ". That is past the 5% falsification bar, so the parallelism is less "
                    "complete than the architecture claims -- the plan's own meta-prediction M1."
                    if wrong
                    else ". That is inside the 5% falsification bar."
                ),
                "verdict": "wrong" if wrong else "right",
                "evidence": {
                    "decay_200": decay_200,
                    "decay_200_ci": ci_200,
                    "accuracy_at_1": position["accuracy"].get(1),
                    "accuracy_at_200": position["accuracy"].get(200),
                    "n_per_position": position["n"].get(200),
                    "knee": knee.get("knee_position"),
                },
            }
        )

    # P8 -- mild contamination, 2-5% drift, better on related and worse on unrelated.
    drift = contamination.get("probability_drift")
    related = contamination.get("by_group", {}).get("related", {})
    unrelated = contamination.get("by_group", {}).get("unrelated", {})
    if drift is None:
        out.append(
            {
                "id": "P8",
                "claim": claims["P8"].claim,
                "outcome": "No question was answered both batched and individually, so drift "
                "was not measured.",
                "verdict": "untestable",
                "evidence": {},
            }
        )
    else:
        # A group with no paired questions has no measured direction. Reading a
        # missing `accuracy_delta` as 0.0 would make "not better" and "not
        # worse" both true and score the direction half as failed on data that
        # was never collected.
        direction_measured = bool(related.get("n")) and bool(unrelated.get("n"))
        related_better = (related.get("accuracy_delta") or 0.0) > 0
        unrelated_worse = (unrelated.get("accuracy_delta") or 0.0) < 0
        falsified = drift > 0.10 or drift == 0.0
        in_band = 0.02 <= drift <= 0.05
        drift_holds = in_band and not falsified
        right = drift_holds and direction_measured and related_better and unrelated_worse
        parts = [
            f"Mean probability drift (total variation between the batched and "
            f"individual answer distributions, which for a noul is the absolute "
            f"difference of the two probabilities) was "
            f"{drift:.4f}; the discrete answer changed on "
            f"{contamination.get('answer_drift', float('nan')):.3f} of questions."
        ]
        if related:
            parts.append(
                f"Related questions: accuracy {related['accuracy_delta']:+.3f} batched "
                f"(drift {related['probability_drift']:.4f}, McNemar p="
                f"{related['mcnemar_p']:.3g})."
            )
        if unrelated:
            parts.append(
                f"Unrelated questions: accuracy {unrelated['accuracy_delta']:+.3f} batched "
                f"(drift {unrelated['probability_drift']:.4f}, McNemar p="
                f"{unrelated['mcnemar_p']:.3g})."
            )
        if falsified:
            parts.append(
                ("Drift above 0.10" if drift > 0.10 else "Drift exactly zero")
                + " -- the plan's falsification condition."
            )
        elif not in_band:
            parts.append(
                "Drift is outside the predicted 2-5% band but inside the 10% "
                "falsification bar."
            )
        if not direction_measured:
            parts.append(
                "One of the two question groups produced no paired answers, so the predicted "
                "direction split -- better on related, worse on unrelated -- was not measured."
            )
        elif not (related_better and unrelated_worse):
            parts.append("The predicted direction split did not hold.")
        out.append(
            {
                "id": "P8",
                "claim": claims["P8"].claim,
                "outcome": " ".join(parts),
                # The drift half alone can falsify the claim, so a missing
                # direction leaves the prediction open rather than wrong.
                "verdict": (
                    "right"
                    if right
                    else "untestable"
                    if drift_holds and not direction_measured
                    else "wrong"
                ),
                "evidence": {
                    "probability_drift": drift,
                    "answer_drift": contamination.get("answer_drift"),
                    "explicit_falsifier_fired": falsified,
                    "in_predicted_band": in_band,
                    "direction_measured": direction_measured,
                    "related": related,
                    "unrelated": unrelated,
                },
            }
        )

    # P9 -- latency roughly flat in question count.
    slope = counts.get("latency_slope_loglog")
    if slope is None:
        out.append(
            {
                "id": "P9",
                "claim": claims["P9"].claim,
                "outcome": "Fewer than two question counts produced a usable p50, so no slope "
                "could be fitted.",
                "verdict": "untestable",
                "evidence": {},
            }
        )
    else:
        ratio = counts.get("latency_ratio_across_range")
        span = counts.get("latency_span") or {}
        linear = slope >= 0.9
        flat = slope <= 0.2
        out.append(
            {
                "id": "P9",
                "claim": claims["P9"].claim,
                "outcome": (
                    f"Fitted log-log slope of p50 latency against question count is "
                    f"{slope:.3f} (0 is flat, 1 is linear). p50 moved by a factor of "
                    f"{ratio:.2f} from {span.get('from_questions')} to "
                    f"{span.get('to_questions')} questions. "
                    if ratio
                    else f"Fitted log-log slope of p50 latency against question count is {slope:.3f}. "
                )
                + (
                    "That scales linearly, which is the plan's falsification condition and "
                    "would destroy the economics."
                    if linear
                    else "Roughly flat, as predicted."
                    if flat
                    else "Sub-linear but not flat: the claim as stated does not hold, though "
                    "the plan's narrower falsifier (scales linearly) was not met."
                )
                + (
                    f" Input tokens grew {counts['input_token_growth_factor']:.1f}x over the "
                    f"same range and latency fits them with a log-log slope of "
                    f"{counts['latency_slope_vs_input_tokens_loglog']:.3f}; a question cannot be "
                    "added without its tokens, so this condition cannot fully separate "
                    "'flat in question count' from 'rising with input size'."
                    if counts.get("latency_slope_vs_input_tokens_loglog") is not None
                    and counts.get("input_token_growth_factor")
                    else ""
                ),
                "verdict": "right" if flat else "wrong",
                "evidence": {
                    "latency_slope_loglog": slope,
                    "latency_slope_vs_input_tokens_loglog": counts.get(
                        "latency_slope_vs_input_tokens_loglog"
                    ),
                    "input_token_growth_factor": counts.get("input_token_growth_factor"),
                    "latency_ratio": ratio,
                    "marginal_input_tokens_per_question": counts.get(
                        "marginal_input_tokens_per_question"
                    ),
                    "marginal_cost_per_question_usd": counts.get(
                        "marginal_cost_per_question_usd"
                    ),
                    "scales_linearly": linear,
                },
            }
        )
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _make_plots(run_dir: Path, position: dict, counts: dict) -> list[str]:
    out: list[str] = []
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    positions = position.get("positions") or []
    if positions:
        acc = [position["accuracy"][p] for p in positions]
        lo = [position["wilson"][p][0] for p in positions]
        hi = [position["wilson"][p][1] for p in positions]
        path = plots.curve(
            positions,
            acc,
            plot_dir / "e3_position.png",
            "E3 -- accuracy against question position",
            "position within the batch",
            "accuracy on the target question",
            n=[position["n"][p] for p in positions],
            band=(lo, hi),
            baseline=position.get("chance_baseline"),
            baseline_label="chance",
            logx=True,
            xticks=positions,
            ylim=(0.0, 1.0),
            note=f"Batch held at {BATCH_QUESTIONS} questions about one state. Bands are Wilson 95%.",
        )
        out.append(str(path.relative_to(run_dir)))

    fillers = position.get("filler_accuracy_by_own_position") or []
    if len(fillers) >= 2:
        path = plots.curve(
            [round(f["mean_position"]) for f in fillers],
            [f["accuracy"] for f in fillers],
            plot_dir / "e3_filler_position.png",
            "E3 -- filler accuracy against its own position",
            "position within the batch (binned)",
            "accuracy on filler questions",
            n=[f["n"] for f in fillers],
            band=([f["wilson"][0] for f in fillers], [f["wilson"][1] for f in fillers]),
            baseline=position.get("filler_chance_baseline"),
            baseline_label="chance",
            ylim=(0.0, 1.0),
            note="Every filler question in the position sweep, binned by where it sat.",
        )
        out.append(str(path.relative_to(run_dir)))

    rows = [r for r in (counts.get("rows") or []) if r["p50_latency_s"]]
    if len(rows) >= 2:
        path = plots.curve(
            [r["questions"] for r in rows],
            {
                "p50": [r["p50_latency_s"] for r in rows],
                "p95": [r["p95_latency_s"] for r in rows],
            },
            plot_dir / "e3_latency.png",
            "E3 -- latency against question count",
            "questions per call",
            "wall clock (s)",
            n=[r["calls"] for r in rows],
            logx=True,
            xticks=[r["questions"] for r in rows],
            note="State held identical across counts. Measured under this harness's own "
            "concurrent load, so read the slope rather than the level.",
        )
        out.append(str(path.relative_to(run_dir)))

    cost_rows = [r for r in (counts.get("rows") or []) if r["cost_per_decision_usd"]]
    if len(cost_rows) >= 2:
        path = plots.curve(
            [r["questions"] for r in cost_rows],
            [r["cost_per_decision_usd"] for r in cost_rows],
            plot_dir / "e3_cost.png",
            "E3 -- cost per decision against question count",
            "questions per call",
            "USD per question",
            n=[r["calls"] for r in cost_rows],
            logx=True,
            logy=True,
            xticks=[r["questions"] for r in cost_rows],
            note=f"Input tokens x ${config.USD_PER_INPUT_TOKEN * 1e9:.0f}/10^9, divided by "
            "the number of questions in the call.",
        )
        out.append(str(path.relative_to(run_dir)))

    acc_rows = [r for r in (counts.get("rows") or []) if r["target_accuracy"] is not None]
    if len(acc_rows) >= 2:
        path = plots.curve(
            [r["questions"] for r in acc_rows],
            [r["target_accuracy"] for r in acc_rows],
            plot_dir / "e3_count_accuracy.png",
            "E3 -- accuracy against question count, position held at 1",
            "questions per call",
            "accuracy on the target question",
            n=[r["target_n"] for r in acc_rows],
            band=(
                [r["target_wilson"][0] for r in acc_rows],
                [r["target_wilson"][1] for r in acc_rows],
            ),
            baseline=counts.get("chance_baseline"),
            baseline_label="chance",
            logx=True,
            xticks=[r["questions"] for r in acc_rows],
            ylim=(0.0, 1.0),
            note="The count half of order-versus-count: the target never moves, only the "
            "number of questions around it.",
        )
        out.append(str(path.relative_to(run_dir)))
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    n = config.n(config.SIZES.batching)

    # The position, count and order conditions share one set of targets, one set
    # of filler blocks and one set of states, so every comparison between them is
    # paired on the instance rather than on two independent samples. One
    # consequence is worth using: position 1 of the position sweep and count 255
    # of the count sweep assemble the same 255 questions in the same order
    # against the same state, so they are byte-identical requests sent twice, and
    # the difference between their accuracies is a free repeatability check
    # against E1's noise floor.
    targets = _target_instances(n)
    blocks = [_block(i, BATCH_QUESTIONS - 1, "batch") for i in range(n)]
    # `attach` raises if the ticket already contains filler vocabulary, which is
    # the one way a filler question could be ambiguous. Let it raise: a shared
    # name would otherwise show up in the results as positional decay that is
    # not there.
    states = [filler.attach(t.state, b) for t, b in zip(targets, blocks)]
    contamination_material = _contamination_material(n)
    interference_material = _interference_material(n)

    # Each condition is scanned and reduced to its metrics as soon as it comes
    # back, and its responses are then dropped. Holding all five conditions'
    # responses at once costs a few hundred megabytes at full scale and buys
    # nothing: the raw record of every call is already on disk in e3.jsonl.
    scan = _Scan()

    def _do(calls: list[Call], label: str, client: JevClient, every: int = 200):
        results = client.run(calls, label=label, progress_every=every)
        scan.ingest(results)
        return results

    with JevClient(run_dir=run_dir, log_name="e3.jsonl") as client:
        position = _analyse_positions(
            _do(_position_calls(targets, blocks, states), "E3/position", client)
        )
        counts = _analyse_counts(
            _do(_count_calls(targets, blocks, states), "E3/count", client)
        )
        order = _analyse_order(
            _do(_order_calls(targets, blocks, states), "E3/order", client), position
        )
        # Batched and individual go out in one list so they interleave in time.
        # Run back to back instead and any drift in the service over the run
        # would be indistinguishable from contamination.
        contamination = _analyse_contamination(
            _do(_contamination_calls(contamination_material), "E3/contamination", client, 500)
        )
        interference = _analyse_interference(
            _do(_interference_calls(interference_material), "E3/interference", client)
        )
        summary = client.summary()

    knee = _knee(position)
    repeatability = _repeatability(position, counts)

    drift = contamination.get("probability_drift")
    slope = counts.get("latency_slope_loglog")
    tier_results = _by_difficulty(position, drift, slope, counts)
    crossings = tiers.boundary_crossings(EXPERIMENT, tier_results)
    decay_only = _by_difficulty(position, None, None, counts)
    decay_crossings = tiers.boundary_crossings(EXPERIMENT, decay_only)

    decay_50 = position.get("decay", {}).get(50)
    decay_50_ci = position.get("decay_ci", {}).get(50)
    n_at_50 = position.get("n", {}).get(50, 0)

    notes = [TARGET_HEADROOM_NOTE]
    notes += _sample_size_notes(n, position, counts, contamination, interference)
    joint = {t.tier for t in tier_results}
    if len(joint) == 1 and len({t.tier for t in decay_only}) > 1:
        notes.append(
            f"Every position grades as {next(iter(joint))} because the rubric's rungs are a "
            "joint claim about decay and drift and the sweep-level drift misses its bar at "
            "every position. The positional finding is in `by_position_decay_only`, which "
            "grades the same sweep on decay alone."
        )
    responses = scan.responses()
    anomalies = _anomalies(position, counts, contamination, interference, order, responses, knee)
    repeat_note = _repeatability_anomaly(repeatability, position)
    if repeat_note:
        anomalies.insert(0, repeat_note)
    failures = scan.failures()

    if decay_50 is None:
        gate = {
            "passed": False,
            "reason": (
                "Decay at position 50 could not be measured -- no usable answers came back "
                "for the target at position 1, at position 50, or both. The plan's Phase 2 "
                "gate is about that number, so it cannot be evaluated and the phase should "
                "not continue on the assumption that it passed."
            ),
        }
    else:
        tripped = decay_50 > GATE_DECAY_50
        gate = {
            "passed": not tripped,
            "reason": (
                f"Positional decay at question 50 is {decay_50:+.3f}"
                + (
                    f" (95% CI {decay_50_ci[0]:+.3f} to {decay_50_ci[1]:+.3f}, n={n_at_50} "
                    "targets per position)"
                    if decay_50_ci
                    else f" (n={n_at_50} targets per position)"
                )
                + (
                    f", above the {GATE_DECAY_50:.0%} gate. Batch size must be capped. The "
                    f"measured knee lies between position {knee['safe_batch_size']}, the "
                    f"largest swept position still within {KNEE_DECAY:.0%} of position 1, and "
                    f"position {knee['knee_position']}, where decay first exceeds it; the cap "
                    f"is therefore {knee['safe_batch_size']} questions until a finer sweep "
                    "locates the knee between those two. E5 and E6 must be re-run within it."
                    + (
                        ""
                        if knee.get("nominal_knee_resolved")
                        else (
                            f" That knee is not resolved by this sample -- the 95% interval on "
                            f"the decay at position {knee['knee_position']} includes zero -- so "
                            "the cap should be confirmed before it is imposed"
                            + (
                                f"; the first decay whose interval excludes zero is at position "
                                f"{knee['knee_position_resolved']}."
                                if knee.get("knee_position_resolved") is not None
                                else ", and no swept position produced a decay whose interval "
                                "excludes zero."
                            )
                        )
                    )
                    if tripped
                    else f", at or below the {GATE_DECAY_50:.0%} gate. No batch-size cap is "
                    "required and E5 and E6 can run at full batch size. Measured knee: "
                    + (
                        f"decay first exceeds {KNEE_DECAY:.0%} at position "
                        f"{knee['knee_position']}"
                        if knee["knee_position"] is not None
                        else f"none within the swept range, so no knee up to position "
                        f"{knee['max_position_swept']}"
                    )
                    + "."
                )
            ),
        }

    result = {
        "experiment": EXPERIMENT,
        # A short name for the report's section heading, which truncates the
        # question at 120 characters and would otherwise cut it mid-word.
        "title": "Batching and parallelism",
        "question": (
            "Does question 200 of a batch get the same quality as question 3, and do "
            "questions batched together contaminate each other?"
        ),
        "headline": {
            "metric": "positional decay at question 50 (accuracy at position 1 minus "
            "accuracy at position 50)",
            "value": decay_50,
            "baseline": 0.0,
            "baseline_name": "zero decay, which is what a genuinely parallel architecture predicts",
            "n": n_at_50,
        },
        "by_difficulty": [t.to_json() for t in tier_results],
        "boundary_crossings": tiers.crossings_json(crossings),
        "plots": _make_plots(run_dir, position, counts),
        "predictions": _score_predictions(position, contamination, counts, knee),
        "anomalies": anomalies,
        "failures": failures,
        "gate": gate,
        "what_this_changes": _what_this_changes(position, counts, knee, gate, contamination),
        # Everything below is detail the report does not print but the analysis
        # rests on, kept in the result file so a reader can check a number
        # without reparsing the log.
        "by_position_decay_only": [t.to_json() for t in decay_only],
        "boundary_crossings_decay_only": tiers.crossings_json(decay_crossings),
        "knee": knee,
        "repeatability": repeatability,
        "baselines": {
            "target_chance": position.get("chance_baseline"),
            "target_keyword_heuristic": position.get("heuristic_baseline"),
            "target_unbatched": counts.get("unbatched_accuracy"),
            "filler_chance": position.get("filler_chance_baseline"),
            "contamination_unbatched": contamination.get("unbatched_accuracy"),
            "interference_hard_majority": interference.get("hard_majority_baseline"),
            "interference_hard_density_heuristic": interference.get("hard_density_baseline"),
        },
        "conditions": {
            "position": position,
            "count": counts,
            "order": order,
            "contamination": contamination,
            "interference": interference,
        },
        "response_scan": responses,
        "notes": notes,
        "sample_sizes": {
            "scale": config.SCALE,
            "instances_per_condition": n,
            "positions": list(POSITIONS),
            "counts": list(COUNTS),
            "batch_questions": BATCH_QUESTIONS,
            "contamination_questions_per_state": CONTAMINATION_QUESTIONS,
            "easy_questions_per_interference_batch": EASY_QUESTIONS,
        },
        "run": summary,
    }
    return result


def _sample_size_notes(
    n: int, position: dict, counts: dict, contamination: dict, interference: dict
) -> list[str]:
    """Where the sample is too small for a number to carry its own weight.

    Stated as a property of the interval rather than as an arbitrary n cutoff:
    what matters is whether the sweep can resolve the boundary it is being
    graded against, and at n=300 per position the 95% interval on a decay is
    roughly +/-0.05, which is exactly the Human bar.
    """
    notes: list[str] = []
    if config.SCALE != 1.0:
        notes.append(
            f"Sample sizes were scaled by {config.SCALE}: {n} instances per condition "
            f"instead of {config.SIZES.batching}. Every number below is a dry run."
        )
    ci = position.get("decay_ci", {}).get(50)
    if ci and (ci[1] - ci[0]) > 2 * KNEE_DECAY:
        notes.append(
            f"The 95% interval on decay at position 50 spans {ci[1] - ci[0]:.3f}, wider than "
            f"the {KNEE_DECAY:.0%} bar it is graded against. The tier at that position is not "
            "resolved by this sample."
        )
    for row in counts.get("rows") or []:
        if row["calls"] and row["calls"] < 20 and row["p95_latency_s"] is not None:
            notes.append(
                f"p95 latency at {row['questions']} questions comes from {row['calls']} calls; "
                "a 95th percentile needs more than 20 samples to mean anything."
            )
            break
    if (contamination.get("paired_questions") or 0) < 30:
        notes.append(
            f"Drift is averaged over {contamination.get('paired_questions')} paired questions, "
            "which is too few to split by group with any confidence."
        )
    if (interference.get("paired_easy_questions") or 0) < 30:
        notes.append(
            f"The interference comparison rests on "
            f"{interference.get('paired_easy_questions')} paired easy questions."
        )
    return notes


def _repeatability_anomaly(repeatability: dict, position: dict) -> str | None:
    """A difference between two identical requests is noise, and noise this big
    would swallow the positional result it sits next to."""
    if not repeatability.get("comparable"):
        return None
    diff = abs(repeatability["difference"])
    if diff <= KNEE_DECAY:
        return None
    largest = max((abs(d) for d in (position.get("decay") or {}).values()), default=0.0)
    head = (
        f"The two byte-identical conditions -- position 1 of the position sweep and count 255 "
        f"of the count sweep -- disagree by {diff:.3f} on the same targets. The largest decay "
        f"measured anywhere in the position sweep is {largest:.3f}, so "
    )
    # Which reading follows depends on how the two numbers compare, and asserting
    # "the same order as the effect" when the effect is eight times the noise
    # would be a confident claim the numbers beside it contradict.
    if largest <= 2 * diff:
        return head + (
            "run-to-run variance is of the same order as the effect and no positional finding "
            "here is safe without E1's noise floor beside it."
        )
    return head + (
        "the largest positional effect does stand clear of this noise, but anything smaller "
        f"than {diff:.3f} in the sweep -- including the decay at any position that does not "
        "exceed it -- is inside run-to-run variance and should be read against E1's noise "
        "floor rather than as a position effect."
    )


def _anomalies(
    position: dict,
    counts: dict,
    contamination: dict,
    interference: dict,
    order: dict,
    scan: dict,
    knee: dict,
) -> list[str]:
    out: list[str] = []

    quantized = scan.get("two_decimal_fraction")
    if quantized is not None and scan.get("probabilities_seen"):
        if quantized > 0.999:
            out.append(
                f"Every one of the {scan['probabilities_seen']} probabilities E3 saw is exactly "
                "representable to two decimal places. The returned probabilities are quantized "
                "at 0.01, which sets a lower bound on any calibration measurement and means a "
                "drift below 0.01 cannot be observed at all."
            )
        elif quantized < 0.99:
            out.append(
                f"{quantized:.3f} of probabilities were two-decimal values, so the quantization "
                "seen elsewhere in this harness is not universal."
            )
    common = scan.get("most_common_decision_probabilities") or []
    decisions = scan.get("decisions_seen") or 0
    if common and decisions and common[0]["count"] / decisions > 0.25:
        out.append(
            f"The decision-carrying probability was {common[0]['value']} on "
            f"{common[0]['count'] / decisions:.2f} of {decisions} answers -- clustering at one "
            "value rather than spreading smoothly."
        )

    diverged = scan.get("confidence_diverged_fraction")
    if diverged:
        out.append(
            f"`confidence` differed from the maximum returned probability on {diverged:.3f} of "
            f"the {scan['confidence_comparisons']} choice and score answers, by "
            f"{scan['confidence_vs_max_probability_mean_abs']:.4f} on average "
            f"(signed mean {scan['confidence_vs_max_probability_mean_signed']:+.4f}). They are "
            "not the same quantity."
        )
    if scan.get("score_answers") and scan.get("score_answers_with_legend"):
        out.append(
            f"{scan['score_answers_with_legend']} of {scan['score_answers']} score answers "
            "carried the undocumented `legend` field mapping rubric indices to labels."
        )
    if scan.get("undocumented_top_level_fields"):
        out.append(
            f"Undocumented top-level response fields: {scan['undocumented_top_level_fields']}."
        )
    if scan.get("undocumented_answer_fields"):
        out.append(f"Undocumented answer fields: {scan['undocumented_answer_fields']}.")

    for name in ("related", "unrelated"):
        group = contamination.get("by_group", {}).get(name)
        if group and group["accuracy_delta"] > 0 and group["mcnemar_p"] < 0.05:
            out.append(
                f"Batched answers were better than unbatched on {name} questions "
                f"({group['accuracy_delta']:+.3f}, McNemar p={group['mcnemar_p']:.3g}). That is "
                "coherence to exploit rather than contamination to avoid."
            )

    filler_acc = position.get("filler_accuracy")
    filler_chance = position.get("filler_chance_baseline")
    if filler_acc is not None and filler_chance is not None:
        if filler_acc <= filler_chance + 0.05:
            out.append(
                f"Filler questions were answered at {filler_acc:.3f} against a "
                f"{filler_chance:.3f} chance baseline, so the batch surrounding the target was "
                "not being answered meaningfully. Any positional result from this condition is "
                "uninterpretable until that is explained."
            )

    curve = position.get("filler_accuracy_by_own_position") or []
    if len(curve) >= 2:
        span = max(c["accuracy"] for c in curve) - min(c["accuracy"] for c in curve)
        if span > 0.05:
            out.append(
                f"Filler accuracy varies by {span:.3f} across position bins, measured on "
                f"{sum(c['n'] for c in curve)} filler answers. That is a second, "
                "higher-resolution reading of positional decay from the same calls."
            )

    if interference.get("easy_accuracy_delta") is not None:
        delta = interference["easy_accuracy_delta"]
        if abs(delta) > 0.02 and (interference.get("mcnemar_p") or 1.0) < 0.05:
            out.append(
                f"Adding one hard 3SAT question moved easy-question accuracy by {delta:+.3f} "
                f"(McNemar p={interference['mcnemar_p']:.3g}). The plan predicted no effect."
            )

    residual = order.get("shuffled_minus_position_matched")
    if residual is not None and abs(residual) > 0.05:
        out.append(
            f"A fully shuffled batch scored {residual:+.3f} against what the designated-position "
            f"curve predicts at the shuffled targets' own realized positions "
            f"({order['accuracy']:.3f} measured against {order['position_matched_expected_accuracy']:.3f} "
            f"expected, n={order['n']}). Placement itself, not just ordinal position, is doing "
            "something. The raw pooled difference is "
            f"{order['shuffled_minus_designated']:+.3f}, most of which is the two arms sampling "
            "different positions rather than an effect."
        )

    if knee.get("knee_position") is not None and not knee.get("nominal_knee_resolved"):
        resolved = knee.get("knee_position_resolved")
        out.append(
            f"A knee appears at position {knee['knee_position']} but its confidence interval "
            "includes zero, so it is not resolved by this sample and should not be treated as "
            "a batch-size cap without more data."
            + (
                f" The first decay whose interval does exclude zero is at position {resolved}, "
                f"so {resolved} is the cap this sample actually supports and "
                f"{knee['safe_batch_size']} is the cap the nominal knee implies."
                if resolved is not None
                else " No swept position produced a decay whose interval excludes zero."
            )
        )
    return out


def _what_this_changes(
    position: dict, counts: dict, knee: dict, gate: dict, contamination: dict
) -> str:
    decay_200 = position.get("decay", {}).get(200)
    slope = counts.get("latency_slope_loglog")
    cap = knee.get("safe_batch_size")
    if not gate["passed"] and knee.get("knee_position") is not None:
        head = (
            f"Batch size has to be capped at about {cap} questions, so the parallel-question "
            "designs are constrained: anything that assumed one call could carry hundreds of "
            f"independent decisions has to shard into batches of {cap} and pay the state's "
            "input tokens once per shard. E5 and E6 must be re-run inside that cap before "
            "their results mean anything."
        )
        if knee.get("nominal_knee_resolved"):
            return head
        resolved = knee.get("knee_position_resolved")
        return head + (
            f" The knee that sets this cap is not resolved by this sample, though: the "
            f"interval on the decay at position {knee['knee_position']} includes zero"
            + (
                f", and the first position whose decay excludes zero is {resolved}. The cap "
                f"worth acting on is somewhere between {cap} and {resolved}, and a finer sweep "
                "would decide it before E5 and E6 are re-sized."
                if resolved is not None
                else ", and no swept position produced a decay whose interval excludes zero. "
                "A finer sweep should locate the knee before E5 and E6 are re-sized."
            )
        )
    if decay_200 is not None and decay_200 <= 0.05:
        flat = slope is not None and slope <= 0.2
        resolved = knee.get("knee_position_resolved")
        head = (
            f"The Phase 2 gate does not bind: accuracy at question 200 is within "
            f"{abs(decay_200):.3f} of accuracy at question 1, so the parallel-question designs "
            "are not constrained by quality and can ask everything they want in one call."
        )
        if resolved is not None:
            head += (
                f" A resolved knee does appear further out, at position {resolved}, so a design "
                f"that wants every question answered equally well should still stay under it."
            )
        return head + " " + (
            "Latency is roughly flat in question count too, so what limits batch size is the "
            "255-option choice cap and the state's token cost, not this model."
            if flat
            else f"Latency does scale with question count (log-log slope {slope:.2f}), so the "
            "limit that binds is throughput rather than quality: batch for cost, not for "
            "accuracy."
            if slope is not None
            else "Latency against question count was not measured, so the throughput limit is "
            "still open."
        )
    return (
        "Positional decay was not measured cleanly enough to say whether the batch-size cap "
        "constrains the parallel-question designs; the position sweep needs to be re-run "
        "before E5 and E6 can be sized."
    )


# --------------------------------------------------------------------------
# Terminal summary
# --------------------------------------------------------------------------


def _f(v: Any, spec: str = ".3f") -> str:
    return "n/a" if v is None else format(v, spec)


def format_report(result: dict) -> str:
    lines: list[str] = []
    h = result["headline"]
    val = "not measured" if h["value"] is None else f"{h['value']:+.4f}"
    lines.append(f"E3 batching -- {h['metric']}: {val} against {h['baseline']:.4g} "
                 f"({h['baseline_name']}), n={h['n']}")

    cond = result["conditions"]
    pos = cond["position"]
    if pos.get("positions"):
        lines.append("  accuracy by position:")
        for p in pos["positions"]:
            lo, hi = pos["wilson"][p]
            lines.append(
                f"    {p:>4}: {pos['accuracy'][p]:.3f} (95% CI {lo:.3f}-{hi:.3f}, "
                f"n={pos['n'][p]})  decay {pos['decay'].get(p, 0.0):+.3f}"
            )
        lines.append(
            f"    chance {pos.get('chance_baseline')}, keyword heuristic "
            f"{pos.get('heuristic_baseline')}, fillers {pos.get('filler_accuracy')} "
            f"on n={pos.get('filler_n')}"
        )

    c = cond["contamination"]
    if c.get("probability_drift") is not None:
        lines.append(
            f"  contamination: probability drift {c['probability_drift']:.4f}, answer drift "
            f"{c['answer_drift']:.3f} over {c['paired_questions']} paired questions"
        )
        for name in ("related", "unrelated"):
            g = c.get("by_group", {}).get(name)
            if g:
                lines.append(
                    f"    {name:<10} drift {g['probability_drift']:.4f}  accuracy "
                    f"{g['accuracy_batched']:.3f} batched vs {g['accuracy_individual']:.3f} "
                    f"individual ({g['direction']}, p={g['mcnemar_p']:.3g})"
                )

    i = cond["interference"]
    if i.get("easy_accuracy_with_hard") is not None:
        lines.append(
            f"  interference: easy accuracy {i['easy_accuracy_with_hard']:.3f} with the hard "
            f"question vs {i['easy_accuracy_without_hard']:.3f} without "
            f"(p={i['mcnemar_p']:.3g}); the hard question itself "
            f"{_f(i['hard_accuracy'])} against majority {_f(i['hard_majority_baseline'])}, "
            f"density heuristic {_f(i['hard_density_baseline'])}"
        )

    o = cond["order"]
    if o.get("accuracy") is not None:
        # The pooled difference is printed because it is the raw comparison, and
        # the residual beside it because the pooled one absorbs the two arms'
        # different position distributions and is the number a reader would
        # otherwise mistake for a placement effect.
        lines.append(
            f"  order: shuffled placement {o['accuracy']:.3f} vs designated pooled "
            f"{o['designated_pooled_accuracy']:.3f} (difference "
            f"{o['shuffled_minus_designated']:+.3f}, of which "
            f"{_f(o.get('shuffled_minus_position_matched'), '+.3f')} survives matching each "
            "shuffled target to the designated curve at its own realized position)"
        )

    k = cond["count"]
    if k.get("latency_slope_loglog") is not None:
        lines.append(
            f"  latency: log-log slope {k['latency_slope_loglog']:.3f} vs question count, "
            f"{_f(k.get('latency_slope_vs_input_tokens_loglog'))} vs input tokens; p50 ratio "
            f"{_f(k.get('latency_ratio_across_range'), '.2f')} across the swept range; marginal "
            f"{_f(k.get('marginal_input_tokens_per_question'), '.1f')} input tokens per question "
            f"(${_f(k.get('marginal_cost_per_question_usd'), '.3g')} each)"
        )

    rp = result["repeatability"]
    if rp.get("comparable"):
        lines.append(
            f"  repeatability: identical request answered {rp['position_sweep_at_1']:.3f} in the "
            f"position sweep and {rp['count_sweep_at_255']:.3f} in the count sweep "
            f"(difference {rp['difference']:+.3f})"
        )

    kn = result["knee"]
    lines.append(
        f"  knee: decay first exceeds {KNEE_DECAY:.0%} at position {kn['knee_position']} "
        f"(resolved: {kn['knee_position_resolved']}); largest clean position "
        f"{kn['safe_batch_size']} of {kn['max_position_swept']} swept"
    )
    for p in result["predictions"]:
        lines.append(f"  {p['id']}: {p['verdict'].upper()} -- {p['outcome']}")
    for note in result.get("notes", []):
        lines.append(f"  note: {note}")
    for a in result["anomalies"]:
        lines.append(f"  ! {a}")
    f = result["failures"]
    lines.append(
        f"  calls: {f['attempted']} attempted, {f['calls']} failed and excluded {f['reasons']}"
    )
    g = result["gate"]
    lines.append(f"  GATE {'passed' if g['passed'] else 'TRIPPED'}: {g['reason']}")
    return "\n".join(lines)
