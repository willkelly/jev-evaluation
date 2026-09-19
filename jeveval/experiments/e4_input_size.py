"""E4 -- input size and encoding.

The plan asks three questions that are usually run as three experiments: how
much state can you send before answers degrade, does position within the state
matter, and does the shape of the state matter. They are one experiment here
because all three are the same confound seen from different sides -- a result
that varies with state size is uninterpretable unless position and encoding are
held fixed, and vice versa. So every arm below holds the other two axes constant
and says so.

Five arms, in the plan's order:

1. DILUTION. One fixed retrieval question, state grown from 200 to 50,000
   estimated tokens with `generators.filler`. Accuracy and ECE against size.
   The needle sits at position 0.0 throughout, the favourable end, so the size
   curve does not have a middle-of-state penalty folded into it.

2. NEEDLE POSITION. The same needle at 0/25/50/75/100% of the way through
   10,000 tokens. `filler.needle_series` and `place_needle` hold total length
   identical across positions -- the same paragraphs and the same separator
   count whatever the insertion index -- because a position effect and a size
   effect are otherwise indistinguishable. `filler.make_needle` is keyed on
   (tokens, seed, index) and not on position, so the five positions of instance
   i carry byte-identical questions and the comparison is paired.

3. ENCODING. `generators.progreach` emits the same program as numbered source,
   AST-as-nested-JSON and a flat CFG edge list. The programs are identical
   across the three encodings, index for index, joined by `meta["program_id"]`,
   so the comparison is paired and uses `metrics.mcnemar` rather than an
   unpaired test.

4. STRUCTURED VERSUS STRINGIFIED. The semantic control set sent as an object and
   then as `json.dumps` of that object with JSON.stringify's separators. Same
   instances, same question order, same option order; only `state` differs.
   Paired again.

5. NOISE FLOOR IN STATE. The same three facts written as a terse record line and
   as a paragraph of verbose prose, inside filler sized so that both arms reach
   the same total state size. Holding the total fixed is what separates this
   from arm 1: arm 1 measures how much irrelevant text the fact survives, arm 5
   measures how the fact's own wording changes retrieval at a fixed amount of
   it.

The difficulty probe the plan calls for
---------------------------------------
`progreach` varies nesting depth, live condition count, and
`constraint_required`. With `constraint_required=False` a line is dead because
control flow cannot reach it, and walking the graph settles the question. With
`constraint_required=True` control flow always reaches the line and reachability
hinges on whether the enclosing integer conditions are jointly satisfiable,
which is a finite-domain CSP -- the same kind of question E2 asks with 3SAT, in a
syntax the model has far more likely seen. Those two are reported as separate
conditions at every depth. A model that handles the first and not the second has
a constraint-solving weakness rather than a graph-traversal one, and that is a
sharper statement than any single reachability number.

What is reported and against what
----------------------------------
Every arm reports accuracy with its chance and majority-class baselines, and the
cheap deterministic heuristic wherever the generator recorded the feature one
would use: "the component this state mentions most" for the needle's choice
question (from `meta["option_mention_counts"]`), keyword argmax for the semantic
control (from `meta["baseline_keyword_prediction"]`), and the best single
threshold on `statement_count` for reachability. The last is fitted in-sample and
is therefore an upper bound on what that feature can do, which is the useful
direction for a baseline.

Tiers are reported per condition within each arm, and boundary crossings are
computed per arm, because a crossing is only meaningful along an ordered axis and
this experiment has four of them (state size, needle position, program depth, and
the two-level encoding contrasts). Every row supplies the full E4 metric set: the
row's own axis supplies its own value and the remaining metrics come from the
experiment-level measurement, so each rung of the plan's ladder can actually be
evaluated and rows stay comparable.

Token accounting
----------------
`filler` sizes its states by a characters/4 estimate and labels it an estimate.
Only `usage.input_tokens` is authoritative, so every call's true input token
count is recorded beside the estimate and the ratio is reported per condition.
The dilution axis is plotted against both.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from itertools import chain
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .. import config, metrics, plots, predictions, tiers
from ..client import Call, CallResult, JevClient
from ..generators import filler, progreach, semantic
from ..instances import Instance, rng_for, shuffled_questions

EXPERIMENT = "E4"
LOG_NAME = "e4.jsonl"

# Arm 5 sits at a size where dilution has begun but has not saturated, so a
# wording effect is not measured on top of a floor or a ceiling.
NOISE_TOKENS = 2000

# ECE over 10 equal-width bins needs mass per bin. Below this many scorable
# probabilities the number is reported as unmeasured rather than as a small
# number that happens to have been computed.
MIN_ECE_N = 50

# Below this many scored observations in the smallest cell of a comparison, a
# few-percent effect is inside the sampling noise, so the predictions that turn
# on such an effect are scored "untestable" with the reason stated. At scale 1.0
# every cell is far above it; this is what keeps a dry run from being reported
# as a falsification.
MIN_N_FOR_SMALL_EFFECT = 100

SIGNIFICANT = 0.05

# The endpoint caps state size and rejects anything over it with HTTP 400 and
# `{"detail": {"error_type": "max_tokens_exceeded"}}`. The plan's dilution
# ladder runs to 50,000 tokens, which is above the cap, so the top of the ladder
# is kept (the rejection is itself the answer to "how much state can you send")
# but each size is probed with a few calls before the rest are issued: a
# condition that cannot succeed should cost a handful of requests against the
# rate limit rather than a full sample.
MAX_TOKENS_ERROR = "max_tokens_exceeded"
CEILING_PROBE = 4

# Probabilities have been observed quantized to 2 decimal places. Recorded as a
# watch item rather than an assumption: the check below measures it.
QUANTIZATION_DP = 2
QUANT_TOL = 1e-9


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------


def _even(n: int) -> int:
    """Round a sample size down to an even number, floor 2.

    `filler.make_needle`, `progreach.generate` and `semantic._questions` all
    alternate ground truth by index, so an even count makes the majority-class
    baseline exactly 0.5 (or exactly 1/k) rather than 0.5 plus a condition-sized
    wobble. An odd count would put a baseline the tier is graded against off by
    1/n for no benefit.
    """
    return max(2, int(n) - (int(n) % 2))


def _mean(xs: Sequence[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or None when it is undefined (n<2, or no variance)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    try:
        return statistics.correlation(list(xs), list(ys))
    except statistics.StatisticsError:
        return None


def _threshold_accuracy(
    feature: Sequence[float], label: Sequence[bool]
) -> tuple[float, float, str] | None:
    """Best accuracy a single threshold on one feature can reach, in-sample.

    Returns (accuracy, threshold, direction) or None when there is too little
    data. Both directions are tried and the better is returned, so this is the
    strongest claim the feature supports and therefore the right thing to put
    beside a model number: if the model does not beat it, twenty lines of code
    would have done the job. Fitted on the same data it is scored on, which
    makes it an upper bound rather than an estimate of held-out performance.
    """
    pairs = [(float(f), bool(y)) for f, y in zip(feature, label)]
    if len(pairs) < 4:
        return None
    cuts = sorted({f for f, _ in pairs})
    best = (-1.0, cuts[0], ">=")
    for cut in cuts:
        for direction in (">=", "<"):
            if direction == ">=":
                hits = sum(1 for f, y in pairs if (f >= cut) == y)
            else:
                hits = sum(1 for f, y in pairs if (f < cut) == y)
            acc = hits / len(pairs)
            if acc > best[0]:
                best = (acc, cut, direction)
    return best


# --------------------------------------------------------------------------
# Observations: one scorable answer, with everything a baseline needs
# --------------------------------------------------------------------------


@dataclass
class Obs:
    condition: str
    arm: str
    pair_key: str  # joins the same item across the arms of a paired comparison
    index: int
    key: str  # question key
    qtype: str
    truth: Any
    predicted: Any
    correct: bool
    p: float | None  # noul probability; None for choice and score
    confidence: float | None
    max_probability: float | None
    chance: float
    # Expected accuracy of the cheap deterministic heuristic on this item:
    # 1.0 right, 0.0 wrong, 1/k when k options tie. None where no heuristic
    # exists for this question type.
    heuristic_score: float | None
    heuristic_name: str | None


@dataclass
class CallStat:
    condition: str
    arm: str
    ok: bool
    latency_s: float
    input_tokens: int
    est_tokens: int | None
    state_chars: int | None


@dataclass
class Batch:
    """One condition's instances and the calls built from them, in step.

    `tokens` is the condition's target state size where it has one, and is what
    the state-size ceiling check compares against. It is None for the arms whose
    states are small and fixed (programs, tickets), which are never at risk of
    the cap.
    """

    condition: str
    arm: str
    instances: list[Instance]
    calls: list[Call]
    tokens: int | None = None


def _answer_space(questions: dict[str, dict]) -> dict:
    """The rubric ids of every score question, in rubric order, or {}.

    A score answer comes back as an index into the wire's `criteria` list, and
    that list holds the rubric's *labels*. Ground truth is a rubric *id*, so
    without the id order the logged answer to a score question cannot be checked
    against the logged truth and that arm stops being rescorable from the JSONL
    alone -- which is what `experiments/__init__` requires of the log. The
    mapping is small and static, so it rides along on every call that carries a
    score question rather than being re-derived from the generator later.
    """
    ids = {
        key: [str(r["id"]) for r in (q.get("rubric") or [])]
        for key, q in questions.items()
        if q["type"] == "score"
    }
    return {"rubric_ids": ids} if ids else {}


def _chance(question: dict) -> float:
    if question["type"] == "choice":
        return 1.0 / max(1, len(question.get("options") or []))
    if question["type"] == "score":
        return 1.0 / max(1, len(question.get("rubric") or []))
    return 0.5


def _predicted(answer: dict) -> Any:
    return answer["predicted"] if answer["type"] == "noul" else answer.get("chosen")


def _heuristic(inst: Instance, key: str, question: dict) -> tuple[float | None, str | None]:
    """The cheap deterministic baseline's score on one question.

    Each branch reads a feature the generator recorded for exactly this purpose;
    none of them looks at the answer. Where a generator states that no cheap
    feature is informative -- the needle's downtime question, whose threshold is
    drawn independently of the answer -- None is returned rather than a rule
    invented here, because a made-up baseline that scores 0.5 is not evidence
    that no baseline exists.
    """
    meta = inst.meta
    if key == "needle_component":
        top = meta.get("most_mentioned_options") or []
        if not top:
            return None, None
        hit = 1.0 / len(top) if meta.get("most_mentioned_is_correct") else 0.0
        return hit, "most-mentioned component"
    if key == semantic.KEY_CHOICE:
        guess = meta.get("baseline_keyword_prediction")
        if guess is None:
            return 0.0, "department keyword argmax"
        return float(guess == inst.truth[key]), "department keyword argmax"
    if key == semantic.KEY_NOUL:
        guess = meta.get("baseline_keyword_prediction")
        asked = meta.get("asked_department")
        if asked is None:
            return None, None
        return float((guess == asked) == inst.truth[key]), "department keyword argmax"
    return None, None


def _observe(batch: Batch, results: Sequence[CallResult]) -> tuple[list[Obs], list[CallStat], dict]:
    """Score one condition's results. Failed calls are counted and dropped.

    `results` is in the order `client.run` was given the calls, which is the
    order of `batch.calls` and therefore of `batch.instances`; nothing is joined
    by id, so a duplicated instance id across two arms cannot cross-contaminate.
    """
    obs: list[Obs] = []
    stats: list[CallStat] = []
    reasons: dict[str, int] = {}
    failed = 0
    excluded = 0

    for inst, result in zip(batch.instances, results):
        est = inst.meta.get("est_tokens")
        chars = inst.meta.get("chars") or inst.meta.get("state_chars")
        stats.append(
            CallStat(
                condition=batch.condition,
                arm=batch.arm,
                ok=result.ok,
                latency_s=result.latency_s,
                input_tokens=result.input_tokens,
                est_tokens=int(est) if est is not None else None,
                state_chars=int(chars) if chars is not None else None,
            )
        )
        if not result.ok:
            failed += 1
            # A failed call loses every answer it would have carried, so the
            # count of excluded data points is not the count of failed calls:
            # the needle conditions ask two questions per call and the semantic
            # ones three.
            excluded += len(inst.questions)
            reason = result.error or f"HTTP {result.http_status}"
            reasons[reason[:120]] = reasons.get(reason[:120], 0) + 1
            continue

        pair_key = str(result.call.meta.get("pair_key") or inst.instance_id)
        for key, answer in result.answers.items():
            question = inst.questions[key]
            truth = inst.truth[key]
            got = _predicted(answer)
            score, hname = _heuristic(inst, key, question)
            obs.append(
                Obs(
                    condition=batch.condition,
                    arm=batch.arm,
                    pair_key=pair_key,
                    index=inst.index,
                    key=key,
                    qtype=answer["type"],
                    truth=truth,
                    predicted=got,
                    correct=bool(got == truth),
                    p=answer.get("p") if answer["type"] == "noul" else None,
                    confidence=answer.get("confidence"),
                    max_probability=answer.get("max_probability"),
                    chance=_chance(question),
                    heuristic_score=score,
                    heuristic_name=hname,
                )
            )

    return obs, stats, {"failed": failed, "excluded": excluded, "reasons": reasons}


# --------------------------------------------------------------------------
# Per-condition aggregation
# --------------------------------------------------------------------------


def _wilson(obs: Sequence[Obs]) -> tuple[float, float]:
    return metrics.wilson_interval(sum(1 for o in obs if o.correct), len(obs))


def _question_stats(obs: Sequence[Obs]) -> dict:
    lo, hi = _wilson(obs)
    truths = [o.truth for o in obs]
    return {
        "n": len(obs),
        "correct": sum(1 for o in obs if o.correct),
        "accuracy": metrics.accuracy([o.correct for o in obs]),
        "chance": _mean([o.chance for o in obs]),
        "majority": metrics.majority_baseline(truths),
        "wilson_95": [lo, hi],
    }


def _calibration(obs: Sequence[Obs]) -> dict:
    """ECE, Brier and AUROC from the noul answers of one condition.

    Reported as unmeasured below MIN_ECE_N rather than computed on a handful of
    points: ten equal-width bins over 12 predictions is not a calibration
    measurement, and a number that looks like one would be graded as one.
    """
    nouls = [o for o in obs if o.qtype == "noul" and o.p is not None]
    out: dict[str, Any] = {"n": len(nouls), "ece": None, "brier": None, "auroc": None,
                           "bins": None, "monotone": None, "note": None}
    if not nouls:
        out["note"] = "no noul answers in this condition"
        return out
    probs = [float(o.p) for o in nouls]
    outcomes = [bool(o.truth) for o in nouls]
    out["brier"] = metrics.brier(probs, outcomes)
    auroc = metrics.auroc(probs, outcomes)
    out["auroc"] = None if math.isnan(auroc) else auroc
    if len(nouls) < MIN_ECE_N:
        out["note"] = f"n={len(nouls)} < {MIN_ECE_N}: ECE not computed, it would not be stable"
        return out
    bins = metrics.reliability_bins(probs, outcomes)
    out["bins"] = bins
    out["ece"] = metrics.ece(probs, outcomes)
    out["monotone"] = metrics.is_monotone(bins)
    return out


def _condition_stats(obs: Sequence[Obs], stats: Sequence[CallStat]) -> dict | None:
    """Everything reported for one condition, or None when nothing scored."""
    if not obs:
        return None
    by_question: dict[str, dict] = {}
    for key in sorted({o.key for o in obs}):
        by_question[key] = _question_stats([o for o in obs if o.key == key])

    heur = [o.heuristic_score for o in obs if o.heuristic_score is not None]
    heur_names = sorted({o.heuristic_name for o in obs if o.heuristic_name})
    lo, hi = _wilson(obs)
    # Weighted by how many answers each question contributed, so a condition
    # mixing an 8-way choice with a yes/no reports the majority-class rate its
    # own answer mix implies rather than an unweighted average of two rates.
    majority = sum(q["majority"] * q["n"] for q in by_question.values()) / len(obs)

    ok = [s for s in stats if s.ok]
    latency = metrics.percentiles([s.latency_s for s in ok]) if ok else None
    true_tokens = [s.input_tokens for s in ok if s.input_tokens]
    est_tokens = [s.est_tokens for s in ok if s.est_tokens]

    return {
        "n": len(obs),
        "correct": sum(1 for o in obs if o.correct),
        "accuracy": metrics.accuracy([o.correct for o in obs]),
        "wilson_95": [lo, hi],
        # Pooled over the questions in the condition; a condition mixing an
        # 8-way choice with a yes/no has a pooled chance rate between the two,
        # and the per-question rates are right below it.
        "chance_baseline": _mean([o.chance for o in obs]),
        "majority_baseline": majority,
        "by_question": by_question,
        "heuristic_baseline": _mean(heur),
        "heuristic_n": len(heur),
        "heuristic_name": ", ".join(heur_names) or None,
        "calibration": _calibration(obs),
        "latency_s": latency,
        "calls_ok": len(ok),
        "input_tokens_mean": _mean(true_tokens),
        "est_tokens_mean": _mean(est_tokens),
        "true_over_estimate": (
            (_mean(true_tokens) / _mean(est_tokens))
            if true_tokens and est_tokens and _mean(est_tokens)
            else None
        ),
    }


def _paired(a: Sequence[Obs], b: Sequence[Obs]) -> dict | None:
    """McNemar on two conditions measured over the same items.

    Items are matched on (pair_key, question key), so a condition carrying two
    questions pairs each of them separately and an item answered in one arm but
    lost to a failed call in the other is dropped from both rather than treated
    as wrong.
    """
    left = {(o.pair_key, o.key): o.correct for o in a}
    right = {(o.pair_key, o.key): o.correct for o in b}
    shared = sorted(set(left) & set(right))
    if not shared:
        return None
    av = [left[k] for k in shared]
    bv = [right[k] for k in shared]
    return {
        "n_pairs": len(shared),
        "accuracy_a": metrics.accuracy(av),
        "accuracy_b": metrics.accuracy(bv),
        "difference": metrics.accuracy(av) - metrics.accuracy(bv),
        "only_a": sum(1 for x, y in zip(av, bv) if x and not y),
        "only_b": sum(1 for x, y in zip(av, bv) if y and not x),
        "mcnemar_p": metrics.mcnemar(av, bv),
    }


# --------------------------------------------------------------------------
# Arm 1 and 2: dilution and needle position
# --------------------------------------------------------------------------


def _needle_condition(tokens: int, position: float) -> str:
    return f"needle/tokens={tokens}/pos={position}"


def _needle_batches(count: int) -> Iterator[Batch]:
    """`filler.difficulty_sweep()` -- the size ladder at position 0, then the
    position ladder at 10,000 tokens.

    The sweep lists the (10000, 0.0) cell once and it serves both arms, which is
    what keeps the position comparison anchored on a cell the size curve also
    reports rather than on a separately drawn one.
    """
    seed = config.seed_for(EXPERIMENT, "needle")
    for difficulty in filler.difficulty_sweep():
        tokens = int(difficulty["tokens"])
        position = float(difficulty["position"])
        condition = _needle_condition(tokens, position)
        instances = filler.generate(difficulty=difficulty, seed=seed, count=count)
        arm = "dilution" if position == filler.DILUTION_POSITION else "position"
        calls = [
            Call(
                experiment=EXPERIMENT,
                condition=condition,
                instance_id=inst.instance_id,
                state=inst.state,
                questions=inst.questions,
                meta={
                    "arm": arm,
                    "truth": inst.truth,
                    "difficulty": inst.difficulty,
                    # The five positions of instance i share one needle, so
                    # this pairs them across positions. The size is in the key
                    # because `make_needle` is keyed on it: two dilution cells
                    # with the same index hold *different* needles, and a key
                    # without the size would silently pair them.
                    "pair_key": f"needle-{tokens}-{inst.index}",
                    **inst.meta,
                    **_answer_space(inst.questions),
                },
            )
            for inst in instances
        ]
        yield Batch(
            condition=condition, arm=arm, instances=instances, calls=calls, tokens=tokens
        )


# --------------------------------------------------------------------------
# Arm 3: encoding
# --------------------------------------------------------------------------


def _program_condition(depth: int, constraint: bool, encoding: str) -> str:
    kind = "constraint" if constraint else "structural"
    return f"progreach/depth={depth}/{kind}/enc={encoding}"


def _program_batches(count: int) -> Iterator[Batch]:
    seed = config.seed_for(EXPERIMENT, "encoding")
    for difficulty in progreach.difficulty_sweep():
        depth = int(difficulty["depth"])
        constraint = bool(difficulty["constraint_required"])
        encoding = str(difficulty["encoding"])
        condition = _program_condition(depth, constraint, encoding)
        instances = progreach.generate(difficulty=difficulty, seed=seed, count=count)
        calls = [
            Call(
                experiment=EXPERIMENT,
                condition=condition,
                instance_id=inst.instance_id,
                state=inst.state,
                questions=inst.questions,
                meta={
                    "arm": "encoding",
                    "truth": inst.truth,
                    "difficulty": inst.difficulty,
                    # The generator's join key: the same program under all three
                    # encodings carries the same program_id.
                    "pair_key": inst.meta["program_id"],
                    **inst.meta,
                    **_answer_space(inst.questions),
                },
            )
            for inst in instances
        ]
        yield Batch(condition=condition, arm="encoding", instances=instances, calls=calls)


# --------------------------------------------------------------------------
# Arm 4: structured versus stringified
# --------------------------------------------------------------------------

STATE_FORMS = ("object", "json-string")

# The plan says `JSON.stringify`, whose output has no spaces after its
# separators. Python's default does, and a whitespace difference is exactly the
# kind of irrelevant formatting the plan asks the report to watch, so it is
# matched rather than left to the default.
_STRINGIFY = dict(separators=(",", ":"), ensure_ascii=False)


def _stateform_batches(count: int) -> Iterator[Batch]:
    """The semantic control set, once as an object and once as its stringified form.

    Drawn at the "hard" level. At "clean" the keyword heuristic scores about
    1.0, so both arms would sit on the ceiling and a real encoding difference
    would be invisible; at "hard" the same heuristic scores about 0.37 and there
    is room for the comparison to move.

    Question key order is shuffled once per instance and reused for both arms.
    Shuffling is required everywhere the experiment is not measuring order
    effects, and reusing the one order is required here specifically: a fresh
    shuffle per arm would put question order inside the thing being compared.
    """
    seed = config.seed_for(EXPERIMENT, "stateform")
    base = semantic.generate(
        difficulty={"level": "hard", "question_type": "all"}, seed=seed, count=count
    )
    ordered: list[Instance] = []
    for inst in base:
        rng = rng_for(f"{EXPERIMENT}:stateform-order", {}, seed, inst.index)
        inst.questions = shuffled_questions(inst.questions, rng)
        ordered.append(inst)

    # Both batches are built before either is yielded. `run` frees a
    # condition's states once it has been scored, and the second arm here is
    # derived from the first arm's instances -- `json.dumps(inst.state)` -- so a
    # generator that built it on resume would stringify states that had already
    # been freed. The tickets are small; this costs nothing.
    built: list[Batch] = []
    for form in STATE_FORMS:
        condition = f"stateform/{form}"
        calls = []
        for inst in ordered:
            state = inst.state if form == "object" else json.dumps(inst.state, **_STRINGIFY)
            calls.append(
                Call(
                    experiment=EXPERIMENT,
                    condition=condition,
                    instance_id=inst.instance_id,
                    state=state,
                    questions=inst.questions,
                    meta={
                        "arm": "stateform",
                        "state_form": form,
                        "truth": inst.truth,
                        "difficulty": inst.difficulty,
                        "pair_key": f"ticket-{inst.index}",
                        "state_chars": len(state if isinstance(state, str) else json.dumps(state)),
                        **inst.meta,
                        **_answer_space(inst.questions),
                    },
                )
            )
        built.append(Batch(condition=condition, arm="stateform", instances=ordered, calls=calls))
    yield from built


# --------------------------------------------------------------------------
# Arm 5: noise floor -- the same fact terse and verbose
# --------------------------------------------------------------------------

FACT_STYLES = ("terse", "verbose")

# Both renderings state the same three facts -- the change request id, the
# component, and the days of downtime -- plus the same walkdown date, and each
# names the component exactly once so that `dilution_corpus(level_component=...)`
# keeps "the component this state mentions most" at chance in both arms. The
# terse form is a record line; the verbose form is the same content spread
# through hedging and scheduling talk that says nothing further about it.
_TERSE_FACT = (
    "{cr_id} | component: {component} | downtime requested: {days} days | "
    "walkdown: {month} {day}"
)

_VERBOSE_FACT = (
    "The change advisory board has now logged change request {cr_id}, and after the "
    "usual back and forth over the schedule it has been attached to the {component} "
    "rather than to any of the other items raised at the same meeting. The downtime "
    "the submitter has asked for, once the pre-checks and the sign-off window are "
    "counted in, comes to {days} days in total, which is what the board minuted. A "
    "walkdown has been pencilled in for {month} {day}, subject to the usual "
    "confirmation from the shift supervisor nearer the time."
)


def _audit_fact_templates() -> None:
    """Hold the two renderings to the same standard as the filler around them.

    `filler` rejects any text of its own containing an answer word ("true",
    "accept") or an answer stem ("reachab", "satisfiab"), so that a filler
    sentence cannot be read as an answer to the question it surrounds. These two
    templates are inserted into filler and must clear the same bar, and the
    check runs at import rather than being asserted in a comment. The
    substituted values come from filler's own audited vocabulary -- `COMPONENTS`
    and `_MONTHS` -- plus an identifier and two integers, so auditing the
    skeleton is sufficient.
    """
    for name, template in (("_TERSE_FACT", _TERSE_FACT), ("_VERBOSE_FACT", _VERBOSE_FACT)):
        filler._audit(re.sub(r"\{[a-z_]+\}", " ", template), f"e4.{name}")
    # Both renderings must name the component exactly once, or
    # `dilution_corpus(level_component=...)` under-levels one arm and "the
    # component this state mentions most" stops being at chance there.
    for name, template in (("_TERSE_FACT", _TERSE_FACT), ("_VERBOSE_FACT", _VERBOSE_FACT)):
        if template.count("{component}") != 1:
            raise AssertionError(f"e4.{name} must name the component exactly once")


_audit_fact_templates()


def _render_fact(needle: filler.Needle, style: str, month: str, day: int) -> str:
    template = _TERSE_FACT if style == "terse" else _VERBOSE_FACT
    return template.format(
        cr_id=needle.cr_id,
        component=needle.component,
        days=needle.days,
        month=month,
        day=day,
    )


def _noise_batches(count: int) -> Iterator[Batch]:
    """One needle per index, rendered both ways inside same-sized filler.

    `dilution_corpus(reserve_chars=...)` sizes the corpus so that the fact plus
    the corpus reaches the requested total, so the verbose arm's longer fact is
    paid for out of the filler and both arms send the same amount of state. The
    fact goes at position 0.0 in both, matching the dilution arm, so position is
    held fixed while wording varies.
    """
    seed = config.seed_for(EXPERIMENT, "noise")
    per_style: dict[str, list[Instance]] = {style: [] for style in FACT_STYLES}
    for i in range(count):
        needle = filler.make_needle(target_tokens=NOISE_TOKENS, seed=seed, index=i)
        date_rng = rng_for(f"{EXPERIMENT}:noise-date", {}, seed, i)
        # Drawn from filler's own month list rather than a local copy, so the
        # fact stays inside the closed vocabulary the audit covers. Drawn once
        # per index and reused for both styles, so the two renderings differ in
        # wording and in nothing else.
        month = date_rng.choice(filler._MONTHS)
        day = date_rng.randrange(1, 29)
        for style in FACT_STYLES:
            fact = _render_fact(needle, style, month, day)
            corpus = filler.dilution_corpus(
                target_tokens=NOISE_TOKENS,
                seed=seed,
                index=i,
                reserve_chars=2 + len(fact),
                level_component=needle.component,
            )
            placement = filler.place_needle(
                paragraphs=corpus.paragraphs, fact=fact, position=filler.DILUTION_POSITION
            )
            state = placement.text
            option_counts = {
                opt: corpus.component_counts.get(opt, 0) + (1 if opt == needle.component else 0)
                for opt in needle.options
            }
            best = max(option_counts.values())
            top = sorted(o for o in option_counts if option_counts[o] == best)
            inst = Instance(
                generator=filler.NAME,
                difficulty={"tokens": NOISE_TOKENS, "fact_style": style},
                seed=seed,
                index=i,
                state=state,
                questions=needle.questions,
                truth=needle.truth,
                meta={
                    "fact_style": style,
                    "fact_chars": len(fact),
                    "target_tokens": NOISE_TOKENS,
                    "est_tokens": filler.estimate_tokens(state),
                    "est_tokens_by_words": filler.estimate_tokens_by_words(state),
                    "token_estimator": "chars/4",
                    "chars": len(state),
                    "words": len(state.split()),
                    "nominal_position": filler.DILUTION_POSITION,
                    "realized_position": placement.realized_position,
                    "n_paragraphs": placement.n_paragraphs,
                    "needle_id": needle.cr_id,
                    "n_options": len(needle.options),
                    "option_mention_counts": option_counts,
                    "most_mentioned_options": top,
                    "most_mentioned_option": top[0],
                    "most_mentioned_is_correct": needle.component in top,
                    "downtime_threshold": needle.threshold,
                    "needle_days": needle.days,
                },
            )
            inst.validate()
            per_style[style].append(inst)

    # Built before either is yielded, for the same reason as the state-form arm:
    # the two styles are paired and must not be affected by `run` freeing the
    # first one's states.
    built: list[Batch] = []
    for style in FACT_STYLES:
        condition = f"factstyle/{style}"
        instances = per_style[style]
        calls = [
            Call(
                experiment=EXPERIMENT,
                condition=condition,
                instance_id=inst.instance_id,
                state=inst.state,
                questions=inst.questions,
                meta={
                    "arm": "noise",
                    "truth": inst.truth,
                    "difficulty": inst.difficulty,
                    "pair_key": f"noise-{inst.index}",
                    **inst.meta,
                    **_answer_space(inst.questions),
                },
            )
            for inst in instances
        ]
        built.append(
            Batch(
                condition=condition,
                arm="noise",
                instances=instances,
                calls=calls,
                tokens=NOISE_TOKENS,
            )
        )
    yield from built


# --------------------------------------------------------------------------
# Issuing one condition, respecting the endpoint's state-size cap
# --------------------------------------------------------------------------


def _all_rejected_for_size(results: Sequence[CallResult]) -> bool:
    return bool(results) and all(
        (not r.ok) and MAX_TOKENS_ERROR in (r.error or "") for r in results
    )


def _execute(
    client: JevClient, batch: Batch, rejected_at: int | None
) -> tuple[list[CallResult], int | None, str | None]:
    """Run one condition. Returns (results, the smallest rejected size, why it stopped).

    Sized conditions are probed before the rest of the sample is issued, and a
    size at or above one already rejected is not attempted at all. Both exist
    because `max_tokens_exceeded` is deterministic in the state size: once
    50,000-token states are refused, issuing 496 more of them buys nothing and
    spends the rate limit that the rest of the experiment needs.
    """
    label = f"E4/{batch.condition}"
    if batch.tokens is not None and rejected_at is not None and batch.tokens >= rejected_at:
        return [], rejected_at, (
            f"not attempted: {rejected_at}-token states were rejected with "
            f"{MAX_TOKENS_ERROR}, and this condition's states are at least as large"
        )

    def note_ceiling(results: list[CallResult], why: str | None) -> tuple[
        list[CallResult], int | None, str | None
    ]:
        if batch.tokens is None or not _all_rejected_for_size(results):
            return results, rejected_at, why
        smallest = batch.tokens if rejected_at is None else min(rejected_at, batch.tokens)
        return results, smallest, why or (
            f"every one of its {len(results)} calls was rejected with {MAX_TOKENS_ERROR}, so this "
            "state size is above the endpoint's cap"
        )

    # A batch no larger than the probe is issued whole; the probe only exists to
    # avoid spending a full sample on a size that cannot succeed, and there is
    # no full sample to save here.
    if batch.tokens is None or len(batch.calls) <= CEILING_PROBE:
        return note_ceiling(client.run(batch.calls, progress_every=100, label=label), None)

    probe = client.run(batch.calls[:CEILING_PROBE], progress_every=0, label=label)
    if _all_rejected_for_size(probe):
        return note_ceiling(
            probe,
            f"stopped after {len(probe)} of {len(batch.calls)} calls: every one was rejected "
            f"with {MAX_TOKENS_ERROR}, so this state size is above the endpoint's cap",
        )
    rest = client.run(batch.calls[CEILING_PROBE:], progress_every=100, label=label)
    return probe + rest, rejected_at, None


# --------------------------------------------------------------------------
# Anomaly checks
# --------------------------------------------------------------------------


def _probability_values(results: Sequence[CallResult]) -> list[float]:
    vals: list[float] = []
    for r in results:
        if not r.ok:
            continue
        for answer in r.answers.values():
            if answer["type"] == "noul":
                vals.append(float(answer["p"]))
            else:
                vals.extend(float(v) for v in (answer.get("probabilities") or {}).values())
    return vals


def _quantization_check(values: Sequence[float]) -> dict:
    """Do the returned probabilities lie on a 2-decimal grid?

    The plan asks the report to watch for probabilities clustering at particular
    values rather than spreading smoothly. Counted over every probability this
    experiment saw -- noul answers and every entry of every returned
    distribution -- so the answer does not depend on which arm is looked at.
    """
    if not values:
        return {"n": 0, "note": "no probabilities observed"}
    on_grid = sum(1 for v in values if abs(round(v, QUANTIZATION_DP) - v) <= QUANT_TOL)
    counts = Counter(round(v, 6) for v in values)
    return {
        "n": len(values),
        "fraction_on_2dp_grid": on_grid / len(values),
        "distinct_values": len(counts),
        "min": min(values),
        "max": max(values),
        "most_common": [v for v, _ in counts.most_common(5)],
    }


def _confidence_check(obs: Sequence[Obs]) -> dict:
    """Does `confidence` track the maximum returned probability?

    Only choice and score carry a confidence field; noul does not, which is why
    nothing in this experiment reports a confidence for a yes/no answer.
    """
    pairs = [
        (o.confidence, o.max_probability)
        for o in obs
        if o.confidence is not None and o.max_probability is not None
    ]
    if not pairs:
        return {"n": 0, "note": "no answer carried both a confidence and a distribution"}
    diffs = [c - m for c, m in pairs]
    return {
        "n": len(pairs),
        "mean_confidence": _mean([c for c, _ in pairs]),
        "mean_max_probability": _mean([m for _, m in pairs]),
        "mean_signed_difference": _mean(diffs),
        "mean_absolute_difference": _mean([abs(d) for d in diffs]),
        "fraction_differing": sum(1 for d in diffs if abs(d) > 0.01) / len(diffs),
        "correlation": _pearson([c for c, _ in pairs], [m for _, m in pairs]),
    }


def _legend_check(results: Sequence[CallResult]) -> dict:
    """The undocumented `legend` field on score answers, reported as found."""
    seen = 0
    total = 0
    example: Any = None
    for r in results:
        if not r.ok:
            continue
        for answer in r.answers.values():
            if answer["type"] != "score":
                continue
            total += 1
            if answer.get("legend"):
                seen += 1
                if example is None:
                    example = answer["legend"]
    return {"score_answers": total, "with_legend": seen, "example": example}


# --------------------------------------------------------------------------
# Plot helper
# --------------------------------------------------------------------------


def _plot(paths: list[str], failures: list[str], run_dir: Path, name: str,
          fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Render one figure, recording rather than raising on failure.

    A plotting error after the calls have been spent must not lose the data, so
    the failure is caught, named in the result, and the run continues. Empty
    conditions are the usual cause at a reduced scale: `plots.curve` refuses an
    empty axis and `plots.reliability_diagram` refuses all-empty bins, both
    deliberately.
    """
    try:
        fn(*args, path=run_dir / "plots" / name, **kwargs)
        paths.append(f"plots/{name}")
    except Exception as exc:  # noqa: BLE001 -- see docstring
        failures.append(f"{name}: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# The experiment
# --------------------------------------------------------------------------


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    # The dilution arm reports ECE per size, so it takes the calibration size;
    # the encoding arm is a paired accuracy comparison, which the plan's
    # methodology section puts at 200. Eight program settings at 200 each is
    # 1,600 programs per encoding, well above the plan's "500 programs per
    # encoding", while keeping each (depth, constraint) cell separately
    # reportable.
    n_cal = _even(config.n(config.SIZES.calibration))
    n_acc = _even(config.n(config.SIZES.accuracy))

    # Built lazily, one condition at a time. The dilution and position arms
    # together are around 260 MB of state strings at scale 1.0, and materialising
    # every condition before the first call would hold all of it for the whole
    # run. Each condition's states are dropped again once it has been scored --
    # see the `state = None` below -- so the peak is one condition's worth.
    planned: Iterator[Batch] = chain(
        _needle_batches(n_cal),
        _program_batches(n_acc),
        _stateform_batches(n_cal),
        _noise_batches(n_cal),
    )
    batches: list[Batch] = []

    all_obs: list[Obs] = []
    all_stats: list[CallStat] = []
    all_results: list[CallResult] = []
    failures = {"calls": 0, "excluded": 0, "reasons": {}}
    stats_by_condition: dict[str, dict] = {}
    obs_by_condition: dict[str, list[Obs]] = {}

    # The smallest state size the endpoint refused, once one has been found.
    rejected_at: int | None = None
    curtailed: dict[str, str] = {}

    with JevClient(run_dir=run_dir, log_name=LOG_NAME) as client:
        for batch in planned:
            batches.append(batch)
            results, rejected_at, why = _execute(client, batch, rejected_at)
            if why:
                curtailed[batch.condition] = why
            all_results.extend(results)
            obs, stats, fail = _observe(batch, results)
            all_obs.extend(obs)
            all_stats.extend(stats)
            obs_by_condition[batch.condition] = obs
            stats_by_condition[batch.condition] = _condition_stats(obs, stats) or {
                "n": 0,
                "note": why or "every call in this condition failed; nothing was scored",
            }
            failures["calls"] += fail["failed"]
            failures["excluded"] += fail["excluded"]
            for reason, count in fail["reasons"].items():
                failures["reasons"][reason] = failures["reasons"].get(reason, 0) + count
            # The state has been sent, logged and scored. Nothing downstream
            # reads it -- the encoding arm's cheap baseline reads meta and truth
            # -- and holding every condition's states to the end of the run is
            # what would make this experiment memory-hungry rather than merely
            # slow.
            for inst in batch.instances:
                inst.state = None
            for call in batch.calls:
                call.state = None
        client_summary = client.summary()

    notes: list[str] = []

    def stat(condition: str) -> dict:
        return stats_by_condition.get(condition) or {}

    def acc(condition: str) -> float | None:
        return stat(condition).get("accuracy")

    # -- arm 1: dilution -------------------------------------------------
    sizes = list(filler.DILUTION_TOKENS)
    dilution = {
        t: stat(_needle_condition(t, filler.DILUTION_POSITION)) for t in sizes
    }
    measured_sizes = [t for t in sizes if dilution[t].get("n")]
    reference_size = measured_sizes[0] if measured_sizes else None
    reference_acc = dilution[reference_size]["accuracy"] if reference_size else None

    def degradation(tokens: int) -> float | None:
        row = dilution.get(tokens) or {}
        if reference_acc is None or not row.get("n"):
            return None
        return reference_acc - row["accuracy"]

    def sweep_degradation(tokens: int) -> float | None:
        """The experiment-level loss to `tokens`, or None when it is not a loss.

        `degradation` is a per-row number: at the reference size it is 0 by
        definition, which is the right thing for that row to carry. The
        experiment-level `degradation_10k` is a different claim -- "accuracy
        lost between the smallest state and a 10,000-token state" -- and if
        every smaller size failed, the reference IS the 10,000-token cell and
        the difference is 0 because it is one cell against itself. Reporting
        that 0 would put "no loss by 10k" in the headline on the strength of a
        single condition, so it is None instead.
        """
        if reference_size is None or reference_size >= tokens:
            return None
        return degradation(tokens)

    if reference_size is not None and reference_size != sizes[0]:
        notes.append(
            f"the {sizes[0]}-token cell produced no scorable answers, so degradation is "
            f"measured against {reference_size} tokens instead and understates the loss"
        )

    # -- arm 2: needle position ------------------------------------------
    positions = list(filler.NEEDLE_POSITIONS)
    position_stats = {
        p: stat(_needle_condition(filler.POSITION_SWEEP_TOKENS, p)) for p in positions
    }
    pos_acc = {p: position_stats[p].get("accuracy") for p in positions}
    measured_pos = [p for p in positions if pos_acc[p] is not None]
    position_effect = (
        max(pos_acc[p] for p in measured_pos) - min(pos_acc[p] for p in measured_pos)
        if len(measured_pos) >= 2
        else None
    )
    middle_dip = None
    if all(pos_acc.get(p) is not None for p in (0.0, 0.5, 1.0)):
        middle_dip = (pos_acc[0.0] + pos_acc[1.0]) / 2.0 - pos_acc[0.5]
    middle_paired = _paired(
        obs_by_condition.get(_needle_condition(filler.POSITION_SWEEP_TOKENS, 0.0), []),
        obs_by_condition.get(_needle_condition(filler.POSITION_SWEEP_TOKENS, 0.5), []),
    )

    # -- arm 3: encoding --------------------------------------------------
    enc_analysis = _encoding_analysis(batches, obs_by_condition, stats_by_condition)
    ladder = enc_analysis["ladder"]
    encoding_cells = enc_analysis["cells"]
    encoding_accuracy = enc_analysis["pooled_accuracy"]
    encoding_spread = enc_analysis["spread"]
    encoding_pairs = enc_analysis["paired_tests"]
    reachability_split = enc_analysis["reachability_split"]
    prog_heuristic = enc_analysis["cheap_heuristic"]
    prog_heuristic_n = enc_analysis["cheap_heuristic_n"]

    # -- arm 4: structured versus stringified -----------------------------
    stateform_paired = _paired(
        obs_by_condition.get("stateform/object", []),
        obs_by_condition.get("stateform/json-string", []),
    )
    stateform_by_question = {}
    for key in (semantic.KEY_NOUL, semantic.KEY_CHOICE, semantic.KEY_SCORE):
        stateform_by_question[key] = _paired(
            [o for o in obs_by_condition.get("stateform/object", []) if o.key == key],
            [o for o in obs_by_condition.get("stateform/json-string", []) if o.key == key],
        )
    stateform_spread = abs(stateform_paired["difference"]) if stateform_paired else None

    # -- arm 5: noise floor -----------------------------------------------
    noise_paired = _paired(
        obs_by_condition.get("factstyle/terse", []),
        obs_by_condition.get("factstyle/verbose", []),
    )
    noise_spread = abs(noise_paired["difference"]) if noise_paired else None

    # -- experiment-level rubric metrics ----------------------------------
    deg_5k = sweep_degradation(5000)
    deg_10k = sweep_degradation(10000)
    deg_50k = sweep_degradation(50000)
    stranded = [
        t
        for t in (5000, 10000, 50000)
        if (dilution.get(t) or {}).get("n") and sweep_degradation(t) is None
    ]
    if stranded:
        notes.append(
            "no dilution cell smaller than "
            + ", ".join(f"{t}" for t in stranded)
            + f" tokens produced a scorable answer, so the loss to {'those sizes' if len(stranded) > 1 else 'that size'} "
            "is not measurable and is reported as unmeasured rather than as zero"
        )
    past_few_thousand_obs = [
        o
        for t in sizes if t >= 5000
        for o in obs_by_condition.get(_needle_condition(t, filler.DILUTION_POSITION), [])
    ]
    past_few_thousand = (
        metrics.accuracy([o.correct for o in past_few_thousand_obs])
        if past_few_thousand_obs
        else None
    )
    past_few_thousand_chance = _mean([o.chance for o in past_few_thousand_obs])

    by_difficulty, crossings, crossing_detail, partial_tiers = _grade(
        deg_5k=deg_5k,
        deg_10k=deg_10k,
        deg_50k=deg_50k,
        position_effect=position_effect,
        middle_dip=middle_dip,
        encoding_spread=encoding_spread,
        dilution=dilution,
        sizes=sizes,
        degradation=degradation,
        position_stats=position_stats,
        pos_acc=pos_acc,
        encoding_cells=encoding_cells,
        prog_heuristic=prog_heuristic,
        stats_by_condition=stats_by_condition,
        stateform_spread=stateform_spread,
        noise_spread=noise_spread,
    )

    plot_paths, plot_failures = _render_plots(
        run_dir,
        dilution=dilution,
        sizes=sizes,
        position_stats=position_stats,
        pos_acc=pos_acc,
        ladder=ladder,
        stats_by_condition=stats_by_condition,
        stateform_paired=stateform_paired,
        stateform_by_question=stateform_by_question,
        noise_paired=noise_paired,
    )

    # -- predictions --------------------------------------------------------
    smallest_dilution_n = min(
        (dilution[t]["n"] for t in sizes if dilution[t].get("n")), default=0
    )
    smallest_position_n = min(
        (position_stats[p]["n"] for p in positions if position_stats[p].get("n")), default=0
    )
    pred = _score_predictions(
        deg_10k=deg_10k,
        deg_50k=deg_50k,
        middle_dip=middle_dip,
        middle_paired=middle_paired,
        dilution_n=smallest_dilution_n,
        position_n=smallest_position_n,
        rejected_at=rejected_at,
        encoding_accuracy=encoding_accuracy,
        encoding_pairs=encoding_pairs,
        reachability_split=reachability_split,
    )

    # -- anomalies ----------------------------------------------------------
    quant = _quantization_check(_probability_values(all_results))
    conf = _confidence_check(all_obs)
    legend = _legend_check(all_results)
    ok_stats = [s for s in all_stats if s.ok]
    latency_vs_tokens = _pearson(
        [float(s.input_tokens) for s in ok_stats], [s.latency_s for s in ok_stats]
    )
    token_ratios = {
        t: dilution[t].get("true_over_estimate") for t in sizes if dilution[t].get("n")
    }

    largest_true = max(
        (s["input_tokens_mean"] for s in dilution.values() if s.get("input_tokens_mean")),
        default=None,
    )
    anomalies = _anomalies(
        rejected_at=rejected_at,
        curtailed=curtailed,
        largest_true_tokens=largest_true,
        partial_tiers=partial_tiers,
        quantization=quant,
        confidence=conf,
        legend=legend,
        latency_vs_input_tokens=latency_vs_tokens,
        token_ratios=token_ratios,
        reachability_split=reachability_split,
        encoding_accuracy=encoding_accuracy,
        stateform_paired=stateform_paired,
        noise_paired=noise_paired,
        prog_heuristic=prog_heuristic,
        stats_by_condition=stats_by_condition,
        crossing_detail=crossing_detail,
        plot_failures=plot_failures,
        notes=notes,
        n_cal=n_cal,
        n_acc=n_acc,
    )

    headline_n = (dilution.get(10000) or {}).get("n")
    return {
        "experiment": EXPERIMENT,
        "question": (
            "How much state can you send before answers degrade, does position within "
            "the state matter, and does the encoding of the state change the answer?"
        ),
        "headline": {
            "metric": (
                "degradation_10k -- accuracy lost between the smallest state and a "
                "10,000-token state"
            ),
            "value": deg_10k,
            "baseline": 0.0,
            "baseline_name": "a flat curve (no loss)",
            "n": headline_n,
        },
        "by_difficulty": by_difficulty,
        "boundary_crossings": crossings,
        "plots": plot_paths,
        "predictions": pred,
        "anomalies": anomalies,
        "failures": failures,
        # Everything below is outside the contract's fixed keys; the report
        # ignores it and `format_report` prints from it, so a rescoring pass has
        # the intermediate numbers without re-deriving them from the log.
        "sample_sizes": {
            "calibration_per_condition": n_cal,
            "accuracy_per_condition": n_acc,
            "scale": config.SCALE,
            "calls_planned": sum(len(b.calls) for b in batches),
            "calls_issued": client_summary.get("total_calls"),
        },
        "state_size_ceiling": {
            "smallest_rejected_target_tokens": rejected_at,
            "error_type": MAX_TOKENS_ERROR if rejected_at else None,
            "largest_measured_true_input_tokens": max(
                (s["input_tokens_mean"] for s in dilution.values() if s.get("input_tokens_mean")),
                default=None,
            ),
            "curtailed_conditions": curtailed,
        },
        "dilution": {
            "sizes": sizes,
            "reference_size": reference_size,
            "per_size": {str(t): dilution[t] for t in sizes},
            "degradation_5k": deg_5k,
            "degradation_10k": deg_10k,
            "degradation_50k": deg_50k,
            "accuracy_past_few_thousand": past_few_thousand,
            "accuracy_past_few_thousand_chance": past_few_thousand_chance,
        },
        "needle_position": {
            "tokens": filler.POSITION_SWEEP_TOKENS,
            "per_position": {str(p): position_stats[p] for p in positions},
            "position_effect": position_effect,
            "middle_dip": middle_dip,
            "middle_vs_edge_paired": middle_paired,
        },
        "encoding": {
            "pooled_accuracy": encoding_accuracy,
            "spread": encoding_spread,
            "paired_tests": encoding_pairs,
            "cells": [
                {
                    "depth": c["depth"],
                    "constraint_required": c["constraint_required"],
                    "accuracy": c["accuracy"],
                    "spread": c["spread"],
                    "n": c["n"],
                    "per_encoding": {
                        e: {
                            "accuracy": c["per_encoding"][e].get("accuracy"),
                            "n": c["per_encoding"][e].get("n"),
                        }
                        for e in progreach.ENCODINGS
                    },
                }
                for c in encoding_cells
            ],
            "reachability_split": reachability_split,
            "cheap_heuristic": (
                {
                    "name": "best in-sample threshold on statement_count (an upper bound)",
                    "accuracy": prog_heuristic[0],
                    "threshold": prog_heuristic[1],
                    "direction": prog_heuristic[2],
                    "n": prog_heuristic_n,
                }
                if prog_heuristic
                else None
            ),
        },
        "state_form": {
            "paired": stateform_paired,
            "by_question": stateform_by_question,
            "per_form": {f: stat(f"stateform/{f}") for f in STATE_FORMS},
        },
        "fact_style": {
            "tokens": NOISE_TOKENS,
            "paired": noise_paired,
            "per_style": {s: stat(f"factstyle/{s}") for s in FACT_STYLES},
        },
        "checks": {
            "quantization": quant,
            "confidence_vs_max_probability": conf,
            "score_legend": legend,
            "latency_vs_input_tokens_r": latency_vs_tokens,
            "true_over_estimated_tokens": token_ratios,
        },
        "crossing_detail": crossing_detail,
        "partial_tiers": partial_tiers,
        "plot_failures": plot_failures,
        "notes": notes,
        "run": client_summary,
    }


def _encoding_analysis(
    batches: Sequence[Batch],
    obs_by_condition: dict[str, list[Obs]],
    stats_by_condition: dict[str, dict],
) -> dict:
    """Arm 3, including the bridge to E2.

    The three encodings of one program share a `program_id`, so every comparison
    here is paired on it: per (depth, constraint_required) cell, then pooled
    across cells. `reachability_split` separates the cells whose answer follows
    from control flow alone from those that need the enclosing integer
    conditions satisfied, which is the distinction the plan wants localised.
    """

    def stat(condition: str) -> dict:
        return stats_by_condition.get(condition) or {}

    def acc(condition: str) -> float | None:
        return stat(condition).get("accuracy")

    ladder = sorted({(int(d["depth"]), bool(d["constraint_required"]))
                     for d in progreach.difficulty_sweep()})
    encoding_cells: list[dict] = []
    for depth, constraint in ladder:
        per_enc = {
            enc: stat(_program_condition(depth, constraint, enc)) for enc in progreach.ENCODINGS
        }
        present = {e: s["accuracy"] for e, s in per_enc.items() if s.get("n")}
        encoding_cells.append(
            {
                "depth": depth,
                "constraint_required": constraint,
                "per_encoding": per_enc,
                "accuracy": _mean(list(present.values())),
                "n": sum(s.get("n", 0) for s in per_enc.values()),
                "spread": (
                    max(present.values()) - min(present.values())
                    if len(present) >= 2
                    else None
                ),
            }
        )

    pooled_encoding: dict[str, list[Obs]] = {enc: [] for enc in progreach.ENCODINGS}
    for depth, constraint in ladder:
        for enc in progreach.ENCODINGS:
            pooled_encoding[enc].extend(
                obs_by_condition.get(_program_condition(depth, constraint, enc), [])
            )
    encoding_accuracy = {
        enc: (metrics.accuracy([o.correct for o in obs]) if obs else None)
        for enc, obs in pooled_encoding.items()
    }
    measured_enc = {e: a for e, a in encoding_accuracy.items() if a is not None}
    encoding_spread = (
        max(measured_enc.values()) - min(measured_enc.values()) if len(measured_enc) >= 2 else None
    )
    encoding_pairs = {
        f"{a}_vs_{b}": _paired(pooled_encoding[a], pooled_encoding[b])
        for a, b in (("cfg", "ast"), ("ast", "source"), ("cfg", "source"))
    }

    # The bridge to E2: structural reachability against constraint-dependent
    # reachability, at the same depths and in the same encodings.
    structural_obs = [
        o for depth, constraint in ladder if not constraint
        for enc in progreach.ENCODINGS
        for o in obs_by_condition.get(_program_condition(depth, constraint, enc), [])
    ]
    constraint_obs = [
        o for depth, constraint in ladder if constraint
        for enc in progreach.ENCODINGS
        for o in obs_by_condition.get(_program_condition(depth, constraint, enc), [])
    ]
    reachability_split = {
        "structural": _question_stats(structural_obs) if structural_obs else None,
        "constraint": _question_stats(constraint_obs) if constraint_obs else None,
        "per_depth": [
            {
                "depth": depth,
                "structural": _mean(
                    [
                        v for v in (
                            acc(_program_condition(depth, False, e)) for e in progreach.ENCODINGS
                        ) if v is not None
                    ]
                ),
                "constraint": _mean(
                    [
                        v for v in (
                            acc(_program_condition(depth, True, e)) for e in progreach.ENCODINGS
                        ) if v is not None
                    ]
                ),
            }
            for depth in sorted({d for d, _ in ladder})
        ],
    }
    if reachability_split["structural"] and reachability_split["constraint"]:
        reachability_split["gap"] = (
            reachability_split["structural"]["accuracy"]
            - reachability_split["constraint"]["accuracy"]
        )
    else:
        reachability_split["gap"] = None

    # The cheap deterministic heuristic for reachability: the best single
    # threshold on the statement count the generator records. It is fitted
    # in-sample, so it is the ceiling of what that feature can do.
    prog_feature: list[float] = []
    prog_label: list[bool] = []
    for batch in batches:
        if batch.arm != "encoding":
            continue
        scored = {o.pair_key for o in obs_by_condition.get(batch.condition, [])}
        for inst in batch.instances:
            if inst.meta["program_id"] in scored:
                prog_feature.append(float(inst.meta["statement_count"]))
                prog_label.append(bool(inst.truth["reachable"]))
    prog_heuristic = _threshold_accuracy(prog_feature, prog_label)

    return {
        "ladder": ladder,
        "cells": encoding_cells,
        "pooled_accuracy": encoding_accuracy,
        "spread": encoding_spread,
        "paired_tests": encoding_pairs,
        "reachability_split": reachability_split,
        "cheap_heuristic": prog_heuristic,
        "cheap_heuristic_n": len(prog_feature),
    }


def _grade(
    *,
    deg_5k: float | None,
    deg_10k: float | None,
    deg_50k: float | None,
    position_effect: float | None,
    middle_dip: float | None,
    encoding_spread: float | None,
    dilution: dict[int, dict],
    sizes: Sequence[int],
    degradation: Callable[[int], float | None],
    position_stats: dict[float, dict],
    pos_acc: dict[float, float | None],
    encoding_cells: Sequence[dict],
    prog_heuristic: tuple[float, float, str] | None,
    stats_by_condition: dict[str, dict],
    stateform_spread: float | None,
    noise_spread: float | None,
) -> tuple[list[dict], dict[str, Any], dict[str, dict], list[dict]]:
    """Tier every condition of every arm, and find each arm's boundary crossings.

    Returns (by_difficulty rows, crossings, per-arm crossing detail, rows whose
    tier rests on part of a rule).
    """

    def stat(condition: str) -> dict:
        return stats_by_condition.get(condition) or {}

    # The experiment-level values of the rubric's sweep-wide metrics. Every
    # graded row starts from these; see `row`. `accuracy_past_few_thousand` is
    # deliberately absent -- it is a statement about a row's own states, so a
    # row whose states are small must not supply it, or the disqualifier would
    # grade a 200-token condition on what happens at 50,000.
    shared = {
        "degradation_5k": deg_5k,
        "degradation_10k": deg_10k,
        "degradation_50k": deg_50k,
        "position_effect": position_effect,
        "middle_dip": middle_dip,
        "encoding_spread": encoding_spread,
    }

    def row(difficulty: dict, s: dict, overrides: dict) -> tiers.TierResult | dict:
        """One graded condition, or a plain "not measured" row.

        Every graded row carries the full E4 metric set. The row's own axis
        supplies its own value through `overrides`; the rest come from the
        experiment-level measurement, so each rung of the plan's ladder is
        evaluable on every row and the rows stay comparable. Without that, a row
        would be graded on whichever criteria it happened to supply and the tiers
        of two arms would not mean the same thing.

        A condition that scored nothing is not graded at all. Because the
        sweep-level metrics are shared, a rung could otherwise be matched on
        those alone and a condition whose every call was refused would come back
        Superhuman. The plan's rule that a failed call is never a data point
        applies to a whole condition of them too.
        """
        if not s.get("n"):
            return {
                "difficulty": difficulty,
                "tier": "not measured",
                "n": 0,
                "metrics": {},
                "metric": tiers.rubric(EXPERIMENT).headline,
                "value": None,
                "baseline": None,
                "baseline_name": tiers.rubric(EXPERIMENT).baseline_label,
                "baseline_source": "missing",
                "reportable": False,
                "fully_evaluated": False,
                "matched_rule": "",
                "reasons": [
                    s.get("note")
                    or "this condition produced no scorable answers, so it has no tier"
                ],
                "signatures": [],
                "missing": [],
                "unevaluated": [],
                "undefined": [],
            }
        full: dict[str, Any] = dict(shared)
        full.update(overrides)
        cal = s.get("calibration") or {}
        full.update(
            {
                "accuracy": s.get("accuracy"),
                "chance_baseline": s.get("chance_baseline"),
                "n": s.get("n"),
                "ece": cal.get("ece"),
            }
        )
        if cal.get("monotone") is not None:
            full["reliability_monotone"] = cal["monotone"]
        if s.get("heuristic_baseline") is not None:
            full["heuristic_baseline"] = s["heuristic_baseline"]

        # Insertion order matters: the report prints only the first four metric
        # keys it meets, in order, so the four that lead are the ones that carry
        # each arm's own axis. accuracy first, then degradation_10k for the
        # dilution rows, position_effect for the position rows, and
        # encoding_spread for the three encoding contrasts. Everything else is
        # still in the row's metrics in the result JSON.
        lead = ("accuracy", "degradation_10k", "position_effect", "encoding_spread")
        m = {k: full[k] for k in lead if k in full}
        m.update({k: v for k, v in full.items() if k not in m})
        return tiers.assign(EXPERIMENT, m, difficulty=difficulty)

    # "Past a few thousand tokens" is read as 5,000 and up, which is where the
    # plan's own dilution boundaries start.
    PAST_FEW_THOUSAND = 5000

    dilution_rows = [
        row(
            {"arm": "dilution", "tokens": t},
            dilution[t],
            {
                # At size t all three read "the loss from the smallest state to
                # this one". The rubric names them after the sizes the plan draws
                # its boundaries at; the values at exactly 5k/10k/50k are the
                # headline and the "dilution" block below.
                "degradation_5k": degradation(t),
                "degradation_10k": degradation(t),
                "degradation_50k": degradation(t),
                "accuracy_past_few_thousand": (
                    dilution[t].get("accuracy") if t >= PAST_FEW_THOUSAND else None
                ),
            },
        )
        for t in sizes
    ]
    best_position = max(
        (a for a in pos_acc.values() if a is not None), default=None
    )
    # A position effect is a difference between positions, so one measured
    # position cannot supply it. Without this the sole surviving row would carry
    # position_effect 0.0 -- "no position effect" -- from a comparison that was
    # never made, and be graded Superhuman on it.
    n_measured_positions = sum(1 for a in pos_acc.values() if a is not None)
    position_rows = [
        row(
            {"arm": "position", "position": p},
            position_stats[p],
            {
                # The glossary defines position_effect as max minus min across
                # positions; per row that is this position's shortfall from the
                # best one, so the worst row carries the whole sweep's effect.
                "position_effect": (
                    best_position - pos_acc[p]
                    if n_measured_positions >= 2
                    and best_position is not None
                    and pos_acc.get(p) is not None
                    else None
                ),
                "accuracy_past_few_thousand": (
                    position_stats[p].get("accuracy")
                    if filler.POSITION_SWEEP_TOKENS >= PAST_FEW_THOUSAND
                    else None
                ),
            },
        )
        for p in pos_acc
    ]

    def encoding_row(cell: dict) -> tiers.TierResult | dict:
        return row(
            {
                "arm": "encoding",
                "kind": "constraint" if cell["constraint_required"] else "structural",
                "depth": cell["depth"],
            },
            {
                "accuracy": cell["accuracy"],
                # progreach alternates its answer by index, so with an even
                # count both the majority-class and the random baseline are
                # exactly 0.5.
                "chance_baseline": 0.5,
                "n": cell["n"],
                "heuristic_baseline": prog_heuristic[0] if prog_heuristic else None,
            },
            # The three per-encoding accuracies behind this spread are in the
            # "encoding" block of the result and in the terminal summary; they
            # are kept out of the row metrics because the report builds one
            # table from the union of every row's keys and three more columns
            # on every row of every arm is not worth it.
            {"encoding_spread": cell["spread"]},
        )

    # Split by constraint_required rather than sorted together: a crossing is a
    # statement about one ordered axis, and structural and constraint-dependent
    # reachability are two different questions that happen to share a depth
    # ladder. Reporting them separately is also the E2 bridge the plan asks for.
    structural_rows = [encoding_row(c) for c in encoding_cells if not c["constraint_required"]]
    constraint_rows = [encoding_row(c) for c in encoding_cells if c["constraint_required"]]
    stateform_rows = [
        row(
            {"arm": "state-form", "form": form},
            stat(f"stateform/{form}"),
            {"encoding_spread": stateform_spread},
        )
        for form in STATE_FORMS
    ]
    noise_rows = [
        row(
            {"arm": "fact-style", "style": style},
            stat(f"factstyle/{style}"),
            {"encoding_spread": noise_spread},
        )
        for style in FACT_STYLES
    ]

    # Crossings are computed per arm: a crossing is a statement about one ordered
    # axis, and sweeping across four concatenated axes would invent knees at the
    # seams between them.
    crossings: dict[str, Any] = {}
    crossing_detail: dict[str, dict[str, Any]] = {}
    arms = (
        ("dilution", dilution_rows),
        ("position", position_rows),
        ("structural reachability by depth", structural_rows),
        ("constraint reachability by depth", constraint_rows),
        ("state form", stateform_rows),
        ("fact style", noise_rows),
    )
    for arm_name, rows in arms:
        # Ungraded conditions are left out: a boundary cannot be crossed at a
        # condition that produced no data, and including it would put the
        # crossing at the wrong difficulty.
        graded = [r for r in rows if isinstance(r, tiers.TierResult)]
        if not graded:
            continue
        detail: dict[str, Any] = {}
        for boundary, crossing in tiers.boundary_crossings(EXPERIMENT, graded).items():
            if crossing is None:
                continue
            crossings[f"{arm_name}: {boundary}"] = crossing.difficulty
            detail[boundary] = crossing.to_json()
        if detail:
            detail["_start_tier"] = graded[0].tier
            crossing_detail[arm_name] = detail

    by_difficulty = [
        r.to_json() if isinstance(r, tiers.TierResult) else r for _, rows in arms for r in rows
    ]

    # Rows whose tier was decided without every criterion of its rung being
    # measured. With the 50,000-token condition unreachable there is no
    # degradation_50k, so the Perfect rung -- "flat to 50k, no position effect,
    # encoding-invariant" -- can be matched on two legs out of three. The tier is
    # a true reading of the rule and a misleading thing to print unqualified, so
    # it is named.
    partial_tiers = [
        {
            "difficulty": r.difficulty,
            "tier": r.tier,
            "unevaluated": list(r.unevaluated),
        }
        for _, rows in arms
        for r in rows
        if isinstance(r, tiers.TierResult)
        and not r.is_fully_evaluated
        and tiers.tier_index(r.tier) < tiers.tier_index("Bad")
    ]
    return by_difficulty, crossings, crossing_detail, partial_tiers


def _render_plots(
    run_dir: Path,
    *,
    dilution: dict[int, dict],
    sizes: Sequence[int],
    position_stats: dict[float, dict],
    pos_acc: dict[float, float | None],
    ladder: Sequence[tuple[int, bool]],
    stats_by_condition: dict[str, dict],
    stateform_paired: dict | None,
    stateform_by_question: dict,
    noise_paired: dict | None,
) -> tuple[list[str], list[str]]:
    """Every figure this experiment emits. Returns (paths written, failures).

    Each figure is guarded on having something to draw, because at a reduced
    scale a condition can be empty and both `plots.curve` and
    `plots.reliability_diagram` refuse an empty axis rather than emitting a
    blank figure with a caption.
    """
    plot_paths: list[str] = []
    plot_failures: list[str] = []
    positions = list(pos_acc)

    def stat(condition: str) -> dict:
        return stats_by_condition.get(condition) or {}

    def acc(condition: str) -> float | None:
        return stat(condition).get("accuracy")
    # -- plots -------------------------------------------------------------
    plotted_sizes = [t for t in sizes if dilution[t].get("n")]
    if plotted_sizes:
        pooled = [dilution[t]["accuracy"] for t in plotted_sizes]
        band = (
            [dilution[t]["wilson_95"][0] for t in plotted_sizes],
            [dilution[t]["wilson_95"][1] for t in plotted_sizes],
        )
        series = {"both questions pooled": pooled}
        for key in ("needle_component", "needle_downtime"):
            vals = [
                (dilution[t].get("by_question", {}).get(key) or {}).get("accuracy")
                for t in plotted_sizes
            ]
            if all(v is not None for v in vals):
                series[key] = vals
        _plot(
            plot_paths, plot_failures, run_dir, "e4_dilution_accuracy.png", plots.curve,
            plotted_sizes, series,
            title="E4.1 needle retrieval against state size",
            xlabel="state size (estimated tokens, log scale)",
            ylabel="accuracy",
            n={t: dilution[t]["n"] for t in plotted_sizes},
            band={"both questions pooled": band},
            baseline=_mean([dilution[t]["chance_baseline"] for t in plotted_sizes]),
            baseline_label="pooled chance",
            logx=True, xticks=plotted_sizes, ylim=(0.0, 1.02),
            n_unit="scored answers per size",
        )
        ece_sizes = [
            t for t in plotted_sizes
            if (dilution[t].get("calibration") or {}).get("ece") is not None
        ]
        if ece_sizes:
            _plot(
                plot_paths, plot_failures, run_dir, "e4_dilution_ece.png", plots.curve,
                ece_sizes, [dilution[t]["calibration"]["ece"] for t in ece_sizes],
                title="E4.1 calibration error against state size",
                xlabel="state size (estimated tokens, log scale)",
                ylabel="ECE (10 equal-width bins)",
                n={t: dilution[t]["calibration"]["n"] for t in ece_sizes},
                series_label="ECE",
                logx=True, xticks=ece_sizes,
                n_unit="yes/no answers per size",
                note=(
                    "ECE comes from the needle_downtime question; noul is the only "
                    "answer type that returns a probability"
                ),
            )
        true_tok = [dilution[t].get("input_tokens_mean") for t in plotted_sizes]
        est_tok = [dilution[t].get("est_tokens_mean") for t in plotted_sizes]
        if all(v for v in true_tok) and all(v for v in est_tok):
            _plot(
                plot_paths, plot_failures, run_dir, "e4_token_accounting.png", plots.curve,
                plotted_sizes,
                {"usage.input_tokens (true)": true_tok, "chars/4 (estimate)": est_tok},
                title="E4.1 true input tokens against the generator's estimate",
                xlabel="target size (estimated tokens, log scale)",
                ylabel="mean tokens per call",
                n={t: dilution[t]["calls_ok"] for t in plotted_sizes},
                logx=True, logy=True, xticks=plotted_sizes,
                n_unit="successful calls per size",
            )
        big = max(plotted_sizes)
        cal = dilution[big].get("calibration") or {}
        if cal.get("bins"):
            _plot(
                plot_paths, plot_failures, run_dir, "e4_reliability_largest.png",
                plots.reliability_diagram, cal["bins"],
                title=f"E4.1 reliability at {big} tokens",
                condition=f"{big} estimated tokens",
                ece=cal["ece"], n=cal["n"], monotone=cal["monotone"],
            )

    plotted_pos = [p for p in positions if pos_acc.get(p) is not None]
    if plotted_pos:
        _plot(
            plot_paths, plot_failures, run_dir, "e4_needle_position.png", plots.curve,
            [f"{int(p * 100)}%" for p in plotted_pos],
            [pos_acc[p] for p in plotted_pos],
            title=f"E4.2 needle position at {filler.POSITION_SWEEP_TOKENS} tokens",
            xlabel="position of the relevant fact through the state",
            ylabel="accuracy",
            n={f"{int(p * 100)}%": position_stats[p]["n"] for p in plotted_pos},
            band=(
                [position_stats[p]["wilson_95"][0] for p in plotted_pos],
                [position_stats[p]["wilson_95"][1] for p in plotted_pos],
            ),
            baseline=_mean([position_stats[p]["chance_baseline"] for p in plotted_pos]),
            baseline_label="pooled chance",
            ylim=(0.0, 1.02),
            n_unit="scored answers per position",
            note="total state size is identical at every position",
        )

    for constraint, label in ((False, "structural"), (True, "constraint-dependent")):
        depths = [d for d, c in ladder if c == constraint]
        series = {}
        for enc in progreach.ENCODINGS:
            vals = [acc(_program_condition(d, constraint, enc)) for d in depths]
            if all(v is not None for v in vals):
                series[enc] = vals
        if depths and series:
            _plot(
                plot_paths, plot_failures, run_dir, f"e4_encoding_{label.split('-')[0]}.png",
                plots.curve, depths, series,
                title=f"E4.3 encoding against depth, {label} reachability",
                xlabel="nesting depth of the target line",
                ylabel="accuracy",
                n={
                    d: sum(
                        stat(_program_condition(d, constraint, e)).get("n", 0)
                        for e in progreach.ENCODINGS
                    )
                    for d in depths
                },
                baseline=0.5, baseline_label="majority class",
                xticks=depths, ylim=(0.0, 1.02),
                n_unit="scored answers per depth, all encodings",
            )

    if stateform_paired:
        series = {}
        for key, label in (
            (semantic.KEY_CHOICE, "department (8-way choice)"),
            (semantic.KEY_NOUL, "department check (yes/no)"),
            (semantic.KEY_SCORE, "urgency (4-point score)"),
        ):
            pair = stateform_by_question.get(key)
            if pair:
                series[label] = [pair["accuracy_a"], pair["accuracy_b"]]
        if series:
            _plot(
                plot_paths, plot_failures, run_dir, "e4_state_form.png", plots.curve,
                ["object", "JSON string"], series,
                title="E4.4 structured state against its stringified equivalent",
                xlabel="how the same state was sent",
                ylabel="accuracy",
                n=stateform_paired["n_pairs"],
                ylim=(0.0, 1.02),
                n_unit="paired items",
                note="same tickets, same question order, same option order",
            )

    if noise_paired:
        series = {}
        for key in ("needle_component", "needle_downtime"):
            vals = [
                (stat(f"factstyle/{s}").get("by_question", {}).get(key) or {}).get("accuracy")
                for s in FACT_STYLES
            ]
            if all(v is not None for v in vals):
                series[key] = vals
        if series:
            _plot(
                plot_paths, plot_failures, run_dir, "e4_fact_style.png", plots.curve,
                list(FACT_STYLES), series,
                title=f"E4.5 terse against verbose wording at {NOISE_TOKENS} tokens",
                xlabel="how the relevant fact was written",
                ylabel="accuracy",
                n=noise_paired["n_pairs"],
                ylim=(0.0, 1.02),
                n_unit="paired items",
                note="total state size held equal; only the fact's wording differs",
            )
    return plot_paths, plot_failures



# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


def _score_predictions(
    *,
    deg_10k: float | None,
    deg_50k: float | None,
    middle_dip: float | None,
    middle_paired: dict | None,
    dilution_n: int,
    position_n: int,
    rejected_at: int | None,
    encoding_accuracy: dict[str, float | None],
    encoding_pairs: dict[str, dict | None],
    reachability_split: dict,
) -> list[dict]:
    """Score every E4 prediction from the plan's table, with its evidence.

    `predictions.for_experiment("E4")` is the source of the list rather than a
    hard-coded pair, so a prediction added to the table cannot be silently
    skipped: anything unrecognised comes back "untestable" naming itself.
    """
    out: list[dict] = []
    for p in predictions.for_experiment(EXPERIMENT):
        if p.id == "P10":
            out.append(
                _score_p10(
                    deg_10k, deg_50k, middle_dip, middle_paired,
                    dilution_n, position_n, rejected_at, p,
                )
            )
        elif p.id == "P11":
            out.append(_score_p11(encoding_accuracy, encoding_pairs, reachability_split, p))
        else:
            out.append(
                {
                    "id": p.id,
                    "claim": p.claim,
                    "outcome": (
                        f"{p.id} is listed against E4 in the plan's prediction table but this "
                        "module was built to test P10 and P11 only; it is reported here rather "
                        "than dropped."
                    ),
                    "verdict": "untestable",
                    "evidence": {"falsified_if": p.falsified_if},
                }
            )
    return out


def _score_p10(
    deg_10k: float | None,
    deg_50k: float | None,
    middle_dip: float | None,
    middle_paired: dict | None,
    dilution_n: int,
    position_n: int,
    rejected_at: int | None,
    p: predictions.Prediction,
) -> dict:
    """P10: 5-15% loss by 10k tokens and a 3-8% middle-of-state dip.

    The plan's falsifier -- "flat to 50k" -- needs a 50,000-token state, which
    the endpoint refuses with `max_tokens_exceeded`. That does not make the
    prediction untestable: its two numeric claims are both about 10,000 tokens
    and are measured. So the numeric claims are scored and the unreachable
    falsifier is stated as unreachable, rather than the whole prediction being
    filed away untested.
    """
    fifty_k_unreachable = rejected_at is not None and rejected_at <= 50000
    evidence = {
        "degradation_10k": deg_10k,
        "degradation_50k": deg_50k,
        "middle_dip": middle_dip,
        "middle_vs_edge_paired": middle_paired,
        "predicted_loss_range": [0.05, 0.15],
        "predicted_dip_range": [0.03, 0.08],
        "smallest_dilution_cell_n": dilution_n,
        "smallest_position_cell_n": position_n,
        "fifty_k_unreachable": fifty_k_unreachable,
        "smallest_rejected_target_tokens": rejected_at,
        "falsified_if": p.falsified_if,
    }
    if deg_10k is None or middle_dip is None:
        return {
            "id": p.id, "claim": p.claim, "verdict": "untestable", "evidence": evidence,
            "outcome": (
                "the 10,000-token cell or the needle-position cells produced no scorable answers, "
                "so neither the 10k loss nor the middle-of-state dip could be measured."
            ),
        }
    if min(dilution_n, position_n) < MIN_N_FOR_SMALL_EFFECT:
        return {
            "id": p.id, "claim": p.claim, "verdict": "untestable", "evidence": evidence,
            "outcome": (
                f"the smallest cell holds {min(dilution_n, position_n)} scored answers, below the "
                f"{MIN_N_FOR_SMALL_EFFECT} this module requires before calling a 3-15% effect. "
                f"Measured anyway: loss to 10k {deg_10k:+.3f}, middle dip {middle_dip:+.3f}. "
                "This is a sample-size limit, not an API limit."
            ),
        }

    loss_in_range = 0.05 <= deg_10k <= 0.15
    dip_in_range = 0.03 <= middle_dip <= 0.08
    dip_p = (middle_paired or {}).get("mcnemar_p")
    fifty = (
        f"the plan's falsifier, flatness to 50,000 tokens, could not be evaluated: the endpoint "
        f"refuses states that large with {MAX_TOKENS_ERROR} (smallest rejected target "
        f"{rejected_at} tokens)"
        if fifty_k_unreachable
        else f"loss to 50,000 tokens {deg_50k:+.3f}"
        if deg_50k is not None
        else "the 50,000-token cell produced no scorable answers"
    )
    detail = (
        f"Accuracy loss from the smallest state to 10,000 tokens {deg_10k:+.3f} "
        f"(predicted 0.05-0.15); middle-of-state dip {middle_dip:+.3f} (predicted 0.03-0.08"
        + (f", paired McNemar p={dip_p:.3g}" if dip_p is not None else "")
        + f"); {fifty}."
    )

    if deg_50k is not None and deg_50k <= tiers.EPS_FLAT:
        return {
            "id": p.id, "claim": p.claim, "verdict": "wrong", "evidence": evidence,
            "outcome": (
                f"falsified on the plan's own condition: the curve is flat to 50k (loss "
                f"{deg_50k:+.3f}, inside the {tiers.EPS_FLAT} the harness reads as flat). {detail}"
            ),
        }
    if loss_in_range and dip_in_range:
        return {
            "id": p.id, "claim": p.claim, "verdict": "right", "evidence": evidence,
            "outcome": f"Both quantities landed in the predicted ranges. {detail}",
        }
    missed = []
    if not loss_in_range:
        missed.append("the 10k loss")
    if not dip_in_range:
        missed.append("the middle-of-state dip")
    return {
        "id": p.id, "claim": p.claim, "verdict": "wrong", "evidence": evidence,
        "outcome": (
            f"{' and '.join(missed)} fell outside the predicted range, and the prediction was "
            f"numeric. {detail}"
        ),
    }


def _score_p11(
    encoding_accuracy: dict[str, float | None],
    encoding_pairs: dict[str, dict | None],
    reachability_split: dict,
    p: predictions.Prediction,
) -> dict:
    evidence = {
        "pooled_accuracy": encoding_accuracy,
        "paired_tests": encoding_pairs,
        "structural_vs_constraint": {
            "structural": (reachability_split.get("structural") or {}).get("accuracy"),
            "constraint": (reachability_split.get("constraint") or {}).get("accuracy"),
            "gap": reachability_split.get("gap"),
        },
        "falsified_if": p.falsified_if,
    }
    measured = {e: a for e, a in encoding_accuracy.items() if a is not None}
    n_pairs = min(
        (t["n_pairs"] for t in encoding_pairs.values() if t), default=0
    )
    if len(measured) < 3:
        return {
            "id": p.id, "claim": p.claim, "verdict": "untestable", "evidence": evidence,
            "outcome": (
                f"only {sorted(measured)} produced scorable answers; the ordering claim needs "
                "all three encodings."
            ),
        }
    ranked = sorted(measured, key=lambda e: -measured[e])
    order = ", ".join(f"{e} {measured[e]:.3f}" for e in ranked)
    ps = {k: (t["mcnemar_p"] if t else None) for k, t in encoding_pairs.items()}
    p_txt = ", ".join(f"{k} p={v:.3g}" for k, v in ps.items() if v is not None)
    detail = f"pooled accuracy, best first: {order}. Paired McNemar: {p_txt}."

    if n_pairs < MIN_N_FOR_SMALL_EFFECT:
        return {
            "id": p.id, "claim": p.claim, "verdict": "untestable", "evidence": evidence,
            "outcome": (
                f"only {n_pairs} paired programs per comparison, below the "
                f"{MIN_N_FOR_SMALL_EFFECT} this module requires before calling an ordering. "
                f"Measured anyway: {detail} This is a sample-size limit, not an API limit."
            ),
        }

    top = max(measured.values())
    winners = sorted(e for e in measured if measured[e] == top)
    any_significant = any(v is not None and v < SIGNIFICANT for v in ps.values())

    # "No difference" is tested before "source wins", because a tie is both and
    # only one of the two can be stated. Reading a three-way tie as "source was
    # the best encoding" would be the plan's other falsifier reported under the
    # wrong name.
    if not any_significant:
        tie = (
            f" The best accuracy is shared by {' and '.join(winners)}."
            if len(winners) > 1
            else ""
        )
        return {
            "id": p.id, "claim": p.claim, "verdict": "wrong", "evidence": evidence,
            "outcome": (
                f"falsified on the plan's 'no difference' condition: no pairwise difference "
                f"reached p<{SIGNIFICANT}.{tie} {detail}"
            ),
        }
    if winners == ["source"]:
        return {
            "id": p.id, "claim": p.claim, "verdict": "wrong", "evidence": evidence,
            "outcome": f"falsified: raw source was the best encoding. {detail}",
        }
    ordered = measured["cfg"] > measured["ast"] > measured["source"]
    cfg_vs_source = ps.get("cfg_vs_source")
    if ordered and cfg_vs_source is not None and cfg_vs_source < SIGNIFICANT:
        return {
            "id": p.id, "claim": p.claim, "verdict": "right", "evidence": evidence,
            "outcome": (
                f"the predicted ordering held -- CFG edge list above AST/JSON above raw source -- "
                f"and CFG beat source at p<{SIGNIFICANT}. {detail}"
            ),
        }
    # Two distinguishable ways to arrive here, and they are different findings,
    # so they are not reported under one sentence: the ordering held but the
    # contrast the prediction rests on -- CFG against source -- did not reach
    # significance, or the ordering itself did not hold.
    if not ordered:
        why = "the predicted ordering did not hold"
    elif cfg_vs_source is None:
        why = (
            "the predicted ordering held but CFG and source share no paired programs, so the "
            "contrast the prediction rests on was never tested"
        )
    else:
        why = (
            f"the predicted ordering held but CFG beat source only at p={cfg_vs_source:.3g}, "
            f"which does not reach p<{SIGNIFICANT}, so the ordering is not established"
        )
    return {
        "id": p.id, "claim": p.claim, "verdict": "wrong", "evidence": evidence,
        "outcome": (
            f"source did not win alone and at least one difference is real, so neither of the "
            f"plan's falsifiers fired cleanly, but {why}. {detail}"
        ),
    }


# --------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------


def _anomalies(
    *,
    rejected_at: int | None,
    curtailed: dict[str, str],
    largest_true_tokens: float | None,
    partial_tiers: list[dict],
    quantization: dict,
    confidence: dict,
    legend: dict,
    latency_vs_input_tokens: float | None,
    token_ratios: dict[int, float | None],
    reachability_split: dict,
    encoding_accuracy: dict[str, float | None],
    stateform_paired: dict | None,
    noise_paired: dict | None,
    prog_heuristic: tuple[float, float, str] | None,
    stats_by_condition: dict[str, dict],
    crossing_detail: dict[str, Any],
    plot_failures: list[str],
    notes: list[str],
    n_cal: int,
    n_acc: int,
) -> list[str]:
    """The report's "unexpected behaviors" section, as far as E4 can see it.

    Written as statements of what was measured, including the negative cases,
    because the plan asks the report to watch for these specifically and "we
    looked and it was not there" is the useful answer to a watch item.
    """
    out: list[str] = []

    if rejected_at is not None:
        out.append(
            f"The endpoint caps state size. States targeted at {rejected_at} estimated tokens were "
            f"rejected with HTTP 400 and error_type {MAX_TOKENS_ERROR!r}, an error the docs do not "
            f"describe. The largest size that answered averaged "
            + (
                f"{largest_true_tokens:,.0f} true input tokens"
                if largest_true_tokens
                else "an unrecorded number of tokens"
            )
            + f", so the cap lies between that and whatever a {rejected_at}-token target costs. "
            "The plan's dilution ladder runs to 50,000 tokens and the top of it is therefore not "
            "reachable on this endpoint, which answers 'how much state can you send' with a hard "
            "limit rather than with a degradation curve. Conditions curtailed: "
            + "; ".join(f"{c} ({why})" for c, why in curtailed.items())
            + "."
        )

    if partial_tiers:
        out.append(
            f"{len(partial_tiers)} condition(s) were tiered on part of a rule, because a criterion "
            "of the rung they matched was never measured: "
            + "; ".join(
                f"{r['difficulty']} graded {r['tier']} without {', '.join(r['unevaluated'])}"
                for r in partial_tiers[:6]
            )
            + ("; and others." if len(partial_tiers) > 6 else ".")
            + " Where degradation_50k is the missing criterion the cause is the state size cap "
            "above, not a gap in the experiment."
        )

    quant = quantization
    if quant.get("n"):
        frac = quant["fraction_on_2dp_grid"]
        out.append(
            f"Probability quantization: {frac:.1%} of the {quant['n']} probabilities returned "
            f"sit exactly on a 2-decimal grid, across {quant['distinct_values']} distinct values "
            f"in [{quant['min']:.3f}, {quant['max']:.3f}]. Most common: "
            f"{', '.join(f'{v:g}' for v in quant['most_common'])}."
            + (
                ""
                if frac > 0.999
                else " Some values are off the grid, so the quantization is not total."
            )
        )

    conf = confidence
    if conf.get("n"):
        out.append(
            f"`confidence` against max probability on the {conf['n']} choice and score answers: "
            f"mean confidence {conf['mean_confidence']:.3f}, mean max probability "
            f"{conf['mean_max_probability']:.3f}, mean signed difference "
            f"{conf['mean_signed_difference']:+.3f}, and they differ by more than 0.01 on "
            f"{conf['fraction_differing']:.1%} of answers"
            + (f" (r={conf['correlation']:.3f})." if conf.get("correlation") is not None else ".")
            + " noul answers carry no confidence field at all, so nothing in this experiment "
            "reports a confidence for a yes/no question."
        )

    if legend.get("score_answers"):
        out.append(
            f"The undocumented `legend` field appeared on {legend['with_legend']} of "
            f"{legend['score_answers']} score answers"
            + (f", e.g. {json.dumps(legend['example'])[:160]}." if legend.get("example") else ".")
        )

    ratios = {t: r for t, r in token_ratios.items() if r}
    if ratios:
        smallest, largest = min(ratios), max(ratios)
        lo, hi = ratios[smallest], ratios[largest]
        if lo - hi > 0.1:
            reading = (
                " The ratio falls as the state grows, which is what a fixed per-call overhead "
                "looks like -- the questions, the options and whatever the endpoint wraps them in "
                "-- rather than a mis-scaled estimator. The small conditions therefore cost "
                "noticeably more than their label says."
            )
        elif hi - lo > 0.1:
            reading = (
                " The ratio rises with state size, which a fixed per-call overhead would not "
                "produce; the estimator is scaling wrongly with the content itself."
            )
        else:
            reading = (
                " The ratio is roughly constant across sizes, so the estimator is off by a "
                "steady factor rather than by a fixed per-call overhead."
            )
        out.append(
            "Token accounting: the true `usage.input_tokens` is "
            f"{lo:.2f}x the filler generator's chars/4 estimate at the {smallest}-token "
            f"condition and {hi:.2f}x it at the {largest}-token condition."
            + reading
            + " Every state size labelled in this experiment is an estimate of the content only."
        )

    if latency_vs_input_tokens is not None:
        out.append(
            f"Latency against input tokens across every E4 call: r={latency_vs_input_tokens:.3f}. "
            "E4 holds the question count fixed within each arm, so this is the size term "
            "on its own."
        )

    split = reachability_split
    if split.get("gap") is not None:
        s = split["structural"]
        c = split["constraint"]
        direction = "better" if split["gap"] > 0 else "worse"
        out.append(
            f"Structural against constraint-dependent reachability: {s['accuracy']:.3f} "
            f"(n={s['n']}) against {c['accuracy']:.3f} (n={c['n']}), a gap of "
            f"{split['gap']:+.3f} -- the model is {direction} when reachability is decided by "
            "control flow alone than when it requires satisfying the enclosing integer conditions. "
            "That is the bridge to E2: a gap here localises the weakness to constraint solving "
            "rather than to graph traversal."
        )

    enc = {e: a for e, a in encoding_accuracy.items() if a is not None}
    if len(enc) >= 2:
        best = max(enc, key=lambda e: enc[e])
        n_enc = (reachability_split.get("structural") or {}).get("n", 0) + (
            reachability_split.get("constraint") or {}
        ).get("n", 0)
        out.append(
            "Encoding: "
            + ", ".join(f"{e} {enc[e]:.3f}" for e in ("source", "ast", "cfg") if e in enc)
            + f"; best is {best}, over {n_enc} scored answers in total. The same programs were "
            "sent all three ways, so this is a paired comparison and the difference is not "
            "instance sampling."
        )

    sf = stateform_paired
    if sf:
        out.append(
            f"Structured against stringified state: object {sf['accuracy_a']:.3f}, JSON string "
            f"{sf['accuracy_b']:.3f} over {sf['n_pairs']} paired items, difference "
            f"{sf['difference']:+.3f}, McNemar p={sf['mcnemar_p']:.3g}. The two states carry the "
            "same bytes of content and differ only in whether they arrive as an object, so a "
            "difference here is the plan's 'answers changing with irrelevant formatting'."
        )

    if noise_paired:
        out.append(
            f"Terse against verbose wording of the same fact at equal total state size: "
            f"{noise_paired['accuracy_a']:.3f} against {noise_paired['accuracy_b']:.3f} over "
            f"{noise_paired['n_pairs']} paired items, difference "
            f"{noise_paired['difference']:+.3f}, McNemar p={noise_paired['mcnemar_p']:.3g}."
        )

    heur = prog_heuristic
    if heur:
        beaten = [
            name
            for name, s in stats_by_condition.items()
            if name.startswith("progreach/") and s.get("accuracy") is not None
            and s["accuracy"] <= heur[0]
        ]
        if beaten:
            out.append(
                f"{len(beaten)} reachability condition(s) did not beat the cheap deterministic "
                f"baseline (best in-sample threshold on statement_count, {heur[0]:.3f}). The "
                "generator balances that feature across the two answers by construction, so a "
                "condition at or below it is at the plan's 'better off with 20 lines of code' bar."
            )

    # Summarised per arm rather than per boundary: an arm that starts at Bad
    # records three boundaries as crossed at its easiest condition, and saying
    # so three times buries everything else in this section.
    for arm_name, detail in crossing_detail.items():
        boundaries = {b: d for b, d in detail.items() if not b.startswith("_")}
        floors = [b for b, d in boundaries.items() if d.get("from_tier") is None]
        recovered = [b for b, d in boundaries.items() if d.get("recovered_at") is not None]
        if floors:
            out.append(
                f"On the '{arm_name}' arm the boundaries {', '.join(floors)} are recorded at its "
                f"easiest condition, because the arm was already at {detail.get('_start_tier')} "
                "there. Those are floors, not knees: nothing in the swept range was above them, so "
                "the crossing points say nothing about where degradation begins."
            )
        if recovered:
            out.append(
                f"On the '{arm_name}' arm the boundaries {', '.join(recovered)} were crossed and "
                "then recovered later in the sweep, so the knee is not clean and those crossing "
                "points should be read as noisy rather than as thresholds."
            )

    for failure in plot_failures:
        out.append(f"A figure could not be rendered and is missing from the report: {failure}")
    out.extend(notes)

    if config.SCALE != 1.0:
        out.append(
            f"Sample sizes were scaled by {config.SCALE} "
            f"({n_cal} per calibration condition, {n_acc} per accuracy condition). "
            "Every number above is from a reduced run and is not comparable to the plan's sizes."
        )
    return out


# --------------------------------------------------------------------------
# Terminal summary
# --------------------------------------------------------------------------


def format_report(result: dict) -> str:
    lines: list[str] = []
    h = result.get("headline") or {}
    val = h.get("value")
    lines.append(
        f"E4 input size and encoding -- {h.get('metric', '')}: "
        + (f"{val:+.3f}" if isinstance(val, (int, float)) else "not measured")
        + f" against {h.get('baseline')} ({h.get('baseline_name')}), n={h.get('n')}"
    )

    dil = result.get("dilution") or {}
    lines.append("  dilution (accuracy by state size, both needle questions pooled):")
    for t in dil.get("sizes", []):
        s = (dil.get("per_size") or {}).get(str(t)) or {}
        if not s.get("n"):
            lines.append(f"    {t:>6} tok: {s.get('note', 'nothing scored')}")
            continue
        cal = s.get("calibration") or {}
        ece = f"ECE {cal['ece']:.3f}" if cal.get("ece") is not None else "ECE n/a"
        lines.append(
            f"    {t:>6} tok: acc {s['accuracy']:.3f} "
            f"[{s['wilson_95'][0]:.3f}-{s['wilson_95'][1]:.3f}] "
            f"chance {s['chance_baseline']:.3f}  {ece}  n={s['n']}  "
            f"true tokens {s.get('input_tokens_mean') or 0:.0f}"
        )
    for key in ("degradation_5k", "degradation_10k", "degradation_50k"):
        v = dil.get(key)
        lines.append(f"    {key}: " + (f"{v:+.3f}" if v is not None else "not measured"))
    ceiling = result.get("state_size_ceiling") or {}
    if ceiling.get("smallest_rejected_target_tokens"):
        lines.append(
            f"    state size cap: {ceiling['smallest_rejected_target_tokens']}-token states were "
            f"rejected with {ceiling['error_type']}; the largest that answered averaged "
            f"{ceiling.get('largest_measured_true_input_tokens') or 0:,.0f} true input tokens"
        )

    pos = result.get("needle_position") or {}
    lines.append(f"  needle position at {pos.get('tokens')} tokens:")
    for p, s in (pos.get("per_position") or {}).items():
        if not s.get("n"):
            lines.append(f"    {p:>5}: nothing scored")
            continue
        lines.append(f"    {float(p) * 100:>4.0f}%: acc {s['accuracy']:.3f}  n={s['n']}")
    lines.append(
        f"    position_effect: {_fmt(pos.get('position_effect'))}  "
        f"middle_dip: {_fmt(pos.get('middle_dip'))}"
    )

    enc = result.get("encoding") or {}
    lines.append("  encoding (same programs, three renderings, paired):")
    for e, a in (enc.get("pooled_accuracy") or {}).items():
        lines.append(f"    {e:<7}: " + (f"{a:.3f}" if a is not None else "nothing scored"))
    for name, test in (enc.get("paired_tests") or {}).items():
        if test:
            lines.append(
                f"    {name}: {test['difference']:+.3f} over {test['n_pairs']} pairs, "
                f"McNemar p={test['mcnemar_p']:.3g}"
            )
    split = enc.get("reachability_split") or {}
    if split.get("gap") is not None:
        lines.append(
            f"    structural {split['structural']['accuracy']:.3f} vs constraint "
            f"{split['constraint']['accuracy']:.3f}  gap {split['gap']:+.3f}"
        )
    cheap = enc.get("cheap_heuristic")
    if cheap:
        lines.append(f"    cheap baseline: {cheap['accuracy']:.3f} ({cheap['name']})")

    sf = (result.get("state_form") or {}).get("paired")
    if sf:
        lines.append(
            f"  state form: object {sf['accuracy_a']:.3f} vs JSON string {sf['accuracy_b']:.3f}, "
            f"{sf['difference']:+.3f}, p={sf['mcnemar_p']:.3g}, n={sf['n_pairs']}"
        )
    ns = (result.get("fact_style") or {}).get("paired")
    if ns:
        lines.append(
            f"  fact style: terse {ns['accuracy_a']:.3f} vs verbose {ns['accuracy_b']:.3f}, "
            f"{ns['difference']:+.3f}, p={ns['mcnemar_p']:.3g}, n={ns['n_pairs']}"
        )

    crossings = result.get("boundary_crossings") or {}
    if crossings:
        lines.append("  tier boundaries crossed:")
        for boundary, where in crossings.items():
            lines.append(f"    {boundary}: {where}")
    else:
        lines.append("  no tier boundary was crossed on any arm")

    for p in result.get("predictions") or []:
        lines.append(f"  {p['id']} {p['verdict'].upper()}: {p['outcome']}")

    f = result.get("failures") or {}
    lines.append(f"  failed calls: {f.get('calls', 0)} (excluded: {f.get('excluded', 0)})")
    for a in result.get("anomalies") or []:
        lines.append(f"  ! {a}")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    return f"{v:+.3f}" if isinstance(v, (int, float)) else "not measured"
