"""E1 -- determinism and the noise floor.

This runs first because every other number in the plan is read against it. If
the same request returns probabilities with a standard deviation of 0.10, then a
three-point accuracy difference in E3 is noise that happens to have a direction,
and reporting it as a finding would be a mistake no later experiment can catch.

Four conditions, all on the same 50 states, all 30 repetitions deep:

    identical       the same request bytes every time
    question_order  question keys presented in a different order per repetition
    key_rename      the question keys renamed, everything else fixed
    option_order    choice options presented in a different order per repetition

Reporting them separately is the point of the design. Condition 1 measures the
model's own variance; 2 through 4 add one presentation change each, so the
difference between a later condition and condition 1 is attributable to that
change rather than to the model. A single pooled sigma would confuse the two and
would not tell a downstream experiment whether randomising option order -- which
the plan requires everywhere else -- is itself introducing variance.

Three choices in here need their reasons stated.

*The conditions are interleaved in time, not run one after another.* All four
conditions' calls go into one list, shuffled under a fixed seed, and are
submitted together. Run in blocks, a silent model change partway through the run
would land entirely inside one condition and be read as an effect of that
condition's presentation change. Interleaving spreads any drift across all four.

*Shift is measured per repetition, not per state.* With 50 states the coarsest
possible flip rate is 2%, which cannot resolve P3's predicted 1-3% option-order
effect or P2's 2% threshold at all. So the shift metrics are the rate at which
an individual repetition's answer differs from condition 1's modal answer for
that state and question, minus the rate at which condition 1's own repetitions
differ from their own mode. That subtraction is what makes the number an effect
of the presentation change rather than a restatement of the model's variance,
and 1500 repetition-level observations per condition resolve it to about 0.1%.
Both raw rates are reported beside the difference so the subtraction is visible.

Condition 1's own rate is measured leaving each repetition out of the mode it is
compared against. The mode is fitted on condition 1's answers, so an in-sample
comparison would hand condition 1 a fitting advantage that the subtraction then
reports as a presentation effect. On a simulated model with no presentation
effect at all and a question it answers 55/45 -- which is what the 3SAT
phase-transition stratum exists to produce -- the in-sample version returned a
mean excess of +5.8%, above P3's 5% falsification threshold. Leave-one-out
returns -0.4%, inside the sampling noise of 1500 observations.

*3SAT states are given a second question that the generator does not produce.*
`sat3` emits one noul per formula. A state with one question cannot show a
question-order effect and has no choice question to shuffle options in, so half
the state set would contribute to condition 1 only. Each 3SAT state therefore
also carries a two-option choice between satisfiable and unsatisfiable, whose
ground truth is the same solver decision already in `meta`. Two options is a
narrow choice, but it is the sharpest possible position test: each of the two
orders gets about half the repetitions.

The probability quantization report lives here rather than in a later experiment
because 50 states times 30 repetitions times four conditions is the densest view
of the returned probability values anywhere in the plan, and the plan's
"unexpected behaviors" section asks specifically whether probabilities cluster
at particular values.
"""

from __future__ import annotations

import hashlib
import random
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import config, metrics, plots, predictions, tiers
from ..client import Call, CallResult, JevClient
from ..generators import sat3, semantic
from ..instances import Instance, choice

EXPERIMENT = "E1"

CONDITIONS: tuple[str, ...] = ("identical", "question_order", "key_rename", "option_order")
REFERENCE_CONDITION = "identical"

# Renaming schemes, cycled across repetitions of the key_rename condition. The
# plan's example is `is_urgent` -> `q1` -> `zzz`; "rotated" is the addition,
# because it is the scheme that would actually catch key leakage. It gives each
# question the *next* question's canonical key, so a model reading the key would
# be reading a key that names a different question about the same state. Opaque
# keys can only show that keys are used; rotated ones show what they are used
# for.
RENAME_SCHEMES: tuple[str, ...] = ("positional", "opaque", "rotated")

# Opaque keys, taken in order. Long enough for the six-question batches this
# experiment never sends, so the list does not have to be extended later.
_OPAQUE_KEYS: tuple[str, ...] = ("zzz", "zzy", "zzx", "zzw", "zzv", "zzu", "zzt", "zzs")

# The 3SAT choice described in the module docstring.
SAT_CHOICE_KEY = "formula_class"
SAT_CHOICE_QUESTION = (
    "Classify the formula: is there an assignment of true or false to each "
    "variable under which every clause contains at least one true literal?"
)
SAT_CHOICE_OPTIONS = [
    {"id": "satisfiable", "label": "Satisfiable -- some assignment makes every clause true."},
    {
        "id": "unsatisfiable",
        "label": "Unsatisfiable -- no assignment makes every clause true.",
    },
]

# Difficulty ladder, ordered easy to hard, where "hard" means the model's answer
# is expected to be least certain rather than least accurate. The semantic
# control is the easy end by construction. Within 3SAT the ends of the ratio
# range are the confident cells -- almost everything is satisfiable at 2.0 and
# almost nothing at 8.0 -- and the phase transition near 4.5 is where a formula
# genuinely could go either way, so that is where any sampling variance should
# show up first.
STRATA: tuple[dict, ...] = (
    {
        "label": "semantic/clean",
        "generator": "semantic",
        "gen_difficulty": {"level": "clean", "question_type": "all"},
        "difficulty": {"domain": "semantic", "level": "clean"},
    },
    {
        "label": "semantic/hard",
        "generator": "semantic",
        "gen_difficulty": {"level": "hard", "question_type": "all"},
        "difficulty": {"domain": "semantic", "level": "hard"},
    },
    {
        "label": "sat3/ratio2.0",
        "generator": "sat3",
        "gen_difficulty": {"n": 20, "ratio": 2.0},
        "difficulty": {"domain": "sat3", "n": 20, "ratio": 2.0},
    },
    {
        "label": "sat3/ratio8.0",
        "generator": "sat3",
        "gen_difficulty": {"n": 20, "ratio": 8.0},
        "difficulty": {"domain": "sat3", "n": 20, "ratio": 8.0},
    },
    {
        "label": "sat3/ratio4.5",
        "generator": "sat3",
        "gen_difficulty": {"n": 20, "ratio": 4.5},
        "difficulty": {"domain": "sat3", "n": 20, "ratio": 4.5},
    },
)

# States are handed out round-robin in this order rather than in difficulty
# order, so that a dry run with two states still draws one from each generator
# instead of two semantic ones.
_ALLOCATION_ORDER: tuple[int, ...] = (0, 2, 1, 4, 3)

# Below this many repetitions a per-state standard deviation is a description of
# two or three numbers rather than an estimate, and the report has to say so.
MIN_REPS_FOR_SIGMA = 5

# A cell -- one (state, question) -- needs this many successful repetitions
# before any within-condition agreement number is computed from it. One
# repetition agrees with itself, is bit-identical to itself and disagrees with
# nothing, so a cell with one answer reports perfect determinism from no
# evidence. That is not a hypothetical: a stratum reduced to a single surviving
# repetition by failed calls was graded Perfect before this bar existed.
MIN_REPS_FOR_AGREEMENT = 2

# The gate from the plan's Phase 1: above this, every downstream comparison is
# inside the noise band.
GATE_SIGMA = 0.15

# The plan's own falsification thresholds for P2 and P3, applied to the
# magnitude of the shift rather than its signed value. "Key renaming has no
# effect" is falsified by an effect in either direction, and "a small effect,
# 1-3%" is not satisfied by a large negative one.
P2_THRESHOLD = 0.02
P3_THRESHOLD = 0.05
# A shift metric whose finest resolvable rate is coarser than the threshold it
# is tested against reports the sample size, not the model. Each prediction is
# marked untestable below its own resolution.
P3_RESOLUTION = 0.01

# A position-bias claim is only reported as an anomaly above this many observed
# choices. The Wilson interval already widens at small n, but a dry run would
# otherwise put "the model prefers position 3" in the report on the strength of
# six answers.
MIN_CHOICES_FOR_POSITION_CLAIM = 30


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _rng(*parts: Any) -> random.Random:
    """A Random determined by its arguments.

    Derived from a digest rather than from `hash()`, whose string hashing is
    salted per process, so a presentation drawn here is reproducible from the
    seed and the log alone.
    """
    payload = "|".join(str(p) for p in (config.MASTER_SEED, EXPERIMENT, *parts))
    digest = hashlib.sha256(payload.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _mode(values: Sequence[Any]) -> Any:
    """The most common value, ties broken by the value's string form.

    Deterministic tie-breaking matters at small repetition counts, where a
    2-2 split is common and an arbitrary winner would make the modal answer --
    and therefore every shift metric measured against it -- depend on dict
    iteration order.
    """
    if not values:
        return None
    counts = Counter(values)
    return min(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0]


def _pstdev(xs: Sequence[float]) -> float | None:
    """Population standard deviation of the observed repetitions, or None.

    Population rather than sample: this describes the spread of the repetitions
    that were actually run, which is what the noise floor is.
    """
    return statistics.pstdev(xs) if len(xs) >= 2 else None


def _mean(xs: Iterable[float]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _allocate(total: int, buckets: int) -> list[int]:
    base, rem = divmod(total, buckets)
    return [base + (1 if i < rem else 0) for i in range(buckets)]


def _discrete(answer: dict) -> Any:
    """The answer as the thing a caller would act on: a label, not a number."""
    if answer["type"] == "noul":
        return bool(answer["predicted"])
    return answer.get("chosen")


def _probability_vector(answer: dict) -> dict[str, float] | None:
    """The returned distribution, keyed by option or rubric id."""
    if answer["type"] == "noul":
        return {"true": answer["p"], "false": 1.0 - answer["p"]}
    probs = answer.get("probabilities")
    return dict(probs) if probs else None


def _scalar(answer: dict) -> float | None:
    """The one continuous number each answer type carries."""
    if answer["type"] == "noul":
        return answer["p"]
    if answer["type"] == "score":
        return answer["score"]
    return answer.get("max_probability")


def _total_variation(a: dict[str, float] | None, b: dict[str, float] | None) -> float | None:
    if not a or not b:
        return None
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


# --------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------


def _sat_with_choice(inst: Instance) -> Instance:
    """A 3SAT instance with the two-option choice added.

    Ground truth is `meta["satisfiable"]`, which is the Minisat22 decision the
    generator already made on this clause list -- not a second derivation that
    could disagree with the noul's truth on the same formula.
    """
    satisfiable = bool(inst.meta["satisfiable"])
    questions = dict(inst.questions)
    questions[SAT_CHOICE_KEY] = choice(SAT_CHOICE_QUESTION, list(SAT_CHOICE_OPTIONS))
    truth = dict(inst.truth)
    truth[SAT_CHOICE_KEY] = "satisfiable" if satisfiable else "unsatisfiable"
    meta = dict(inst.meta)
    # The cheap density heuristic predicts the noul; state its choice-shaped
    # form too, so the baseline can be computed for both questions offline.
    meta["baseline_density_choice"] = (
        "satisfiable" if inst.meta["baseline_density_pred"] else "unsatisfiable"
    )
    meta["e1_added_choice"] = SAT_CHOICE_KEY
    out = replace(inst, questions=questions, truth=truth, meta=meta)
    out.validate()
    return out


def build_states(n_states: int) -> list[tuple[dict, Instance]]:
    """(stratum, instance) pairs, `n_states` of them, spread across the ladder."""
    counts = [0] * len(STRATA)
    for i, k in zip(_ALLOCATION_ORDER, _allocate(n_states, len(STRATA))):
        counts[i] = k

    out: list[tuple[dict, Instance]] = []
    for stratum, count in zip(STRATA, counts):
        if count == 0:
            continue
        seed = config.seed_for(EXPERIMENT, stratum["label"])
        if stratum["generator"] == "semantic":
            instances = semantic.generate(
                difficulty=stratum["gen_difficulty"], seed=seed, count=count
            )
        else:
            instances = [
                _sat_with_choice(inst)
                for inst in sat3.generate(
                    difficulty=stratum["gen_difficulty"], seed=seed, count=count
                )
            ]
        out.extend((stratum, inst) for inst in instances)
    return out


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def _canonical(inst: Instance) -> tuple[list[str], dict[str, list[str]]]:
    """The presentation every condition varies from: one fixed key order and one
    fixed option order per state, drawn once and reused across all repetitions.

    Drawn per state rather than fixed globally so that condition 1 is not itself
    measuring one particular question order, and fixed across repetitions so
    that condition 1 really is the same request bytes every time.
    """
    rng = _rng("canonical", inst.instance_id)
    keys = list(inst.questions)
    rng.shuffle(keys)
    option_order = {}
    for key in keys:
        q = inst.questions[key]
        if q["type"] == "choice":
            ids = [o["id"] for o in q["options"]]
            rng.shuffle(ids)
            option_order[key] = ids
    return keys, option_order


def _with_options(q: dict, order: Sequence[str]) -> dict:
    by_id = {o["id"]: o for o in q["options"]}
    out = dict(q)
    out["options"] = [by_id[i] for i in order]
    return out


def _rename(keys: Sequence[str], scheme: str) -> dict[str, str]:
    """presented key -> canonical key, for one renaming scheme."""
    if scheme == "positional":
        new = [f"q{i + 1}" for i in range(len(keys))]
    elif scheme == "opaque":
        if len(keys) > len(_OPAQUE_KEYS):
            raise ValueError(f"no opaque key for {len(keys)} questions")
        new = list(_OPAQUE_KEYS[: len(keys)])
    elif scheme == "rotated":
        new = [keys[(i + 1) % len(keys)] for i in range(len(keys))]
    else:
        raise ValueError(f"unknown rename scheme {scheme!r}")
    return dict(zip(new, keys))


def present(inst: Instance, condition: str, rep: int) -> tuple[dict[str, dict], dict]:
    """The questions as sent for one (state, condition, repetition), plus the
    presentation metadata that makes the call rescorable from the log alone."""
    keys, option_order = _canonical(inst)
    key_map = {k: k for k in keys}
    scheme = None

    if condition == "question_order":
        keys = list(keys)
        _rng("question_order", inst.instance_id, rep).shuffle(keys)
        key_map = {k: k for k in keys}
    elif condition == "key_rename":
        scheme = RENAME_SCHEMES[rep % len(RENAME_SCHEMES)]
        key_map = _rename(keys, scheme)
        keys = list(key_map)
    elif condition == "option_order":
        rng = _rng("option_order", inst.instance_id, rep)
        option_order = {k: list(v) for k, v in option_order.items()}
        for ids in option_order.values():
            rng.shuffle(ids)
    elif condition != REFERENCE_CONDITION:
        raise ValueError(f"unknown condition {condition!r}")

    questions: dict[str, dict] = {}
    for presented_key in keys:
        canonical_key = key_map[presented_key]
        q = inst.questions[canonical_key]
        if canonical_key in option_order:
            q = _with_options(q, option_order[canonical_key])
        questions[presented_key] = q

    presentation = {
        "condition": condition,
        "repetition": rep,
        "key_map": key_map,
        "question_order": [key_map[k] for k in keys],
        "option_order": option_order,
        "rename_scheme": scheme,
    }
    return questions, presentation


def build_calls(states: Sequence[tuple[dict, Instance]], n_reps: int) -> list[Call]:
    """Every call this experiment makes, interleaved across conditions.

    One shuffled list rather than four blocks: see the module docstring. The
    shuffle is seeded, so the submission order is reproducible from the log.
    """
    calls: list[Call] = []
    for stratum, inst in states:
        for condition in CONDITIONS:
            for rep in range(n_reps):
                questions, presentation = present(inst, condition, rep)
                calls.append(
                    Call(
                        experiment=EXPERIMENT,
                        condition=condition,
                        instance_id=inst.instance_id,
                        repetition=rep,
                        state=inst.state,
                        questions=questions,
                        meta={
                            "truth": inst.truth,
                            "difficulty": inst.difficulty,
                            "generator": inst.generator,
                            "stratum": stratum["label"],
                            "stratum_difficulty": stratum["difficulty"],
                            "presentation": presentation,
                            "instance_meta": inst.meta,
                        },
                    )
                )
    _rng("submission-order").shuffle(calls)
    return calls


# --------------------------------------------------------------------------
# Collecting answers
# --------------------------------------------------------------------------


class Collected:
    """Successful answers, indexed the way every metric below wants to read them.

    `by_cell[(condition, instance_id, canonical_key)]` is the list of
    repetitions, each carrying the neutral answer and where the chosen option
    sat in the order it was presented in.
    """

    def __init__(self) -> None:
        self.by_cell: dict[tuple[str, str, str], list[dict]] = {}
        self.stratum_of: dict[str, str] = {}
        self.qtype_of: dict[tuple[str, str], str] = {}
        self.truth_of: dict[tuple[str, str], Any] = {}
        self.instance_meta: dict[str, dict] = {}
        self.failures = 0
        self.failure_reasons: Counter = Counter()
        self.lost_observations = 0
        self.latencies: list[float] = []
        self.model_versions: set[str] = set()

    def add(self, result: CallResult) -> None:
        call = result.call
        if not result.ok:
            self.failures += 1
            reason = result.error or f"HTTP {result.http_status}"
            self.failure_reasons[reason.split(":")[0][:60]] += 1
            self.lost_observations += len(call.questions)
            return
        self.latencies.append(result.latency_s)
        if result.model_version:
            self.model_versions.add(result.model_version)
        meta = call.meta
        key_map = meta["presentation"]["key_map"]
        option_order = meta["presentation"]["option_order"]
        self.stratum_of[call.instance_id] = meta["stratum"]
        self.instance_meta[call.instance_id] = meta["instance_meta"]
        for presented_key, answer in result.answers.items():
            canonical = key_map[presented_key]
            self.qtype_of[(call.instance_id, canonical)] = answer["type"]
            self.truth_of[(call.instance_id, canonical)] = meta["truth"].get(canonical)
            position = None
            if answer["type"] == "choice" and answer.get("chosen") is not None:
                order = option_order.get(canonical) or []
                if answer["chosen"] in order:
                    position = order.index(answer["chosen"])
            cell = self.by_cell.setdefault((call.condition, call.instance_id, canonical), [])
            cell.append(
                {
                    "rep": call.repetition,
                    "answer": answer,
                    "position": position,
                    "n_options": len(option_order.get(canonical) or []) or None,
                    "rename_scheme": meta["presentation"]["rename_scheme"],
                }
            )

    def cells(self, condition: str) -> dict[tuple[str, str], list[dict]]:
        return {
            (iid, key): reps
            for (cond, iid, key), reps in self.by_cell.items()
            if cond == condition
        }


# --------------------------------------------------------------------------
# Per-condition metrics
# --------------------------------------------------------------------------


def condition_metrics(
    data: Collected, condition: str, instance_ids: Sequence[str] | None = None
) -> dict:
    """Variance within one condition, over one set of states.

    Everything here is a within-condition number. Nothing is compared against
    another condition, so a large sigma here is the model's own variance under
    that presentation and not an effect of the presentation change.
    """
    wanted = None if instance_ids is None else set(instance_ids)
    cells = {
        k: v
        for k, v in data.cells(condition).items()
        if wanted is None or k[0] in wanted
    }

    noul_sigmas: list[float] = []
    noul_spreads: list[float] = []
    score_sigmas: list[float] = []
    agreements: list[float] = []
    exact_choice = 0
    choice_cells = 0
    sampling_agreement: list[float] = []
    # Kept per option count: an 8-option choice and a 2-option choice have
    # different uniform references, so pooling their positions would compare
    # each against the wrong one.
    positions_by_k: dict[int, Counter] = {}
    # None until a cell with at least two repetitions is seen: one answer is
    # trivially identical to itself, and reporting that as determinism is how a
    # stratum left with a single surviving call gets graded Perfect.
    bit_identical: bool | None = None
    observations = 0
    noul_values: list[float] = []
    thin_cells = 0

    for (_iid, _key), reps in cells.items():
        answers = [r["answer"] for r in reps]
        observations += len(answers)
        qtype = answers[0]["type"]
        discrete = [_discrete(a) for a in answers]
        comparable = len(answers) >= MIN_REPS_FOR_AGREEMENT
        if not comparable:
            thin_cells += 1
        if comparable:
            if bit_identical is None:
                bit_identical = True
            if len(set(map(str, discrete))) > 1:
                bit_identical = False

        if qtype == "noul":
            ps = [a["p"] for a in answers]
            noul_values.extend(ps)
            if comparable and len(set(ps)) > 1:
                bit_identical = False
            sigma = _pstdev(ps)
            if sigma is not None:
                noul_sigmas.append(sigma)
                noul_spreads.append(max(ps) - min(ps))
        elif qtype == "choice":
            # Position counts come from every answer; only the agreement
            # numbers need two repetitions to mean anything.
            for r in reps:
                if r["position"] is not None and r["n_options"]:
                    positions_by_k.setdefault(r["n_options"], Counter())[r["position"]] += 1
            # What agreement would look like if the model drew its answer from
            # the distribution it returned. Nothing says it does, but if the
            # observed agreement matches this, the variation is consistent with
            # sampling rather than with an unstable encoder.
            for a in answers:
                probs = a.get("probabilities") or {}
                if probs:
                    sampling_agreement.append(sum(p * p for p in probs.values()))
            if not comparable:
                continue
            choice_cells += 1
            modal = _mode(discrete)
            agreement = sum(1 for d in discrete if d == modal) / len(discrete)
            agreements.append(agreement)
            exact_choice += int(agreement == 1.0)
        elif qtype == "score":
            values = [a["score"] for a in answers]
            if comparable and len(set(values)) > 1:
                bit_identical = False
            sigma = _pstdev(values)
            if sigma is not None:
                score_sigmas.append(sigma)

    n_states = len({iid for iid, _ in cells})
    out: dict[str, Any] = {
        "condition": condition,
        "prob_sigma": _mean(noul_sigmas),
        "prob_spread_mean": _mean(noul_spreads),
        "prob_spread_max": max(noul_spreads) if noul_spreads else None,
        "choice_agreement": _mean(agreements),
        "choice_exact_match_rate": (exact_choice / choice_cells) if choice_cells else None,
        "choice_agreement_if_sampling": _mean(sampling_agreement),
        "score_sigma": _mean(score_sigmas),
        "bit_identical": bit_identical,
        "n_states": n_states,
        "n_noul_states": len(noul_sigmas),
        "n_choice_states": choice_cells,
        # Cells that came back with a single successful repetition and so
        # contribute observations but no agreement or determinism evidence.
        "single_repetition_cells": thin_cells,
        "n": observations,
        "n_noul": len(noul_values),
        "distinct_noul_values": len(set(noul_values)),
    }
    out["position_bias"] = [
        _position_bias(k, counts, order_randomized=condition == "option_order")
        for k, counts in sorted(positions_by_k.items())
    ]
    return out


def _position_bias(n_options: int, positions: Counter, *, order_randomized: bool) -> dict:
    """How often the chosen option sat at each position in the presented order.

    Only evidence of position bias where option order was shuffled per
    repetition. In the other three conditions the order is fixed per state, so
    position is confounded with which option happens to be correct, and the
    counts are recorded without being tested against uniform.

    The test is the Wilson interval on the most-chosen position: if it excludes
    1/k, position is doing something the options themselves do not explain. The
    interval is Bonferroni-widened to 1 - 0.05/k, because the position under
    test was chosen for being the largest of k. At 8 options -- the semantic
    control's choice -- a model with no position preference whatever trips a
    plain 95% interval on 23% of simulated runs, which would put "the model
    prefers position 5" in the report roughly one run in four. The widened
    interval holds that near 2.5%.
    """
    total = sum(positions.values())
    uniform = 1.0 / n_options
    out: dict[str, Any] = {
        "n_options": n_options,
        "n": total,
        "counts": dict(sorted(positions.items())),
        "uniform_rate": uniform,
        "order_randomized": order_randomized,
    }
    if not order_randomized:
        out["note"] = (
            "option order is fixed per state in this condition, so position is "
            "confounded with option identity and is not tested against uniform"
        )
        return out
    top_pos, top_n = max(positions.items(), key=lambda kv: (kv[1], -kv[0]))
    conf = 1.0 - 0.05 / n_options
    lo, hi = metrics.wilson_interval(top_n, total, conf)
    out.update(
        {
            "max_position": top_pos,
            "max_rate": top_n / total,
            "max_rate_ci": [lo, hi],
            "ci_confidence": conf,
            "ci_note": (
                f"{conf * 100:.2f}% interval: 95% Bonferroni-corrected for the "
                f"{n_options} positions the maximum was selected from"
            ),
            "excludes_uniform": lo > uniform or hi < uniform,
            "max_deviation": max(abs(c / total - uniform) for c in positions.values()),
        }
    )
    return out


# --------------------------------------------------------------------------
# Cross-condition shift
# --------------------------------------------------------------------------


def _reference_modes(
    data: Collected, instance_ids: Sequence[str] | None = None
) -> dict[tuple[str, str], Any]:
    """Condition 1's modal answer per (state, question).

    Cells with fewer than `MIN_REPS_FOR_AGREEMENT` successful repetitions are
    left out entirely, which drops them from every shift metric as well. A mode
    drawn from one answer is that answer, and every comparison against it is a
    comparison against a single observation.
    """
    wanted = None if instance_ids is None else set(instance_ids)
    return {
        (iid, key): _mode([_discrete(r["answer"]) for r in reps])
        for (iid, key), reps in data.cells(REFERENCE_CONDITION).items()
        if (wanted is None or iid in wanted) and len(reps) >= MIN_REPS_FOR_AGREEMENT
    }


MARGIN_BANDS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# Gap bands are narrow at the bottom because that is the only region where the
# argmax can move: the probability's own standard deviation is around 0.01-0.02,
# so a top-two gap of that size is where flips live.
GAP_BANDS: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.25, 1.0)


def _top_two_gap(answer: dict) -> float | None:
    """Distance between the best and second-best option.

    This, rather than the winner's probability, is what predicts whether a
    repeated call returns the same discrete answer. A distribution can be flat
    -- a winner at 0.30 -- and still be answered identically every time if the
    runner-up is far below it; a distribution can be sharp and still flip if two
    options are a hundredth apart.

    The distinction is not hypothetical. A semantic-routing instance measured by
    hand returned two different departments across 30 identical calls, 18 to 12,
    while the pooled choice agreement over E1's own 1500 observations was
    1.0000: E1's ten hard states simply contained no near-tie. Reporting the
    pooled number alone would tell a reader the model is deterministic on
    choices, when what is deterministic is the distribution, not the argmax of a
    distribution whose top two are within its own jitter.

    A noul is the two-outcome case of the same quantity: its options are p and
    1 - p, so the gap is |2p - 1|.
    """
    if answer.get("type") == "noul":
        p = answer.get("p")
        return abs(2.0 * float(p) - 1.0) if isinstance(p, (int, float)) else None
    probs = answer.get("probabilities") or {}
    if len(probs) < 2:
        top = answer.get("max_probability")
        return float(top) if isinstance(top, (int, float)) else None
    ordered = sorted((float(v) for v in probs.values()), reverse=True)
    return ordered[0] - ordered[1]


def _decidedness(answer: dict) -> float | None:
    """How committed one answer is, on a common 0-to-1 scale.

    A choice or score answer reports its own max probability. A noul reports
    only a probability, so its distance from 0.5 is doubled to put it on the
    same scale -- 0.5 becomes 0 and 0.0 or 1.0 becomes 1. This is a rescaling,
    not a confidence field; noul answers do not carry one.
    """
    if answer.get("type") == "noul":
        p = answer.get("p")
        return abs(float(p) - 0.5) * 2.0 if isinstance(p, (int, float)) else None
    top = answer.get("max_probability")
    return float(top) if isinstance(top, (int, float)) else None


def agreement_by_margin(data: Collected) -> dict:
    """Repeat-agreement as a function of how decided the model was.

    The difficulty strata answer "is the model stable on hard instances". This
    answers the operationally useful version of the same question: given only
    what the model itself returns, does a confident answer come back the same
    way next time? That matters because the deployment pattern the vendor
    recommends -- act when confident, escalate when not -- assumes it does.

    Measured during development on single instances, agreement was 30/30 on a
    clean semantic state and 18/12 on an ambiguous one, so a single pooled
    agreement number would average a near-deterministic regime together with a
    near-coin-flip one and describe neither. Splitting by the model's own margin
    is free: it reuses the repetitions already collected for condition 1.
    """
    buckets: dict[str, dict] = {}
    for lo, hi in zip(MARGIN_BANDS, MARGIN_BANDS[1:]):
        # The top band is closed so a fully committed answer is counted.
        label = f"{lo:.1f}-{hi:.1f}"
        buckets[label] = {"lo": lo, "hi": hi, "cells": 0, "observations": 0,
                          "agreements": 0, "sigmas": []}

    for (_iid, _key), reps in data.cells(REFERENCE_CONDITION).items():
        if len(reps) < MIN_REPS_FOR_AGREEMENT:
            continue
        margins = [m for m in (_decidedness(r["answer"]) for r in reps) if m is not None]
        if not margins:
            continue
        margin = sum(margins) / len(margins)
        label = None
        for lo, hi in zip(MARGIN_BANDS, MARGIN_BANDS[1:]):
            if lo <= margin < hi or (hi == MARGIN_BANDS[-1] and margin >= lo):
                label = f"{lo:.1f}-{hi:.1f}"
                break
        if label is None:
            continue
        b = buckets[label]
        answers = [_discrete(r["answer"]) for r in reps]
        # Held out for the same reason shift_metrics holds it out: a mode fitted
        # on its own sample agrees with itself by construction.
        flags = _leave_one_out_flags(answers)
        b["cells"] += 1
        b["observations"] += len(flags)
        b["agreements"] += len(flags) - sum(flags)
        probs = [r["answer"].get("p") for r in reps if r["answer"].get("type") == "noul"]
        probs = [p for p in probs if isinstance(p, (int, float))]
        if len(probs) >= MIN_REPS_FOR_SIGMA:
            sd = _pstdev(probs)
            if sd is not None:
                b["sigmas"].append(sd)

    rows = []
    for label, b in buckets.items():
        if not b["observations"]:
            continue
        rows.append({
            "margin_band": label,
            "cells": b["cells"],
            "observations": b["observations"],
            "agreement": b["agreements"] / b["observations"],
            "mean_sigma": (sum(b["sigmas"]) / len(b["sigmas"])) if b["sigmas"] else None,
        })
    spread = None
    if len(rows) >= 2:
        spread = max(r["agreement"] for r in rows) - min(r["agreement"] for r in rows)
    gap = agreement_by_top_two_gap(data)
    return {
        "bands": rows,
        "agreement_spread_across_bands": spread,
        "by_top_two_gap": gap["bands"],
        "narrowest_gap_observed": gap["narrowest_gap_observed"],
        "near_tie_cells": gap["near_tie_cells"],
        "coverage_warning": gap["coverage_warning"],
        "note": (
            "Decidedness is the returned max probability for choice and score, "
            "and |p - 0.5| * 2 for noul, which carries no confidence field. The "
            "top-two-gap split is the one that predicts whether the discrete "
            "answer repeats; see _top_two_gap."
        ),
    }


def agreement_by_top_two_gap(data: Collected) -> dict:
    """Repeat-agreement as a function of the gap between the top two options.

    Separate from the margin split because the two can disagree: a winner at
    0.30 with nothing else above 0.05 repeats perfectly, and a winner at 0.90
    with a runner-up at 0.89 does not.

    `coverage_warning` exists because a pooled agreement of 1.0 is only evidence
    that the model is stable where it was *measured*. If no cell in the sample
    had a top-two gap near the probability's own jitter, then the sample never
    tested the case where flipping is possible, and the report must say so
    rather than letting the reader generalise from it.
    """
    buckets: dict[str, dict] = {}
    for lo, hi in zip(GAP_BANDS, GAP_BANDS[1:]):
        buckets[f"{lo:.2f}-{hi:.2f}"] = {
            "cells": 0, "observations": 0, "agreements": 0
        }
    narrowest: float | None = None
    near_tie = 0

    for (_iid, _key), reps in data.cells(REFERENCE_CONDITION).items():
        if len(reps) < MIN_REPS_FOR_AGREEMENT:
            continue
        gaps = [g for g in (_top_two_gap(r["answer"]) for r in reps) if g is not None]
        if not gaps:
            continue
        gap = sum(gaps) / len(gaps)
        narrowest = gap if narrowest is None else min(narrowest, gap)
        if gap < GAP_BANDS[1]:
            near_tie += 1
        label = None
        for lo, hi in zip(GAP_BANDS, GAP_BANDS[1:]):
            if lo <= gap < hi or (hi == GAP_BANDS[-1] and gap >= lo):
                label = f"{lo:.2f}-{hi:.2f}"
                break
        if label is None:
            continue
        b = buckets[label]
        flags = _leave_one_out_flags([_discrete(r["answer"]) for r in reps])
        b["cells"] += 1
        b["observations"] += len(flags)
        b["agreements"] += len(flags) - sum(flags)

    rows = [
        {
            "gap_band": label,
            "cells": b["cells"],
            "observations": b["observations"],
            "agreement": b["agreements"] / b["observations"],
        }
        for label, b in buckets.items()
        if b["observations"]
    ]
    warning = None
    if near_tie == 0:
        # `narrowest` is None when no cell yielded a gap at all -- which is the
        # every-call-failed case, and is exactly when this branch runs. Formatting
        # it unguarded crashed the whole experiment on the failure mode the plan
        # calls most likely, losing 6000 spent calls and Phase 1's gate with it.
        seen = (
            f"The narrowest gap seen was {narrowest:.3f}."
            if narrowest is not None
            else "No gap could be measured at all, so no cell produced a usable answer."
        )
        warning = (
            f"No cell in this sample had a top-two gap below {GAP_BANDS[1]:.2f}, "
            f"the scale of the probability's own standard deviation. {seen} "
            "Agreement measured here therefore says nothing about near-ties, "
            "which is the case where a repeated call can return a different "
            "option. A hand-probed routing instance with a near-tie returned "
            "two different answers across 30 identical calls, 18 to 12."
        )
    return {
        "bands": rows,
        "narrowest_gap_observed": narrowest,
        "near_tie_cells": near_tie,
        "coverage_warning": warning,
    }


def _leave_one_out_flags(answers: Sequence[Any]) -> list[int]:
    """Per repetition: did it differ from the mode of the *other* repetitions?

    Condition 1 is the only condition scored against a mode fitted on its own
    answers, and a mode is by construction the value its own sample disagrees
    with least often. Comparing it in-sample while comparing every other
    condition out-of-sample gives condition 1 a fitting advantage, and
    `shift_metrics` would report that advantage as a presentation effect.

    The size of it is not academic. With a question the model answers 55/45 --
    which is what the 3SAT phase-transition stratum is chosen to produce -- a
    simulated model with no presentation effect whatsoever showed a mean excess
    of +5.8%, above P3's 5% falsification threshold and nearly three times P2's.
    Holding each repetition out makes both sides of the subtraction predictions
    about an answer that was not used to fit the reference.
    """
    if len(answers) < MIN_REPS_FOR_AGREEMENT:
        return []
    rest = list(answers)
    out: list[int] = []
    for i, a in enumerate(answers):
        out.append(int(a != _mode(rest[:i] + rest[i + 1 :])))
    return out


def _disagreement(
    data: Collected,
    condition: str,
    reference: dict[tuple[str, str], Any],
    *,
    qtypes: set[str] | None = None,
    instance_ids: Sequence[str] | None = None,
    leave_one_out: bool = False,
) -> tuple[int, int, dict[tuple[str, str], list[int]]]:
    """(disagreeing repetitions, total repetitions, per-cell 0/1 lists).

    `leave_one_out` is for condition 1 measured against itself; see
    `_leave_one_out_flags`.
    """
    wanted = None if instance_ids is None else set(instance_ids)
    bad = total = 0
    per_cell: dict[tuple[str, str], list[int]] = {}
    for (iid, key), reps in data.cells(condition).items():
        if wanted is not None and iid not in wanted:
            continue
        if (iid, key) not in reference:
            continue
        if qtypes is not None and data.qtype_of.get((iid, key)) not in qtypes:
            continue
        # By repetition index, not by arrival order. The McNemar pairing below
        # is positional, and arrival order is thread-completion order -- so
        # without this the p-value differs between the live run and a rescore
        # of the same log, and the plan requires every number to come back from
        # the log plus the seed.
        answers = [_discrete(r["answer"]) for r in sorted(reps, key=lambda r: r["rep"])]
        if leave_one_out:
            flags = _leave_one_out_flags(answers)
            if not flags:
                continue
        else:
            ref = reference[(iid, key)]
            flags = [int(a != ref) for a in answers]
        per_cell[(iid, key)] = flags
        bad += sum(flags)
        total += len(flags)
    return bad, total, per_cell


def shift_metrics(
    data: Collected,
    condition: str,
    *,
    qtypes: set[str] | None = None,
    instance_ids: Sequence[str] | None = None,
) -> dict:
    """How much a presentation change moves the answer, net of model variance.

    `excess` is the headline: the condition's repetition-level disagreement with
    condition 1's modal answer, minus condition 1's own disagreement with that
    same mode, the latter measured leave-one-out so that both sides are
    out-of-sample. A presentation change with no effect gives zero; the model's
    own instability cancels. See `_leave_one_out_flags` for why the holdout is
    not a refinement but the thing that makes the subtraction mean anything.

    Cells with fewer than `MIN_REPS_FOR_AGREEMENT` successful repetitions of
    condition 1 are excluded from both sides, since there is no mode to hold a
    repetition out of.
    """
    wanted = None if instance_ids is None else set(instance_ids)
    reference = _reference_modes(data, instance_ids)
    bad_c, n_c, cells_c = _disagreement(
        data, condition, reference, qtypes=qtypes, instance_ids=instance_ids
    )
    bad_r, n_r, cells_r = _disagreement(
        data,
        REFERENCE_CONDITION,
        reference,
        qtypes=qtypes,
        instance_ids=instance_ids,
        leave_one_out=True,
    )
    if n_c == 0 or n_r == 0:
        return {
            "condition": condition,
            "excess": None,
            "n": n_c,
            "note": "no comparable observations",
        }

    rate_c = bad_c / n_c
    rate_r = bad_r / n_r
    lo_c, hi_c = metrics.wilson_interval(bad_c, n_c)
    lo_r, hi_r = metrics.wilson_interval(bad_r, n_r)
    disjoint = lo_c > hi_r or lo_r > hi_c

    # Paired by state and question: McNemar over the cells both conditions
    # cover, comparing whether each repetition matched the reference mode.
    p_value = None
    shared = sorted(set(cells_c) & set(cells_r))
    if shared:
        a: list[int] = []
        b: list[int] = []
        for cell in shared:
            width = min(len(cells_c[cell]), len(cells_r[cell]))
            a.extend(1 - f for f in cells_c[cell][:width])
            b.extend(1 - f for f in cells_r[cell][:width])
        if a:
            p_value = metrics.mcnemar(a, b)

    # The coarse view the plan's prose describes: did the modal answer move?
    modal_moved = modal_total = 0
    for (iid, key), reps in data.cells(condition).items():
        if wanted is not None and iid not in wanted:
            continue
        if (iid, key) not in reference:
            continue
        if qtypes is not None and data.qtype_of.get((iid, key)) not in qtypes:
            continue
        modal_total += 1
        modal_moved += int(_mode([_discrete(r["answer"]) for r in reps]) != reference[(iid, key)])

    # And the continuous view, which a discrete answer hides: how far the whole
    # returned distribution moved.
    tv: list[float] = []
    for (iid, key), reps in data.cells(condition).items():
        if wanted is not None and iid not in wanted:
            continue
        ref_reps = data.by_cell.get((REFERENCE_CONDITION, iid, key))
        if not ref_reps:
            continue
        if qtypes is not None and data.qtype_of.get((iid, key)) not in qtypes:
            continue
        d = _total_variation(
            _mean_distribution([r["answer"] for r in reps]),
            _mean_distribution([r["answer"] for r in ref_reps]),
        )
        if d is not None:
            tv.append(d)

    return {
        "condition": condition,
        "excess": rate_c - rate_r,
        "rate": rate_c,
        "reference_rate": rate_r,
        "rate_ci95": [lo_c, hi_c],
        "reference_rate_ci95": [lo_r, hi_r],
        "distinguishable_from_zero": disjoint,
        "mcnemar_p": p_value,
        "modal_shift": (modal_moved / modal_total) if modal_total else None,
        "modal_n": modal_total,
        "distribution_shift_tv": _mean(tv),
        "n": n_c,
        "reference_n": n_r,
        "resolution": 1.0 / n_c,
    }


def _mean_distribution(answers: Sequence[dict]) -> dict[str, float] | None:
    """The mean returned distribution over repetitions."""
    vectors = [v for v in (_probability_vector(a) for a in answers) if v]
    if not vectors:
        return None
    keys = set().union(*(set(v) for v in vectors))
    return {k: sum(v.get(k, 0.0) for v in vectors) / len(vectors) for k in keys}


def rename_scheme_breakdown(data: Collected) -> dict[str, dict]:
    """key_rename split by scheme.

    Worth separating because the schemes ask different questions. Opaque keys
    test whether the key is read at all; rotated keys -- where each question
    carries the canonical name of a different question about the same state --
    test whether a key that is read is also believed.
    """
    reference = _reference_modes(data)
    out: dict[str, dict] = {}
    for (iid, key), reps in data.cells("key_rename").items():
        if (iid, key) not in reference:
            continue
        ref = reference[(iid, key)]
        for r in reps:
            scheme = r["rename_scheme"] or "unknown"
            slot = out.setdefault(scheme, {"disagreements": 0, "n": 0})
            slot["n"] += 1
            slot["disagreements"] += int(_discrete(r["answer"]) != ref)
    bad_r, n_r, _ = _disagreement(
        data, REFERENCE_CONDITION, reference, leave_one_out=True
    )
    reference_rate = (bad_r / n_r) if n_r else None
    for scheme, slot in out.items():
        slot["rate"] = slot["disagreements"] / slot["n"] if slot["n"] else None
        slot["excess"] = (
            None if slot["rate"] is None or reference_rate is None
            else slot["rate"] - reference_rate
        )
    return out


# --------------------------------------------------------------------------
# Accuracy, with the baselines the rubric requires
# --------------------------------------------------------------------------


def accuracy_with_baselines(data: Collected, instance_ids: Sequence[str] | None = None) -> dict:
    """Accuracy of the modal answer, beside random, majority and cheap baselines.

    E1 is not an accuracy experiment, but the rubric refuses a number without
    its baseline and these come free from ground truth already in `meta`. They
    also serve as a sanity check: a stratum whose accuracy sits at chance while
    its sigma is zero is confidently and stably wrong, which is a different
    finding from noisy.
    """
    wanted = None if instance_ids is None else set(instance_ids)
    per_type: dict[str, dict] = {}
    heuristic_hits = heuristic_total = 0

    for (iid, key), reps in data.cells(REFERENCE_CONDITION).items():
        if wanted is not None and iid not in wanted:
            continue
        truth = data.truth_of.get((iid, key))
        if truth is None:
            continue
        qtype = data.qtype_of[(iid, key)]
        modal = _mode([_discrete(r["answer"]) for r in reps])
        slot = per_type.setdefault(
            qtype, {"correct": 0, "n": 0, "truths": [], "random_baseline": []}
        )
        slot["correct"] += int(modal == truth)
        slot["n"] += 1
        slot["truths"].append(truth)
        probs = reps[0]["answer"].get("probabilities") or {}
        slot["random_baseline"].append(1.0 / len(probs) if probs else 0.5)

        # The cheap deterministic heuristic the rubric asks for, from the
        # features the generators put in meta: clause density on 3SAT, keyword
        # argmax on the semantic control. A keyword argmax that finds no keyword
        # returns None; that counts as a miss rather than being skipped, because
        # a baseline you would actually deploy has to return something.
        meta = data.instance_meta.get(iid) or {}
        applies = False
        guess = None
        if qtype == "choice" and "baseline_density_choice" in meta:
            applies, guess = True, meta["baseline_density_choice"]
        elif qtype == "noul" and "baseline_density_pred" in meta:
            applies, guess = True, bool(meta["baseline_density_pred"])
        elif qtype == "choice" and "baseline_keyword_prediction" in meta:
            applies, guess = True, meta["baseline_keyword_prediction"]
        if applies:
            heuristic_total += 1
            heuristic_hits += int(guess == truth)

    out: dict[str, Any] = {}
    for qtype, slot in per_type.items():
        lo, hi = metrics.wilson_interval(slot["correct"], slot["n"])
        out[qtype] = {
            "accuracy": slot["correct"] / slot["n"],
            "n": slot["n"],
            "wilson95": [lo, hi],
            "random_baseline": _mean(slot["random_baseline"]),
            "majority_baseline": metrics.majority_baseline(slot["truths"]),
        }
    out["cheap_heuristic"] = {
        "accuracy": (heuristic_hits / heuristic_total) if heuristic_total else None,
        "n": heuristic_total,
        "description": "keyword argmax on the semantic control, clause density on 3SAT",
    }
    return out


# --------------------------------------------------------------------------
# Quantization and confidence
# --------------------------------------------------------------------------


def _granularity(values: Sequence[float]) -> dict:
    distinct = sorted(set(values))
    gaps = [b - a for a, b in zip(distinct, distinct[1:]) if b - a > 1e-12]
    out: dict[str, Any] = {
        "n": len(values),
        "distinct": len(distinct),
        "min": distinct[0] if distinct else None,
        "max": distinct[-1] if distinct else None,
        "min_gap": min(gaps) if gaps else None,
    }
    for step, name in ((0.01, "multiples_of_0.01"), (0.001, "multiples_of_0.001")):
        out[name] = all(abs(v / step - round(v / step)) < 1e-6 for v in distinct)
    counts = Counter(values)
    total = len(values) or 1
    out["most_common"] = [
        {"value": v, "count": c, "share": c / total} for v, c in counts.most_common(8)
    ]
    # Small enough to print in full when it is; otherwise the summary above is
    # what the report needs.
    if len(distinct) <= 128:
        out["values"] = distinct
        out["value_counts"] = {v: counts[v] for v in distinct}
    return out


def quantization_report(data: Collected) -> dict:
    """What values the API actually returns, and whether they are on a grid."""
    families: dict[str, list[float]] = {
        "noul_probability": [],
        "choice_probability": [],
        "choice_confidence": [],
        "score_probability": [],
        "score_confidence": [],
        "score_value": [],
    }
    confidence_gaps: list[float] = []
    confidence_exceeds_max = 0
    confidence_n = 0

    for reps in data.by_cell.values():
        for r in reps:
            a = r["answer"]
            if a["type"] == "noul":
                families["noul_probability"].append(a["p"])
                continue
            probs = a.get("probabilities") or {}
            key = "choice_probability" if a["type"] == "choice" else "score_probability"
            families[key].extend(probs.values())
            if a["type"] == "score":
                families["score_value"].append(a["score"])
            conf = a.get("confidence")
            if conf is not None:
                families[
                    "choice_confidence" if a["type"] == "choice" else "score_confidence"
                ].append(conf)
                top = a.get("max_probability")
                if top is not None:
                    confidence_n += 1
                    confidence_gaps.append(conf - top)
                    confidence_exceeds_max += int(conf > top + 1e-9)

    out: dict[str, Any] = {
        family: _granularity(values) for family, values in families.items() if values
    }
    if confidence_n:
        out["confidence_vs_max_probability"] = {
            "n": confidence_n,
            "mean_signed_gap": _mean(confidence_gaps),
            "mean_absolute_gap": _mean(abs(g) for g in confidence_gaps),
            "max_absolute_gap": max(abs(g) for g in confidence_gaps),
            "fraction_differing": sum(1 for g in confidence_gaps if abs(g) > 1e-9)
            / confidence_n,
            "fraction_above_max_probability": confidence_exceeds_max / confidence_n,
        }
    return out


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


def score_predictions(
    reference: dict, shifts: dict[str, dict], n_reps: int, schemes: dict[str, dict]
) -> list[dict]:
    """P1, P2 and P3, each with the evidence that decided it.

    A prediction whose claim has two clauses is scored against both clauses, and
    the outcome text says separately whether the plan's own stated falsifier
    fired -- those are not the same test, and collapsing them would hide the
    cases where the claim failed without being formally falsified.
    """
    out: list[dict] = []
    table = {p.id: p for p in predictions.for_experiment(EXPERIMENT)}

    # -- P1 ---------------------------------------------------------------
    sigma = reference.get("prob_sigma")
    agreement = reference.get("choice_agreement")
    if sigma is None or agreement is None:
        out.append(
            {
                "id": "P1",
                "claim": table["P1"].claim,
                "verdict": "untestable",
                "outcome": (
                    "No probability sigma or choice agreement could be measured: "
                    f"sigma={sigma}, agreement={agreement}. Both need at least two "
                    "successful repetitions of the identical condition."
                ),
                "evidence": {"prob_sigma": sigma, "choice_agreement": agreement},
            }
        )
    else:
        falsifier = sigma > 0.05
        holds = sigma <= 0.02 and agreement >= 0.98
        out.append(
            {
                "id": "P1",
                "claim": table["P1"].claim,
                "verdict": "right" if holds else "wrong",
                "outcome": (
                    f"sigma = {sigma:.4f} on noul probabilities and choice agreement "
                    f"= {agreement:.4f} over {n_reps} repetitions of the identical "
                    f"condition. The claim's own thresholds are sigma <=0.02 and "
                    f"agreement >=0.98, {'both met' if holds else 'not both met'}. "
                    f"The plan's stated falsifier, sigma >0.05, "
                    f"{'fired' if falsifier else 'did not fire'}."
                ),
                "evidence": {
                    "prob_sigma": sigma,
                    "choice_agreement": agreement,
                    "spread_max": reference.get("prob_spread_max"),
                    "bit_identical": reference.get("bit_identical"),
                    "explicit_falsifier_fired": falsifier,
                },
            }
        )

    # -- P2 ---------------------------------------------------------------
    rename = shifts.get("key_rename") or {}
    scheme_text = ", ".join(
        f"{name}={v['excess'] * 100:+.2f}%"
        for name, v in sorted(schemes.items())
        if v.get("excess") is not None
    )
    excess = rename.get("excess")
    if excess is None:
        out.append(
            {
                "id": "P2",
                "claim": table["P2"].claim,
                "verdict": "untestable",
                "outcome": "No key_rename observations could be paired with the identical condition.",
                "evidence": rename,
            }
        )
    elif rename.get("resolution", 1.0) > P2_THRESHOLD:
        # The same resolution argument the module docstring makes for P3. A run
        # with 40 paired repetitions resolves 2.5% at best, so "no shift above
        # 2%" would be a statement about the sample size rather than the model.
        out.append(
            {
                "id": "P2",
                "claim": table["P2"].claim,
                "verdict": "untestable",
                "outcome": (
                    f"The measured shift is {excess * 100:+.2f}% but with only "
                    f"{rename['n']} paired repetitions the finest resolvable rate is "
                    f"{rename['resolution'] * 100:.2f}%, coarser than the "
                    f"{P2_THRESHOLD * 100:.0f}% falsification threshold itself. "
                    "Needs a full-scale run."
                ),
                "evidence": {"key_rename": rename, "by_scheme": schemes},
            }
        )
    else:
        # Magnitude, not signed size: a renaming that moved answers 9% the other
        # way is a key effect too, and "no effect" is a claim about zero.
        moved = abs(excess) > P2_THRESHOLD
        out.append(
            {
                "id": "P2",
                "claim": table["P2"].claim,
                "verdict": "wrong" if moved else "right",
                "outcome": (
                    f"Renaming the question keys moved {excess * 100:+.2f}% of answers "
                    f"net of the model's own variance ({rename['rate'] * 100:.2f}% of "
                    f"repetitions disagreed with the identical condition's modal answer, "
                    f"against {rename['reference_rate'] * 100:.2f}% within the identical "
                    f"condition itself measured leave-one-out, n={rename['n']}). "
                    f"{'Distinguishable' if rename.get('distinguishable_from_zero') else 'Not distinguishable'}"
                    f" from zero by non-overlapping Wilson intervals"
                    + (
                        f"; McNemar p={rename['mcnemar_p']:.3g}"
                        if rename.get("mcnemar_p") is not None
                        else ""
                    )
                    + f". Falsification threshold is a shift of more than "
                    f"{P2_THRESHOLD * 100:.0f}% in either direction"
                    + (
                        ", which this point estimate exceeds although the interval "
                        "includes zero, so the falsification rests on a number the "
                        "data cannot separate from no effect"
                        if moved and not rename.get("distinguishable_from_zero")
                        else ""
                    )
                    + "."
                    + (
                        " The shift is negative: renaming the keys made the answers "
                        "*more* stable than leaving them alone, which no account of "
                        "key handling predicts and is more likely a measurement "
                        "artefact than a finding."
                        if excess < -P2_THRESHOLD
                        else ""
                    )
                    + (f" By scheme: {scheme_text}." if scheme_text else "")
                ),
                "evidence": {"key_rename": rename, "by_scheme": schemes},
            }
        )

    # -- P3 ---------------------------------------------------------------
    order = shifts.get("option_order") or {}
    effect = order.get("excess")
    if effect is None:
        out.append(
            {
                "id": "P3",
                "claim": table["P3"].claim,
                "verdict": "untestable",
                "outcome": "No option_order observations could be paired with the identical condition.",
                "evidence": order,
            }
        )
    elif order.get("resolution", 1.0) > P3_RESOLUTION:
        out.append(
            {
                "id": "P3",
                "claim": table["P3"].claim,
                "verdict": "untestable",
                "outcome": (
                    f"The measured effect is {effect * 100:+.2f}% but with only "
                    f"{order['n']} paired repetitions the finest resolvable rate is "
                    f"{order['resolution'] * 100:.2f}%, which cannot separate the "
                    "predicted 1-3% band from zero. Needs a full-scale run."
                ),
                "evidence": order,
            }
        )
    else:
        zero = not order.get("distinguishable_from_zero")
        # Magnitude again: an effect of -9% is as far from "small, 1-3%" as
        # +9%, and reading only the signed value scores it as a hit.
        too_big = abs(effect) > P3_THRESHOLD
        out.append(
            {
                "id": "P3",
                "claim": table["P3"].claim,
                "verdict": "wrong" if (zero or too_big) else "right",
                "outcome": (
                    f"Shuffling choice option order changed the chosen option on "
                    f"{effect * 100:+.2f}% of repetitions net of the model's own "
                    f"variance ({order['rate'] * 100:.2f}% against "
                    f"{order['reference_rate'] * 100:.2f}% measured leave-one-out, "
                    f"n={order['n']}). "
                    + (
                        "The effect is not distinguishable from zero by "
                        "non-overlapping Wilson intervals, and P3 is falsified by a "
                        "zero effect as well as by a large one."
                        if zero
                        else "The effect is distinguishable from zero."
                    )
                    + (
                        f" Its magnitude exceeds the {P3_THRESHOLD * 100:.0f}% "
                        "falsification threshold, so every choice-based experiment "
                        "downstream needs option order randomised to stay valid."
                        if too_big and not zero
                        else ""
                    )
                    + (
                        f" The point estimate is beyond {P3_THRESHOLD * 100:.0f}% but "
                        "the interval includes zero, so that reading rests on a number "
                        "the data cannot separate from no effect."
                        if too_big and zero
                        else ""
                    )
                    + (
                        " The effect is negative: shuffling the options made the "
                        "chosen option *more* stable than leaving them fixed, which "
                        "is not a position effect and points at the measurement "
                        "rather than at the model."
                        if effect < -P3_THRESHOLD
                        else ""
                    )
                ),
                "evidence": order,
            }
        )
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _plots(
    run_dir: Path,
    per_condition: list[dict],
    per_stratum: list[dict],
    quantization: dict,
) -> list[str]:
    out: list[str] = []
    plot_dir = Path(run_dir) / "plots"

    usable = [c for c in per_condition if c.get("prob_sigma") is not None]
    if usable:
        plots.curve(
            [c["condition"] for c in usable],
            {
                "noul probability sigma": [c["prob_sigma"] for c in usable],
                "choice disagreement (1 - agreement)": [
                    1.0 - c["choice_agreement"] if c.get("choice_agreement") is not None
                    else float("nan")
                    for c in usable
                ],
            },
            plot_dir / "e1_variance_by_condition.png",
            "Variation across repetitions, by condition",
            "condition",
            "variation (0 = deterministic)",
            n={c["condition"]: c["n"] for c in usable},
            n_unit="answers per condition",
            baseline=0.0,
            baseline_label="a deterministic system",
            note="conditions 2-4 add one presentation change each to condition 1",
        )
        out.append("plots/e1_variance_by_condition.png")

    strata = [s for s in per_stratum if s["metrics"].get("prob_sigma") is not None]
    if strata:
        plots.curve(
            [s["label"] for s in strata],
            [s["metrics"]["prob_sigma"] for s in strata],
            plot_dir / "e1_sigma_by_difficulty.png",
            "Noul probability sigma by state difficulty",
            "state difficulty, easy to hard",
            "sigma of returned probability",
            n={s["label"]: s["n"] for s in strata},
            n_unit="answers per stratum",
            baseline=0.0,
            baseline_label="a deterministic system",
            series_label="identical condition",
        )
        out.append("plots/e1_sigma_by_difficulty.png")

    spectrum = quantization.get("noul_probability") or {}
    values = spectrum.get("values")
    if values and len(values) > 1:
        counts = spectrum["value_counts"]
        plots.curve(
            values,
            [counts[v] for v in values],
            plot_dir / "e1_probability_spectrum.png",
            "Distinct noul probabilities returned",
            "returned probability",
            "times returned",
            n=spectrum["n"],
            n_unit="noul answers",
            series_label="frequency",
            note=(
                f"{spectrum['distinct']} distinct values, smallest gap "
                f"{spectrum['min_gap']:.4g}"
                if spectrum.get("min_gap")
                else "one distinct value"
            ),
        )
        out.append("plots/e1_probability_spectrum.png")
    return out


# --------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------


def _anomalies(
    quantization: dict,
    per_condition: list[dict],
    shifts: dict[str, dict],
    n_reps: int,
    data: Collected,
) -> list[str]:
    out: list[str] = []

    noul = quantization.get("noul_probability")
    if noul:
        grid = (
            "0.01" if noul["multiples_of_0.01"]
            else "0.001" if noul["multiples_of_0.001"]
            else None
        )
        out.append(
            f"Returned noul probabilities: {noul['distinct']} distinct values over "
            f"{noul['n']} answers, range {noul['min']:.4g}-{noul['max']:.4g}, smallest "
            f"gap between adjacent values "
            + (f"{noul['min_gap']:.4g}" if noul["min_gap"] else "n/a")
            + ". "
            + (
                f"Every value is a multiple of {grid}, so the probability is quantized "
                f"at that granularity."
                if grid
                else "The values do not all lie on a 0.001 grid, so the probability is "
                "not quantized in the way the plan's watch list anticipated."
            )
        )
        top = noul["most_common"][0] if noul["most_common"] else None
        if top and top["share"] >= 0.10:
            out.append(
                f"One probability value, {top['value']:.4g}, accounts for "
                f"{top['share'] * 100:.1f}% of all noul answers. The plan asks "
                "specifically about probabilities clustering rather than spreading."
            )

    conf = quantization.get("confidence_vs_max_probability")
    if conf:
        if conf["fraction_differing"] > 0.0:
            out.append(
                f"`confidence` differs from the maximum returned probability on "
                f"{conf['fraction_differing'] * 100:.1f}% of choice and score answers "
                f"(mean signed gap {conf['mean_signed_gap']:+.4f}, largest absolute gap "
                f"{conf['max_absolute_gap']:.4f}"
                + (
                    f", and it exceeds the maximum probability on "
                    f"{conf['fraction_above_max_probability'] * 100:.1f}% of them"
                    if conf["fraction_above_max_probability"] > 0
                    else ""
                )
                + "). The two are therefore not the same quantity."
            )
        else:
            out.append(
                "`confidence` equalled the maximum returned probability on every "
                f"choice and score answer ({conf['n']} of them), so on this data it "
                "carries no information beyond the distribution."
            )

    for cond in per_condition:
        for bias in cond.get("position_bias") or []:
            if not bias.get("excludes_uniform"):
                continue
            if bias["n"] < MIN_CHOICES_FOR_POSITION_CLAIM:
                continue
            out.append(
                f"In condition {cond['condition']}, across the {bias['n_options']}-option "
                f"choices, the chosen option sat at position {bias['max_position']} on "
                f"{bias['max_rate'] * 100:.1f}% of answers "
                f"({bias['ci_confidence'] * 100:.2f}% CI "
                f"{bias['max_rate_ci'][0] * 100:.1f}-{bias['max_rate_ci'][1] * 100:.1f}%, "
                f"corrected for the {bias['n_options']} positions searched, "
                f"n={bias['n']}), against {bias['uniform_rate'] * 100:.1f}% if position "
                "were irrelevant. That is a position bias rather than an option "
                "preference: option order was reshuffled every repetition."
            )

    thin = {
        c["condition"]: c["single_repetition_cells"]
        for c in per_condition
        if c.get("single_repetition_cells")
    }
    if thin:
        out.append(
            "Some (state, question) cells came back with a single successful "
            f"repetition and so carry no determinism evidence at all: {thin}. They "
            "are counted in n and excluded from every agreement and shift number; "
            "a stratum made entirely of them has no tier."
        )

    for name, shift in shifts.items():
        excess = shift.get("excess")
        if excess is not None and excess < -P2_THRESHOLD:
            out.append(
                f"Condition {name} disagreed with the identical condition's modal "
                f"answer {abs(excess) * 100:.2f}% *less* often than the identical "
                "condition disagreed with it itself. A presentation change that "
                "stabilises the answer is not something any account of the "
                "architecture predicts, and the first thing to suspect is the "
                "measurement rather than the model."
            )
        tv = shift.get("distribution_shift_tv")
        if tv is not None and shift.get("modal_shift") == 0.0 and tv > 0.01:
            out.append(
                f"Condition {name} moved no modal answer at all, but the returned "
                f"distributions still moved by {tv:.4f} in mean total variation. The "
                "presentation change reaches the probabilities without reaching the "
                "decisions."
            )

    if len(data.model_versions) > 1:
        out.append(
            f"More than one model version answered during E1: "
            f"{sorted(data.model_versions)}. The variance measured here includes a "
            "model change and is not a noise floor."
        )

    if n_reps < MIN_REPS_FOR_SIGMA:
        out.append(
            f"Only {n_reps} repetitions per state were run (the plan asks for 30), so "
            "every sigma here describes that many numbers rather than estimating a "
            "distribution. Treat the conditions as a smoke test of the method."
        )
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    n_states = config.n(config.SIZES.determinism_states)
    n_reps = config.n(config.SIZES.determinism_reps)
    states = build_states(n_states)
    calls = build_calls(states, n_reps)

    with JevClient(run_dir=run_dir, log_name="e1.jsonl") as client:
        results = client.run(calls, progress_every=200, label="E1")
        summary = client.summary()

    data = Collected()
    for result in results:
        data.add(result)

    per_condition = [condition_metrics(data, c) for c in CONDITIONS]
    by_condition = {c["condition"]: c for c in per_condition}
    reference = by_condition[REFERENCE_CONDITION]

    shifts = {
        "question_order": shift_metrics(data, "question_order"),
        "key_rename": shift_metrics(data, "key_rename"),
        "option_order": shift_metrics(data, "option_order", qtypes={"choice"}),
    }
    schemes = rename_scheme_breakdown(data)

    # The rubric's own metric names, assembled once and reused for the overall
    # tier and for each stratum.
    def rubric_metrics(cond: dict, sh: dict[str, dict]) -> dict:
        return {
            "prob_sigma": cond.get("prob_sigma"),
            "choice_agreement": cond.get("choice_agreement"),
            "option_order_flip_rate": (sh.get("option_order") or {}).get("excess"),
            "key_rename_shift": (sh.get("key_rename") or {}).get("excess"),
            "question_order_shift": (sh.get("question_order") or {}).get("excess"),
            "bit_identical": cond.get("bit_identical"),
            "n": cond.get("n"),
        }

    # -- tier as a function of difficulty --------------------------------
    per_stratum: list[dict] = []
    for stratum in STRATA:
        ids = [
            inst.instance_id for s, inst in states if s["label"] == stratum["label"]
        ]
        ids = [i for i in ids if i in data.stratum_of]
        if not ids:
            continue
        cond = condition_metrics(data, REFERENCE_CONDITION, ids)
        sh = {
            "question_order": shift_metrics(data, "question_order", instance_ids=ids),
            "key_rename": shift_metrics(data, "key_rename", instance_ids=ids),
            "option_order": shift_metrics(
                data, "option_order", qtypes={"choice"}, instance_ids=ids
            ),
        }
        per_stratum.append(
            {
                "label": stratum["label"],
                "difficulty": stratum["difficulty"],
                "n": cond["n"],
                "n_states": cond["n_states"],
                "metrics": rubric_metrics(cond, sh),
                "condition": cond,
                "shifts": sh,
                "accuracy": accuracy_with_baselines(data, ids),
                # Whether any determinism evidence survived for this stratum at
                # all. `tiers.assign` grades a stratum that supplied nothing as
                # the residual Bad and says so in its reasons, which is honest
                # per row and wrong as a sweep: it would put a Human->Bad
                # boundary crossing at the easiest difficulty in the ladder
                # purely because those calls failed.
                "measured": any(
                    cond.get(k) is not None
                    for k in ("prob_sigma", "choice_agreement", "bit_identical")
                ),
            }
        )

    graded = tiers.tier_by_difficulty(
        EXPERIMENT, [(s["difficulty"], s["metrics"]) for s in per_stratum]
    )
    crossings = tiers.boundary_crossings(
        EXPERIMENT, [g for s, g in zip(per_stratum, graded) if s["measured"]]
    )
    by_difficulty = []
    for stratum, result in zip(per_stratum, graded):
        row = result.to_json()
        row["label"] = stratum["label"]
        row["accuracy"] = stratum["accuracy"]
        row["measured"] = stratum["measured"]
        by_difficulty.append(row)

    # -- per-condition tiers ---------------------------------------------
    condition_rows = []
    for cond in per_condition:
        name = cond["condition"]
        sh = {k: v for k, v in shifts.items() if k == name}
        graded_condition = tiers.assign(
            EXPERIMENT, rubric_metrics(cond, sh), difficulty={"condition": name}
        )
        row = graded_condition.to_json()
        row["condition"] = name
        row["detail"] = cond
        row["shift_vs_identical"] = shifts.get(name)
        condition_rows.append(row)

    overall = tiers.assign(EXPERIMENT, rubric_metrics(reference, shifts))
    quantization = quantization_report(data)

    # -- gate -------------------------------------------------------------
    sigmas = {
        c["condition"]: c["prob_sigma"]
        for c in per_condition
        if c.get("prob_sigma") is not None
    }
    # The gate is about the model's own variance, which is condition 1. A run
    # that measured one of the presentation conditions and not the reference has
    # not established a noise floor, and passing it on whichever sigma happened
    # to survive would clear Phase 1 on missing evidence.
    if reference.get("prob_sigma") is None:
        gate = {
            "passed": False,
            "reason": (
                "No probability sigma could be measured on the identical condition, "
                "so the noise floor beneath every downstream comparison is unknown. "
                "The plan gates Phase 1 on this number; without it, nothing after E1 "
                "is interpretable."
                + (
                    f" Other conditions did return a sigma ({sigmas}), but those "
                    "carry a presentation change and are not the model's own "
                    "variance."
                    if sigmas
                    else ""
                )
            ),
            "sigma_by_condition": sigmas,
            "threshold": GATE_SIGMA,
        }
    else:
        worst_condition = max(sigmas, key=lambda k: sigmas[k])
        worst = sigmas[worst_condition]
        caveat = (
            f" Measured over {n_reps} repetitions, fewer than the plan's 30, so this "
            "is a provisional reading."
            if n_reps < MIN_REPS_FOR_SIGMA
            else ""
        )
        gate = {
            "passed": worst <= GATE_SIGMA,
            "reason": (
                f"Worst per-condition sigma is {worst:.4f} ({worst_condition}); the "
                f"identical condition alone is {reference['prob_sigma']:.4f}"
                + f". The gate trips above {GATE_SIGMA}."
                + (
                    " Every downstream comparison is inside the noise band; the plan "
                    "says to stop and redesign around repeated sampling."
                    if worst > GATE_SIGMA
                    else " Differences larger than roughly twice this are outside the "
                    "noise floor and may be read as findings."
                )
                + caveat
            ),
            "sigma_by_condition": sigmas,
            "threshold": GATE_SIGMA,
        }

    result = {
        "experiment": EXPERIMENT,
        "title": "Determinism and the noise floor",
        "question": tiers.rubric(EXPERIMENT).question,
        "headline": {
            "metric": "prob_sigma (identical condition)",
            "value": reference.get("prob_sigma"),
            "baseline": 0.0,
            "baseline_name": "a deterministic system, sigma",
            # The noul answers the sigma is computed from, not every answer in
            # the condition: choice and score answers are not in this number.
            "n": reference.get("n_noul", 0),
        },
        "by_difficulty": by_difficulty,
        "boundary_crossings": tiers.crossings_json(crossings),
        "plots": _plots(run_dir, per_condition, per_stratum, quantization),
        "predictions": score_predictions(reference, shifts, n_reps, schemes),
        "anomalies": _anomalies(quantization, per_condition, shifts, n_reps, data),
        "failures": {
            "calls": data.failures,
            "excluded": data.lost_observations,
            "reasons": dict(data.failure_reasons),
        },
        "gate": gate,
        # Everything past here is E1's own detail rather than the shared contract.
        "conditions": condition_rows,
        "overall_tier": overall.to_json(),
        "rename_schemes": schemes,
        "quantization": quantization,
        # Agreement conditioned on the model's own margin, not on the instance's
        # designed difficulty. The two answer different questions and the report
        # needs both.
        "agreement_by_margin": agreement_by_margin(data),
        "accuracy": accuracy_with_baselines(data),
        "sizes": {
            "states": n_states,
            "repetitions": n_reps,
            "conditions": len(CONDITIONS),
            "calls_planned": len(calls),
            "scale": config.SCALE,
            "plan_states": config.SIZES.determinism_states,
            "plan_repetitions": config.SIZES.determinism_reps,
        },
        "run": summary,
    }
    return result


def format_report(result: dict) -> str:
    lines = [f"E1 -- {result['title']}"]
    h = result["headline"]
    value = h["value"]
    lines.append(
        f"  headline: {h['metric']} = "
        + ("n/a" if value is None else f"{value:.4f}")
        + f" vs {h['baseline']:.4f} ({h['baseline_name']}), n={h['n']}"
    )
    lines.append("  per condition:")
    for row in result["conditions"]:
        d = row["detail"]
        sigma = d.get("prob_sigma")
        agree = d.get("choice_agreement")
        shift = (row.get("shift_vs_identical") or {}).get("excess")
        lines.append(
            f"    {d['condition']:<15} sigma="
            + ("n/a   " if sigma is None else f"{sigma:.4f}")
            + "  choice agreement="
            + ("n/a  " if agree is None else f"{agree:.4f}")
            + ("" if shift is None else f"  shift vs identical={shift * 100:+.2f}%")
            + f"  tier={row['tier']}"
        )
    lines.append("  tier by difficulty:")
    for row in result["by_difficulty"]:
        m = row["metrics"]
        sigma = m.get("prob_sigma")
        lines.append(
            f"    {row['label']:<16} n={row['n']:<5} sigma="
            + ("n/a" if sigma is None else f"{sigma:.4f}")
            + (
                "  tier=UNMEASURED (no repetition survived; the tier below is the "
                f"rubric's residual, not a reading: {row['tier']})"
                if not row.get("measured", True)
                else f"  tier={row['tier']}"
            )
        )
    for boundary, where in (result.get("boundary_crossings") or {}).items():
        lines.append(f"    boundary {boundary} crossed at {where}")
    lines.append("  predictions:")
    for p in result["predictions"]:
        lines.append(f"    {p['id']}: {p['verdict'].upper()} -- {p['outcome']}")
    f = result["failures"]
    lines.append(
        f"  failures: {f['calls']} calls failed, {f['excluded']} answers excluded "
        f"{f['reasons'] or ''}"
    )
    gate = result["gate"]
    lines.append(f"  gate: {'passed' if gate['passed'] else 'TRIPPED'} -- {gate['reason']}")
    for a in result["anomalies"]:
        lines.append(f"  anomaly: {a}")
    return "\n".join(lines)
