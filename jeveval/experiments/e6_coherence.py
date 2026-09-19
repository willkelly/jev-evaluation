"""E6 -- cross-call coherence.

Enrollment makes answers coherent inside one call. Across separate calls there
is no shared state and no memory, so nothing enforces that three pairwise
comparisons compose into an order, that P(A) times P(B|A) equals P(A and B), or
that P(X) and P(not X) sum to one. This module measures how far apart those
separately-obtained answers drift.

Five conditions, each a different kind of incoherence:

  transitivity        a > b, b > c and a > c as three SEPARATE calls; count the
                      triples whose three answers form a cycle. Split by whether
                      the items are close or far apart in the true order.
  enrolled triples    the same triples, one call, the 6 orderings as a choice.
                      Paired on identical material, so the before/after is a
                      within-triple comparison rather than a matched sample.
  product rule        P(A), P(B), P(B|A) and P(A and B) as four separate calls,
                      checked against P(A)P(B|A) = P(A and B).
  temporal stability  the same queries replayed hours and days later, to catch a
                      silent model update during early access.
  negation symmetry   "is X true" and "is X false" as separate calls.

Three design decisions are worth stating because they are easy to get wrong.

*The separate calls really are separate.* The three pairwise questions of a
triple are three requests, and their issue order is shuffled across the whole
condition so a triple's three calls are not adjacent in the stream. Batching
them would measure within-call coherence, which is E5's subject, and would make
the cycle rate meaninglessly low.

*A cycle cannot occur in the enrolled form.* A choice over the 6 total orders
has no cyclic option, so the enrolled violation rate is zero by construction
rather than by measurement. P17 is therefore scored "untestable as specified"
and the substantive comparison -- does one enrolled call recover the true order
more often than three separate calls do -- is reported in its place, paired and
with a McNemar p-value.

*Accuracy is kept out of the tier metrics.* The grading keys are
`pairwise_accuracy` and `pairwise_chance_baseline` rather than `accuracy` and
`chance_baseline`, so `tiers.GLOBAL_DISQUALIFIERS` does not read them. E6 grades
coherence. A model that answered every comparison by alphabetical order would be
perfectly transitive and exactly at chance, and feeding its accuracy to the
global "at or below chance is Doesn't work" row would grade a coherent model on
a property this experiment is not measuring. The accuracy numbers and the
lookup-table baseline that beats them are reported in full under
`conditions.transitivity`; they are just not what assigns the tier.

A noul answer carries no confidence field, so wherever a confidence signal is
wanted on a yes/no question this module reports the probability's distance from
0.5 under the name `mean_noul_margin` and says so rather than presenting some
other quantity as the planned one.

Temporal stability spans a day and a single process cannot wait for it, so the
t=0 pass writes its exact queries and answers to `e6_temporal_passes.json` in
the run directory and a later invocation replays them against that baseline:

    JEV_E6_TEMPORAL_ONLY=1 python run.py e6 --run-id <the same run id>

That reruns only the temporal condition, merges the refreshed section into the
existing `e6_result.json`, and leaves the other four conditions' numbers alone.
Run it about an hour after the first pass and again about a day after.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .. import config, metrics, plots, predictions, tiers
from ..client import Call, CallResult, JevClient
from ..generators import ordering, semantic
from ..instances import noul, rng_for

EXPERIMENT = "E6"
LOG_NAME = "e6.jsonl"
TEMPORAL_FILE = "e6_temporal_passes.json"
RESULT_FILE = "e6_result.json"
TEMPORAL_ONLY_ENV = "JEV_E6_TEMPORAL_ONLY"

# Three independent binary answers form a cyclic tournament in 2 of the 8
# equally likely outcomes. This is the null the plan's "Doesn't work" tier is
# defined against, so it is the headline's baseline.
RANDOM_CYCLE_RATE = 0.25

# The planned checkpoints. Only used to label a pass; the elapsed time actually
# measured is what gets reported.
PLANNED_CHECKPOINTS_H = (0.0, 1.0, 24.0)

# Probabilities arrive quantized to two decimals, so a difference of one step is
# not evidence of anything. Used as the tolerance on the conjunction and Frechet
# bound checks, which would otherwise count rounding as incoherence.
QUANT_STEP = 0.01

URGENT_QUESTION = "Does this support ticket need resolution today or sooner?"
CONJUNCTION_QUESTION = (
    "Should this support ticket be handled by the {label} team AND does it need "
    "resolution today or sooner?"
)
NEGATED_ROUTING_QUESTION = (
    "Is it false that this support ticket should be handled by the {label} team?"
)
URGENT_SEVERITIES = ("high", "critical")

TEMPORAL_KINDS = ("ordering_noul", "ordering_choice", "semantic_choice", "semantic_score")


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------
#
# All of these return None rather than 0.0 on an empty input. A metric computed
# over nothing is not zero, and a zero that reaches `tiers.assign` is graded as
# though it had been measured.


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def _wilson(k: int, n: int) -> tuple[float, float] | None:
    return metrics.wilson_interval(k, n) if n else None


def _fmt(v: Any, spec: str = ".4g") -> str:
    return "n/a" if v is None else format(v, spec)


def _tv_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _close_rung() -> dict:
    """The middle close rung of the ordering ladder.

    Chosen by position rather than written out, so a change to the generator's
    ladder moves this with it instead of silently leaving a rung name that no
    longer exists.
    """
    close = [d for d in ordering.difficulty_sweep() if d["distance"] == "close"]
    return dict(close[len(close) // 2])


def _rung_key(d: dict) -> str:
    return f"{d['distance']}-gap{d['min_gap']}to{d['max_gap']}"


# --------------------------------------------------------------------------
# Sample sizes
# --------------------------------------------------------------------------


def _sizes() -> dict:
    """Every count this experiment draws, all through `config.n`.

    The ordering ladder has five rungs and the plan asks for the triples split
    by distance, so the triple budget is divided evenly across the rungs rather
    than spent on one: the closest rung has only 116 distinct triples and asking
    it for the whole budget would report 500 nominally independent triples whose
    effective count was 116.
    """
    sweep = ordering.difficulty_sweep()
    triples_total = config.n(config.SIZES.coherence_triples)
    per_rung = max(1, triples_total // len(sweep))
    units = config.n(config.SIZES.accuracy)
    return {
        "triples_per_rung": per_rung,
        "n_rungs": len(sweep),
        "triples_total": per_rung * len(sweep),
        "product_rule_units_per_level": max(1, units // 2),
        "negation_props_per_domain": max(1, units // 2),
        # Two calls per query per pass, so this is half the accuracy budget.
        "temporal_queries": max(2, units // 2),
        "scale": config.SCALE,
    }


def _planned_calls(sizes: dict) -> dict:
    triples = sizes["triples_total"]
    return {
        "transitivity": 3 * triples,
        "enrolled_triples": triples,
        "product_rule": 4 * 2 * sizes["product_rule_units_per_level"],
        "negation_symmetry": 2 * 2 * sizes["negation_props_per_domain"],
        "temporal_stability_per_pass": 2 * sizes["temporal_queries"],
    }


# --------------------------------------------------------------------------
# Running and collecting calls
# --------------------------------------------------------------------------


def _per_condition_cost(results: list[CallResult]) -> dict:
    """Latency percentiles and cost per decision, per condition.

    The plan asks for both per condition rather than per run, because they vary
    with state size and question count and E6's conditions differ in both: a
    6-way choice over three element names and a conditioned support ticket are
    not the same request.
    """
    buckets: dict[str, list[CallResult]] = {}
    for r in results:
        if r.ok:
            buckets.setdefault(r.call.condition, []).append(r)
    out: dict[str, dict] = {}
    for condition, rs in sorted(buckets.items()):
        lat = metrics.percentiles([r.latency_s for r in rs], (50, 95))
        itok = sum(r.input_tokens for r in rs)
        questions = sum(len(r.call.questions) for r in rs)
        out[condition] = {
            "calls": len(rs),
            "latency_p50_s": lat.get("p50"),
            "latency_p95_s": lat.get("p95"),
            "input_tokens": itok,
            "cost_usd": itok * config.USD_PER_INPUT_TOKEN,
            "cost_per_decision_usd": (
                metrics.cost_per_decision(itok, questions) if questions else None
            ),
        }
    return out


def _key(condition: str, instance_id: str, repetition: int = 0) -> tuple:
    return (condition, instance_id, repetition)


def _run(client: JevClient, calls: list[Call], label: str) -> tuple[dict, dict, list[CallResult]]:
    """Issue a batch and index the results by (condition, instance, repetition).

    A failed call is returned in the index as a CallResult with ok=False; the
    scorers drop the whole data point it belonged to rather than filling it in.
    """
    if not calls:
        return {}, {"calls": 0, "reasons": {}}, []
    results = client.run(calls, progress_every=250, label=f"{EXPERIMENT}/{label}")
    index: dict[tuple, CallResult] = {}
    failures: dict[str, Any] = {"calls": 0, "reasons": {}}
    for r in results:
        index[_key(r.call.condition, r.call.instance_id, r.call.repetition)] = r
        if not r.ok:
            failures["calls"] += 1
            reason = (r.error or f"HTTP {r.http_status}")[:120]
            failures["reasons"][reason] = failures["reasons"].get(reason, 0) + 1
    return index, failures, results


def _answer(index: dict, condition: str, instance_id: str, qkey: str, repetition: int = 0):
    r = index.get(_key(condition, instance_id, repetition))
    if r is None or not r.ok:
        return None
    return r.answers.get(qkey)


# --------------------------------------------------------------------------
# Condition 1 and 2: transitivity, separate calls versus one enrolled call
# --------------------------------------------------------------------------


def _build_ordering(per_rung: int) -> tuple[list[Call], dict[str, dict]]:
    """Four instances per triple: three separate pairwise calls and one enrolled.

    The generator emits them in that order and ties them together with
    `meta["triple_id"]`, so the enrolled call asks about exactly the same three
    elements as the three pairwise calls and the comparison is paired.
    """
    calls: list[Call] = []
    units: dict[str, dict] = {}
    for d in ordering.difficulty_sweep():
        seed = config.seed_for(EXPERIMENT, "ordering-" + _rung_key(d))
        insts = ordering.generate(difficulty=d, seed=seed, count=4 * per_rung)
        pool = insts[0].meta["n_distinct_triples"] if insts else 0
        for inst in insts:
            role = inst.meta["role"]
            tid = inst.meta["triple_id"]
            unit = units.setdefault(
                tid,
                {
                    "triple_id": tid,
                    "rung": _rung_key(d),
                    "difficulty": dict(d),
                    "distance_class": d["distance"],
                    "pool": pool,
                    "instances": {},
                },
            )
            unit["instances"][role] = inst
            calls.append(
                Call(
                    experiment=EXPERIMENT,
                    condition="enrolled-triples" if role == "enrolled" else "transitivity",
                    state=inst.state,
                    questions=inst.questions,
                    instance_id=inst.instance_id,
                    # Truth and the whole generator meta are written into the
                    # call, so the JSONL log alone is enough to recount cycles
                    # offline.
                    meta={"truth": inst.truth, "difficulty": dict(inst.difficulty), **inst.meta},
                )
            )
    return calls, units


def _tournament(edges: list[tuple[str, str]]) -> tuple[dict[str, int], bool]:
    """Win counts and whether the three edges form a cycle.

    Three items with one directed edge per pair: the win counts are [0, 1, 2] in
    some order for a total order, and [1, 1, 1] for a cycle. Nothing else is
    possible, which is why a cycle needs no path search.
    """
    wins: dict[str, int] = {}
    for winner, loser in edges:
        wins[winner] = wins.get(winner, 0) + 1
        wins.setdefault(loser, 0)
    return wins, sorted(wins.values()) == [1, 1, 1]


def _score_triple(unit: dict, index: dict) -> dict:
    """One triple: the three separate answers, the enrolled answer, and the
    coherence of each."""
    out: dict[str, Any] = {
        "triple_id": unit["triple_id"],
        "rung": unit["rung"],
        "distance_class": unit["distance_class"],
        "pairwise_ok": True,
        "enrolled_ok": True,
    }
    edges: list[tuple[str, str]] = []
    probs: list[float] = []
    correct = 0
    asked = 0

    for role in ("ab", "bc", "ac"):
        inst = unit["instances"].get(role)
        if inst is None:
            out["pairwise_ok"] = False
            break
        ans = _answer(index, "transitivity", inst.instance_id, "greater")
        left, right = inst.meta["left"], inst.meta["right"]
        if ans is None:
            out["pairwise_ok"] = False
            continue
        p = ans["p"]
        probs.append(p)
        asked += 1
        correct += int(ans["predicted"] == inst.truth["greater"])
        edges.append((left, right) if ans["predicted"] else (right, left))

    if out["pairwise_ok"] and len(edges) == 3:
        wins, cyclic = _tournament(edges)
        out["cyclic"] = cyclic
        out["pairwise_correct"] = correct
        out["pairwise_asked"] = asked
        out["mean_margin"] = _mean([abs(p - 0.5) for p in probs])
        out["exact_half"] = sum(1 for p in probs if p == 0.5)
        induced = sorted(wins, key=lambda nm: -wins[nm])
        truth_order = unit["instances"]["ab"].meta["correct_ordering"]
        out["separate_order_correct"] = (not cyclic) and induced == truth_order
        out["induced_order"] = induced
    else:
        out["pairwise_ok"] = False

    enrolled = unit["instances"].get("enrolled")
    if enrolled is None:
        out["enrolled_ok"] = False
    else:
        ans = _answer(index, "enrolled-triples", enrolled.instance_id, "ordering")
        if ans is None:
            out["enrolled_ok"] = False
        else:
            out["enrolled_chosen"] = ans["chosen"]
            out["enrolled_correct"] = ans["chosen"] == enrolled.truth["ordering"]
            out["enrolled_confidence"] = ans.get("confidence")
            out["enrolled_max_probability"] = ans.get("max_probability")
    return out


def _aggregate_triples(scored: list[dict]) -> dict:
    usable = [s for s in scored if s["pairwise_ok"]]
    cycles = sum(1 for s in usable if s["cyclic"])
    pw_correct = sum(s["pairwise_correct"] for s in usable)
    pw_asked = sum(s["pairwise_asked"] for s in usable)
    margins = [s["mean_margin"] for s in usable if s["mean_margin"] is not None]
    enrolled = [s for s in scored if s["enrolled_ok"]]
    return {
        "n_triples": len(usable),
        "n_triples_excluded": len(scored) - len(usable),
        "cycles": cycles,
        "cycle_rate": _rate(cycles, len(usable)),
        "cycle_rate_ci95": _wilson(cycles, len(usable)),
        "separate_order_correct_rate": _rate(
            sum(1 for s in usable if s["separate_order_correct"]), len(usable)
        ),
        "pairwise_accuracy": _rate(pw_correct, pw_asked),
        "pairwise_n": pw_asked,
        "pairwise_accuracy_ci95": _wilson(pw_correct, pw_asked),
        # No confidence field exists on a noul answer. This is the probability's
        # distance from 0.5, reported under its own name so it is not mistaken
        # for the confidence the plan asked for.
        "mean_noul_margin": _mean(margins),
        "exact_half_answers": sum(s["exact_half"] for s in usable),
        "n_enrolled": len(enrolled),
        "n_enrolled_excluded": len(scored) - len(enrolled),
        "enrolled_accuracy": _rate(
            sum(1 for s in enrolled if s["enrolled_correct"]), len(enrolled)
        ),
        # Structural, not measured: a choice over the 6 total orders has no
        # cyclic option.
        "enrolled_cycle_rate": 0.0 if enrolled else None,
    }


def _constant_answerer_cycle_rate(units: dict[str, dict]) -> dict[str, float | None]:
    """Cycle rate of a deterministic "always yes" answerer, per rung.

    Computed from the orientations the generator drew, not assumed, because the
    orientations are random per question and the realized rate differs from its
    0.25 expectation by a few points at these sample sizes.
    """
    per_rung: dict[str, list[bool]] = {}
    for unit in units.values():
        edges = []
        ok = True
        for role in ("ab", "bc", "ac"):
            inst = unit["instances"].get(role)
            if inst is None:
                ok = False
                break
            edges.append((inst.meta["left"], inst.meta["right"]))
        if not ok:
            continue
        per_rung.setdefault(unit["rung"], []).append(_tournament(edges)[1])
    return {k: _mean([float(v) for v in vs]) for k, vs in per_rung.items()}


# --------------------------------------------------------------------------
# Condition 3: the product rule across four separate calls
# --------------------------------------------------------------------------


def _build_product_rule(per_level: int) -> tuple[list[Call], list[dict]]:
    """P(A), P(B), P(B|A) and P(A and B) about one support ticket.

    A is "this ticket belongs to team T" and B is "this ticket needs resolution
    today or sooner". The conditional is asked by putting A into the state as a
    confirmed routing decision, which is the only conditioning channel the API
    has -- there is no top-level instructions field and no conversation.

    P(B) on its own is the fourth call, and it is not in the plan. Without it a
    product-rule violation cannot be told apart from the model ignoring the
    conditioning assertion entirely, which would reduce the test to one of
    independence. |P(B|A) - P(B)| is reported as `conditioning_shift`.

    T is the correct team on even indices and a wrong one on odd, so P(A) is
    asked across its whole range rather than only where the answer is yes.
    """
    calls: list[Call] = []
    units: list[dict] = []
    for level in ("clean", "hard"):
        seed = config.seed_for(EXPERIMENT, f"product-{level}")
        for i in range(per_level):
            ticket = semantic.ticket_for(seed=seed, index=i, level=level)
            rng = rng_for(EXPERIMENT, {"condition": "product", "level": level}, seed, i)
            if i % 2 == 0:
                asked = ticket.department
            else:
                asked = rng.choice([d for d in semantic.DEPARTMENTS if d != ticket.department])
            label = semantic.DEPARTMENT_LABELS[asked]
            a_truth = asked == ticket.department
            b_truth = ticket.severity in URGENT_SEVERITIES
            unit_id = f"e6-product-{level}-{i:04d}"
            state = ticket.as_state()
            conditioned = dict(state)
            conditioned["triage"] = {"assigned_team": label, "assignment_confirmed": True}

            parts = [
                ("A", state, noul(semantic.NOUL_QUESTION.format(label=label)), a_truth),
                ("B", state, noul(URGENT_QUESTION), b_truth),
                ("B_given_A", conditioned, noul(URGENT_QUESTION), b_truth),
                ("AB", state, noul(CONJUNCTION_QUESTION.format(label=label)), a_truth and b_truth),
            ]
            unit = {
                "unit_id": unit_id,
                "level": level,
                "asked_department": asked,
                "department": ticket.department,
                "severity": ticket.severity,
                "a_truth": a_truth,
                "b_truth": b_truth,
                "ab_truth": a_truth and b_truth,
                "parts": [p[0] for p in parts],
            }
            units.append(unit)
            for role, st, question, truth in parts:
                calls.append(
                    Call(
                        experiment=EXPERIMENT,
                        condition="product-rule",
                        state=st,
                        questions={"q": question},
                        instance_id=f"{unit_id}-{role}",
                        meta={
                            "truth": {"q": truth},
                            "difficulty": {"level": level},
                            "unit_id": unit_id,
                            "role": role,
                            "asked_department": asked,
                            "department": ticket.department,
                            "severity": ticket.severity,
                            "a_truth": a_truth,
                            "b_truth": b_truth,
                            "ab_truth": a_truth and b_truth,
                        },
                    )
                )
    return calls, units


def _score_product_rule(units: list[dict], index: dict) -> dict:
    rows: list[dict] = []
    excluded = 0
    for unit in units:
        ps = {}
        for role in ("A", "B", "B_given_A", "AB"):
            ans = _answer(index, "product-rule", f"{unit['unit_id']}-{role}", "q")
            if ans is None:
                ps = {}
                break
            ps[role] = ans["p"]
        if not ps:
            excluded += 1
            continue
        pa, pb, pba, pab = ps["A"], ps["B"], ps["B_given_A"], ps["AB"]
        implied = pa * pba
        lo = max(0.0, pa + pb - 1.0)
        hi = min(pa, pb)
        rows.append(
            {
                "unit_id": unit["unit_id"],
                "level": unit["level"],
                "p_a": pa,
                "p_b": pb,
                "p_b_given_a": pba,
                "p_ab": pab,
                "implied_ab": implied,
                "signed_error": implied - pab,
                "abs_error": abs(implied - pab),
                "conditioning_shift": abs(pba - pb),
                # A conjunction cannot be more likely than either conjunct. The
                # quantization step is allowed for so rounding is not counted.
                "conjunction_fallacy": pab > hi + QUANT_STEP,
                "frechet_violation": pab < lo - QUANT_STEP or pab > hi + QUANT_STEP,
            }
        )

    def block(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0, "product_rule_error": None}
        return {
            "n": len(subset),
            "product_rule_error": _mean([r["abs_error"] for r in subset]),
            "product_rule_signed_error": _mean([r["signed_error"] for r in subset]),
            "conditioning_shift": _mean([r["conditioning_shift"] for r in subset]),
            "conjunction_fallacy_rate": _rate(
                sum(1 for r in subset if r["conjunction_fallacy"]), len(subset)
            ),
            "frechet_violation_rate": _rate(
                sum(1 for r in subset if r["frechet_violation"]), len(subset)
            ),
            "mean_p_a": _mean([r["p_a"] for r in subset]),
            "mean_p_b": _mean([r["p_b"] for r in subset]),
            "mean_p_b_given_a": _mean([r["p_b_given_a"] for r in subset]),
            "mean_p_ab": _mean([r["p_ab"] for r in subset]),
        }

    by_level = {lvl: block([r for r in rows if r["level"] == lvl]) for lvl in ("clean", "hard")}
    overall = block(rows)
    overall["n_excluded"] = excluded
    overall["by_level"] = by_level
    overall["permutation_null"] = _product_rule_null(rows)
    # The 20-lines-of-code alternative: never ask for the conjunction, compute
    # it from the two answers you already have. Its error is zero by definition,
    # which is the whole tabling argument in one number.
    overall["tabled_baseline_error"] = 0.0 if rows else None
    return overall


def _product_rule_null(rows: list[dict], n_boot: int = 200) -> float | None:
    """Mean |P(A)P(B|A) - P(A and B)| when the three answers are re-paired at random.

    The floor a genuinely incoherent model would sit at, given the marginal
    distribution of probabilities it actually produced. Without it a
    product-rule error of 0.15 has nothing to be compared against: it might be
    large, or it might be what any three unrelated numbers in this range give.
    """
    if len(rows) < 2:
        return None
    pa = [r["p_a"] for r in rows]
    pba = [r["p_b_given_a"] for r in rows]
    pab = [r["p_ab"] for r in rows]
    rng = random.Random(config.seed_for(EXPERIMENT, "product-null"))
    totals = []
    for _ in range(n_boot):
        i = list(range(len(rows)))
        j = list(range(len(rows)))
        rng.shuffle(i)
        rng.shuffle(j)
        totals.append(_mean([abs(pa[k] * pba[i[k]] - pab[j[k]]) for k in range(len(rows))]))
    return _mean(totals)


# --------------------------------------------------------------------------
# Condition 5: negation symmetry
# --------------------------------------------------------------------------


def _build_negation(per_domain: int) -> tuple[list[Call], list[dict]]:
    """"Is X true" and "is X false" as two separate calls over the same state.

    Two domains, because an asymmetry confined to one would be a fact about that
    wording rather than about the model. The ordering domain reuses the
    generator's own rendered question verbatim for the positive framing, so the
    positive half is exactly the question condition 1 asks.
    """
    calls: list[Call] = []
    props: list[dict] = []

    rung = _close_rung()
    seed = config.seed_for(EXPERIMENT, "negation-ordering")
    insts = [
        inst
        for inst in ordering.generate(difficulty=rung, seed=seed, count=4 * per_domain)
        if inst.meta["role"] == "ab"
    ]
    for inst in insts[:per_domain]:
        left, right, attr = inst.meta["left"], inst.meta["right"], inst.meta["attribute"]
        props.append(
            {
                "prop_id": f"e6-neg-ordering-{inst.index:04d}",
                "domain": "ordering-close",
                "state": inst.state,
                "positive": inst.questions["greater"],
                "negative": noul(f"Is it false that {left} has a higher {attr} than {right}?"),
                "truth": bool(inst.truth["greater"]),
                "meta": {"rank_distance": inst.meta["rank_distance"], "left": left, "right": right},
            }
        )

    seed = config.seed_for(EXPERIMENT, "negation-semantic")
    for i in range(per_domain):
        ticket = semantic.ticket_for(seed=seed, index=i, level="clean")
        rng = rng_for(EXPERIMENT, {"condition": "negation", "domain": "semantic"}, seed, i)
        if i % 2 == 0:
            asked = ticket.department
        else:
            asked = rng.choice([d for d in semantic.DEPARTMENTS if d != ticket.department])
        label = semantic.DEPARTMENT_LABELS[asked]
        props.append(
            {
                "prop_id": f"e6-neg-semantic-{i:04d}",
                "domain": "semantic",
                "state": ticket.as_state(),
                "positive": noul(semantic.NOUL_QUESTION.format(label=label)),
                "negative": noul(NEGATED_ROUTING_QUESTION.format(label=label)),
                "truth": asked == ticket.department,
                "meta": {"asked_department": asked, "department": ticket.department},
            }
        )

    for prop in props:
        for framing, question, truth in (
            ("pos", prop["positive"], prop["truth"]),
            ("neg", prop["negative"], not prop["truth"]),
        ):
            calls.append(
                Call(
                    experiment=EXPERIMENT,
                    condition="negation-symmetry",
                    state=prop["state"],
                    questions={"q": question},
                    instance_id=f"{prop['prop_id']}-{framing}",
                    meta={
                        "truth": {"q": truth},
                        "difficulty": {"domain": prop["domain"]},
                        "prop_id": prop["prop_id"],
                        "framing": framing,
                        "positive_truth": prop["truth"],
                        **prop["meta"],
                    },
                )
            )
    return calls, props


def _score_negation(props: list[dict], index: dict) -> dict:
    rows: list[dict] = []
    excluded = 0
    for prop in props:
        pos = _answer(index, "negation-symmetry", f"{prop['prop_id']}-pos", "q")
        neg = _answer(index, "negation-symmetry", f"{prop['prop_id']}-neg", "q")
        if pos is None or neg is None:
            excluded += 1
            continue
        rows.append(
            {
                "prop_id": prop["prop_id"],
                "domain": prop["domain"],
                "p_true": pos["p"],
                "p_false": neg["p"],
                "sum": pos["p"] + neg["p"],
                "abs_error": abs(pos["p"] + neg["p"] - 1.0),
                "signed_error": pos["p"] + neg["p"] - 1.0,
                "yes_to_both": pos["p"] > 0.5 and neg["p"] > 0.5,
                "no_to_both": pos["p"] < 0.5 and neg["p"] < 0.5,
                "pos_correct": pos["predicted"] == prop["truth"],
                "neg_correct": neg["predicted"] == (not prop["truth"]),
            }
        )

    def block(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0, "negation_sum_error": None}
        return {
            "n": len(subset),
            "negation_sum_error": _mean([r["abs_error"] for r in subset]),
            "negation_signed_error": _mean([r["signed_error"] for r in subset]),
            "yes_to_both_rate": _rate(sum(1 for r in subset if r["yes_to_both"]), len(subset)),
            "no_to_both_rate": _rate(sum(1 for r in subset if r["no_to_both"]), len(subset)),
            "accuracy_positive_framing": _rate(
                sum(1 for r in subset if r["pos_correct"]), len(subset)
            ),
            "accuracy_negated_framing": _rate(
                sum(1 for r in subset if r["neg_correct"]), len(subset)
            ),
        }

    domains = ["ordering-close", "semantic"]
    by_domain = {d: block([r for r in rows if r["domain"] == d]) for d in domains}
    overall = block(rows)
    overall["n_excluded"] = excluded
    overall["by_domain"] = by_domain
    overall["permutation_null"] = _negation_null(rows)
    return overall


def _negation_null(rows: list[dict], n_boot: int = 200) -> float | None:
    """Mean |P(X) + P(not X) - 1| when the two halves are re-paired at random."""
    if len(rows) < 2:
        return None
    pt = [r["p_true"] for r in rows]
    pf = [r["p_false"] for r in rows]
    rng = random.Random(config.seed_for(EXPERIMENT, "negation-null"))
    totals = []
    for _ in range(n_boot):
        j = list(range(len(rows)))
        rng.shuffle(j)
        totals.append(_mean([abs(pt[k] + pf[j[k]] - 1.0) for k in range(len(rows))]))
    return _mean(totals)


# --------------------------------------------------------------------------
# Condition 4: temporal stability, as a resumable checkpoint
# --------------------------------------------------------------------------


def _build_temporal_queries(n_queries: int) -> list[dict]:
    """A fixed query set spanning all three question types.

    Cycled across the four kinds by index rather than sampled, so a run at any
    scale covers every kind in the same order and the set is reproducible from
    the seed alone. Drift confined to one question type is a different diagnosis
    from drift across all of them.
    """
    rung = _close_rung()
    ord_insts = ordering.generate(
        difficulty=rung,
        seed=config.seed_for(EXPERIMENT, "temporal-ordering"),
        count=4 * n_queries,
    )
    ord_noul = [i for i in ord_insts if i.meta["role"] == "ab"]
    ord_choice = [i for i in ord_insts if i.meta["role"] == "enrolled"]
    sem_choice = semantic.generate(
        difficulty={"level": "clean", "question_type": "choice"},
        seed=config.seed_for(EXPERIMENT, "temporal-semantic-choice"),
        count=n_queries,
    )
    sem_score = semantic.generate(
        difficulty={"level": "clean", "question_type": "score"},
        seed=config.seed_for(EXPERIMENT, "temporal-semantic-score"),
        count=n_queries,
    )
    pools = {
        "ordering_noul": ord_noul,
        "ordering_choice": ord_choice,
        "semantic_choice": sem_choice,
        "semantic_score": sem_score,
    }
    counters = {k: 0 for k in pools}

    queries: list[dict] = []
    for i in range(n_queries):
        kind = TEMPORAL_KINDS[i % len(TEMPORAL_KINDS)]
        pool = pools[kind]
        if not pool:
            continue
        inst = pool[counters[kind] % len(pool)]
        counters[kind] += 1
        queries.append(
            {
                "query_id": f"e6-temporal-{i:04d}",
                "kind": kind,
                "state": inst.state,
                "questions": inst.questions,
                "truth": inst.truth,
                "meta": {"source_instance": inst.instance_id, "kind": kind},
            }
        )
    return queries


def _temporal_calls(queries: list[dict], pass_index: int) -> list[Call]:
    """Two calls per query: the measurement and an immediate repeat.

    The repeat is what makes drift interpretable. Without a same-session noise
    floor measured in the same pass, a mean probability change of 0.03 a day
    later cannot be told from the model's own answer-to-answer variation, and
    this condition cannot depend on E1 having been run into the same run directory.
    """
    calls: list[Call] = []
    for q in queries:
        for rep in (0, 1):
            calls.append(
                Call(
                    experiment=EXPERIMENT,
                    condition="temporal-stability",
                    state=q["state"],
                    questions=q["questions"],
                    instance_id=q["query_id"],
                    repetition=rep,
                    meta={
                        "truth": q["truth"],
                        "difficulty": {"pass": pass_index, "kind": q["kind"]},
                        "pass_index": pass_index,
                        "kind": q["kind"],
                        **q["meta"],
                    },
                )
            )
    return calls


def _collect_pass_answers(queries: list[dict], index: dict, rep: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for q in queries:
        r = index.get(_key("temporal-stability", q["query_id"], rep))
        if r is None or not r.ok:
            continue
        out[q["query_id"]] = r.answers
    return out


def _compare_answers(a: dict[str, dict], b: dict[str, dict], kinds: dict[str, str]) -> dict:
    """Drift between two sets of answers to the same queries, split by kind."""
    buckets: dict[str, dict[str, list]] = {}
    for qid, answers_a in a.items():
        answers_b = b.get(qid)
        if answers_b is None:
            continue
        kind = kinds.get(qid, "unknown")
        slot = buckets.setdefault(
            kind, {"noul_delta": [], "noul_flip": [], "choice_change": [], "choice_tv": [],
                   "score_delta": [], "score_change": [], "n": 0}
        )
        slot["n"] += 1
        for qkey, ans_a in answers_a.items():
            ans_b = answers_b.get(qkey)
            if ans_b is None or ans_a.get("type") != ans_b.get("type"):
                continue
            if ans_a["type"] == "noul":
                slot["noul_delta"].append(abs(ans_b["p"] - ans_a["p"]))
                slot["noul_flip"].append(float(ans_a["predicted"] != ans_b["predicted"]))
            elif ans_a["type"] == "choice":
                slot["choice_change"].append(float(ans_a["chosen"] != ans_b["chosen"]))
                slot["choice_tv"].append(
                    _tv_distance(ans_a.get("probabilities") or {}, ans_b.get("probabilities") or {})
                )
            elif ans_a["type"] == "score":
                sa, sb = ans_a.get("score"), ans_b.get("score")
                if sa is not None and sb is not None:
                    slot["score_delta"].append(abs(sb - sa))
                slot["score_change"].append(float(ans_a.get("chosen") != ans_b.get("chosen")))

    def summarize(slot: dict) -> dict:
        return {
            "n": slot["n"],
            "noul_mean_abs_delta": _mean(slot["noul_delta"]),
            "noul_max_abs_delta": max(slot["noul_delta"]) if slot["noul_delta"] else None,
            "noul_flip_rate": _mean(slot["noul_flip"]),
            "choice_change_rate": _mean(slot["choice_change"]),
            "choice_mean_tv": _mean(slot["choice_tv"]),
            "score_mean_abs_delta": _mean(slot["score_delta"]),
            "score_change_rate": _mean(slot["score_change"]),
        }

    pooled = {"noul_delta": [], "noul_flip": [], "choice_change": [], "choice_tv": [],
              "score_delta": [], "score_change": [], "n": 0}
    for slot in buckets.values():
        for k, v in slot.items():
            if k == "n":
                pooled["n"] += v
            else:
                pooled[k].extend(v)
    out = summarize(pooled)
    out["by_kind"] = {k: summarize(v) for k, v in sorted(buckets.items())}
    return out


def _temporal_pass(client: JevClient, run_dir: Path, sizes: dict) -> tuple[dict, dict, list]:
    """Run one pass: the t=0 baseline if none exists, otherwise a follow-up."""
    path = run_dir / TEMPORAL_FILE
    if path.exists():
        store = json.loads(path.read_text())
        queries = store["queries"]
        fresh_baseline = False
    else:
        queries = _build_temporal_queries(sizes["temporal_queries"])
        store = {
            "experiment": EXPERIMENT,
            "condition": "temporal-stability",
            "master_seed": config.MASTER_SEED,
            "scale": config.SCALE,
            "n_queries": len(queries),
            "queries": queries,
            "passes": [],
        }
        fresh_baseline = True

    pass_index = len(store["passes"])
    started = time.time()
    index, failures, results = _run(
        client, _temporal_calls(queries, pass_index), f"temporal-pass{pass_index}"
    )
    first = _collect_pass_answers(queries, index, 0)
    repeat = _collect_pass_answers(queries, index, 1)
    kinds = {q["query_id"]: q["kind"] for q in queries}
    versions = sorted({r.model_version for r in results if r.ok and r.model_version})

    t0 = store["passes"][0]["started"] if store["passes"] else started
    record = {
        "index": pass_index,
        "started": started,
        "elapsed_hours": (started - t0) / 3600.0,
        "model_versions": versions,
        "failures": failures,
        "n_answered": len(first),
        # Counted separately from `n_answered`: a query whose measurement call
        # succeeded and whose repeat failed still loses its noise-floor data
        # point, and that loss has to reach the run's excluded count rather than
        # disappearing into a `within_pass_repeat` computed over fewer queries.
        "n_repeat_answered": len(repeat),
        "answers": first,
        "repeat_answers": repeat,
        "within_pass_repeat": _compare_answers(first, repeat, kinds) if repeat else None,
    }
    store["passes"].append(record)
    path.write_text(json.dumps(store, indent=2, default=str))

    # The report never needs the stored answer bodies; they stay on disk so a
    # later pass can score against them.
    passes_summary = []
    baseline_answers = store["passes"][0]["answers"]
    for p in store["passes"]:
        drift = (
            _compare_answers(baseline_answers, p["answers"], kinds)
            if p["index"] > 0
            else None
        )
        passes_summary.append(
            {
                "index": p["index"],
                "label": _pass_label(p["elapsed_hours"], p["index"]),
                "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p["started"])),
                "elapsed_hours": p["elapsed_hours"],
                "model_versions": p["model_versions"],
                "n_answered": p["n_answered"],
                # `.get` with a fallback: a passes file written before this key
                # existed still has the answer bodies to count.
                "n_repeat_answered": p.get(
                    "n_repeat_answered", len(p.get("repeat_answers") or {})
                ),
                "failures": p["failures"],
                "within_pass_repeat": p.get("within_pass_repeat"),
                "drift_vs_t0": drift,
            }
        )

    all_versions = sorted({v for p in store["passes"] for v in p["model_versions"]})
    kinds_used = sorted({q["kind"] for q in queries})
    condition = {
        "n_queries": len(queries),
        # What was actually drawn, not the four kinds the set cycles over: below
        # four queries the later kinds never appear, and claiming coverage this
        # run does not have would overstate what the drift measurement covers.
        "question_kinds": kinds_used,
        "question_kinds_not_covered": [k for k in TEMPORAL_KINDS if k not in kinds_used],
        "passes_run": len(store["passes"]),
        "passes": passes_summary,
        "model_versions_across_passes": all_versions,
        "model_version_changed": len(all_versions) > 1,
        "baseline_file": str(path.relative_to(run_dir)),
        "fresh_baseline": fresh_baseline,
        "how_to_run_later_passes": (
            f"Run `{TEMPORAL_ONLY_ENV}=1 python run.py e6 --run-id <this run id>` about an "
            "hour after the first pass and again about a day after. It replays the queries "
            f"saved in {TEMPORAL_FILE}, scores them against the t=0 answers, merges the "
            f"refreshed temporal section into {RESULT_FILE} without touching the other "
            f"conditions, and spends {2 * len(queries)} calls per pass."
        ),
        "interpretation": (
            "Drift is only readable against the within-pass repeat noise measured in "
            "the same pass, which is reported beside it. Drift at or below that noise "
            "is sampling variation, not a model change."
        ),
    }
    if len(store["passes"]) < 2:
        condition["note"] = (
            "Only the t=0 pass has run, so no drift is measurable yet. The within-pass "
            "repeat figures are the noise floor a later pass will be read against."
        )
    return condition, failures, results


def _pass_label(elapsed_hours: float, index: int) -> str:
    if index == 0:
        return "t=0"
    nearest = min(PLANNED_CHECKPOINTS_H, key=lambda h: abs(h - elapsed_hours))
    return f"t=+{elapsed_hours:.2f}h (nearest planned checkpoint t={nearest:g}h)"


# --------------------------------------------------------------------------
# Anomalies the plan asks every experiment to watch for
# --------------------------------------------------------------------------


def _watch_list_anomalies(results: list[CallResult], triples: list[dict]) -> list[str]:
    out: list[str] = []
    nouls: list[float] = []
    conf_vs_max: list[tuple[float, float]] = []
    for r in results:
        if not r.ok:
            continue
        for ans in r.answers.values():
            if ans["type"] == "noul":
                nouls.append(ans["p"])
            elif ans["type"] in ("choice", "score"):
                c, m = ans.get("confidence"), ans.get("max_probability")
                if c is not None and m is not None:
                    conf_vs_max.append((c, m))

    if nouls:
        off_grid = [p for p in nouls if abs(p * 100 - round(p * 100)) > 1e-9]
        distinct = sorted(set(nouls))
        if not off_grid:
            out.append(
                f"Every noul probability in E6 ({len(nouls)} answers, {len(distinct)} distinct "
                "values) was an exact multiple of 0.01. Consistent with the 2-decimal "
                "quantization on the plan's watch list; no counter-evidence seen here."
            )
        else:
            out.append(
                f"{len(off_grid)} of {len(nouls)} noul probabilities were NOT multiples of "
                f"0.01 (e.g. {off_grid[:3]}), so the 2-decimal quantization is not universal."
            )
        halves = sum(1 for p in nouls if p == 0.5)
        if halves:
            out.append(
                f"{halves} of {len(nouls)} noul answers came back exactly 0.5. This module "
                "resolves those to 'yes' via the p >= 0.5 convention, which biases the "
                "induced tournament; the count is reported so the effect can be bounded."
            )
        counts = Counter(nouls)
        common = sorted(((c, v) for v, c in counts.items()), reverse=True)[:3]
        top = ", ".join(f"{v} x{c}" for c, v in common)
        if common and common[0][0] / len(nouls) > 0.25:
            out.append(
                f"Noul probabilities cluster: the most common values are {top} out of "
                f"{len(nouls)} answers, so the distribution is not spread smoothly."
            )

    if conf_vs_max:
        diffs = [abs(c - m) for c, m in conf_vs_max]
        mean_diff = _mean(diffs) or 0.0
        if mean_diff > 0.01:
            out.append(
                f"`confidence` diverged from max-probability on choice/score answers by "
                f"{mean_diff:.3f} on average (max {max(diffs):.3f}, n={len(diffs)}). The two "
                "are not the same quantity."
            )
        else:
            out.append(
                f"`confidence` tracked max-probability to within {mean_diff:.3f} on average "
                f"across {len(diffs)} choice/score answers in E6."
            )

    usable = [t for t in triples if t["pairwise_ok"]]
    if usable:
        cyclic = [t for t in usable if t["cyclic"]]
        if cyclic:
            margins = [t["mean_margin"] for t in cyclic if t["mean_margin"] is not None]
            clean = [t["mean_margin"] for t in usable if not t["cyclic"] and t["mean_margin"] is not None]
            mc, mk = _mean(margins), _mean(clean)
            if mc is not None and mk is not None:
                out.append(
                    f"Triples that cycled were answered with a mean distance from 0.5 of "
                    f"{mc:.3f} against {mk:.3f} for triples that did not. Since noul carries "
                    "no confidence field, that distance is the only confidence-like signal "
                    "available, and it is what would have to be thresholded to filter cycles."
                )
    return out


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------

MIN_N_FOR_RATE = 30
MIN_N_FOR_MEAN = 20

# What a rung with no usable triple is called. Deliberately not one of the
# rubric's tier names: the report prints this string in the tier column, and it
# has to read as "this was not measured" rather than as a grade.
UNMEASURED_TIER = "unmeasured"


def _score_predictions(close: dict, enrolled: dict, product: dict) -> list[dict]:
    table = {p.id: p for p in predictions.for_experiment(EXPERIMENT)}
    out: list[dict] = []

    # -- P16 ---------------------------------------------------------------
    rate = close.get("cycle_rate")
    n_close = close.get("n_triples", 0)
    ci = close.get("cycle_rate_ci95")
    if rate is None or n_close < MIN_N_FOR_RATE:
        verdict = "untestable"
        outcome = (
            f"Only {n_close} close-pair triples were scored, below the {MIN_N_FOR_RATE} "
            "needed to tell a 3-10% rate from 0% or from 20%. The prediction is not "
            "scorable at this sample size; rerun at scale 1.0."
        )
    elif 0.03 <= rate <= 0.10:
        verdict = "right"
        outcome = (
            f"Cycle rate on close pairs was {rate:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}, "
            f"n={n_close} triples), inside the predicted 3-10% band and well under the "
            f"{RANDOM_CYCLE_RATE} rate independent answers imply."
        )
    elif rate > 0.20 or rate < 0.005:
        verdict = "wrong"
        outcome = (
            f"Cycle rate on close pairs was {rate:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}, "
            f"n={n_close}), which meets the plan's own falsification condition "
            f"({'>20%' if rate > 0.20 else 'approximately 0%'})."
        )
    else:
        verdict = "wrong"
        outcome = (
            f"Cycle rate on close pairs was {rate:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}, "
            f"n={n_close}), outside the predicted 3-10% band but short of the plan's "
            "stated falsification threshold of >20% or approximately 0%. The claim is "
            "not supported; it is not falsified on the plan's own terms."
        )
    out.append(
        {
            "id": "P16",
            "claim": table["P16"].claim,
            "outcome": outcome,
            "verdict": verdict,
            "evidence": {
                "cycle_rate_close": rate,
                "ci95": ci,
                "n_triples_close": n_close,
                "random_cycle_rate": RANDOM_CYCLE_RATE,
                "constant_answerer_cycle_rate": close.get("constant_answerer_cycle_rate"),
                "falsified_if": table["P16"].falsified_if,
            },
        }
    )

    # -- P17 ---------------------------------------------------------------
    # Untestable as specified: the enrolled form has no cyclic option, so its
    # violation rate is a property of the encoding, not a measurement. The
    # paired order-recovery comparison is reported in its place.
    sep_acc = enrolled.get("separate_order_correct_rate")
    enr_acc = enrolled.get("enrolled_accuracy")
    n_paired = enrolled.get("n_paired", 0)
    mcnemar_p = enrolled.get("mcnemar_p")
    comparison = (
        f"On the same {n_paired} triples, one enrolled call recovered the true order "
        f"{_fmt(enr_acc, '.3f')} of the time against {_fmt(sep_acc, '.3f')} for the three "
        f"separate calls (McNemar p={_fmt(mcnemar_p, '.3g')})."
        if n_paired
        else "No triple had all four calls succeed, so the paired comparison is empty."
    )
    if enrolled.get("small_n_note"):
        comparison += f" That figure is thin: {enrolled['small_n_note']}."
    out.append(
        {
            "id": "P17",
            "claim": table["P17"].claim,
            "outcome": (
                "Untestable as specified. A choice over the 6 total orderings has no cyclic "
                "option, so the enrolled violation rate is 0.000 by construction rather than "
                "by measurement, and 'cut violations below 2%' cannot discriminate between a "
                "model that reasons well and one that does not. Scoring it 'right' would put "
                "a fact about the encoding into the prediction hit rate. The testable content "
                "of the claim -- that enrolling the outcome space beats making cross-calls -- "
                f"is the paired order-recovery comparison. {comparison}"
            ),
            "verdict": "untestable",
            "evidence": {
                "enrolled_cycle_rate": enrolled.get("enrolled_cycle_rate"),
                "enrolled_order_accuracy": enr_acc,
                "separate_order_accuracy": sep_acc,
                "n_paired_triples": n_paired,
                "mcnemar_p": mcnemar_p,
                "paired_bootstrap": enrolled.get("paired_bootstrap"),
                "why_untestable": (
                    "the enrolled option set contains only total orders, so a cycle is "
                    "unrepresentable"
                ),
            },
        }
    )

    # -- P18 ---------------------------------------------------------------
    err = product.get("product_rule_error")
    n_units = product.get("n", 0)
    null = product.get("permutation_null")
    if err is None or n_units < MIN_N_FOR_MEAN:
        verdict = "untestable"
        outcome = (
            f"Only {n_units} product-rule units were scored, below the {MIN_N_FOR_MEAN} "
            "needed for the mean to separate 0.05 from 0.15. Not scorable at this sample "
            "size; rerun at scale 1.0."
        )
    elif 0.10 <= err <= 0.20:
        verdict = "right"
        outcome = (
            f"Mean |P(A)P(B|A) - P(A and B)| was {err:.3f} over {n_units} units, inside the "
            f"predicted 0.1-0.2 band, against a re-pairing null of {_fmt(null, '.3f')} and "
            "0.000 for the same quantity computed rather than asked."
        )
    elif err <= 0.05:
        verdict = "wrong"
        outcome = (
            f"Mean |P(A)P(B|A) - P(A and B)| was {err:.3f} over {n_units} units, within the "
            "0.05 the plan named as its falsification condition. The product rule holds "
            "across separate calls better than predicted."
        )
    else:
        verdict = "wrong"
        outcome = (
            f"Mean |P(A)P(B|A) - P(A and B)| was {err:.3f} over {n_units} units, outside the "
            "predicted 0.1-0.2 band but above the 0.05 that would have falsified it. The "
            "claim is not supported; it is not falsified on the plan's own terms."
        )
    out.append(
        {
            "id": "P18",
            "claim": table["P18"].claim,
            "outcome": outcome,
            "verdict": verdict,
            "evidence": {
                "product_rule_error": err,
                "n_units": n_units,
                "permutation_null": null,
                "tabled_baseline_error": product.get("tabled_baseline_error"),
                "conditioning_shift": product.get("conditioning_shift"),
                "conjunction_fallacy_rate": product.get("conjunction_fallacy_rate"),
                "by_level": product.get("by_level"),
                "falsified_if": table["P18"].falsified_if,
            },
        }
    )
    return out


# --------------------------------------------------------------------------
# What this changes
# --------------------------------------------------------------------------


def _cycle_horizon(rate: float) -> int | None:
    """Smallest item count m at which a pairwise-built ranking is more likely
    than not to contain at least one cycle, given a per-triple cycle rate.

    Treats the C(m, 3) triples as independent, which they are not -- they share
    pairs, so this understates the real horizon somewhat. It is reported as an
    order-of-magnitude figure for exactly that reason.
    """
    if rate <= 0.0 or rate >= 1.0:
        return None
    target = math.log(0.5) / math.log(1.0 - rate)
    for m in range(3, 2001):
        if math.comb(m, 3) >= target:
            return m
    return None


def _what_this_changes(close: dict, far: dict, product: dict, negation: dict) -> str:
    rate = close.get("cycle_rate")
    perr = product.get("product_rule_error")
    nerr = negation.get("negation_sum_error")
    if rate is None and perr is None and nerr is None:
        return (
            "Nothing was measured at a usable sample size, so this run says nothing about "
            "the tabling requirement either way."
        )
    thin = [
        f"{name} (n={got}, needs {want})"
        for name, got, want in (
            ("close-pair triples", close.get("n_triples", 0), MIN_N_FOR_RATE),
            ("product-rule units", product.get("n", 0), MIN_N_FOR_MEAN),
            ("negation propositions", negation.get("n", 0), MIN_N_FOR_MEAN),
        )
        if got < want
    ]
    if thin:
        return (
            f"Measured at too small a sample to support a conclusion: {'; '.join(thin)}. "
            f"The numbers this run produced were a close-pair cycle rate of {_fmt(rate, '.3f')}, "
            f"a product-rule error of {_fmt(perr, '.3f')} and a negation-sum error of "
            f"{_fmt(nerr, '.3f')}, but none of them separates the plan's thresholds at this "
            "size, so this run neither supports nor undermines the tabling requirement. "
            "Rerun at scale 1.0 before reading it either way."
        )
    incoherent = any(
        v is not None and v > t
        for v, t in ((rate, 0.0), (perr, 0.05), (nerr, 0.02))
    )
    horizon = _cycle_horizon(rate) if rate else None
    pieces = [
        f"Close-pair triples cycled at {_fmt(rate, '.3f')} and far-pair triples at "
        f"{_fmt(far.get('cycle_rate'), '.3f')}, against the {RANDOM_CYCLE_RATE} independent "
        f"answers imply; the product rule was off by {_fmt(perr, '.3f')} across separate "
        f"calls, and P(X) + P(not X) missed 1 by {_fmt(nerr, '.3f')}."
    ]
    if incoherent:
        pieces.append(
            "Those numbers support the plan's tabling requirement: anything that needs a "
            "consistent model theory -- a Prolog-style fact store, a ranking used as ground "
            "truth, a derived knowledge graph -- must table every derived fact on first use "
            "and read the table thereafter, because asking twice can return answers that "
            "cannot both be true."
        )
        if horizon:
            pieces.append(
                f"At the measured rate a ranking built from pairwise calls over about "
                f"{horizon} items is more likely than not to contain at least one cycle, so "
                "the requirement binds at small scale, not only in the limit."
            )
        pieces.append(
            "The reason is soundness rather than cost: the tabled value is the only thing "
            "that makes two derivations of the same fact agree, and the cheap alternative "
            "on this task -- a sorted lookup table -- is both exact and free."
        )
    else:
        pieces.append(
            "Those numbers do not support the tabling requirement as a soundness measure: "
            "answers obtained in separate calls composed within the tolerance the plan set, "
            "so caching derived facts remains a cost optimisation rather than a correctness "
            "one."
        )
    pieces.append(
        "Enrollment removes the problem only where the whole question fits in one call; a "
        "choice over the 6 orderings of three items cannot express a cycle, but the same "
        "trick does not scale to a fact store, and 255 options is the ceiling."
    )
    return " ".join(pieces)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _plots(run_dir: Path, rungs: list[dict], product: dict, negation: dict, temporal: dict) -> list[str]:
    out: list[str] = []
    plot_dir = run_dir / "plots"

    scored = [r for r in rungs if r["cycle_rate"] is not None]
    if scored:
        xs = [r["rung"] for r in scored]
        ys = [r["cycle_rate"] for r in scored]
        cis = [r["cycle_rate_ci95"] or (r["cycle_rate"], r["cycle_rate"]) for r in scored]
        series = {
            "separate calls": ys,
            "one enrolled call": [0.0] * len(scored),
            "always-yes answerer": [
                r["constant_answerer_cycle_rate"] if r["constant_answerer_cycle_rate"] is not None else 0.0
                for r in scored
            ],
        }
        path = plots.curve(
            xs,
            series,
            plot_dir / "e6_cycle_rate_by_distance.png",
            "E6: transitivity violations across separate calls",
            "item distance in the true order (easy to hard)",
            "fraction of triples forming a cycle",
            n=[r["n_triples"] for r in scored],
            band={"separate calls": ([c[0] for c in cis], [c[1] for c in cis])},
            baseline=RANDOM_CYCLE_RATE,
            baseline_label="independent answers imply",
            n_unit="triples per rung",
            note="enrolled is 0 by construction: a choice over total orders has no cyclic option",
        )
        out.append(str(path.relative_to(run_dir)))

    levels = [(lvl, product.get("by_level", {}).get(lvl, {})) for lvl in ("clean", "hard")]
    levels = [(lvl, b) for lvl, b in levels if b.get("product_rule_error") is not None]
    if levels:
        path = plots.curve(
            [lvl for lvl, _ in levels],
            {"|P(A)P(B|A) - P(A and B)|": [b["product_rule_error"] for _, b in levels]},
            plot_dir / "e6_product_rule.png",
            "E6: product rule across four separate calls",
            "ticket wording level",
            "mean absolute product-rule error",
            n=[b["n"] for _, b in levels],
            baseline=product.get("permutation_null"),
            baseline_label="answers re-paired at random",
            n_unit="units per level",
            note="0.000 is what computing the conjunction from the two answers would give",
        )
        out.append(str(path.relative_to(run_dir)))

    domains = [(d, b) for d, b in (negation.get("by_domain") or {}).items()
               if b.get("negation_sum_error") is not None]
    if domains:
        path = plots.curve(
            [d for d, _ in domains],
            {"|P(X) + P(not X) - 1|": [b["negation_sum_error"] for _, b in domains]},
            plot_dir / "e6_negation_symmetry.png",
            "E6: negation symmetry across separate calls",
            "domain",
            "mean deviation of P(X) + P(not X) from 1",
            n=[b["n"] for _, b in domains],
            baseline=negation.get("permutation_null"),
            baseline_label="answers re-paired at random",
            n_unit="propositions per domain",
        )
        out.append(str(path.relative_to(run_dir)))

    passes = [p for p in temporal.get("passes", []) if p.get("drift_vs_t0")]
    if passes:
        xs = [p["elapsed_hours"] for p in passes]
        drift = [p["drift_vs_t0"].get("noul_mean_abs_delta") or 0.0 for p in passes]
        noise = [
            (p.get("within_pass_repeat") or {}).get("noul_mean_abs_delta") or 0.0 for p in passes
        ]
        path = plots.curve(
            xs,
            {"drift vs t=0": drift, "within-pass repeat noise": noise},
            plot_dir / "e6_temporal_drift.png",
            "E6: answer drift over time",
            "hours since the t=0 pass",
            "mean |delta p| on noul answers",
            n=[p["n_answered"] for p in passes],
            n_unit="queries per pass",
            note="drift at or below the repeat noise is sampling variation, not a model change",
        )
        out.append(str(path.relative_to(run_dir)))

    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    if os.environ.get(TEMPORAL_ONLY_ENV):
        return _run_temporal_only(run_dir)
    return _run_full(run_dir)


def _run_temporal_only(run_dir: Path) -> dict:
    """A follow-up temporal pass, merged into the result the full run wrote.

    run.py overwrites e6_result.json with whatever this returns, so the previous
    result is loaded and only its temporal section replaced. Without that, an
    operator running the t=24h pass would destroy the other four conditions'
    numbers.
    """
    sizes = _sizes()
    previous_path = run_dir / RESULT_FILE
    previous: dict | None = None
    if previous_path.exists():
        try:
            previous = json.loads(previous_path.read_text())
        except json.JSONDecodeError:
            previous = None

    with JevClient(run_dir=run_dir, log_name=LOG_NAME) as client:
        temporal, failures, results = _temporal_pass(client, run_dir, sizes)
        summary = client.summary()

    if previous is None:
        return {
            "experiment": EXPERIMENT,
            "question": tiers.rubric(EXPERIMENT).question,
            "headline": None,
            "by_difficulty": [],
            "boundary_crossings": {},
            "plots": _plots(run_dir, [], {}, {}, temporal),
            "predictions": [],
            "anomalies": [
                f"{TEMPORAL_ONLY_ENV} was set but no {RESULT_FILE} was found in this run "
                "directory, so only the temporal condition ran and there is nothing to "
                "merge it into."
            ],
            "failures": {"calls": failures["calls"], "excluded": 0, "reasons": failures["reasons"]},
            "conditions": {"temporal_stability": temporal},
            "run": summary,
            "what_this_changes": (
                "Only the temporal pass ran, so this invocation says nothing about the "
                "tabling requirement."
            ),
        }

    previous.setdefault("conditions", {})["temporal_stability"] = temporal
    previous["plots"] = sorted(
        set(previous.get("plots") or []) | set(_plots(run_dir, [], {}, {}, temporal))
    )
    anomalies = [
        a for a in (previous.get("anomalies") or []) if not a.startswith("Temporal pass")
    ]
    anomalies.append(
        f"Temporal pass {temporal['passes_run'] - 1} ran "
        f"{temporal['passes'][-1]['elapsed_hours']:.2f}h after t=0; model version(s) seen "
        f"across passes: {', '.join(temporal['model_versions_across_passes']) or 'none'}"
        + (" -- THE MODEL VERSION CHANGED MID-RUN." if temporal["model_version_changed"] else ".")
    )
    previous["anomalies"] = anomalies
    previous["temporal_only_refresh"] = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": (
            "Only the temporal condition was rerun. Every other number in this file is "
            "from the original full run and its calls were not re-spent, so the top-level "
            "`failures` block still counts that run only -- this pass's failures are under "
            "conditions.temporal_stability.passes[-1].failures."
        ),
        "run": summary,
    }
    return previous


def _run_full(run_dir: Path) -> dict:
    sizes = _sizes()
    planned = _planned_calls(sizes)
    notes: list[str] = []
    if config.SCALE != 1.0:
        notes.append(
            f"Sample sizes were scaled by {config.SCALE}. Every n below is the scaled one."
        )

    ordering_calls, units = _build_ordering(sizes["triples_per_rung"])
    pool = min((u["pool"] for u in units.values()), default=0)
    if pool and sizes["triples_per_rung"] > pool:
        notes.append(
            f"A rung was asked for {sizes['triples_per_rung']} triples but has only {pool} "
            "distinct ones, so its effective sample size is the smaller number."
        )
    # The three calls of a triple are shuffled into the stream rather than
    # issued back to back. There is no shared state across calls, so this should
    # not matter -- and if it does, that is itself a finding.
    random.Random(config.seed_for(EXPERIMENT, "issue-order")).shuffle(ordering_calls)

    product_calls, product_units = _build_product_rule(sizes["product_rule_units_per_level"])
    negation_calls, negation_props = _build_negation(sizes["negation_props_per_domain"])

    all_results: list[CallResult] = []
    failure_calls = 0
    failure_reasons: dict[str, int] = {}

    def tally(f: dict) -> None:
        nonlocal failure_calls
        failure_calls += f["calls"]
        for k, v in f["reasons"].items():
            failure_reasons[k] = failure_reasons.get(k, 0) + v

    with JevClient(run_dir=run_dir, log_name=LOG_NAME) as client:
        ord_index, f1, r1 = _run(client, ordering_calls, "transitivity+enrolled")
        prod_index, f2, r2 = _run(client, product_calls, "product-rule")
        neg_index, f3, r3 = _run(client, negation_calls, "negation")
        temporal, f4, r4 = _temporal_pass(client, run_dir, sizes)
        summary = client.summary()
    for f in (f1, f2, f3, f4):
        tally(f)
    all_results = r1 + r2 + r3 + r4

    # -- transitivity and enrolled ----------------------------------------
    scored = [_score_triple(u, ord_index) for u in units.values()]
    const_rates = _constant_answerer_cycle_rate(units)
    by_rung: list[dict] = []
    for d in ordering.difficulty_sweep():
        key = _rung_key(d)
        rows = [s for s in scored if s["rung"] == key]
        agg = _aggregate_triples(rows)
        agg.update(
            {
                "rung": key,
                "difficulty": dict(d),
                "distance_class": d["distance"],
                "constant_answerer_cycle_rate": const_rates.get(key),
                "random_cycle_rate": RANDOM_CYCLE_RATE,
                "pairwise_chance_baseline": 0.5,
                "pairwise_majority_baseline": _pairwise_majority(units, key),
                "table_lookup_baseline_accuracy": 1.0,
            }
        )
        by_rung.append(agg)

    close = _pool_rows([s for s in scored if s["distance_class"] == "close"], const_rates, "close")
    far = _pool_rows([s for s in scored if s["distance_class"] == "far"], const_rates, "far")
    enrolled = _enrolled_comparison(scored)

    product = _score_product_rule(product_units, prod_index)
    negation = _score_negation(negation_props, neg_index)

    # -- tiers -------------------------------------------------------------
    #
    # Every rung is graded on its OWN cycle rate. Of the plan's five rows only
    # two are qualified by distance -- "Superhuman: <1% cycles on close pairs,
    # <0.1% on far pairs" and "Human: 3-8% cycles on close pairs". Perfect, Bad
    # and Doesn't-work are stated about cycles unqualified, and the harness's
    # rubric keys all of them on `cycle_rate_close`. So each rung's own rate goes
    # under `cycle_rate_close`, where the unqualified rows read it, and
    # `cycle_rate_far` additionally carries the rung's own rate on a far rung, so
    # the stricter far threshold applies exactly where the plan says it does. On a
    # close rung `cycle_rate_far` carries the run's pooled far rate, because
    # Superhuman is a joint claim about both classes rather than a per-condition
    # one.
    #
    # The first version of this passed the pooled close rate to the far rungs.
    # It graded every far rung on a number measured somewhere else, which
    # flattened the whole curve to one tier and put every boundary crossing on
    # the easiest rung -- the single overall grade the plan says throws away the
    # finding.
    #
    # A rung on which no triple produced three usable answers is left out of the
    # grading entirely. `tiers.assign` falls through to Bad as its residual, and
    # because E6 also hands it the whole-experiment product-rule and negation
    # numbers the rung does not even look unevaluated -- it comes back Bad,
    # reportable and fully evaluated, and the boundary-crossing search then reads
    # that Bad as the rung where the model stopped working. That is a failed call
    # defaulted into a finding, one level up from the answers.
    sweep_metrics: list[tuple[dict, dict]] = []
    graded_rungs: list[dict] = []
    rung_metrics: dict[str, dict] = {}
    for row in by_rung:
        m: dict[str, Any] = {
            "cycle_rate": row["cycle_rate"],
            "cycle_rate_close": row["cycle_rate"],
            "cycle_rate_far": (
                row["cycle_rate"] if row["distance_class"] == "far" else far.get("cycle_rate")
            ),
            "random_cycle_rate": RANDOM_CYCLE_RATE,
            # Whole-experiment quantities. Perfect is defined as zero cycles AND
            # a product rule within 0.02 AND negation summing to 1.00 +/- 0.01,
            # so a rung cannot reach it on its cycle rate alone.
            "product_rule_error": product.get("product_rule_error"),
            "negation_sum_error": negation.get("negation_sum_error"),
            "pairwise_accuracy": row["pairwise_accuracy"],
            "pairwise_chance_baseline": 0.5,
            "enrolled_cycle_rate": row["enrolled_cycle_rate"],
            "n": row["n_triples"],
        }
        rung_metrics[row["rung"]] = m
        if row["n_triples"]:
            sweep_metrics.append((row["difficulty"], m))
            graded_rungs.append(row)

    tier_results = tiers.tier_by_difficulty(EXPERIMENT, sweep_metrics)
    crossings = tiers.boundary_crossings(EXPERIMENT, tier_results)
    graded = {row["rung"]: t for t, row in zip(tier_results, graded_rungs)}
    difficulty_rows = []
    thin_rungs = []
    unmeasured_rungs = []
    for row in by_rung:
        t = graded.get(row["rung"])
        if t is None:
            unmeasured_rungs.append(row["rung"])
            difficulty_rows.append(
                {
                    "difficulty": row["difficulty"],
                    "tier": UNMEASURED_TIER,
                    "n": 0,
                    "metrics": dict(rung_metrics[row["rung"]]),
                    "metric": "cycle_rate_close",
                    "value": None,
                    "baseline": RANDOM_CYCLE_RATE,
                    "baseline_name": "independent answers imply",
                    "baseline_source": "assumed",
                    "reportable": False,
                    "fully_evaluated": False,
                    "matched_rule": "",
                    "reasons": [
                        "no triple on this rung produced three usable answers, so it has "
                        f"no cycle rate; {UNMEASURED_TIER!r} is the absence of a grade, not "
                        "a grade, and the rung is left out of the boundary-crossing search"
                    ],
                    "signatures": [],
                    "missing": ["cycle_rate_close", "cycle_rate_far"],
                    "unevaluated": [],
                    "undefined": [],
                    "small_n": True,
                    "note": "n=0 triples: every call on this rung failed or was excluded",
                }
            )
            continue
        d = t.to_json()
        d["small_n"] = row["n_triples"] < MIN_N_FOR_RATE
        if d["small_n"]:
            d["note"] = (
                f"n={row['n_triples']} triples: a cycle rate cannot separate the plan's "
                f"thresholds below {MIN_N_FOR_RATE}, so this tier is arithmetic rather than "
                "a measurement"
            )
            thin_rungs.append(f"{row['rung']} (n={row['n_triples']})")
        difficulty_rows.append(d)

    # -- assembly ----------------------------------------------------------
    #
    # Every data point a failed call cost, counted once. The enrolled call and
    # the temporal repeat each carry their own data point -- a paired
    # before/after entry and a noise-floor entry -- so a failure there is an
    # exclusion even when the triple's three pairwise calls or the measurement
    # call succeeded.
    last_pass = temporal["passes"][-1] if temporal["passes"] else None
    temporal_excluded = 0
    if last_pass is not None:
        temporal_excluded = max(0, temporal["n_queries"] - last_pass["n_answered"]) + max(
            0, temporal["n_queries"] - last_pass["n_repeat_answered"]
        )
    excluded = (
        sum(r["n_triples_excluded"] for r in by_rung)
        + sum(r["n_enrolled_excluded"] for r in by_rung)
        + product.get("n_excluded", 0)
        + negation.get("n_excluded", 0)
        + temporal_excluded
    )
    anomalies = _watch_list_anomalies(all_results, scored)
    versions = summary.get("model_versions") or []
    if len(versions) > 1:
        anomalies.append(
            f"More than one model version answered during E6 ({', '.join(versions)}). "
            "Cross-condition comparisons within this experiment are not safe."
        )
    if temporal["model_version_changed"]:
        anomalies.append(
            "The model version changed between temporal passes: "
            f"{', '.join(temporal['model_versions_across_passes'])}."
        )
    if unmeasured_rungs:
        anomalies.append(
            f"No triple survived on {len(unmeasured_rungs)} rung(s) -- "
            f"{', '.join(unmeasured_rungs)}. They carry no cycle rate, are reported as "
            f"'{UNMEASURED_TIER}' rather than graded, and are excluded from the "
            "boundary-crossing search, so no tier boundary below is attributed to them."
        )
    if thin_rungs:
        anomalies.append(
            "Run note: these rungs were scored on too few triples for their tier to mean "
            f"anything -- {', '.join(thin_rungs)}. A cycle rate needs about "
            f"{MIN_N_FOR_RATE} triples before it separates 0%, 3-10% and 20%."
        )
    if (
        product.get("n", 0) >= MIN_N_FOR_MEAN
        and product.get("conditioning_shift") is not None
        and product["conditioning_shift"] < 0.01
    ):
        anomalies.append(
            f"Asserting A in the state moved P(B) by only {product['conditioning_shift']:.3f}, "
            "so the conditional call is close to a repeat of the marginal call. The "
            "product-rule test then reduces to a test of independence, which is worth "
            "knowing before reading its number as a Bayesian failure."
        )
    if negation.get("n", 0) >= MIN_N_FOR_MEAN and negation.get("yes_to_both_rate"):
        anomalies.append(
            f"On {negation['yes_to_both_rate']:.1%} of propositions the model answered above "
            "0.5 to both 'is X true' and 'is X false' in separate calls."
        )
    for note in notes:
        anomalies.append(f"Run note: {note}")

    headline_n = close.get("n_triples", 0)
    result = {
        "experiment": EXPERIMENT,
        "question": tiers.rubric(EXPERIMENT).question,
        "headline": {
            "metric": "cycle_rate_close",
            "value": close.get("cycle_rate"),
            "baseline": RANDOM_CYCLE_RATE,
            "baseline_name": "independent answers imply 0.25",
            "n": headline_n,
        },
        "by_difficulty": difficulty_rows,
        "boundary_crossings": tiers.crossings_json(crossings),
        "plots": _plots(run_dir, by_rung, product, negation, temporal),
        "predictions": _score_predictions(close, enrolled, product),
        "anomalies": anomalies,
        "failures": {
            "calls": failure_calls,
            "excluded": excluded,
            "reasons": failure_reasons,
        },
        "conditions": {
            "transitivity": {
                "by_rung": by_rung,
                "close": close,
                "far": far,
                "baselines": {
                    "random_cycle_rate": RANDOM_CYCLE_RATE,
                    "constant_answerer_cycle_rate": const_rates,
                    "pairwise_chance": 0.5,
                    "table_lookup_accuracy": 1.0,
                    "table_lookup_cycle_rate": 0.0,
                },
                "baseline_note": (
                    "The cheap deterministic baseline for ordering elements by atomic number "
                    "is a sorted lookup table: accuracy 1.0, cycle rate 0.0. It is reported "
                    "here rather than fed to the tier rubric, because E6 grades coherence and "
                    "the rubric's accuracy rows would otherwise assign a tier on a property "
                    "this experiment is not measuring."
                ),
                "confidence_note": (
                    "noul answers carry no confidence field, so `mean_noul_margin` is the mean "
                    "distance of the probability from 0.5 and is reported under that name."
                ),
            },
            "enrolled_triples": enrolled,
            "product_rule": product,
            "temporal_stability": temporal,
            "negation_symmetry": negation,
        },
        "per_condition": _per_condition_cost(all_results),
        "sample_sizes": {**sizes, "planned_calls": planned},
        "notes": notes,
        "run": summary,
        "what_this_changes": _what_this_changes(close, far, product, negation),
    }
    return result


def _pairwise_majority(units: dict[str, dict], rung: str) -> float | None:
    """Majority-class accuracy on the pairwise labels actually drawn.

    The generator randomizes which item is named first, so the yes-rate is 0.50
    only to within sampling error. Reporting against the realized majority rather
    than against a nominal 0.5 is what keeps a 52% accuracy from reading as
    above chance.
    """
    labels: list[bool] = []
    for u in units.values():
        if u["rung"] != rung:
            continue
        for role in ("ab", "bc", "ac"):
            inst = u["instances"].get(role)
            if inst is not None:
                labels.append(bool(inst.truth["greater"]))
    return metrics.majority_baseline(labels) if labels else None


def _pool_rows(rows: list[dict], const_rates: dict, distance: str) -> dict:
    agg = _aggregate_triples(rows)
    rungs = sorted({r["rung"] for r in rows})
    const = [const_rates[k] for k in rungs if const_rates.get(k) is not None]
    agg.update(
        {
            "distance_class": distance,
            "rungs": rungs,
            "constant_answerer_cycle_rate": _mean(const),
            "random_cycle_rate": RANDOM_CYCLE_RATE,
        }
    )
    return agg


def _enrolled_comparison(scored: list[dict]) -> dict:
    """The paired before/after: three separate calls against one enrolled call.

    Restricted to triples where all four calls succeeded, so the comparison is on
    identical material rather than on two differently-thinned samples.
    """
    paired = [s for s in scored if s["pairwise_ok"] and s["enrolled_ok"]]
    if not paired:
        return {
            "n_paired": 0,
            "note": "no triple had all four of its calls succeed",
            "enrolled_cycle_rate": None,
        }
    sep = [bool(s["separate_order_correct"]) for s in paired]
    enr = [bool(s["enrolled_correct"]) for s in paired]
    out: dict[str, Any] = {
        "n_paired": len(paired),
        "separate_order_correct_rate": _mean([float(v) for v in sep]),
        "enrolled_accuracy": _mean([float(v) for v in enr]),
        "enrolled_random_baseline": 1.0 / 6.0,
        # Three coin flips produce 8 equally likely orientations: 2 are cyclic
        # and recover no order, and the other 6 each give a different total
        # order. So a random answerer recovers the true order 1 time in 8, not
        # 1 in 4 -- 0.25 is the rate at which it produces a *cycle*, which is a
        # different event and the wrong baseline to put beside an accuracy.
        "separate_random_baseline": 1.0 / 8.0,
        "cycle_rate_separate": _mean([float(s["cyclic"]) for s in paired]),
        "enrolled_cycle_rate": 0.0,
        "enrolled_cycle_rate_note": (
            "0 by construction: the 6 options are total orders, so a cycle cannot be "
            "selected. This is a property of the encoding, not a measurement of the model."
        ),
        "by_distance": {},
    }
    for distance in ("close", "far"):
        subset = [s for s in paired if s["distance_class"] == distance]
        if not subset:
            continue
        out["by_distance"][distance] = {
            "n": len(subset),
            "separate_order_correct_rate": _mean(
                [float(s["separate_order_correct"]) for s in subset]
            ),
            "enrolled_accuracy": _mean([float(s["enrolled_correct"]) for s in subset]),
            "cycle_rate_separate": _mean([float(s["cyclic"]) for s in subset]),
        }
    out["mcnemar_p"] = metrics.mcnemar(enr, sep)
    if len(paired) >= 10:
        out["paired_bootstrap"] = metrics.paired_bootstrap(
            enr, sep, metrics.accuracy, n_boot=2000, seed=config.seed_for(EXPERIMENT, "enrolled")
        )
    else:
        out["paired_bootstrap"] = None
        out["small_n_note"] = (
            f"only {len(paired)} paired triples: the bootstrap interval was skipped and the "
            "McNemar p-value is not meaningful at this size"
        )
    confs = [s["enrolled_confidence"] for s in paired if s.get("enrolled_confidence") is not None]
    maxes = [
        s["enrolled_max_probability"] for s in paired if s.get("enrolled_max_probability") is not None
    ]
    if confs and len(confs) == len(maxes):
        out["confidence_minus_max_probability"] = _mean(
            [c - m for c, m in zip(confs, maxes)]
        )
    return out


# --------------------------------------------------------------------------
# Terminal summary
# --------------------------------------------------------------------------


def format_report(result: dict) -> str:
    lines = [f"{EXPERIMENT} — {result.get('question', '')}"]
    h = result.get("headline") or {}
    lines.append(
        f"  headline: {h.get('metric')} = {_fmt(h.get('value'), '.4f')} vs "
        f"{_fmt(h.get('baseline'), '.4f')} ({h.get('baseline_name')}), n={h.get('n')}"
    )
    cond = result.get("conditions") or {}
    tr = cond.get("transitivity") or {}
    for distance in ("close", "far"):
        block = tr.get(distance) or {}
        if not block.get("n_triples"):
            continue
        ci = block.get("cycle_rate_ci95") or (None, None)
        lines.append(
            f"  {distance:<5} pairs: cycles {_fmt(block.get('cycle_rate'), '.4f')} "
            f"(95% CI {_fmt(ci[0], '.3f')}-{_fmt(ci[1], '.3f')}, n={block['n_triples']}), "
            f"always-yes baseline {_fmt(block.get('constant_answerer_cycle_rate'), '.3f')}, "
            f"pairwise accuracy {_fmt(block.get('pairwise_accuracy'), '.3f')}"
        )
    enr = cond.get("enrolled_triples") or {}
    if enr.get("n_paired"):
        lines.append(
            f"  enrolled: order recovered {_fmt(enr.get('enrolled_accuracy'), '.3f')} vs "
            f"{_fmt(enr.get('separate_order_correct_rate'), '.3f')} from three separate calls "
            f"(n={enr['n_paired']}, McNemar p={_fmt(enr.get('mcnemar_p'), '.3g')}); "
            "enrolled cycles 0 by construction"
        )
    pr = cond.get("product_rule") or {}
    if pr.get("n"):
        lines.append(
            f"  product rule: |P(A)P(B|A) - P(A and B)| = {_fmt(pr.get('product_rule_error'), '.4f')} "
            f"vs re-pairing null {_fmt(pr.get('permutation_null'), '.3f')} and 0.000 if tabled "
            f"(n={pr['n']}, conditioning shift {_fmt(pr.get('conditioning_shift'), '.3f')})"
        )
    ng = cond.get("negation_symmetry") or {}
    if ng.get("n"):
        lines.append(
            f"  negation: |P(X)+P(not X)-1| = {_fmt(ng.get('negation_sum_error'), '.4f')} vs "
            f"re-pairing null {_fmt(ng.get('permutation_null'), '.3f')} (n={ng['n']}, signed "
            f"{_fmt(ng.get('negation_signed_error'), '+.3f')})"
        )
    tp = cond.get("temporal_stability") or {}
    if tp:
        lines.append(
            f"  temporal: {tp.get('passes_run')} pass(es), "
            f"{tp.get('n_queries')} queries, versions "
            f"{', '.join(tp.get('model_versions_across_passes') or []) or 'none'}"
            + (" -- VERSION CHANGED" if tp.get("model_version_changed") else "")
        )
        if tp.get("note"):
            lines.append(f"    {tp['note']}")
    for row in result.get("by_difficulty") or []:
        lines.append(
            f"  tier @ {row.get('difficulty')}: {row.get('tier')} "
            f"({row.get('metric')}={_fmt(row.get('value'), '.4f')}, n={row.get('n')})"
        )
    for boundary, where in (result.get("boundary_crossings") or {}).items():
        lines.append(f"  {boundary}: crossed at {where}")
    for p in result.get("predictions") or []:
        lines.append(f"  {p['id']}: {p['verdict'].upper()} — {p['outcome'][:160]}")
    f = result.get("failures") or {}
    lines.append(f"  failures: {f.get('calls', 0)} calls, {f.get('excluded', 0)} data points excluded")
    for a in result.get("anomalies") or []:
        lines.append(f"  ! {a}")
    lines.append(f"  what this changes: {result.get('what_this_changes', '')}")
    return "\n".join(lines)
