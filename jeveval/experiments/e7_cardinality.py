"""E7 -- cardinality: how far a single `choice` scales, and what to do instead.

Five conditions, all over the faceted catalogue in `generators/taxonomy.py`:

1. **Knee-finding.** The same 300 objects presented at 2, 4, 8, 16, 32, 64, 128,
   200 and 255 options. The item is a function of its index alone, so every
   cardinality sees the same material and the comparison is within item.
2. **Distractor quality.** Cardinality fixed at 64, the distractor pool varied
   from `far` to `same_object`. The plan predicts similarity matters more than
   count; this is the condition that can tell them apart, because uniform
   sampling in condition 1 holds the per-option similarity distribution constant
   while only the count moves.
3. **Two-stage versus flat at 255.** The flat arm is condition 1's 255-option
   cell -- literally the same calls, so the comparison is paired on identical
   option sets rather than on matched samples. The two-stage arm asks a `noul`
   per candidate in batches of 64 as a filter that never picks, keeps the top 16
   and makes one explicit choice over the survivors. That is five calls against
   one, and the comparison reports total calls, total input tokens and total
   latency, because a pattern that wins on accuracy while costing five calls has
   not obviously won.
4. **Hierarchical descent.** Four adaptive calls down department -> object class
   -> period -> regional school, against one flat choice over a 255-item
   shortlist from a simulated retriever. Per-level error attribution comes out
   of the same calls.
5. **Distribution shape.** Entropy against cardinality from every choice already
   made, plus a `score` sweep over 2 to 100 ordered date bands, because the
   confidence-versus-max-probability divergence the plan flags was first seen on
   a score question.

Three things about the endpoint shape the module and are worth stating here.

*Top-5 is free but has to be tie-aware.* Every probability the API returns is
quantised, so at 255 options dozens of candidates share a value and a naive
"is the truth in the top 5 of a sorted list" would be decided by dictionary
order. `_topk_expectation` returns the probability that the truth lands in the
top k under uniform tie-breaking, which is the unbiased estimate of what a
system reading that distribution would get.

*Quantisation caps entropy.* If probabilities come back on a 0.01 grid, a
distribution over 255 options has at most 100 non-zero entries, so its entropy
cannot exceed ln(100) whatever the model believes -- and the rubric's
`entropy_ratio_255`, which divides by ln(255), is then biased downward by about
28% and makes the distribution look sharper than it is. Both normalisations are
reported and the artifact goes in `anomalies`.

*A noul carries no confidence.* Condition 3's filter therefore ranks candidates
by the returned probability itself, which is the only signal a noul gives.

Failed calls are counted and excluded, never defaulted. An instance whose
two-stage filter lost a batch, or whose descent lost a level, is dropped from
that arm entirely rather than scored on a partial answer: a descent missing
level 2 is not a wrong answer, it is no answer.

Ground truth, the presented option ids and the correct option's position all
travel in `Call.meta`, so every number here can be recomputed from the JSONL log
without re-spending a call.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .. import config, metrics, plots, predictions, tiers
from ..client import Call, CallResult, JevClient
from ..generators import taxonomy as tx
from ..instances import Instance, choice, noul, rng_for, shuffled_options, shuffled_questions

EXPERIMENT = "E7"
QUESTION = (
    "Does choice quality hold across the 2-to-255 option range, and is the "
    "documented two-stage pattern actually better than one flat choice?"
)

# Condition 2 holds cardinality here and varies only the pool the distractors
# come from. 64 is the plan's fixed point and is inside the nearest pool, which
# holds 159 leaves.
DISTRACTOR_CARDINALITY = 64
# Ordered far to near by tree distance, with one exception at the end.
# `same_period_region` is not a nearer rung on the same scale: its distractors
# are far by tree distance but they all share the truth's period and region, so
# the date-and-gazetteer baseline drops to 1/64 there and only reading the
# description separates them. It is the rung that asks whether the model is
# doing the part of the task a lookup table cannot.
SIMILARITY_LADDER: tuple[str, ...] = (
    "far", "uniform", "same_domain", "same_object", "same_period_region",
)

# Condition 3. The per-call question limit is undocumented and is E3's subject,
# not this module's, so the filter stays well inside any plausible cap; 64 nouls
# per call makes 255 candidates four calls.
STAGE1_BATCH = 64
STAGE2_SURVIVORS = 16

# A drop of five points from the two-option ceiling is what this module calls a
# knee. Stated as a constant because the number decides P19 and should not be
# buried in the function that scores it.
KNEE_DROP = 0.05
NEAR_ZERO_RECOVERY = 0.05

# Below these counts a rate or a calibration error is reported with a warning
# attached rather than silently. Ten equal-width ECE bins need real mass per bin.
MIN_N_FOR_RATE = 20
MIN_N_FOR_ECE = 100
# The department-preference check splits its sample across eight cells, so it
# needs eight times the mass a single rate does before a spike is reportable.
MIN_N_FOR_ID_BIAS = MIN_N_FOR_RATE * 8

# The probability grid the endpoint has been observed to quantise to.
PROB_GRID = 0.01


# --------------------------------------------------------------------------
# Scoring one answer
# --------------------------------------------------------------------------


def _calls(instances: Sequence[Instance], condition: str) -> list[Call]:
    return [
        Call(
            experiment=EXPERIMENT,
            condition=condition,
            instance_id=inst.instance_id,
            state=inst.state,
            questions=inst.questions,
            meta={"truth": inst.truth, "difficulty": inst.difficulty,
                  "item_index": inst.index, **inst.meta},
        )
        for inst in instances
    ]


def _on_grid(values: Sequence[float], grid: float = PROB_GRID) -> bool:
    return all(abs(v / grid - round(v / grid)) < 1e-6 for v in values)


def _topk_expectation(vec: Sequence[float], truth_ix: int, k: int) -> float:
    """P(truth is in the top k) under uniform tie-breaking.

    Quantised probabilities make ties the common case at high cardinality, so a
    sort-and-slice would be deciding the answer by whatever order the options
    happened to arrive in. This returns what a system breaking those ties at
    random would actually get.
    """
    n = len(vec)
    if k >= n:
        return 1.0
    p = vec[truth_ix]
    greater = sum(1 for v in vec if v > p)
    tied = sum(1 for v in vec if v == p)
    if greater >= k:
        return 0.0
    if greater + tied <= k:
        return 1.0
    return (k - greater) / tied


def _shot(
    result: CallResult,
    *,
    key: str,
    option_ids: Sequence[str],
    truth_id: str,
    extra: dict | None = None,
) -> dict | None:
    """One scored choice answer, or None when the call failed."""
    if not result.ok:
        return None
    answer = result.answers.get(key)
    if answer is None or answer.get("type") != "choice":
        return None
    probs = answer.get("probabilities") or {}
    vec = [float(probs.get(oid, 0.0)) for oid in option_ids]
    n = len(option_ids)
    truth_ix = option_ids.index(truth_id)
    total = sum(vec)
    support = sum(1 for v in vec if v > 0.0)
    quantised = _on_grid([v for v in vec if v > 0.0])

    ent: float | None
    try:
        ent = metrics.entropy(vec, tol=0.25)
    except ValueError:
        # A distribution that does not sum to 1 within a quarter is not a
        # distribution; the sum is recorded so the report can say what came back
        # instead of quietly reporting an entropy computed from a renormalised
        # vector.
        ent = None

    chosen = answer.get("chosen")
    shot = {
        "instance_id": result.call.instance_id,
        "item_index": result.call.meta.get("item_index"),
        "n_options": n,
        "truth": truth_id,
        "chosen": chosen,
        "correct": chosen == truth_id,
        "p_truth": vec[truth_ix],
        "max_p": max(vec) if vec else 0.0,
        "confidence": answer.get("confidence"),
        "top5": _topk_expectation(vec, truth_ix, 5),
        "entropy": ent,
        "entropy_ratio": (ent / math.log(n)) if (ent is not None and n > 1) else None,
        # The ceiling a 0.01 grid actually permits: at most 100 non-zero
        # entries, so at 255 options ln(255) is not a reachable maximum.
        "entropy_ratio_achievable": (
            ent / math.log(min(n, int(round(1 / PROB_GRID))))
            if (ent is not None and min(n, int(round(1 / PROB_GRID))) > 1)
            else None
        ),
        "prob_sum": total,
        "support": support,
        "quantised": quantised,
        "correct_position": result.call.meta.get("correct_position"),
        "chosen_position": option_ids.index(chosen) if chosen in option_ids else None,
        "latency_s": result.latency_s,
        "input_tokens": result.input_tokens,
        "distribution": vec,
    }
    if chosen in option_ids:
        try:
            shot["error_distance"] = tx.tree_distance(tx.path_of(chosen), tx.path_of(truth_id))
        except ValueError:
            shot["error_distance"] = None
    if extra:
        shot.update(extra)
    return shot


def _safe_percentiles(values: Sequence[float]) -> dict | None:
    return metrics.percentiles(values) if values else None


def _mean(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _ece_of(probs: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    if len(probs) < 2:
        return None
    return metrics.ece(list(probs), list(outcomes))


def _aggregate(shots: Sequence[dict], *, overlap: Sequence[float] = (),
               lookup: Sequence[float] = ()) -> dict:
    """Everything a condition reports, guarded for an empty or tiny sample."""
    n = len(shots)
    if n == 0:
        return {"n": 0, "note": "every call in this condition failed or was excluded"}
    hits = sum(1 for s in shots if s["correct"])
    lo, hi = metrics.wilson_interval(hits, n)
    max_ps = [s["max_p"] for s in shots]
    correct = [s["correct"] for s in shots]
    confidences = [s["confidence"] for s in shots if s["confidence"] is not None]
    out: dict[str, Any] = {
        "n": n,
        "top1": hits / n,
        "top1_ci95": [lo, hi],
        "top5": _mean([s["top5"] for s in shots]),
        "chance": _mean([1.0 / s["n_options"] for s in shots]),
        "majority_class": metrics.majority_baseline([s["truth"] for s in shots]),
        "entropy_nats": _mean([s["entropy"] for s in shots]),
        "entropy_ratio": _mean([s["entropy_ratio"] for s in shots]),
        "entropy_ratio_achievable": _mean([s["entropy_ratio_achievable"] for s in shots]),
        "mean_max_p": _mean(max_ps),
        "mean_p_truth": _mean([s["p_truth"] for s in shots]),
        "mean_confidence": _mean(confidences) if confidences else None,
        "ece_max_p": _ece_of(max_ps, correct),
        "ece_confidence": (
            _ece_of(confidences, [s["correct"] for s in shots if s["confidence"] is not None])
            if confidences
            else None
        ),
        "mean_prob_sum": _mean([s["prob_sum"] for s in shots]),
        # An arm can hand in a shot with no distribution at all -- the two-stage
        # filter losing the truth is a real outcome with no stage-2 vector behind
        # it -- so these skip the shots that have no support to report rather
        # than reading a missing distribution as an unquantised one.
        "mean_support": _mean(
            [float(s["support"]) for s in shots if s.get("support") is not None]
        ),
        "fraction_on_001_grid": _mean(
            [1.0 if s["quantised"] else 0.0 for s in shots
             if s.get("quantised") is not None]
        ),
        "latency_s": _safe_percentiles([s["latency_s"] for s in shots]),
        "mean_input_tokens": _mean([float(s["input_tokens"]) for s in shots]),
        "cost_per_decision_usd": metrics.cost_per_decision(
            _mean([float(s["input_tokens"]) for s in shots]) or 0.0, 1
        ),
    }
    if overlap:
        out["overlap_baseline"] = _mean(list(overlap))
    if lookup:
        # The rubric's "below a cheap deterministic baseline" signature is
        # evaluated against this one, since it is the stronger of the two.
        out["heuristic_baseline"] = _mean(list(lookup))
    if n >= MIN_N_FOR_ECE:
        bins = metrics.reliability_bins(max_ps, correct)
        out["reliability_monotone"] = metrics.is_monotone(bins)
        out["reliability_bins"] = bins
    if n < MIN_N_FOR_RATE:
        out["small_sample"] = (
            f"n={n}: rates here carry a 95% interval of roughly "
            f"{hi - lo:.2f} and should not be read as a measurement"
        )
    if n < MIN_N_FOR_ECE:
        out["ece_small_sample"] = (
            f"n={n}: ten equal-width bins over {n} predictions leaves most bins "
            "empty or nearly so, so ECE here is not stable"
        )
    return out


# --------------------------------------------------------------------------
# Failure bookkeeping
# --------------------------------------------------------------------------


def _new_ledger() -> dict:
    return {"calls": 0, "failed": 0, "excluded": 0, "reasons": Counter()}


def _reason(r: CallResult) -> str:
    if r.http_status and r.http_status != 200:
        return f"HTTP {r.http_status}"
    return (r.error or "unknown").split("\n")[0][:80]


def _tally(ledger: dict, results: Sequence[CallResult], condition: str) -> None:
    for r in results:
        ledger["calls"] += 1
        if not r.ok:
            ledger["failed"] += 1
            ledger["reasons"][f"{condition}: {_reason(r)}"] += 1


def _exclude(ledger: dict, count: int, why: str) -> None:
    if count:
        ledger["excluded"] += count
        ledger["reasons"][why] += count


# --------------------------------------------------------------------------
# Condition 1 -- knee-finding
# --------------------------------------------------------------------------


def _run_knee(client: JevClient, seed: int, n: int, ledger: dict) -> dict:
    per_k: dict[int, dict] = {}
    for k in tx.CARDINALITIES:
        difficulty = {**tx.DEFAULT_DIFFICULTY, "n_options": k}
        insts = tx.generate(difficulty=difficulty, seed=seed, count=n)
        results = client.run(_calls(insts, f"knee-{k}"), progress_every=100,
                             label=f"E7/knee-{k}")
        _tally(ledger, results, f"knee-{k}")
        by_id = {i.instance_id: i for i in insts}
        shots, overlap, lookup = [], [], []
        for r in results:
            inst = by_id[r.call.instance_id]
            s = _shot(
                r,
                key=tx.KEY_CHOICE,
                option_ids=inst.meta["option_ids"],
                truth_id=inst.truth[tx.KEY_CHOICE],
                extra={"distractor_profile": inst.meta["distractor_profile"]},
            )
            if s is None:
                continue
            s["nearest_distractor"] = min(
                (int(d) for d in inst.meta["distractor_distance_counts"]),
                default=len(tx.LEVELS),
            )
            shots.append(s)
            overlap.append(inst.meta["baseline_overlap_expectation"])
            lookup.append(inst.meta["baseline_lookup_expectation"])
        _exclude(ledger, len(insts) - len(shots), f"knee-{k}: no usable answer")
        per_k[k] = {"instances": insts, "shots": shots,
                    "agg": _aggregate(shots, overlap=overlap, lookup=lookup)}
    return per_k


# --------------------------------------------------------------------------
# Condition 2 -- distractor quality at a fixed 64 options
# --------------------------------------------------------------------------


def _run_distractors(client: JevClient, seed: int, n: int, ledger: dict,
                     uniform_cell: dict) -> dict:
    """The similarity ladder at 64 options.

    `uniform` is condition 1's 64-option cell rather than a fresh draw: it is
    the same items with the same seed and the same difficulty dict, so rerunning
    it would spend 300 calls to reproduce numbers already paid for, and any
    difference between the two would be sampling noise reported as an effect.
    """
    out: dict[str, dict] = {"uniform": uniform_cell}
    for profile in SIMILARITY_LADDER:
        if profile == "uniform":
            continue
        difficulty = {
            **tx.DEFAULT_DIFFICULTY,
            "n_options": DISTRACTOR_CARDINALITY,
            "distractor_profile": profile,
        }
        insts = tx.generate(difficulty=difficulty, seed=seed, count=n)
        results = client.run(_calls(insts, f"distractor-{profile}"), progress_every=100,
                             label=f"E7/distractor-{profile}")
        _tally(ledger, results, f"distractor-{profile}")
        by_id = {i.instance_id: i for i in insts}
        shots, overlap, lookup = [], [], []
        for r in results:
            inst = by_id[r.call.instance_id]
            s = _shot(r, key=tx.KEY_CHOICE, option_ids=inst.meta["option_ids"],
                      truth_id=inst.truth[tx.KEY_CHOICE],
                      extra={"distractor_profile": profile})
            if s is None:
                continue
            shots.append(s)
            overlap.append(inst.meta["baseline_overlap_expectation"])
            lookup.append(inst.meta["baseline_lookup_expectation"])
        _exclude(ledger, len(insts) - len(shots), f"distractor-{profile}: no usable answer")
        out[profile] = {"instances": insts, "shots": shots,
                        "agg": _aggregate(shots, overlap=overlap, lookup=lookup)}
    return out


# --------------------------------------------------------------------------
# Condition 3 -- two-stage against flat, both at 255
# --------------------------------------------------------------------------


def _run_two_stage(client: JevClient, seed: int, ledger: dict, flat_cell: dict) -> dict:
    """Filter with a noul per candidate, then choose over the survivors.

    The filter's questions are shuffled within their batch and the candidates
    are shuffled across batches, so neither the taxonomy's order nor a
    within-call position effect lines up with which candidate is correct.
    """
    insts: list[Instance] = flat_cell["instances"]
    stage1: list[Call] = []
    tie_keys: dict[int, dict[str, float]] = {}

    for inst in insts:
        options = inst.questions[tx.KEY_CHOICE]["options"]
        truth_id = inst.truth[tx.KEY_CHOICE]
        rng = rng_for(f"{EXPERIMENT}:two-stage", {}, seed, inst.index)
        order = list(options)
        rng.shuffle(order)
        # A per-candidate random key, drawn once, breaks probability ties in the
        # filter. Without it the survivors of a tie would be whichever candidate
        # the taxonomy happened to enumerate first.
        tie_keys[inst.index] = {o["id"]: rng.random() for o in options}
        for b in range(0, len(order), STAGE1_BATCH):
            batch = order[b:b + STAGE1_BATCH]
            mapping = {f"c{j}": o["id"] for j, o in enumerate(batch)}
            questions = {
                f"c{j}": noul(tx.CANDIDATE_QUESTION.format(label=o["label"]))
                for j, o in enumerate(batch)
            }
            stage1.append(
                Call(
                    experiment=EXPERIMENT,
                    condition="two-stage-filter",
                    instance_id=inst.instance_id,
                    state=inst.state,
                    questions=shuffled_questions(questions, rng),
                    meta={
                        "item_index": inst.index,
                        "batch": b // STAGE1_BATCH,
                        "candidate_by_key": mapping,
                        "leaf_id": truth_id,
                        "truth": {k: (v == truth_id) for k, v in mapping.items()},
                    },
                )
            )

    results = client.run(stage1, progress_every=100, label="E7/two-stage-filter")
    _tally(ledger, results, "two-stage-filter")

    filter_p: dict[int, dict[str, float]] = defaultdict(dict)
    broken: set[int] = set()
    filter_latency: dict[int, list[float]] = defaultdict(list)
    filter_tokens: dict[int, int] = defaultdict(int)
    for r in results:
        idx = r.call.meta["item_index"]
        if not r.ok:
            broken.add(idx)
            continue
        filter_latency[idx].append(r.latency_s)
        filter_tokens[idx] += r.input_tokens
        for qkey, answer in r.answers.items():
            candidate = r.call.meta["candidate_by_key"].get(qkey)
            if candidate is not None and answer.get("type") == "noul":
                filter_p[idx][candidate] = answer["p"]

    stage2: list[Call] = []
    survivors_by_index: dict[int, list[dict]] = {}
    for inst in insts:
        idx = inst.index
        options = inst.questions[tx.KEY_CHOICE]["options"]
        if idx in broken or len(filter_p[idx]) != len(options):
            broken.add(idx)
            continue
        keys = tie_keys[idx]
        ranked = sorted(options, key=lambda o: (-filter_p[idx][o["id"]], keys[o["id"]]))
        survivors = ranked[:STAGE2_SURVIVORS]
        survivors_by_index[idx] = survivors
        rng = rng_for(f"{EXPERIMENT}:two-stage-pick", {}, seed, idx)
        q = shuffled_options(choice(tx.LEAF_QUESTION, survivors), rng)
        stage2.append(
            Call(
                experiment=EXPERIMENT,
                condition="two-stage-pick",
                instance_id=inst.instance_id,
                state=inst.state,
                questions={tx.KEY_CHOICE: q},
                meta={
                    "item_index": idx,
                    "truth": inst.truth,
                    "option_ids": [o["id"] for o in q["options"]],
                    "survivor_ids": [o["id"] for o in survivors],
                    "filter_recall": inst.truth[tx.KEY_CHOICE] in {o["id"] for o in survivors},
                    "filter_p_truth": filter_p[idx].get(inst.truth[tx.KEY_CHOICE]),
                    "stage": 2,
                },
            )
        )
    _exclude(ledger, len(broken), "two-stage: filter incomplete for this item")

    pick_results = client.run(stage2, progress_every=100, label="E7/two-stage-pick")
    _tally(ledger, pick_results, "two-stage-pick")

    shots: list[dict] = []
    lost_pick = 0
    by_index = {i.index: i for i in insts}
    for r in pick_results:
        idx = r.call.meta["item_index"]
        inst = by_index[idx]
        truth_id = inst.truth[tx.KEY_CHOICE]
        survivor_ids = r.call.meta["survivor_ids"]
        recall = r.call.meta["filter_recall"]
        if not r.ok:
            lost_pick += 1
            continue
        # The truth may not be among the survivors at all, in which case there
        # is no truth index in the stage-2 distribution. That is a real result
        # for the pattern -- the filter lost it -- not a failed call, so it is
        # scored as wrong rather than excluded.
        if not recall:
            answer = r.answers.get(tx.KEY_CHOICE) or {}
            shots.append({
                "instance_id": r.call.instance_id, "item_index": idx,
                "n_options": len(survivor_ids), "truth": truth_id,
                "chosen": answer.get("chosen"), "correct": False,
                "p_truth": 0.0, "max_p": answer.get("max_probability") or 0.0,
                "confidence": answer.get("confidence"), "top5": 0.0,
                "entropy": None, "entropy_ratio": None,
                "entropy_ratio_achievable": None, "prob_sum": None,
                "support": None, "quantised": None, "correct_position": None,
                "chosen_position": None, "latency_s": r.latency_s,
                "input_tokens": r.input_tokens, "distribution": [],
                "filter_recall": False,
            })
            continue
        s = _shot(r, key=tx.KEY_CHOICE, option_ids=r.call.meta["option_ids"],
                  truth_id=truth_id, extra={"filter_recall": True})
        if s is None:
            lost_pick += 1
            continue
        shots.append(s)
    # A pick that failed leaves the two-stage arm entirely, so it has to appear
    # in the ledger: otherwise the arm's denominator shrinks with nothing in the
    # failure report to account for it.
    _exclude(ledger, lost_pick, "two-stage: the pick call had no usable answer")

    totals = []
    for s in shots:
        idx = s["item_index"]
        lats = filter_latency.get(idx) or [0.0]
        totals.append({
            "item_index": idx,
            "calls": len(lats) + 1,
            "input_tokens": filter_tokens.get(idx, 0) + s["input_tokens"],
            "latency_serial_s": sum(lats) + s["latency_s"],
            # The filter batches are independent, so a deployment would issue
            # them together; only the pick has to wait. Both numbers are
            # reported because which one applies depends on the deployment.
            "latency_parallel_s": max(lats) + s["latency_s"],
        })

    agg = _aggregate(shots)
    recalls = [s.get("filter_recall") for s in shots if s.get("filter_recall") is not None]
    return {
        "shots": shots,
        "agg": agg,
        "filter_recall_at_16": (sum(1 for x in recalls if x) / len(recalls)) if recalls else None,
        "survivors": STAGE2_SURVIVORS,
        "filter_batch_size": STAGE1_BATCH,
        "per_item_cost": {
            "calls": _mean([float(t["calls"]) for t in totals]),
            "input_tokens": _mean([float(t["input_tokens"]) for t in totals]),
            "latency_serial_s": _safe_percentiles([t["latency_serial_s"] for t in totals]),
            "latency_parallel_s": _safe_percentiles([t["latency_parallel_s"] for t in totals]),
            "cost_usd": (
                (_mean([float(t["input_tokens"]) for t in totals]) or 0.0)
                * config.USD_PER_INPUT_TOKEN
            ),
        },
        "excluded_items": len(broken) + lost_pick,
        "excluded_filter_incomplete": len(broken),
        "excluded_pick_failed": lost_pick,
    }


# --------------------------------------------------------------------------
# Condition 4 -- hierarchical descent
# --------------------------------------------------------------------------


def _descend(
    client: JevClient,
    seed: int,
    items_by_index: dict[int, tx.Item],
    prefixes: dict[int, tuple],
    *,
    start_level: int,
    condition: str,
    tag: str,
    ledger: dict,
) -> dict[int, dict]:
    """Run the levels below `start_level` as a sequence of adaptive batches.

    Each level is one batch of concurrent calls whose option set depends on what
    the level above chose, so the levels are sequential and the items within a
    level are not. The same function runs the free-running descent and the
    second beam branch, which is what makes the two comparable: a difference
    between them cannot come from a difference in how they were asked.
    """
    state = {
        idx: {"prefix": tuple(p), "levels": [], "alive": True, "failed_at": None}
        for idx, p in prefixes.items()
    }
    for level in range(start_level, len(tx.LEVELS)):
        key = f"q_level{level}"
        calls: list[Call] = []
        for idx, st in state.items():
            if not st["alive"]:
                continue
            prefix = st["prefix"]
            item = items_by_index[idx]
            parent_label = tx.facet_label(len(prefix) - 1, prefix) if prefix else None
            rng = rng_for(f"{EXPERIMENT}:{tag}", {"level": level}, seed, idx)
            q = shuffled_options(
                choice(tx.level_question(level, parent_label), tx.children(prefix)), rng
            )
            option_ids = [o["id"] for o in q["options"]]
            prefix_correct = prefix == tuple(item.path[:level])
            true_child = (
                tx.node_id((*item.path[:level], item.path[level])) if prefix_correct else None
            )
            calls.append(
                Call(
                    experiment=EXPERIMENT,
                    condition=condition,
                    instance_id=f"{tag}-{idx}-L{level}",
                    state=item.state,
                    questions={key: q},
                    meta={
                        "item_index": idx,
                        "level": level,
                        "level_name": tx.LEVELS[level],
                        "prefix": list(prefix),
                        "prefix_correct": prefix_correct,
                        "option_ids": option_ids,
                        "correct_position": (
                            option_ids.index(true_child) if true_child else None
                        ),
                        # None where the prefix is already wrong: at that point
                        # no option in this call is correct, which is the whole
                        # point of the compounding measurement.
                        "truth": {key: true_child},
                        "leaf_id": item.leaf_id,
                    },
                )
            )
        if not calls:
            break
        results = client.run(calls, progress_every=100, label=f"E7/{tag}-L{level}")
        _tally(ledger, results, f"{condition}-L{level}")
        for r in results:
            idx = r.call.meta["item_index"]
            st = state[idx]
            if not r.ok:
                st["alive"] = False
                st["failed_at"] = level
                continue
            answer = r.answers.get(key) or {}
            option_ids = r.call.meta["option_ids"]
            chosen = answer.get("chosen")
            if chosen not in option_ids:
                st["alive"] = False
                st["failed_at"] = level
                continue
            probs = {oid: float((answer.get("probabilities") or {}).get(oid, 0.0))
                     for oid in option_ids}
            true_child = r.call.meta["truth"][key]
            ranked = sorted(option_ids, key=lambda o: -probs[o])
            st["levels"].append({
                "level": level,
                "prefix_correct": r.call.meta["prefix_correct"],
                "chosen": chosen,
                "true_child": true_child,
                "correct": (true_child is not None and chosen == true_child),
                "p_chosen": probs[chosen],
                "p_true_child": probs.get(true_child) if true_child else None,
                "rank_true_child": (
                    ranked.index(true_child) + 1 if true_child in probs else None
                ),
                "runner_up": ranked[1] if len(ranked) > 1 else None,
                "p_runner_up": probs[ranked[1]] if len(ranked) > 1 else None,
                "confidence": answer.get("confidence"),
                # Kept so the confidence fit can use descent answers too: they
                # are the only choice answers in E7 at cardinalities below 8.
                "distribution": [probs[o] for o in option_ids],
                "latency_s": r.latency_s,
                "input_tokens": r.input_tokens,
            })
            st["prefix"] = tx.path_of(chosen)
    return state


def _shortlist_calls(items: Sequence[tx.Item], seed: int) -> tuple[list[Call], dict[int, dict]]:
    calls, aux = [], {}
    for item in items:
        rng = rng_for(f"{EXPERIMENT}:shortlist", {}, seed, item.index)
        paths = tx.shortlist_paths(item.path, size=255, rng=rng)
        q = shuffled_options(choice(tx.LEAF_QUESTION, tx.options_for(paths)), rng)
        option_ids = [o["id"] for o in q["options"]]
        shown = [tx.path_of(o) for o in option_ids]
        base = tx.overlap_baseline(item.tokens, shown, item.path)
        lookup = tx.lookup_baseline(item.path, shown)
        composition = Counter(tx.tree_distance(p, item.path) for p in shown if p != item.path)
        meta = {
            "item_index": item.index,
            "truth": {tx.KEY_CHOICE: item.leaf_id},
            "option_ids": option_ids,
            "correct_position": option_ids.index(item.leaf_id),
            "n_options": len(option_ids),
            "shortlist": "simulated retriever, enriched toward the true subtree",
            "distractor_distance_counts": {str(k): v for k, v in sorted(composition.items())},
            "baseline_overlap_expectation": base["expectation"],
            "baseline_lookup_expectation": lookup["expectation"],
            **item.meta(),
        }
        aux[item.index] = meta
        calls.append(
            Call(
                experiment=EXPERIMENT,
                condition="flat-shortlist-255",
                instance_id=f"shortlist-{item.index}",
                state=item.state,
                questions={tx.KEY_CHOICE: q},
                meta=meta,
            )
        )
    return calls, aux


def _run_hierarchical(client: JevClient, seed: int, items: Sequence[tx.Item],
                      ledger: dict) -> dict:
    by_index = {it.index: it for it in items}
    descent = _descend(
        client, seed, by_index, {it.index: () for it in items},
        start_level=0, condition="descent", tag="descent", ledger=ledger,
    )

    complete = {
        idx: st for idx, st in descent.items()
        if st["alive"] and len(st["levels"]) == len(tx.LEVELS)
    }
    _exclude(ledger, len(descent) - len(complete), "descent: a level had no usable answer")

    leaf_correct = {idx: st["prefix"] == by_index[idx].path for idx, st in complete.items()}
    n_complete = len(complete)

    # Per-level conditional accuracy, on the items whose prefix was still right
    # when that level was asked. This is the quantity the compounding claim is
    # about: a level's own difficulty, with the levels above it held correct.
    per_level = []
    for level in range(len(tx.LEVELS)):
        rows = [st["levels"][level] for st in complete.values() if st["levels"][level]["prefix_correct"]]
        hits = sum(1 for r in rows if r["correct"])
        lo, hi = metrics.wilson_interval(hits, len(rows))
        per_level.append({
            "level": level,
            "name": tx.LEVELS[level],
            "n_options": tx.LEVEL_WIDTHS[level],
            "n_reached_with_correct_prefix": len(rows),
            "conditional_accuracy": (hits / len(rows)) if rows else None,
            "ci95": [lo, hi],
            "chance": 1.0 / tx.LEVEL_WIDTHS[level],
            "mean_p_chosen": _mean([r["p_chosen"] for r in rows]),
            "mean_confidence": _mean([r["confidence"] for r in rows]),
            "latency_s": _safe_percentiles([r["latency_s"] for r in rows]),
        })

    first_error = Counter()
    for idx, st in complete.items():
        where = next((r["level"] for r in st["levels"] if r["prefix_correct"] and not r["correct"]),
                     None)
        first_error["none" if where is None else f"level{where}"] += 1

    product = 1.0
    known = True
    for row in per_level:
        if row["conditional_accuracy"] is None:
            known = False
            break
        product *= row["conditional_accuracy"]
    observed = (sum(1 for v in leaf_correct.values() if v) / n_complete) if n_complete else None

    recovery = _run_recovery(client, seed, by_index, complete, ledger)

    shortlist_calls, shortlist_meta = _shortlist_calls(items, seed)
    shortlist_results = client.run(shortlist_calls, progress_every=100,
                                   label="E7/flat-shortlist-255")
    _tally(ledger, shortlist_results, "flat-shortlist-255")
    flat_shots, flat_overlap, flat_lookup = [], [], []
    for r in shortlist_results:
        meta = r.call.meta
        s = _shot(r, key=tx.KEY_CHOICE, option_ids=meta["option_ids"],
                  truth_id=meta["truth"][tx.KEY_CHOICE], extra={"shortlist": True})
        if s is None:
            continue
        flat_shots.append(s)
        flat_overlap.append(meta["baseline_overlap_expectation"])
        flat_lookup.append(meta["baseline_lookup_expectation"])
    _exclude(ledger, len(shortlist_calls) - len(flat_shots),
             "flat-shortlist-255: no usable answer")

    descent_tokens = _mean([
        float(sum(r["input_tokens"] for r in st["levels"])) for st in complete.values()
    ]) or 0.0
    descent_latency = [sum(r["latency_s"] for r in st["levels"]) for st in complete.values()]

    lo, hi = metrics.wilson_interval(sum(1 for v in leaf_correct.values() if v), n_complete)
    return {
        "descent": {
            "n": n_complete,
            "leaf_accuracy": observed,
            "ci95": [lo, hi],
            "chance": 1.0 / tx.N_LEAVES,
            "calls_per_item": len(tx.LEVELS),
            "mean_input_tokens": descent_tokens,
            "cost_usd_per_item": descent_tokens * config.USD_PER_INPUT_TOKEN,
            "latency_serial_s": _safe_percentiles(descent_latency),
            "per_level": per_level,
            "first_error_level": dict(first_error),
            "compounding": {
                "product_of_conditional_accuracies": product if known else None,
                "observed_leaf_accuracy": observed,
                "gap": (observed - product) if (known and observed is not None) else None,
                "note": (
                    "In a strict tree the two are equal by construction: a wrong turn "
                    "puts the correct leaf outside every later option set, so errors "
                    "multiply. A gap materially above zero would mean the descent is "
                    "not behaving like a partition and should be investigated."
                ),
            },
        },
        "recovery": recovery,
        "flat_shortlist": {
            "n": len(flat_shots),
            **_aggregate(flat_shots, overlap=flat_overlap, lookup=flat_lookup),
            "shortlist": "simulated retriever, enriched toward the true subtree",
            "composition_example": (
                shortlist_meta[items[0].index]["distractor_distance_counts"] if items else {}
            ),
        },
        "_flat_shots": flat_shots,
        "_leaf_correct": leaf_correct,
        "_level_rows": [row for st in complete.values() for row in st["levels"]],
    }


def _run_recovery(client: JevClient, seed: int, by_index: dict[int, tx.Item],
                  complete: dict[int, dict], ledger: dict) -> dict:
    """Can a descent that turns wrong at the root get back to the right leaf?

    Under a strict top-1 descent it cannot, and that number is 0 by
    construction rather than by measurement, so reporting only it would make
    P21 unfalsifiable. The measured quantity is a width-2 beam at the root: keep
    the runner-up department from the level-0 distribution -- which costs no
    extra call at the level where the error happened -- descend it, and let the
    two branches arbitrate on the product of their chosen probabilities.

    Items whose runner-up is also the wrong department cannot be recovered by a
    width-2 beam whatever the deeper levels say, so they count as not recovered
    and no calls are spent on them.
    """
    wrong_root = [
        idx for idx, st in complete.items()
        if st["levels"][0]["prefix_correct"] and not st["levels"][0]["correct"]
    ]
    if not wrong_root:
        return {
            "n_wrong_root": 0,
            "strict_top1_recovery": 0.0,
            "beam2_recovery": None,
            "note": "no descent took a wrong turn at the root, so recovery is untestable here",
        }

    true_domain_is_runner_up = []
    rank_of_true = Counter()
    for idx in wrong_root:
        row = complete[idx]["levels"][0]
        rank_of_true[row["rank_true_child"]] += 1
        if row["runner_up"] == row["true_child"]:
            true_domain_is_runner_up.append(idx)

    beam_state: dict[int, dict] = {}
    if true_domain_is_runner_up:
        beam_state = _descend(
            client, seed, by_index,
            {idx: tx.path_of(complete[idx]["levels"][0]["runner_up"])
             for idx in true_domain_is_runner_up},
            start_level=1, condition="descent-beam2", tag="beam2", ledger=ledger,
        )

    arbitrated = oracle = 0
    usable = 0
    unmeasured = 0
    for idx in true_domain_is_runner_up:
        st = beam_state.get(idx)
        if st is None or not st["alive"] or len(st["levels"]) != len(tx.LEVELS) - 1:
            # The runner-up branch was the true one but its descent lost a call,
            # so this item has no outcome. Counting it as "not recovered" would
            # push the rate down with a failure, which is exactly what the plan
            # forbids; it leaves the sample instead.
            unmeasured += 1
            continue
        usable += 1
        truth = by_index[idx].path
        branch2_leaf = st["prefix"]
        branch1 = complete[idx]["levels"]
        s1 = 1.0
        for row in branch1:
            s1 *= row["p_chosen"]
        s2 = branch1[0]["p_runner_up"] or 0.0
        for row in st["levels"]:
            s2 *= row["p_chosen"]
        if branch2_leaf == truth:
            oracle += 1
        winner = branch2_leaf if s2 > s1 else complete[idx]["prefix"]
        if winner == truth:
            arbitrated += 1

    n_wrong = len(wrong_root)
    _exclude(ledger, unmeasured, "descent-beam2: the recovery branch lost a call")
    # Items whose runner-up was the wrong department need no call: no width-2
    # beam can reach the truth through either branch, so they are measured
    # outcomes, not gaps. Only the failed descents leave the denominator.
    measured = n_wrong - unmeasured
    return {
        "n_wrong_root": n_wrong,
        "n_measured": measured,
        "n_unmeasured_call_failed": unmeasured,
        "strict_top1_recovery": 0.0,
        "strict_note": (
            "0 by construction, not by measurement: the tree is a strict "
            "partition, so after a wrong turn the correct leaf is absent from "
            "every later option set"
        ),
        "true_department_was_runner_up": len(true_domain_is_runner_up) / n_wrong,
        "rank_of_true_department_when_wrong": {
            str(k): v for k, v in sorted(
                rank_of_true.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)
            )
        },
        "beam2_descended": usable,
        "beam2_recovery": (arbitrated / measured) if measured else None,
        "beam2_recovery_oracle_branch_choice": (oracle / measured) if measured else None,
        "beam2_extra_calls_per_recovered_item": (
            (usable * (len(tx.LEVELS) - 1) / arbitrated) if arbitrated else None
        ),
        "note": (
            "beam2_recovery is the fraction of wrong-root descents that end at the "
            "correct leaf once the runner-up department is kept and the two branches "
            "arbitrate on the product of their chosen probabilities. The oracle "
            "variant is the same with the branch chosen correctly at the end, and is "
            "the ceiling a width-2 beam can reach."
        ),
    }


# --------------------------------------------------------------------------
# Condition 5 -- distribution shape
# --------------------------------------------------------------------------


def _score_shot(r: CallResult, rubric: Sequence[dict], truth_id: str,
                date_cue: str) -> dict | None:
    if not r.ok:
        return None
    answer = r.answers.get(tx.KEY_SCORE)
    if answer is None or answer.get("type") != "score":
        return None
    k = len(rubric)
    idx_probs = answer.get("index_probabilities") or {}
    if answer.get("chosen") is None:
        # `chosen` comes from the returned distribution, so an answer without one
        # names no rubric point. Scoring that as a miss would default a call the
        # endpoint did not really answer; it leaves the sample and is counted.
        return None
    vec = [float(idx_probs.get(i, 0.0)) for i in range(k)]
    truth_ix = int(truth_id[1:])
    chosen = answer.get("chosen")
    chosen_ix = (
        int(chosen[1:]) if isinstance(chosen, str) and chosen[1:].isdigit() else None
    )
    try:
        ent = metrics.entropy(vec, tol=0.25)
    except ValueError:
        ent = None
    raw = answer.get("score")
    return {
        "item_index": r.call.meta.get("item_index"),
        "rubric_size": k,
        "date_cue": date_cue,
        "truth": truth_id,
        "chosen": chosen,
        "correct": chosen == truth_id,
        "within_one_band": (chosen_ix is not None and abs(chosen_ix - truth_ix) <= 1),
        "band_error": (abs(chosen_ix - truth_ix) if chosen_ix is not None else None),
        "continuous_score_error": (abs(float(raw) - truth_ix) if raw is not None else None),
        "max_p": max(vec) if vec else 0.0,
        "p_truth": vec[truth_ix] if 0 <= truth_ix < k else 0.0,
        "confidence": answer.get("confidence"),
        "entropy": ent,
        "entropy_ratio": (ent / math.log(k)) if (ent is not None and k > 1) else None,
        "prob_sum": sum(vec),
        "support": sum(1 for v in vec if v > 0.0),
        "quantised": _on_grid([v for v in vec if v > 0.0]),
        "has_legend": answer.get("legend") is not None,
        "latency_s": r.latency_s,
        "distribution": vec,
    }


def _probe_score_cap(client: JevClient, seed: int, ledger: dict) -> dict:
    """One call at 20 rubric points, to record the cap rather than assume it.

    The endpoint answered "Too many score levels. Must have at most 10 levels."
    when this module was written. The probe costs one call and keeps that fact a
    measurement in this run's log instead of a comment that may have gone stale.
    """
    insts = tx.generate(
        difficulty={"question_type": "score", "rubric_size": tx.RUBRIC_CAP_PROBE},
        seed=seed,
        count=1,
    )
    results = client.run(_calls(insts, "score-cap-probe"), progress_every=0,
                         label="E7/score-cap-probe")
    _tally(ledger, results, f"score-cap-probe at {tx.RUBRIC_CAP_PROBE} levels (expected 400)")
    r = results[0] if results else None
    return {
        "levels_requested": tx.RUBRIC_CAP_PROBE,
        "accepted": bool(r and r.ok),
        "http_status": r.http_status if r else None,
        "error": (r.error or "")[:200] if r else None,
        "documented_cap_used": tx.MAX_SCORE_LEVELS,
    }


def _run_shape(client: JevClient, seed: int, n: int, ledger: dict) -> dict:
    """A `score` sweep over 2 to 100 ordered date bands.

    Rubric length is the cardinality axis for `score`, and half the items carry
    a vague date so the sample spans genuinely uncertain answers as well as
    determinate ones -- a fit of confidence against the distribution needs
    spread in the distribution to fit anything.
    """
    per_k: dict[int, dict] = {}
    for k in tx.RUBRIC_SIZES:
        difficulty = {"question_type": "score", "rubric_size": k, "date_cue": "mixed"}
        insts = tx.generate(difficulty=difficulty, seed=seed, count=n)
        results = client.run(_calls(insts, f"score-{k}"), progress_every=100,
                             label=f"E7/score-{k}")
        _tally(ledger, results, f"score-{k}")
        by_id = {i.instance_id: i for i in insts}
        shots = []
        for r in results:
            inst = by_id[r.call.instance_id]
            s = _score_shot(r, inst.questions[tx.KEY_SCORE]["rubric"],
                            inst.truth[tx.KEY_SCORE], inst.meta["date_cue"])
            if s is not None:
                shots.append(s)
        _exclude(ledger, len(insts) - len(shots), f"score-{k}: no usable answer")
        hits = sum(1 for s in shots if s["correct"])
        lo, hi = metrics.wilson_interval(hits, len(shots))
        per_k[k] = {
            "shots": shots,
            "n": len(shots),
            "accuracy": (hits / len(shots)) if shots else None,
            "ci95": [lo, hi],
            "chance": 1.0 / k,
            "within_one_band": _mean([1.0 if s["within_one_band"] else 0.0 for s in shots]),
            "entropy_nats": _mean([s["entropy"] for s in shots]),
            "entropy_ratio": _mean([s["entropy_ratio"] for s in shots]),
            "mean_max_p": _mean([s["max_p"] for s in shots]),
            "mean_confidence": _mean([s["confidence"] for s in shots]),
            "mean_support": _mean([float(s["support"]) for s in shots]),
            "continuous_score_error": _mean([s["continuous_score_error"] for s in shots]),
            "legend_returned": _mean([1.0 if s["has_legend"] else 0.0 for s in shots]),
            "by_date_cue": {
                cue: {
                    "n": len([s for s in shots if s["date_cue"] == cue]),
                    "accuracy": _mean(
                        [1.0 if s["correct"] else 0.0 for s in shots if s["date_cue"] == cue]
                    ),
                    "mean_max_p": _mean([s["max_p"] for s in shots if s["date_cue"] == cue]),
                    "mean_confidence": _mean(
                        [s["confidence"] for s in shots if s["date_cue"] == cue]
                    ),
                }
                for cue in ("exact", "vague")
            },
        }
        if len(shots) < MIN_N_FOR_RATE:
            # The same guard `_aggregate` puts on a choice condition. A rubric
            # accuracy read off a handful of items is an interval, not a rate,
            # and it is reported as one here rather than in the run's prose.
            per_k[k]["small_sample"] = (
                f"n={len(shots)}: rates here carry a 95% interval of roughly "
                f"{hi - lo:.2f} and should not be read as a measurement"
            )
        if not shots:
            per_k[k]["note"] = (
                f"every call at rubric size {k} failed; the endpoint may not accept "
                "a rubric this long"
            )
    return per_k


# --------------------------------------------------------------------------
# What `confidence` actually is
# --------------------------------------------------------------------------


def _functionals(vec: Sequence[float]) -> dict | None:
    """Candidate closed forms for `confidence`, computed from the distribution."""
    total = sum(vec)
    k = len(vec)
    if total <= 0.0 or k < 2:
        return None
    p = sorted((v / total for v in vec), reverse=True)
    h = -sum(x * math.log(x) for x in p if x > 0.0)
    support = sum(1 for x in p if x > 0.0)
    return {
        "max_p": p[0],
        "margin_over_runner_up": p[0] - p[1],
        "collision_probability": sum(x * x for x in p),
        "normalised_negentropy": 1.0 - h / math.log(k),
        "chance_corrected_max_p": (p[0] - 1.0 / k) / (1.0 - 1.0 / k),
        "max_p_squared": p[0] * p[0],
        "one_over_support": 1.0 / support,
    }


def _fit_one(xs: Sequence[float], ys: Sequence[float]) -> dict:
    import numpy as np

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    out: dict[str, Any] = {
        "n": int(x.size),
        "identity_mae": float(np.mean(np.abs(y - x))),
        "identity_exact_rate": float(np.mean(np.abs(y - x) <= 0.005)),
    }
    if x.size >= 3 and float(x.std()) > 1e-9 and float(y.std()) > 1e-9:
        out["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
        design = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out["slope"] = float(coef[0])
        out["intercept"] = float(coef[1])
        out["r_squared"] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
        out["linear_mae"] = float(np.mean(np.abs(y - pred)))
    else:
        out["note"] = "too few samples, or no spread in one variable, to fit"
    return out


def _confidence_fit(pairs: Sequence[tuple[float, Sequence[float]]], label: str) -> dict:
    """Fit `confidence` against candidate functionals of the distribution.

    The plan asks whether confidence diverges from max-probability and says the
    divergence would be a usable signal if it means something. Reporting that
    the two differ is not an answer, so this fits a small family of closed forms
    a vendor might plausibly be computing and reports which one tracks it.
    """
    rows: dict[str, list[float]] = defaultdict(list)
    conf: list[float] = []
    for confidence, vec in pairs:
        f = _functionals(vec)
        if f is None or confidence is None:
            continue
        conf.append(float(confidence))
        for name, value in f.items():
            rows[name].append(value)
    if len(conf) < 3:
        return {"n": len(conf), "note": f"{label}: too few usable answers to fit"}
    fits = {name: _fit_one(values, conf) for name, values in rows.items()}
    best_identity = min(fits, key=lambda k: fits[k]["identity_mae"])
    with_r2 = {k: v for k, v in fits.items() if v.get("r_squared") is not None}
    best_linear = max(with_r2, key=lambda k: with_r2[k]["r_squared"]) if with_r2 else None
    return {
        "n": len(conf),
        "mean_confidence": sum(conf) / len(conf),
        "fits": fits,
        "best_as_identity": best_identity,
        "best_identity_mae": fits[best_identity]["identity_mae"],
        "best_linear": best_linear,
        "best_linear_r_squared": with_r2[best_linear]["r_squared"] if best_linear else None,
        "confidence_equals_max_probability": (
            fits["max_p"]["identity_exact_rate"] if "max_p" in fits else None
        ),
    }


# --------------------------------------------------------------------------
# Cross-cutting checks the plan asks every experiment to watch for
# --------------------------------------------------------------------------


def _quantisation_facts(shots: Sequence[dict]) -> dict:
    usable = [s for s in shots if s.get("prob_sum")]
    if not usable:
        return {"n": 0}
    grid = sum(1 for s in usable if s.get("quantised")) / len(usable)
    big = [s for s in usable if s["n_options"] >= 128]
    return {
        "n": len(usable),
        "fraction_on_001_grid": grid,
        "mean_prob_sum": _mean([s["prob_sum"] for s in usable]),
        "min_prob_sum": min(s["prob_sum"] for s in usable),
        "max_prob_sum": max(s["prob_sum"] for s in usable),
        "mean_support_at_128_plus": _mean([float(s["support"]) for s in big]) if big else None,
        "max_support_seen": max(int(s["support"]) for s in usable if s["support"] is not None),
    }


def _position_bias(shots: Sequence[dict]) -> dict | None:
    """Accuracy by where the correct option sat, and where wrong picks landed."""
    rows = [s for s in shots if s.get("correct_position") is not None and s["n_options"] > 2]
    if len(rows) < 3 * MIN_N_FOR_RATE:
        return None
    thirds: dict[int, list[bool]] = defaultdict(list)
    for s in rows:
        third = min(2, int(3 * s["correct_position"] / s["n_options"]))
        thirds[third].append(bool(s["correct"]))
    by_third = {}
    for third in (0, 1, 2):
        vals = thirds.get(third, [])
        hits = sum(1 for v in vals if v)
        lo, hi = metrics.wilson_interval(hits, len(vals))
        by_third[["first", "middle", "last"][third]] = {
            "n": len(vals), "accuracy": (hits / len(vals)) if vals else None, "ci95": [lo, hi]
        }
    wrong = [s for s in rows if not s["correct"] and s.get("chosen_position") is not None]
    return {
        "accuracy_by_position_third": by_third,
        "mean_normalised_position_of_wrong_pick": _mean(
            [s["chosen_position"] / max(1, s["n_options"] - 1) for s in wrong]
        ),
        "n_wrong": len(wrong),
        "note": "0.5 is what no position bias looks like for the wrong-pick position",
    }


def _id_bias(shots: Sequence[dict]) -> dict | None:
    """Is any department over-chosen relative to how often it was offered?"""
    rows = [s for s in shots if s.get("chosen") and s["n_options"] >= 64]
    # Eight departments need real mass before a spike means anything: at twenty
    # answers the most-chosen of eight cells is several times the flat rate on
    # noise alone, and this feeds `anomalies`, which the report presents as
    # findings.
    if len(rows) < MIN_N_FOR_ID_BIAS:
        return None
    chosen = Counter()
    for s in rows:
        try:
            chosen[tx.path_of(s["chosen"])[0]] += 1
        except (ValueError, IndexError):
            continue
    if not chosen:
        return None
    total = sum(chosen.values())
    expected = total / tx.N_DEPARTMENTS
    worst = max(chosen.items(), key=lambda kv: abs(kv[1] - expected))
    return {
        "n": total,
        "chosen_department_counts": {tx.DEPARTMENTS[d].id: c for d, c in sorted(chosen.items())},
        "expected_if_flat": expected,
        "largest_deviation": {
            "department": tx.DEPARTMENTS[worst[0]].id,
            "observed": worst[1],
            "ratio_to_expected": worst[1] / expected if expected else None,
        },
        "departments_never_chosen": [
            tx.DEPARTMENTS[d].id for d in range(tx.N_DEPARTMENTS) if d not in chosen
        ],
        "note": (
            "the truth is drawn uniformly over departments, so a flat chooser and "
            "an accurate chooser both produce a flat distribution here; a spike is a "
            "preference for a department rather than for a position"
        ),
    }


# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------


def _sweep_metrics(k: int, agg: dict, reference: float | None,
                   acc_at_8: float | None) -> dict:
    """The rubric's metrics read as though the sweep ended at this cardinality.

    E7's rubric is written in whole-sweep terms -- drop from 8 to 128, drop to
    64, drop to 255 -- but the plan requires a tier per difficulty, so each
    cardinality is graded on the same quantities measured up to itself. The
    boundary crossings then name the cardinality at which the sweep falls out of
    a tier, which is the thing the plan actually asks for.
    """
    accuracy = agg.get("top1")
    m: dict[str, Any] = {
        "n": agg.get("n"),
        "accuracy": accuracy,
        "accuracy_255": accuracy,
        "chance_baseline": agg.get("chance"),
        "entropy_ratio_255": agg.get("entropy_ratio"),
    }
    # ECE and monotonicity are withheld below the mass ten equal-width bins
    # need. A rung decided by a three-sample calibration error is not a rung,
    # and tiers.assign marks the result as resting on part of its rule when a
    # criterion is absent -- which is the honest reading.
    if agg.get("ece_max_p") is not None and (agg.get("n") or 0) >= MIN_N_FOR_ECE:
        m["ece"] = agg["ece_max_p"]
    if agg.get("heuristic_baseline") is not None:
        m["heuristic_baseline"] = agg["heuristic_baseline"]
    # Only supplied where there is enough mass for the bins to mean anything: a
    # non-monotone reliability curve is a global disqualifier, and reading one
    # off twenty predictions would send the condition to "Doesn't work" on noise.
    if agg.get("reliability_monotone") is not None:
        m["reliability_monotone"] = agg["reliability_monotone"]
    if reference is not None and accuracy is not None:
        drop = reference - accuracy
        m["drop_to_255"] = drop
        m["drop_to_64"] = drop
        m["drop_by_32"] = drop
    if acc_at_8 is not None and accuracy is not None and k >= 8:
        m["drop_8_to_128"] = acc_at_8 - accuracy
    return m


def _near_miss_decomposition(knee: dict) -> dict:
    """Split the knee into "more options" and "a near miss got drawn in".

    Under uniform sampling the per-option similarity distribution is the same at
    every cardinality, but the chance that a set of 255 contains a leaf one or
    two levels from the truth is far higher than for a set of 4. Accuracy with
    and without such a neighbour present separates the two effects using calls
    already spent.
    """
    out: dict[str, Any] = {}
    for k in tx.CARDINALITIES:
        shots = [s for s in knee[k]["shots"] if s.get("nearest_distractor") is not None]
        if not shots:
            continue
        row: dict[str, Any] = {}
        for name, keep in (("near_miss_present", lambda d: d <= 2),
                           ("no_near_miss", lambda d: d > 2)):
            subset = [s for s in shots if keep(s["nearest_distractor"])]
            hits = sum(1 for s in subset if s["correct"])
            lo, hi = metrics.wilson_interval(hits, len(subset))
            row[name] = {
                "n": len(subset),
                "accuracy": (hits / len(subset)) if subset else None,
                "ci95": [lo, hi],
            }
        out[str(k)] = row
    out["note"] = (
        "a near miss is a distractor within two levels of the truth: the same "
        "object class, differing only in period or region"
    )
    return out


def _paired_correct(a: Sequence[dict], b: Sequence[dict]) -> tuple[list[bool], list[bool]]:
    left = {s["item_index"]: bool(s["correct"]) for s in a if s.get("item_index") is not None}
    right = {s["item_index"]: bool(s["correct"]) for s in b if s.get("item_index") is not None}
    common = sorted(set(left) & set(right))
    return [left[i] for i in common], [right[i] for i in common]


def _compare(a: Sequence[dict], b: Sequence[dict], a_name: str, b_name: str) -> dict:
    left, right = _paired_correct(a, b)
    if not left:
        return {"n": 0, "note": f"no item was answered under both {a_name} and {b_name}"}
    out = {
        "n": len(left),
        f"{a_name}_accuracy": sum(left) / len(left),
        f"{b_name}_accuracy": sum(right) / len(right),
        "difference": (sum(left) - sum(right)) / len(left),
        "mcnemar_p": metrics.mcnemar(left, right),
    }
    if len(left) >= MIN_N_FOR_RATE:
        out["paired_bootstrap"] = metrics.paired_bootstrap(
            left, right, metrics.accuracy, n_boot=2000
        )
    return out


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------
#
# Scored against the claim as written, not only against the plan's falsification
# column. The column gives a sufficient condition for "wrong"; it is not a
# necessary one, and a knee at 8 options would satisfy "not flat to 255" while
# plainly contradicting "knee at 30-60". Marking that right would defeat the
# point of the table, which the plan says is that wrong predictions are the
# valuable output.

# The knee sweep samples cardinality on a coarse grid, so the knee is known only
# to lie in the half-open interval between the last cardinality that held and
# the first that did not. P19 counts as right when that interval overlaps 30-60.
P19_WINDOW = (30, 60)
P20_BAND = (0.10, 0.20)


def _score_predictions(knee: dict, two_stage: dict, hier: dict,
                       flat255: Sequence[dict]) -> list[dict]:
    table = {p.id: p for p in predictions.for_experiment(EXPERIMENT)}
    out: list[dict] = []

    # -- P19: the knee --------------------------------------------------
    accs = [(k, knee[k]["agg"].get("top1")) for k in tx.CARDINALITIES
            if knee[k]["agg"].get("n")]
    usable = [(k, a) for k, a in accs if a is not None]
    if len(usable) < 2:
        out.append({
            "id": "P19", "claim": table["P19"].claim, "verdict": "untestable",
            "outcome": "fewer than two cardinalities returned usable answers",
            "evidence": {"cardinalities_with_data": [k for k, _ in usable]},
        })
    else:
        reference = usable[0][1]
        knee_at = last_good = None
        for k, a in usable:
            if reference - a >= KNEE_DROP:
                knee_at = k
                break
            last_good = k
        lower = (last_good + 1) if last_good else usable[0][0]
        if knee_at is None:
            verdict = "wrong"
            outcome = (
                f"no knee: top-1 accuracy at {usable[-1][0]} options is "
                f"{usable[-1][1]:.3f} against {reference:.3f} at {usable[0][0]}, a drop of "
                f"{reference - usable[-1][1]:.3f}, below the {KNEE_DROP:.2f} threshold. "
                "This is the plan's own falsification condition, flat to 255."
            )
        else:
            overlaps = not (knee_at < P19_WINDOW[0] or lower > P19_WINDOW[1])
            verdict = "right" if overlaps else "wrong"
            outcome = (
                f"the first {KNEE_DROP:.0%} drop from the {usable[0][0]}-option ceiling "
                f"of {reference:.3f} appears at {knee_at} options "
                f"({dict(usable)[knee_at]:.3f}), so the knee lies in "
                f"({lower - 1}, {knee_at}]. The predicted window is "
                f"{P19_WINDOW[0]}-{P19_WINDOW[1]}, which that interval "
                f"{'overlaps' if overlaps else 'does not reach'}."
            )
        out.append({
            "id": "P19", "claim": table["P19"].claim, "verdict": verdict,
            "outcome": outcome,
            "evidence": {
                "accuracy_by_cardinality": {str(k): a for k, a in usable},
                "knee_threshold": KNEE_DROP,
                "knee_at": knee_at,
                "last_cardinality_holding": last_good,
            },
        })

    # -- P20: two-stage against flat at 255 ------------------------------
    cmp20 = _compare(two_stage.get("shots") or [], list(flat255), "two_stage", "flat")
    if not cmp20.get("n"):
        out.append({
            "id": "P20", "claim": table["P20"].claim, "verdict": "untestable",
            "outcome": "no item was answered under both the flat and the two-stage arm",
            "evidence": cmp20,
        })
    else:
        diff = cmp20["difference"]
        p = cmp20["mcnemar_p"]
        tie = diff <= 0 or p >= 0.05
        in_band = P20_BAND[0] <= diff <= P20_BAND[1]
        verdict = "right" if (not tie and in_band) else "wrong"
        cost = two_stage.get("per_item_cost") or {}
        flat_tokens = _mean([float(s["input_tokens"]) for s in flat255]) or 0.0
        verdict_text = (
            "Flat wins or ties, which is the plan's falsification condition."
            if tie
            else "Two-stage wins"
            + ("." if in_band else ", but by a margin outside the predicted 10-20 points.")
        )
        cost_text = (
            f" It costs {cost.get('calls'):.0f} calls and "
            f"{(cost.get('input_tokens') or 0) / flat_tokens:.1f}x the input tokens of the "
            "one flat call."
            if flat_tokens and cost.get("calls")
            else " Its call and token cost could not be totalled."
        )
        outcome = (
            f"two-stage {cmp20['two_stage_accuracy']:.3f} against flat "
            f"{cmp20['flat_accuracy']:.3f} on {cmp20['n']} shared items, a difference of "
            f"{diff:+.3f} (McNemar p={p:.3g}). " + verdict_text + cost_text
        )
        out.append({
            "id": "P20", "claim": table["P20"].claim, "verdict": verdict,
            "outcome": outcome,
            "evidence": {
                "comparison": cmp20,
                "two_stage_cost": cost,
                "flat_mean_input_tokens": flat_tokens,
                "filter_recall_at_16": two_stage.get("filter_recall_at_16"),
            },
        })

    # -- P21: recovery from a wrong turn ---------------------------------
    rec = hier.get("recovery") or {}
    measured = rec.get("beam2_recovery")
    if not rec.get("n_wrong_root"):
        out.append({
            "id": "P21", "claim": table["P21"].claim, "verdict": "untestable",
            "outcome": (
                "no descent took a wrong turn at the root, so there is nothing to "
                "recover from; the compounding claim cannot be exercised on this sample"
            ),
            "evidence": rec,
        })
    elif measured is None:
        out.append({
            "id": "P21", "claim": table["P21"].claim, "verdict": "untestable",
            "outcome": "the beam-2 arm returned no usable descent",
            "evidence": rec,
        })
    else:
        verdict = "right" if measured <= NEAR_ZERO_RECOVERY else "wrong"
        outcome = (
            f"strict top-1 descent recovers 0 of {rec['n_wrong_root']} wrong root turns, "
            "but that is 0 by construction rather than by measurement -- the tree is a "
            "partition. The measured quantity is a width-2 beam at the root, which "
            f"recovers {measured:.3f} of them "
            f"({rec.get('beam2_recovery_oracle_branch_choice') or 0.0:.3f} with the "
            "branch chosen correctly at the end), because the true department was the "
            "runner-up in "
            f"{rec.get('true_department_was_runner_up'):.3f} of wrong turns. "
            + ("That is near zero as predicted." if verdict == "right" else
               "That is materially above zero, so the claim as written does not hold"
               + (" and the plan's 20% falsification line is crossed."
                  if measured > 0.20 else "."))
        )
        out.append({
            "id": "P21", "claim": table["P21"].claim, "verdict": verdict,
            "outcome": outcome, "evidence": rec,
        })

    scored = {p["id"] for p in out}
    for pid, pred in table.items():
        if pid not in scored:
            out.append({
                "id": pid, "claim": pred.claim, "verdict": "untestable",
                "outcome": "this experiment produced no measurement bearing on it",
                "evidence": {},
            })
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _safe_plot(paths: list[str], notes: list[str], name: str, fn, *args, **kwargs) -> None:
    """Draw a figure, or note why it could not be drawn.

    A plotting exception after the calls are paid for would lose the whole
    experiment, and a missing figure is a far smaller loss than a lost run.
    """
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - see docstring
        notes.append(f"harness: figure {name} was not drawn ({type(exc).__name__}: {exc})")
        return
    paths.append(f"plots/{name}.png")


def _figures(run_dir: Path, knee: dict, similarity: dict, two_stage: dict,
             hier: dict, shape: dict, conf_pairs: Sequence[tuple[float, Sequence[float]]],
             notes: list[str]) -> list[str]:
    plot_dir = Path(run_dir) / "plots"
    paths: list[str] = []

    ks = [k for k in tx.CARDINALITIES if knee[k]["agg"].get("n")]
    if ks:
        aggs = [knee[k]["agg"] for k in ks]
        ns = [a["n"] for a in aggs]
        top1 = [a["top1"] for a in aggs]
        band = ([a["top1_ci95"][0] for a in aggs], [a["top1_ci95"][1] for a in aggs])
        series = {
            "top-1": top1,
            "top-5 (tie-aware)": [a["top5"] for a in aggs],
            "random 1/N": [a["chance"] for a in aggs],
        }
        if all(a.get("heuristic_baseline") is not None for a in aggs):
            series["date+gazetteer baseline"] = [a["heuristic_baseline"] for a in aggs]
        _safe_plot(
            paths, notes, "e7_accuracy_vs_cardinality", plots.curve,
            ks, series, plot_dir / "e7_accuracy_vs_cardinality.png",
            "Choice accuracy against option count",
            "options offered", "accuracy",
            n=ns, band={"top-1": band}, logx=True, xticks=ks, ylim=(0.0, 1.02),
            n_unit="items per cardinality",
            note="same items at every cardinality; token overlap is omitted here "
                 "because it sits within about 2x of 1/N at every cardinality",
        )
        ent = {
            "H / ln(N)": [a["entropy_ratio"] or 0.0 for a in aggs],
            "H / ln(reachable support)": [a["entropy_ratio_achievable"] or 0.0 for a in aggs],
        }
        _safe_plot(
            paths, notes, "e7_entropy_vs_cardinality", plots.curve,
            ks, ent, plot_dir / "e7_entropy_vs_cardinality.png",
            "Entropy of the returned distribution", "options offered",
            "entropy as a fraction of its maximum",
            n=ns, logx=True, xticks=ks, ylim=(0.0, 1.02),
            n_unit="items per cardinality",
            note="1.0 is uniform; the second series divides by the ceiling a 0.01 grid allows",
        )
        biggest = ks[-1]
        shots = knee[biggest]["shots"]
        if len(shots) >= MIN_N_FOR_RATE:
            bins = metrics.reliability_bins([s["max_p"] for s in shots],
                                            [s["correct"] for s in shots])
            _safe_plot(
                paths, notes, "e7_reliability_255", plots.reliability_diagram,
                bins, plot_dir / "e7_reliability_255.png",
                f"Is the top probability calibrated at {biggest} options?",
                condition=f"{biggest} options", n=len(shots),
                xlabel="max probability returned", ylabel="top-1 accuracy",
            )

    order = [p for p in SIMILARITY_LADDER if (similarity.get(p) or {}).get("agg", {}).get("n")]
    if order:
        aggs = [similarity[p]["agg"] for p in order]
        _safe_plot(
            paths, notes, "e7_distractor_similarity", plots.curve,
            order, {"top-1": [a["top1"] for a in aggs],
                    "random 1/64": [a["chance"] for a in aggs]},
            plot_dir / "e7_distractor_similarity.png",
            f"Distractor similarity at {DISTRACTOR_CARDINALITY} options",
            "distractor pool, far to near", "accuracy",
            n=[a["n"] for a in aggs],
            band={"top-1": ([a["top1_ci95"][0] for a in aggs],
                            [a["top1_ci95"][1] for a in aggs])},
            ylim=(0.0, 1.02), n_unit="items per pool",
        )

    strategies, values, counts = [], [], []
    if knee[255]["agg"].get("n"):
        strategies.append("flat 255\n(uniform)")
        values.append(knee[255]["agg"]["top1"])
        counts.append(knee[255]["agg"]["n"])
    if (two_stage.get("agg") or {}).get("n"):
        strategies.append(f"two-stage\n(255 -> {STAGE2_SURVIVORS})")
        values.append(two_stage["agg"]["top1"])
        counts.append(two_stage["agg"]["n"])
    if (hier.get("flat_shortlist") or {}).get("n"):
        strategies.append("flat 255\n(shortlist)")
        values.append(hier["flat_shortlist"]["top1"])
        counts.append(hier["flat_shortlist"]["n"])
    if (hier.get("descent") or {}).get("leaf_accuracy") is not None:
        strategies.append("descent\n(4 calls)")
        values.append(hier["descent"]["leaf_accuracy"])
        counts.append(hier["descent"]["n"])
    if len(strategies) >= 2:
        _safe_plot(
            paths, notes, "e7_strategies_at_255", plots.curve,
            strategies, {"accuracy": values}, plot_dir / "e7_strategies_at_255.png",
            "Four ways to resolve one label", "strategy", "accuracy",
            n=counts, ylim=(0.0, 1.02), n_unit="items per strategy",
            note="the descent resolves 10,240 leaves; the flat arms resolve 255",
        )

    if len(conf_pairs) >= MIN_N_FOR_RATE:
        buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for confidence, vec in conf_pairs:
            f = _functionals(vec)
            if f is None or confidence is None:
                continue
            buckets[min(9, int(f["max_p"] * 10))].append((f["max_p"], float(confidence)))
        filled = sorted(b for b in buckets if buckets[b])
        if len(filled) >= 2:
            xs = [_mean([m for m, _ in buckets[b]]) for b in filled]
            ys = [_mean([c for _, c in buckets[b]]) for b in filled]
            _safe_plot(
                paths, notes, "e7_confidence_vs_maxprob", plots.curve,
                xs, {"mean confidence": ys, "identity": list(xs)},
                plot_dir / "e7_confidence_vs_maxprob.png",
                "Does `confidence` track the top probability?",
                "max probability returned", "confidence returned",
                n=[len(buckets[b]) for b in filled], ylim=(0.0, 1.02),
                n_unit="answers per bin",
            )

    rubric_ks = [k for k in tx.RUBRIC_SIZES if shape.get(k, {}).get("n")]
    if rubric_ks:
        rows = [shape[k] for k in rubric_ks]
        _safe_plot(
            paths, notes, "e7_score_rubric_size", plots.curve,
            rubric_ks,
            {"accuracy": [r["accuracy"] for r in rows],
             "random 1/K": [r["chance"] for r in rows],
             "mean confidence": [r["mean_confidence"] or 0.0 for r in rows],
             "mean max probability": [r["mean_max_p"] or 0.0 for r in rows]},
            plot_dir / "e7_score_rubric_size.png",
            "Score questions against rubric length", "ordered rubric points",
            "accuracy / probability", n=[r["n"] for r in rows], logx=True,
            xticks=rubric_ks, ylim=(0.0, 1.02), n_unit="items per rubric size",
        )
    return paths


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _anomalies(knee: dict, all_shots: Sequence[dict], shape: dict,
               conf_choice: dict, conf_score: dict, recovery: dict) -> list[str]:
    out: list[str] = []

    # A width-2 beam that reaches the truth and then discards it is a finding
    # about the arbitration rule rather than about the beam, and nothing else in
    # the result says so: P21's verdict rests on the arbitrated rate, which is
    # the one a deployment would get.
    arb = recovery.get("beam2_recovery")
    oracle = recovery.get("beam2_recovery_oracle_branch_choice")
    if arb is not None and oracle is not None and oracle - arb >= 0.20:
        out.append(
            f"Beam arbitration throws away recoveries it already has: over "
            f"{recovery.get('n_measured')} wrong root turns the runner-up branch "
            f"ends at the correct leaf {oracle:.1%} of the time, but choosing "
            f"between the two branches on the product of their chosen "
            f"probabilities keeps only {arb:.1%}. The gap is a property of the "
            "tie-break, not of the beam, so a better branch score would recover "
            "most of it."
        )

    q = _quantisation_facts(all_shots)
    if q.get("n"):
        out.append(
            f"Probability quantisation: {q['fraction_on_001_grid']:.1%} of "
            f"{q['n']} choice distributions have every non-zero entry on a 0.01 "
            f"grid; the largest support seen anywhere is {q['max_support_seen']} "
            f"options, and at 128 options or more the mean number of non-zero "
            f"entries is {q['mean_support_at_128_plus']}. Returned distributions "
            f"sum to between {q['min_prob_sum']:.3f} and {q['max_prob_sum']:.3f}."
        )
        if q["fraction_on_001_grid"] > 0.9:
            ceiling = math.log(int(round(1 / PROB_GRID)))
            out.append(
                "Entropy at high cardinality is capped by that grid, not by the "
                f"model: a 0.01 grid allows at most {int(round(1 / PROB_GRID))} "
                f"non-zero entries, so entropy cannot exceed {ceiling:.2f} nats "
                f"against the ln(255)={math.log(255):.2f} the rubric's "
                "entropy_ratio_255 divides by. The 255-option distribution is "
                "therefore about "
                f"{1 - ceiling / math.log(255):.0%} sharper-looking than it can "
                "actually be; both normalisations are reported in by_difficulty "
                "and in the entropy figure."
            )

    for kind, fit in (("choice", conf_choice), ("score", conf_score)):
        if not fit.get("fits"):
            continue
        out.append(
            f"On {kind} answers, `confidence` equals the max probability in "
            f"{fit['confidence_equals_max_probability']:.1%} of {fit['n']} answers. "
            f"The closed form it matches best with no fitted scale is "
            f"{fit['best_as_identity']} (mean absolute error "
            f"{fit['best_identity_mae']:.4f}); the best linear fit is on "
            f"{fit.get('best_linear') or 'nothing -- no candidate had spread to fit'}"
            + (f" with R^2 {fit['best_linear_r_squared']:.4f}."
               if fit.get("best_linear_r_squared") is not None else ".")
        )

    pos = _position_bias(all_shots)
    if pos:
        thirds = pos["accuracy_by_position_third"]
        out.append(
            "Position: accuracy when the correct option sits in the first, middle "
            f"and last third of the list is {_num(thirds['first']['accuracy'])}, "
            f"{_num(thirds['middle']['accuracy'])} and "
            f"{_num(thirds['last']['accuracy'])}; over {pos['n_wrong']} wrong picks "
            "the chosen position averages "
            f"{_num(pos['mean_normalised_position_of_wrong_pick'])} of the way down "
            "the list, against the 0.5 that no bias would give."
        )
    ids = _id_bias(all_shots)
    if ids:
        dev = ids["largest_deviation"]
        never = ids["departments_never_chosen"]
        out.append(
            "Option-id preference: the department furthest from the flat rate is "
            f"{dev['department']} at {dev['ratio_to_expected']:.2f}x it, over "
            f"{ids['n']} answers at 64 options or more"
            + (f"; never chosen at all: {', '.join(never)}" if never else "")
            + ". The truth is uniform over departments, so anything far from 1.0 is "
            "a standing preference rather than accuracy."
        )

    dead = [k for k in tx.RUBRIC_SIZES if not shape.get(k, {}).get("n")]
    if dead:
        out.append(
            f"Score rubric sizes {dead} returned no usable answer at all; the "
            "endpoint may cap rubric length, which is undocumented."
        )
    legend = [shape[k].get("legend_returned") for k in tx.RUBRIC_SIZES if shape.get(k, {}).get("n")]
    if legend and all(v == 1.0 for v in legend if v is not None):
        out.append(
            "Every score answer carried the undocumented `legend` field mapping "
            "rubric indices to the labels sent."
        )

    lat = [(k, knee[k]["agg"].get("latency_s")) for k in tx.CARDINALITIES
           if knee[k]["agg"].get("latency_s")]
    if len(lat) >= 2:
        out.append(
            f"Latency against option count: p50 {lat[0][1]['p50']:.3f}s at "
            f"{lat[0][0]} options against {lat[-1][1]['p50']:.3f}s at {lat[-1][0]}, "
            f"p95 {lat[0][1]['p95']:.3f}s against {lat[-1][1]['p95']:.3f}s."
        )
    return out


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    n_items = config.n(config.SIZES.cardinality)
    n_score = config.n(config.SIZES.accuracy)
    seed = config.seed_for(EXPERIMENT, "catalogue")
    score_seed = config.seed_for(EXPERIMENT, "score")
    ledger = _new_ledger()
    notes: list[str] = []
    items = tx.items(seed=seed, count=n_items)

    with JevClient(run_dir=run_dir, log_name="e7.jsonl") as client:
        knee = _run_knee(client, seed, n_items, ledger)
        similarity = _run_distractors(client, seed, n_items, ledger,
                                      knee[DISTRACTOR_CARDINALITY])
        two_stage = _run_two_stage(client, seed, ledger, knee[255])
        hier = _run_hierarchical(client, seed, items, ledger)
        shape = _run_shape(client, score_seed, n_score, ledger)
        score_cap = _probe_score_cap(client, score_seed, ledger)
        run_summary = client.summary()

    # -- pooled views ----------------------------------------------------
    all_shots: list[dict] = []
    for cell in knee.values():
        all_shots.extend(cell["shots"])
    for profile, cell in similarity.items():
        if profile != "uniform":  # uniform IS the 64-option knee cell
            all_shots.extend(cell["shots"])
    all_shots.extend(two_stage.get("shots") or [])
    all_shots.extend(hier.get("_flat_shots") or [])

    choice_pairs = [
        (s["confidence"], s["distribution"])
        for s in all_shots
        if s.get("confidence") is not None and s.get("distribution")
    ]
    choice_pairs.extend(
        (row["confidence"], row["distribution"])
        for row in (hier.get("_level_rows") or [])
        if row.get("confidence") is not None and row.get("distribution")
    )
    score_pairs = [
        (s["confidence"], s["distribution"])
        for k in tx.RUBRIC_SIZES
        for s in (shape.get(k, {}).get("shots") or [])
        if s.get("confidence") is not None and s.get("distribution")
    ]
    conf_choice = _confidence_fit(choice_pairs, "choice")
    conf_score = _confidence_fit(score_pairs, "score")

    # -- tiers -----------------------------------------------------------
    reference = knee[tx.CARDINALITIES[0]]["agg"].get("top1")
    acc_at_8 = knee[8]["agg"].get("top1")
    card_sweep = [
        ({"n_options": k}, _sweep_metrics(k, knee[k]["agg"], reference, acc_at_8))
        for k in tx.CARDINALITIES
        if knee[k]["agg"].get("n")
    ]
    card_results = tiers.tier_by_difficulty(EXPERIMENT, card_sweep) if card_sweep else []
    crossings = tiers.boundary_crossings(EXPERIMENT, card_results) if card_results else {}
    sim_sweep = [
        (
            {"n_options": DISTRACTOR_CARDINALITY, "distractor_profile": profile},
            _sweep_metrics(DISTRACTOR_CARDINALITY, similarity[profile]["agg"],
                           reference, acc_at_8),
        )
        for profile in SIMILARITY_LADDER
        if (similarity.get(profile) or {}).get("agg", {}).get("n")
    ]
    sim_results = tiers.tier_by_difficulty(EXPERIMENT, sim_sweep) if sim_sweep else []

    # -- comparisons -----------------------------------------------------
    flat255 = knee[255]["shots"]
    descent_pseudo = [
        {"item_index": idx, "correct": ok}
        for idx, ok in (hier.get("_leaf_correct") or {}).items()
    ]
    comparisons = {
        "two_stage_vs_flat_255": _compare(two_stage.get("shots") or [], flat255,
                                          "two_stage", "flat"),
        "descent_vs_flat_shortlist": _compare(descent_pseudo,
                                              hier.get("_flat_shots") or [],
                                              "descent", "flat_shortlist"),
        "note": (
            "descent_vs_flat_shortlist is not like for like: the descent resolves "
            f"all {tx.N_LEAVES} leaves in four calls, the flat arm resolves 255 in "
            "one, and the shortlist comes from a simulated retriever. The uniform "
            "255-option cell of the knee sweep is the contrast for how much the "
            "shortlist's enrichment costs."
        ),
    }

    preds = _score_predictions(knee, two_stage, hier, flat255)
    anomalies = _anomalies(knee, all_shots, shape, conf_choice, conf_score,
                           hier.get("recovery") or {})
    anomalies.append(
        f"Score rubrics are capped far below choice: {score_cap['levels_requested']} "
        f"ordered levels was "
        + ("accepted, so the cap has moved since this module was written."
           if score_cap["accepted"]
           else f"rejected with HTTP {score_cap['http_status']} "
                f"({score_cap['error']}). Choice takes 255 options; score takes "
                f"{score_cap['documented_cap_used']} levels, which is undocumented "
                "and limits how far the score cardinality axis can be swept.")
    )
    withheld = [
        k for k in tx.CARDINALITIES
        if knee[k]["agg"].get("n") and knee[k]["agg"]["n"] < MIN_N_FOR_ECE
    ]
    if withheld:
        notes.append(
            f"ECE and reliability monotonicity were withheld from the tier at "
            f"cardinalities {withheld}: fewer than {MIN_N_FOR_ECE} predictions leaves "
            "ten equal-width bins too sparse to grade on. Those tiers therefore rest "
            "on the accuracy criteria alone, which their `unevaluated` field records."
        )
    plot_paths = _figures(run_dir, knee, similarity, two_stage, hier, shape,
                          choice_pairs, notes)

    largest = next((k for k in reversed(tx.CARDINALITIES) if knee[k]["agg"].get("n")), None)
    if largest is None:
        headline = {"metric": "top-1 accuracy", "value": None, "baseline": None,
                    "baseline_name": "random (1/N)", "n": 0}
    else:
        cell = knee[largest]["agg"]
        headline = {
            "metric": f"top-1 accuracy at {largest} options",
            "value": cell["top1"],
            "baseline": cell["chance"],
            "baseline_name": f"random (1/{largest})",
            "n": cell["n"],
        }

    notes.append(
        "Top-5 is 1.0 by definition at 2 and 4 options, where every option is in "
        "the top five, so only the cardinalities from 8 upward carry information "
        "there. It is computed tie-aware: with the returned distribution quantised "
        "to 0.01, most options at high cardinality share a probability, and the "
        "reported value is the chance the truth lands in the top five under uniform "
        "tie-breaking rather than under whatever order the options arrived in."
    )
    notes.append(
        "Two cheap baselines are reported beside every accuracy. `overlap_baseline` "
        "is token overlap between the state and the option labels. The state never "
        "reuses a label's head word, so the overlap is usually tied across the whole "
        "option set and the baseline sits near 1/N -- measured at 1.0x chance at 2 "
        "options rising to about 2.4x at 200, which is still far below any usable "
        "accuracy. `heuristic_baseline` is a date parser plus a gazetteer: it resolves "
        "the period and the region exactly and guesses uniformly among the options "
        "consistent with both. The rubric's 'below a cheap deterministic baseline' "
        "signature is evaluated against the second one. Its lookups are exact "
        "against this generator's vocabulary, so it is an upper bound on what "
        "twenty lines of code would really buy."
    )
    if config.SCALE != 1.0:
        notes.append(
            f"Sample sizes were scaled by {config.SCALE}: {n_items} items per "
            f"cardinality against the plan's {config.SIZES.cardinality}, and "
            f"{n_score} per rubric size. Every rate below carries the interval that "
            "implies and should not be read as a measurement."
        )

    return {
        "experiment": EXPERIMENT,
        "question": QUESTION,
        "headline": headline,
        "by_difficulty": [r.to_json() for r in (*card_results, *sim_results)],
        "boundary_crossings": tiers.crossings_json(crossings) if crossings else {},
        "plots": plot_paths,
        "predictions": preds,
        "anomalies": anomalies,
        "failures": {
            "calls": ledger["failed"],
            "excluded": ledger["excluded"],
            "attempted_calls": ledger["calls"],
            "reasons": dict(ledger["reasons"]),
        },
        "summary": {
            "knee": {str(k): knee[k]["agg"] for k in tx.CARDINALITIES},
            "distractor_quality": {p: similarity[p]["agg"] for p in SIMILARITY_LADDER
                                   if similarity.get(p)},
            "two_stage": {k: v for k, v in two_stage.items() if k != "shots"},
            "hierarchical": {k: v for k, v in hier.items() if not k.startswith("_")},
            "distribution_shape": {
                str(k): {kk: vv for kk, vv in shape[k].items() if kk != "shots"}
                for k in tx.RUBRIC_SIZES if k in shape
            },
            "confidence_vs_distribution": {"choice": conf_choice, "score": conf_score},
            "score_level_cap": score_cap,
            "comparisons": comparisons,
            "similarity_tiers": [r.to_json() for r in sim_results],
            "near_miss_decomposition": _near_miss_decomposition(knee),
        },
        "sample_sizes": {
            "items_per_cardinality": n_items,
            "items_per_rubric_size": n_score,
            "scale": config.SCALE,
            "plan_cardinality_n": config.SIZES.cardinality,
            "taxonomy_leaves": tx.N_LEAVES,
            "seed": seed,
        },
        "run": run_summary,
        "notes": notes,
    }


def _num(value: Any, places: int = 3) -> str:
    return "  -  " if value is None else f"{value:.{places}f}"


def format_report(result: dict) -> str:
    s = result.get("summary", {})
    lines = [f"E7 cardinality -- {result['question']}"]
    h = result["headline"]
    val = "n/a" if h["value"] is None else f"{h['value']:.3f}"
    base = "n/a" if h["baseline"] is None else f"{h['baseline']:.4f}"
    lines.append(f"  headline: {h['metric']} = {val} vs {h['baseline_name']} {base}, n={h['n']}")

    lines.append(
        "  options | top-1 | top-5 | random | lookup | overlap | H/lnN | n"
    )
    for k in tx.CARDINALITIES:
        cell = (s.get("knee") or {}).get(str(k)) or {}
        if not cell.get("n"):
            continue
        lines.append(
            f"    {k:>4}  | {_num(cell.get('top1'))} | {_num(cell.get('top5'))} | "
            f"{_num(cell.get('chance'), 4)} | {_num(cell.get('heuristic_baseline'))} | "
            f"{_num(cell.get('overlap_baseline'))} | {_num(cell.get('entropy_ratio'))} | "
            f"{cell['n']}"
        )

    lines.append(f"  distractor pool at {DISTRACTOR_CARDINALITY} options:")
    for profile in SIMILARITY_LADDER:
        cell = (s.get("distractor_quality") or {}).get(profile) or {}
        if cell.get("n"):
            lines.append(f"    {profile:<19} {cell['top1']:.3f}  n={cell['n']}")

    cmp20 = (s.get("comparisons") or {}).get("two_stage_vs_flat_255") or {}
    if cmp20.get("n"):
        cost = (s.get("two_stage") or {}).get("per_item_cost") or {}
        lines.append(
            f"  two-stage {cmp20['two_stage_accuracy']:.3f} vs flat "
            f"{cmp20['flat_accuracy']:.3f} ({cmp20['difference']:+.3f}, "
            f"McNemar p={cmp20['mcnemar_p']:.3g}) at {cost.get('calls')} calls and "
            f"${cost.get('cost_usd', 0):.6f} per decision"
        )
    hier = s.get("hierarchical") or {}
    desc = hier.get("descent") or {}
    if desc.get("leaf_accuracy") is not None:
        lines.append(
            f"  descent over {tx.N_LEAVES} leaves: {desc['leaf_accuracy']:.3f} leaf "
            f"accuracy in {desc['calls_per_item']} calls; per-level "
            + ", ".join(
                f"{row['name']}={row['conditional_accuracy']:.3f}"
                for row in desc.get("per_level", [])
                if row.get("conditional_accuracy") is not None
            )
        )
        rec = hier.get("recovery") or {}
        if rec.get("n_wrong_root"):
            lines.append(
                f"  recovery from a wrong root turn: strict 0.000 (by construction), "
                f"width-2 beam {rec.get('beam2_recovery')}, over "
                f"{rec['n_wrong_root']} wrong turns"
            )
    flat_sl = hier.get("flat_shortlist") or {}
    if flat_sl.get("n"):
        lines.append(f"  flat 255-item shortlist: {flat_sl['top1']:.3f}  n={flat_sl['n']}")

    for boundary, where in (result.get("boundary_crossings") or {}).items():
        lines.append(f"  {boundary}: crossed at {where}")
    for anomaly in result.get("anomalies", []):
        lines.append(f"  anomaly: {anomaly}")
    for p in result["predictions"]:
        lines.append(f"  {p['id']} {p['verdict'].upper()}: {p['outcome']}")
    f = result["failures"]
    lines.append(
        f"  calls attempted {f['attempted_calls']}, failed {f['calls']}, "
        f"excluded {f['excluded']}"
    )
    for note in result.get("notes", []):
        lines.append(f"  note: {note}")
    return "\n".join(lines)
