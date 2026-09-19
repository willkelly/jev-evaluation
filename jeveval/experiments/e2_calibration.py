"""E2 -- calibration via the 3SAT phase transition.

The plan calls ECE "the single most important number in this whole plan" and
calibration "the entire reason to choose this model over an LLM". Vendor
calibration is calibration against the vendor's distribution, which is not this
one, so the question is whether the returned probabilities mean anything here.

Random 3SAT answers that question more cheaply than anything else in the plan.
The satisfiable fraction falls from 1 to 0 across clause-to-variable ratio, so
the sweep supplies a target probability curve with no labelling at all: the
solver decides every instance, and the fraction it decides SAT at each ratio is
what a calibrated P(SAT) should average to there.

Five things shape this module.

**Clause density alone predicts satisfiability well.** Accuracy on the plain
sweep is therefore not evidence of reasoning, and a model reading only the
DIMACS header would score well on it. Every ratio that can support one also gets
a density-matched control from `sat3.generate_matched` -- equal SAT and UNSAT at
the same n and the same m -- where the majority baseline and the density
heuristic are both exactly 0.5. Rejection sampling cannot fill both halves at
the ends of the ratio range, so the matched conditions come from
`sat3.matched_sweep()` and are generated with `strict=False`; the realised
counts and the ratios that were out of reach are written to
`e2_conditions.json` and reported, rather than an unbalanced set being passed
off as a control.

**The gate is on the semantic control, not on 3SAT.** The plan is explicit that
bad calibration on 3SAT alone is an expected off-distribution result and is not
a gate. So a calibration condition runs on the support-ticket control as well,
at the "hard" level: at "clean" a keyword baseline scores 1.000 and the
condition cannot discriminate a working model from a broken one.

**Scoring reads the JSONL log, not the live results.** `run` spends the calls
and then calls `score`, which reconstructs every number from the log alone.
That is the plan's requirement -- "metrics can be recomputed without re-spending
calls" -- and routing the live path through the same code is the only way to
know the log really is sufficient. `score(run_dir)` can be called on its own
afterwards to rescore or to redefine a metric.

**Calls are submitted in one seeded shuffle, not ratio by ratio.** The
instances are generated in ratio order and the headline figure is a curve in
ratio, so sending them in that order would make elapsed time a proxy for the
x-axis across tens of thousands of calls: any drift in the service during the
run would land in one stretch of the sweep and be read off the figure as a
property of the ratio. E1 interleaves its four conditions for the same reason.

**Three of the rubric's metrics are properties of a sweep, not of a ratio.**
`curve_max_abs_error`, `crossover_ratio` and `prob_monotone` describe a whole
ratio sweep at one instance size, so each is stamped onto every row of that
size. That is what makes a non-monotone predicted curve tier the entire sweep
"Doesn't work", which is the plan's intent: a probability that does not fall
with density is not a probability about satisfiability.

At the plan's own sample sizes this is the largest experiment in it: 75 sweep
cells at 500 instances, 41 attainable matched cells at 200, and 500 semantic
control instances, for 46,200 calls, each carrying one noul.

Two recorded departures from the plan's text, both stated in the result:

* Per-ratio AUROC is undefined wherever the solver decided every instance the
  same way, which is most of the sweep outside the transition. `metrics.auroc`
  returns NaN there and `tiers.assign` treats NaN as unmeasured rather than as a
  failed bar. The pooled AUROC on the matched set is the discrimination number
  that is not confounded by density, and it is reported separately.
* The plan asks for 500 instances per condition wherever ECE is reported, and
  allows 200 for a pure accuracy comparison. The matched control is scored on
  accuracy against 0.5, so it uses `SIZES.accuracy`; the sweep and the semantic
  control report ECE and use `SIZES.calibration`.
"""

from __future__ import annotations

import json
import math
import random
from statistics import NormalDist
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .. import config, logstore, metrics, plots, predictions, tiers
from ..client import Call, JevClient
from ..generators import sat3, semantic
from ..instances import Instance, rng_for, shuffled_questions

EXPERIMENT = "E2"
LOG_NAME = "e2.jsonl"
CONDITIONS_FILE = "e2_conditions.json"

QUESTION = (
    "Do the returned probabilities mean anything on a distribution the vendor "
    "did not calibrate against?"
)

# n=20 is the plan's primary sweep; 10 and 50 exist to separate instance size
# from ratio. Kept in this order so the primary sweep is first everywhere.
PRIMARY_N = 20
SIZES_SWEPT = (20, 10, 50)

# "Use the 'hard' level of the semantic sweep, not 'clean'": at clean the
# keyword heuristic scores 1.000, so the condition cannot discriminate.
CONTROL_LEVEL = "hard"

# The plan's gate: "if calibration on the semantic control is worse than ECE
# 0.30, the core value proposition does not hold on your data."
GATE_ECE = 0.30

# A 10-bin ECE on a handful of predictions is not a measurement. The plan asks
# for 500; below this many scorable control predictions the gate is reported as
# not evaluated rather than passed on noise. Dry runs land here by design.
MIN_GATE_N = 100

# Below this many predictions a per-condition ECE and Brier are still computed,
# because the sweep's shape is the point and a gap in it is worse than a noisy
# point, but the row is flagged so nothing reads them as measurements.
MIN_STABLE_N = 50

# --- thresholds the harness supplies where the plan gives a word -------------
#
# Transcribed the way tiers.py does it: where the plan names a number the number
# is used; where it names a shape, the number here is the harness's reading and
# is reported as such.

# P6 asks whether the predicted curve is "flatter than the true curve at 4.26".
# Slopes are fitted over ratios within this distance of the transition.
SLOPE_HALFWIDTH = 0.75
# "Flatter" is read as a predicted slope whose magnitude is below this fraction
# of the true curve's. 0.8 leaves a 20% band that counts as neither flatter nor
# sharper, so a curve that merely tracks the true one is not scored as flat.
FLATNESS_RATIO = 0.80

# Monotonicity of the predicted curve in ratio. A false positive here sends an
# entire sweep to "Doesn't work", so the test is deliberately conservative:
# every observed probability so far is quantized to 2 decimals, and a rise of
# one quantization step is not a reversal, so that is the floor.
MONOTONE_MIN_STEP = 0.02
# A cell below this many predictions does not take part. Probabilities pile up
# against 0 and 1 across most of this sweep, so a small cell's sample spread
# understates how much its mean moves between draws, and the normal band
# computed from it would be far too tight. Under the plan's 500 per ratio every
# cell qualifies; at a dry-run scale none does and the test reports itself
# untested rather than guessing.
MONOTONE_MIN_COUNT = 30
# Observed probabilities are quantized to 2 decimals, so a cell whose answers
# are all identical still carries the grid's uncertainty rather than none.
PROBABILITY_STEP = 0.01
# Family-wise error rate for the all-pairs comparison, Bonferroni-corrected
# across the pairs actually compared. A 25-ratio sweep is 300 comparisons, and
# at an uncorrected 95% about fifteen of them fire on a perfectly monotone
# curve -- which would tier nearly every real sweep "Doesn't work" on noise.
MONOTONE_ALPHA = 0.05

# Bulky generator meta that is not needed to rescore. The clause list and the
# satisfying assignment are already implied by the DIMACS state in the logged
# request, and sat3's own self-check is what verifies the encoder; carrying a
# second copy on each of ~46k records would roughly double the log for no
# rescoring value.
_BULKY_META = ("clauses", "assignment")


# --------------------------------------------------------------------------
# Instances and calls
# --------------------------------------------------------------------------


@dataclass
class _Group:
    """One named block of the plan, so the run prints what it is about to spend
    per sweep rather than per ratio. It is a unit of reporting, not of
    submission: every group's calls go into one shuffled list before any of them
    is sent. The per-call `condition` still names the size and the ratio."""

    label: str
    calls: list[Call] = field(default_factory=list)


@dataclass
class _Plan:
    groups: list[_Group] = field(default_factory=list)
    conditions: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _even(k: int) -> int:
    """`generate_matched` requires an even count so the two halves are equal."""
    return max(2, k - (k % 2))


def _condition_key(kind: str, n: int, ratio: float) -> str:
    return f"{kind}-n{n}-r{ratio:g}"


def _call_meta(kind: str, instance: Instance) -> dict:
    """Everything the offline scorer needs, and nothing it does not.

    Ground truth travels here so the JSONL log alone can be rescored. The
    question key travels with it because the scorer must not guess which answer
    in a response is the one with truth attached.
    """
    if len(instance.questions) != 1:
        raise ValueError(
            f"{instance.instance_id}: E2 scores one noul per call, got "
            f"{len(instance.questions)} questions"
        )
    meta = {k: v for k, v in instance.meta.items() if k not in _BULKY_META}
    meta.update(
        {
            "kind": kind,
            "generator": instance.generator,
            "difficulty": instance.difficulty,
            "question_key": next(iter(instance.questions)),
            "truth": instance.truth,
        }
    )
    return meta


def _calls(kind: str, condition: str, instances: Iterable[Instance]) -> list[Call]:
    out: list[Call] = []
    for inst in instances:
        # Every call carries exactly one noul and no choice, so this shuffle is
        # a no-op today. It is applied anyway so that adding a second question
        # to this experiment cannot silently reintroduce position bias.
        rng = rng_for(EXPERIMENT, inst.difficulty, inst.seed, inst.index)
        out.append(
            Call(
                experiment=EXPERIMENT,
                condition=condition,
                instance_id=inst.instance_id,
                state=inst.state,
                questions=shuffled_questions(inst.questions, rng),
                meta=_call_meta(kind, inst),
            )
        )
    return out


def _build_plan() -> _Plan:
    """Generate every instance and lay out the calls.

    Sample sizes come from `config.SIZES` through `config.n`, which applies the
    dry-run scale factor, so nothing here hardcodes a count.
    """
    sweep_n = config.n(config.SIZES.calibration)
    matched_n = _even(config.n(config.SIZES.accuracy))
    control_n = config.n(config.SIZES.calibration)

    plan = _Plan()

    # -- the plain sweep, one group per instance size --------------------
    for size in SIZES_SWEPT:
        group = _Group(label=f"sweep-n{size}")
        for difficulty in sat3.difficulty_sweep():
            if difficulty["n"] != size:
                continue
            ratio = float(difficulty["ratio"])
            key = _condition_key("sweep", size, ratio)
            instances = sat3.generate(
                difficulty=difficulty,
                seed=config.seed_for(EXPERIMENT, key),
                count=sweep_n,
            )
            group.calls.extend(_calls("sweep", key, instances))
            plan.conditions.append(
                {
                    "condition": key,
                    "kind": "sweep",
                    "n": size,
                    "ratio": ratio,
                    "requested": sweep_n,
                    "generated": len(instances),
                    "true_sat_fraction": (
                        sum(i.meta["satisfiable"] for i in instances) / len(instances)
                        if instances
                        else None
                    ),
                }
            )
        plan.groups.append(group)

    # -- the density-matched control -------------------------------------
    attainable = {(d["n"], float(d["ratio"])) for d in sat3.matched_sweep()}
    for size in SIZES_SWEPT:
        group = _Group(label=f"matched-n{size}")
        for difficulty in sat3.difficulty_sweep():
            if difficulty["n"] != size:
                continue
            ratio = float(difficulty["ratio"])
            key = _condition_key("matched", size, ratio)
            if (size, ratio) not in attainable:
                # Out of the measured range where rejection sampling can reach
                # both classes. Recorded rather than skipped in silence: which
                # ratios have no control is part of the result.
                plan.conditions.append(
                    {
                        "condition": key,
                        "kind": "matched",
                        "n": size,
                        "ratio": ratio,
                        "requested": matched_n,
                        "generated": 0,
                        "attainable": False,
                    }
                )
                continue
            # strict=False returns a shorter but still balanced set rather than
            # raising when one half runs dry inside the budget.
            instances = sat3.generate_matched(
                difficulty=difficulty,
                seed=config.seed_for(EXPERIMENT, key),
                count=matched_n,
                strict=False,
            )
            plan.conditions.append(
                {
                    "condition": key,
                    "kind": "matched",
                    "n": size,
                    "ratio": ratio,
                    "requested": matched_n,
                    "generated": len(instances),
                    "attainable": bool(instances),
                    "short": bool(instances) and len(instances) < matched_n,
                }
            )
            if not instances:
                plan.notes.append(
                    f"matched control at n={size} ratio={ratio:g}: neither half could be "
                    "sampled within the budget, so the condition was not run"
                )
                continue
            if len(instances) < matched_n:
                plan.notes.append(
                    f"matched control at n={size} ratio={ratio:g}: {len(instances)} of "
                    f"{matched_n} instances, still balanced"
                )
            group.calls.extend(_calls("matched", key, instances))
        if group.calls:
            plan.groups.append(group)

    # -- the semantic control, which is what the gate reads ---------------
    control_key = f"control-{CONTROL_LEVEL}"
    control = semantic.generate_noul(
        seed=config.seed_for(EXPERIMENT, control_key),
        count=control_n,
        level=CONTROL_LEVEL,
    )
    plan.groups.append(_Group(label=control_key, calls=_calls("control", control_key, control)))
    plan.conditions.append(
        {
            "condition": control_key,
            "kind": "control",
            "level": CONTROL_LEVEL,
            "requested": control_n,
            "generated": len(control),
        }
    )
    return plan


# --------------------------------------------------------------------------
# Reading the log back
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    """One scorable prediction, reconstructed from one logged record."""

    condition: str
    kind: str
    n_vars: int | None
    ratio: float | None
    p: float
    truth: bool
    heuristic: bool | None
    latency_s: float
    instance_id: str


def _heuristic_prediction(kind: str, meta: dict) -> bool | None:
    """The cheap deterministic baseline's answer for one instance.

    3SAT: read m and n out of the header and predict SAT below the asymptotic
    threshold. That is all a header-reading baseline can see, and on the matched
    set it is constant, which is what makes the matched set a control.

    Semantic control: the keyword baseline names a department; the noul asks
    whether the ticket belongs to one particular department, so the baseline
    answers yes when those agree. Where no keyword fires it answers no, which is
    the better of its two constant fallbacks on a balanced yes/no set.
    """
    if kind == "control":
        guess = meta.get("baseline_keyword_prediction")
        asked = meta.get("asked_department")
        if asked is None:
            return None
        return guess == asked
    pred = meta.get("baseline_density_pred")
    return None if pred is None else bool(pred)


# The fields `logstore.run_stats` reads. A full E2 log is around 46k records
# whose bulk is the DIMACS state in each logged request, and holding all of them
# as Python objects to compute a p95 costs hundreds of megabytes. Scanning once
# and keeping this projection instead leaves run_stats as the single definition
# of those run-level numbers without retaining the states.
_STAT_FIELDS = (
    "ts",
    "outcome",
    "latency_s",
    "input_tokens",
    "output_tokens",
    "model_version",
    "http_status",
    "transport_error",
)


ABANDONED_REASON = "retries exhausted: the call never reached a terminal record"
UNLOGGED_REASON = "generated but absent from the log: no attempt of any kind was recorded"


def _scan(path: Path) -> tuple[list[_Row], dict, list[dict], Counter[str]]:
    """Stream the log once: scorable rows, failure accounting, run-level facts.

    A failed call is never a data point: it is counted, its reason is recorded,
    and it is excluded. Nothing is defaulted to 0.5 or to the majority class.

    A call that exhausts its retries is the one failure the log does not mark
    as such. The client writes `outcome: "retry"` on every attempt including the
    last, so such a call leaves retry history and no terminal record at all, and
    counting only terminal records would report it as neither scored nor
    failed -- the sample would shrink and the result would still say zero
    failures. So the keys seen only in retry history are collected and counted
    as the failed calls they are.

    The fourth return value is the number of calls seen per condition, terminal
    or abandoned, which `score` reconciles against the conditions file to catch
    the other way a call can vanish: generated, then never logged at all,
    because the run was killed partway through.

    Re-running the same experiment into the same run directory appends to the
    log, and instance ids are deterministic, so the newest terminal record of
    each (condition, instance, repetition) wins and earlier attempts at the same
    one are dropped rather than double counted.
    """
    reasons: Counter[str] = Counter()
    stat_records: list[dict] = []
    # (timestamp, row or None, failure reason or None) per call, newest kept.
    latest: dict[tuple[str, str, int], tuple[float, _Row | None, str | None]] = {}
    retried: set[tuple[str, str, int]] = set()

    for record in logstore.read(path):
        stat_records.append({k: record.get(k) for k in _STAT_FIELDS})
        if record.get("experiment") != EXPERIMENT:
            continue
        outcome = record.get("outcome")
        if outcome not in ("ok", "error", "parse_error", "retry"):
            continue
        key = (
            str(record.get("condition")),
            str(record.get("instance_id")),
            int(record.get("repetition") or 0),
        )
        if outcome == "retry":
            retried.add(key)
            continue
        ts = float(record.get("ts") or 0.0)
        if key in latest and latest[key][0] > ts:
            continue
        latest[key] = (ts, *_extract(record))

    rows: list[_Row] = []
    failed_calls = 0
    for _, row, reason in latest.values():
        if row is not None:
            rows.append(row)
            continue
        reasons[reason or "unknown"] += 1
        if not (reason or "").startswith("ok:"):
            failed_calls += 1

    abandoned = retried - set(latest)
    if abandoned:
        reasons[ABANDONED_REASON] += len(abandoned)
        failed_calls += len(abandoned)

    attempted: Counter[str] = Counter(cond for cond, _, _ in latest)
    attempted.update(cond for cond, _, _ in abandoned)

    failures = {
        "calls": failed_calls,
        "excluded": len(latest) - len(rows) + len(abandoned),
        "reasons": dict(reasons),
    }
    return rows, failures, stat_records, attempted


def _unlogged(planned: Sequence[dict], attempted: Counter[str]) -> dict[str, int]:
    """Calls the plan generated that the log has no record of, per condition.

    The conditions file is written before the first call goes out, so a
    shortfall against its `generated` count is a call that was planned and never
    logged -- the run was killed, or the process died mid-flight. Counting it is
    the difference between a report that says the sweep ran at 500 per ratio and
    one that says how many answers each ratio actually rests on.
    """
    out: dict[str, int] = {}
    for c in planned:
        want = int(c.get("generated") or 0)
        got = attempted.get(str(c.get("condition")), 0)
        if want > got:
            out[str(c.get("condition"))] = want - got
    return out


def _extract(record: dict) -> tuple[_Row | None, str | None]:
    """One logged record to either a scorable row or a reason it is not one.

    A reason prefixed "ok:" means the call itself succeeded but the response
    could not be scored, which is a different fact from a failed call and is
    counted separately.
    """
    meta = record.get("meta") or {}
    if record.get("outcome") != "ok":
        return None, f"{record.get('outcome')}:{record.get('http_status')}"
    qkey = meta.get("question_key")
    answer = logstore.answers_of(record).get(qkey) if qkey else None
    truth = (meta.get("truth") or {}).get(qkey)
    if answer is None or answer.get("type") != "noul" or not isinstance(truth, bool):
        return None, "ok: answer or ground truth missing from the record"
    p = float(answer["p"])
    if not math.isfinite(p):
        return None, "ok: non-finite probability returned"
    kind = str(meta.get("kind") or "")
    return (
        _Row(
            condition=str(record.get("condition")),
            kind=kind,
            n_vars=int(meta["n"]) if meta.get("n") is not None else None,
            ratio=float(meta["ratio"]) if meta.get("ratio") is not None else None,
            p=p,
            truth=truth,
            heuristic=_heuristic_prediction(kind, meta),
            latency_s=float(record.get("latency_s") or 0.0),
            instance_id=str(record.get("instance_id")),
        ),
        None,
    )


# --------------------------------------------------------------------------
# Per-condition metrics
# --------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _cell(rows: Sequence[_Row]) -> dict:
    """Every metric for one condition. `rows` must be non-empty."""
    ps = [r.p for r in rows]
    ys = [r.truth for r in rows]
    correct = [(r.p >= 0.5) == r.truth for r in rows]
    bins = metrics.reliability_bins(ps, ys)
    heuristic_rows = [r for r in rows if r.heuristic is not None]
    return {
        "n": len(rows),
        "ece": metrics.ece(ps, ys),
        "brier": metrics.brier(ps, ys),
        "auroc": metrics.auroc(ps, ys),
        "accuracy": metrics.accuracy(correct),
        "mean_predicted": _mean(ps),
        "true_fraction": _mean([float(y) for y in ys]),
        "majority_baseline": metrics.majority_baseline(ys),
        "random_baseline": metrics.random_baseline(2),
        "heuristic_baseline": (
            metrics.accuracy([r.heuristic == r.truth for r in heuristic_rows])
            if heuristic_rows
            else None
        ),
        "reliability_monotone": metrics.is_monotone(bins),
        # Spread of the predicted probability, which only the monotonicity test
        # on the sweep-level curve consumes.
        "predicted_sd": _stdev(ps),
        "stable": len(rows) >= MIN_STABLE_N,
        "bins": bins,
        "latency": metrics.percentiles([r.latency_s for r in rows]),
    }


def _by_cell(rows: Iterable[_Row], kind: str) -> dict[tuple[int, float], dict]:
    grouped: dict[tuple[int, float], list[_Row]] = {}
    for r in rows:
        if r.kind != kind or r.n_vars is None or r.ratio is None:
            continue
        grouped.setdefault((r.n_vars, r.ratio), []).append(r)
    return {k: _cell(v) for k, v in sorted(grouped.items())}


def _curve_non_increasing(
    points: Sequence[tuple[float, float, float, int]],
) -> tuple[bool, bool, dict | None]:
    """Is mean predicted P(SAT) non-increasing in ratio?

    Returns (monotone, tested, worst reversal). `points` are
    (ratio, mean, standard deviation, count) in ratio order.

    Every pair is compared, not only adjacent ones, for the reason
    `metrics.is_monotone` gives: a curve that slides upward over eight ratios,
    each step inside its own band, is not monotone in ratio, and an
    adjacent-only test calls it monotone.

    Comparing every pair means comparing many pairs, and at an uncorrected 95%
    a perfectly monotone sweep produces about fifteen "reversals" by chance.
    Since a reversal tiers the whole sweep "Doesn't work", the threshold is
    Bonferroni-corrected across the pairs actually compared, and floored at one
    step of the observed 2-decimal quantization.

    The largest reversal is returned whether or not it clears the threshold, so
    the report can state how big the worst wiggle was rather than only whether
    a boolean tripped.
    """
    usable = [(x, m, s, c) for x, m, s, c in points if c >= MONOTONE_MIN_COUNT]
    if len(usable) < 3:
        return True, False, None
    pairs = len(usable) * (len(usable) - 1) // 2
    z = NormalDist().inv_cdf(1.0 - MONOTONE_ALPHA / (2 * pairs))

    monotone = True
    worst: dict | None = None
    for i in range(len(usable)):
        x_i, m_i, s_i, c_i = usable[i]
        se_i = max(s_i, PROBABILITY_STEP) / math.sqrt(c_i)
        for j in range(i + 1, len(usable)):
            x_j, m_j, s_j, c_j = usable[j]
            rise = m_j - m_i
            if rise <= 0:
                continue
            se_j = max(s_j, PROBABILITY_STEP) / math.sqrt(c_j)
            tol = max(MONOTONE_MIN_STEP, z * math.sqrt(se_i**2 + se_j**2))
            if worst is None or rise - tol > worst["rise"] - worst["tolerance"]:
                worst = {
                    "from_ratio": x_i,
                    "to_ratio": x_j,
                    "from_mean": m_i,
                    "to_mean": m_j,
                    "rise": rise,
                    "tolerance": tol,
                }
            if rise > tol:
                monotone = False
    return monotone, True, worst


def _slope(xs: Sequence[float], ys: Sequence[float], centre: float, halfwidth: float) -> float | None:
    """Least-squares slope of y against x over the window around `centre`.

    None when fewer than three sampled ratios fall in the window, which is the
    only honest answer: a slope through two points at a 0.25 step is a
    difference, not a trend.
    """
    pts = [(x, y) for x, y in zip(xs, ys) if abs(x - centre) <= halfwidth]
    if len(pts) < 3:
        return None
    mx = _mean([x for x, _ in pts])
    my = _mean([y for _, y in pts])
    denom = sum((x - mx) ** 2 for x, _ in pts)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / denom


def _sweep_summary(cells: dict[tuple[int, float], dict], size: int) -> dict | None:
    """The sweep-level facts for one instance size: the two curves and their shape."""
    items = sorted((ratio, cell) for (n, ratio), cell in cells.items() if n == size)
    if not items:
        return None
    ratios = [r for r, _ in items]
    predicted = [c["mean_predicted"] for _, c in items]
    truth = [c["true_fraction"] for _, c in items]
    counts = [c["n"] for _, c in items]
    monotone, monotone_tested, worst_reversal = _curve_non_increasing(
        [(r, c["mean_predicted"], c["predicted_sd"], c["n"]) for r, c in items]
    )
    pred_slope = _slope(ratios, predicted, plots.SAT_TRANSITION, SLOPE_HALFWIDTH)
    true_slope = _slope(ratios, truth, plots.SAT_TRANSITION, SLOPE_HALFWIDTH)
    return {
        "n": size,
        "ratios": ratios,
        "counts": counts,
        "predicted": predicted,
        "truth": truth,
        "curve_max_abs_error": max(abs(p - t) for p, t in zip(predicted, truth)),
        "crossover_ratio": plots.crossover_x(ratios, predicted),
        "true_crossover_ratio": plots.crossover_x(ratios, truth),
        "prob_monotone": monotone,
        "prob_monotone_tested": monotone_tested,
        "worst_reversal": worst_reversal,
        "predicted_span": max(predicted) - min(predicted),
        "predicted_slope_at_transition": pred_slope,
        "true_slope_at_transition": true_slope,
    }


def _pool(rows: Sequence[_Row]) -> dict | None:
    return _cell(rows) if rows else None


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


def _wilson_disjoint(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Do two (successes, trials) rates have non-overlapping 95% intervals?

    Used instead of a bare difference so that "symmetric" -- P4's falsifier --
    means "no difference this sample can resolve" rather than "not exactly
    equal", which no finite sample ever is.
    """
    a_lo, a_hi = metrics.wilson_interval(*a)
    b_lo, b_hi = metrics.wilson_interval(*b)
    return a_lo > b_hi or b_lo > a_hi


def _band(rows: Sequence[_Row], lo: float | None, hi: float | None) -> dict | None:
    """Accuracy and mean predicted P(SAT) over a slice of the ratio axis."""
    sel = [
        r
        for r in rows
        if r.ratio is not None and (lo is None or r.ratio >= lo) and (hi is None or r.ratio <= hi)
    ]
    if not sel:
        return None
    return {
        "n": len(sel),
        "accuracy": sum(1 for r in sel if (r.p >= 0.5) == r.truth) / len(sel),
        "mean_predicted": _mean([r.p for r in sel]),
        "true_fraction": sum(1 for r in sel if r.truth) / len(sel),
    }


def _score_p4(rows: Sequence[_Row]) -> dict:
    """P4: asymmetric failure, better at low ratio (SAT) than high (UNSAT).

    Measured per true class rather than per ratio band. The plan's reasoning is
    that satisfiability has a shallow witness and unsatisfiability requires
    exhaustive proof, which is a claim about the answer, not about the density;
    per-class accuracy states it directly and is not reweighted by how many
    instances each ratio happens to contribute. The low and high ratio bands the
    plan names in prose -- "ratio 2" and "ratio 8" -- are reported beside it.
    """
    pred = predictions.BY_ID["P4"]
    sat = [r for r in rows if r.truth]
    unsat = [r for r in rows if not r.truth]
    evidence: dict[str, Any] = {
        "n_sat": len(sat),
        "n_unsat": len(unsat),
        "ratio_le_3": _band(rows, None, 3.0),
        "ratio_ge_7": _band(rows, 7.0, None),
    }
    if not sat or not unsat:
        return _verdict(
            pred,
            "untestable",
            f"the n={PRIMARY_N} sweep returned {len(sat)} SAT and {len(unsat)} UNSAT "
            "scorable instances, so one class has no accuracy to compare",
            evidence,
        )

    sat_hits = sum(1 for r in sat if r.p >= 0.5)
    unsat_hits = sum(1 for r in unsat if r.p < 0.5)
    acc_sat = sat_hits / len(sat)
    acc_unsat = unsat_hits / len(unsat)
    evidence.update(
        {
            "accuracy_on_sat_instances": acc_sat,
            "accuracy_on_unsat_instances": acc_unsat,
            "difference": acc_sat - acc_unsat,
            "predicted_sat_rate": sum(1 for r in rows if r.p >= 0.5) / len(rows),
        }
    )

    if not _wilson_disjoint((sat_hits, len(sat)), (unsat_hits, len(unsat))):
        return _verdict(
            pred,
            "wrong",
            f"symmetric: accuracy {acc_sat:.3f} on SAT instances against "
            f"{acc_unsat:.3f} on UNSAT, a difference of {acc_sat - acc_unsat:+.3f} whose "
            "95% intervals overlap, which is the plan's own falsifier",
            evidence,
        )
    constant = ""
    if evidence["predicted_sat_rate"] > 0.95:
        constant = (
            " -- but the model answered SAT on "
            f"{evidence['predicted_sat_rate']:.1%} of instances, so the asymmetry is "
            "a constant answer rather than a shallower witness"
        )
    elif evidence["predicted_sat_rate"] < 0.05:
        constant = (
            " -- but the model answered UNSAT on "
            f"{1 - evidence['predicted_sat_rate']:.1%} of instances, so the asymmetry is a "
            "constant answer rather than anything about the two classes"
        )
    if acc_sat > acc_unsat:
        note = constant
        return _verdict(
            pred,
            "right",
            f"accuracy {acc_sat:.3f} on SAT instances against {acc_unsat:.3f} on UNSAT, "
            f"a gap of {acc_sat - acc_unsat:+.3f} with non-overlapping 95% intervals{note}",
            evidence,
        )
    return _verdict(
        pred,
        "wrong",
        f"better at high ratio, not low: accuracy {acc_unsat:.3f} on UNSAT instances "
        f"against {acc_sat:.3f} on SAT{constant}",
        evidence,
    )


def _score_p5(matched: dict | None, per_size: dict) -> dict:
    """P5: density-matched accuracy 0.55-0.65; falsified above 0.75.

    The band is tested against the point estimate and the falsifier against the
    interval. Testing the band against the interval instead reads "right" off
    any measurement loose enough to overlap it -- a matched accuracy of 0.43
    with a 0.31-0.56 interval would be scored as confirming a prediction of
    0.55-0.65, which is the opposite of what happened. The interval's job here
    is to say when the measurement cannot separate the band from the falsifier
    at all, which is an untestable result rather than a verdict.
    """
    pred = predictions.BY_ID["P5"]
    if not matched:
        return _verdict(
            pred,
            "untestable",
            "no density-matched instance at n="
            f"{PRIMARY_N} produced a scorable answer, so there is no matched accuracy",
            {"n": 0},
        )
    n = matched["n"]
    acc = matched["accuracy"]
    lo, hi = metrics.wilson_interval(round(acc * n), n)
    evidence = {
        "matched_accuracy": acc,
        "matched_baseline": 0.5,
        "n": n,
        "wilson_95": [lo, hi],
        "by_instance_size": per_size,
    }
    if n < MIN_STABLE_N:
        return _verdict(
            pred,
            "untestable",
            f"matched accuracy {acc:.3f} on only {n} instances (95% CI {lo:.3f}-{hi:.3f}); "
            "the interval spans the predicted band, the falsifier and chance at once",
            evidence,
        )
    if lo > 0.75:
        return _verdict(
            pred,
            "wrong",
            f"matched accuracy {acc:.3f} (95% CI {lo:.3f}-{hi:.3f}, n={n}) is above the "
            "0.75 the plan named as its falsifier -- the model is not merely reading "
            "clause density",
            evidence,
        )
    if lo <= 0.55 and hi >= 0.75:
        return _verdict(
            pred,
            "untestable",
            f"matched accuracy {acc:.3f} on {n} instances has a 95% interval "
            f"{lo:.3f}-{hi:.3f} that covers the predicted band and the 0.75 falsifier at "
            "once, so it cannot tell them apart",
            evidence,
        )
    if 0.55 <= acc <= 0.65:
        return _verdict(
            pred,
            "right",
            f"matched accuracy {acc:.3f} (95% CI {lo:.3f}-{hi:.3f}, n={n}) against the "
            "0.5 balanced baseline falls in the predicted 0.55-0.65 band",
            evidence,
        )
    side = "below" if acc < 0.55 else "above"
    chance = " and below the 0.5 a coin gets on a balanced set" if acc < 0.5 else ""
    # The falsifier branch above is deliberately conservative, testing 0.75
    # against the interval's lower bound. A point estimate over 0.75 whose
    # interval still reaches below it has met the plan's literal condition
    # without establishing it, and reporting only "above the band" would hide
    # that the named falsifier is in play.
    if acc > 0.75:
        chance = (
            f", above the 0.75 the plan named as its falsifier -- though the interval reaches "
            f"down to {lo:.3f}, so this sample does not establish the falsifier"
        )
    return _verdict(
        pred,
        "wrong",
        f"matched accuracy {acc:.3f} (95% CI {lo:.3f}-{hi:.3f}, n={n}) lies {side} the "
        f"predicted 0.55-0.65 band{chance}",
        evidence,
    )


def _score_p6(summary: dict | None) -> dict:
    """P6: the predicted curve is flatter than the true curve at 4.26.

    Falsified by "a sharp crossover within 0.5", so both halves are measured:
    the slope of each curve over the ratios within SLOPE_HALFWIDTH of 4.26, and
    where the predicted curve crosses 0.5.
    """
    pred = predictions.BY_ID["P6"]
    if not summary:
        return _verdict(pred, "untestable", f"no n={PRIMARY_N} sweep was scored", {})
    p_slope = summary["predicted_slope_at_transition"]
    t_slope = summary["true_slope_at_transition"]
    crossover = summary["crossover_ratio"]
    evidence = {
        "predicted_slope_at_transition": p_slope,
        "true_slope_at_transition": t_slope,
        "crossover_ratio": crossover,
        "true_crossover_ratio": summary["true_crossover_ratio"],
        "slope_window": [plots.SAT_TRANSITION - SLOPE_HALFWIDTH, plots.SAT_TRANSITION + SLOPE_HALFWIDTH],
        "flatness_threshold": FLATNESS_RATIO,
        "prob_monotone": summary["prob_monotone"],
        "prob_monotone_tested": summary["prob_monotone_tested"],
    }
    if p_slope is None or t_slope is None or abs(t_slope) < 1e-9:
        return _verdict(
            pred,
            "untestable",
            "the slope window around 4.26 holds too few sampled ratios, or the solver "
            "curve is flat there, so there is nothing to be flatter than",
            evidence,
        )
    ratio = abs(p_slope) / abs(t_slope)
    evidence["flatness_ratio"] = ratio
    near = crossover is not None and abs(crossover - plots.SAT_TRANSITION) <= 0.5
    evidence["crossover_within_0.5_of_4.26"] = near
    evidence["runs_backwards"] = p_slope > 0 > t_slope
    if ratio < FLATNESS_RATIO:
        where = (
            f"crosses 0.5 at {crossover:.2f}" if crossover is not None else "never crosses 0.5"
        )
        # A curve going the other way is literally not dropping as sharply as
        # the true one, so the plan's claim holds -- but reading that as a
        # confirmation without saying which way the curve runs would bury the
        # larger finding, which is that it runs backwards.
        #
        # What that costs the sweep is a separate question and is not assumed
        # here. The slope is fitted over the seven ratios within SLOPE_HALFWIDTH
        # of 4.26; the disqualifier that tiers a sweep "Doesn't work" comes from
        # the all-pairs monotonicity test over all 25, and a local rise well
        # inside that test's tolerance does not trip it. Saying it did would be
        # a claim about the by_difficulty table that the table does not make.
        backwards = ""
        if evidence["runs_backwards"]:
            if not summary["prob_monotone_tested"]:
                consequence = (
                    "whether the curve rises across the sweep as a whole could not be tested "
                    "at this sample size, so no row was tiered on it"
                )
            elif summary["prob_monotone"]:
                consequence = (
                    "the rise stays inside the tolerance of the sweep-wide monotonicity test, "
                    "which compares all sampled ratios rather than this window, so no row was "
                    "tiered \"Doesn't work\" for it"
                )
            else:
                consequence = (
                    "the sweep-wide monotonicity test agrees, and every row of this sweep is "
                    "tiered \"Doesn't work\" for it"
                )
            backwards = (
                " Across the slope window the predicted curve RISES with ratio rather than "
                "falling, so it is not merely flatter than the true curve; it runs the wrong "
                f"way, and {consequence}."
            )
        return _verdict(
            pred,
            "right",
            f"the predicted curve changes by {p_slope:+.3f} per unit ratio against the "
            f"solver curve's {t_slope:+.3f}, {ratio:.2f} times as steep, and {where}."
            + backwards,
            evidence,
        )
    if near:
        return _verdict(
            pred,
            "wrong",
            f"sharp crossover: the predicted curve is {ratio:.2f} times as steep as the "
            f"solver curve at the transition and crosses 0.5 at {crossover:.2f}, within "
            "0.5 of 4.26 -- the plan's explicit falsifier",
            evidence,
        )
    return _verdict(
        pred,
        "wrong",
        f"not flatter: the predicted curve is {ratio:.2f} times as steep as the solver "
        "curve at the transition, though it is offset rather than crossing near 4.26",
        evidence,
    )


def _verdict(pred: predictions.Prediction, verdict: str, outcome: str, evidence: dict) -> dict:
    return {
        "id": pred.id,
        "claim": pred.claim,
        "falsified_if": pred.falsified_if,
        "verdict": verdict,
        "outcome": outcome,
        "evidence": evidence,
    }


# --------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------


def _quantization(rows: Sequence[_Row]) -> dict:
    """Are the returned probabilities on a 2-decimal grid, and do they cluster?

    Both are on the plan's watch list for the report's "unexpected behaviors"
    section, and E2 returns tens of thousands of nouls, which is the largest
    sample of bare probabilities any experiment here produces.
    """
    ps = [r.p for r in rows]
    counts = Counter(round(p, 6) for p in ps)
    on_grid = sum(1 for p in ps if abs(p * 100 - round(p * 100)) < 1e-6)
    return {
        "n": len(ps),
        "distinct_values": len(counts),
        "fraction_on_2dp_grid": on_grid / len(ps),
        "most_common": [
            {"value": v, "share": c / len(ps)} for v, c in counts.most_common(6)
        ],
        "min": min(ps),
        "max": max(ps),
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _emit(
    paths: list[str],
    notes: list[str],
    run_dir: Path,
    out: Path,
    fn: Callable[..., Path],
    *args,
    **kwargs,
) -> None:
    """Render one figure, recording a note instead of losing the run if it fails.

    E2 is the most expensive experiment in the plan. A figure that cannot be
    drawn -- an empty condition at a dry-run scale, a degenerate axis -- must not
    discard 46k calls' worth of scoring, and the log is on disk either way so
    `score` can be re-run once the cause is fixed.

    `out` is passed separately as well as positionally because the plot
    functions take it in different argument positions, and a failure note has to
    be able to name which figure failed.
    """
    try:
        path = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the reason is reported, not swallowed
        notes.append(f"figure {out.name} was not rendered: {exc}")
        return
    paths.append(str(Path(path).relative_to(run_dir)))


def _plots(
    run_dir: Path,
    summaries: dict[int, dict],
    sweep_cells: dict[tuple[int, float], dict],
    matched_cells: dict[tuple[int, float], dict],
    pooled: dict[str, dict | None],
    control: dict | None,
    notes: list[str],
) -> list[str]:
    out: list[str] = []
    plot_dir = run_dir / "plots"

    # The headline figure, primary size first.
    for size in SIZES_SWEPT:
        s = summaries.get(size)
        if not s:
            continue
        _emit(
            out, notes, run_dir, plot_dir / f"e2_curve_n{size}.png",
            plots.calibration_curve_vs_truth,
            s["ratios"], s["predicted"], s["truth"],
            plot_dir / f"e2_curve_n{size}.png",
            n=s["counts"],
            condition=f"random 3SAT, n={size} variables, unmatched sweep",
            crossover=s["crossover_ratio"],
            note=(
                "The solver curve is the target. "
                + (
                    f"It crosses 0.5 at {s['true_crossover_ratio']:.2f} at this size, not at "
                    "4.26, which is the asymptotic threshold."
                    if s["true_crossover_ratio"] is not None
                    else "It does not cross 0.5 within the swept range."
                )
            ),
        )

    for size in SIZES_SWEPT:
        cell = pooled.get(f"sweep-{size}")
        if cell:
            _emit(
                out, notes, run_dir, plot_dir / f"e2_reliability_sweep_n{size}.png",
                plots.reliability_diagram,
                cell["bins"], plot_dir / f"e2_reliability_sweep_n{size}.png",
                f"Reliability, 3SAT unmatched sweep (n={size})",
                condition=f"pooled over ratios 2.0-8.0, n={size} variables",
                ece=cell["ece"], n=cell["n"], monotone=cell["reliability_monotone"],
            )
        matched = pooled.get(f"matched-{size}")
        if matched:
            _emit(
                out, notes, run_dir, plot_dir / f"e2_reliability_matched_n{size}.png",
                plots.reliability_diagram,
                matched["bins"], plot_dir / f"e2_reliability_matched_n{size}.png",
                f"Reliability, density-matched control (n={size})",
                condition=f"equal SAT and UNSAT at each ratio, n={size} variables",
                ece=matched["ece"], n=matched["n"], monotone=matched["reliability_monotone"],
            )

    transition = pooled.get("transition")
    if transition:
        _emit(
            out, notes, run_dir, plot_dir / "e2_reliability_transition_n20.png",
            plots.reliability_diagram,
            transition["bins"], plot_dir / "e2_reliability_transition_n20.png",
            f"Reliability at the transition (n={PRIMARY_N}, ratio 4.0-5.0)",
            condition="the ratios where satisfiability is genuinely uncertain",
            ece=transition["ece"], n=transition["n"],
            monotone=transition["reliability_monotone"],
        )

    if control:
        _emit(
            out, notes, run_dir, plot_dir / "e2_reliability_control.png",
            plots.reliability_diagram,
            control["bins"], plot_dir / "e2_reliability_control.png",
            "Reliability, semantic control (support-ticket routing, hard)",
            condition="the gate condition: ECE above 0.30 here stops the run",
            ece=control["ece"], n=control["n"], monotone=control["reliability_monotone"],
        )

    # ECE against ratio. All three sizes sweep the same 25 ratios, so they share
    # one axis; a size whose sweep is incomplete is left off rather than plotted
    # against the wrong ratios.
    ece_series: dict[str, list[float]] = {}
    ratio_axis: list[float] = []
    axis_counts: list[int] = []
    for size in SIZES_SWEPT:
        s = summaries.get(size)
        if not s:
            continue
        if not ratio_axis:
            ratio_axis, axis_counts = s["ratios"], s["counts"]
        if s["ratios"] == ratio_axis:
            ece_series[f"n={size}"] = [sweep_cells[(size, r)]["ece"] for r in s["ratios"]]
    if ece_series:
        _emit(
            out, notes, run_dir, plot_dir / "e2_ece_by_ratio.png",
            plots.curve,
            ratio_axis, ece_series, plot_dir / "e2_ece_by_ratio.png",
            "Expected calibration error against clause-to-variable ratio",
            "clause-to-variable ratio m/n", "ECE (10 equal-width bins)",
            n=axis_counts,
            n_unit="instances per ratio",
            baseline=config.ECE_TIERS[1][0],
            baseline_label="superhuman tier",
            note="Lower is better. The tier bands are 0.02 / 0.05 / 0.15 / 0.30.",
        )

    # Matched accuracy against ratio. The attainable ratios differ per size, so
    # the axis is their union and a size contributes NaN where it has no control.
    matched_ratios = sorted({r for _, r in matched_cells})
    if matched_ratios:
        series = {}
        for size in SIZES_SWEPT:
            vals = [
                matched_cells[(size, r)]["accuracy"] if (size, r) in matched_cells else float("nan")
                for r in matched_ratios
            ]
            if any(not math.isnan(v) for v in vals):
                series[f"n={size}"] = vals
        _emit(
            out, notes, run_dir, plot_dir / "e2_matched_accuracy.png",
            plots.curve,
            matched_ratios, series, plot_dir / "e2_matched_accuracy.png",
            "Accuracy on the density-matched control",
            "clause-to-variable ratio m/n", "accuracy",
            n=[matched_cells[k]["n"] for k in sorted(matched_cells)],
            n_unit="instances per ratio",
            baseline=0.5, baseline_label="balanced by construction",
            ylim=(0.0, 1.05),
            note="Gaps are ratios where rejection sampling cannot reach both classes.",
        )

    s = summaries.get(PRIMARY_N)
    if s:
        _emit(
            out, notes, run_dir, plot_dir / f"e2_accuracy_n{PRIMARY_N}.png",
            plots.curve,
            s["ratios"],
            {
                "model": [sweep_cells[(PRIMARY_N, r)]["accuracy"] for r in s["ratios"]],
                "clause-density heuristic": [
                    sweep_cells[(PRIMARY_N, r)]["heuristic_baseline"] for r in s["ratios"]
                ],
                "majority class": [
                    sweep_cells[(PRIMARY_N, r)]["majority_baseline"] for r in s["ratios"]
                ],
            },
            plot_dir / f"e2_accuracy_n{PRIMARY_N}.png",
            f"Accuracy on the unmatched sweep against its baselines (n={PRIMARY_N})",
            "clause-to-variable ratio m/n", "accuracy",
            n=s["counts"], n_unit="instances per ratio",
            ylim=(0.0, 1.05),
            note="Beating the density heuristic here is the only thing this figure can show.",
        )
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(run_dir: Path) -> dict:
    """Recompute every E2 number from the JSONL log alone.

    Separate from `run` so the experiment can be rescored, or a metric
    redefined, without re-spending a call. `run` calls this; nothing else is
    computed from the live results.
    """
    run_dir = Path(run_dir)
    rows, failures, stat_records, attempted = _scan(run_dir / LOG_NAME)
    log_stats = logstore.run_stats(stat_records)
    anomalies: list[str] = []
    notes: list[str] = []

    planned: list[dict] = []
    conditions_path = run_dir / CONDITIONS_FILE
    if conditions_path.exists():
        planned = json.loads(conditions_path.read_text())
    else:
        notes.append(
            f"{CONDITIONS_FILE} is absent, so which matched ratios were attempted and "
            "found unattainable cannot be stated; only the ones that produced calls are "
            "known, and calls that were generated but never logged cannot be detected"
        )

    unlogged = _unlogged(planned, attempted)
    if unlogged:
        missing = sum(unlogged.values())
        failures["calls"] += missing
        failures["excluded"] += missing
        failures["reasons"][UNLOGGED_REASON] = missing
        failures["unlogged_by_condition"] = unlogged
        notes.append(
            f"{missing} calls across {len(unlogged)} conditions were generated but never "
            "reached the log, so the run did not finish what it planned. They are counted as "
            "failures and excluded; the affected conditions rest on fewer answers than "
            "their requested size. Worst: "
            + ", ".join(
                f"{c} ({k} missing)"
                for c, k in sorted(unlogged.items(), key=lambda kv: -kv[1])[:6]
            )
            + "."
        )
    abandoned = failures["reasons"].get(ABANDONED_REASON, 0)
    if abandoned:
        notes.append(
            f"{abandoned} calls exhausted their retries and never produced a terminal record. "
            "The client marks every attempt of such a call, including the last, as retry "
            "history, so they are recovered from that history and counted as failed calls "
            "rather than being absent from both the scored rows and the failure count."
        )

    if not rows:
        return _empty_result(failures, notes, log_stats)

    sweep_cells = _by_cell(rows, "sweep")
    matched_cells = _by_cell(rows, "matched")
    control = _pool([r for r in rows if r.kind == "control"])
    summaries = {size: s for size in SIZES_SWEPT if (s := _sweep_summary(sweep_cells, size))}

    pooled: dict[str, dict | None] = {}
    for size in SIZES_SWEPT:
        pooled[f"sweep-{size}"] = _pool([r for r in rows if r.kind == "sweep" and r.n_vars == size])
        pooled[f"matched-{size}"] = _pool(
            [r for r in rows if r.kind == "matched" and r.n_vars == size]
        )
    pooled["transition"] = _pool(
        [
            r
            for r in rows
            if r.kind == "sweep"
            and r.n_vars == PRIMARY_N
            and r.ratio is not None
            and 4.0 <= r.ratio <= 5.0
        ]
    )
    pooled["control"] = control

    # -- tier per condition ----------------------------------------------
    by_difficulty: list[dict] = []
    crossings: dict[str, Any] = {}
    crossing_detail: dict[str, Any] = {}
    for size in SIZES_SWEPT:
        s = summaries.get(size)
        if not s:
            continue
        results = []
        for ratio in s["ratios"]:
            results.append(
                tiers.assign(
                    EXPERIMENT,
                    _row_metrics(sweep_cells[(size, ratio)], matched_cells.get((size, ratio)), s),
                    difficulty={"n": size, "ratio": ratio},
                )
            )
        by_difficulty.extend(_row_json(r) for r in results)
        found = tiers.boundary_crossings(EXPERIMENT, results)
        crossing_detail.update(
            {f"n={size} {b}": (c.to_json() if c else None) for b, c in found.items()}
        )
        # Built here rather than with tiers.crossings_json because difficulty is
        # not monotone in ratio on this sweep -- instances are easy at both ends
        # and hardest near the transition -- so "crossed at ratio 2.5" alone
        # reads as "worse from 2.5 onward", which is the opposite of what
        # happens. Carrying the recovery ratio in the value says so on the line
        # the report prints.
        for boundary, c in found.items():
            if c is None:
                continue
            value = dict(c.difficulty) if isinstance(c.difficulty, dict) else {"at": c.difficulty}
            if isinstance(c.last_above, dict):
                value["last_above_ratio"] = c.last_above.get("ratio")
            if isinstance(c.recovered_at, dict):
                value["recovers_at_ratio"] = c.recovered_at.get("ratio")
            crossings[f"n={size} {boundary}"] = value

    if control:
        # Graded with the calibration rubric rather than E2's: the control has
        # no matched set and no ratio, and the only question asked of it is
        # whether its probabilities mean what they say.
        control_row = tiers.assign(
            "calibration",
            {
                "ece": control["ece"],
                "reliability_monotone": control["reliability_monotone"],
                "ece_baseline": 0.0,
                "n": control["n"],
                "brier": control["brier"],
                "auroc": control["auroc"],
                "accuracy": control["accuracy"],
                "chance_baseline": max(control["random_baseline"], control["majority_baseline"]),
                "heuristic_baseline": control["heuristic_baseline"],
            },
            difficulty={"set": "semantic control", "level": CONTROL_LEVEL},
        )
        by_difficulty.append(_row_json(control_row))

    # -- headline: the matched set at the primary size --------------------
    primary_matched = pooled.get(f"matched-{PRIMARY_N}")
    headline = {
        "metric": "matched_accuracy",
        "value": primary_matched["accuracy"] if primary_matched else None,
        "baseline": 0.5,
        "baseline_name": (
            f"density-matched control at n={PRIMARY_N}, balanced by construction "
            "(majority class and clause-density heuristic both exactly 0.5)"
        ),
        "n": primary_matched["n"] if primary_matched else 0,
    }

    # -- predictions -------------------------------------------------------
    primary_rows = [r for r in rows if r.kind == "sweep" and r.n_vars == PRIMARY_N]
    matched_by_size = {
        f"n={size}": {
            "accuracy": pooled[f"matched-{size}"]["accuracy"],
            "n": pooled[f"matched-{size}"]["n"],
        }
        for size in SIZES_SWEPT
        if pooled.get(f"matched-{size}")
    }
    scored = [
        _score_p4(primary_rows),
        _score_p5(primary_matched, matched_by_size),
        _score_p6(summaries.get(PRIMARY_N)),
    ]
    _assert_every_prediction_scored(scored)

    # -- gate ---------------------------------------------------------------
    gate = _gate(control)

    # -- anomalies ----------------------------------------------------------
    quant = _quantization(rows)
    anomalies.extend(
        _anomalies(
            quant,
            summaries,
            sweep_cells,
            matched_cells,
            pooled,
            planned,
            by_difficulty,
            log_stats,
        )
    )
    anomalies.extend(notes)

    return {
        "experiment": EXPERIMENT,
        "question": QUESTION,
        "headline": headline,
        "by_difficulty": by_difficulty,
        "boundary_crossings": crossings,
        "boundary_crossings_detail": crossing_detail,
        "plots": _plots(run_dir, summaries, sweep_cells, matched_cells, pooled, control, anomalies),
        "predictions": scored,
        "anomalies": anomalies,
        "failures": failures,
        "gate": gate,
        "summary": _summary(pooled, summaries, control, quant),
        "sweeps": {str(size): s for size, s in summaries.items()},
        "conditions": planned,
        "sample_sizes": _sample_sizes(sweep_cells, matched_cells, control, planned),
        "log": log_stats,
    }


def _sample_sizes(
    sweep_cells: dict[tuple[int, float], dict],
    matched_cells: dict[tuple[int, float], dict],
    control: dict | None,
    planned: Sequence[dict],
) -> dict:
    """The sample sizes the numbers actually rest on.

    Read from what was scored rather than from `config.n` at scoring time. The
    two differ whenever a log is rescored in a shell with a different
    `JEV_SCALE`, and reporting the configured size next to numbers computed from
    a tenth as much data is exactly the silently-shrunk sample the plan warns
    about. `requested_per_ratio` comes from the conditions file written when the
    calls were generated, so the gap between asked for and obtained is visible.
    """

    def span(cells: dict[tuple[int, float], dict]) -> dict | None:
        counts = [c["n"] for c in cells.values()]
        if not counts:
            return None
        return {"min": min(counts), "max": max(counts), "conditions": len(counts)}

    requested = {
        kind: sorted({c["requested"] for c in planned if c.get("kind") == kind})
        for kind in ("sweep", "matched", "control")
    }
    return {
        "source": "scored predictions in the log; requested_per_ratio from "
        f"{CONDITIONS_FILE}, written when the calls were generated",
        "sweep_per_ratio": span(sweep_cells),
        "matched_per_ratio": span(matched_cells),
        "control": control["n"] if control else 0,
        "requested_per_ratio": {k: v for k, v in requested.items() if v},
    }


def _row_json(result: tiers.TierResult) -> dict:
    """One by_difficulty row. `n` is dropped from the metric bag because the row
    already carries it as its own field, and the report would print it twice."""
    row = result.to_json()
    row["metrics"].pop("n", None)
    return row


def _row_metrics(cell: dict, matched: dict | None, summary: dict) -> dict:
    """The metric bag `tiers.assign` grades one ratio on.

    Ordered so that the four the report prints are the four that matter.

    `accuracy` and `chance_baseline` are supplied only where a matched control
    exists, and are the matched numbers. The global "at or below chance" row
    would otherwise fire on every ratio outside the transition, where the solver
    decides every instance the same way and no model can beat the base rate --
    an unmeasurable cell, not a failure. The unmatched accuracy and its majority
    and density baselines are still reported in the row for the reader.
    """
    out: dict[str, Any] = {
        # Insertion order is the order the report's table discovers columns in,
        # and it prints the first four. AUROC and matched accuracy lead because
        # they are the two that matter most, but neither exists at a ratio where
        # the solver decided every instance alike, so the four that actually
        # reach the table on most sweeps are the four after them: calibration
        # error, sharpness, accuracy, and the predicted probability itself.
        "ece": cell["ece"],
        "brier": cell["brier"],
        "auroc": cell["auroc"],
        "matched_accuracy": matched["accuracy"] if matched else None,
        "unmatched_accuracy": cell["accuracy"],
        "mean_predicted": cell["mean_predicted"],
        "true_fraction": cell["true_fraction"],
        "majority_baseline": cell["majority_baseline"],
        "density_heuristic_baseline": cell["heuristic_baseline"],
        "curve_max_abs_error": summary["curve_max_abs_error"],
        "crossover_ratio": summary["crossover_ratio"],
        "true_crossover_ratio": summary["true_crossover_ratio"],
        # None, not True, when the monotonicity test could not run: fewer than
        # three ratios reached MONOTONE_MIN_COUNT. `tiers.assign` treats None as
        # a metric it was not given and flags the row as tiered on part of its
        # rule, which is the honest report. Passing the default True would tell
        # the rubric the curve had been checked and had passed.
        "prob_monotone": summary["prob_monotone"] if summary["prob_monotone_tested"] else None,
        "reliability_monotone": cell["reliability_monotone"],
        "n": cell["n"],
    }
    if matched:
        out["matched_baseline"] = 0.5
        out["matched_n"] = matched["n"]
        out["accuracy"] = matched["accuracy"]
        out["chance_baseline"] = 0.5
        out["heuristic_baseline"] = 0.5
    return out


def _gate(control: dict | None) -> dict:
    """Phase 1's calibration gate, read off the semantic control.

    The plan gates on ECE above 0.30 on the control and says in as many words
    that bad calibration on 3SAT alone is not a gate. A reliability curve that
    is not monotone trips it too: the plan's own calibration ladder makes a
    non-monotone curve "Doesn't work" whatever its ECE, and that is the same
    statement as "the probabilities do not mean anything here", which is what
    the gate exists to detect. `metrics.is_monotone` requires the reversal to
    clear both bins' Wilson intervals, so this does not fire on noise.
    """
    if control is None:
        return {
            "passed": True,
            "evaluated": False,
            "reason": "the semantic control produced no scorable answer, so the calibration "
            "gate was NOT evaluated. It is reported as passed only because there is nothing "
            "to fail it on; treat Phase 1 as ungated until the control runs.",
        }
    ece = control["ece"]
    n = control["n"]
    if n < MIN_GATE_N:
        return {
            "passed": True,
            "evaluated": False,
            "control_ece": ece,
            "n": n,
            "reason": f"semantic control ECE {ece:.3f} on only {n} predictions; the gate needs "
            f"at least {MIN_GATE_N} for a 10-bin ECE to mean anything (the plan asks for 500), "
            "so it was NOT evaluated.",
        }
    failures = []
    if ece > GATE_ECE:
        failures.append(f"ECE {ece:.3f} exceeds {GATE_ECE:.2f}")
    if not control["reliability_monotone"]:
        failures.append(
            "the control's reliability curve is not monotone, which the plan's calibration "
            "ladder tiers as 'Doesn't work' regardless of ECE"
        )
    if failures:
        return {
            "passed": False,
            "evaluated": True,
            "control_ece": ece,
            "n": n,
            "reason": "Calibration does not hold on the semantic control: "
            + "; ".join(failures)
            + ". Per the plan the core value proposition does not hold on this data and the "
            "remaining experiments are academic.",
        }
    return {
        "passed": True,
        "evaluated": True,
        "control_ece": ece,
        "n": n,
        "reason": f"semantic control ECE {ece:.3f} at n={n} is within the {GATE_ECE:.2f} gate "
        f"({tiers.calibration_tier(ece, monotone=True)} on the calibration ladder), and its "
        "reliability curve is monotone. 3SAT calibration is not gated on: the plan calls bad "
        "calibration there an expected off-distribution result.",
    }


def _assert_every_prediction_scored(scored: Sequence[dict]) -> None:
    """Every E2 prediction must carry a verdict, including an untestable one.

    Quietly dropping a prediction the experiment failed to test is the one
    failure mode the report cannot detect for itself.
    """
    wanted = {p.id for p in predictions.for_experiment(EXPERIMENT)}
    got = {s["id"] for s in scored}
    if wanted != got:
        raise AssertionError(
            f"E2 must score {sorted(wanted)}; scored {sorted(got)}"
        )


def _difficulty_str(d: Any) -> str:
    if isinstance(d, dict):
        return ", ".join(f"{k}={v}" for k, v in sorted(d.items()))
    return str(d)


def _anomalies(
    quant: dict,
    summaries: dict[int, dict],
    sweep_cells: dict[tuple[int, float], dict],
    matched_cells: dict[tuple[int, float], dict],
    pooled: dict[str, dict | None],
    planned: Sequence[dict],
    by_difficulty: Sequence[dict],
    log_stats: dict,
) -> list[str]:
    out: list[str] = []

    if quant["fraction_on_2dp_grid"] >= 0.999:
        out.append(
            f"Every one of the {quant['n']} noul probabilities returned in E2 lies exactly on "
            f"a 2-decimal grid ({quant['distinct_values']} distinct values between "
            f"{quant['min']:.2f} and {quant['max']:.2f}). The probabilities are quantized to "
            "two decimals, which puts a floor of about 0.005 under any ECE."
        )
    elif quant["fraction_on_2dp_grid"] < 0.99:
        out.append(
            f"{1 - quant['fraction_on_2dp_grid']:.1%} of returned probabilities are off the "
            "2-decimal grid, contradicting the quantization seen elsewhere in this harness."
        )
    top = quant["most_common"][0]
    if top["share"] >= 0.15:
        out.append(
            f"Returned probabilities cluster: {top['share']:.1%} of all E2 answers are exactly "
            f"{top['value']:.2f}. Top values: "
            + ", ".join(f"{m['value']:.2f} ({m['share']:.1%})" for m in quant["most_common"][:4])
            + "."
        )

    out.append(
        "E2 asks only nouls, and a noul answer carries no confidence field, so this "
        "experiment contributes no evidence either way on whether `confidence` diverges from "
        "max-probability. Where a confidence signal is needed here the distance of the "
        "probability from 0.5 is the only one available."
    )

    for size, s in summaries.items():
        w = s["worst_reversal"]
        if not s["prob_monotone"]:
            out.append(
                f"Mean predicted P(SAT) is NOT monotone in ratio at n={size}: it reads "
                f"{w['to_mean']:.3f} at ratio {w['to_ratio']:g} against {w['from_mean']:.3f} at "
                f"ratio {w['from_ratio']:g}, a rise of {w['rise']:.3f} against a tolerance of "
                f"{w['tolerance']:.3f}. The plan tiers this whole sweep 'Doesn't work' "
                "regardless of its ECE."
            )
        elif not s["prob_monotone_tested"]:
            out.append(
                f"Monotonicity of the predicted curve at n={size} could not be tested: fewer "
                f"than three ratios reached {MONOTONE_MIN_COUNT} scorable instances."
            )
        elif w is not None and w["rise"] > MONOTONE_MIN_STEP:
            out.append(
                f"The predicted curve at n={size} rises by {w['rise']:.3f} between ratio "
                f"{w['from_ratio']:g} and {w['to_ratio']:g}, which is more than one "
                "quantization step but inside the tolerance the monotonicity test applies, "
                "so the sweep was not tiered non-monotone on it."
            )
        if s["predicted_span"] < 0.05:
            out.append(
                f"At n={size} mean predicted P(SAT) spans only {s['predicted_span']:.3f} across "
                f"ratios {s['ratios'][0]:g}-{s['ratios'][-1]:g}, against a true satisfiable "
                f"fraction that spans {max(s['truth']) - min(s['truth']):.3f}. The probability "
                "barely responds to the input at all."
            )
        if s["crossover_ratio"] is None:
            out.append(
                f"At n={size} the predicted curve never crosses 0.5 across ratios "
                f"{s['ratios'][0]:g}-{s['ratios'][-1]:g} (mean predicted P(SAT) runs "
                f"{max(s['predicted']):.2f} down to {min(s['predicted']):.2f}), so it has no "
                "phase transition to compare against 4.26."
            )
        elif s["true_crossover_ratio"] is not None:
            gap = s["crossover_ratio"] - s["true_crossover_ratio"]
            if abs(gap) > 0.25:
                out.append(
                    f"At n={size} the predicted curve crosses 0.5 at {s['crossover_ratio']:.2f} "
                    f"while the solver curve crosses at {s['true_crossover_ratio']:.2f} "
                    f"({gap:+.2f}). The rubric scores the crossover against the asymptotic "
                    "4.26, but the empirical threshold at finite n is the honest comparison."
                )

    for size in SIZES_SWEPT:
        cell = pooled.get(f"sweep-{size}")
        if cell and not cell["reliability_monotone"]:
            out.append(
                f"The pooled reliability curve for the n={size} sweep is not monotone: observed "
                "frequency falls as predicted probability rises, by more than sampling noise "
                "explains. Anti-correlated calibration is 'Doesn't work' whatever the ECE."
            )
        matched = pooled.get(f"matched-{size}")
        if matched and matched["accuracy"] < 0.5 and matched["n"] >= MIN_STABLE_N:
            lo, _ = metrics.wilson_interval(
                round(matched["accuracy"] * matched["n"]), matched["n"]
            )
            out.append(
                f"Density-matched accuracy at n={size} is {matched['accuracy']:.3f}, below the "
                f"0.5 a coin gets on a balanced set (n={matched['n']}, 95% lower bound {lo:.3f})."
            )

    beaten = [
        (size, ratio)
        for (size, ratio), cell in sweep_cells.items()
        if cell["heuristic_baseline"] is not None
        and cell["accuracy"] + 0.02 < cell["heuristic_baseline"]
    ]
    if beaten:
        out.append(
            f"The clause-density heuristic -- read m and n from the DIMACS header, predict SAT "
            f"below {sat3.DENSITY_THRESHOLD} -- beats the model by more than 2 points at "
            f"{len(beaten)} of {len(sweep_cells)} sweep cells. Per the rubric that is the 'Bad' "
            "row: you would be better off with 20 lines of code."
        )

    unattainable = [c for c in planned if c.get("kind") == "matched" and not c.get("attainable")]
    short = [c for c in planned if c.get("kind") == "matched" and c.get("short")]
    if unattainable:
        by_size: dict[Any, list[str]] = {}
        for c in unattainable:
            by_size.setdefault(c["n"], []).append(f"{c['ratio']:g}")
        out.append(
            "No density-matched control exists at these ratios, where rejection sampling "
            "cannot reach both classes: "
            + "; ".join(f"n={k}: {', '.join(v)}" for k, v in sorted(by_size.items()))
            + ". The matched result covers only the ratios listed as attainable."
        )
    if short:
        out.append(
            f"{len(short)} matched conditions returned fewer instances than requested but "
            "stayed balanced: "
            + ", ".join(f"n={c['n']} r={c['ratio']:g} ({c['generated']})" for c in short[:8])
            + ("..." if len(short) > 8 else "")
        )

    versions = log_stats["model_versions"]
    if len(versions) > 1:
        out.append(
            f"More than one model version answered during E2 ({', '.join(versions)}). "
            "Calibration numbers pooled across versions are not comparable."
        )

    partial = [r for r in by_difficulty if not r.get("fully_evaluated", True)]
    if partial:
        out.append(
            f"{len(partial)} of {len(by_difficulty)} conditions were tiered on part of their "
            "rung's rule, almost always because no density-matched control exists at that "
            "ratio. A tier awarded on ECE alone says the probabilities are well shaped there; "
            "it is not evidence that anything other than clause density produced them. "
            "Examples: "
            + "; ".join(
                f"{_difficulty_str(r['difficulty'])} {r['tier']} "
                f"(unmeasured: {', '.join(r['unevaluated'])})"
                for r in partial[:4]
            )
            + "."
        )

    thin = [
        f"n={size} r={ratio:g}"
        for (size, ratio), cell in sweep_cells.items()
        if not cell["stable"]
    ]
    if thin:
        out.append(
            f"{len(thin)} of {len(sweep_cells)} sweep cells hold fewer than {MIN_STABLE_N} "
            "scorable predictions, so their ECE is reported but is not a measurement "
            f"(the plan asks for 500 per condition). First few: {', '.join(thin[:6])}."
        )

    thin_matched = [
        f"n={size} r={ratio:g} ({cell['n']})"
        for (size, ratio), cell in matched_cells.items()
        if not cell["stable"]
    ]
    if thin_matched:
        out.append(
            f"{len(thin_matched)} of {len(matched_cells)} density-matched cells hold fewer "
            f"than {MIN_STABLE_N} scorable predictions. Matched accuracy is this experiment's "
            "headline metric and the rubric reads it for the 'at or below chance' rung, so "
            "wherever it is this thin the tier at that ratio rests on a handful of instances "
            "rather than on a measurement -- a matched cell of two is one coin flip away from "
            f"tiering its ratio 'Doesn't work'. First few: {', '.join(thin_matched[:6])}."
        )
    return out


def _summary(
    pooled: dict[str, dict | None],
    summaries: dict[int, dict],
    control: dict | None,
    quant: dict,
) -> dict:
    """The numbers format_report prints and the report's prose quotes."""

    def pick(cell: dict | None) -> dict | None:
        if not cell:
            return None
        return {
            k: cell[k]
            for k in (
                "n",
                "ece",
                "brier",
                "auroc",
                "accuracy",
                "mean_predicted",
                "true_fraction",
                "majority_baseline",
                "random_baseline",
                "heuristic_baseline",
                "reliability_monotone",
                "latency",
            )
        }

    out: dict[str, Any] = {
        "pooled": {k: pick(v) for k, v in pooled.items()},
        "calibration_tier": {},
        "probability_values": quant,
    }
    for key, cell in pooled.items():
        if cell:
            out["calibration_tier"][key] = tiers.calibration_tier(
                cell["ece"], monotone=cell["reliability_monotone"]
            )
    for size, s in summaries.items():
        out[f"curve_n{size}"] = {
            "curve_max_abs_error": s["curve_max_abs_error"],
            "crossover_ratio": s["crossover_ratio"],
            "true_crossover_ratio": s["true_crossover_ratio"],
            "prob_monotone": s["prob_monotone"],
            "prob_monotone_tested": s["prob_monotone_tested"],
            "predicted_slope_at_transition": s["predicted_slope_at_transition"],
            "true_slope_at_transition": s["true_slope_at_transition"],
        }
    if control:
        out["gate_metric"] = {"control_ece": control["ece"], "n": control["n"]}
    return out


def _empty_result(failures: dict, notes: list[str], log_stats: dict) -> dict:
    """What to return when nothing was scorable.

    Reported as an absence rather than as a zero: an experiment with no data has
    no tier, no headline and no gate verdict, and saying so is different from
    saying the model scored badly.
    """
    untestable = [
        _verdict(p, "untestable", "no scorable E2 prediction was logged", {})
        for p in predictions.for_experiment(EXPERIMENT)
    ]
    return {
        "experiment": EXPERIMENT,
        "question": QUESTION,
        "headline": {
            "metric": "matched_accuracy",
            "value": None,
            "baseline": 0.5,
            "baseline_name": "density-matched control, balanced by construction",
            "n": 0,
        },
        "by_difficulty": [],
        "boundary_crossings": {},
        "plots": [],
        "predictions": untestable,
        "anomalies": notes
        + ["E2 produced no scorable answer; every number below is absent, not zero."],
        "failures": failures,
        "gate": {
            "passed": True,
            "evaluated": False,
            "reason": "E2 logged no scorable answer, so the calibration gate was NOT "
            "evaluated. Treat Phase 1 as ungated.",
        },
        "log": log_stats,
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _submission_order(plan: _Plan) -> list[Call]:
    """Every call in one list, interleaved across conditions under a fixed seed.

    The instances are generated ratio by ratio and size by size, and submitting
    them in that order would make elapsed time a near-perfect proxy for ratio
    across a run of tens of thousands of calls. Any drift in the service during
    it -- a silent model change, a cache warming, a degraded shard -- would then
    land in one contiguous stretch of the ratio axis and be read straight off
    the headline figure as a property of the ratio. E1 interleaves its four
    conditions for the same reason.

    The shuffle is seeded from the master seed, so the submission order is
    reproducible, and every call carries its own condition in the log, so
    nothing in `score` depends on the order calls were sent in.
    """
    calls = [c for group in plan.groups for c in group.calls]
    random.Random(config.seed_for(EXPERIMENT, "submission-order")).shuffle(calls)
    return calls


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    plan = _build_plan()
    calls = _submission_order(plan)
    (run_dir / CONDITIONS_FILE).write_text(json.dumps(plan.conditions, indent=2))
    print(f"  [E2] {len(calls)} calls across {len(plan.conditions)} conditions")
    for group in plan.groups:
        if group.calls:
            print(f"  [E2]   {group.label}: {len(group.calls)} calls")
    print("  [E2] submitted in one seeded shuffle, not in ratio order, so that "
          "drift during the run does not read as a ratio effect")
    for note in plan.notes:
        print(f"  [E2] {note}")

    with JevClient(run_dir=run_dir, log_name=LOG_NAME) as client:
        client.run(calls, progress_every=250, label="E2")
        summary = client.summary()

    result = score(run_dir)
    result["run"] = summary
    result["generation_notes"] = plan.notes
    return result


def format_report(result: dict) -> str:
    lines: list[str] = [f"E2 calibration -- {result['question']}"]
    s = result.get("summary") or {}
    pooled = s.get("pooled") or {}
    tier = s.get("calibration_tier") or {}

    gate = result.get("gate") or {}
    state = "passed" if gate.get("passed") else "TRIPPED"
    if not gate.get("evaluated", True):
        state = "NOT EVALUATED"
    lines.append(f"  gate ({state}): {gate.get('reason', '')}")

    h = result.get("headline") or {}
    val = h.get("value")
    lines.append(
        "  headline: matched accuracy "
        + (f"{val:.3f}" if isinstance(val, float) else "n/a")
        + f" vs 0.500 balanced baseline, n={h.get('n', 0)}"
    )

    for key, label in (
        (f"sweep-{PRIMARY_N}", f"3SAT sweep n={PRIMARY_N}"),
        ("transition", "3SAT at the transition"),
        (f"matched-{PRIMARY_N}", f"matched control n={PRIMARY_N}"),
        ("control", "semantic control (hard)"),
    ):
        cell = pooled.get(key)
        if not cell:
            continue
        auroc = cell["auroc"]
        lines.append(
            f"  {label:<28} n={cell['n']:<6} ECE {cell['ece']:.3f} [{tier.get(key, '?')}]  "
            f"Brier {cell['brier']:.3f}  "
            f"AUROC {'n/a' if auroc is None or math.isnan(auroc) else f'{auroc:.3f}'}  "
            f"acc {cell['accuracy']:.3f} (majority {cell['majority_baseline']:.3f}, "
            f"heuristic "
            + (
                f"{cell['heuristic_baseline']:.3f}"
                if cell["heuristic_baseline"] is not None
                else "n/a"
            )
            + ")"
        )

    lines.append(
        "  (AUROC on the unmatched sweep is confounded by clause density, which predicts "
        "satisfiability on its own; the matched row is the density-free discrimination.)"
    )

    for size in SIZES_SWEPT:
        curve = s.get(f"curve_n{size}")
        if not curve:
            continue
        cross = curve["crossover_ratio"]
        true_cross = curve["true_crossover_ratio"]
        lines.append(
            f"  curve n={size:<3} max |predicted - true| {curve['curve_max_abs_error']:.3f}  "
            f"crossover "
            + (f"{cross:.2f}" if cross is not None else "never")
            + " vs solver "
            + (f"{true_cross:.2f}" if true_cross is not None else "never")
            + "  monotone in ratio: "
            + (
                str(curve["prob_monotone"])
                if curve.get("prob_monotone_tested", True)
                else "untested (too few instances per ratio)"
            )
        )

    for p in result.get("predictions") or []:
        mark = {"right": "right", "wrong": "WRONG", "untestable": "untestable"}.get(
            p["verdict"], p["verdict"]
        )
        lines.append(f"  {p['id']} {mark}: {p['outcome']}")

    f = result.get("failures") or {}
    lines.append(f"  failed calls: {f.get('calls', 0)}, excluded: {f.get('excluded', 0)}")
    for a in result.get("anomalies") or []:
        lines.append(f"  ! {a}")
    return "\n".join(lines)
