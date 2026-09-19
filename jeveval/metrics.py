"""Scoring metrics shared by every experiment.

The plan defines its metrics once and has nine experiments reference them, so
they are implemented once here rather than inline per experiment. Three
consequences shape this module:

*ECE and the reliability diagram come from the same binning.* `ece` is computed
from the output of `reliability_bins`, not alongside it. A headline ECE that
disagreed with the plotted diagram would be very hard to notice and would
invalidate the most important number in the report.

*No sklearn, no scipy.* The library is stdlib only. The plan requires that
metrics be recomputable offline from the JSONL log with a single command, and a
reader reproducing a number should not have to reconstruct the solver and
plotting stack to do it. AUROC is therefore computed from the Mann-Whitney U
identity with average ranks for ties, which is exact rather than approximate.
The self-check below does cross-check against scipy and numpy where they cover
the same ground, but skips those comparisons if the imports fail.

*Baselines and intervals live here too, not only the headline metrics.* The
rubric refuses a tier assignment that is not reported next to its baseline, and
an accuracy reported without an interval invites over-reading a 3-point
difference that is inside the noise floor E1 measures. `majority_baseline`,
`random_baseline`, `wilson_interval`, `mcnemar` and `paired_bootstrap` are here
so that no experiment has to reimplement them to satisfy that.

Bad input raises rather than returning a sentinel. The one exception is `auroc`,
which returns NaN when a condition contains only one class, because that is a
legitimate property of the data and not a caller error -- see its docstring.

Run the self-check with:  python -m jeveval.metrics
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from collections import Counter
from functools import lru_cache
from collections.abc import Callable, Sequence
from statistics import NormalDist
from typing import Any

from . import config

__all__ = [
    "accuracy",
    "auroc",
    "brier",
    "cost_per_decision",
    "ece",
    "ece_tier",
    "entropy",
    "is_monotone",
    "kl_divergence",
    "majority_baseline",
    "mcnemar",
    "paired_bootstrap",
    "percentiles",
    "random_baseline",
    "reliability_bins",
    "wilson_interval",
]


# --------------------------------------------------------------------------
# Input coercion
# --------------------------------------------------------------------------


def _as_probs(probs: Sequence[float], name: str = "probs") -> list[float]:
    out = []
    for p in probs:
        p = float(p)
        if math.isnan(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"{name} contains {p!r}, which is not a probability in [0, 1]")
        out.append(p)
    return out


def _as_outcomes(outcomes: Sequence[Any], name: str = "outcomes") -> list[float]:
    """Accept bools (noul ground truth is a bool) or 0/1, nothing else."""
    out = []
    for o in outcomes:
        if isinstance(o, bool):
            out.append(1.0 if o else 0.0)
        elif o == 0 or o == 1:
            out.append(float(o))
        else:
            raise ValueError(f"{name} contains {o!r}; outcomes must be bool or 0/1")
    return out


def _pairs(probs: Sequence[float], outcomes: Sequence[Any]) -> tuple[list[float], list[float]]:
    if len(probs) != len(outcomes):
        raise ValueError(f"length mismatch: {len(probs)} probs, {len(outcomes)} outcomes")
    # `len(...) == 0` rather than `not probs`: an experiment holding its
    # predictions in a numpy array would otherwise get numpy's "truth value of
    # an array is ambiguous" out of the emptiness guard, which reads like a
    # metrics bug and hides the real call. Everything downstream already accepts
    # numpy scalars, so the array itself is fine.
    if len(probs) == 0:
        raise ValueError("no predictions to score")
    return _as_probs(probs), _as_outcomes(outcomes)


def _as_distribution(xs: Sequence[float], tol: float, name: str) -> list[float]:
    vals = [float(x) for x in xs]
    if not vals:
        raise ValueError(f"{name} is empty")
    for v in vals:
        if math.isnan(v) or v < 0.0:
            raise ValueError(f"{name} contains {v!r}; a distribution must be non-negative")
    total = sum(vals)
    if total <= 0.0:
        raise ValueError(f"{name} sums to {total!r}")
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"{name} sums to {total!r}, not 1 within tol={tol!r}. "
            "Pass a looser tol if the source rounds its probabilities."
        )
    return [v / total for v in vals]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def _bin_index(p: float, edges: Sequence[float]) -> int:
    """Index of the equal-width bin holding `p`, lower edge inclusive.

    Computed by bisecting the explicit edge list rather than as `int(p *
    n_bins)`. The multiplication is not reliably exact: `(29 / 100) * 100` is
    28.999999999999996, so a prediction of exactly 0.29 lands in [0.28, 0.29)
    under the arithmetic version. The smallest n_bins where this bites is 22,
    and it bites 3 edges at n_bins=100 -- but *not* at n_bins 10 or 3, where the
    products happen to round back exactly. So the plan's own 10 bins cannot
    demonstrate the difference, which is why the self-check asserts the edge
    cases at 22 and 100 as well: a future reader who tests only at 10 will
    conclude the bisect is unnecessary and reintroduce the bug for every finer
    diagram.

    A prediction sitting exactly on an edge belongs to the bin above it.
    """
    i = bisect_right(edges, p) - 1
    return min(max(i, 0), len(edges) - 2)


def reliability_bins(
    probs: Sequence[float],
    outcomes: Sequence[Any],
    n_bins: int = 10,
    *,
    conf: float = 0.95,
) -> list[dict]:
    """Per-bin calibration summary -- what the reliability diagram plots.

    Bins are equal-width over [0, 1]: [0, 0.1), [0.1, 0.2), ... [0.9, 1.0], with
    the top bin closed so that a prediction of exactly 1.0 has somewhere to go.

    Empty bins are returned too, with count 0 and None for every statistic. The
    diagram needs the full axis, and a bin with no mass is itself worth seeing:
    the plan flags probabilities clustering at particular values as an
    unexpected behavior to report, and that shows up as empty bins.

    Each bin carries a Wilson interval on its observed frequency, so the diagram
    can show error bars and `is_monotone` can tell a real reversal from noise.
    """
    ps, ys = _pairs(probs, outcomes)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    edges = [i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    sum_p = [0.0] * n_bins
    sum_y = [0.0] * n_bins
    for p, y in zip(ps, ys):
        b = _bin_index(p, edges)
        counts[b] += 1
        sum_p[b] += p
        sum_y[b] += y

    total = len(ps)
    out: list[dict] = []
    for i in range(n_bins):
        c = counts[i]
        if c:
            mean_p = sum_p[i] / c
            observed = sum_y[i] / c
            lo, hi = wilson_interval(int(round(sum_y[i])), c, conf)
            gap = abs(mean_p - observed)
        else:
            mean_p = observed = lo = hi = gap = None
        out.append(
            {
                "bin": i,
                "lo_edge": edges[i],
                "hi_edge": edges[i + 1],
                "count": c,
                "weight": c / total,
                "mean_predicted": mean_p,
                "observed": observed,
                "observed_lo": lo,
                "observed_hi": hi,
                "gap": gap,
            }
        )
    return out


def ece(probs: Sequence[float], outcomes: Sequence[Any], n_bins: int = 10) -> float:
    """Expected calibration error, exactly as the plan defines it.

    Ten equal-width probability bins; the weighted mean of
    |mean predicted probability - observed frequency| per bin, weighted by the
    fraction of predictions in each bin. Empty bins contribute nothing.

    Range is [0, 1]. 0 is perfect; 1 is reached only by predicting 0 for every
    event that happens and 1 for every event that does not.
    """
    return sum(
        b["weight"] * b["gap"] for b in reliability_bins(probs, outcomes, n_bins) if b["count"]
    )


def brier(probs: Sequence[float], outcomes: Sequence[Any]) -> float:
    """Mean squared error between predicted probability and outcome.

    Catches sharpness that ECE misses: a model that answers 0.5 to everything on
    a balanced set has ECE 0 and Brier 0.25.
    """
    ps, ys = _pairs(probs, outcomes)
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def is_monotone(bins: Sequence[dict], *, conf: float = 0.95, strict: bool = False) -> bool:
    """Does observed frequency rise with predicted probability across the bins?

    The rubric tiers a non-monotone reliability curve as "Doesn't work"
    regardless of its ECE, so the report carries this as a separate fact.

    Because that consequence is severe, the default is noise-aware: a drop
    counts against monotonicity only when the two bins' Wilson intervals do not
    overlap, i.e. when sampling noise does not plausibly explain it. Ten bins
    over 500 instances produce small reversals by chance often enough that a
    literal test would tier most conditions as "Doesn't work" on noise alone --
    around 70% of perfectly calibrated samples at n=500, which the self-check
    measures and prints. Pass strict=True for the literal reading: any decrease
    at all.

    The noise-aware test compares *every* pair of bins, not only adjacent ones.
    Adjacent-only misses the case the plan singles out by name. A curve whose
    observed frequency slides from 0.90 in the lowest bin to 0.18 in the highest
    is anti-correlated -- AUROC 0.23, the "confidently wrong in a way that
    correlates with nothing" of the "Doesn't work" row -- but at 50 instances
    per bin each individual step of 0.08 sits inside its Wilson interval, so an
    adjacent-only test reports it as monotone. Comparing all pairs catches it on
    the first-to-last comparison. It costs nothing in false positives: on
    perfectly calibrated samples both tests flag 0.0% at n=500 and 0.1% at
    n=200, and the self-check asserts both the catch and the false-positive rate.

    Empty bins are skipped. Fewer than two non-empty bins is vacuously monotone,
    which is itself worth noticing -- it means the predictions were concentrated
    in one bin.
    """
    points = [(b["count"], b["observed"]) for b in bins if b["count"] and b["observed"] is not None]
    if len(points) < 2:
        return True
    if strict:
        return all(b[1] >= a[1] for a, b in zip(points, points[1:]))
    # Wilson bounds are recomputed at this function's `conf` rather than read
    # from the bins' own observed_lo/observed_hi, which were computed at
    # whatever conf reliability_bins was called with.
    #
    # "Some earlier bin's lower bound clears some later bin's upper bound" is
    # the all-pairs condition, and it needs only the running maximum of the
    # lower bounds seen so far, so it is one pass rather than O(bins^2). The
    # number of non-empty bins is bounded by the number of predictions, not by
    # the plan's 10, so a fine-grained diagram over a large sweep would
    # otherwise be quadratic in it.
    best_lo = -1.0
    for c, o in points:
        lo, hi = wilson_interval(int(round(o * c)), c, conf)
        if best_lo > hi:
            return False
        best_lo = max(best_lo, lo)
    return True


def ece_tier(value: float, *, monotone: bool | None = None) -> str:
    """Map an ECE onto the plan's calibration tiers.

    Thresholds come from config.ECE_TIERS, which transcribes "On calibration
    tiers specifically": <=0.02 perfect, <=0.05 superhuman, <=0.15 human,
    <=0.30 bad, above that doesn't work.

    The same paragraph makes a non-monotone curve "doesn't work" whatever the
    ECE, so pass `monotone=is_monotone(bins)` to apply that rule here rather
    than remembering it at each of the nine call sites.

    NaN raises: a condition with no scorable data has no tier, and silently
    tiering it as the worst would report a harness failure as a model result.
    """
    if math.isnan(value):
        raise ValueError("ECE is NaN; a condition with no scorable data has no tier")
    if value < 0.0:
        raise ValueError(f"ECE cannot be negative, got {value!r}")
    if monotone is False:
        return config.ECE_WORST
    # Sorted rather than taken in listed order: a config edit that reordered
    # the tiers would otherwise silently return the wrong one.
    for threshold, name in sorted(config.ECE_TIERS):
        if value <= threshold:
            return name
    return config.ECE_WORST


# --------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------


def auroc(scores: Sequence[float], labels: Sequence[Any]) -> float:
    """Area under the ROC curve, via the Mann-Whitney U identity.

    AUROC equals the probability that a randomly chosen positive outranks a
    randomly chosen negative, counting a tie as half. Computed as

        (sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg)

    over combined ranks with ties assigned their average rank. That is exact and
    equals the trapezoidal ROC area; the naive version that assigns tied scores
    arbitrary distinct ranks silently inflates AUROC toward 1, which matters
    here because the plan expects probabilities to cluster at particular values
    and clustering means ties.

    Scores need not be probabilities -- only their order matters, so a
    confidence or a raw margin works.

    Returns NaN when either class is absent. AUROC is undefined then, and NaN is
    returned rather than raising because a legitimately one-class condition does
    occur in this plan: 3SAT at ratio 8.0 is nearly all UNSAT, and one sweep
    point being undefined should not abort the sweep. Callers must skip NaN
    rather than average it in.
    """
    if len(scores) != len(labels):
        raise ValueError(f"length mismatch: {len(scores)} scores, {len(labels)} labels")
    if len(scores) == 0:
        raise ValueError("no scores to rank")
    s = []
    for x in scores:
        x = float(x)
        if math.isnan(x):
            raise ValueError("scores contain NaN")
        s.append(x)
    y = _as_outcomes(labels, "labels")

    n_pos = int(sum(y))
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = sorted(range(len(s)), key=lambda i: s[i])
    rank_sum_pos = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # ranks are 1-based
        for k in range(i, j + 1):
            if y[order[k]] == 1.0:
                rank_sum_pos += avg_rank
        i = j + 1

    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


# --------------------------------------------------------------------------
# Distribution shape
# --------------------------------------------------------------------------


def kl_divergence(
    p: Sequence[float],
    q: Sequence[float],
    *,
    eps: float = 0.0,
    tol: float = 1e-6,
) -> float:
    """KL(p || q) in nats.

    Used in E5 to ask whether a returned joint is more than the outer product of
    its marginals: KL near 0 means enrollment bought nothing.

    Zeros. The convention 0 * log(0/q) = 0 is applied, so p_i == 0 contributes
    nothing whatever q_i is -- that is the limit, not a fudge. A zero in q under
    non-zero p is genuinely infinite divergence and returns inf by default,
    because that is the true answer and because silently flooring it would turn
    "the model assigned zero mass to something that happened" into a modest
    finite number. When the caller needs a finite figure -- a product of
    marginals can contain an exact zero, and averaging a column of infs is
    useless -- pass eps > 0 to floor q at eps and renormalize, and say so in the
    report.

    Both arguments must already be normalized to within `tol` of 1. They are
    renormalized exactly afterwards, so KL(p || p) is exactly 0.
    """
    if len(p) != len(q):
        raise ValueError(f"length mismatch: {len(p)} vs {len(q)}")
    if eps < 0.0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    pv = _as_distribution(p, tol, "p")
    qv = _as_distribution(q, tol, "q")
    if eps > 0.0:
        qv = [max(v, eps) for v in qv]
        s = sum(qv)
        qv = [v / s for v in qv]
    total = 0.0
    for pi, qi in zip(pv, qv):
        if pi == 0.0:
            continue
        if qi == 0.0:
            return float("inf")
        total += pi * math.log(pi / qi)
    return total


def entropy(p: Sequence[float], *, base: float | None = None, tol: float = 1e-6) -> float:
    """Shannon entropy, in nats by default.

    E7 reports this against cardinality to ask whether the returned distribution
    stays informative at 255 options or flattens into noise. Comparing across
    cardinalities needs the ceiling divided out, since the uniform maximum is
    log(n): divide by `entropy([1/n] * n)`, or pass base=n to read the result
    directly as a fraction of maximum.
    """
    vals = _as_distribution(p, tol, "p")
    h = -sum(v * math.log(v) for v in vals if v > 0.0)
    if base is not None:
        if base <= 1.0:
            raise ValueError(f"base must be > 1, got {base!r}")
        h /= math.log(base)
    return h


# --------------------------------------------------------------------------
# Accuracy and baselines
# --------------------------------------------------------------------------


def accuracy(correct: Sequence[Any]) -> float:
    """Fraction correct, from a sequence of per-instance bools."""
    vals = _as_outcomes(correct, "correct")
    if not vals:
        raise ValueError("no instances to score")
    return sum(vals) / len(vals)


def majority_baseline(labels: Sequence[Any]) -> float:
    """Accuracy of always predicting the most common label.

    The rubric requires this beside every accuracy number, because on a
    class-imbalanced condition -- 3SAT at ratio 8.0, where nearly everything is
    UNSAT -- accuracy alone says nothing.
    """
    if len(labels) == 0:
        raise ValueError("no labels")
    counts = Counter(labels)
    return max(counts.values()) / len(labels)


def random_baseline(n_classes: int) -> float:
    """Accuracy of guessing uniformly among `n_classes` options."""
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    return 1.0 / n_classes


@lru_cache(maxsize=None)
def _z(conf: float) -> float:
    """Two-sided normal quantile for `conf`.

    Cached because `wilson_interval` is called once per reliability bin and
    `reliability_bins` is called once per bootstrap resample, which turns a few
    hundred thousand identical NormalDist constructions into one.
    """
    return NormalDist().inv_cdf(1.0 - (1.0 - conf) / 2.0)


def wilson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    Wilson rather than the normal approximation because several conditions here
    land near 0 or 1 -- forced Sudoku moves, injection success rate -- where the
    normal interval runs outside [0, 1] and has poor coverage. It needs no
    continuity fudge at k=0 or k=n.

    n == 0 returns (0.0, 1.0): nothing is known.
    """
    if n != int(n) or k != int(k):
        # A fractional success count means the caller passed a rate where a
        # count belongs -- wilson_interval(0.25, 10) for "25% of 10" is a real
        # mistake and returns a plausible-looking interval for 2.5 successes.
        raise ValueError(f"k and n must be whole counts, got k={k!r}, n={n!r}")
    k, n = int(k), int(n)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not (0 <= k <= n):
        raise ValueError(f"k={k} out of range for n={n}")
    if not (0.0 < conf < 1.0):
        raise ValueError(f"conf must be in (0, 1), got {conf!r}")
    if n == 0:
        return (0.0, 1.0)
    z = _z(conf)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    # k == 0 and k == n are exact at the closed end; taken from the algebra
    # rather than from the subtraction, which leaves a few 1e-17 of residue and
    # would print a reliability bin's lower bound as 2.8e-17 instead of 0.
    return (lo, hi)


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics, matching numpy's default."""
    if not sorted_vals:
        raise ValueError("no values")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = q * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[int(rank)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (rank - lo)


def percentiles(xs: Sequence[float], ps: Sequence[float] = (50, 95)) -> dict[str, float]:
    """Percentiles of `xs`, keyed "p50", "p95" and so on.

    The plan reports p50 and p95 wall clock per condition. Keys are strings so
    the result drops straight into the JSON summary beside the other numbers.

    An empty `xs` raises rather than returning NaN. Empty means every call in
    that condition failed, and the plan is explicit that a failed call is
    recorded as a failure with a count, never defaulted into a data point. A NaN
    or an infinite value raises for the same reason: a timed-out call recorded
    as `inf` latency would otherwise silently carry all the way to a reported
    p95 of `inf`, and a timeout is a failure with a count, not a slow success.
    """
    vals = sorted(float(x) for x in xs)
    if not vals:
        raise ValueError("no values to take percentiles of")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(
            "values contain NaN or infinity; a failed or timed-out call is "
            "counted as a failure, not entered as a latency"
        )
    out: dict[str, float] = {}
    for p in ps:
        if not (0.0 <= p <= 100.0):
            raise ValueError(f"percentile must be in [0, 100], got {p!r}")
        out[f"p{p:g}"] = _quantile(vals, p / 100.0)
    return out


# --------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------


def mcnemar(a_correct: Sequence[Any], b_correct: Sequence[Any]) -> float:
    """Exact two-sided McNemar p-value for two conditions on the same instances.

    E4 compares three encodings of the same programs and E7 compares two-stage
    against flat on the same items, so the comparison is paired and an unpaired
    test would throw away the pairing and lose power.

    Only the discordant pairs carry information: `only_a` is a right and b
    wrong, `only_b` is a wrong and b right, and under the null those split
    Binomial(only_a + only_b, 0.5). Pairs where both conditions agree, right or
    wrong, are ignored. The exact binomial tail is used rather than the
    chi-square approximation, which is unreliable when there are few discordant
    pairs -- and a condition producing few disagreements is exactly where this
    test gets used.

    With no discordant pairs the two conditions are indistinguishable on this
    data and the p-value is 1.0.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(f"length mismatch: {len(a_correct)} vs {len(b_correct)}")
    if len(a_correct) == 0:
        raise ValueError("no paired observations")
    a = _as_outcomes(a_correct, "a_correct")
    b = _as_outcomes(b_correct, "b_correct")
    only_a = sum(1 for x, y in zip(a, b) if x == 1.0 and y == 0.0)
    only_b = sum(1 for x, y in zip(a, b) if x == 0.0 and y == 1.0)
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(
    a: Sequence[Any],
    b: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    n_boot: int = 10000,
    seed: int = config.MASTER_SEED,
    *,
    conf: float = 0.95,
) -> dict:
    """Bootstrap CI on statistic(a) - statistic(b), resampling instances in pairs.

    `a[i]` and `b[i]` are the same instance under two conditions, and each
    resample draws one index vector applied to both. That preserves the pairing,
    so instance difficulty cancels out and the interval reflects only the
    difference between conditions -- which is the whole point of measuring both
    conditions on the same instances.

    `statistic` is any function from a sample to a number: `accuracy` for a
    difference in accuracy, a lambda over (prob, outcome) tuples for a
    difference in ECE.

    Seeded and reproducible: the same (a, b, statistic, n_boot, seed) gives a
    byte-identical result. Resampling uses stdlib `random.Random` rather than
    numpy so reproduction does not depend on a numpy version.

    The p-value is the bootstrap proportion of resamples falling on the null
    side of zero, doubled. It is a rough two-sided figure, not an exact test;
    use `mcnemar` when the statistic is accuracy and an exact p is wanted. It
    cannot resolve below 2 / n_boot, so a reported 0.0 means "smaller than the
    bootstrap can see" and should be written up that way, not as zero.

    Cost is n_boot evaluations of `statistic` over samples of len(a). At the
    default 10000 and 500 instances that is a few seconds.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("no paired observations")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not (0.0 < conf < 1.0):
        raise ValueError(f"conf must be in (0, 1), got {conf!r}")

    stat_a = float(statistic(a))
    stat_b = float(statistic(b))
    rng = random.Random(seed)
    idx_pool = range(n)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choices(idx_pool, k=n)
        diffs.append(float(statistic([a[i] for i in idx])) - float(statistic([b[i] for i in idx])))
    diffs.sort()

    alpha = (1.0 - conf) / 2.0
    n_le = sum(1 for d in diffs if d <= 0.0)
    n_ge = sum(1 for d in diffs if d >= 0.0)
    return {
        "statistic_a": stat_a,
        "statistic_b": stat_b,
        "difference": stat_a - stat_b,
        "ci_lo": _quantile(diffs, alpha),
        "ci_hi": _quantile(diffs, 1.0 - alpha),
        "conf": conf,
        "p_value": min(1.0, 2.0 * min(n_le, n_ge) / n_boot),
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
    }


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def cost_per_decision(input_tokens: float, n_questions: int) -> float:
    """USD per question for one call: input tokens x price / questions asked.

    Output tokens are free under this pricing, so they do not appear. Dividing
    by the question count is what makes the number comparable across batch
    sizes, and E3 uses the resulting curve to test the claim that the marginal
    cost of question 200 is zero.
    """
    if input_tokens < 0:
        raise ValueError(f"input_tokens must be >= 0, got {input_tokens!r}")
    if n_questions < 1:
        raise ValueError(f"n_questions must be >= 1, got {n_questions}")
    return input_tokens * config.USD_PER_INPUT_TOKEN / n_questions


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    summary: list[tuple[str, str]] = []
    checks = 0

    def record(label: str, value: Any) -> None:
        summary.append((label, f"{value:.6g}" if isinstance(value, float) else str(value)))

    def close(x: float, y: float, tol: float = 1e-9) -> bool:
        global checks
        checks += 1
        assert abs(x - y) <= tol, f"{x!r} != {y!r} within {tol!r}"
        return True

    def check(cond: bool, msg: str) -> None:
        global checks
        checks += 1
        assert cond, msg

    # -- binning ------------------------------------------------------------
    # A probability sitting exactly on an edge belongs to the bin above it.
    edge_bins = reliability_bins([i / 10 for i in range(11)], [1] * 11, 10)
    check([b["count"] for b in edge_bins] == [1] * 9 + [2], "edge probabilities mis-binned")
    check(len(reliability_bins([0.5], [1], 10)) == 10, "empty bins must still be returned")
    check(
        sum(b["weight"] for b in reliability_bins([0.1, 0.4, 0.9], [0, 1, 1], 10)) == 1.0,
        "bin weights must sum to 1",
    )
    three = reliability_bins([1 / 3, 2 / 3], [0, 1], 3)
    check([b["count"] for b in three] == [0, 1, 1], "edge mis-binned at n_bins=3")
    # n_bins 10 and 3 above cannot discriminate the bisect from `int(p *
    # n_bins)` -- their products round back exactly. 22 is the smallest n_bins
    # that can, and 100 fails on 3 of its edges. Without these two, the whole
    # justification in _bin_index is untested and reads as superstition.
    for nb in (22, 100):
        edge_probs = [i / nb for i in range(nb)]
        got = [_bin_index(p, [i / nb for i in range(nb + 1)]) for p in edge_probs]
        check(got == list(range(nb)), f"edge mis-binned at n_bins={nb}")
        naive = [int(p * nb) for p in edge_probs]
        check(naive != got, f"n_bins={nb} was chosen because int(p*n_bins) differs there")
    check(
        _bin_index(1.0, [i / 10 for i in range(11)]) == 9,
        "a prediction of exactly 1.0 belongs in the closed top bin",
    )

    # -- ECE ----------------------------------------------------------------
    # Hand-worked: gaps 0.05, 0.85, 0.05 in three singleton bins.
    close(ece([0.05, 0.15, 0.95], [0, 1, 1]), 0.95 / 3, 1e-12)
    # Confidently wrong in both directions: the maximum ECE is exactly 1.
    close(ece([1.0, 1.0, 0.0, 0.0], [0, 0, 1, 1]), 1.0, 0.0)
    record("ece(confidently wrong)", ece([1.0, 1.0, 0.0, 0.0], [0, 0, 1, 1]))
    # A large perfectly calibrated sample: outcome ~ Bernoulli(p), p ~ U(0, 1).
    rng = random.Random(20260919)
    cal_p = [rng.random() for _ in range(200_000)]
    cal_y = [rng.random() < p for p in cal_p]
    ece_cal = ece(cal_p, cal_y)
    check(ece_cal < 0.005, f"calibrated sample should have ECE near 0, got {ece_cal}")
    record("ece(calibrated, n=200000)", ece_cal)
    # Always answering 0.5 on a balanced set: ECE 0, Brier 0.25. ECE alone
    # cannot see that the predictions are useless; this is why Brier is here.
    close(ece([0.5] * 200, [i % 2 for i in range(200)]), 0.0, 1e-12)
    close(brier([0.5] * 200, [i % 2 for i in range(200)]), 0.25, 1e-12)
    # ECE must equal the weighted gap sum of the bins the diagram plots.
    bins_cal = reliability_bins(cal_p, cal_y)
    close(ece_cal, sum(b["weight"] * b["gap"] for b in bins_cal if b["count"]), 1e-12)

    # -- Brier --------------------------------------------------------------
    # (0.01 + 0.01 + 0.64 + 0.09) / 4
    close(brier([0.1, 0.9, 0.8, 0.3], [0, 1, 0, 0]), 0.1875, 1e-12)
    close(brier([1.0, 0.0], [True, False]), 0.0, 0.0)
    close(brier([0.0, 1.0], [True, False]), 1.0, 0.0)
    record("brier(hand-worked)", brier([0.1, 0.9, 0.8, 0.3], [0, 1, 0, 0]))

    # -- AUROC --------------------------------------------------------------
    close(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0, 0.0)
    close(auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0, 0.0)
    close(auroc([0.5] * 4, [0, 0, 1, 1]), 0.5, 0.0)
    close(auroc([0.5] * 100, [i % 2 for i in range(100)]), 0.5, 0.0)
    # Hand-worked tie: ranks 1, 2.5, 2.5, 4; positives hold 2.5 and 4.
    # (6.5 - 3) / 4 = 0.875. Pairwise: 1 + 0.5 + 1 + 1 over 4 pairs.
    close(auroc([1, 2, 2, 3], [0, 0, 1, 1]), 0.875, 1e-12)
    record("auroc(hand-worked tie)", auroc([1, 2, 2, 3], [0, 0, 1, 1]))
    check(math.isnan(auroc([0.1, 0.2, 0.3], [0, 0, 0])), "one class absent must give NaN")
    check(math.isnan(auroc([0.1, 0.2, 0.3], [1, 1, 1])), "one class absent must give NaN")
    # Invariant under a monotone transform of the scores; complementary under
    # flipped labels.
    rng = random.Random(7)
    s_rand = [rng.random() for _ in range(200)]
    y_rand = [rng.random() < 0.3 + 0.4 * s for s in s_rand]
    close(auroc(s_rand, y_rand), auroc([math.log(s + 1) for s in s_rand], y_rand), 1e-12)
    close(auroc(s_rand, y_rand) + auroc(s_rand, [not y for y in y_rand]), 1.0, 1e-12)
    # Against brute-force pair counting on a tie-heavy sample. Scores rounded to
    # one decimal so that ties dominate, which is where the rank handling has to
    # be right -- and the plan expects returned probabilities to cluster.
    rng = random.Random(11)
    s_tied = [round(rng.random(), 1) for _ in range(300)]
    y_tied = [rng.random() < 0.2 + 0.6 * s for s in s_tied]
    pos = [s for s, y in zip(s_tied, y_tied) if y]
    neg = [s for s, y in zip(s_tied, y_tied) if not y]
    brute = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg) / (
        len(pos) * len(neg)
    )
    close(auroc(s_tied, y_tied), brute, 1e-12)
    record("auroc(tie-heavy) vs brute force", auroc(s_tied, y_tied))

    # -- KL and entropy -----------------------------------------------------
    uni4 = [0.25] * 4
    close(kl_divergence(uni4, uni4), 0.0, 0.0)
    close(kl_divergence([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]), 0.0, 0.0)
    close(kl_divergence([1.0, 0.0], [0.5, 0.5]), math.log(2), 1e-12)
    # p_i == 0 contributes nothing even though q_i > 0.
    close(kl_divergence([0.0, 1.0], [0.5, 0.5]), math.log(2), 1e-12)
    # A zero in q under non-zero p is infinite, and eps makes it finite.
    check(math.isinf(kl_divergence([0.5, 0.5], [1.0, 0.0])), "zero in q must give inf")
    finite = kl_divergence([0.5, 0.5], [1.0, 0.0], eps=1e-9)
    check(math.isfinite(finite) and finite > 5.0, f"eps-floored KL looks wrong: {finite}")
    check(kl_divergence(uni4, [0.7, 0.1, 0.1, 0.1]) > 0.0, "KL to a different q must be positive")
    record("kl(uniform4 || skewed)", kl_divergence(uni4, [0.7, 0.1, 0.1, 0.1]))
    close(entropy(uni4), math.log(4), 1e-12)
    close(entropy(uni4, base=4), 1.0, 1e-12)
    close(entropy([1.0, 0.0, 0.0]), 0.0, 0.0)
    close(entropy([0.5, 0.5], base=2), 1.0, 1e-12)
    # Entropy falls as the distribution sharpens -- the E7 flattening check.
    check(entropy([0.97, 0.01, 0.01, 0.01]) < entropy([0.4, 0.3, 0.2, 0.1]) < entropy(uni4),
          "entropy must decrease as the distribution sharpens")
    record("entropy(uniform255, base=255)", entropy([1 / 255] * 255, base=255))

    # -- baselines and intervals -------------------------------------------
    close(majority_baseline([1, 1, 1, 0]), 0.75, 1e-12)
    close(majority_baseline([True] * 90 + [False] * 10), 0.90, 1e-12)
    close(random_baseline(4), 0.25, 1e-12)
    close(random_baseline(255), 1 / 255, 1e-15)
    close(accuracy([True, True, False, True]), 0.75, 1e-12)
    # Published Wilson 95% interval for 0/10 is (0, 0.2775).
    lo, hi = wilson_interval(0, 10)
    close(lo, 0.0, 0.0)
    close(hi, 0.2775, 1e-4)
    lo, hi = wilson_interval(5, 10)
    close(lo + hi, 1.0, 1e-12)
    close(lo, 0.2366, 1e-4)
    check(wilson_interval(0, 0) == (0.0, 1.0), "n=0 must give the whole interval")
    # A wider interval for less data, and the plan's point: 0.50 vs 0.53 on 200
    # instances is not a finding.
    lo_small, hi_small = wilson_interval(50, 100)
    lo_big, hi_big = wilson_interval(500, 1000)
    check((hi_small - lo_small) > 3 * (hi_big - lo_big), "interval must narrow with n")
    check(wilson_interval(100, 200)[1] > 0.53, "0.50 vs 0.53 at n=200 is inside the interval")
    w_lo, w_hi = wilson_interval(100, 200)
    record("wilson(100/200)", f"({w_lo:.4f}, {w_hi:.4f})")

    # -- monotonicity -------------------------------------------------------
    rising = reliability_bins([0.05] * 100 + [0.95] * 100, [0] * 100 + [1] * 100)
    falling = reliability_bins([0.05] * 100 + [0.95] * 100, [1] * 100 + [0] * 100)
    check(is_monotone(rising), "a clean rising curve is monotone")
    check(not is_monotone(falling), "an inverted curve is not monotone")
    check(is_monotone(bins_cal), "the calibrated sample's curve is monotone")
    # A small dip on thin bins is noise under the default and a violation under
    # strict -- this is the difference the default exists to make.
    noisy = reliability_bins([0.05] * 10 + [0.15] * 10, [1] * 5 + [0] * 5 + [1] * 4 + [0] * 6)
    check(is_monotone(noisy), "a dip inside sampling noise must not flag")
    check(not is_monotone(noisy, strict=True), "strict must flag any decrease")
    one_bin = reliability_bins([0.5] * 50, [1] * 25 + [0] * 25)
    check(is_monotone(one_bin), "a single occupied bin is vacuously monotone")

    # The case the plan names: anti-correlated calibration, which belongs in
    # "Doesn't work" regardless of accuracy. Observed frequency slides from 0.90
    # to 0.18 across ten bins of 50 -- but each individual step of 0.08 is
    # inside its Wilson interval, so comparing only adjacent bins reports this
    # curve as monotone. This is the assertion that requires the all-pairs test.
    anti_p: list[float] = []
    anti_y: list[int] = []
    for i in range(10):
        hits = round((0.90 - 0.08 * i) * 50)
        anti_p += [0.05 + 0.1 * i] * 50
        anti_y += [1] * hits + [0] * (50 - hits)
    anti_bins = reliability_bins(anti_p, anti_y)
    check(not is_monotone(anti_bins), "an anti-correlated curve must not pass as monotone")
    check(auroc(anti_p, anti_y) < 0.3, "the anti-correlated curve should also show in AUROC")
    check(
        all(
            anti_bins[i]["observed"] > anti_bins[i + 1]["observed"] for i in range(9)
        ),
        "the anti-correlated fixture must actually decline at every step",
    )
    # ...and it is missed by the adjacent-only rule, which is why that rule was
    # replaced. Asserted so the replacement is not quietly reverted.
    adjacent_only = True
    pts = [(b["count"], b["observed"]) for b in anti_bins if b["count"]]
    for (c0, o0), (c1, o1) in zip(pts, pts[1:]):
        if wilson_interval(round(o0 * c0), c0)[0] > wilson_interval(round(o1 * c1), c1)[1]:
            adjacent_only = False
    check(adjacent_only, "fixture must be one that adjacent-only comparison misses")
    record("anti-correlated curve: adjacent-only says monotone", adjacent_only)

    # The false-positive rates both branches trade off, measured rather than
    # asserted from the armchair: on perfectly calibrated data the default must
    # stay quiet, and strict must be shown to be unusable as a default.
    rng = random.Random(505)
    fp_default = fp_strict = 0
    trials = 200
    for _ in range(trials):
        fp_p = [rng.random() for _ in range(500)]
        fp_y = [rng.random() < q for q in fp_p]
        fp_b = reliability_bins(fp_p, fp_y)
        fp_default += not is_monotone(fp_b)
        fp_strict += not is_monotone(fp_b, strict=True)
    check(fp_default <= trials * 0.02, f"default flagged {fp_default}/{trials} calibrated samples")
    check(fp_strict > trials * 0.4, f"strict flagged only {fp_strict}/{trials}; check the fixture")
    record(
        "false non-monotone on calibrated n=500",
        f"default {fp_default}/{trials}, strict {fp_strict}/{trials}",
    )

    # -- tiers --------------------------------------------------------------
    check(ece_tier(0.00) == "Perfect", "0.00 is Perfect")
    check(ece_tier(0.02) == "Perfect", "the 0.02 boundary is inclusive")
    check(ece_tier(0.021) == "Superhuman", "just past 0.02 is Superhuman")
    check(ece_tier(0.05) == "Superhuman", "the 0.05 boundary is inclusive")
    check(ece_tier(0.15) == "Human", "the 0.15 boundary is inclusive")
    check(ece_tier(0.30) == "Bad", "the 0.30 boundary is inclusive")
    check(ece_tier(0.31) == config.ECE_WORST, "past 0.30 does not work")
    check(ece_tier(0.01, monotone=False) == config.ECE_WORST, "non-monotone overrides the ECE")
    check(ece_tier(0.01, monotone=True) == "Perfect", "monotone=True does not change the tier")
    record("ece_tier(0.021)", ece_tier(0.021))

    # -- latency ------------------------------------------------------------
    pc = percentiles([1, 2, 3, 4])
    close(pc["p50"], 2.5, 1e-12)
    close(pc["p95"], 3.85, 1e-12)
    close(percentiles([5], (50, 95))["p95"], 5.0, 0.0)
    edge = percentiles(list(range(101)), (0, 50, 100))
    close(edge["p0"], 0.0, 0.0)
    close(edge["p50"], 50.0, 1e-12)
    close(edge["p100"], 100.0, 0.0)
    # Unsorted input must give the same answer as sorted.
    rng = random.Random(3)
    lat = [rng.expovariate(1 / 0.2) for _ in range(1000)]
    check(percentiles(lat) == percentiles(sorted(lat, reverse=True)), "percentiles must sort")
    lat_pc = percentiles(lat)
    record("percentiles(exp latency)", f"p50={lat_pc['p50']:.4f} p95={lat_pc['p95']:.4f}")

    # -- McNemar ------------------------------------------------------------
    # 10 discordant pairs all one way: 2 * (1/1024).
    close(mcnemar([1] * 10 + [1] * 5, [0] * 10 + [1] * 5), 2 / 1024, 1e-15)
    # 9 discordant, 8 one way: 2 * (1 + 9) / 512.
    close(mcnemar([1] * 8 + [0] + [1] * 5, [0] * 8 + [1] + [1] * 5), 20 / 512, 1e-15)
    # Evenly split discordants: the doubled tail exceeds 1 and clamps.
    close(mcnemar([1] * 5 + [0] * 5, [0] * 5 + [1] * 5), 1.0, 0.0)
    # Identical conditions have no discordant pairs and no evidence.
    close(mcnemar([1, 0, 1, 1], [1, 0, 1, 1]), 1.0, 0.0)
    # Concordant errors do not count: only the disagreements carry information.
    close(
        mcnemar([1] * 10 + [0] * 90, [0] * 10 + [0] * 90),
        mcnemar([1] * 10, [0] * 10),
        1e-15,
    )
    record("mcnemar(10 vs 0 discordant)", mcnemar([1] * 10 + [1] * 5, [0] * 10 + [1] * 5))

    # -- paired bootstrap ---------------------------------------------------
    a_acc = [True] * 60 + [False] * 40
    b_acc = [True] * 50 + [False] * 50
    boot1 = paired_bootstrap(a_acc, b_acc, accuracy, n_boot=2000, seed=1)
    boot2 = paired_bootstrap(a_acc, b_acc, accuracy, n_boot=2000, seed=1)
    check(boot1 == boot2, "same seed must reproduce the bootstrap exactly")
    # A different seed must resample differently, but must not move the
    # observed difference, which is computed on the real sample. Checked with a
    # continuous statistic: accuracy on 100 items takes values on a 0.01 grid,
    # so two seeds routinely land on the same CI endpoint by coincidence.
    rng = random.Random(42)
    cont_a = [rng.gauss(0.6, 0.2) for _ in range(100)]
    cont_b = [rng.gauss(0.5, 0.2) for _ in range(100)]

    def mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs)

    cont1 = paired_bootstrap(cont_a, cont_b, mean, n_boot=500, seed=1)
    cont2 = paired_bootstrap(cont_a, cont_b, mean, n_boot=500, seed=2)
    check(cont1["ci_lo"] != cont2["ci_lo"], "a different seed must resample differently")
    close(cont1["difference"], cont2["difference"], 0.0)
    close(boot1["difference"], 0.1, 1e-12)
    check(boot1["ci_lo"] < 0.1 < boot1["ci_hi"], "the CI must contain the observed difference")
    # No overlap between conditions: the difference is 1.0 with no uncertainty.
    ident = paired_bootstrap([1.0] * 50, [0.0] * 50, mean, n_boot=200, seed=5)
    close(ident["difference"], 1.0, 1e-12)
    close(ident["ci_lo"], 1.0, 1e-12)
    close(ident["ci_hi"], 1.0, 1e-12)
    close(ident["p_value"], 0.0, 0.0)
    # Identical conditions: difference 0, and the p-value cannot reject.
    same = paired_bootstrap(a_acc, list(a_acc), accuracy, n_boot=200, seed=5)
    close(same["difference"], 0.0, 0.0)
    close(same["p_value"], 1.0, 0.0)
    # The pairing is the whole point of this function, and every check above
    # still passes if the two conditions are resampled with independent index
    # vectors. This one does not. Two conditions that differ by a constant on
    # every instance, over instances that differ wildly from each other: paired
    # resampling cancels the instance effect and gives a zero-width interval,
    # while unpaired resampling leaves an interval as wide as the instance
    # spread (about +/-0.55 here).
    rng = random.Random(808)
    base = [rng.gauss(0.0, 2.0) for _ in range(200)]
    pair_a = [x + 0.25 for x in base]
    paired = paired_bootstrap(pair_a, base, mean, n_boot=300, seed=1)
    close(paired["difference"], 0.25, 1e-12)
    check(
        paired["ci_hi"] - paired["ci_lo"] < 1e-12,
        f"paired resampling must cancel the instance effect, got width "
        f"{paired['ci_hi'] - paired['ci_lo']}",
    )
    record("paired bootstrap CI width on a constant shift", paired["ci_hi"] - paired["ci_lo"])
    # The documented ECE use case: a statistic over (prob, outcome) pairs.
    ece_pairs_a = [(rng.random(), rng.random() < 0.5) for _ in range(120)]
    ece_pairs_b = [(rng.random(), rng.random() < 0.5) for _ in range(120)]
    ece_boot = paired_bootstrap(
        ece_pairs_a,
        ece_pairs_b,
        lambda s: ece([q for q, _ in s], [o for _, o in s]),
        n_boot=200,
        seed=2,
    )
    check(
        ece_boot["ci_lo"] <= ece_boot["difference"] <= ece_boot["ci_hi"],
        "the ECE-difference bootstrap must bracket its own point estimate",
    )
    record(
        "bootstrap(0.60 vs 0.50)",
        f"{boot1['difference']:.3f} "
        f"[{boot1['ci_lo']:.3f}, {boot1['ci_hi']:.3f}] p={boot1['p_value']:.3f}",
    )

    # -- cost ---------------------------------------------------------------
    close(cost_per_decision(800, 1), 800 * 42 / 1e9, 1e-18)
    close(cost_per_decision(800, 200), 800 * 42 / 1e9 / 200, 1e-20)
    # The plan's own budget line: ~92M input tokens is about $4.
    close(cost_per_decision(92_000_000, 1), 3.864, 1e-9)
    record("cost_per_decision(800 tok, 1 q)", cost_per_decision(800, 1))
    record("cost_per_decision(800 tok, 200 q)", cost_per_decision(800, 200))

    # -- bad input is refused, not absorbed ---------------------------------
    def raises(fn: Callable[[], Any], msg: str) -> None:
        global checks
        checks += 1
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(msg)

    raises(lambda: ece([0.5, 1.5], [0, 1]), "a probability above 1 must raise")
    raises(lambda: ece([0.5], [0, 1]), "a length mismatch must raise")
    raises(lambda: ece([], []), "an empty sample must raise")
    raises(lambda: ece([0.5, 0.5], [0, 2]), "a non-binary outcome must raise")
    raises(lambda: ece([0.5, 0.5], [0, 0.5]), "a fractional outcome must raise")
    raises(lambda: ece_tier(float("nan")), "a NaN ECE must raise")
    raises(lambda: entropy([0.5, 0.4]), "an unnormalized distribution must raise")
    raises(lambda: kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4]), "mismatched supports must raise")
    raises(lambda: kl_divergence([0.5, -0.5, 1.0], [0.3, 0.3, 0.4]), "negative mass must raise")
    raises(lambda: percentiles([]), "empty latency must raise, not default to NaN")
    raises(lambda: percentiles([1.0, float("inf")]), "an infinite latency must raise")
    raises(lambda: percentiles([1.0, float("nan")]), "a NaN latency must raise")
    raises(lambda: percentiles([1.0], (101,)), "a percentile above 100 must raise")
    raises(lambda: wilson_interval(11, 10), "k > n must raise")
    raises(lambda: wilson_interval(0.25, 10), "a rate passed where a count belongs must raise")
    raises(lambda: reliability_bins([0.5], [1], 0), "n_bins of 0 must raise")
    raises(lambda: auroc([0.1, 0.2], [0, 1, 1]), "an AUROC length mismatch must raise")
    raises(lambda: auroc([float("nan"), 0.2], [0, 1]), "a NaN score must raise")
    raises(lambda: entropy([0.5, 0.5], base=1.0), "an entropy base of 1 must raise")
    raises(lambda: kl_divergence([0.5, 0.5], [0.5, 0.5], eps=-1e-9), "a negative eps must raise")
    raises(lambda: mcnemar([1, 0], [1, 0, 1]), "a McNemar length mismatch must raise")
    raises(lambda: paired_bootstrap([1], [1], accuracy, n_boot=0), "n_boot of 0 must raise")
    raises(lambda: accuracy([]), "an empty accuracy sample must raise")
    raises(lambda: cost_per_decision(800, 0), "zero questions must raise")
    raises(lambda: random_baseline(0), "zero classes must raise")

    # -- cross-checks against scipy and numpy, when available ---------------
    cross = "skipped (scipy/numpy absent)"
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        pass
    else:
        u = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
        close(auroc(s_tied, y_tied), float(u) / (len(pos) * len(neg)), 1e-12)
        close(auroc([1, 2, 2, 3], [0, 0, 1, 1]),
              float(stats.mannwhitneyu([2, 3], [1, 2], alternative="greater").statistic) / 4, 1e-12)
        close(mcnemar([1] * 10 + [1] * 5, [0] * 10 + [1] * 5),
              float(stats.binomtest(0, 10, 0.5).pvalue), 1e-15)
        close(mcnemar([1] * 8 + [0] + [1] * 5, [0] * 8 + [1] + [1] * 5),
              float(stats.binomtest(1, 9, 0.5).pvalue), 1e-15)
        p_np = percentiles(lat, (50, 95, 99))
        for q in (50, 95, 99):
            close(p_np[f"p{q}"], float(np.percentile(lat, q)), 1e-12)
        close(entropy([0.1, 0.2, 0.3, 0.4]), float(stats.entropy([0.1, 0.2, 0.3, 0.4])), 1e-12)
        close(kl_divergence(uni4, [0.7, 0.1, 0.1, 0.1]),
              float(stats.entropy(uni4, [0.7, 0.1, 0.1, 0.1])), 1e-12)
        # A numpy array must score, not raise. Every emptiness guard here tests
        # `len(x) == 0` rather than `not x` for this reason: an experiment that
        # keeps its predictions in an array would otherwise get numpy's "truth
        # value of an array is ambiguous" out of a metrics function and have to
        # go looking for a bug that is not there.
        np_p = np.array([0.1, 0.4, 0.9])
        np_y = np.array([0, 1, 1])
        close(ece(np_p, np_y), ece(np_p.tolist(), np_y.tolist()), 0.0)
        close(brier(np_p, np_y), brier(np_p.tolist(), np_y.tolist()), 0.0)
        close(auroc(np_p, np_y), auroc(np_p.tolist(), np_y.tolist()), 0.0)
        close(accuracy(np.array([True, False, True])), 2 / 3, 1e-12)
        close(mcnemar(np.array([1, 1, 0]), np.array([0, 1, 0])), mcnemar([1, 1, 0], [0, 1, 0]), 0.0)
        cross = "scipy mannwhitneyu / binomtest / entropy, numpy percentile, numpy arrays"

    # -- summary ------------------------------------------------------------
    record(
        "reliability bins (calibrated sample)",
        f"{sum(1 for b in bins_cal if b['count'])}/10 occupied",
    )
    record("monotone (calibrated sample)", is_monotone(bins_cal))
    record("tier (calibrated sample)", ece_tier(ece_cal, monotone=is_monotone(bins_cal)))
    record("cross-checks", cross)

    width = max(len(label) for label, _ in summary)
    print("jeveval.metrics self-check")
    print("-" * (width + 26))
    for label, value in summary:
        print(f"{label:<{width}}  {value}")
    print("-" * (width + 26))
    print(f"{checks} assertions passed")
    sys.exit(0)
