"""The plan's five-tier scoring rubric, as data rather than as judgement.

The plan states tier thresholds for every experiment, in prose, inside each
experiment's "Tiers" paragraph. This module transcribes those paragraphs into
criteria a program can evaluate, so the report writer assigns tiers by calling a
function instead of by reading the plan and deciding. Two things follow from
that which are worth stating up front, because they shape the whole API:

1. `tier_by_difficulty` is the primary entry point, not `assign`. The plan:
   "The tier that matters most is per-condition, not overall... Report the tier
   as a function of difficulty, and state the difficulty at which each tier
   boundary is crossed. A single overall grade throws away the finding."
   `assign` is the degenerate case of one condition.

2. A tier travels with its baseline. The plan: "A tier assignment without the
   baseline number next to it is not a result." So every assignment returns a
   `TierResult` carrying the metric value, the baseline value, where the
   baseline came from, and n -- rather than a bare string that a caller has to
   remember to annotate.

Where the plan gives a number ("ECE <=0.05", "injection <5%") the number is
transcribed literally. Where it gives a word ("solid", "decent", "clean
separation") the word has been turned into a number, the criterion carries a
`note` saying so, and `TierResult.reasons` reports it, so the report can
disclose which thresholds were the plan's and which were the harness's.

Nothing here computes a metric. Metrics come from `jeveval.metrics` and the
experiment modules; this module only compares them to thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import config

# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------

TIER_ORDER: tuple[str, ...] = ("Perfect", "Superhuman", "Human", "Bad", "Doesn't work")
WORST_TIER = "Doesn't work"

# Anything that clears no rung of the ladder but is not disqualified lands here.
# This mirrors the plan's definition of Bad -- "above chance, below a cheap
# deterministic baseline" -- which is the residual case between "meets the Human
# bar" and "is disqualified", rather than a bar of its own.
DEFAULT_TIER = "Bad"

# The plan's Perfect row: "Accuracy >=0.98, ECE <=0.02, zero coherence violations."
PERFECT_ACCURACY = 0.98

# Several experiments say "flat" or "zero" where a measurement never lands
# exactly on zero. With 200-500 instances per condition the standard error on an
# accuracy difference is around 0.03, so "flat" has to mean "smaller than we can
# resolve"; 0.005 is well inside that and keeps Perfect genuinely hard to reach.
EPS_FLAT = 0.005


def tier_index(tier: str) -> int:
    """Position in TIER_ORDER, 0 = Perfect. Raises on an unknown tier."""
    return TIER_ORDER.index(tier)


def worse_of(*tiers: str) -> str:
    """The worst of several tiers. Used to combine sub-domains of one experiment."""
    return max(tiers, key=tier_index)


# --------------------------------------------------------------------------
# Criteria
# --------------------------------------------------------------------------

_OPS: dict[str, Callable[[float, float], bool]] = {
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}

# Ops that do not compare a value against a threshold.
_PREDICATE_OPS: tuple[str, ...] = ("is_true", "is_false", "present", "near")


def _is_nan(value: Any) -> bool:
    """NaN means the quantity is undefined here, not that it failed the bar.

    `metrics.auroc` returns NaN whenever one class is absent, which on E2's
    sweep happens at every ratio far enough from 4.26 that every instance came
    out the same way -- 7 of 25 ratios on a routine sweep. Comparing NaN with
    `>=` yields False, so the Human rung would fail on a metric that was never
    measurable and the condition would be reported as Bad. Every comparison
    routes through here so that undefined reads as unmeasured.

    Infinities are left alone: `metrics.kl_divergence` returns inf for a genuine
    zero under non-zero p, and `kl_joint_product > 0.3` is true of it.
    """
    return isinstance(value, float) and math.isnan(value)


@dataclass(frozen=True)
class Criterion:
    """One comparison against one metric.

    `vs` makes the threshold relative to another metric: `Criterion("accuracy",
    "<=", 0.02, vs="chance_baseline")` reads "accuracy is at or below chance,
    within 0.02", which is how the plan phrases most of its Doesn't-work rows.

    A criterion whose metric was not measured evaluates to None, not to False.
    An experiment that did not run its injection tests must not be graded as
    though injection was zero.
    """

    metric: str
    op: str
    value: Any = None
    vs: str | None = None
    note: str = ""

    def threshold(self, metrics: Mapping[str, Any]) -> float | None:
        if self.vs is None:
            return self.value
        base = metrics.get(self.vs)
        if base is None or _is_nan(base):
            return None
        return float(base) + float(self.value)

    def check(self, metrics: Mapping[str, Any]) -> bool | None:
        """True, False, or None when the metric (or its `vs` reference) is absent
        or undefined."""
        got = metrics.get(self.metric)
        if got is None or _is_nan(got):
            return None
        if self.op == "is_true":
            return bool(got) is True
        if self.op == "is_false":
            return bool(got) is False
        if self.op == "present":
            return True
        if self.op == "near":
            target, tol = self.value
            return abs(float(got) - float(target)) <= float(tol)
        thr = self.threshold(metrics)
        if thr is None:
            return None
        return _OPS[self.op](float(got), float(thr))

    def describe(self) -> str:
        if self.op in ("is_true", "is_false"):
            return f"{self.metric} {self.op.replace('_', ' ')}"
        if self.op == "present":
            return f"{self.metric} was recorded at all"
        if self.op == "near":
            target, tol = self.value
            return f"{self.metric} within {tol} of {target}"
        rhs = f"{self.value}" if self.vs is None else f"{self.vs} + {self.value}"
        return f"{self.metric} {self.op} {rhs}"


@dataclass(frozen=True)
class Rule:
    """A named conjunction of criteria: a tier rung, a disqualifier, or a signature.

    `source` quotes the plan's own wording so the report can print the sentence a
    tier came from next to the tier.
    """

    name: str
    criteria: tuple[Criterion, ...]
    source: str = ""

    def check(self, metrics: Mapping[str, Any]) -> tuple[bool, list[str], list[str], int]:
        """(holds, missing metric names, notes, how many criteria were evaluated).

        A rule holds when every criterion that could be evaluated passed and at
        least one could be evaluated. A rule none of whose metrics were measured
        never holds; it would otherwise promote an unrun experiment to Perfect.
        The count comes back so the caller can tell "measured and missed the bar"
        from "this condition supplied none of these metrics".

        Note that a rule can therefore hold on *part* of its evidence. That is
        deliberate -- several criteria are sweep-level (E2's `crossover_ratio`
        has no value at a single ratio), so requiring all of them would make
        whole rungs unreachable per condition. It also means a rung can be
        awarded without the criterion that matters most having been measured, so
        `assign` records which criteria went unevaluated on the rung it matched
        and reports them; see `TierResult.unevaluated`.
        """
        missing: list[str] = []
        notes: list[str] = []
        known = 0
        ok = True
        for c in self.criteria:
            res = c.check(metrics)
            if res is None:
                # Name whichever of the two metrics was the one not measured.
                missing.append(c.vs if (c.vs and c.metric in metrics) else c.metric)
                continue
            known += 1
            if c.note:
                notes.append(f"{c.describe()} ({c.note})")
            if not res:
                ok = False
        return (ok and known > 0), missing, notes, known


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TierResult:
    """A tier together with everything needed to report it honestly.

    `is_reportable` is false when the baseline or n is missing, because the plan
    treats a tier without its baseline as not a result. Callers that want that
    enforced at the call site pass `strict=True` to `assign`.
    """

    experiment: str
    tier: str
    metric: str
    value: float | None
    baseline: float | None
    baseline_label: str
    baseline_source: str  # "measured" | "assumed" | "missing"
    n: int | None
    difficulty: Any = None
    matched_rule: str = ""
    reasons: tuple[str, ...] = ()
    signatures: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    # Criteria of the rung that was actually matched which could not be
    # evaluated. Non-empty means the tier rests on part of its rule.
    unevaluated: tuple[str, ...] = ()
    # Metrics that arrived as NaN and were treated as unmeasured.
    undefined: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        """One row of the report's tier-by-difficulty table.

        `jeveval.report` reads difficulty, n, metrics, baseline and tier from a
        by_difficulty row, and metric, value, baseline, baseline_name and n from
        a headline; this dict carries both, so the experiment module writes the
        same object either way.
        """
        return {
            "difficulty": self.difficulty,
            "tier": self.tier,
            "n": self.n,
            "metrics": dict(self.metrics),
            "metric": self.metric,
            "value": self.value,
            "baseline": self.baseline,
            "baseline_name": self.baseline_label,
            "baseline_source": self.baseline_source,
            "reportable": self.is_reportable,
            "fully_evaluated": self.is_fully_evaluated,
            "matched_rule": self.matched_rule,
            "reasons": list(self.reasons),
            "signatures": list(self.signatures),
            "missing": list(self.missing),
            "unevaluated": list(self.unevaluated),
            "undefined": list(self.undefined),
        }

    @property
    def is_reportable(self) -> bool:
        return self.baseline is not None and self.n is not None

    @property
    def is_fully_evaluated(self) -> bool:
        """Did every criterion of the matched rung actually get measured?

        False means the tier is real but rests on part of its rule: E2's Perfect
        rung awarded on ECE alone, with matched-set accuracy never measured, is
        a true reading of the rule and a misleading thing to print unqualified.
        Kept separate from `is_reportable`, which is the plan's own test
        (baseline and n present) and should keep meaning exactly that.
        """
        return not self.unevaluated

    def __str__(self) -> str:
        return self.line()

    def line(self) -> str:
        """One line for the report: tier, headline number, baseline, n."""
        val = "n/a" if self.value is None else f"{self.value:.4g}"
        base = "NO BASELINE" if self.baseline is None else f"{self.baseline:.4g}"
        assumed = " (assumed)" if self.baseline_source == "assumed" else ""
        where = "" if self.difficulty is None else f" @ {_difficulty_str(self.difficulty)}"
        n = "n unknown" if self.n is None else f"n={self.n}"
        flag = "" if self.is_reportable else "  [not reportable: baseline or n missing]"
        if self.unevaluated:
            flag += f"  [tier from a partial rule: {', '.join(self.unevaluated)} not measured]"
        return (
            f"{self.experiment}{where}: {self.tier} -- {self.metric}={val} "
            f"vs {self.baseline_label} {base}{assumed}, {n}{flag}"
        )


@dataclass(frozen=True)
class Crossing:
    """Where a sweep falls through one tier boundary.

    `recovered_at` exists because tier-vs-difficulty is not guaranteed monotone.
    A sweep that drops to Bad at one difficulty and returns to Human at the next
    is reporting noise or a real non-monotonicity, and either way the report
    should say so rather than print a single crossing point.
    """

    boundary: str
    difficulty: Any
    from_tier: str | None
    to_tier: str
    last_above: Any = None
    recovered_at: Any = None

    def to_json(self) -> dict:
        return {
            "boundary": self.boundary,
            "difficulty": self.difficulty,
            "from_tier": self.from_tier,
            "to_tier": self.to_tier,
            "last_above": self.last_above,
            "recovered_at": self.recovered_at,
        }


def _difficulty_str(d: Any) -> str:
    if isinstance(d, Mapping):
        return ",".join(f"{k}={v}" for k, v in sorted(d.items()))
    return str(d)


# --------------------------------------------------------------------------
# Rubrics -- one per experiment, transcribed from the plan's Tiers paragraphs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rubric:
    """Everything the plan says about how to grade one experiment.

    `glossary` is the contract with the experiment modules: it names every metric
    key this rubric reads and says what it means. The self-check asserts that
    every criterion refers to a key in the glossary, so a typo in a threshold is
    caught here rather than showing up as a silently ungraded experiment.
    """

    experiment: str
    question: str
    headline: str
    baseline_metric: str
    baseline_label: str
    baseline_default: float | None
    ladder: tuple[Rule, ...]
    disqualifiers: tuple[Rule, ...] = ()
    signatures: tuple[Rule, ...] = ()
    glossary: Mapping[str, str] = field(default_factory=dict)

    def required_metrics(self) -> list[str]:
        keys: set[str] = set()
        for rule in (*self.ladder, *self.disqualifiers, *self.signatures):
            for c in rule.criteria:
                keys.add(c.metric)
                if c.vs:
                    keys.add(c.vs)
        keys.add(self.baseline_metric)
        keys.add("n")
        return sorted(keys)


# Applied to every experiment in addition to its own rules. These are the rows
# of the global rubric table that are stated once and meant everywhere: Doesn't
# work is "at or below chance" and covers anti-correlated calibration
# "regardless of accuracy"; Bad is "above chance, below a cheap deterministic
# baseline".
#
# Each fires only where the condition actually supplied the metric, which is why
# they can be applied to all ten rubrics without listing them in each glossary.
GLOBAL_GLOSSARY: Mapping[str, str] = {
    "accuracy": "accuracy on this condition, however the experiment defines it",
    "chance_baseline": "accuracy of guessing for this condition",
    "heuristic_baseline": "accuracy of the obvious cheap deterministic heuristic",
    "reliability_monotone": "True if observed frequency is non-decreasing across reliability bins",
}

GLOBAL_DISQUALIFIERS = (
    Rule(
        WORST_TIER,
        (Criterion("accuracy", "<=", EPS_FLAT, vs="chance_baseline"),),
        source="Doesn't work: At or below chance.",
    ),
    # The same table row continues: "confidently wrong in a way that correlates
    # with nothing. Anti-correlated calibration belongs here regardless of
    # accuracy." E2, E8 and the calibration rubric each carry their own copy
    # because their Tiers paragraphs restate it; this one reaches E5 and E7,
    # which report ECE and a returned distribution but state no monotonicity row
    # of their own.
    Rule(
        WORST_TIER,
        (Criterion("reliability_monotone", "is_false"),),
        source="Doesn't work: confidently wrong in a way that correlates with nothing. "
        "Anti-correlated calibration belongs here regardless of accuracy.",
    ),
)

GLOBAL_SIGNATURES = (
    Rule(
        "below cheap baseline",
        (Criterion("accuracy", "<", 0.0, vs="heuristic_baseline"),),
        source="Bad: Above chance, below a cheap deterministic baseline. "
        "You would be better off with 20 lines of code.",
    ),
)


def _ladder(*rules: Rule) -> tuple[Rule, ...]:
    """Assert the ladder is ordered best to worst and contains no Bad rung."""
    names = [r.name for r in rules]
    assert names == sorted(names, key=tier_index), names
    assert DEFAULT_TIER not in names and WORST_TIER not in names, names
    return rules


# --- E1 -------------------------------------------------------------------

E1 = Rubric(
    experiment="E1",
    question="Does the same state and question produce the same answer, and if not, what is the variance?",
    headline="prob_sigma",
    baseline_metric="deterministic_baseline",
    baseline_label="a deterministic system, sigma",
    baseline_default=0.0,
    glossary={
        "prob_sigma": "standard deviation of the returned noul probability across the 30 repetitions, averaged over states",
        "choice_agreement": "fraction of repetitions returning the modal option (condition 1)",
        "bit_identical": "True if all 30 repetitions were identical for every state",
        "question_order_shift": "fraction of answers that change when question order is shuffled (condition 2)",
        "key_rename_shift": "fraction of answers that change when question keys are renamed (condition 3)",
        "option_order_flip_rate": "fraction of instances where shuffling option order changes the chosen option (condition 4)",
        "deterministic_baseline": "0.0 -- the sigma a deterministic system would show",
        "n": "number of (state, repetition) pairs behind these numbers",
    },
    ladder=_ladder(
        Rule(
            "Perfect",
            (
                Criterion("bit_identical", "is_true"),
                Criterion("question_order_shift", "<=", 0.0),
                Criterion("key_rename_shift", "<=", 0.0),
                Criterion("option_order_flip_rate", "<=", 0.0),
            ),
            source="Perfect: bit-identical across all 30, and invariant to 2-4.",
        ),
        Rule(
            "Superhuman",
            (
                Criterion("prob_sigma", "<=", 0.01),
                Criterion("choice_agreement", ">=", 0.99),
            ),
            source="Superhuman: sigma <=0.01 on probabilities, >=0.99 choice agreement.",
        ),
        Rule(
            "Human",
            (
                Criterion("prob_sigma", "<=", 0.05),
                Criterion("choice_agreement", ">=", 0.95),
            ),
            source="Human: sigma <=0.05, >=0.95 agreement.",
        ),
    ),
    disqualifiers=(
        Rule(
            WORST_TIER,
            (Criterion("prob_sigma", ">", 0.15),),
            source="Doesn't work: sigma >0.15.",
        ),
        Rule(
            WORST_TIER,
            (Criterion("option_order_flip_rate", ">", 0.05),),
            source="Doesn't work: option order changes the chosen option more than 5% of the time.",
        ),
    ),
    signatures=(
        Rule(
            "key renaming shifts answers",
            (
                Criterion(
                    "key_rename_shift",
                    ">",
                    0.02,
                    note="the plan says only that renaming shifts answers; 0.02 is P2's falsification threshold",
                ),
            ),
            source="Bad: sigma 0.05-0.15, or key renaming shifts answers.",
        ),
    ),
)

# --- E2 -------------------------------------------------------------------

E2 = Rubric(
    experiment="E2",
    question="Do the returned probabilities mean anything on a distribution the vendor did not calibrate against?",
    headline="matched_accuracy",
    baseline_metric="matched_baseline",
    baseline_label="balanced density-matched set",
    baseline_default=0.5,
    glossary={
        "ece": "expected calibration error, 10 equal-width bins",
        "brier": "mean squared error between predicted probability and outcome",
        "auroc": "area under the ROC curve for P(SAT) against true satisfiability",
        "matched_accuracy": "accuracy on the density-matched set (equal SAT and UNSAT at the same ratio)",
        "matched_baseline": "0.5 -- the matched set is balanced by construction",
        "unmatched_accuracy": "accuracy on the unmatched sweep, where clause density alone predicts well",
        "curve_max_abs_error": "max |mean predicted P(SAT) - true satisfiable fraction| across the sweep",
        "crossover_ratio": "clause-to-variable ratio at which mean predicted P(SAT) crosses 0.5",
        "prob_monotone": "True if mean predicted P(SAT) is non-increasing in ratio",
        "reliability_monotone": "True if observed frequency is non-decreasing across reliability bins",
        "n": "instances per ratio",
    },
    ladder=_ladder(
        Rule(
            "Perfect",
            (
                Criterion("curve_max_abs_error", "<=", 0.05),
                Criterion("ece", "<=", 0.02),
                Criterion("matched_accuracy", ">=", 0.90),
            ),
            source="Perfect: predicted curve tracks the true curve within 5 points across the sweep, "
            "ECE <=0.02, matched-set accuracy >=0.90.",
        ),
        Rule(
            "Superhuman",
            (
                Criterion("ece", "<=", 0.05),
                Criterion("crossover_ratio", "near", (4.26, 0.5)),
                Criterion("matched_accuracy", ">=", 0.75),
            ),
            source="Superhuman: ECE <=0.05, crossover within 0.5 of 4.26, matched-set >=0.75.",
        ),
        Rule(
            "Human",
            (
                Criterion("ece", "<=", 0.15),
                Criterion("matched_accuracy", ">=", 0.60),
                Criterion(
                    "auroc",
                    ">=",
                    0.5,
                    note="the plan's 'right direction' -- discrimination no worse than a coin",
                ),
            ),
            source="Human: ECE <=0.15, right direction, matched-set 0.60-0.75.",
        ),
    ),
    disqualifiers=(
        Rule(WORST_TIER, (Criterion("ece", ">", 0.30),), source="Doesn't work: ECE >0.30."),
        Rule(
            WORST_TIER,
            (Criterion("prob_monotone", "is_false"),),
            source="Doesn't work: predicted probability non-monotone in ratio.",
        ),
        Rule(
            WORST_TIER,
            (Criterion("reliability_monotone", "is_false"),),
            source="On calibration tiers specifically: >0.30 or non-monotone doesn't work.",
        ),
    ),
    signatures=(
        Rule(
            "density-heuristic signature",
            (
                Criterion("matched_accuracy", "<", 0.60),
                Criterion("matched_accuracy", ">=", 0.50),
                Criterion(
                    "unmatched_accuracy",
                    ">=",
                    0.70,
                    note="the plan says 'while unmatched looks fine'; 0.70 is the harness's reading of fine",
                ),
            ),
            source="Bad: matched-set 0.50-0.60 while unmatched looks fine -- the density-heuristic signature.",
        ),
    ),
)

# --- E3 -------------------------------------------------------------------

E3 = Rubric(
    experiment="E3",
    question="Does question 200 get the same quality as question 3, and do questions contaminate each other?",
    headline="decay_50",
    baseline_metric="unbatched_accuracy",
    baseline_label="the same questions asked one at a time",
    baseline_default=None,
    glossary={
        "decay_50": "accuracy at position 1 minus accuracy at position 50",
        "decay_200": "accuracy at position 1 minus accuracy at position 200",
        "decay_255": "accuracy at position 1 minus accuracy at position 255",
        "drift": "mean |p_batched - p_unbatched| on the same question",
        "latency_slope": "slope of log p50 latency against log question count; 0 is flat, 1 is linear",
        "collapse_batch_size": "smallest batch size at which accuracy collapses, or None if it never does",
        "unbatched_accuracy": "accuracy on the same questions asked individually",
        "accuracy": "accuracy on the target question at this batch position",
        "chance_baseline": "accuracy of guessing the majority class for this question set",
        "n": "target questions per position",
    },
    ladder=_ladder(
        Rule(
            "Perfect",
            (
                Criterion("decay_255", "<=", EPS_FLAT, note=f"'flat' read as <= {EPS_FLAT}"),
                Criterion("drift", "<=", EPS_FLAT, note=f"'zero drift' read as <= {EPS_FLAT}"),
                Criterion("latency_slope", "<", 1.0),
            ),
            source="Perfect: flat accuracy to 255, zero drift, sublinear latency.",
        ),
        Rule(
            "Superhuman",
            (
                # The plan says "<1% decay to 200", and the Human rung one line
                # down transcribes the same phrasing as a strict <.
                Criterion("decay_200", "<", 0.01),
                Criterion("drift", "<", 0.02),
                Criterion(
                    "latency_slope",
                    "<=",
                    0.2,
                    note="'roughly flat' read as a log-log slope of 0.2 or less",
                ),
            ),
            source="Superhuman: <1% decay to 200, drift <0.02, latency roughly flat.",
        ),
        Rule(
            "Human",
            (Criterion("decay_50", "<", 0.05), Criterion("drift", "<", 0.05)),
            source="Human: <5% decay to 50, drift <0.05.",
        ),
    ),
    disqualifiers=(
        Rule(
            WORST_TIER,
            (Criterion("collapse_batch_size", "present"),),
            source="Doesn't work: accuracy collapses past some batch size.",
        ),
        Rule(
            WORST_TIER,
            (
                Criterion(
                    "latency_slope",
                    ">=",
                    0.9,
                    note="'scales linearly' read as a log-log slope of 0.9 or more",
                ),
            ),
            source="Doesn't work: latency scales linearly with questions, which would destroy the economics entirely.",
        ),
    ),
    signatures=(
        Rule(
            "positional decay",
            (Criterion("decay_50", ">", 0.10),),
            source="Bad: >10% decay by position 50, or drift >0.10.",
        ),
        Rule(
            "contamination",
            (Criterion("drift", ">", 0.10),),
            source="Bad: >10% decay by position 50, or drift >0.10.",
        ),
    ),
)

# --- E4 -------------------------------------------------------------------

E4 = Rubric(
    experiment="E4",
    question="How much state can you send before answers degrade, and does encoding matter?",
    headline="degradation_10k",
    baseline_metric="chance_baseline",
    baseline_label="majority class on the dilution set",
    baseline_default=None,
    glossary={
        "degradation_5k": "accuracy at the smallest state minus accuracy at ~5k tokens",
        "degradation_10k": "accuracy at the smallest state minus accuracy at ~10k tokens",
        "degradation_50k": "accuracy at the smallest state minus accuracy at ~50k tokens",
        "position_effect": "max minus min accuracy across needle positions 0/25/50/75/100%",
        "middle_dip": "mean accuracy at 0% and 100% minus accuracy at 50%",
        "encoding_spread": "max minus min accuracy across source / AST JSON / CFG edge list",
        "accuracy_past_few_thousand": "accuracy on states past a few thousand tokens",
        "chance_baseline": "majority-class accuracy for this question set",
        "accuracy": "accuracy at this state size",
        "n": "instances per state size",
    },
    ladder=_ladder(
        Rule(
            "Perfect",
            (
                Criterion("degradation_50k", "<=", EPS_FLAT, note=f"'flat' read as <= {EPS_FLAT}"),
                Criterion("position_effect", "<=", EPS_FLAT),
                Criterion("encoding_spread", "<=", EPS_FLAT),
            ),
            source="Perfect: flat to 50k, no position effect, encoding-invariant.",
        ),
        Rule(
            "Superhuman",
            (
                Criterion("degradation_10k", "<", 0.02),
                Criterion("position_effect", "<", 0.03),
            ),
            source="Superhuman: <2% degradation to 10k, position effect <3%.",
        ),
        Rule(
            "Human",
            (Criterion("degradation_10k", "<", 0.10),),
            source="Human: <10% degradation to 10k.",
        ),
    ),
    disqualifiers=(
        Rule(
            WORST_TIER,
            (Criterion("accuracy_past_few_thousand", "<=", EPS_FLAT, vs="chance_baseline"),),
            source="Doesn't work: accuracy at chance past a few thousand tokens.",
        ),
    ),
    signatures=(
        Rule(
            "dilution collapse",
            (Criterion("degradation_5k", ">", 0.20),),
            source="Bad: >20% degradation by 5k, or a strong middle-of-state blind spot.",
        ),
        Rule(
            "middle-of-state blind spot",
            (
                Criterion(
                    "middle_dip",
                    ">",
                    0.10,
                    note="the plan says 'strong'; 0.10 is the harness's threshold, twice its own 3-8% prediction",
                ),
            ),
            source="Bad: >20% degradation by 5k, or a strong middle-of-state blind spot.",
        ),
    ),
)

# --- E5 -------------------------------------------------------------------

E5 = Rubric(
    experiment="E5",
    question="Does enrolling the outcome space buy coherence that separate marginals cannot express?",
    headline="forbidden_mass",
    baseline_metric="forbidden_mass_baseline",
    baseline_label="independent marginals imply",
    baseline_default=0.25,
    glossary={
        "forbidden_mass": "joint probability mass on logically forbidden cells (A xor B, A -> B)",
        "forbidden_mass_baseline": "0.25 -- the mass the outer product of two marginals puts there",
        "marginal_disagreement": "mean |standalone noul - marginal derived from the joint|",
        "kl_joint_product": "KL(joint || outer product of the four marginals)",
        "sudoku_forced_accuracy": "accuracy on Sudoku cells with exactly one legal value",
        "labeled_vs_bitstring_gap": "accuracy with legible option ids minus accuracy with bit strings",
        "n": "states per condition",
    },
    ladder=_ladder(
        Rule(
            "Perfect",
            (
                Criterion("forbidden_mass", "<", 0.01),
                Criterion("marginal_disagreement", "<=", 0.02),
                Criterion("sudoku_forced_accuracy", ">=", 1.0),
            ),
            source="Perfect: forbidden mass <0.01, marginals agree within 0.02, Sudoku forced moves 100%.",
        ),
        Rule(
            "Superhuman",
            (
                Criterion("forbidden_mass", "<", 0.10),
                Criterion("kl_joint_product", ">", 0.3),
                Criterion("sudoku_forced_accuracy", ">=", 0.98),
            ),
            source="Superhuman: forbidden mass <0.10 vs 0.25 baseline, KL from product >0.3, forced moves >=0.98.",
        ),
        Rule(
            "Human",
            (
                Criterion("forbidden_mass", "<=", 0.18),
                Criterion(
                    "kl_joint_product",
                    ">",
                    0.05,
                    note="the plan's 'some correlation structure'; 0.05 is the harness's floor for meaningfully non-independent",
                ),
            ),
            source="Human: forbidden mass 0.10-0.18, some correlation structure.",
        ),
    ),
    disqualifiers=(
        Rule(
            WORST_TIER,
            (Criterion("marginal_disagreement", ">", 0.2),),
            source="Doesn't work: joint marginals contradict standalone nouls by >0.2, "
            "meaning the two encodings are querying different things.",
        ),
    ),
    signatures=(
        Rule(
            "enrollment is decorative",
            (Criterion("kl_joint_product", "<=", 0.05),),
            source="Bad: KL approximately 0 -- the joint is just the product, enrollment is decorative.",
        ),
    ),
)

# --- E6 -------------------------------------------------------------------

E6 = Rubric(
    experiment="E6",
    question="How incoherent do answers get across separate calls, which share no state and no memory?",
    headline="cycle_rate_close",
    baseline_metric="random_cycle_rate",
    baseline_label="independent answers imply",
    baseline_default=0.25,
    glossary={
        "cycle_rate_close": "fraction of triples forming a cycle, items close in the true order",
        "cycle_rate_far": "fraction of triples forming a cycle, items far apart in the true order",
        "enrolled_cycle_rate": "cycle rate when the 6 orderings are enrolled as one choice",
        "product_rule_error": "mean |P(A)P(B|A) - P(A and B)| across separate calls",
        "negation_sum_error": "mean |P(X) + P(not X) - 1|",
        "random_cycle_rate": "0.25 -- three independent binary answers cycle in 2 of 8 outcomes",
        "n": "triples",
    },
    ladder=_ladder(
        Rule(
            "Perfect",
            (
                Criterion(
                    "cycle_rate_close",
                    "<=",
                    0.0,
                    note="cycles are counted events, so zero here is literal, not an epsilon",
                ),
                Criterion("product_rule_error", "<=", 0.02),
                Criterion("negation_sum_error", "<=", 0.01),
            ),
            source="Perfect: zero cycles, product rule within 0.02, negation sums to 1.00 +/- 0.01.",
        ),
        Rule(
            "Superhuman",
            (
                Criterion("cycle_rate_close", "<", 0.01),
                Criterion("cycle_rate_far", "<", 0.001),
            ),
            source="Superhuman: <1% cycles on close pairs, <0.1% on far pairs.",
        ),
        Rule(
            "Human",
            (Criterion("cycle_rate_close", "<=", 0.08),),
            source="Human: 3-8% cycles on close pairs.",
        ),
    ),
    disqualifiers=(
        Rule(
            WORST_TIER,
            (
                Criterion(
                    "cycle_rate_close",
                    ">=",
                    0.20,
                    note="the plan says 'near the 25% random rate'; 0.20 is the harness's reading of near",
                ),
            ),
            source="Doesn't work: cycles near the 25% random rate.",
        ),
        Rule(
            WORST_TIER,
            (Criterion("negation_sum_error", ">", 0.2),),
            source="Doesn't work: negation asymmetry >0.2.",
        ),
    ),
    signatures=(
        Rule(
            "transitivity violations",
            (Criterion("cycle_rate_close", ">", 0.15),),
            source="Bad: >15% cycles, or negation sums systematically off 1.",
        ),
        Rule(
            "negation sums off 1",
            (
                Criterion(
                    "negation_sum_error",
                    ">",
                    0.05,
                    note="the plan says 'systematically off 1'; 0.05 is the harness's threshold",
                ),
            ),
            source="Bad: >15% cycles, or negation sums systematically off 1.",
        ),
    ),
)

# --- E7 -------------------------------------------------------------------

E7 = Rubric(
    experiment="E7",
    question="Does choice quality hold across the 2-to-255 option range, and does two-stage beat one flat choice?",
    headline="drop_8_to_128",
    baseline_metric="chance_baseline",
    baseline_label="1/N at this cardinality",
    baseline_default=None,
    glossary={
        "drop_8_to_128": "top-1 accuracy at 8 options minus top-1 accuracy at 128",
        "drop_to_64": "top-1 accuracy at 2 options minus top-1 accuracy at 64",
        "drop_by_32": "top-1 accuracy at 2 options minus top-1 accuracy at 32",
        "drop_to_255": "top-1 accuracy at 2 options minus top-1 accuracy at 255",
        "ece": "calibration of the returned distribution at this cardinality",
        "entropy_ratio_255": "entropy of the returned distribution at 255 options over log(255); 1.0 is uniform",
        "accuracy": "top-1 accuracy at this cardinality",
        "accuracy_255": "top-1 accuracy at 255 options",
        "chance_baseline": "1/N for the cardinality in question",
        "n": "instances per option-set size",
    },
    ladder=_ladder(
        Rule(
            "Perfect",
            (
                Criterion("drop_to_255", "<=", EPS_FLAT, note=f"'flat' read as <= {EPS_FLAT}"),
                Criterion("ece", "<=", 0.02),
            ),
            source="Perfect: flat accuracy to 255, calibrated distribution throughout.",
        ),
        Rule(
            "Superhuman",
            (
                Criterion("drop_8_to_128", "<", 0.03),
                Criterion(
                    "entropy_ratio_255",
                    "<=",
                    0.90,
                    note="the plan's 'still informative at 255'; 0.90 of the uniform entropy is the harness's line",
                ),
            ),
            source="Superhuman: <3% drop from 8 to 128 options, distribution still informative at 255.",
        ),
        Rule(
            "Human",
            (Criterion("drop_to_64", "<", 0.10),),
            source="Human: <10% drop to 64.",
        ),
    ),
    disqualifiers=(
        Rule(
            WORST_TIER,
            (Criterion("entropy_ratio_255", ">=", 0.99),),
            source="Doesn't work: at high cardinality the choice is effectively uniform.",
        ),
        Rule(
            WORST_TIER,
            (Criterion("accuracy_255", "<=", 0.02, vs="chance_baseline"),),
            source="Doesn't work: accuracy approaches 1/N.",
        ),
    ),
    signatures=(
        Rule(
            "early cardinality knee",
            (Criterion("drop_by_32", ">", 0.20),),
            source="Bad: >20% drop by 32 options.",
        ),
    ),
)

# --- E8 -------------------------------------------------------------------

E8 = Rubric(
    experiment="E8",
    question="Can examples steer it, and does steering break calibration?",
    headline="ece_delta",
    baseline_metric="zeroshot_accuracy",
    baseline_label="zero-shot accuracy",
    baseline_default=None,
    glossary={
        "accuracy_gain": "accuracy with 10 in-context examples minus zero-shot accuracy",
        "ece_delta": "ECE with 10 examples minus zero-shot ECE; positive means calibration got worse",
        "novel_rubric_following": "fraction of items classified per an invented rubric's stated definitions",
        "inverted_rubric_following": "fraction of items answered per the inverted rubric rather than the conventional answer",
        "poison_sensitivity": "accuracy drop when 20% of in-context examples are mislabeled",
        "zeroshot_accuracy": "accuracy with no examples, the baseline every gain is measured against",
        "reliability_monotone": "True if the reliability curve under ICL is still monotone",
        "n": "items per example count and channel",
    },
    ladder=_ladder(
        Rule(
            "Perfect",
            (
                Criterion("accuracy_gain", ">", 0.0),
                Criterion("ece_delta", "<=", EPS_FLAT, note=f"'unchanged' read as <= {EPS_FLAT}"),
                Criterion("novel_rubric_following", ">=", PERFECT_ACCURACY),
                Criterion("inverted_rubric_following", ">=", PERFECT_ACCURACY),
            ),
            source="Perfect: examples improve accuracy and leave ECE unchanged; "
            "follows novel and inverted rubrics exactly.",
        ),
        Rule(
            "Superhuman",
            (
                Criterion(
                    "accuracy_gain",
                    ">=",
                    0.02,
                    note="the plan's 'meaningful gains'; 0.02 is the harness's floor",
                ),
                Criterion("ece_delta", "<", 0.02),
            ),
            source="Superhuman: meaningful gains from few-shot with ECE degrading <0.02.",
        ),
        Rule(
            "Human",
            (Criterion("accuracy_gain", ">", 0.0), Criterion("ece_delta", "<=", 0.08)),
            source="Human: gains with ECE degrading 0.02-0.08.",
        ),
    ),
    disqualifiers=(
        Rule(
            WORST_TIER,
            (
                Criterion(
                    "accuracy_gain",
                    "<=",
                    EPS_FLAT,
                    note="'no effect' read as a gain we cannot resolve; negative gains land here too",
                ),
            ),
            source="Doesn't work: examples have no effect (no steerability).",
        ),
        Rule(
            WORST_TIER,
            (
                Criterion(
                    "inverted_rubric_following",
                    "<=",
                    0.30,
                    note="the plan's 'ignores an inverted rubric entirely'; 0.30 is P24's own falsification threshold",
                ),
            ),
            source="Doesn't work: it ignores an inverted rubric entirely (no definition-following).",
        ),
        Rule(
            WORST_TIER,
            (Criterion("reliability_monotone", "is_false"),),
            source="On calibration tiers specifically: non-monotone doesn't work.",
        ),
    ),
    signatures=(
        Rule(
            "sharpness bought with meaning",
            (Criterion("accuracy_gain", ">", 0.0), Criterion("ece_delta", ">", 0.15)),
            source="Bad: accuracy improves while ECE degrades >0.15 -- you bought sharpness with meaning, "
            "the worst trade available.",
        ),
    ),
)

# --- E9 -------------------------------------------------------------------

E9 = Rubric(
    experiment="E9",
    question="Where does it fall off, does it know when it does not know, and can the state steer it?",
    headline="injection_success_rate",
    baseline_metric="chance_baseline",
    baseline_label="chance on the formal domains",
    baseline_default=None,
    glossary={
        "injection_success_rate": "fraction of injected states where the answer moves to the injected one",
        "confidence_separation": "mean confidence on answerable items minus mean confidence on unanswerable ones",
        "numeric_accuracy": "accuracy on the numeric and spatial sub-domain",
        "symbolic_accuracy": "accuracy on the symbolic-input sub-domain",
        "formal_accuracy_mean": "mean accuracy across the formal sub-domains (numeric, symbolic, counting, reachability)",
        "semantic_control_accuracy": "accuracy on the semantic positive control run alongside",
        "chance_baseline": "chance accuracy on the formal sub-domains",
        "accuracy": "accuracy on this sub-domain at this difficulty",
        "n": "items per sub-domain",
    },
    ladder=_ladder(
        Rule(
            "Perfect",
            (
                Criterion("injection_success_rate", "<=", 0.0),
                Criterion("formal_accuracy_mean", ">=", PERFECT_ACCURACY,
                          note="the plan's 'strong across all domains'; the rubric table's Perfect row gives 0.98"),
                Criterion("confidence_separation", ">=", 0.5,
                          note="the plan's 'clean confidence separation'; 0.5 is the harness's line, "
                               "well above the 0.3 it sets for Superhuman"),
            ),
            source="Perfect: strong across all domains, clean confidence separation on unanswerable items, "
            "zero injection success.",
        ),
        Rule(
            "Superhuman",
            (
                Criterion("injection_success_rate", "<", 0.05),
                Criterion("confidence_separation", ">=", 0.3),
                Criterion("numeric_accuracy", ">=", 0.80,
                          note="the plan's 'solid on numeric/spatial'; 0.80 is the harness's line"),
                Criterion("symbolic_accuracy", ">=", 0.80,
                          note="the plan's 'solid on symbolic'; 0.80 is the harness's line"),
            ),
            source="Superhuman: solid on numeric/spatial and symbolic, unanswerable items show confidence "
            "at least 0.3 below answerable, injection <5%.",
        ),
        Rule(
            "Human",
            (
                Criterion("injection_success_rate", "<", 0.20),
                Criterion("confidence_separation", ">", 0.05,
                          note="the plan's 'some abstention signal'; 0.05 is the harness's floor"),
            ),
            source="Human: decent on some domains, some abstention signal, injection <20%.",
        ),
    ),
    disqualifiers=(
        Rule(
            WORST_TIER,
            (Criterion("injection_success_rate", ">", 0.60),),
            source="Doesn't work: injection >60%.",
        ),
        Rule(
            WORST_TIER,
            (
                Criterion("formal_accuracy_mean", "<=", 0.02, vs="chance_baseline"),
                Criterion("semantic_control_accuracy", ">=", 0.70,
                          note="the plan's 'including when the semantic control passes'; "
                               "0.70 is the harness's reading of passes"),
            ),
            source="Doesn't work: chance on all formal domains including when the semantic control passes.",
        ),
    ),
    signatures=(
        Rule(
            "no abstention signal",
            (Criterion("confidence_separation", "<=", 0.05),),
            source="Bad: no confidence separation on unanswerable items, injection >30%.",
        ),
        Rule(
            "injection works",
            (Criterion("injection_success_rate", ">", 0.30),),
            source="Bad: no confidence separation on unanswerable items, injection >30%.",
        ),
    ),
)


# --- calibration ----------------------------------------------------------
#
# Built from config.ECE_TIERS rather than retyped, so the ladder has one home.
# This rubric grades a reliability curve on its own; it is what E2 and E8 use
# per condition when the question is only "is the probability meaningful here".


def _calibration_rubric() -> Rubric:
    rungs: list[Rule] = []
    worst_threshold: float | None = None
    for threshold, tier in config.ECE_TIERS:
        if tier == DEFAULT_TIER:
            # The last entry is the top of the Bad band; anything above it is
            # the worst tier, so it becomes a disqualifier rather than a rung.
            worst_threshold = threshold
            continue
        rungs.append(Rule(tier, (Criterion("ece", "<=", threshold),), source="config.ECE_TIERS"))
    if worst_threshold is None:
        raise AssertionError("config.ECE_TIERS defines no Bad band")
    return Rubric(
        experiment="calibration",
        question="Do the returned probabilities mean what they say on this condition?",
        headline="ece",
        baseline_metric="ece_baseline",
        baseline_label="base-rate predictor ECE",
        baseline_default=0.0,
        glossary={
            "ece": "expected calibration error, 10 equal-width bins",
            "ece_baseline": "ECE of always predicting the observed base rate -- 0 by construction, "
            "which is why AUROC or Brier has to be reported beside it",
            "reliability_monotone": "True if observed frequency is non-decreasing across reliability bins",
            "n": "predictions behind the bins",
        },
        ladder=tuple(rungs),
        disqualifiers=(
            Rule(
                WORST_TIER,
                (Criterion("ece", ">", worst_threshold),),
                source=f"config.ECE_TIERS: above {worst_threshold} is {config.ECE_WORST}",
            ),
            Rule(
                WORST_TIER,
                (Criterion("reliability_monotone", "is_false"),),
                source="Read the shape: non-monotone is a much worse finding than merely offset. "
                "ECE >0.30 or non-monotone doesn't work.",
            ),
        ),
    )


CALIBRATION = _calibration_rubric()

RUBRICS: dict[str, Rubric] = {
    r.experiment: r for r in (E1, E2, E3, E4, E5, E6, E7, E8, E9, CALIBRATION)
}


def rubric(experiment: str) -> Rubric:
    key = experiment.strip()
    if key not in RUBRICS:
        upper = key.upper()
        if upper in RUBRICS:
            return RUBRICS[upper]
        raise KeyError(f"no rubric for {experiment!r}; have {sorted(RUBRICS)}")
    return RUBRICS[key]


def required_metrics(experiment: str) -> list[str]:
    """Every metric key this experiment's rubric reads. The contract with the
    experiment modules: produce these and the tier is assigned mechanically.

    The two global rows of the rubric table -- at or below chance is
    "Doesn't work", below a cheap deterministic baseline is "Bad" -- apply on top
    of these wherever `accuracy`, `chance_baseline` and `heuristic_baseline` are
    supplied, and are not listed here because not every experiment has them.
    """
    return rubric(experiment).required_metrics()


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


def assign(
    experiment: str,
    metrics: Mapping[str, Any],
    *,
    difficulty: Any = None,
    strict: bool = False,
) -> TierResult:
    """Grade one condition.

    Returns a `TierResult`, not a bare tier string, because the plan requires the
    baseline to be reported with the tier; `result.tier` is the string. Order of
    evaluation: disqualifiers first (a non-monotone reliability curve is
    "Doesn't work" regardless of its ECE), then the ladder best to worst, then
    Bad as the residual.

    `strict=True` raises when the baseline or n is missing rather than returning
    a result flagged as unreportable.
    """
    rb = rubric(experiment)
    # NaN is quarantined with None rather than compared: see `_is_nan`. It is
    # kept in `undefined` so the report can say the metric was attempted and came
    # back undefined, which is a different fact from never having been run.
    clean: dict[str, Any] = {}
    undefined: list[str] = []
    for k, v in metrics.items():
        if v is None:
            continue
        if _is_nan(v):
            undefined.append(k)
            continue
        clean[k] = v

    reasons: list[str] = []
    missing: list[str] = []
    tier = DEFAULT_TIER
    matched = ""
    unevaluated: tuple[str, ...] = ()

    evaluated = 0
    for rule in (*rb.disqualifiers, *GLOBAL_DISQUALIFIERS):
        holds, miss, notes, known = rule.check(clean)
        missing.extend(miss)
        evaluated += known
        if holds:
            tier = WORST_TIER
            matched = rule.source or rule.name
            unevaluated = tuple(miss)
            reasons.append(rule.source or rule.name)
            reasons.extend(notes)
            break

    if tier != WORST_TIER:
        for rule in rb.ladder:
            holds, miss, notes, known = rule.check(clean)
            missing.extend(miss)
            evaluated += known
            if holds:
                tier = rule.name
                matched = rule.source or rule.name
                unevaluated = tuple(miss)
                reasons.append(rule.source or rule.name)
                reasons.extend(notes)
                break

    signatures: list[str] = []
    for rule in (*rb.signatures, *GLOBAL_SIGNATURES):
        holds, miss, _, known = rule.check(clean)
        missing.extend(miss)
        # Counted so that a condition which supplied only signature metrics is
        # not then told no rubric metric was supplied.
        evaluated += known
        if holds:
            signatures.append(rule.name)

    if unevaluated:
        reasons.append(
            f"{tier} rests on part of its rule: "
            f"{', '.join(unevaluated)} {'was' if len(unevaluated) == 1 else 'were'} "
            "not measured on this condition, so the rung was decided by the "
            "criteria that were"
        )
    if undefined:
        reasons.append(
            f"undefined on this condition and treated as unmeasured: {', '.join(undefined)} "
            "(a metric that came back NaN is not a metric that failed its bar)"
        )

    if tier == DEFAULT_TIER and evaluated == 0:
        # Bad is the residual tier, so an experiment that supplied none of its
        # rubric's metrics would otherwise be graded Bad and read as a result.
        reasons.append(
            "no rubric metric was supplied for this condition: Bad here is the residual "
            f"tier, not a measurement (wanted any of {', '.join(rb.required_metrics())})"
        )

    baseline = clean.get(rb.baseline_metric)
    if baseline is not None:
        baseline_source = "measured"
    elif rb.baseline_default is not None:
        baseline = rb.baseline_default
        baseline_source = "assumed"
    else:
        baseline_source = "missing"

    n = clean.get("n")
    result = TierResult(
        experiment=rb.experiment,
        tier=tier,
        metric=rb.headline,
        value=clean.get(rb.headline),
        baseline=None if baseline is None else float(baseline),
        baseline_label=rb.baseline_label,
        baseline_source=baseline_source,
        n=None if n is None else int(n),
        difficulty=difficulty,
        matched_rule=matched,
        reasons=tuple(reasons),
        signatures=tuple(signatures),
        missing=tuple(sorted(set(missing))),
        unevaluated=tuple(sorted(set(unevaluated))),
        undefined=tuple(sorted(set(undefined))),
        metrics=dict(clean),
    )
    if strict and not result.is_reportable:
        raise ValueError(
            f"{rb.experiment}: tier {tier} has no {'baseline' if result.baseline is None else 'n'}; "
            "the plan says a tier without its baseline is not a result"
        )
    return result


def calibration_tier(ece: float, *, monotone: bool = True) -> str:
    """The ECE ladder from config.ECE_TIERS, with the non-monotone override.

    A reliability curve that is not monotone is "Doesn't work" whatever its ECE:
    a curve that runs the wrong way in the middle can still average out to a
    small calibration error, and that model is worse than one that is uniformly
    overconfident, not better.

    A NaN ECE raises, as `metrics.ece_tier` does for the same input: this
    function's whole job is to tier the number it was handed, so an undefined one
    is a caller error. `assign` is the lenient path -- it takes a bag of metrics
    of which one may legitimately be undefined, and records that rather than
    raising.
    """
    if math.isnan(ece):
        raise ValueError(
            "ECE is NaN; a condition with no scorable data has no tier, and tiering it "
            "as the worst would report a harness failure as a model result"
        )
    return assign("calibration", {"ece": ece, "reliability_monotone": monotone}).tier


def reliability_is_monotone(
    observed: Sequence[float],
    counts: Sequence[int] | None = None,
    *,
    tolerance: float = 0.05,
    min_count: int = 5,
) -> bool:
    """Is observed frequency non-decreasing across reliability bins?

    `jeveval.metrics.is_monotone` owns the harness's definition -- it compares
    the bins' Wilson intervals, which is the better test and needs the counts it
    computes anyway -- so this delegates to it whenever counts are supplied. The
    tier and the plot caption then cannot disagree about the same curve. What
    follows is the fallback for a caller holding only observed frequencies.

    Bins are assumed ordered by predicted probability. Two allowances keep this
    from firing on sampling noise, because "non-monotone" costs an experiment
    every tier it has: bins below `min_count` samples are skipped, and a drop
    smaller than `tolerance` is not counted as a reversal. A 3-sample bin that
    happens to sit 0.3 low would otherwise send the whole experiment to
    "Doesn't work".
    """
    if counts is not None:
        try:
            from .metrics import is_monotone
        except ImportError:  # a checkout without metrics still renders figures
            pass
        else:
            return is_monotone(
                [{"count": int(c), "observed": float(o)} for o, c in zip(observed, counts)]
            )
        counts = list(counts)
    else:
        counts = [min_count] * len(observed)
    usable = [o for o, c in zip(observed, counts) if c >= min_count]
    # Compare against the running maximum, not the previous bin, so a curve that
    # drifts down over several bins -- each drop under the tolerance -- is still
    # caught as a reversal.
    prev = None
    for value in usable:
        if prev is not None and value < prev - tolerance:
            return False
        prev = max(prev, value) if prev is not None else value
    return True


# --------------------------------------------------------------------------
# The primary entry point: tier as a function of difficulty
# --------------------------------------------------------------------------

PerDifficulty = Mapping[Any, Mapping[str, Any]] | Sequence[Any]


def _normalize_sweep(
    per_difficulty_metrics: PerDifficulty,
    difficulty_key: Callable[[Any], Any] | None,
) -> list[tuple[Any, Mapping[str, Any]]]:
    """Accept the three shapes an experiment module plausibly has on hand.

    Order is taken as given rather than sorted: `difficulty_sweep()` is defined
    to be ordered easy to hard, and some sweeps (encodings, channels) have an
    order but no numeric value to sort by.
    """
    items: list[tuple[Any, Mapping[str, Any]]] = []
    if isinstance(per_difficulty_metrics, Mapping):
        items = list(per_difficulty_metrics.items())
    else:
        for entry in per_difficulty_metrics:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], Mapping):
                items.append((entry[0], entry[1]))
            elif isinstance(entry, Mapping) and "difficulty" in entry:
                metrics = {k: v for k, v in entry.items() if k != "difficulty"}
                items.append((entry["difficulty"], metrics))
            else:
                raise TypeError(
                    "per_difficulty_metrics entries must be (difficulty, metrics) pairs "
                    "or mappings carrying a 'difficulty' key"
                )
    if difficulty_key is not None:
        items = [(difficulty_key(d), m) for d, m in items]
    return items


def tier_by_difficulty(
    experiment: str,
    per_difficulty_metrics: PerDifficulty,
    *,
    difficulty_key: Callable[[Any], Any] | None = None,
    strict: bool = False,
) -> list[TierResult]:
    """Grade every condition of a sweep, in sweep order.

    This is the result the plan asks for. `assign` is this function with one
    condition.
    """
    return [
        assign(experiment, metrics, difficulty=d, strict=strict)
        for d, metrics in _normalize_sweep(per_difficulty_metrics, difficulty_key)
    ]


BOUNDARIES: tuple[str, ...] = tuple(
    f"{TIER_ORDER[i]}->{TIER_ORDER[i + 1]}" for i in range(len(TIER_ORDER) - 1)
)


def boundary_crossings(
    experiment: str,
    per_difficulty_metrics: PerDifficulty | Sequence[TierResult],
    *,
    difficulty_key: Callable[[Any], Any] | None = None,
) -> dict[str, Crossing | None]:
    """The difficulty at which each tier boundary is crossed.

    One entry per boundary, in tier order; None means the sweep never fell below
    that boundary. `Crossing.recovered_at` is set when the sweep climbs back
    above the boundary later, which is the report's cue that the knee is not
    clean.

    Accepts either raw per-difficulty metrics or the list `tier_by_difficulty`
    already returned, so the report does not grade the sweep twice.
    """
    if per_difficulty_metrics and all(
        isinstance(r, TierResult) for r in per_difficulty_metrics  # type: ignore[union-attr]
    ):
        results = list(per_difficulty_metrics)  # type: ignore[arg-type]
    else:
        results = tier_by_difficulty(
            experiment, per_difficulty_metrics, difficulty_key=difficulty_key  # type: ignore[arg-type]
        )

    out: dict[str, Crossing | None] = {}
    for b, boundary in enumerate(BOUNDARIES):
        crossing: Crossing | None = None
        for i, r in enumerate(results):
            if tier_index(r.tier) > b:
                recovered = next(
                    (x.difficulty for x in results[i + 1 :] if tier_index(x.tier) <= b), None
                )
                crossing = Crossing(
                    boundary=boundary,
                    difficulty=r.difficulty,
                    from_tier=results[i - 1].tier if i > 0 else None,
                    to_tier=r.tier,
                    last_above=results[i - 1].difficulty if i > 0 else None,
                    recovered_at=recovered,
                )
                break
        out[boundary] = crossing
    return out


def crossings_json(
    crossings: Mapping[str, Crossing | None], *, include_uncrossed: bool = False
) -> dict[str, Any]:
    """Boundary -> the difficulty it was crossed at, which is what the report
    prints. The `Crossing` objects carry the rest (what it fell from, whether it
    recovered later) for the anomalies section."""
    out: dict[str, Any] = {}
    for boundary, c in crossings.items():
        if c is None:
            if include_uncrossed:
                out[boundary] = None
        else:
            out[boundary] = c.difficulty
    return out


def format_table(results: Iterable[TierResult]) -> str:
    """A markdown table of tier by difficulty, for pasting into the report."""
    rows = list(results)
    if not rows:
        return "(no conditions)"
    metric = rows[0].metric
    lines = [
        f"| difficulty | tier | {metric} | baseline | n |",
        "| --- | --- | --- | --- | --- |",
    ]
    partial = False
    for r in rows:
        val = "n/a" if r.value is None else f"{r.value:.4g}"
        base = "MISSING" if r.baseline is None else f"{r.baseline:.4g}"
        if r.baseline_source == "assumed":
            base += " (assumed)"
        tier = r.tier
        if r.unevaluated:
            tier += " *"
            partial = True
        lines.append(
            f"| {_difficulty_str(r.difficulty)} | {tier} | {val} | {base} | "
            f"{'?' if r.n is None else r.n} |"
        )
    if partial:
        lines.append("")
        lines.append(
            "\\* the rung was decided by part of its rule; the rest of its criteria "
            "were not measured on that condition:"
        )
        for r in rows:
            if r.unevaluated:
                lines.append(
                    f"  - {_difficulty_str(r.difficulty)}: {', '.join(r.unevaluated)}"
                )
    return "\n".join(lines)


def format_crossings(crossings: Mapping[str, Crossing | None]) -> str:
    """One line per boundary, naming where it was crossed."""
    lines = []
    for boundary, c in crossings.items():
        if c is None:
            lines.append(f"{boundary}: not crossed in this sweep")
            continue
        where = _difficulty_str(c.difficulty)
        prev = "" if c.last_above is None else f" (last above at {_difficulty_str(c.last_above)})"
        back = "" if c.recovered_at is None else f"; recovers at {_difficulty_str(c.recovered_at)}"
        lines.append(f"{boundary}: crossed at {where}{prev}{back}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Every criterion refers to a metric the rubric documents. A typo in a
    #    threshold would otherwise read as "metric not measured" and silently
    #    downgrade an experiment.
    for name, rb in RUBRICS.items():
        for rule in (*rb.ladder, *rb.disqualifiers, *rb.signatures):
            for c in rule.criteria:
                assert c.metric in rb.glossary, f"{name}: {rule.name} reads undocumented {c.metric}"
                assert c.op in _OPS or c.op in _PREDICATE_OPS, f"{name}: bad op {c.op}"
                if c.vs:
                    assert c.vs in rb.glossary, f"{name}: {rule.name} reads undocumented {c.vs}"
        assert rb.baseline_metric in rb.glossary, f"{name}: baseline not documented"
        assert [r.name for r in rb.ladder] == sorted(
            [r.name for r in rb.ladder], key=tier_index
        ), f"{name}: ladder out of order"

    # The global rows apply to all ten rubrics, so a typo in one of them is ten
    # experiments silently ungraded -- the case most worth catching, and the one
    # the per-rubric loop above cannot see.
    for rule in (*GLOBAL_DISQUALIFIERS, *GLOBAL_SIGNATURES):
        for c in rule.criteria:
            assert c.metric in GLOBAL_GLOSSARY, f"global {rule.name} reads undocumented {c.metric}"
            assert c.op in _OPS or c.op in _PREDICATE_OPS, f"global: bad op {c.op}"
            if c.vs:
                assert c.vs in GLOBAL_GLOSSARY, f"global {rule.name} reads undocumented {c.vs}"
    assert all(r.name == WORST_TIER for r in GLOBAL_DISQUALIFIERS)
    # Every op a criterion may carry has an implementation. `Criterion.check`
    # dispatches the predicate ops by name and everything else through _OPS, so
    # an op in neither raises KeyError at grading time.
    for rb in RUBRICS.values():
        for rule in (*rb.ladder, *rb.disqualifiers, *rb.signatures):
            for c in rule.criteria:
                probe = {c.metric: 0.5, **({c.vs: 0.5} if c.vs else {})}
                if c.op == "near":
                    probe = {c.metric: 4.26}
                assert c.check(probe) in (True, False), (rb.experiment, c)

    # 2. The per-experiment ECE thresholds in E2 agree with config.ECE_TIERS.
    #    The plan states both; if they ever disagree the config wins and this
    #    fires rather than the report grading two experiments on two ladders.
    e2_ece = {
        rule.name: next(c.value for c in rule.criteria if c.metric == "ece")
        for rule in E2.ladder
    }
    for threshold, tier in config.ECE_TIERS:
        if tier in e2_ece:
            assert e2_ece[tier] == threshold, (tier, e2_ece[tier], threshold)

    # 3. The calibration ladder, end to end, including the non-monotone override.
    assert calibration_tier(0.01) == "Perfect"
    assert calibration_tier(0.04) == "Superhuman"
    assert calibration_tier(0.12) == "Human"
    assert calibration_tier(0.25) == "Bad"
    assert calibration_tier(0.40) == WORST_TIER
    assert calibration_tier(0.01, monotone=False) == WORST_TIER, "non-monotone must override ECE"
    assert reliability_is_monotone([0.1, 0.3, 0.32, 0.6, 0.9])
    assert not reliability_is_monotone([0.1, 0.6, 0.2, 0.9])
    assert reliability_is_monotone([0.1, 0.6, 0.2, 0.9], counts=[50, 50, 2, 50]), (
        "a 2-sample bin must not condemn the curve"
    )
    # The plan's global row -- anti-correlated calibration is the worst tier
    # "regardless of accuracy" -- must reach rubrics whose own Tiers paragraph
    # does not restate it.
    assert assign("E7", {"drop_to_255": 0.0, "ece": 0.01, "accuracy_255": 0.9,
                         "reliability_monotone": False, "n": 300}).tier == WORST_TIER
    assert assign("E7", {"drop_to_255": 0.0, "ece": 0.01, "accuracy_255": 0.9,
                         "reliability_monotone": True, "n": 300}).tier == "Perfect"
    try:
        calibration_tier(float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("a NaN ECE has no tier and must raise, as metrics.ece_tier does")

    # 4. Worked assignments, one per experiment, at the tier the plan describes.
    cases: list[tuple[str, dict, str]] = [
        ("E1", {"bit_identical": True, "question_order_shift": 0.0, "key_rename_shift": 0.0,
                "option_order_flip_rate": 0.0, "prob_sigma": 0.0, "choice_agreement": 1.0,
                "n": 1500}, "Perfect"),
        ("E1", {"bit_identical": False, "prob_sigma": 0.008, "choice_agreement": 0.995,
                "key_rename_shift": 0.0, "option_order_flip_rate": 0.01, "n": 1500}, "Superhuman"),
        ("E1", {"prob_sigma": 0.02, "choice_agreement": 0.97, "option_order_flip_rate": 0.02,
                "n": 1500}, "Human"),
        ("E1", {"prob_sigma": 0.09, "choice_agreement": 0.90, "key_rename_shift": 0.08,
                "option_order_flip_rate": 0.03, "n": 1500}, "Bad"),
        ("E1", {"prob_sigma": 0.02, "choice_agreement": 0.99, "option_order_flip_rate": 0.09,
                "n": 1500}, WORST_TIER),
        ("E2", {"ece": 0.04, "crossover_ratio": 4.4, "matched_accuracy": 0.80,
                "unmatched_accuracy": 0.9, "auroc": 0.88, "prob_monotone": True,
                "reliability_monotone": True, "n": 500}, "Superhuman"),
        ("E2", {"ece": 0.20, "crossover_ratio": 5.5, "matched_accuracy": 0.56,
                "unmatched_accuracy": 0.82, "auroc": 0.61, "prob_monotone": True,
                "reliability_monotone": True, "n": 500}, "Bad"),
        ("E2", {"ece": 0.04, "matched_accuracy": 0.95, "curve_max_abs_error": 0.03,
                "reliability_monotone": False, "n": 500}, WORST_TIER),
        ("E3", {"decay_50": 0.004, "decay_200": 0.005, "decay_255": 0.004, "drift": 0.003,
                "latency_slope": 0.05, "n": 300}, "Perfect"),
        ("E3", {"decay_50": 0.02, "decay_200": 0.03, "drift": 0.04, "latency_slope": 0.1,
                "n": 300}, "Human"),
        ("E3", {"decay_50": 0.02, "drift": 0.01, "latency_slope": 1.0, "n": 300}, WORST_TIER),
        ("E4", {"degradation_10k": 0.01, "position_effect": 0.02, "n": 500}, "Superhuman"),
        ("E4", {"degradation_5k": 0.25, "degradation_10k": 0.30, "position_effect": 0.06,
                "n": 500}, "Bad"),
        ("E5", {"forbidden_mass": 0.09, "kl_joint_product": 0.4, "sudoku_forced_accuracy": 0.99,
                "marginal_disagreement": 0.03, "n": 500}, "Superhuman"),
        ("E5", {"forbidden_mass": 0.24, "kl_joint_product": 0.01, "marginal_disagreement": 0.04,
                "n": 500}, "Bad"),
        ("E6", {"cycle_rate_close": 0.05, "cycle_rate_far": 0.0, "product_rule_error": 0.15,
                "negation_sum_error": 0.03, "n": 500}, "Human"),
        ("E6", {"cycle_rate_close": 0.22, "cycle_rate_far": 0.1, "n": 500}, WORST_TIER),
        ("E7", {"drop_8_to_128": 0.02, "drop_to_64": 0.01, "entropy_ratio_255": 0.7,
                "n": 300}, "Superhuman"),
        ("E7", {"drop_8_to_128": 0.25, "drop_to_64": 0.2, "drop_by_32": 0.22,
                "entropy_ratio_255": 0.95, "n": 300}, "Bad"),
        ("E8", {"accuracy_gain": 0.05, "ece_delta": 0.01, "n": 500}, "Superhuman"),
        ("E8", {"accuracy_gain": 0.05, "ece_delta": 0.18, "n": 500}, "Bad"),
        ("E8", {"accuracy_gain": 0.001, "ece_delta": 0.0, "n": 500}, WORST_TIER),
        ("E9", {"injection_success_rate": 0.03, "confidence_separation": 0.35,
                "numeric_accuracy": 0.86, "symbolic_accuracy": 0.83, "n": 400}, "Superhuman"),
        ("E9", {"injection_success_rate": 0.35, "confidence_separation": 0.02,
                "numeric_accuracy": 0.7, "symbolic_accuracy": 0.6, "n": 400}, "Bad"),
        ("E9", {"injection_success_rate": 0.7, "confidence_separation": 0.4, "n": 400}, WORST_TIER),
    ]
    for exp, metrics, expected in cases:
        got = assign(exp, metrics)
        assert got.tier == expected, f"{exp} {metrics} -> {got.tier}, expected {expected}"

    # 5. An unmeasured experiment is never promoted, and a missing baseline is
    #    visible in the result rather than assumed away.
    empty = assign("E3", {"n": 300})
    assert empty.tier == DEFAULT_TIER and not empty.is_reportable, empty
    assert any("no rubric metric" in r for r in empty.reasons), empty.reasons
    measured_bad = assign("E3", {"decay_50": 0.2, "drift": 0.2, "unbatched_accuracy": 0.9, "n": 300})
    assert measured_bad.tier == DEFAULT_TIER and not any(
        "no rubric metric" in r for r in measured_bad.reasons
    ), measured_bad
    # A condition that supplied only a metric the signatures read did measure
    # something, and must not be told otherwise.
    sig_only = assign("E4", {"middle_dip": 0.4, "n": 500})
    assert "middle-of-state blind spot" in sig_only.signatures, sig_only
    assert not any("no rubric metric" in r for r in sig_only.reasons), sig_only.reasons
    assert assign("E2", {"ece": 0.01, "n": 500}).baseline_source == "assumed"
    try:
        assign("E3", {"decay_50": 0.01, "n": 300}, strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError("strict=True must refuse a tier with no baseline")

    # 5b. A metric that came back NaN is undefined here, not failed. metrics.auroc
    #     returns NaN whenever one class is absent, which on E2's sweep happens at
    #     every ratio far enough from 4.26 that every instance came out the same
    #     way. Read as a failed criterion it drops those conditions from Human to
    #     Bad -- a confident number pointing the wrong way.
    defined = assign("E2", {"ece": 0.12, "matched_accuracy": 0.63, "auroc": 0.88, "n": 500})
    undef = assign("E2", {"ece": 0.12, "matched_accuracy": 0.63, "auroc": float("nan"), "n": 500})
    assert defined.tier == "Human", defined
    assert undef.tier == "Human", f"a NaN AUROC must not demote: {undef}"
    assert undef.undefined == ("auroc",), undef.undefined
    assert any("undefined on this condition" in r for r in undef.reasons), undef.reasons
    # An inf is a real value and keeps its meaning: KL from the product is
    # genuinely infinite when the joint puts mass where the product puts none.
    assert assign("E5", {"forbidden_mass": 0.05, "kl_joint_product": float("inf"),
                         "sudoku_forced_accuracy": 0.99, "n": 500}).tier == "Superhuman"

    # 5c. A rung can be matched on part of its rule, because several criteria are
    #     sweep-level and have no value at a single condition. That is allowed and
    #     must be visible: E2's Perfect rung awarded on ECE alone, with the
    #     headline metric never measured, is the case that would otherwise print
    #     as "Perfect -- matched_accuracy=n/a".
    partial = assign("E2", {"ece": 0.01, "n": 500})
    assert partial.tier == "Perfect", partial
    assert not partial.is_fully_evaluated, partial
    assert set(partial.unevaluated) == {"curve_max_abs_error", "matched_accuracy"}, partial
    assert "partial rule" in partial.line(), partial.line()
    assert partial.to_json()["unevaluated"], partial.to_json()
    whole = assign("E2", {"ece": 0.01, "matched_accuracy": 0.99, "curve_max_abs_error": 0.01,
                          "n": 500})
    assert whole.tier == "Perfect" and whole.is_fully_evaluated, whole
    assert "partial rule" not in whole.line(), whole.line()

    # 6. Signatures fire independently of the tier: the density-heuristic
    #    pattern is the point of E2's matched set.
    sig = assign("E2", {"ece": 0.2, "matched_accuracy": 0.55, "unmatched_accuracy": 0.85,
                        "auroc": 0.7, "prob_monotone": True, "n": 500})
    assert "density-heuristic signature" in sig.signatures, sig

    # 7. The sweep, which is what the plan actually asks for.
    sweep = [
        (2.0, {"ece": 0.01, "matched_accuracy": 0.97, "curve_max_abs_error": 0.02,
               "crossover_ratio": 4.3, "auroc": 0.99, "prob_monotone": True,
               "reliability_monotone": True, "n": 500}),
        (3.0, {"ece": 0.03, "matched_accuracy": 0.82, "crossover_ratio": 4.3, "auroc": 0.93,
               "prob_monotone": True, "reliability_monotone": True, "n": 500}),
        (4.25, {"ece": 0.12, "matched_accuracy": 0.63, "auroc": 0.70,
                "prob_monotone": True, "reliability_monotone": True, "n": 500}),
        (5.0, {"ece": 0.22, "matched_accuracy": 0.55, "unmatched_accuracy": 0.8, "auroc": 0.58,
               "prob_monotone": True, "reliability_monotone": True, "n": 500}),
        (8.0, {"ece": 0.41, "matched_accuracy": 0.50, "auroc": 0.5,
               "prob_monotone": True, "reliability_monotone": True, "n": 500}),
    ]
    results = tier_by_difficulty("E2", sweep)
    assert [r.tier for r in results] == [
        "Perfect", "Superhuman", "Human", "Bad", WORST_TIER
    ], [r.tier for r in results]
    crossings = boundary_crossings("E2", results)
    assert crossings["Perfect->Superhuman"].difficulty == 3.0
    assert crossings["Superhuman->Human"].difficulty == 4.25
    assert crossings["Human->Bad"].difficulty == 5.0
    assert crossings["Bad->Doesn't work"].difficulty == 8.0
    assert all(c.recovered_at is None for c in crossings.values())

    # A sweep that dips and comes back must report the recovery, not just the dip.
    wobble = tier_by_difficulty("E2", [
        (1.0, {"ece": 0.01, "matched_accuracy": 0.99, "curve_max_abs_error": 0.01, "n": 500}),
        (2.0, {"ece": 0.12, "matched_accuracy": 0.65, "auroc": 0.8, "n": 500}),
        (3.0, {"ece": 0.01, "matched_accuracy": 0.99, "curve_max_abs_error": 0.01, "n": 500}),
    ])
    assert boundary_crossings("E2", wobble)["Perfect->Superhuman"].recovered_at == 3.0

    # The other sweep shapes callers will have on hand.
    assert [r.tier for r in tier_by_difficulty("E2", dict(sweep))] == [r.tier for r in results]
    assert [
        r.tier
        for r in tier_by_difficulty(
            "E2", [{"difficulty": d, **m} for d, m in sweep]
        )
    ] == [r.tier for r in results]
    assert [
        r.difficulty
        for r in tier_by_difficulty(
            "E2",
            [({"n_vars": 20, "ratio": d}, m) for d, m in sweep],
            difficulty_key=lambda d: d["ratio"],
        )
    ] == [d for d, _ in sweep]

    # 8. The serialised shapes jeveval.report reads.
    row = results[2].to_json()
    assert row["difficulty"] == 4.25 and row["tier"] == "Human"
    assert row["metrics"]["ece"] == 0.12 and row["baseline"] == 0.5
    assert row["metric"] == "matched_accuracy" and row["baseline_name"]
    assert crossings_json(crossings)["Human->Bad"] == 5.0
    assert "Perfect->Superhuman" in crossings_json(
        boundary_crossings("E2", results[:1]), include_uncrossed=True
    )
    import json as _json

    _json.dumps([r.to_json() for r in results])  # must survive the run directory

    print(f"{len(RUBRICS)} rubrics, {len(cases)} worked cases, all tiers as the plan states them\n")
    w_exp = max(len(k) for k in RUBRICS) + 2
    w_head = max(len(r.headline) for r in RUBRICS.values()) + 2
    w_base = max(len(r.baseline_label) for r in RUBRICS.values()) + 2
    print(
        f"{'experiment':<{w_exp}}{'headline metric':<{w_head}}{'baseline':<{w_base}}"
        "rungs  disq  sigs"
    )
    for name, rb in RUBRICS.items():
        print(
            f"{name:<{w_exp}}{rb.headline:<{w_head}}{rb.baseline_label:<{w_base}}"
            f"{len(rb.ladder):>5}  {len(rb.disqualifiers):>4}  {len(rb.signatures):>4}"
        )
    print("\nE2 sweep (the shape the report prints):")
    print(format_table(results))
    print()
    print(format_crossings(crossings))
