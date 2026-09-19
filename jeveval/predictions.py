"""The plan's prediction table, P1 through P28, as data.

Transcribed verbatim from the "Prediction summary" section so the report can
score predictions mechanically rather than by re-reading prose. Each entry keeps
the plan's own falsification condition, because a prediction is only worth
recording if it was falsifiable before the data arrived.

The plan notes that a high hit rate here would be mildly suspicious -- it would
suggest the experiments are too easy -- so the report states the hit rate plainly
either way, and flags wrong predictions prominently rather than burying them.

A prediction can also come out "untestable as specified". P25 is that by
construction: the plan declines to predict numeric and spatial performance and
asks for the result to be reported instead. Others may become untestable if the
API turns out not to expose what the prediction assumed -- noul answers carry no
confidence field, for instance, which changes how P26 has to be measured.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    id: str
    experiment: str
    claim: str
    falsified_if: str


TABLE: list[Prediction] = [
    Prediction("P1", "E1", "sigma <=0.02 on probabilities, >=0.98 choice agreement", "sigma >0.05"),
    Prediction("P2", "E1", "Key renaming has no effect", "Answers shift >2%"),
    Prediction("P3", "E1", "Option order has a small effect, 1-3%", "Effect >5% or exactly 0"),
    Prediction("P4", "E2", "Asymmetric failure: better at low ratio (SAT) than high (UNSAT)", "Symmetric, or better at high"),
    Prediction("P5", "E2", "Density-matched accuracy 0.55-0.65 -- mostly reading clause density", "Matched accuracy >0.75"),
    Prediction("P6", "E2", "Predicted-probability curve flatter than true curve at 4.26", "Sharp crossover within 0.5"),
    Prediction("P7", "E3", "No positional decay -- architecture is genuinely parallel", ">5% decay by position 200"),
    Prediction("P8", "E3", "Mild contamination, 2-5% drift; better on related questions, worse on unrelated", "Drift >10%, or zero drift"),
    Prediction("P9", "E3", "Latency roughly flat in question count", "Scales linearly"),
    Prediction("P10", "E4", "5-15% accuracy loss by 10k tokens; 3-8% middle-of-state dip", "Flat to 50k"),
    Prediction("P11", "E4", "CFG edge list > AST/JSON > raw source for reachability", "Source wins, or no difference"),
    Prediction("P12", "E5", "Enrollment helps: forbidden-cell mass 0.08-0.15 vs 0.25 product baseline", "Mass ~=0.25"),
    Prediction("P13", "E5", "KL(joint || product) in 0.2-0.5 -- meaningfully non-independent", "KL ~=0"),
    Prediction("P14", "E5", "Labeled option ids beat bit strings by >=15%", "Gap <5%"),
    Prediction("P15", "E5", "Sudoku near-perfect on forced moves, near-random by 5 options", "Strong at 5+ options"),
    Prediction("P16", "E6", "3-10% transitivity violations on close pairs", ">20% or ~=0%"),
    Prediction("P17", "E6", "Enrolled triples cut violations below 2%", "No improvement"),
    Prediction("P18", "E6", "Product rule violated by 0.1-0.2", "Within 0.05"),
    Prediction("P19", "E7", "Cardinality knee at 30-60 options", "Flat to 255"),
    Prediction("P20", "E7", "Two-stage beats flat at 255 by 10-20%", "Flat wins or ties"),
    Prediction("P21", "E7", "Near-zero recovery from a wrong turn in hierarchical descent", "Recovery >20%"),
    Prediction("P22", "E8", "ICL degrades calibration: ECE +0.03-0.10 with 10 examples", "ECE unchanged or improved"),
    Prediction("P23", "E8", "State-channel examples beat instructions-channel", "Instructions win"),
    Prediction("P24", "E8", "Inverted rubric followed ~70%, prior leaks through", ">95% or <30%"),
    Prediction("P25", "E9", "No confident prediction on numeric/spatial -- flagged as open", "n/a -- report the result"),
    Prediction("P26", "E9", "Poor abstention: confidence on nonsense states is moderate, not low", "Clean separation >=0.3"),
    Prediction("P27", "E9", "Injection success 10-35%", "<5% or >60%"),
    Prediction("P28", "E9", "Counting degrades sharply past ~20 elements", "Holds past 50"),
]

BY_ID = {p.id: p for p in TABLE}


def for_experiment(experiment: str) -> list[Prediction]:
    """Every prediction belonging to an experiment, e.g. "E2"."""
    key = experiment.upper()
    return [p for p in TABLE if p.experiment == key]


# The plan's two meta-predictions, called out separately because they are claims
# about which results will matter rather than about what the numbers will be.
META_PREDICTIONS = [
    (
        "M1",
        "The most likely genuine surprise is P7 being wrong in the interesting "
        "direction -- decay exists, revealing the parallelism is less complete "
        "than advertised.",
    ),
    (
        "M2",
        "The most consequential single result is P26: if confidence does not "
        "track ignorance, the 'act when confident, escalate when not' deployment "
        "pattern stops working regardless of how good the accuracy numbers look.",
    ),
]


def score(verdicts: list[dict]) -> dict:
    """Summarise scored predictions.

    `verdicts` is a list of {"id", "verdict", ...}. Untestable predictions are
    excluded from the hit rate rather than counted as misses, and reported
    separately, since counting P25 as a miss would misrepresent a prediction the
    plan deliberately declined to make.
    """
    seen = {v["id"]: v for v in verdicts}
    right = [i for i, v in seen.items() if v.get("verdict") == "right"]
    wrong = [i for i, v in seen.items() if v.get("verdict") == "wrong"]
    untestable = [i for i, v in seen.items() if v.get("verdict") == "untestable"]
    unscored = [p.id for p in TABLE if p.id not in seen]
    testable = len(right) + len(wrong)
    return {
        "right": sorted(right),
        "wrong": sorted(wrong),
        "untestable": sorted(untestable),
        "unscored": sorted(unscored),
        "hit_rate": (len(right) / testable) if testable else None,
        "n_testable": testable,
    }
