"""Experiment modules E1 through E9.

Every module exposes:

    run(run_dir: Path) -> dict
        Execute the experiment against the live API, write its raw calls to the
        run directory's JSONL log, compute its metrics offline from what it
        logged, emit its plots, and return a result dict in the shape below.

    format_report(result: dict) -> str        (optional)
        A short human-readable summary for the terminal. The markdown report is
        built from the result dict, not from this.

The result dict follows the plan's "Report format" section, which fixes both the
contents and the order in which an experiment must report:

    {
      "experiment": "E1",
      "question":   one sentence, the thing this experiment asks,

      "headline":   {"metric": str, "value": float, "baseline": float,
                     "baseline_name": str, "n": int},
          The headline number with its baseline beside it. The plan is explicit
          that a number without its baseline is not a result, so the baseline is
          part of the structure rather than something prose is trusted to add.

      "by_difficulty": [
          {"difficulty": {...}, "n": int, "metrics": {...},
           "tier": str, "baseline": float}, ...
      ],
          Tier as a function of difficulty. The plan: "A single overall grade
          throws away the finding."

      "boundary_crossings": {"Superhuman->Human": <difficulty>, ...},
          The difficulty at which each tier boundary is crossed, named.

      "plots":      [relative paths to PNGs under the run directory],

      "predictions": [
          {"id": "P1", "claim": str, "outcome": str,
           "verdict": "right" | "wrong" | "untestable", "evidence": {...}}, ...
      ],
          Scored against jeveval.predictions. Wrong predictions are the most
          valuable output and the report flags them prominently, so a module
          must not quietly omit one it failed.

      "anomalies":  [str, ...],
          Anything surprising, including things no experiment was designed to
          test. These feed the report's required "unexpected behaviors" section.

      "failures":   {"calls": int, "excluded": int, "reasons": {...}},
          Failed calls are reported as a count and excluded, never defaulted.

      "gate":       {"passed": bool, "reason": str}      (only where the plan
                    defines a gate: E1 and E2 in Phase 1, E3 in Phase 2)
    }

Two rules that apply to every module:

Ground truth goes into `Call.meta` on the way out, so the JSONL log is
self-sufficient and every metric can be recomputed offline without re-spending
calls.

Randomise question order within every batch and option order within every
choice, via `instances.shuffled_questions` and `instances.shuffled_options`,
unless the experiment is specifically measuring order effects. Position bias
would otherwise confound every choice-based result in the plan.
"""

from __future__ import annotations

ALL = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"]
