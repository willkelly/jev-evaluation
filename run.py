#!/usr/bin/env python
"""Command-line entry point for the jev evaluation.

The plan's run order is not a suggestion -- early results invalidate later
experiments, and two of the phase boundaries are hard gates:

    Phase 0  smoke      auth, three question types, semantic control above chance
    Phase 1  E1, E2     determinism and calibration
      gate:  determinism sigma > 0.15 means every downstream comparison is
             inside the noise band; control ECE > 0.30 means the core value
             proposition does not hold on this data
    Phase 2  E3, E7     batching and cardinality -- the use case
      gate:  positional decay over 10% by question 50 caps batch size, and E5
             and E6 must then be re-run within that cap
    Phase 3  E5, E6     enrollment and cross-call coherence
    Phase 4  E4, E8, E9 input size, in-context learning, distribution edges
             (E9's injection tests run last)

So `run.py phase1` stops if its gate trips, and `run.py all` runs the phases in
order and stops at the first gate that trips. Running a single experiment by
name bypasses the gates deliberately, for iteration.

Usage:
    run.py smoke                 Phase 0 only
    run.py phase1 | phase2 | ... run a phase, honouring its gate
    run.py e1 | e2 | ... | e9    one experiment, no gating
    run.py all                   every phase in order, stopping at a tripped gate
    run.py report                rebuild the report from an existing run's log
    run.py status                what the current run directory contains

Options:
    --run-id ID     name the run directory (default: a timestamp)
    --scale F       multiply every sample size by F, for dry runs. Recorded in
                    the report, because a silently shrunk sample size is how a
                    noisy result gets published as a finding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from jeveval import config


PHASES: dict[str, list[str]] = {
    "phase1": ["e1", "e2"],
    "phase2": ["e3", "e7"],
    "phase3": ["e5", "e6"],
    "phase4": ["e4", "e8", "e9"],
}

EXPERIMENT_MODULES = {
    "e1": "e1_determinism",
    "e2": "e2_calibration",
    "e3": "e3_batching",
    "e4": "e4_input_size",
    "e5": "e5_enrollment",
    "e6": "e6_coherence",
    "e7": "e7_cardinality",
    "e8": "e8_icl",
    "e9": "e9_edges",
}


def default_run_id() -> str:
    return time.strftime("run-%Y%m%d-%H%M%S")


def load_experiment(name: str):
    import importlib

    mod = EXPERIMENT_MODULES[name]
    return importlib.import_module(f"jeveval.experiments.{mod}")


def cmd_smoke(run_dir: Path) -> int:
    from jeveval import smoke

    report = smoke.run(run_dir)
    print(smoke.format_report(report))
    return 0 if report["passed"] else 1


def cmd_experiment(name: str, run_dir: Path) -> int:
    mod = load_experiment(name)
    result = mod.run(run_dir)
    out = run_dir / f"{name}_result.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n{name.upper()} written to {out}")
    if hasattr(mod, "format_report"):
        print(mod.format_report(result))
    gate = result.get("gate")
    if gate and not gate.get("passed", True):
        print(f"\nGATE TRIPPED after {name.upper()}: {gate.get('reason')}")
        return 2
    return 0


def cmd_phase(phase: str, run_dir: Path) -> int:
    for name in PHASES[phase]:
        print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
        rc = cmd_experiment(name, run_dir)
        if rc != 0:
            return rc
    return 0


def cmd_all(run_dir: Path) -> int:
    rc = cmd_smoke(run_dir)
    if rc != 0:
        print("\nPhase 0 failed. The plan says to fix the harness before believing "
              "anything else, so the run stops here.")
        return rc
    for phase in ("phase1", "phase2", "phase3", "phase4"):
        print(f"\n{'#' * 70}\n{phase.upper()}\n{'#' * 70}")
        rc = cmd_phase(phase, run_dir)
        if rc != 0:
            return rc
    return cmd_report(run_dir)


def cmd_report(run_dir: Path) -> int:
    from jeveval import report

    path = report.build(run_dir)
    print(f"report written to {path}")
    return 0


def cmd_status(run_dir: Path) -> int:
    from jeveval import logstore

    if not run_dir.exists():
        print(f"{run_dir} does not exist")
        return 1
    print(f"run directory: {run_dir}")
    for log in sorted(run_dir.glob("*.jsonl")):
        records, stats = logstore.read_with_stats(log)
        rs = logstore.run_stats(records)
        print(f"\n  {log.name}: {stats['records']} records, outcomes={stats['outcomes']}")
        print(f"    experiments: {', '.join(x for x in stats['experiments'] if x)}")
        print(
            f"    calls={rs['calls']} ok={rs['successful']} failed={rs['failed']} "
            f"retries={rs['retries']}"
        )
        print(
            f"    model={','.join(rs['model_versions'])} cost=${rs['cost_usd']:.6f} "
            f"p50={rs['latency_p50']} p95={rs['latency_p95']}"
        )
    for j in sorted(run_dir.glob("*_result.json")):
        print(f"  {j.name}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run the jev evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--scale", type=float, default=None)
    args = parser.parse_args(argv)

    if args.scale is not None:
        # config reads this at import time, so set it before anything that
        # depends on a sample size is imported.
        os.environ["JEV_SCALE"] = str(args.scale)
        config.SCALE = args.scale

    run_id = args.run_id or default_run_id()
    run_dir = config.run_dir(run_id)
    # Written once, by the command that spends the calls. `report` and `status`
    # read a run rather than producing one, so they must not restamp it: doing
    # so replaces the real start time and, worse, the real `scale`, which is
    # what makes the report print its "SAMPLE SIZES SCALED BY" warning. A run
    # made at --scale 0.02 and reported later without the flag would silently
    # lose that warning and read as a full run.
    meta_path = run_dir / "run_meta.json"
    if args.command.lower() not in ("report", "status") or not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "started": time.time(),
                    "command": args.command,
                    "scale": config.SCALE,
                    "master_seed": config.MASTER_SEED,
                    "model_alias": config.MODEL_ALIAS,
                    "api_url": config.API_URL,
                },
                indent=2,
            )
        )
    if config.SCALE != 1.0:
        print(
            f"NOTE: sample sizes scaled by {config.SCALE}. This is recorded in the "
            "run metadata and must appear in the report."
        )
    print(f"run directory: {run_dir}\n")

    cmd = args.command.lower()
    if cmd == "smoke":
        return cmd_smoke(run_dir)
    if cmd in PHASES:
        return cmd_phase(cmd, run_dir)
    if cmd in EXPERIMENT_MODULES:
        return cmd_experiment(cmd, run_dir)
    if cmd == "all":
        return cmd_all(run_dir)
    if cmd == "report":
        return cmd_report(run_dir)
    if cmd == "status":
        return cmd_status(run_dir)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
