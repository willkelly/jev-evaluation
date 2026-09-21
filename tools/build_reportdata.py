#!/usr/bin/env python
"""Extract a run's results into the payload the report template renders.

`tools/reportdata.json` is what `build_report.py` embeds in the page, and until
now it had no producer: it was assembled by hand in a scratch directory that
did not survive, which made the most visible artefact of the evaluation the one
thing a checkout could not rebuild. This is that producer.

It reads a run directory -- the nine `e*_result.json` files, the per-experiment
`run` blocks inside them, and the logs for the latency percentiles the result
files do not carry -- and writes the same shape. The predictions, their verdicts
and the two meta-predictions come from `jeveval.predictions`, which is the
plan's own copy of them, so a scoreboard cannot drift from the plan it scores.

Nothing here is authored. The experiment's name, its question, the rule that
set each grade and the caveats attached to it all come out of the result file;
this only renames the fields to what the template reads.

    python tools/build_reportdata.py --run full-20260919
    python tools/build_reportdata.py --run von-20260921 --runs-dir ../../von-runtime/runs \\
                                     --out tools/reportdata-von.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jeveval import predictions  # noqa: E402

# The plan's run order, which is the order the report presents them in: an
# early result can invalidate a later experiment, so E7 runs with E3 and E4
# runs in the last phase. Sorting the files by name would reorder the report.
RUN_ORDER = ["E1", "E2", "E3", "E7", "E5", "E6", "E4", "E8", "E9"]

def scan_log(log: Path) -> tuple[list[float], float | None, float | None]:
    """Every call's latency, and the first and last timestamps.

    Latencies are pooled across experiments rather than averaged per
    experiment, because a median of medians is not a median. Only calls that
    succeeded count: a call that timed out and was retried contributes its
    timeout to the record, and reporting that as service latency would
    describe the harness rather than the service.

    The timestamps give the run's wall clock, which is not the sum of the
    experiments' elapsed times: phases overlap, and the sum understates a
    138-minute run as 28 minutes.
    """
    lat: list[float] = []
    first = last = None
    if not log.exists():
        return lat, first, last
    with log.open() as fh:
        for line in fh:
            i = line.find('"latency_s":')
            if i >= 0 and '"outcome": "ok"' in line:
                try:
                    lat.append(float(line[i + 12 : line.index(",", i + 12)]))
                except (ValueError, IndexError):
                    pass
            j = line.find('"ts":')
            if j >= 0:
                try:
                    t = float(line[j + 5 : line.index(",", j + 5)])
                except (ValueError, IndexError):
                    continue
                first = t if first is None else min(first, t)
                last = t if last is None else max(last, t)
    return lat, first, last


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def row_of(block: dict) -> dict:
    """One difficulty's row, in the shape the template's tables expect."""
    return {
        "d": block.get("difficulty"),
        "n": (block.get("metrics") or {}).get("n", block.get("n")),
        "metric": block.get("metric"),
        "value": block.get("value"),
        "baseline": block.get("baseline"),
        "bname": block.get("baseline_name"),
        "bsrc": block.get("baseline_source"),
        "tier": block.get("tier"),
        # The grade's rule, then every caveat the grader attached to it: a
        # metric that came back undefined, or a named failure signature. The
        # rule alone reads as a cleaner result than the row actually was.
        "why": "; ".join(
            list(block.get("reasons") or ([block["matched_rule"]]
                                          if block.get("matched_rule") else []))
            + list(block.get("signatures") or [])
        ) or None,
        "mn": block.get("metric_n"),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="full-20260919")
    ap.add_argument("--runs-dir", default=None, help="defaults to this repo's runs/")
    ap.add_argument("--out", default=None, help="defaults to tools/reportdata.json")
    args = ap.parse_args(argv)

    base = Path(args.runs_dir).resolve() if args.runs_dir else ROOT / "runs"
    run_dir = base / args.run
    results = sorted(run_dir.glob("e*_result.json"))
    if not results:
        print(f"no e*_result.json in {run_dir}", file=sys.stderr)
        return 2

    agg = {"calls": 0, "failed": 0, "retries": 0, "throttles": 0, "cost": 0.0,
           "itok": 0, "otok": 0, "seconds": 0.0}
    models: set[str] = set()
    rps: dict[str, float] = {}
    tps: dict[str, float] = {}
    mtok: dict[str, float] = {}
    thr: dict[str, int] = {}
    lat_all: list[float] = []
    wall: list[float | None] = [None, None]
    experiments = []

    for path in results:
        doc = json.loads(path.read_text())
        eid = doc.get("experiment") or path.stem.split("_")[0].upper()
        run = doc.get("run") or {}
        agg["calls"] += run.get("total_calls", 0)
        agg["failed"] += run.get("failed_calls", 0)
        agg["retries"] += run.get("total_retries", 0)
        agg["throttles"] += run.get("throttle_events", 0)
        agg["cost"] += run.get("cost_usd", 0.0) or 0.0
        agg["itok"] += run.get("input_tokens", 0)
        agg["otok"] += run.get("output_tokens", 0)
        agg["seconds"] += run.get("elapsed_s", 0.0) or 0.0
        models.update(run.get("model_versions") or [])
        if run.get("sustained_req_per_s"):
            rps[eid] = run["sustained_req_per_s"]
        secs = run.get("elapsed_s") or 0.0
        if secs and run.get("input_tokens"):
            tps[eid] = run["input_tokens"] / secs
        if run.get("total_calls"):
            mtok[eid] = run.get("input_tokens", 0) / run["total_calls"]
        if run.get("throttle_events"):
            thr[eid] = run["throttle_events"]

        lat, first, last = scan_log(run_dir / f"{eid.lower()}.jsonl")
        lat_all.extend(lat)
        if first is not None:
            wall[0] = first if wall[0] is None else min(wall[0], first)
            wall[1] = last if wall[1] is None else max(wall[1], last)

        experiments.append({
            "id": eid,
            "title": doc.get("title", ""),
            "question": doc.get("question", ""),
            "headline": doc.get("headline"),
            "gate": doc.get("gate"),
            "rows": [row_of(b) for b in doc.get("by_difficulty") or []],
            "crossings": doc.get("boundary_crossings"),
            "crossings_axis": doc.get("boundary_crossings_axis"),
            "predictions": [
                {"id": p.get("id"), "claim": p.get("claim"),
                 "outcome": p.get("outcome"), "verdict": p.get("verdict")}
                for p in doc.get("predictions") or []
            ],
            "anomalies": doc.get("anomalies") or [],
            "notes": doc.get("notes") or [],
            # The template resolves plots against the run directory itself.
            "plots": [Path(x).name for x in (doc.get("plots") or [])],
            "what": doc.get("what_this_changes", doc.get("what", "")),
        })

    experiments.sort(key=lambda e: (RUN_ORDER.index(e["id"])
                                    if e["id"] in RUN_ORDER else len(RUN_ORDER)))

    # The scoreboard is the plan's, keyed by the verdicts this run recorded.
    verdicts = {p["id"]: p["verdict"]
                for e in experiments for p in e["predictions"] if p.get("id")}
    right = sorted(k for k, v in verdicts.items() if v == "right")
    wrong = sorted(k for k, v in verdicts.items() if v == "wrong")
    untestable = sorted(k for k, v in verdicts.items() if v not in ("right", "wrong"))
    scored = {p.id for p in predictions.TABLE} & set(verdicts)
    testable = len(right) + len(wrong)
    score = {
        "right": right, "wrong": wrong, "untestable": untestable,
        # A prediction the run never reached is not a prediction that held.
        "unscored": sorted({p.id for p in predictions.TABLE} - scored),
        "hit_rate": round(len(right) / testable, 2) if testable else None,
        "n_testable": testable,
    }

    meta = json.loads((run_dir / "run_meta.json").read_text()) \
        if (run_dir / "run_meta.json").exists() else {}

    payload = {
        "run": {
            "model": ", ".join(sorted(models)) or meta.get("model_alias", ""),
            "calls": agg["calls"],
            "ok": agg["calls"] - agg["failed"],
            "failed": agg["failed"],
            "retries": agg["retries"],
            "throttles": agg["throttles"],
            "cost": agg["cost"],
            "itok": agg["itok"],
            "otok": agg["otok"],
            "minutes": ((wall[1] - wall[0]) / 60.0
                        if wall[0] is not None and wall[1] is not None
                        else agg["seconds"] / 60.0),
            "p50": pct(lat_all, 0.50),
            "p95": pct(lat_all, 0.95),
            "rps": rps, "tps": tps, "mtok": mtok, "thr": thr,
            "scale": meta.get("scale", 1.0),
        },
        "experiments": experiments,
        "score": score,
        "meta_predictions": list(predictions.META_PREDICTIONS),
        "table": [
            {"id": t.id, "exp": t.experiment, "claim": t.claim, "falsified": t.falsified_if}
            for t in predictions.TABLE
        ],
    }

    out = Path(args.out) if args.out else ROOT / "tools" / "reportdata.json"
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    r = payload["run"]
    print(f"{len(experiments)} experiments, {r['calls']:,} calls, {r['failed']} failed, "
          f"{r['minutes']:.0f} min, model {r['model'] or '(none)'}")
    print(f"predictions: {len(score['right'])} right, {len(score['wrong'])} wrong, "
          f"{len(score['untestable'])} untestable")
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} "
          f"({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
