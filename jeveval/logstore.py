"""Offline reading of the raw JSONL call log.

The plan requires that every metric be recomputable from the log alone, without
re-spending calls: "All analysis runs offline against that log, so metrics can be
recomputed without re-spending calls." That is only true if the log is
self-sufficient, so two conventions hold everywhere in this harness.

First, ground truth travels in `Call.meta` and is therefore written into every
record. An experiment that keeps truth only in its own memory produces a log that
cannot be rescored after the process exits.

Second, answers are re-parsed here from the logged wire response rather than
carried over from the live run. Reading the log is the same code path whether the
call happened a second ago or last week, so a metric redefined tomorrow sees
exactly what the model actually returned.

Retries appear as multiple records for the same (instance_id, repetition). Only
the terminal record of each attempt chain carries an outcome of "ok", "error" or
"parse_error"; records with outcome "retry" are the retry history and are counted
separately rather than being mistaken for data points.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


def read(path: str | Path) -> Iterator[dict]:
    """Yield every record in a JSONL log, skipping unparseable lines.

    A truncated final line is expected when a run is killed mid-write, so it is
    skipped rather than raising -- but the count of skipped lines is worth
    reporting, which `read_with_stats` does.
    """
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_with_stats(path: str | Path) -> tuple[list[dict], dict]:
    """All records plus a summary of what the log contains."""
    p = Path(path)
    records, bad = [], 0
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    outcomes: dict[str, int] = defaultdict(int)
    for r in records:
        outcomes[r.get("outcome", "unknown")] += 1
    return records, {
        "records": len(records),
        "unparseable_lines": bad,
        "outcomes": dict(outcomes),
        "experiments": sorted({r.get("experiment", "") for r in records}),
    }


def terminal(records: Iterable[dict]) -> Iterator[dict]:
    """Records that are a call's final outcome, excluding retry history."""
    for r in records:
        if r.get("outcome") in ("ok", "error", "parse_error"):
            yield r


def successful(records: Iterable[dict]) -> Iterator[dict]:
    for r in records:
        if r.get("outcome") == "ok":
            yield r


def select(
    records: Iterable[dict],
    *,
    experiment: str | None = None,
    condition: str | None = None,
) -> list[dict]:
    out = []
    for r in records:
        if experiment is not None and r.get("experiment") != experiment:
            continue
        if condition is not None and r.get("condition") != condition:
            continue
        out.append(r)
    return out


def group_by(records: Iterable[dict], key: str) -> dict[Any, list[dict]]:
    """Group records by a top-level field or a dotted path into `meta`."""
    out: dict[Any, list[dict]] = defaultdict(list)
    for r in records:
        out[_dig(r, key)].append(r)
    return dict(out)


def _dig(record: dict, key: str) -> Any:
    if "." not in key:
        return record.get(key)
    cur: Any = record
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


# --------------------------------------------------------------------------
# Re-parsing answers from the logged wire response
# --------------------------------------------------------------------------


def answers_of(record: dict) -> dict[str, dict]:
    """Neutral answers for one successful record, parsed from the log.

    This mirrors `wire.parse_answer`, but works from the logged wire request
    rather than from a live neutral question spec -- the log holds the wire form,
    which is what the model actually saw. For a score question the wire criteria
    list supplies the index-to-label mapping, so rubric labels come back even
    though the caller's rubric ids are not in the log.
    """
    resp = record.get("response") or {}
    req = record.get("request") or {}
    wire_questions = (req.get("questions") or {})
    answers = resp.get("answers") or {}
    out: dict[str, dict] = {}

    for key, ans in answers.items():
        wq = wire_questions.get(key, {})
        atype = ans.get("type")
        if atype == "noul":
            p = float(ans.get("noul", float("nan")))
            out[key] = {"type": "noul", "p": p, "predicted": p >= 0.5, "confidence": None}
        elif atype == "choice":
            probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
            out[key] = {
                "type": "choice",
                "chosen": str(ans.get("choice")),
                "confidence": _f(ans.get("confidence")),
                "probabilities": probs,
                "max_probability": max(probs.values()) if probs else None,
                # Wire option order, which is what a position-bias analysis needs.
                "option_order": list((wq.get("criteria") or {}).keys()),
            }
        elif atype == "score":
            idx = {int(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
            labels = wq.get("criteria") or []
            # Ground truth for a score question is the caller's rubric id, but
            # the wire carries only labels. The client records the id list per
            # score question for exactly this reason; when it is present, key
            # the result by id so an offline rescore matches a live one. Without
            # it, fall back to labels and say so, rather than returning labels
            # under a name that implies they are ids -- that silently rescores
            # every score question as wrong.
            rubric_ids = (record.get("rubric_ids") or {}).get(key) or []
            keyed_by_id = bool(rubric_ids) and len(rubric_ids) >= len(labels)

            def name_at(i: int) -> str | None:
                if keyed_by_id and i < len(rubric_ids):
                    return str(rubric_ids[i])
                if i < len(labels):
                    lab = labels[i]
                    return str(lab.get("label") if isinstance(lab, dict) else lab)
                return None

            by_name: dict[str, float] = {}
            for i in range(len(labels)):
                nm = name_at(i)
                if nm is not None and i in idx:
                    by_name[nm] = idx[i]
            chosen = None
            if idx:
                best = max(idx, key=lambda i: idx[i])
                chosen = name_at(best)
            out[key] = {
                "type": "score",
                "score": _f(ans.get("score")),
                "chosen": chosen,
                "chosen_is_rubric_id": keyed_by_id,
                "confidence": _f(ans.get("confidence")),
                "probabilities": by_name,
                "index_probabilities": idx,
                "legend": ans.get("legend"),
                "max_probability": max(idx.values()) if idx else None,
            }
    return out


def _f(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


# --------------------------------------------------------------------------
# Run-level facts the report header needs
# --------------------------------------------------------------------------


# The only fields any run-level statistic reads. Projecting onto these while
# streaming, rather than materialising every record, is the difference between
# a few megabytes and several gigabytes: a parsed record costs roughly 3.5x its
# JSONL text, and E3 alone logs around 450 MB because it sends up to 255
# questions per call.
_STAT_FIELDS = (
    "ts",
    "outcome",
    "latency_s",
    "input_tokens",
    "output_tokens",
    "model_version",
    "http_status",
    "transport_error",
    "experiment",
)


def scan_stats(paths: Iterable[str | Path]) -> dict:
    """Run-level statistics, streamed.

    Same numbers as `run_stats`, without ever holding more than one record.
    Also returns the achieved request rate per experiment, because the pooled
    figure spans the idle time between experiments -- and E6's temporal
    condition deliberately writes into the same directory hours later, which
    would drag a 130 req/s run toward zero. The plan treats the sustained rate
    as a finding in its own right, so it has to be the rate something actually
    sustained.
    """
    latencies: list[float] = []
    itok = otok = 0
    versions: set[str] = set()
    calls = ok = retries = throttles = attempts = 0
    lo_ts: float | None = None
    hi_ts: float | None = None
    per_exp: dict[str, dict] = {}

    for p in paths:
        try:
            fh = Path(p).open(encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                attempts += 1
                ts = r.get("ts")
                if isinstance(ts, (int, float)):
                    lo_ts = ts if lo_ts is None else min(lo_ts, ts)
                    hi_ts = ts if hi_ts is None else max(hi_ts, ts)
                outcome = r.get("outcome")
                if r.get("http_status") in (429, 503) or r.get("transport_error"):
                    throttles += 1
                if outcome == "retry":
                    retries += 1
                    continue
                if outcome not in ("ok", "error", "parse_error"):
                    continue
                calls += 1
                exp = r.get("experiment") or ""
                slot = per_exp.setdefault(exp, {"calls": 0, "lo": None, "hi": None})
                slot["calls"] += 1
                if isinstance(ts, (int, float)):
                    slot["lo"] = ts if slot["lo"] is None else min(slot["lo"], ts)
                    slot["hi"] = ts if slot["hi"] is None else max(slot["hi"], ts)
                if outcome != "ok":
                    continue
                ok += 1
                lat = r.get("latency_s")
                if isinstance(lat, (int, float)):
                    latencies.append(float(lat))
                itok += int(r.get("input_tokens") or 0)
                otok += int(r.get("output_tokens") or 0)
                v = r.get("model_version")
                if v:
                    versions.add(str(v))

    latencies.sort()
    span = (hi_ts - lo_ts) if (lo_ts is not None and hi_ts is not None) else 0.0
    rates = {
        e: (s["calls"] / (s["hi"] - s["lo"]))
        for e, s in per_exp.items()
        if s["lo"] is not None and s["hi"] is not None and s["hi"] > s["lo"] and s["calls"] > 1
    }
    return {
        "attempts_logged": attempts,
        "calls": calls,
        "successful": ok,
        "failed": calls - ok,
        "retries": retries,
        "throttle_or_transport_events": throttles,
        "input_tokens": itok,
        "output_tokens": otok,
        "cost_usd": itok * (42.0 / 1e9),
        "model_versions": sorted(versions),
        "wall_clock_s": span,
        "sustained_req_per_s": (calls / span) if span > 0 else None,
        "req_per_s_by_experiment": rates,
        "latency_p50": latencies[len(latencies) // 2] if latencies else None,
        "latency_p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
    }


def run_stats(records: Iterable[dict]) -> dict:
    records = list(records)
    term = list(terminal(records))
    ok = [r for r in term if r.get("outcome") == "ok"]
    lat = sorted(r.get("latency_s", 0.0) for r in ok)
    itok = sum(int(r.get("input_tokens") or 0) for r in ok)
    otok = sum(int(r.get("output_tokens") or 0) for r in ok)
    versions = sorted({r.get("model_version") for r in ok if r.get("model_version")})
    ts = [r.get("ts", 0.0) for r in records if r.get("ts")]
    span = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
    retries = sum(1 for r in records if r.get("outcome") == "retry")
    throttles = sum(
        1 for r in records if r.get("http_status") in (429, 503) or r.get("transport_error")
    )
    return {
        "attempts_logged": len(records),
        "calls": len(term),
        "successful": len(ok),
        "failed": len(term) - len(ok),
        "retries": retries,
        "throttle_or_transport_events": throttles,
        "input_tokens": itok,
        "output_tokens": otok,
        "cost_usd": itok * (42.0 / 1e9),
        "model_versions": versions,
        "wall_clock_s": span,
        "sustained_req_per_s": (len(term) / span) if span > 0 else None,
        "latency_p50": lat[len(lat) // 2] if lat else None,
        "latency_p95": lat[int(len(lat) * 0.95)] if lat else None,
    }
