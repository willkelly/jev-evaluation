"""The markdown report.

Structure and ordering come from the plan's "Report format" section, which is
prescriptive, so this module follows it literally rather than inventing a layout:

  * a header carrying the versioned model string, date, total calls, total spend,
    sustained request rate and any rate limiting encountered;
  * per experiment, in the plan's order: the question in one sentence, the
    headline number with its baseline beside it, the tier as a function of
    difficulty naming where each boundary is crossed, the plot, prediction versus
    outcome, and anything anomalous;
  * a required "unexpected behaviors" section, separate from the per-experiment
    results;
  * a required "what this changes" section;
  * the prediction scoreboard.

The report is rebuilt from the run directory alone. Note the limit on that: it
re-renders the numbers stored in `*_result.json` and streams the JSONL only for
the run-level header. Recomputing a redefined metric from the log needs a
`score(run_dir)` in the experiment, which only E2 currently has.

Two things this module refuses to do quietly. It will not print a headline number
without a baseline next to it, and it will not print a single overall grade in
place of a tier-by-difficulty table; both are failures the plan calls out by
name. Where an experiment did not supply them, the report says so in the text
rather than omitting the row.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import config, logstore, predictions

ORDER = ["E0", "E1", "E2", "E3", "E7", "E5", "E6", "E4", "E8", "E9"]

# The behaviours the plan says to watch for and report even though no experiment
# specifically tests them.
WATCH_LIST = [
    "Probabilities clustering at particular values (0.5, 0.9) rather than "
    "spreading smoothly -- a sign of quantization or a trained-in prior",
    "`confidence` diverging from max-probability in a patterned way",
    "Answers changing with irrelevant formatting (whitespace, key order, casing)",
    "Latency correlating with anything other than input size",
    "Any response shape not documented -- extra fields, unexpected error codes",
    "Systematic bias toward particular option positions or particular ids",
    "Cases where batched answers were better than unbatched, not just different",
    "Silent behavior change across the run",
]


def _load_results(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(run_dir.glob("*_result.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        key = str(data.get("experiment") or path.stem.split("_")[0]).upper()
        out[key] = data
    smoke = run_dir / "smoke_report.json"
    if smoke.exists():
        out["E0"] = json.loads(smoke.read_text())
    return out


def _log_stats(run_dir: Path) -> dict:
    """Run-level statistics, streamed from every log in the directory.

    Materialising the records instead costs roughly 3.5x the JSONL size in RAM
    -- several gigabytes at full scale, since E3 alone logs around 450 MB --
    to compute eight scalars.
    """
    return logstore.scan_stats(sorted(run_dir.glob("*.jsonl")))


def _per_experiment_rates(stats: dict) -> str:
    """Achieved request rate per experiment, since the pooled figure spans idle
    time between experiments and is not the rate anything actually sustained."""
    rates = stats.get("req_per_s_by_experiment") or {}
    parts = [f"{e} {r:.0f}/s" for e, r in sorted(rates.items()) if e]
    return (" — per experiment: " + ", ".join(parts)) if parts else ""


def _header(run_dir: Path, st: dict, meta: dict) -> list[str]:
    versions = ", ".join(st["model_versions"]) or "unknown"
    rate = st["sustained_req_per_s"]
    lines = [
        "# Jev evaluation report",
        "",
        f"- **Model**: `{versions}` (requested alias `{meta.get('model_alias', config.MODEL_ALIAS)}`)",
        f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Run id**: `{run_dir.name}`",
        f"- **Endpoint**: `{meta.get('api_url', config.API_URL)}`",
        f"- **Master seed**: `{meta.get('master_seed', config.MASTER_SEED)}`",
        f"- **Total calls**: {st['calls']:,} ({st['successful']:,} succeeded, "
        f"{st['failed']:,} failed, {st['retries']:,} retries logged)",
        f"- **Total spend**: ${st['cost_usd']:.4f} "
        f"({st['input_tokens']:,} input tokens, {st['output_tokens']:,} output tokens)",
        # Pooled across the whole directory this number is wrong whenever the
        # run spans idle time -- E6's temporal condition deliberately writes
        # into the same directory hours later, which would drag a 130 req/s run
        # toward zero. The plan treats the achieved rate as a finding in its own
        # right, so the per-experiment figures are what to read.
        f"- **Sustained request rate**: "
        + (f"{rate:.1f} req/s pooled" if rate else "not measurable")
        + f" over {st['wall_clock_s'] / 60:.1f} minutes"
        + _per_experiment_rates(st),
        f"- **Latency**: p50 "
        + (f"{st['latency_p50'] * 1000:.0f} ms" if st["latency_p50"] else "n/a")
        + ", p95 "
        + (f"{st['latency_p95'] * 1000:.0f} ms" if st["latency_p95"] else "n/a"),
        f"- **Throttling / transport events**: {st['throttle_or_transport_events']:,}",
    ]
    scale = meta.get("scale", 1.0)
    if scale and float(scale) != 1.0:
        lines.append(
            f"- **SAMPLE SIZES SCALED BY {scale}** -- every n below is reduced "
            "by this factor relative to the plan. Numbers are not comparable to "
            "a full run."
        )
    if len(st["model_versions"]) > 1:
        lines.append(
            f"- **WARNING**: more than one model version answered during this run "
            f"({versions}). Results are only meaningful against a pinned version; "
            "treat cross-experiment comparisons as suspect."
        )
    lines.append("")
    return lines


def _experiment_section(key: str, res: dict) -> list[str]:
    lines = [f"## {key} — {res.get('title', res.get('question', ''))[:120]}", ""]

    question = res.get("question")
    if question:
        lines += [f"**Question.** {question}", ""]

    # 2. Headline number, with its baseline beside it.
    h = res.get("headline")
    if h:
        base = h.get("baseline")
        base_txt = (
            f"{base:.4g} ({h.get('baseline_name', 'baseline')})"
            if isinstance(base, (int, float))
            else "**no baseline reported — per the plan this is not a result**"
        )
        val = h.get("value")
        val_txt = f"{val:.4g}" if isinstance(val, (int, float)) else str(val)
        lines += [
            f"**Headline.** {h.get('metric', 'metric')} = **{val_txt}** "
            f"against {base_txt}, n = {h.get('n', '?')}.",
            "",
        ]
    else:
        lines += ["**Headline.** Not reported by this experiment.", ""]

    # 3. Tier as a function of difficulty.
    rows = res.get("by_difficulty") or []
    if rows:
        lines += ["**Tier by difficulty.**", ""]
        # The decisive columns: which metric set the tier, what it was measured
        # against, and why. An earlier version printed the first four metric
        # keys it happened to encounter and a bare baseline number, which on the
        # E2 sweep dropped AUROC and every baseline name -- so a row showed a
        # near-perfect ECE beside "Doesn't work" with nothing to explain it, and
        # a reader could not tell a calibrated model from one whose ECE merely
        # matches the base rate because it answers the same way every time.
        lines.append(
            "| difficulty | n | deciding metric | value | baseline | tier | why |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            base = _fmt(r.get("baseline"))
            bname = r.get("baseline_name")
            bsrc = r.get("baseline_source")
            if bname:
                base = f"{base} ({bname}"
                base += f", {bsrc})" if bsrc and bsrc != "measured" else ")"
            elif r.get("baseline") is not None:
                base = f"{base} (**unnamed**)"
            why = "; ".join(
                list(r.get("reasons") or []) + list(r.get("signatures") or [])
            )
            if not r.get("reportable", True):
                why = ("not reportable; " + why).strip("; ")
            # The row's n is the condition's, but the tier may have been decided
            # on a sub-sample with its own n -- E2's matched control is 200 of a
            # 500-instance sweep cell. Printing only the larger number attributes
            # the tier to more evidence than it had.
            n_txt = str(r.get("n", "?"))
            metric = r.get("metric")
            m = r.get("metrics") or {}
            metric_n = m.get(f"{metric}_n")
            if metric_n is None and isinstance(metric, str) and metric.startswith("matched"):
                metric_n = m.get("matched_n")
            if metric_n is not None and metric_n != r.get("n"):
                n_txt = f"{n_txt} ({metric} n={_fmt(metric_n)})"
            lines.append(
                f"| {_fmt_difficulty(r.get('difficulty'))} | {n_txt} "
                f"| {r.get('metric', '—')} | {_fmt(r.get('value'))} | {base} "
                f"| {r.get('tier', '?')} | {why or '—'} |"
            )
        lines.append("")

        # Every metric each row computed, in full. Truncating this is how a
        # discrimination number like AUROC goes missing next to a calibration
        # number, which the plan warns about by name: "A model can be badly
        # calibrated but perfectly discriminating; you need both numbers to tell
        # those apart."
        metric_keys: list[str] = []
        for r in rows:
            for k in (r.get("metrics") or {}):
                if k not in metric_keys:
                    metric_keys.append(k)
        if metric_keys:
            lines += ["<details><summary>All measured metrics per difficulty</summary>", ""]
            lines.append("| difficulty | n | " + " | ".join(metric_keys) + " |")
            lines.append("| --- | --- | " + " | ".join("---" for _ in metric_keys) + " |")
            for r in rows:
                m = r.get("metrics") or {}
                lines.append(
                    f"| {_fmt_difficulty(r.get('difficulty'))} | {r.get('n', '?')} | "
                    + " | ".join(_fmt(m.get(k)) for k in metric_keys)
                    + " |"
                )
            lines += ["", "</details>", ""]

        lines += _crossings_lines(res, rows)
    else:
        lines += [
            "_This experiment reported no tier-by-difficulty breakdown. The plan "
            "requires one: a single overall grade throws away the finding._",
            "",
        ]

    # 4. The plot.
    for plot in res.get("plots") or []:
        lines.append(f"![{key} {Path(plot).stem}]({plot})")
    if res.get("plots"):
        lines.append("")

    # 5. Prediction versus outcome.
    preds = res.get("predictions") or []
    if preds:
        lines += ["**Prediction versus outcome.**", ""]
        lines.append("| # | predicted | observed | verdict |")
        lines.append("| --- | --- | --- | --- |")
        for p in preds:
            pid = p.get("id", "?")
            claim = p.get("claim") or (
                predictions.BY_ID[pid].claim if pid in predictions.BY_ID else ""
            )
            verdict = p.get("verdict", "?")
            mark = {"right": "right", "wrong": "**WRONG**", "untestable": "untestable"}.get(
                verdict, verdict
            )
            lines.append(f"| {pid} | {claim} | {p.get('outcome', '')} | {mark} |")
        lines.append("")

    # 6. Anything anomalous. `notes` is a parallel channel several experiments
    # use for caveats that are not anomalies -- a plot that failed to render, a
    # measurement substituted for the one the plan named, a baseline that was
    # assumed rather than measured. Leaving it unread hid exactly the kind of
    # qualification the report exists to surface.
    anomalies = list(res.get("anomalies") or [])
    notes = list(res.get("notes") or [])
    if anomalies:
        lines += ["**Anomalies.**", ""]
        lines += [f"- {a}" for a in anomalies]
        lines.append("")
    if notes:
        lines += ["**Notes and caveats.**", ""]
        lines += [f"- {n}" for n in notes]
        lines.append("")

    fails = res.get("failures") or {}
    if fails.get("calls") or fails.get("excluded"):
        lines += [
            f"**Failed calls.** {fails.get('calls', 0)} failed, "
            f"{fails.get('excluded', 0)} excluded from metrics. "
            f"{fails.get('reasons', '')}",
            "",
        ]

    gate = res.get("gate")
    if gate:
        # A gate can pass because it was measured and cleared, or because there
        # was nothing to measure it against. Those are opposite facts and the
        # bolded verdict is what a reader carries away, so an unevaluated gate
        # must not print as "passed" with the explanation buried in prose after
        # it.
        if gate.get("evaluated") is False:
            state = (
                "**NOT EVALUATED** (did not pass; there was nothing to measure "
                "it against)"
            )
        elif gate.get("passed"):
            state = "passed"
        else:
            state = "**TRIPPED**"
        lines += [f"**Gate.** {state}. {gate.get('reason', '')}", ""]

    return lines


def _crossings_lines(res: dict, rows: list[dict]) -> list[str]:
    """Where each tier boundary was crossed, without inventing crossings.

    `tiers.boundary_crossings` reports the first row whose tier is below each
    boundary. When a sweep starts already below a boundary it never crossed it
    -- it was never above it -- and the Crossing carries `from_tier: None` to
    say so. Printing those as crossings turns a sweep that failed every bar into
    what reads as graceful degradation through four tiers, all at the easiest
    difficulty.

    Some experiments also sweep several unrelated conditions rather than one
    difficulty ladder, in which case a single crossings list over row order is
    meaningless. Those modules say so in `boundary_crossings_axis` and supply
    `boundary_crossings_by_condition`; both are preferred here when present.
    """
    lines: list[str] = []
    axis = res.get("boundary_crossings_axis")
    by_condition = res.get("boundary_crossings_by_condition")
    detail = res.get("boundary_crossings_detail")
    crossings = res.get("boundary_crossings") or {}

    tiers_seen = [r.get("tier") for r in rows if r.get("tier")]
    never_cleared = bool(tiers_seen) and set(tiers_seen) <= {"Doesn't work"}

    if axis:
        lines += [f"_Difficulty axis for crossings: {axis}_", ""]

    genuine: dict[str, Any] = {}
    if isinstance(detail, dict) and detail:
        for boundary, c in detail.items():
            if not isinstance(c, dict):
                continue
            if c.get("from_tier") is None:
                continue
            genuine[boundary] = c.get("difficulty")
    elif crossings and not never_cleared:
        genuine = dict(crossings)

    if genuine:
        lines += ["**Tier boundaries crossed at:**", ""]
        for boundary, where in genuine.items():
            lines.append(f"- {boundary}: {_fmt_difficulty(where)}")
        lines.append("")
    elif never_cleared:
        lines += [
            "_No tier boundary was crossed: every difficulty in this sweep is "
            "already at the lowest tier, so the sweep never cleared any bar. "
            "This is not degradation across difficulty._",
            "",
        ]
    else:
        lines += [
            "_No tier boundary was crossed within the difficulty range swept._",
            "",
        ]

    if isinstance(by_condition, dict) and by_condition:
        lines += ["<details><summary>Boundary crossings per condition</summary>", ""]
        for cond, cross in by_condition.items():
            inner = ", ".join(
                f"{b} at {_fmt_difficulty(w)}" for b, w in (cross or {}).items()
            )
            lines.append(f"- **{cond}**: {inner or 'none crossed'}")
        lines += ["", "</details>", ""]
    return lines


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if v is None:
        return "—"
    return str(v)


def _fmt_difficulty(d: Any) -> str:
    if isinstance(d, dict):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in d.items())
    return _fmt(d)


def _unexpected_section(results: dict[str, dict]) -> list[str]:
    lines = [
        "## Unexpected behaviors",
        "",
        "Collected across every experiment, including things no experiment was "
        "designed to test.",
        "",
    ]
    any_found = False
    for key in ORDER:
        res = results.get(key)
        if not res:
            continue
        for a in res.get("anomalies") or []:
            lines.append(f"- **{key}**: {a}")
            any_found = True
    if not any_found:
        lines.append("- None recorded.")
    lines += ["", "**Specifically watched for:**", ""]
    for item in WATCH_LIST:
        lines.append(f"- {item}")
    lines.append("")
    return lines


def _predictions_section(results: dict[str, dict]) -> list[str]:
    verdicts: list[dict] = []
    for res in results.values():
        verdicts.extend(res.get("predictions") or [])
    summary = predictions.score(verdicts)
    lines = ["## Prediction scoreboard", ""]
    hr = summary["hit_rate"]
    lines.append(
        f"- Right: {len(summary['right'])} / {summary['n_testable']} testable"
        + (f" (hit rate {hr:.0%})" if hr is not None else "")
    )
    lines.append(f"- Wrong: {len(summary['wrong'])} — {', '.join(summary['wrong']) or 'none'}")
    lines.append(
        f"- Untestable as specified: {', '.join(summary['untestable']) or 'none'}"
    )
    if summary["unscored"]:
        lines.append(
            f"- **Not yet scored**: {', '.join(summary['unscored'])} — these "
            "experiments have not run, or did not report a verdict."
        )
    lines += [
        "",
        "The plan notes that a high hit rate here would be mildly suspicious, "
        "since it would suggest the experiments are too easy. Wrong predictions "
        "are the most valuable output and are listed above rather than buried.",
        "",
        "**Meta-predictions.**",
        "",
    ]
    for pid, text in predictions.META_PREDICTIONS:
        lines.append(f"- {pid}: {text}")
    lines.append("")
    return lines


def _what_this_changes(results: dict[str, dict]) -> list[str]:
    """The plan asks for three to five sentences on which architectures survive.

    The judgements depend on results that may not exist yet, so this states what
    each conclusion hinges on and fills in the ones it can, rather than
    fabricating a verdict for an experiment that has not run.
    """
    lines = ["## What this changes", ""]
    e6 = results.get("E6")
    e3 = results.get("E3")
    e9 = results.get("E9")

    if e6:
        lines.append(
            f"- **Tabling requirement (E6).** {e6.get('what_this_changes', 'See E6 above.')}"
        )
    else:
        lines.append(
            "- **Tabling requirement (E6).** Not yet measured. Whether a derived "
            "fact must be tabled on first use and treated as authoritative "
            "thereafter depends on the cross-call transitivity and product-rule "
            "results."
        )
    if e3:
        lines.append(
            f"- **Batch-size cap (E3).** {e3.get('what_this_changes', 'See E3 above.')}"
        )
    else:
        lines.append(
            "- **Batch-size cap (E3).** Not yet measured. Whether the "
            "parallel-question designs are constrained depends on where, if "
            "anywhere, positional decay appears."
        )
    if e9:
        lines.append(
            f"- **Judging user-supplied content (E9).** "
            f"{e9.get('what_this_changes', 'See E9 above.')}"
        )
    else:
        lines.append(
            "- **Judging user-supplied content (E9).** Not yet measured. The "
            "injection success rate decides whether moderation, trust-and-safety "
            "and agent-output verification are viable at all."
        )
    lines.append("")
    return lines


def build(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    results = _load_results(run_dir)
    stats = _log_stats(run_dir)
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    lines: list[str] = []
    lines += _header(run_dir, stats, meta)

    if not results:
        lines += [
            "> No experiment results found in this run directory. Run "
            "`run.py smoke` or a phase first.",
            "",
        ]

    for key in ORDER:
        if key in results:
            lines += _experiment_section(key, results[key])
    for key in sorted(set(results) - set(ORDER)):
        lines += _experiment_section(key, results[key])

    lines += _unexpected_section(results)
    lines += _predictions_section(results)
    lines += _what_this_changes(results)

    lines += [
        "## Raw artifacts",
        "",
        f"- JSONL of every call and retry: `{run_dir}/*.jsonl`",
        f"- Per-experiment result JSON: `{run_dir}/*_result.json`",
        f"- Plots: `{run_dir}/plots/`",
        f"- Master seed `{meta.get('master_seed', config.MASTER_SEED)}`; every "
        "instance is reproducible from (generator, difficulty, seed, index).",
        f"- Rebuild this document from the stored results: "
        f"`python run.py report --run-id {run_dir.name}`. Note that this "
        "re-renders the numbers in `*_result.json`; it does not recompute them "
        "from the call log.",
        "- **Offline recomputation is only partly implemented.** The plan makes "
        "it a non-negotiable, and the *data* satisfies it: every call, retry and "
        "failure is in the JSONL with its ground truth in `meta.truth`, and "
        "`jeveval.logstore` re-parses answers from the logged wire response. But "
        "only E2 implements a `score(run_dir)` that rebuilds its metrics from "
        "the log. For the other experiments a redefined metric currently costs a "
        "re-run rather than a rescore.",
        "",
    ]

    out = run_dir / "report.md"
    out.write_text("\n".join(lines))
    return out
