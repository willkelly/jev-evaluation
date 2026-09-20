#!/usr/bin/env python
"""Regenerate PROMPTING.md, and re-inject the prose into the built report.

Both the standalone guide and the report page are rendered from one source,
`tools/prose.json`, so a rule cannot say one thing in the repository and
another on the page. This script is that renderer. It was previously a throwaway
in a scratch directory, which meant the guide could not be rebuilt from a clean
checkout; keeping it here is the point.

    python tools/build_guide.py            # rewrite PROMPTING.md and report.html
    python tools/build_guide.py --check    # fail if either is out of date

After changing `tools/prose.json`, run this and then `tools/verify_citations.py`,
which fails if any number in the new prose does not trace to a measured value.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROSE = ROOT / "tools" / "prose.json"
GUIDE = ROOT / "PROMPTING.md"
REPORT = ROOT / "runs" / "full-20260919" / "report.html"
REPORT_URL = "https://willkelly.github.io/jev-evaluation/runs/full-20260919/report.html"


def to_markdown(fragment: str | None) -> str:
    """The prose holds small HTML fragments; the guide needs markdown."""
    t = re.sub(r"</p>\s*<p>", "\n\n", fragment or "")
    t = re.sub(r"</?p>", "", t)
    for pattern, repl in (
        (r"<strong>(.*?)</strong>", r"**\1**"),
        (r"<em>(.*?)</em>", r"*\1*"),
        (r"<code>(.*?)</code>", r"`\1`"),
        (r"<h4>(.*?)</h4>", r"**\1**\n"),
    ):
        t = re.sub(pattern, repl, t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    return html.unescape(re.sub(r"[ \t]+", " ", t)).strip()


def fence(code: str) -> str:
    return "json" if code.lstrip().startswith("{") else "python"


def slug(rule: dict) -> str:
    return re.sub(r"[^a-z0-9]+", "-", rule["title"].lower().replace("’", "")).strip("-")


def render_guide(prose: dict) -> str:
    rules = prose["_rules"]
    out = [
        "# A prompting guide for jev", "",
        "Twelve rules that follow from [the evaluation](README.md), in four groups. Each gives",
        "the requests to write, the mistakes to avoid, the measurement behind the advice, and a",
        "link to the experiment and the code that produced it.", "",
        "Figures come from one of two places. Those attributed to an experiment are from the full",
        "run against `jev-1.13.0`, and the sample size is given with each. Those described as",
        "measured live were run while writing this guide, on sixty problems per condition. Both are",
        "reproducible from the seeds in the generators linked below, and every number here is",
        "checked against the measured results by `tools/verify_citations.py`.", "",
        "A difference is treated as real here only if it exceeds the model's own variation between",
        "identical requests, which the first experiment puts at 0.011. Anything smaller than roughly",
        "twice that is not a finding, and is not used as one below.", "",
        f"The full report, with figures, is at <{REPORT_URL}>.", "", "---", "",
    ]
    group = None
    for r in rules:
        if r.get("group") != group:
            group = r["group"]
            out += [f"### {group}", ""]
        out.append(f"{r['n']}. [{r['title']}](#{r['n']}-{slug(r)})")
    out.append("")

    group = None
    for r in rules:
        if r.get("group") != group:
            group = r["group"]
            out += ["---", "", f"# {group}", ""]
        out += [f"## {r['n']}. {r['title']}", "", r["rule"], ""]
        for it in r["items"]:
            tag = "**Do**" if it["kind"] == "do" else "**Don't**"
            out += [f"{tag} — {it['label']}", "", f"```{fence(it['code'])}", it["code"], "```", ""]
        if r.get("evidence"):
            out += [to_markdown(r["evidence"]), ""]
        if r.get("table"):
            t = r["table"]
            out += ["| " + " | ".join(t["head"]) + " |",
                    "|" + "|".join(["---"] * len(t["head"])) + "|"]
            out += ["| " + " | ".join(row) + " |" for row in t["rows"]] + [""]
        if r.get("example"):
            e = r["example"]
            out += ["Text added to an otherwise ordinary support ticket:", "",
                    "> " + e["injected"], "",
                    "| Correct department | Answer, clean | Answer, with that text added |",
                    "|---|---|---|",
                    f"| {e['true']} | {e['clean']} | **{e['attacked']}** |", ""]
        if r.get("warn"):
            out += ["> " + to_markdown(r["warn"]).replace("\n\n", "\n>\n> "), ""]
        if r.get("from"):
            bits = [f"[{f['label']}]({REPORT_URL}#x{f['id']}) — "
                    f"[`{f['path'].split('/')[-1]}`]({f['path']})" for f in r["from"]]
            gens = sorted({g for f in r["from"] for g in prose[f["id"]]["source"]["generators"]})
            out += ["*Produced by " + " and ".join(bits) + ". Problems and their answers come from " +
                    ", ".join(f"[`{g.split('/')[-1]}`]({g})" for g in gens) + ".*", ""]
    out += ["---", "", "## Where these come from", "",
            "Every figure above is either from the full run of nine experiments described in",
            "[the plan](jev-evaluation-plan.md), or measured directly while writing this guide.",
            "Ground truth always comes from a solver or from construction, never from the model",
            "and never from another model.", ""]
    return "\n".join(out)


def render_report(prose: dict, current: str) -> str:
    """Swap the prose the page carries, leaving its markup and data untouched."""
    payload = json.dumps(prose, separators=(",", ":"), ensure_ascii=False)
    payload = payload.replace("</", "<\\/").replace("<!--", "<\\!--")
    new, n = re.subn(r'(<script type="application/json" id="prose">).*?(</script>)',
                     lambda m: m.group(1) + payload + m.group(2), current, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("could not find the prose block in the report page")
    return new


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="fail instead of writing")
    args = ap.parse_args(argv)

    prose = json.loads(PROSE.read_text())
    guide = render_guide(prose)
    report = render_report(prose, REPORT.read_text())

    stale = []
    if GUIDE.read_text() != guide:
        stale.append(str(GUIDE.relative_to(ROOT)))
    if REPORT.read_text() != report:
        stale.append(str(REPORT.relative_to(ROOT)))

    if args.check:
        if stale:
            print("out of date with tools/prose.json: " + ", ".join(stale))
            return 1
        print("PROMPTING.md and report.html are current")
        return 0

    GUIDE.write_text(guide)
    REPORT.write_text(report)
    print(f"wrote PROMPTING.md ({len(guide.splitlines())} lines) and report.html "
          f"({len(report) // 1024} KB) from {len(prose['_rules'])} rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
