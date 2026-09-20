#!/usr/bin/env python
"""Build the HTML report and the prompting guide from their sources.

The published page is a template plus two JSON payloads, and for a while those
lived in a scratch directory rather than in the repository. That made the most
visible artefact of the evaluation the one thing that could not be rebuilt from
a checkout, so they are here now and this script is what assembles them:

    tools/report_template.html   layout, styling and the rendering code
    tools/reportdata.json        extracted from the nine *_result.json files
    tools/prose.json             the authored text and the twelve rules

    python tools/build_report.py

It writes `runs/<run>/report.html` and `PROMPTING.md`, and both are generated
from the same `prose.json`, so the guide in the repository and the guide inside
the report cannot disagree.

Run `tools/verify_citations.py` afterwards. It reads the same `prose.json` and
fails if a number in the prose does not trace to a measured value.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
REPORT_URL = "https://willkelly.github.io/jev-evaluation/runs/full-20260919/report.html"


def _embed(text: str) -> str:
    """Make a JSON payload safe to sit inside a <script> block."""
    return text.replace("</", "<\\/").replace("<!--", "<\\!--")


def build_html(run: str) -> Path:
    tpl = (TOOLS / "report_template.html").read_text()
    data = (TOOLS / "reportdata.json").read_text()
    prose = (TOOLS / "prose.json").read_text()
    page = tpl.replace("__DATA__", _embed(data)).replace("__PROSE__", _embed(prose))
    if "__DATA__" in page or "__PROSE__" in page:
        raise SystemExit("template placeholders were not both replaced")
    out = ROOT / "runs" / run / "report.html"
    out.write_text(page)
    return out


def _md(fragment: str | None) -> str:
    """The prose carries a little HTML; markdown needs it back as markdown."""
    t = re.sub(r"</p>\s*<p>", "\n\n", fragment or "")
    t = re.sub(r"</?p>", "", t)
    for pattern, repl in (
        (r"<strong>(.*?)</strong>", r"**\1**"),
        (r"<em>(.*?)</em>", r"*\1*"),
        (r"<code>(.*?)</code>", r"`\1`"),
    ):
        t = re.sub(pattern, repl, t, flags=re.S)
    return html.unescape(re.sub(r"[ \t]+", " ", re.sub(r"<[^>]+>", "", t))).strip()


def _lang(code: str) -> str:
    return "json" if code.lstrip().startswith("{") else "python"


def _slug(rule: dict) -> str:
    return re.sub(r"[^a-z0-9]+", "-", rule["title"].lower().replace("’", "")).strip("-")


def build_guide() -> Path:
    prose = json.loads((TOOLS / "prose.json").read_text())
    rules = prose["_rules"]
    out: list[str] = [
        "# A prompting guide for jev", "",
        "Twelve rules that follow from [the evaluation](README.md), in four groups. Each gives",
        "the requests to write, the mistakes to avoid, the measurement behind the advice, and a",
        "link to the experiment and the code that produced it.", "",
        "Figures come from one of two places. Those attributed to an experiment are from the full",
        "run against `jev-1.13.0`, and the sample size is given with each. Those described as",
        "measured live were run while writing this guide, on sixty problems per condition, and are",
        "recorded in [`runs/guide-demos/`](runs/guide-demos/). Both are reproducible from the seeds",
        "in the generators linked below, and `tools/verify_citations.py` checks that every number",
        "here traces to one of them.", "",
        "A difference is treated as real here only if it exceeds the model's own variation between",
        "identical requests, which the first experiment puts at 0.011. Anything smaller than roughly",
        "twice that is not a finding, and is not used as one below.", "",
        f"The full report, with figures, is at <{REPORT_URL}>.", "", "---", "",
    ]
    last = None
    for r in rules:
        if r.get("group") != last:
            last = r["group"]
            out += [f"### {last}", ""]
        out.append(f"{r['n']}. [{r['title']}](#{r['n']}-{_slug(r)})")
    out.append("")

    last = None
    for r in rules:
        if r.get("group") != last:
            last = r["group"]
            out += ["---", "", f"# {last}", ""]
        out += [f"## {r['n']}. {r['title']}", "", r["rule"], ""]
        for it in r["items"]:
            tag = "**Do**" if it["kind"] == "do" else "**Don't**"
            out += [f"{tag} — {it['label']}", "", f"```{_lang(it['code'])}", it["code"], "```", ""]
        if r.get("evidence"):
            out += [_md(r["evidence"]), ""]
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
            out += ["> " + _md(r["warn"]).replace("\n\n", "\n>\n> "), ""]
        if r.get("from"):
            bits = [f"[{f['label']}]({REPORT_URL}#x{f['id']}) — "
                    f"[`{f['path'].split('/')[-1]}`]({f['path']})" for f in r["from"]]
            gens = sorted({g for f in r["from"] for g in prose[f["id"]]["source"]["generators"]})
            out += ["*Produced by " + " and ".join(bits) + ". Problems and their answers come from "
                    + ", ".join(f"[`{g.split('/')[-1]}`]({g})" for g in gens) + ".*", ""]
    out += ["---", "", "## Where these come from", "",
            "Every figure above is either from the full run of nine experiments described in",
            "[the plan](jev-evaluation-plan.md), or measured directly while writing this guide and",
            "recorded in [`runs/guide-demos/`](runs/guide-demos/). Ground truth always comes from a",
            "solver or from construction, never from the model and never from another model.", ""]

    path = ROOT / "PROMPTING.md"
    path.write_text("\n".join(out))
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="full-20260919")
    args = ap.parse_args(argv)
    page = build_html(args.run)
    guide = build_guide()
    print(f"wrote {page.relative_to(ROOT)}  ({page.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {guide.relative_to(ROOT)}  ({len(guide.read_text().splitlines())} lines)")
    print("\nnow run: python tools/verify_citations.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
