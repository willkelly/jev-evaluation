#!/usr/bin/env python
"""Check that every number in the written report traces to a measured value.

The report and the prompting guide are prose written by hand around figures
produced by the experiments. Nothing stops the two drifting: a number can be
mistyped, quoted from an earlier run, rounded in a flattering direction, or
simply invented. Reading for that is exactly the kind of check a person does
badly and a program does well, so this does it mechanically.

How it works. Every numeric leaf in the deterministic outputs -- the nine
`*_result.json` files from the full run and the live measurements in
`runs/guide-demos/` -- goes into an index keyed by value, remembering where it
came from. Every number in the authored prose is then looked up in that index,
at a tolerance set by how precisely it was written: a figure given as `0.894`
must match to three decimals, one given as `35` need only match to the nearest
whole number.

Many figures are derived rather than copied -- a ratio of two measurements, a
percentage, a difference, a sum of shares. Those would all show as unmatched
against a naive index, so the index also holds pairwise ratios, differences and
percentages of values found within the same file. That is a large search space
and it makes false matches possible, which is why the output distinguishes a
direct match from a derived one and reports how many paths a value matched:
a figure matching one path is evidence, a figure matching two hundred is not.

What this cannot check is whether a correctly-sourced number is used to support
the right claim. That is left to a reader, and this tool exists to leave them a
much shorter list to read.

    python tools/verify_citations.py                 # summary, non-zero on failure
    python tools/verify_citations.py --json out.json # full detail
    python tools/verify_citations.py --show-matched  # print sources for everything
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent

# Numbers so common, or so obviously structural, that matching them proves
# nothing and failing to match them means nothing.
TRIVIAL = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 100, 255, 0.0, 0.5, 1.0}

# Figures that are definitional, quoted from the vendor's documentation, or
# otherwise not produced by this evaluation. Each needs a stated reason.
ALLOWED: dict[str, str] = {
    "64": "documented context limit, from the vendor's model page",
    "32": "documented context limit for state plus the longest question",
    "4.26": "the clause ratio of the satisfiability phase transition, a property of the problem",
    "1.13": "the model version string",
    "20260919": "the master seed",
    "42": "the vendor's price, dollars per billion input tokens",
    "0.01": "the grid the returned probabilities lie on",
    "0.02": "twice the measured noise floor, the threshold this report uses",
    "0.011": "the measured noise floor, quoted from experiment 1",
    "0.3": "the plan's stated threshold for a usable confidence separation",
    "0.95": "a confidence threshold chosen for illustration",
    "0.8": "a confidence threshold chosen for illustration",
    "4417": "an invented order number in an example request",
    "812": "an invented account age in an example request",
    "1041": "an invented department id in an example request",
    "1042": "an invented department id in an example request",
    "2213": "an invented department id in an example request",
}


def walk(obj: Any, path: str = "") -> Iterator[tuple[str, float]]:
    """Every numeric leaf in a JSON document, with the path that reaches it."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            yield path, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        # Numbers embedded in generated sentences count as measured: the
        # experiments write their own outcome text and the report quotes it.
        for m in re.finditer(r"-?\d+(?:\.\d+)?", obj):
            try:
                yield f"{path}(text)", float(m.group())
            except ValueError:
                pass


class Index:
    """Measured values, and the simple combinations of them."""

    def __init__(self) -> None:
        self.direct: dict[float, list[str]] = defaultdict(list)
        self.derived: dict[float, list[str]] = defaultdict(list)

    def add_file(self, path: Path, label: str) -> int:
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return 0
        vals: list[tuple[str, float]] = list(walk(doc, label))
        for p, v in vals:
            # 10 decimals, not 6: a cost per decision of 4.4e-06 collapses to
            # 4e-06 at six and then never matches what the prose quotes.
            self.direct[round(v, 10)].append(p)
        # Derived figures, within one file only. Ratios and differences across
        # unrelated files would match almost anything.
        nums = [(p, v) for p, v in vals if abs(v) > 1e-9][:1200]
        for i, (pa, a) in enumerate(nums):
            for pb, b in nums[i + 1 : i + 60]:
                if abs(b) > 1e-9:
                    for val, how in ((a / b, "/"), (b / a, "/"), (a - b, "-"), (b - a, "-")):
                        if math.isfinite(val) and abs(val) < 1e7:
                            self.derived[round(val, 10)].append(f"{pa} {how} {pb}")
        for p, v in vals:
            self.derived[round(v * 100, 10)].append(f"{p} as a percentage")
        return len(vals)

    def find(self, value: float, decimals: int) -> tuple[str, list[str]]:
        tol = 0.5 * (10 ** -decimals) if decimals else 0.5
        for kind, table in (("direct", self.direct), ("derived", self.derived)):
            hits: list[str] = []
            for v, paths in table.items():
                if abs(v - value) <= tol:
                    hits.extend(paths)
                    if len(hits) > 400:
                        break
            if hits:
                return kind, hits
        return "none", []


NUMBER = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(?![\w])")


def claims(prose: dict) -> Iterator[tuple[str, str, float, int]]:
    """Every number written by hand, with where it sits and its precision."""

    def scan(where: str, text: Any) -> Iterator[tuple[str, str, float, int]]:
        if not isinstance(text, str):
            return
        plain = re.sub(r"<[^>]+>", " ", text)
        for m in NUMBER.finditer(plain):
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
            except ValueError:
                continue
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            start = max(0, m.start() - 60)
            ctx = " ".join(plain[start : m.end() + 60].split())
            yield where, ctx, val, decimals

    for r in prose.get("_rules", []):
        base = f"rule {r['n']}"
        for field in ("rule", "evidence", "warn"):
            yield from scan(f"{base}.{field}", r.get(field))
        for it in r.get("items", []):
            yield from scan(f"{base}.item[{it['label'][:28]}]", it.get("code"))
        for row in (r.get("table") or {}).get("rows", []):
            for cell in row:
                yield from scan(f"{base}.table", cell)
    for key, block in prose.items():
        if not isinstance(block, dict):
            continue
        for name in ("calibration", "batching", "encoding", "abstention", "injection"):
            sec = block.get(name)
            if not isinstance(sec, dict):
                continue
            for field in ("intro", "analysis", "second", "caveat"):
                yield from scan(f"{key}.{name}.{field}", sec.get(field))
            for row in (sec.get("table") or {}).get("rows", []):
                for cell in row:
                    yield from scan(f"{key}.{name}.table", cell)
        for field in ("asks", "method", "found", "surprise", "learned"):
            yield from scan(f"{key}.{field}", block.get(field))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prose", default=None, help="prose JSON; defaults to the one beside this tool")
    ap.add_argument("--json", dest="out", default=None)
    ap.add_argument("--show-matched", action="store_true")
    args = ap.parse_args(argv)

    prose_path = Path(args.prose) if args.prose else ROOT / "tools" / "prose.json"
    if not prose_path.exists():
        print(f"no prose file at {prose_path}", file=sys.stderr)
        return 2
    prose = json.loads(prose_path.read_text())

    idx = Index()
    sources = 0
    for f in sorted((ROOT / "runs" / "full-20260919").glob("*_result.json")):
        sources += idx.add_file(f, f.stem)
    for f in sorted((ROOT / "runs" / "guide-demos").glob("*.json")):
        sources += idx.add_file(f, f"demo:{f.stem}")

    matched, derived, unmatched, allowed = [], [], [], []
    for where, ctx, val, dec in claims(prose):
        key = ("%g" % val)
        if val in TRIVIAL:
            continue
        if key in ALLOWED:
            allowed.append({"where": where, "value": val, "why": ALLOWED[key], "context": ctx})
            continue
        kind, hits = idx.find(val, dec)
        rec = {"where": where, "value": val, "decimals": dec, "context": ctx,
               "n_sources": len(hits), "sources": hits[:4]}
        (matched if kind == "direct" else derived if kind == "derived" else unmatched).append(rec)

    total = len(matched) + len(derived) + len(unmatched) + len(allowed)
    print(f"indexed {sources:,} measured values from "
          f"{len(list((ROOT/'runs'/'full-20260919').glob('*_result.json')))} result files "
          f"and {len(list((ROOT/'runs'/'guide-demos').glob('*.json')))} live-measurement files")
    print(f"checked {total} numbers written by hand\n")
    print(f"  {len(matched):>4}  match a measured value directly")
    print(f"  {len(derived):>4}  match a ratio, difference or percentage of measured values")
    print(f"  {len(allowed):>4}  documented constants, listed in ALLOWED with a reason")
    print(f"  {len(unmatched):>4}  UNMATCHED")

    weak = [m for m in matched + derived if m["n_sources"] > 60]
    if weak:
        print(f"\n  {len(weak)} matched so many values that the match is not evidence "
              "(a reader should confirm these by hand)")

    if unmatched:
        print("\nUnmatched — no measured value, ratio or difference equals these:")
        for u in sorted(unmatched, key=lambda x: x["where"]):
            print(f"  {u['where']:<40} {u['value']:<12g} …{u['context'][:88]}")
    if args.show_matched:
        print("\nMatched:")
        for m in sorted(matched + derived, key=lambda x: x["where"]):
            print(f"  {m['where']:<40} {m['value']:<12g} {m['n_sources']:>4} src  {m['sources'][:1]}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"matched": matched, "derived": derived, "unmatched": unmatched,
             "allowed": allowed, "weak": weak}, indent=1))
        print(f"\nfull detail written to {args.out}")
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
