#!/usr/bin/env python
"""How should a question point at one of many subjects sharing a request?

The guide's second rule tells readers not to put several subjects in one
request. The measurement behind it puts sixty support tickets in one state and
asks sixty questions, each naming its ticket by position -- "Ticket 1: which
department?" -- and that scores far below asking one ticket per request. The
rule concluded that carrying several subjects is the mistake.

A later probe on a different task suggested the conclusion was too broad: there,
identifying each item by quoting it rather than by numbering it recovered the
whole loss. This adds that third arm to the ticket measurement, so the two
readings can be told apart on the task the rule is actually written about.

    A  one ticket per request                       the baseline
    B  all sixty in one state, questions by position
    C  all sixty in one state, questions by sender and date

B and C send byte-identical states and the same eight options. Only the wording
that picks out the ticket differs, which is the whole of the comparison.

What C may not do is hand over the answer. This generator builds a ticket so
that its subject line and its opening sentence carry keywords of its own
department -- that is what makes the keyword baseline work -- so quoting either
would let the question classify the ticket on the harness's behalf and the arm
would score well for the wrong reason. The sender's name, address and the date
received contain no department keyword on any ticket, and are unique across the
sixty, so they identify without classifying. The script asserts both properties
before sending anything.

    python tools/ticket_binding.py --blocks 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jeveval.client import JevClient, Call            # noqa: E402
from jeveval.generators import semantic               # noqa: E402

DEPARTMENTS = {
    "account_access": "Logins, passwords, lockouts and multi-factor problems",
    "billing": "Payments, invoices, refunds and subscription charges",
    "careers": "Job applications, interviews and recruitment",
    "press": "Media enquiries, interviews and fact-checking",
    "privacy": "Data protection, retention, erasure and access requests",
    "sales": "Quotes, upgrades, new business and contract terms",
    "shipping": "Deliveries, tracking, damage in transit and returns",
    "technical": "Faults, errors, outages and anything not working",
}
OPTIONS = [{"id": k, "label": v} for k, v in DEPARTMENTS.items()]


def identifier(t) -> str:
    """How arm C names a ticket: metadata fields only, never its content."""
    return f"{t.customer} <{t.customer_email}>, received {t.received}"


def check_no_leak(tickets) -> None:
    """Refuse to run if the identifier would classify the ticket for the model."""
    for t in tickets:
        counts = semantic.keyword_counts(identifier(t))
        if sum(counts.values()):
            raise SystemExit(
                f"identifier for {t.customer!r} carries department keywords {counts}; "
                "arm C would be scoring the harness, not the model")
    ids = {identifier(t) for t in tickets}
    if len(ids) != len(tickets):
        raise SystemExit(f"identifiers are not unique: {len(ids)} for {len(tickets)} tickets")


def build(block: int, tickets) -> list[Call]:
    calls: list[Call] = []
    for i, t in enumerate(tickets):                                     # A
        calls.append(Call("TICKET-BINDING", "A-one-per-request", t.as_state(),
                          {"dept": {"type": "choice", "options": OPTIONS,
                                    "question": "Which department should handle this ticket?"}},
                          instance_id=f"b{block}-t{i}",
                          meta={"block": block, "i": i, "truth": {"dept": t.department}}))

    # One state, shared verbatim by B and C.
    state = {"tickets": [t.as_state() for t in tickets]}
    truth = {f"t{i+1:03d}": t.department for i, t in enumerate(tickets)}

    calls.append(Call("TICKET-BINDING", "B-by-position", state,                # B
                      {f"t{i+1:03d}": {"type": "choice", "options": OPTIONS,
                                       "question": f"Ticket {i+1}: which department "
                                                   "should handle it?"}
                       for i in range(len(tickets))},
                      instance_id=f"b{block}", meta={"block": block, "truth": truth}))

    calls.append(Call("TICKET-BINDING", "C-by-sender", state,                  # C
                      {f"t{i+1:03d}": {"type": "choice", "options": OPTIONS,
                                       "question": f"The ticket from {identifier(t)}: "
                                                   "which department should handle it?"}
                       for i, t in enumerate(tickets)},
                      instance_id=f"b{block}", meta={"block": block, "truth": truth}))
    return calls


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--per-block", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out", default=str(ROOT / "runs" / "guide-demos"))
    ap.add_argument("--score-only", action="store_true",
                    help="rescore the existing log without sending anything")
    args = ap.parse_args(argv)

    blocks = {b: semantic.tickets(seed=args.seed + b, count=args.per_block, level="clean")
              for b in range(args.blocks)}
    for ts in blocks.values():
        check_no_leak(ts)
    print(f"{args.blocks} blocks x {args.per_block} tickets; identifiers carry no department "
          f"keyword and are unique within every block")

    calls = [c for b, ts in blocks.items() for c in build(b, ts)]
    print(f"{len(calls)} requests: " + ", ".join(
        f"{c}={sum(1 for x in calls if x.condition == c)}"
        for c in sorted({x.condition for x in calls})))

    out = Path(args.out)
    log = out / "ticket-binding.jsonl"
    if not args.score_only:
        with JevClient(run_dir=out, log_name="ticket-binding.jsonl") as client:
            client.run(calls, label="ticket binding", progress_every=50)
    elif not log.exists():
        raise SystemExit(f"--score-only needs {log}, which does not exist")

    # Scored from the log rather than from the returned objects, so that a
    # scoring change costs nothing to re-run. The wire parser renames a choice's
    # selected option to `chosen`; the log keeps the endpoint's own `choice`.
    arms: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "correct": 0, "calls": 0, "itok": 0, "lat": []})
    per_position: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    with log.open() as fh:
        for raw in fh:
            rec = json.loads(raw)
            if rec.get("outcome") != "ok" or rec.get("experiment") != "TICKET-BINDING":
                continue
            cond = rec["condition"]
            a = arms[cond]
            a["calls"] += 1
            a["itok"] += rec.get("input_tokens") or 0
            a["lat"].append(rec["latency_s"])
            answers = rec["response"].get("answers") or {}
            for key, want in rec["meta"]["truth"].items():
                ans = answers.get(key) or {}
                got = ans.get("choice", ans.get("chosen"))
                a["n"] += 1
                a["correct"] += got == want
                pos = rec["meta"]["i"] if cond.startswith("A") else int(key[1:]) - 1
                per_position[cond][pos].append(int(got == want))
    if not any(a["n"] for a in arms.values()):
        raise SystemExit("nothing scored -- the log holds no TICKET-BINDING records")

    payload = {
        "source": "tools/ticket_binding.py",
        "question": "how a question should point at one of many subjects in a shared request",
        "design": {"blocks": args.blocks, "tickets_per_block": args.per_block,
                   "seed": args.seed, "options": len(OPTIONS), "chance": 1 / len(OPTIONS),
                   "identifier": "sender name, address and date received",
                   "identifier_leaks_department": False,
                   "state_identical_between_B_and_C": True},
        "arms": {},
    }
    for cond in sorted(arms):
        a = arms[cond]
        payload["arms"][cond] = {
            "requests": a["calls"], "judgements": a["n"],
            "accuracy": a["correct"] / a["n"] if a["n"] else None,
            "correct": a["correct"],
            "input_tokens": a["itok"],
            "input_tokens_per_judgement": a["itok"] / a["n"] if a["n"] else None,
            "latency_p50": sorted(a["lat"])[len(a["lat"]) // 2] if a["lat"] else None,
        }
    # Does accuracy decay with position in the list? That is the other story
    # a numbered arm could be telling.
    for cond, pos in per_position.items():
        thirds = [[], [], []]
        for p, hits in pos.items():
            thirds[min(2, p * 3 // args.per_block)].extend(hits)
        payload["arms"][cond]["accuracy_by_third_of_list"] = [
            sum(t) / len(t) if t else None for t in thirds]

    path = out / "ticket-binding.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"\n{'arm':<20} {'reqs':>5} {'judged':>7} {'accuracy':>9} {'by third of the list':>26}")
    for cond in sorted(payload["arms"]):
        a = payload["arms"][cond]
        thirds = "  ".join(f"{x:.3f}" if x is not None else "  --"
                           for x in a["accuracy_by_third_of_list"])
        print(f"{cond:<20} {a['requests']:>5} {a['judgements']:>7} {a['accuracy']:>9.3f} "
              f"{thirds:>26}")
    print(f"\nchance is {1/len(OPTIONS):.3f}; wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
