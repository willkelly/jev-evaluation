# jev evaluation harness

Implements [`jev-evaluation-plan.md`](jev-evaluation-plan.md): nine experiments
against TypeSafe's `jev` decision model, with instance generators that carry
their own ground truth, a metrics library, and a report writer.

## Setup

This machine runs Guix, whose Python profile already provides working numpy,
scipy and matplotlib. A broken numpy in `~/.local/lib/python3.11/site-packages`
shadows the good one for the system interpreter, and a virtualenv excludes user
site-packages, so creating the venv *with* system site-packages is what fixes
both problems at once:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install python-sat
```

Do not `pip install numpy` here — the PyPI manylinux wheel cannot find
`libz.so.1` on this system. The Guix copy works; use it.

HTTP goes through the standard library rather than httpx. The Guix profile
precedes the virtualenv on `sys.path`, so a pip-installed package can be
shadowed by an older Guix copy of one of its dependencies, which is what broke
httpx by way of anyio and typing_extensions. The only request this harness makes
is a JSON POST, so `http.client` with one keep-alive connection per worker
thread costs nothing and removes the whole problem.

## The API key

The key lives in `pass`, and decrypting it requires a physical YubiKey touch
with a short timeout, so it is fetched as rarely as possible — once per session,
not once per request.

```sh
scripts/prime-key.sh           # fetch once, cache for the session (prompts for a touch)
scripts/prime-key.sh --check   # is the cache warm?
scripts/prime-key.sh --clear   # drop it
```

The cache lives in `$XDG_RUNTIME_DIR`, which is tmpfs: RAM-backed, mode 0700,
and gone at logout. Nothing is written to disk, nothing is committed, and
`jeveval.auth.redact` strips the key from anything logged or printed. Setting
`TYPESAFE_API_KEY` in the environment overrides all of this.

## Running

```sh
.venv/bin/python run.py smoke            # Phase 0: auth, three question types, control
.venv/bin/python run.py phase1           # E1 determinism, E2 calibration  (gated)
.venv/bin/python run.py phase2           # E3 batching, E7 cardinality     (gated)
.venv/bin/python run.py phase3           # E5 enrollment, E6 coherence
.venv/bin/python run.py phase4           # E4 input size, E8 ICL, E9 edges
.venv/bin/python run.py all              # every phase in order, stopping at a tripped gate
.venv/bin/python run.py e2               # one experiment, bypassing the gates
.venv/bin/python run.py report --run-id run-20260919-130000
.venv/bin/python run.py status --run-id run-20260919-130000
```

`--scale F` multiplies every sample size, for dry runs. It is recorded in the
run metadata and printed in the report header, because a silently shrunk sample
size is how a noisy result gets published as a finding. Use `--scale 0.02` to
rehearse the whole plan for a few hundred calls before committing to the full
run.

Phase 0 is a hard gate. Every formal domain in the plan is off-distribution for
this model, so a bad result there is ambiguous between "hard domain", "broken
harness" and "weak model"; the semantic control is the only thing that separates
them. If the control fails, nothing downstream is interpretable.

## Layout

```
jeveval/
  auth.py         key acquisition; one touch per session
  wire.py         neutral question/answer types <-> the real wire format
  client.py       adaptive concurrency, retry with backoff, JSONL logging
  logstore.py     offline reading of the log; every metric recomputable from it
  metrics.py      ECE, Brier, AUROC, reliability bins, KL, Wilson, paired tests
  plots.py        reliability diagrams and curves
  tiers.py        the plan's five-tier rubric, as data
  predictions.py  the plan's P1-P28 table, so predictions are scored mechanically
  smoke.py        Phase 0
  report.py       the markdown report
  generators/     3SAT, graph reachability, program reachability, Sudoku,
                  pairwise ordering, DFA, Dyck words, the semantic control,
                  and filler/dilution material
  experiments/    E1 .. E9
runs/<run-id>/    calls.jsonl, *_result.json, plots/, report.md
```

## What the endpoint actually accepts

The plan was written from documentation and press coverage, and its paraphrase
of the request shape is wrong in three ways that matter. Confirmed by probing:

- The prompt field is `instructions`, not `question`. A `criteria` field also
  exists and is typed per question kind: an object for `noul`, an object of
  `{option_id: description}` for `choice`, and an ordered list for `score`.
- There is no top-level `instructions`; sending one is a 400. E8's
  "examples in instructions" channel is therefore the per-question field.
- A `noul` answer is a bare probability with **no confidence field**. Only
  `choice` and `score` carry `confidence`. Anything asking about confidence on a
  yes/no question — E9's abstention test — has to use the probability's distance
  from 0.5 instead, and the report says so.

`choice` accepts at most 255 options, confirmed: 256 is rejected. The versioned
model string observed is `jev-1.13.0`, and `score` returns an undocumented
`legend` field mapping rubric indices back to labels.

## Reproducibility

Every instance is a deterministic function of `(generator, difficulty, seed,
index)`, so instance 400 of a condition can be regenerated without generating
the 399 before it. Every call, retry and failure is appended to JSONL before any
metric is computed, and all analysis runs offline against that log — so a metric
can be redefined and the whole report rebuilt without re-spending a call. A
failed call is recorded as a failure and excluded with a count; it is never
defaulted to 0.5, to `False`, or to the majority class.
