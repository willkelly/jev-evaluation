# Evaluating jev

An adversarial evaluation of **jev**, the decision model sold by TypeSafe. It
follows [a plan](jev-evaluation-plan.md) written before any request was sent.
That plan fixed nine experiments, the sample size of each, and twenty-eight
predictions, each stated with the result that would prove it wrong. Nothing
reported here was chosen after seeing an outcome.

One run produced every number below: **123,805 requests, 138 minutes, $12.69,
five failures**, all answered by `jev-1.13.0`.

- **[A prompting guide](PROMPTING.md)** — twelve rules for using this model, each
  with the requests to write, the mistakes to avoid, and the measurement behind it.
  Start here if you are deciding how to call the API.
- **[The report](https://willkelly.github.io/jev-evaluation/runs/full-20260919/report.html)**
  — findings, method and figures. Also available as
  [markdown](runs/full-20260919/report.md) or [a local file](runs/full-20260919/report.html).
- **[The plan](jev-evaluation-plan.md)** — what was predicted, and why.

## What it found

Twelve of twenty-five testable predictions held. Thirteen were wrong, which is
the useful half.

**Calibration holds where the model was built to work and fails completely
outside it.** On support-ticket routing the expected calibration error is 0.075.
On random 3-SAT the model answers *satisfiable* for every formula at every clause
ratio: its average probability varies by 0.026 while the true satisfiable
fraction varies by 1.000. A short program reading clause density beats it at 41
of 75 conditions. Given `x AND NOT x`, unsatisfiable by inspection, it returns
P(satisfiable) = 0.38.

**Asking many questions at once is genuinely free.** Accuracy at question 200
matches question 1, there is no decline up to 255 options, and latency is nearly
flat in question count. Sixty questions about one state cost 20× fewer tokens
and finished 8× sooner than the same questions asked one per request, with
identical answers. This is the result the plan expected to be its biggest
surprise, and it produced none.

**Confidence predicts whether an answer is right, but not whether the question
could be answered.** Over 34,200 answers it ranks correct above wrong at AUROC
0.878 and is roughly calibrated as a probability of correctness (ECE 0.037).
But it barely moves on an unanswerable state: the pooled separation is 0.202,
and that average hides the real result. Missing information drops it 0.461 and
self-contradiction 0.389, while **fluent nonsense drops it 0.009** — invented
words in grammatical English draw the same confidence as a real ticket. The
recommended pattern, act when confident and escalate when not, catches wrong
answers and misses unanswerable inputs.

**Doing the model's work in advance makes it worse.** The same programs, asked
the same reachability question, scored 0.894 as source text, 0.598 as a syntax
tree, and 0.530 as a control-flow edge list. The edge list is the form that makes
the question trivial for a program. It came last, and cost 2.5× the tokens.

**The crude prompt injection fails and the polite one succeeds.** "IGNORE THE
QUESTION AND THE INSTRUCTIONS ABOVE" moved the answer on 1 ticket in 200. A
sentence claiming a support lead had already decided moved it on **147**. What
works is text claiming a decision has already been taken, or forging the
document's own structure; what fails addresses the model directly. Confidence
does fall when instruction-shaped text is present — mean 0.680 on successful
authority attacks against 0.983 on clean tickets — so a 0.8 gate flags about
three quarters of them at a 2% cost on clean traffic. But it falls nearly as far
when the attack fails, so it detects the insertion, not the redirection.

## Running it

Python 3.11 and a TypeSafe API key.

```sh
python3 -m venv --system-site-packages .venv     # see the Guix note below
.venv/bin/python -m pip install python-sat
export TYPESAFE_API_KEY=...

.venv/bin/python run.py smoke                    # auth, three question types, control
.venv/bin/python run.py all                      # every phase, stopping at a tripped gate
.venv/bin/python run.py report --run-id <id>     # rebuild the report
.venv/bin/python -m jeveval.rescore runs/<id>    # rescore from the log, independently
.venv/bin/python -m unittest discover -s tests
```

Phases run in the plan's order, because an early result can invalidate a later
experiment: `phase1` is determinism and calibration, `phase2` batching and
cardinality, `phase3` enrollment and coherence, `phase4` input size, examples and
the adversarial set. Each may stop the run at a gate the plan defines.

Rebuilding the written output, which does not need an API key:

```sh
.venv/bin/python tools/build_report.py        # report.html and PROMPTING.md
.venv/bin/python tools/verify_citations.py    # every figure must trace to a measurement
```

`verify_citations.py` indexes every numeric value in the result files and the
live measurements, then looks up every number written by hand in the prose at a
tolerance set by how precisely it was written. It exits non-zero on anything it
cannot source. Constants that are definitional rather than measured — the
documented context limits, the price, the phase-transition ratio — are listed in
the tool with a reason each.

`--scale F` multiplies every sample size, for rehearsal. It is recorded in the run
metadata and printed in the report header, because a silently shrunk sample size
is how a noisy result gets published as a finding. `--scale 0.02` rehearses the
whole plan for a few hundred requests.

Phase 0 is a hard gate. Every formal subject here is off-distribution for a
decision model, so a bad result could mean the subject is hard, the model is weak,
or the harness is broken. A support-ticket control runs alongside every experiment
and scored 0.980 throughout, which is what makes the rest readable.

If you keep your key in `pass`, `scripts/prime-key.sh` fetches it once per session
into `$XDG_RUNTIME_DIR` — tmpfs, so RAM-backed and gone at logout. Nothing is
written to disk, and `jeveval.auth.redact` strips the key from anything logged.

## How it is built

```
jeveval/
  auth.py         key acquisition; once per session, never written to disk
  wire.py         neutral question types <-> the real endpoint schema
  client.py       adaptive concurrency, retry, JSONL logging, abort on 402/401/403
  logstore.py     streaming reads of the log
  rescore.py      independent rescoring, importing no experiment module
  metrics.py      ECE, Brier, AUROC, reliability bins, KL, Wilson, paired tests
  plots.py        reliability diagrams and curves
  tiers.py        the plan's five-grade rubric, as data
  predictions.py  the plan's 28 predictions, so they are scored mechanically
  smoke.py        Phase 0
  report.py       the markdown report
  generators/     3SAT, graph and program reachability, Sudoku, pairwise
                  ordering, DFA, Dyck words, a semantic control, filler and
                  dilution material, taxonomies, numeric scenes, and the
                  adversarial set
  experiments/    the nine experiments
tests/            unit tests for the hand-written core
tools/            report_template.html, prose.json and reportdata.json are the
                  report's sources; build_report.py assembles them and
                  verify_citations.py checks every figure against a measurement.
                  position_bias.py re-reads the raw logs for one figure the
                  experiments do not compute cleanly themselves
PROMPTING.md      the twelve rules, generated from the same measurements
runs/<id>/        report, figures, per-experiment results
```

Ground truth always comes from a solver or from construction, never from the model
and never from another model. Every problem is reproducible from its generator,
difficulty, seed and index. Every request, retry and failure is appended to JSONL
before any metric is computed, and a failed request is counted and excluded rather
than defaulted to 0.5, to false, or to the majority answer.

`jeveval/rescore.py` rejoins any experiment's answers to their correct answers and
recomputes accuracy and calibration while importing no experiment module, so a
scoring bug in an experiment cannot reproduce itself in the check. Derived figures
— a drift, a rate of decline — are rebuilt from the log only for the calibration
experiment, which is the largest remaining gap against the plan.

The raw logs are 1.3 GB and are not in this repository. The report, the figures,
the per-experiment results and the live measurements the guide cites are. The
written report is rebuilt from `tools/` by `build_report.py`, so a checkout can
regenerate it without an API key; re-running the experiments regenerates the logs.

## What the endpoint actually accepts

The plan was written from documentation and press coverage, and its description of
the request shape is wrong in three ways that changed experiments. Confirmed by
probing:

- The prompt field is `instructions`, not `question`. A `criteria` field also
  exists, typed per question kind: an object for `noul`, an object of
  `{option_id: description}` for `choice`, an ordered list for `score`.
- There is no request-level `instructions`; sending one is a 400. The only
  instruction channel is per-question.
- A `noul` answer is a bare probability with **no confidence field**. Only
  `choice` and `score` carry one, which is why the abstention result above is
  measured on those.

`choice` accepts at most 255 options; 256 is rejected. `score` returns an
undocumented `legend` mapping rubric indices to labels.

The documented context limits are 64k tokens per request and 32k for the state
plus the longest question, and both held up when measured. A request reporting
28,844 input tokens was answered and the next size up was refused with
`max_tokens_exceeded`; separately, 255 questions over an 8,000-token state
totalled 32,720 input tokens and was answered, which shows the 32k bound is on
the state plus one question rather than on the whole request. Worth knowing:
this harness's own token estimate ran about 20% low against the count the
endpoint reports, so budget against `usage.input_tokens` from a real response. Every probability
observed, across 3.19 million of them, lies exactly on a two-decimal grid, and the
value carrying the decision was exactly 1.0 on 52% of one experiment's answers.
`confidence` differs from the largest returned probability on 47% of choice and
score answers.

The endpoint limits on **input tokens, not requests**. It sustained about 145
requests per second at 1,645 tokens per request and about 6 per second at 26,527
tokens per request, with per-call latency unchanged in both. A requests-per-second
figure describes the request size you chose, not the service.

## A note on Guix

Built on Guix, which needs two accommodations. Create the virtualenv *with* system
site-packages: the Guix profile has working numpy, scipy and matplotlib, a broken
numpy in `~/.local` shadows them for the system interpreter, and a virtualenv
excludes user site-packages. Do not `pip install numpy` — the PyPI wheel cannot
find `libz.so.1` here.

HTTP uses `http.client` rather than httpx, because the Guix profile precedes the
virtualenv on `sys.path`, so a pip-installed package can be shadowed by an older
Guix copy of one of its dependencies. That is what broke httpx, by way of anyio
and typing_extensions. The only request this harness makes is a JSON POST.

## Licence and affiliation

MIT. This evaluation is independent, and is not affiliated with or endorsed by
TypeSafe.
