# Live measurements taken while writing the guide

These are the figures the prompting guide describes as "measured live", as
distinct from those drawn from the full run in `../full-20260919/`. Each was
produced against `jev-1.13.0` while the guide was being written, on sixty
problems per condition unless the file says otherwise.

| file | what it holds |
| --- | --- |
| `guide.json` | batching one subject against many, encoding, confidence thresholds, injection by technique, a trivially unsatisfiable formula |
| `abst.json` | confidence by kind of unanswerable state, with a verbatim example of each |
| `conf.json` | whether confidence predicts correctness, computed offline from the call logs over 34,200 answers |
| `ctx.json` | the state-size sweep across the documented 32k boundary, and a question-count sweep against the 64k per-request limit |
| `cal.json` | one satisfiable and one unsatisfiable formula with the probability returned for each |
| `encodings.json` | one program rendered as source, syntax tree and control-flow graph |
| `ticket-binding.json` | how a question should point at one of many subjects sharing a request: by position against by sender, over five blocks of sixty tickets |
| `semantic-grep.json` | a yes/no predicate asked of every line of a document, four ways; kept for where it puts its probabilities rather than for the predicate |

`ticket-binding.jsonl` and `semantic-grep.jsonl` are the only raw call logs kept here. The full run's logs
are 1.3 GB and are excluded; these are 1.4 MB and 1.5 MB, and keeping them means
`tools/ticket_binding.py --score-only` and `tools/semantic_grep.py` can
recompute those results from a checkout without an API key. They hold no
credentials: the client never writes the request headers.

They are kept so that every number in the guide traces to a file in this
repository. `tools/verify_citations.py` checks exactly that, and fails if a
figure in the prose cannot be matched to a value here or in the full run.
