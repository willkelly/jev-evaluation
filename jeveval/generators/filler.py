"""Filler material for E3 (batching) and E4 (input size).

E3 asks whether question 200 of a batch is answered as well as question 3. That
only means something if the other 199 questions are real questions with real
answers; a batch padded with nonsense would measure how the model handles
nonsense. E4 asks whether a known fact survives being buried in 50,000 tokens,
which needs 50,000 tokens of content that is plausible and says nothing about
the fact.

Both needs are served by one thing -- a synthetic maintenance log -- so it is
built once, here. `filler_block` turns the log into answerable questions whose
truth is known by construction. `dilution_corpus` renders it at a requested
size. `needle_series` inserts one relevant sentence into it at a requested
position, holding total size constant.

The module also satisfies the generator contract: `generate` emits the E4
dilution and needle-position instances, whose difficulty is state size and
needle position.

How the filler avoids answering the question it surrounds
---------------------------------------------------------
Every filler sentence is a claim about one invented entity -- a work order, a
sensor, an outage slot, a stock item, a procedure -- named by an identifier from
a prefix this project reserves (see RESERVED_PREFIXES). No filler sentence
states a general proposition, so nothing in it can entail or contradict a claim
about a 3SAT formula, a program, a graph or a support ticket. The vocabulary is
closed: every word the filler can emit comes from the lists in this module, so
no content is ever copied out of the state the filler is attached to.

Three mechanisms enforce that rather than leaving it to inspection:

- `_audit` runs over every template and vocabulary item at import and rejects
  any that contains an answer word ("true", "no", "accept") or an answer stem
  ("satisfiab", "reachab"). A filler sentence therefore cannot contain a token
  a grader would read as an answer to someone else's question.
- `collisions` reports any reserved prefix or proper name from this module that
  already occurs in a caller's state. E3 should call it before attaching a
  block. A shared name is the one way a filler question could become ambiguous,
  and it should fail at generation time instead of turning up later as an
  unexplained accuracy drop.
- Every filler question is anchored on an identifier ("work order WO-431105"),
  never on a bare attribute, so even a coincidental shared word leaves the
  question's referent unambiguous.

How a question avoids answering itself
--------------------------------------
Separately from not answering its neighbours, a question must not answer
itself. Two constructions here had that fault and were rebuilt:

- The needle's yes/no question shows a threshold and asks whether the needle
  beats it. Drawing the needle's duration first and the threshold around it --
  the obvious order -- makes low thresholds nearly always beatable and high
  ones nearly never, so a model that never finds the needle scores 0.86 from
  the question alone. Worse for E4, that 0.86 does not move with state size, so
  the retrieval curve comes out flat. The threshold is now drawn first, from
  one range whatever the answer, and the duration placed on the required side
  of it. The same applies to the filler's sensor question, whose threshold now
  sits within 2.0 units of the reading rather than within 20.
- The needle names a component, and the corpus keeps its own component
  mentions level, so the needle's component was the most-mentioned one in the
  assembled state and counting mentions answered the choice question 0.5 to 0.7
  of the time against a chance rate of 0.125. The corpus now emits the needle's
  component one time fewer (`dilution_corpus(level_component=...)`), which puts
  that baseline at chance from 1,000 tokens up. At 200 tokens it is about 0.3
  and cannot be helped -- a 200-token state names five components -- so every
  instance records whether the correct option was the most-mentioned one, and
  the smallest condition can be read with that held out.

Token counts
------------
Token counts here are estimates and are labelled as such everywhere they are
recorded. The estimator is `len(text) / 4` rounded -- the usual rule of thumb
for English under a BPE vocabulary. For the text this module emits -- ordinary
English sentences plus alphanumeric identifiers -- expect it within roughly 15%
of the true count. Identifiers like WO-431105 split into several tokens and
bias the estimate low; common English words bias it high. As a cross-check, a
word-based estimator (words / 0.75) reads about 8% lower than the
character-based one on this corpus at every size, which brackets the likely
error without settling it.

Only `usage.input_tokens` from the response is authoritative. Every instance
records `chars`, `words` and both estimates in meta, so the ratio can be
refitted against real usage after a run without regenerating anything, and a
state labelled 10,000 tokens can be relabelled with what it actually cost.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Sequence

from ..instances import Instance, choice, noul, rng_for, score, shuffled_options, shuffled_questions

NAME = "filler"


# --------------------------------------------------------------------------
# Closed vocabulary
# --------------------------------------------------------------------------

COMPONENTS: tuple[str, ...] = (
    "Larchmont pump array",
    "Calder feedwater line",
    "Tolman bearing housing",
    "Vance intake manifold",
    "Alder condenser bank",
    "Hollis actuator rack",
    "Dunmore seal cartridge",
    "Kestrel vent stack",
    "Palmer chiller loop",
    "Ridgeway drive coupling",
    "Oakhurst filter bank",
    "Sandhurst flow meter",
)

# Surnames, component heads and identifier prefixes are all chosen to be
# disjoint from every other generator's vocabulary. That was checked against
# the string literals of every module in the package, and `collisions` is what
# keeps it true as those modules change: E3 pairs a filler block with a state
# from one of them, and a shared name is the one way a filler question could
# become ambiguous. Names near-miss-equal to another module's were avoided too
# (no Okonkwo beside semantic.py's Okafor), since `collisions` matches whole
# words and would not report those.
TECHNICIANS: tuple[str, ...] = (
    "Brannigan",
    "Thorsby",
    "Velankar",
    "Quintero",
    "Ashgrove",
    "Merrowe",
    "Skagen",
    "Duvall",
    "Calloway",
    "Halvarsson",
)

DEPOTS: tuple[str, ...] = ("Granville", "Halden", "Ternbury", "Mossvale", "Keelbridge")

_MONTHS: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# Identifier prefixes this module owns. Nothing else in the project emits them,
# which is what lets `collisions` be a cheap and sufficient safety check.
RESERVED_PREFIXES: tuple[str, ...] = ("WO-", "SEN-", "OS-", "STK-", "PR-", "CR-")

# Words and stems that must never appear in filler. A filler sentence carrying
# one of these could be read as an answer to the surrounding question even
# though it is about an unrelated entity.
_FORBIDDEN_WORDS = frozenset({
    "yes", "no", "true", "false", "correct", "incorrect", "wrong", "answer",
    "valid", "invalid", "accept", "accepts", "accepted", "reject", "rejects",
    "rejected", "approve", "approved", "deny", "denied", "proof", "prove",
    "proven", "theorem", "lemma", "holds", "fails", "failure",
})
_FORBIDDEN_STEMS = ("satisfiab", "unsat", "reachab", "contradic", "tautolog")


def _audit(text: str, where: str) -> None:
    low = text.lower()
    for stem in _FORBIDDEN_STEMS:
        if stem in low:
            raise AssertionError(f"{where}: filler text contains answer stem {stem!r}: {text!r}")
    for word in re.findall(r"[a-z]+", low):
        if word in _FORBIDDEN_WORDS:
            raise AssertionError(f"{where}: filler text contains answer word {word!r}: {text!r}")


def collisions(text: str) -> list[str]:
    """Filler terms that already occur in `text`.

    A caller assembling an E3 batch should run this over its own state first.
    An empty list means no filler question can be confused with the state the
    filler is attached to; a non-empty one means a filler identifier or proper
    name is ambiguous and that state should be skipped or the filler reseeded.
    """
    found: set[str] = set()
    for prefix in RESERVED_PREFIXES:
        if re.search(re.escape(prefix) + r"\d", text):
            found.add(prefix)
    for term in COMPONENTS + TECHNICIANS + DEPOTS:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            found.add(term)
    return sorted(found)


# --------------------------------------------------------------------------
# Token estimation
# --------------------------------------------------------------------------

CHARS_PER_TOKEN = 4.0
WORDS_PER_TOKEN = 0.75


def estimate_tokens(text: str) -> int:
    """Primary estimator: characters / 4. See the module docstring on accuracy."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def estimate_tokens_by_words(text: str) -> int:
    """Secondary estimator: words / 0.75.

    Recorded alongside the primary one so a large disagreement between the two
    is visible in the log. It is not used to size anything.
    """
    return max(1, round(len(text.split()) / WORDS_PER_TOKEN))


# --------------------------------------------------------------------------
# Corpus construction
# --------------------------------------------------------------------------

_WORK_ORDER = "work_order"
_SENSOR = "sensor"
_DILUTION_KINDS: tuple[str, ...] = (
    _WORK_ORDER, _SENSOR, "outage", "stock", "procedure", "note", "count",
)
# Filler questions are only asked about these two kinds, so a block is built
# from them alone; every sentence in a block is then one a question can target.
_BLOCK_KINDS: tuple[str, ...] = (_WORK_ORDER, _SENSOR)

_DURATION_BUCKETS: tuple[dict, ...] = (
    {"id": "lt30", "label": "under 30 minutes", "lo": 5, "hi": 30},
    {"id": "m30_89", "label": "30 to 89 minutes", "lo": 30, "hi": 90},
    {"id": "m90_239", "label": "90 to 239 minutes", "lo": 90, "hi": 240},
    {"id": "ge240", "label": "240 minutes or more", "lo": 240, "hi": 601},
)
DURATION_RUBRIC: list[dict] = [{"id": b["id"], "label": b["label"]} for b in _DURATION_BUCKETS]

_TEMPLATES: dict[str, str] = {
    _WORK_ORDER: (
        "Work order {id} was opened by {technician} on {month} {day} and concerns "
        "the {component}; it ran for {minutes} minutes under shift {shift}."
    ),
    _SENSOR: (
        "Sensor {id} on the {component} reported a mean of {reading} units over "
        "{article} {window}-day window ending in {month}."
    ),
    "outage": (
        "Outage slot {id} is scheduled for the {component} in {month} and releases "
        "{days} days of downtime."
    ),
    "stock": (
        "Stock item {id} is held at {depot} depot with a reorder threshold of "
        "{units} units and a lead time of {lead} days."
    ),
    "procedure": (
        "Procedure {id} requires a walkdown of the {component} before any lift over "
        "{kilograms} kilograms, and lists {technician} as the current owner."
    ),
    "note": "Procedure {id} was last reviewed in {month} by {technician}.",
    "count": "{depot} depot logged {units} walkdowns in {month}.",
}

_ID_SPECS: dict[str, tuple[str, int, int, int]] = {
    _WORK_ORDER: ("WO-", 6, 100000, 800000),
    _SENSOR: ("SEN-", 5, 10000, 80000),
    "outage": ("OS-", 4, 1000, 8000),
    "stock": ("STK-", 5, 10000, 80000),
    "procedure": ("PR-", 4, 1000, 8000),
    "note": ("PR-", 4, 1000, 8000),
}

# Kinds whose sentence names a component, and kinds whose sentence names a
# technician. Only these advance the corresponding cycle, which is what keeps
# the mentions balanced (see _Cycle).
_COMPONENT_KINDS = frozenset({_WORK_ORDER, _SENSOR, "outage", "procedure"})
_TECHNICIAN_KINDS = frozenset({_WORK_ORDER, "procedure", "note"})


class _Cycle:
    """Draws from a pool without replacement, reshuffling once it empties.

    Uniform independent draws would leave the mention counts of the twelve
    components unequal by a few, and a choice question whose correct option is
    also the most-mentioned one would then be answerable by counting. Cycling
    keeps every count within 1 of every other, which removes that cue by
    construction rather than leaving it to chance.

    `peek` exists because the corpus builder generates candidate sentences it
    then throws away. A discarded candidate must not advance the cycle -- if it
    does, the cycle balances the draws rather than the text, and the mention
    counts of the text are neither balanced nor equal to what gets recorded.
    So the builder peeks the component, renders candidates against it, and
    calls `take` only for the candidate it keeps.

    `charged` pre-spends one draw on an item the caller will mention itself.
    E4's needle names its component once, outside the corpus; without this the
    needle's component ends up mentioned one more time than every other, and
    the correct option is the most-mentioned one in the assembled state, which
    is the exact cue the cycling was introduced to remove.
    """

    def __init__(
        self, pool: Sequence[str], rng: random.Random, *, charged: str | None = None
    ) -> None:
        self._pool = tuple(pool)
        self._rng = rng
        self._bag: list[str] = []
        if charged is not None:
            if charged not in self._pool:
                raise ValueError(f"{charged!r} is not in the pool")
            self._bag = [item for item in self._pool if item != charged]
            rng.shuffle(self._bag)

    def peek(self) -> str:
        """The item `take` would return, without consuming it."""
        if not self._bag:
            self._bag = list(self._pool)
            self._rng.shuffle(self._bag)
        return self._bag[-1]

    def take(self) -> str:
        item = self.peek()
        self._bag.pop()
        return item

    # `next` is `take`; kept as the name the phase-1 loop reads best with.
    next = take


class _Writer:
    """Emits log entries. Identifiers are sequential from a random base so that
    no two entries in one corpus can share an id -- a duplicate id would make a
    filler question ambiguous, and random draws collide often enough to matter
    across thousands of instances."""

    def __init__(self, rng: random.Random, *, charged_component: str | None = None) -> None:
        self.rng = rng
        self.components = _Cycle(COMPONENTS, rng, charged=charged_component)
        self.technicians = _Cycle(TECHNICIANS, rng)
        self._next: dict[str, int] = {}
        self._stride: dict[str, int] = {}

    def _ident(self, kind: str) -> str:
        prefix, width, lo, hi = _ID_SPECS[kind]
        if prefix not in self._next:
            self._next[prefix] = self.rng.randrange(lo, hi)
            self._stride[prefix] = self.rng.randrange(1, 6)
        value = self._next[prefix]
        self._next[prefix] = value + self._stride[prefix]
        return f"{prefix}{value:0{width}d}"

    def entry(
        self, kind: str, *, component: str | None = None, technician: str | None = None
    ) -> dict:
        """One log entry.

        `component` and `technician` may be supplied by a caller that has
        already peeked them, so that a candidate entry which is generated and
        then discarded does not advance the cycles. Passing them does not
        consume a draw; leaving them out does.
        """
        rng = self.rng
        e: dict[str, Any] = {"kind": kind, "month": rng.choice(_MONTHS)}
        if kind in _ID_SPECS:
            e["id"] = self._ident(kind)
        if kind in _COMPONENT_KINDS:
            e["component"] = component if component is not None else self.components.next()
        if kind == _WORK_ORDER:
            bucket = _DURATION_BUCKETS[rng.randrange(len(_DURATION_BUCKETS))]
            e["technician"] = technician if technician is not None else self.technicians.next()
            e["minutes"] = rng.randrange(bucket["lo"], bucket["hi"])
            e["duration_bucket"] = bucket["id"]
            e["shift"] = rng.randrange(1, 4)
            e["day"] = rng.randrange(1, 29)
        elif kind == _SENSOR:
            e["reading"] = rng.randrange(150, 950) / 10.0
            e["window"] = rng.randrange(3, 31)
            e["article"] = _article(e["window"])
        elif kind == "outage":
            e["days"] = rng.randrange(1, 12)
        elif kind == "stock":
            e["depot"] = rng.choice(DEPOTS)
            e["units"] = rng.randrange(12, 900)
            e["lead"] = rng.randrange(2, 60)
        elif kind == "procedure":
            e["technician"] = technician if technician is not None else self.technicians.next()
            e["kilograms"] = rng.randrange(5, 400)
        elif kind == "note":
            e["technician"] = technician if technician is not None else self.technicians.next()
        elif kind == "count":
            e["depot"] = rng.choice(DEPOTS)
            e["units"] = rng.randrange(3, 90)
        return e


def _article(n: int) -> str:
    """"a" or "an" for a number read aloud. The filler has to read as real log
    text -- the plan asks for plausible, and "a 11-day window" is not."""
    return "an" if str(n)[0] == "8" or str(n).startswith(("11", "18")) else "a"


def _render(entry: dict) -> str:
    return _TEMPLATES[entry["kind"]].format(**entry)


@dataclass(frozen=True)
class Corpus:
    """A rendered block of filler, plus the facts a baseline would want.

    `component_counts` is here because the obvious cheap answer to "which
    component is X scheduled for" is "whichever component this state talks about
    most". Recording the counts lets that baseline be computed offline instead
    of assumed away. It counts the components of the entries actually rendered
    into `paragraphs`, so it is what a reader would get by counting the text --
    not what the cycle handed out, which includes candidates the size fit
    generated and discarded.
    """

    paragraphs: list[str]
    entries: list[dict]
    component_counts: dict[str, int]
    chars: int

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


# Phase 1 fills to within this many characters of the target, then a best-fit
# pass closes the gap with whole sentences and a reference tag closes the rest.
_TAIL_MARGIN_CHARS = 260
_TAIL_POOL = 24
_TAIL_STEPS = 64


def _tag(width: int, rng: random.Random) -> str:
    """A reference tag of exactly `width` characters, or "" if none fits.

    The last sentence carries this so a corpus can land on an exact character
    budget. Whole sentences are too coarse -- the shortest is around 40
    characters, which is 10 estimated tokens of error on a 200-token state.
    """
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    if width >= 8:
        body = "".join(rng.choice(alphabet) for _ in range(width - 7))
        return f" [ref {body}]"
    if width >= 4:
        body = "".join(rng.choice(alphabet) for _ in range(width - 3))
        return f" ({body})"
    return ""


def _build(
    *,
    target_chars: int,
    rng: random.Random,
    kinds: Sequence[str],
    level_for: str | None = None,
) -> Corpus:
    """Render entries until the text is exactly `target_chars` long.

    `level_for` names a component the caller will mention once itself, outside
    this corpus; it is emitted one time fewer here so that the counts in the
    assembled text come out level. See `_Cycle`.
    """
    writer = _Writer(rng, charged_component=level_for)
    paragraphs: list[list[str]] = []
    entries: list[dict] = []
    chars = 0
    para_len = 0
    para_target = rng.randrange(3, 7)
    cursor = 0

    def append(sentence: str, entry: dict | None) -> None:
        nonlocal chars, para_len, para_target
        if not paragraphs or para_len >= para_target:
            paragraphs.append([])
            para_len = 0
            para_target = rng.randrange(3, 7)
            chars += 2 if len(paragraphs) > 1 else 0
        else:
            chars += 1
        paragraphs[-1].append(sentence)
        chars += len(sentence)
        para_len += 1
        if entry is not None:
            entries.append(entry)

    while target_chars - chars > _TAIL_MARGIN_CHARS:
        kind = kinds[cursor % len(kinds)]
        cursor += 1
        entry = writer.entry(kind)
        append(_render(entry), entry)

    # Best fit: repeatedly take the longest candidate that still fits, so the
    # corpus never overshoots the budget and the residue is closed by the tag.
    for _ in range(_TAIL_STEPS):
        # One peeked component and technician for the whole candidate pool, so
        # that a candidate which loses does not consume a draw and so that the
        # longest-fitting choice is not also a vote for the longest component
        # name. The winner consumes them; the losers are discarded.
        component = writer.components.peek()
        technician = writer.technicians.peek()
        candidates = []
        for offset in range(_TAIL_POOL):
            kind = kinds[(cursor + offset) % len(kinds)]
            entry = writer.entry(kind, component=component, technician=technician)
            candidates.append((entry, _render(entry)))
        best = None
        for entry, sentence in candidates:
            if chars + 1 + len(sentence) <= target_chars:
                if best is None or len(sentence) > len(best[1]):
                    best = (entry, sentence)
        if best is None:
            break
        if best[0]["kind"] in _COMPONENT_KINDS:
            writer.components.take()
        if best[0]["kind"] in _TECHNICIAN_KINDS:
            writer.technicians.take()
        cursor += 1
        append(best[1], best[0])

    if paragraphs:
        tag = _tag(target_chars - chars, rng)
        if tag:
            last = paragraphs[-1][-1]
            paragraphs[-1][-1] = last[:-1] + tag + last[-1]
            chars += len(tag)

    rendered = [" ".join(para) for para in paragraphs]
    return Corpus(
        paragraphs=rendered,
        entries=entries,
        component_counts=_component_counts(entries),
        chars=chars,
    )


def _component_counts(entries: Sequence[dict]) -> dict[str, int]:
    """Component mentions in `entries`. One mention per entry that names one."""
    counts = {component: 0 for component in COMPONENTS}
    for entry in entries:
        if "component" in entry:
            counts[entry["component"]] += 1
    return counts


MIN_TOKENS = 50


def dilution_corpus(
    *,
    target_tokens: int,
    seed: int,
    index: int = 0,
    reserve_chars: int = 0,
    level_component: str | None = None,
) -> Corpus:
    """Filler sized so that `reserve_chars` plus the corpus estimates to
    `target_tokens`.

    `reserve_chars` is how much of the budget the caller will spend on its own
    content -- for `needle_series`, the needle sentence and its separator. The
    corpus rng is keyed on `target_tokens` only, deliberately excluding needle
    position, so all five positions of one instance share one corpus and the
    total size is identical across them.

    `level_component` is a component the caller's own content mentions once. It
    is emitted one time fewer here, so counting component mentions in the
    assembled state does not single it out. A caller inserting a fact that
    names a component should pass it; leaving it out leaves the fact's
    component mentioned once more than every other, which makes "the component
    this state mentions most" a 0.5-to-0.7 accurate answer to a question whose
    chance rate is 0.125.
    """
    if target_tokens < MIN_TOKENS:
        raise ValueError(f"target_tokens must be at least {MIN_TOKENS}, got {target_tokens}")
    target_chars = round(target_tokens * CHARS_PER_TOKEN) - reserve_chars
    if target_chars < 120:
        raise ValueError(
            f"reserve_chars={reserve_chars} leaves {target_chars} characters for filler"
        )
    rng = rng_for(NAME, {"tokens": int(target_tokens), "part": "corpus"}, seed, index)
    return _build(
        target_chars=target_chars,
        rng=rng,
        kinds=_DILUTION_KINDS,
        level_for=level_component,
    )


def dilution_text(*, target_tokens: int, seed: int, index: int = 0) -> str:
    """Filler text whose character-based estimate is `target_tokens`.

    Accuracy of the fit, not of the estimator: the returned text estimates to
    within 1 token of the request. The estimator's own error against real
    tokenization is the ~15% discussed in the module docstring.
    """
    return dilution_corpus(target_tokens=target_tokens, seed=seed, index=index).text


# --------------------------------------------------------------------------
# Needle placement (E4 item 2)
# --------------------------------------------------------------------------

NEEDLE_POSITIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Placement:
    """One assembled state with the needle at one position."""

    text: str
    position: float
    realized_position: float
    paragraph_index: int
    n_paragraphs: int


def place_needle(*, paragraphs: Sequence[str], fact: str, position: float) -> Placement:
    """Insert `fact` as its own paragraph at `position` of the way through.

    Total length is the same for every position: the same paragraphs and the
    same separator count are used whatever the insertion index. That is the
    whole point of doing it this way -- if size moved with position, a position
    effect and a size effect would be indistinguishable.

    Paragraphs have unequal lengths, so the insertion index that best matches
    the requested fraction is chosen and the fraction actually achieved is
    reported as `realized_position`.
    """
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"position must be in [0, 1], got {position}")
    if not paragraphs:
        raise ValueError("cannot place a needle in an empty corpus")
    lengths = [len(p) for p in paragraphs]
    total = sum(lengths) + 2 * (len(paragraphs) - 1)
    prefix = 0
    fractions = []
    for i in range(len(paragraphs) + 1):
        fractions.append(prefix / total if total else 0.0)
        if i < len(paragraphs):
            prefix += lengths[i] + (2 if i < len(paragraphs) - 1 else 0)
    # Anchor the endpoints exactly: 0.0 is before everything, 1.0 after.
    fractions[-1] = 1.0
    best = min(range(len(fractions)), key=lambda i: (abs(fractions[i] - position), i))
    parts = list(paragraphs[:best]) + [fact] + list(paragraphs[best:])
    return Placement(
        text="\n\n".join(parts),
        position=position,
        realized_position=fractions[best],
        paragraph_index=best,
        n_paragraphs=len(paragraphs),
    )


def needle_series(
    *,
    fact: str,
    target_tokens: int,
    seed: int,
    index: int = 0,
    positions: Sequence[float] = NEEDLE_POSITIONS,
    level_component: str | None = None,
) -> list[Placement]:
    """The same fact in the same filler at several positions, all one size.

    Returned together rather than one call per position so that sharing the
    corpus is structural and cannot be got wrong by a caller that varies a seed
    along with the position.

    Pass `level_component` when `fact` names a component; see
    `dilution_corpus`.
    """
    corpus = dilution_corpus(
        target_tokens=target_tokens,
        seed=seed,
        index=index,
        reserve_chars=2 + len(fact),
        level_component=level_component,
    )
    return [place_needle(paragraphs=corpus.paragraphs, fact=fact, position=p) for p in positions]


# --------------------------------------------------------------------------
# The needle itself
# --------------------------------------------------------------------------

NEEDLE_OPTIONS = 8

# The needle's downtime question shows a threshold and asks whether the needle
# beats it. These bounds keep `days` in 2..14 -- plural, plausible, and small
# enough to be read off the sentence -- while letting the threshold be drawn
# from the same range whichever way the answer goes. A gap of 0 makes the
# comparison exact ("more than 7" against 7 days); the largest gap leaves three
# days of slack, so retrieving the number approximately is not always enough
# and is not always useless either.
_NEEDLE_THRESHOLD_LO = 4
_NEEDLE_THRESHOLD_HI = 11
_NEEDLE_MAX_GAP = 2


@dataclass(frozen=True)
class Needle:
    """One relevant fact and the questions that retrieve it.

    Two questions, about two different attributes of the one sentence. Two
    rather than one because E4 reports ECE as well as accuracy and ECE needs a
    probability, which only a noul returns. Different attributes because two
    questions about the same attribute in one call would let enrollment answer
    one from the other, which is E5's subject and not E4's.

    The count is fixed at two across every condition so that E4's size and
    position curves are not also a batch-size curve.
    """

    cr_id: str
    component: str
    days: int
    threshold: int
    sentence: str
    questions: dict[str, dict]
    truth: dict[str, Any]
    options: list[str]


def make_needle(*, target_tokens: int, seed: int, index: int) -> Needle:
    """Build the needle for instance `index`.

    Keyed on `target_tokens`, `seed` and `index` only -- not on position -- so
    that the five position conditions of one instance carry byte-identical
    questions and differ solely in where the sentence sits.
    """
    rng = rng_for(NAME, {"tokens": int(target_tokens), "part": "needle"}, seed, index)
    cr_id = f"CR-{rng.randrange(10000, 99999)}"
    component = rng.choice(COMPONENTS)
    # Alternate the noul's truth by index so the majority-class baseline is 0.5
    # exactly, rather than 0.5 in expectation with a condition-sized wobble.
    want_true = index % 2 == 0
    # Draw the threshold -- the only number the question itself shows -- from
    # one distribution whatever the answer, then place `days` on the required
    # side of it. Drawing `days` first and the threshold around it, which is
    # the obvious way round, makes the threshold itself the answer: a low
    # threshold has to be beaten and a high one cannot be, so a model that
    # never finds the needle scores 0.86 by reading the question alone, and
    # E4's retrieval curve comes out flat because that 0.86 does not care how
    # big the state is. Here P(true | threshold) is 0.5 at every threshold.
    threshold = rng.randrange(_NEEDLE_THRESHOLD_LO, _NEEDLE_THRESHOLD_HI + 1)
    gap = rng.randrange(0, _NEEDLE_MAX_GAP + 1)
    days = threshold + 1 + gap if want_true else threshold - gap
    month = rng.choice(_MONTHS)
    day = rng.randrange(1, 29)
    sentence = (
        f"Change request {cr_id} is scheduled for the {component}, requests {days} days "
        f"of downtime, and sets its walkdown for {month} {day}."
    )

    distractors = [c for c in COMPONENTS if c != component]
    rng.shuffle(distractors)
    options = [component] + distractors[: NEEDLE_OPTIONS - 1]

    questions = {
        "needle_component": shuffled_options(
            choice(f"Which component is change request {cr_id} scheduled for?", options), rng
        ),
        "needle_downtime": noul(
            f"Does change request {cr_id} request more than {threshold} days of downtime?"
        ),
    }
    truth: dict[str, Any] = {
        "needle_component": component,
        "needle_downtime": days > threshold,
    }
    return Needle(
        cr_id=cr_id,
        component=component,
        days=days,
        threshold=threshold,
        sentence=sentence,
        questions=shuffled_questions(questions, rng),
        truth=truth,
        options=options,
    )


# --------------------------------------------------------------------------
# Filler questions (E3 item 1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FillerBlock:
    """Filler text plus questions about it, with truth by construction."""

    text: str
    records: list[dict]
    questions: dict[str, dict]
    truth: dict[str, Any]
    meta: dict

    def __len__(self) -> int:
        return len(self.questions)

    @property
    def state_records(self) -> list[dict]:
        """The records with derived fields dropped, for sending as state.

        `duration_bucket` is the answer to the score question. It stays on
        `records` because the self-check and any offline audit want it, and it
        must never reach the state, so the split is made here rather than left
        to each caller.
        """
        drop = ("kind", "duration_bucket")
        return [{k: v for k, v in r.items() if k not in drop} for r in self.records]


def _probe(actual: str, pool: Sequence[str], want_true: bool, rng: random.Random) -> str:
    if want_true:
        return actual
    others = [x for x in pool if x != actual]
    return others[rng.randrange(len(others))]


def _wo_question(
    entry: dict, template: int, rng: random.Random, want_true: bool | None
) -> tuple[dict, Any]:
    wid = entry["id"]
    if template == 0:
        who = _probe(entry["technician"], TECHNICIANS, bool(want_true), rng)
        return noul(f"Did {who} open work order {wid}?"), who == entry["technician"]
    if template == 1:
        pool = [c for c in COMPONENTS if c != entry["component"]]
        rng.shuffle(pool)
        opts = [entry["component"]] + pool[: NEEDLE_OPTIONS - 1]
        q = shuffled_options(choice(f"Which component does work order {wid} concern?", opts), rng)
        return q, entry["component"]
    if template == 2:
        q = score(f"How long did work order {wid} take?", DURATION_RUBRIC)
        return q, entry["duration_bucket"]
    shift = _probe(str(entry["shift"]), ("1", "2", "3"), bool(want_true), rng)
    return noul(f"Was work order {wid} logged under shift {shift}?"), int(shift) == entry["shift"]


def _sen_question(
    entry: dict, template: int, rng: random.Random, want_true: bool | None
) -> tuple[dict, Any]:
    sid = entry["id"]
    if template == 0:
        pool = [c for c in COMPONENTS if c != entry["component"]]
        rng.shuffle(pool)
        opts = [entry["component"]] + pool[: NEEDLE_OPTIONS - 1]
        q = shuffled_options(choice(f"Which component is sensor {sid} mounted on?", opts), rng)
        return q, entry["component"]
    reading = entry["reading"]
    # A narrow span, for the reason given at _NEEDLE_THRESHOLD_LO: the
    # threshold is the only number the question shows, so if it sits far below
    # the reading when the answer is yes and far above it when the answer is
    # no, its absolute value answers the question. Readings span 15.0 to 94.9;
    # a threshold within 2.0 of one is drawn from very nearly the same
    # distribution either way, and a single cut on it scores 0.52 rather than
    # the 0.65 a +/-20 span allowed.
    span = rng.randrange(1, _SENSOR_MAX_SPAN_TENTHS + 1) / 10.0
    threshold = round(reading - span, 1) if want_true else round(reading + span, 1)
    q = noul(f"Did sensor {sid} report a mean above {threshold} units?")
    return q, reading > threshold


# Tenths of a unit; see the comment in _sen_question.
_SENSOR_MAX_SPAN_TENTHS = 20

_WO_TEMPLATES = 4
_SEN_TEMPLATES = 2
_TEMPLATES_PER_RECORD = _WO_TEMPLATES + _SEN_TEMPLATES
_MIN_RECORDS_PER_KIND = 16

# Which (kind, template) pairs produce a noul. The caller needs to know before
# it builds the question, so that it can hand down a polarity that alternates
# and the block's yes/no answers come out balanced exactly rather than in
# expectation. A block whose fillers are 70% yes would let a model that answers
# yes to everything look like it read them.
_IS_NOUL: dict[tuple[str, int], bool] = {
    ("wo", 0): True, ("wo", 1): False, ("wo", 2): False, ("wo", 3): True,
    ("sen", 0): False, ("sen", 1): True,
}


def filler_block(*, seed: int, count: int, index: int = 0, key_prefix: str = "f") -> FillerBlock:
    """`count` distinct, answerable questions about one generated log.

    The log is sized so that six question templates over equal numbers of work
    orders and sensors cover `count` without ever asking the same question
    twice: 255 questions -- E3's position 200 plus E7's 255-option batches --
    need 43 records of each kind, and larger counts scale the log rather than
    repeating.

    Slots are taken one per template round before a second is taken from any,
    then shuffled, so even a ten-question block carries all three question
    types rather than ten copies of whichever template came first.
    """
    if count < 1:
        raise ValueError("count must be positive")
    per_kind = max(_MIN_RECORDS_PER_KIND, -(-count // _TEMPLATES_PER_RECORD))
    rng = rng_for(NAME, {"fillers": int(count), "part": "block"}, seed, index)

    writer = _Writer(rng)
    records: list[dict] = []
    for i in range(per_kind):
        records.append(writer.entry(_WORK_ORDER))
        records.append(writer.entry(_SENSOR))
    rng.shuffle(records)

    work_orders = [r for r in records if r["kind"] == _WORK_ORDER]
    sensors = [r for r in records if r["kind"] == _SENSOR]
    rounds: list[list[tuple[dict, str, int]]] = []
    for t in range(_WO_TEMPLATES):
        rounds.append([(r, "wo", t) for r in work_orders])
    for t in range(_SEN_TEMPLATES):
        rounds.append([(r, "sen", t) for r in sensors])
    capacity = sum(len(rd) for rd in rounds)
    if capacity < count:  # pragma: no cover - per_kind is sized to prevent this
        raise ValueError(f"capacity {capacity} below requested count {count}")

    # Take one slot from each template round before taking a second from any,
    # so a small block gets a mix of the three question types rather than
    # `count` copies of whichever template happens to come first.
    slots: list[tuple[dict, str, int]] = []
    depth = 0
    while len(slots) < count:
        for rd in rounds:
            if depth < len(rd):
                slots.append(rd[depth])
                if len(slots) == count:
                    break
        depth += 1
    rng.shuffle(slots)

    questions: dict[str, dict] = {}
    truth: dict[str, Any] = {}
    noul_seq = 0
    for i, (record, kind, template) in enumerate(slots):
        want_true: bool | None = None
        if _IS_NOUL[(kind, template)]:
            want_true = noul_seq % 2 == 0
            noul_seq += 1
        if kind == "wo":
            q, t = _wo_question(record, template, rng, want_true)
        else:
            q, t = _sen_question(record, template, rng, want_true)
        key = f"{key_prefix}{i:03d}"
        questions[key] = q
        truth[key] = t

    paragraphs: list[list[str]] = []
    for i, record in enumerate(records):
        if i % 4 == 0:
            paragraphs.append([])
        paragraphs[-1].append(_render(record))
    text = "\n\n".join(" ".join(p) for p in paragraphs)

    nouls = [v for k, v in truth.items() if questions[k]["type"] == "noul"]
    meta = {
        "n_questions": len(questions),
        "n_records": len(records),
        "chars": len(text),
        "est_tokens": estimate_tokens(text),
        "est_tokens_by_words": estimate_tokens_by_words(text),
        "token_estimator": "chars/4",
        "noul_true_fraction": (sum(nouls) / len(nouls)) if nouls else None,
        "type_counts": {
            t: sum(1 for q in questions.values() if q["type"] == t)
            for t in ("noul", "choice", "score")
        },
        "component_counts": _component_counts(records),
    }
    return FillerBlock(text=text, records=records, questions=questions, truth=truth, meta=meta)


def _check_free(state: Any) -> None:
    hits = collisions(state if isinstance(state, str) else json.dumps(state, default=str))
    if hits:
        raise ValueError(
            "state already contains filler vocabulary "
            f"{hits}; a filler question about it would be ambiguous"
        )


def attach(state: Any, block: FillerBlock, *, key: str = "maintenance_log") -> Any:
    """Put a block's text into a state of whatever shape the generator used.

    Silently dropping the block would leave E3 asking unanswerable questions and
    reading the result as positional decay, so every shape is handled explicitly
    and an unknown one raises.

    The collision check runs here rather than being left to the caller. It is
    the only enforcement that survives the other generators changing their own
    vocabularies, and a shared name should stop an E3 condition at generation
    time instead of showing up in the results as positional decay that is not
    there.
    """
    _check_free(state)
    if isinstance(state, str):
        return f"{state}\n\n{block.text}"
    if isinstance(state, dict):
        if key in state:
            raise ValueError(f"state already has a {key!r} key; pass a different key=")
        return {**state, key: block.state_records}
    if isinstance(state, list):
        return list(state) + [{key: block.state_records}]
    raise TypeError(f"cannot attach filler to a state of type {type(state).__name__}")


@dataclass(frozen=True)
class Batch:
    """A filler block with one target question at a known position."""

    questions: dict[str, dict]
    truth: dict[str, Any]
    target_key: str
    position: int
    meta: dict


def batch_with_target(
    *,
    block: FillerBlock,
    target: dict,
    target_truth: Any,
    position: int,
    key_format: str = "q{:03d}",
) -> Batch:
    """Insert `target` at 1-indexed `position` among a block's questions.

    Every key is renumbered to one uniform format including the target's. The
    plan forbids question ids from encoding the answer; a target that kept a
    distinctive key while the filler used another would go further and mark
    which question is being scored, and any position effect measured that way
    would be uninterpretable.
    """
    total = len(block.questions) + 1
    if not 1 <= position <= total:
        raise ValueError(f"position must be in 1..{total}, got {position}")
    items: list[tuple[dict, Any]] = [(block.questions[k], block.truth[k]) for k in block.questions]
    idx = position - 1
    items.insert(idx, (target, target_truth))

    questions: dict[str, dict] = {}
    truth: dict[str, Any] = {}
    target_key = ""
    for i, (q, t) in enumerate(items):
        key = key_format.format(i)
        questions[key] = q
        truth[key] = t
        if i == idx:
            target_key = key
    meta = dict(block.meta)
    meta.update({"n_questions": total, "target_position": position, "target_key": target_key})
    return Batch(
        questions=questions, truth=truth, target_key=target_key, position=position, meta=meta
    )


# --------------------------------------------------------------------------
# Generator contract: the E4 dilution and needle-position instances
# --------------------------------------------------------------------------

DILUTION_TOKENS: tuple[int, ...] = (200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
DILUTION_POSITION = 0.0
POSITION_SWEEP_TOKENS = 10000


def difficulty_sweep() -> list[dict]:
    """Size ladder at a fixed position, then a position ladder at a fixed size.

    The sizes are E4's stated range, 200 to 50,000, roughly doubling so the knee
    can be located to within a factor of two wherever it falls, with points at
    5k and 10k because that is where E4 draws its tier boundaries.

    The two arms are separate on purpose. Varying size and position together
    would leave a size effect and a position effect indistinguishable, which is
    the one thing E4 item 2 exists to separate. Dilution is measured with the
    fact first, the favourable end, so the size curve does not have a
    middle-of-state penalty folded into it; position is then measured at 10,000
    tokens, where E4's position tiers are set and where the corpus has enough
    paragraphs for 25% steps to be meaningful. The (10000, 0.0) condition serves
    both arms and is listed once.
    """
    sweep = [{"tokens": t, "position": DILUTION_POSITION} for t in DILUTION_TOKENS]
    sweep.extend(
        {"tokens": POSITION_SWEEP_TOKENS, "position": p}
        for p in NEEDLE_POSITIONS
        if not (p == DILUTION_POSITION and POSITION_SWEEP_TOKENS in DILUTION_TOKENS)
    )
    return sweep


def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]:
    tokens = int(difficulty["tokens"])
    position = float(difficulty["position"])
    out: list[Instance] = []
    for i in range(count):
        needle = make_needle(target_tokens=tokens, seed=seed, index=i)
        corpus = dilution_corpus(
            target_tokens=tokens,
            seed=seed,
            index=i,
            reserve_chars=2 + len(needle.sentence),
            level_component=needle.component,
        )
        placement = place_needle(
            paragraphs=corpus.paragraphs, fact=needle.sentence, position=position
        )
        state = placement.text

        # Mentions in the assembled state, needle included -- what an offline
        # frequency baseline would actually count. Recording the corpus counts
        # alone would understate the correct option by exactly one and make the
        # baseline look weaker than it is.
        option_counts = {
            opt: corpus.component_counts.get(opt, 0) + (1 if opt == needle.component else 0)
            for opt in needle.options
        }
        best = max(option_counts.values())
        top = sorted(o for o in option_counts if option_counts[o] == best)
        meta = {
            "target_tokens": tokens,
            "est_tokens": estimate_tokens(state),
            "est_tokens_by_words": estimate_tokens_by_words(state),
            "token_estimator": "chars/4",
            "chars": len(state),
            "words": len(state.split()),
            "nominal_position": position,
            "realized_position": placement.realized_position,
            "paragraph_index": placement.paragraph_index,
            "n_paragraphs": placement.n_paragraphs,
            "n_sentences": len(corpus.entries),
            "needle_id": needle.cr_id,
            "needle_occurrences": state.count(needle.cr_id),
            # Baseline features, recorded so that both cheap heuristics can be
            # scored offline rather than argued about. For the choice question
            # it is "the option this state mentions most", which the leveling
            # in `dilution_corpus` holds to chance from 1,000 tokens up but
            # which still reaches about 0.3 at 200, where only a handful of
            # components are named at all. For the noul it is "guess from the
            # threshold", which `make_needle` holds to chance at every size.
            "n_options": len(needle.options),
            "option_mention_counts": option_counts,
            "most_mentioned_options": top,
            "most_mentioned_option": top[0],
            "most_mentioned_is_correct": needle.component in top,
            "option_mentions_tied": len(set(option_counts.values())) == 1,
            "downtime_threshold": needle.threshold,
            "needle_days": needle.days,
            "n_questions": len(needle.questions),
        }
        inst = Instance(
            generator=NAME,
            difficulty={"tokens": tokens, "position": position},
            seed=seed,
            index=i,
            state=state,
            questions=needle.questions,
            truth=needle.truth,
            meta=meta,
        )
        inst.validate()
        out.append(inst)
    return out


# --------------------------------------------------------------------------
# Import-time audit of the static vocabulary
# --------------------------------------------------------------------------

def _audit_all() -> None:
    for name, items in (
        ("COMPONENTS", COMPONENTS),
        ("TECHNICIANS", TECHNICIANS),
        ("DEPOTS", DEPOTS),
        ("_MONTHS", _MONTHS),
    ):
        for item in items:
            _audit(item, name)
    for kind, template in _TEMPLATES.items():
        _audit(re.sub(r"\{[a-z_]+\}", " ", template), f"_TEMPLATES[{kind!r}]")
    for bucket in _DURATION_BUCKETS:
        _audit(bucket["label"], "_DURATION_BUCKETS")


_audit_all()


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

def _selfcheck() -> None:
    seed = 20260919

    print(
        f"{'tokens':>7} {'pos':>5} {'est':>7} {'err':>5} {'chars':>8} {'paras':>6} "
        f"{'realized':>9} {'est/word':>8}"
    )
    print("-" * 61)

    for d in difficulty_sweep():
        insts = generate(difficulty=d, seed=seed, count=2)
        assert len(insts) == 2
        for inst in insts:
            state = inst.state
            m = inst.meta
            n = make_needle(target_tokens=d["tokens"], seed=seed, index=inst.index)

            # The needle is present exactly once, by sentence and by identifier.
            assert state.count(n.sentence) == 1, f"needle sentence x{state.count(n.sentence)}"
            assert state.count(n.cr_id) == 1, f"needle id x{state.count(n.cr_id)}"
            assert m["needle_occurrences"] == 1

            # Ground truth is the construction, not a reading of the text.
            assert inst.truth["needle_component"] == n.component
            assert inst.truth["needle_downtime"] == (n.days > n.threshold)
            assert isinstance(inst.truth["needle_downtime"], bool)
            opts = {o["id"] for o in inst.questions["needle_component"]["options"]}
            assert n.component in opts and len(opts) == NEEDLE_OPTIONS

            # Size fit and estimator bookkeeping.
            err = m["est_tokens"] - d["tokens"]
            assert abs(err) <= 1, f"size fit off by {err} at {d}"
            assert m["chars"] == len(state)

            # No answer words, and no reserved-prefix leakage beyond the needle.
            _audit(state, "generated state")
            assert state.count("CR-") == 1

            # The recorded frequency baseline must be what a reader counting
            # the state would get. Two bugs hid here: the corpus builder used
            # to count candidate sentences it discarded, and the count used to
            # omit the needle's own mention of its component.
            for opt, n_opt in m["option_mention_counts"].items():
                assert state.count(opt) == n_opt, (opt, state.count(opt), n_opt)
            assert (n.component in m["most_mentioned_options"]) == m["most_mentioned_is_correct"]

            # And the correct option must not stand out by frequency. The
            # needle names its component once, so the corpus emits it once
            # fewer; without that the answer is the unique most-mentioned
            # option and the choice question is answerable by counting.
            full = {c: state.count(c) for c in COMPONENTS}
            assert max(full.values()) - min(full.values()) <= 1, full

        a, b = insts
        assert a.instance_id != b.instance_id
        d0 = insts[0]
        print(
            f"{d['tokens']:>7} {d['position']:>5.2f} {d0.meta['est_tokens']:>7} "
            f"{d0.meta['est_tokens'] - d['tokens']:>5} {d0.meta['chars']:>8} "
            f"{d0.meta['n_paragraphs']:>6} {d0.meta['realized_position']:>9.3f}"
            f" {d0.meta['est_tokens_by_words']:>8}"
        )

    # Constant size across needle positions -- the separation E4 item 2 rests on.
    for tokens in (2000, POSITION_SWEEP_TOKENS):
        sizes, realized, facts = set(), [], set()
        for p in NEEDLE_POSITIONS:
            inst = generate(difficulty={"tokens": tokens, "position": p}, seed=seed, count=1)[0]
            sizes.add(len(inst.state))
            realized.append(inst.meta["realized_position"])
            facts.add(inst.meta["needle_id"])
            assert inst.state.count(inst.meta["needle_id"]) == 1
        assert len(sizes) == 1, f"state size varies with position at {tokens}: {sorted(sizes)}"
        assert len(facts) == 1, "needle differs across positions; not a paired comparison"
        assert realized == sorted(realized), f"realized position not monotone: {realized}"
        assert realized[0] == 0.0 and realized[-1] == 1.0
        print(f"constant size at {tokens:>6} tokens: {sizes.pop():>8} chars, realized {realized}")

    # The needle's phrasing is not unique in the state, so it cannot be found by
    # matching the verb; only its identifier is unique.
    for tokens in (1000, 10000):
        inst = generate(difficulty={"tokens": tokens, "position": 0.5}, seed=seed, count=1)[0]
        assert inst.state.count("is scheduled for the") > 1
        assert inst.state.count("days of downtime") > 1
        counts = inst.meta["option_mention_counts"]
        assert max(counts.values()) - min(counts.values()) <= 1, counts

    # A question must not answer itself. The threshold is the only number the
    # needle's noul shows, so if its value moved with the answer a model that
    # never located the needle would still score well -- and would score just
    # as well at 50,000 tokens as at 200, flattening the curve E4 exists to
    # measure. The check is on the two class means because that is what a
    # threshold-only rule exploits; drawing days first and the threshold around
    # it, the obvious construction, separated them by about 5 and let such a
    # rule reach 0.86.
    thresholds: dict[bool, list[int]] = {True: [], False: []}
    for inst in generate(difficulty={"tokens": 1000, "position": 0.0}, seed=seed, count=600):
        shown = re.search(r"more than (\d+) days", inst.questions["needle_downtime"]["question"])
        assert shown
        thresholds[inst.truth["needle_downtime"]].append(int(shown.group(1)))
    assert len(thresholds[True]) == len(thresholds[False]) == 300
    span = set(range(_NEEDLE_THRESHOLD_LO, _NEEDLE_THRESHOLD_HI + 1))
    assert set(thresholds[True]) == set(thresholds[False]) == span
    lo, hi = (sum(v) / len(v) for v in (thresholds[False], thresholds[True]))
    # 300 draws per class from a range with sd 2.3 put the standard error of
    # this difference at 0.19, so the bound is four of those -- loose enough
    # not to fail on a seed, tight enough that the old construction, which
    # separated the means by about 5, could not have squeaked through.
    assert abs(hi - lo) < 0.8, f"needle threshold leaks the answer: {lo:.2f} vs {hi:.2f}"
    print(f"needle threshold mean: {lo:.2f} when no, {hi:.2f} when yes")

    # The same trap in the filler's sensor question, which E3 scores by
    # position. Here the guarantee is checked per question rather than as a
    # class mean: readings run 15.0 to 94.9 and are not controlled, so the two
    # class means differ by several units on some seeds through sampling alone
    # and a bound on them would pass or fail by luck. What the construction
    # actually promises is that the threshold sits within _SENSOR_MAX_SPAN of
    # the reading it asks about, whichever way the answer goes, and that is
    # what makes the two marginals nearly the same. The original +/-20 span
    # would fail this on nearly every question.
    limit = _SENSOR_MAX_SPAN_TENTHS / 10.0
    probe = filler_block(seed=seed, count=1200)
    readings = {r["id"]: r["reading"] for r in probe.records if r["kind"] == _SENSOR}
    marks: dict[bool, list[float]] = {True: [], False: []}
    for key, q in probe.questions.items():
        shown = re.fullmatch(
            r"Did sensor (SEN-\d+) report a mean above ([\d.]+) units\?", q["question"]
        )
        if not shown:
            continue
        reading, mark = readings[shown.group(1)], float(shown.group(2))
        assert 0.0 < mark, f"implausible threshold {mark} for a reading of {reading}"
        assert abs(mark - reading) <= limit + 1e-9, f"threshold {mark} is far from {reading}"
        assert (reading > mark) == probe.truth[key]
        marks[probe.truth[key]].append(mark - reading)
    assert len(marks[True]) + len(marks[False]) > 100
    lo, hi = (sum(v) / len(v) for v in (marks[True], marks[False]))
    print(
        f"sensor threshold sits {lo:+.2f} from the reading when yes, {hi:+.2f} when no "
        f"(both within {limit} of it, over {len(marks[True]) + len(marks[False])} questions)"
    )

    # needle_series makes the same guarantee without going through generate().
    series = needle_series(fact="Marker XYZ.", target_tokens=3000, seed=seed)
    assert len({len(p.text) for p in series}) == 1
    assert all(p.text.count("Marker XYZ.") == 1 for p in series)

    # Determinism: same arguments, byte-identical instances.
    for d in (difficulty_sweep()[0], difficulty_sweep()[-1], {"tokens": 5000, "position": 0.5}):
        first = [x.to_json() for x in generate(difficulty=d, seed=seed, count=3)]
        second = [x.to_json() for x in generate(difficulty=d, seed=seed, count=3)]
        assert first == second, f"generate is not deterministic at {d}"
    assert dilution_text(target_tokens=1200, seed=seed) == dilution_text(
        target_tokens=1200, seed=seed
    )

    # Filler blocks: distinct, answerable, balanced, and enough of them.
    for count in (10, 64, 200, 255, 400, 1000):
        block = filler_block(seed=seed, count=count)
        assert len(block.questions) == count
        texts = [q["question"] for q in block.questions.values()]
        assert len(set(texts)) == count, f"{count - len(set(texts))} duplicate filler questions"
        _audit(block.text, "filler block text")
        ids = [r["id"] for r in block.records]
        assert len(set(ids)) == len(ids), "duplicate record ids in one block"
        counts = block.meta["component_counts"]
        assert max(counts.values()) - min(counts.values()) <= 1, counts
        # Ground truth is checkable against the records that produced it.
        by_id = {r["id"]: r for r in block.records}
        for key, q in block.questions.items():
            found = re.search(r"\b(WO-\d+|SEN-\d+)\b", q["question"])
            assert found, q["question"]
            rec = by_id[found.group(1)]
            t = block.truth[key]
            if q["type"] == "choice":
                assert t == rec["component"]
            elif q["type"] == "score":
                assert t == rec["duration_bucket"]
            else:
                assert isinstance(t, bool)
        frac = block.meta["noul_true_fraction"]
        n_nouls = block.meta["type_counts"]["noul"]
        assert abs(frac - 0.5) <= 0.5 / max(1, n_nouls), f"filler nouls imbalanced: {frac}"
        assert min(block.meta["type_counts"].values()) > 0 or count < 3
        # A block is only useful if validate() accepts it on a real Instance.
        Instance(
            generator=NAME, difficulty={"fillers": count}, seed=seed, index=0,
            state=block.text, questions=block.questions, truth=block.truth,
        ).validate()
        assert filler_block(seed=seed, count=count).text == block.text
        print(
            f"filler block n={count:>3}: {block.meta['n_records']:>3} records, "
            f"{block.meta['est_tokens']:>5}/{block.meta['est_tokens_by_words']:>5} est tokens "
            f"(chars/words), types {block.meta['type_counts']}, noul true {frac:.2f}"
        )

    # attach() must not put a derived field into the state: duration_bucket is
    # the answer to the score questions.
    block = filler_block(seed=seed, count=64)
    assert attach("p cnf 3 2\n", block).endswith(block.text)
    for shape in (attach({"formula": "x"}, block), attach([{"formula": "x"}], block)):
        flat = shape if isinstance(shape, list) else [shape]
        for record in flat[-1]["maintenance_log"]:
            assert "duration_bucket" not in record and "kind" not in record
    try:
        attach({"maintenance_log": 1}, block)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("attach overwrote an existing key")
    for hostile in ("ticket from Brannigan about the Larchmont pump array", "see WO-100001"):
        try:
            attach(hostile, block)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"attach accepted a colliding state: {hostile!r}")
        try:
            attach({"body": hostile}, block)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"attach accepted a colliding dict state: {hostile!r}")

    # Target insertion: uniform keys, target where it was asked for.
    block = filler_block(seed=seed, count=254)
    target = noul("Is this formula the one described above?")
    for pos in (1, 5, 20, 50, 100, 200, 255):
        batch = batch_with_target(block=block, target=target, target_truth=True, position=pos)
        keys = list(batch.questions)
        assert len(keys) == 255
        assert keys[pos - 1] == batch.target_key
        assert batch.questions[batch.target_key] is target
        assert all(re.fullmatch(r"q\d{3}", k) for k in keys)
        assert set(batch.truth) == set(batch.questions)

    # Collision check against states the other generators actually emit.
    samples = {
        "dimacs": "p cnf 20 85\n1 -4 17 0\n-3 9 -12 0\n",
        "python": "def f(x):\n    if x > 5:\n        return 1\n    return 0\n",
        "edges": '{"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]]}',
        "ticket": "Customer cannot log in after the password reset email expired.",
    }
    for name, text in samples.items():
        assert collisions(text) == [], f"{name} collides: {collisions(text)}"
    assert "Larchmont pump array" in collisions("we replaced the Larchmont pump array")
    assert "WO-" in collisions("see WO-123456")

    print("\nOK")


if __name__ == "__main__":
    _selfcheck()
