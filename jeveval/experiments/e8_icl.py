"""E8 -- in-context learning, and what steering costs in calibration.

The plan asks two things here and only one of them is about accuracy. Whether
examples steer the model at all is worth knowing. Whether steering breaks the
probabilities is the question that decides whether the model is still worth
using, because calibration is the reason to choose it over an LLM. So ECE is the
headline of this module and accuracy is reported beside it, not the other way
round.

TWO BASE TASKS, AND WHY THERE HAD TO BE TWO. The brief said to use the semantic
control at its "hard" level, on the grounds that the clean level is solved by a
keyword heuristic and leaves accuracy no room to move. Probing the live endpoint
showed the hard level leaves this model no room either: on 40 hard tickets it
scored 40/40 on the yes/no routing question (AUROC 1.00) and 40/40 on the 8-way
routing choice, against a keyword heuristic of 0.60 and 0.43 respectively. It
also scored 40/40 on yes/no probes built from the distractor departments the
hard level mentions and explicitly rules out, so the saturation is not an
artifact of easy negatives. An ICL sweep on a task the model already solves can
measure a calibration change but cannot measure steerability, and poisoning the
examples cannot move a number that is pinned at 1.0. So this module runs the
sweep on two questions from that same generator and level:

  routing (noul)  -- accuracy 1.00 zero-shot, ECE 0.077. The calibration arm.
                     A noul answer is a bare probability against a binary
                     outcome, which is ECE exactly as the plan defines it. This
                     arm carries the headline. Because accuracy cannot rise
                     here, any ECE the examples cost is a calibration loss that
                     buys nothing -- the plan's worst trade, in its pure form.
  urgency (score) -- accuracy 0.90 zero-shot against a 0.25 chance baseline and
                     a 0.30 majority class. The steerability arm. Examples have
                     room to help and poisoning has room to hurt. Its ECE is
                     top-label calibration -- the probability on the rubric
                     point the model picked, against whether that point was
                     right -- which is a weaker notion than the routing arm's
                     and is labelled as such everywhere it appears.

Both arms are graded, but not by the same rubric. The E8 ladder's rungs are
written in terms of accuracy_gain, and on a saturated task its "examples have no
effect (no steerability)" disqualifier fires on a ceiling rather than on a model
that ignores its examples. So the routing arm is graded by the calibration
rubric, which grades a reliability curve on its own terms, and the urgency arm
by the E8 rubric. Which rubric graded which row is stated in the result.

The rest of the shape, and the reason for each choice:

*The instructions channel is the per-question `instructions` string.* There is no
request-level instructions field -- sending one is a 400. The plan's channel
comparison is therefore narrower here than it was written to be: per-question
instructions against state, not a system prompt against state. P23 is scored on
that narrower comparison and says so in its outcome.

*Every cell of an arm runs the same items in the same order.* Accuracy and ECE
differences are then paired differences on identical instances, so McNemar and
the paired bootstrap apply and item difficulty cancels.

*An exemplar never shares a template with the item it accompanies.* The generator
renders from a fixed template set, so an exemplar drawn from the item's own
template would hand over the answer. Eligibility is filtered on template id.

*The exemplar set is nested across example counts.* The k=1 block is a subset of
the k=3 block is a subset of the k=10 block, so the sweep varies the count and
not the material. Presentation order inside a block is shuffled per item, which
is why the nesting is stated of the set rather than of the sequence.

*Poisoning swaps the labels of two examples that carry different labels.*
Relabelling at random would also move the label marginal, and a shifted prior on
the answer would be indistinguishable from the examples' content mattering. A
swap holds the marginal exactly fixed while mislabelling 20% of the block, so
what moves is only the example-to-label mapping. For the binary routing arm a
swap is the same thing as flipping one YES and one NO.

*The novel rubric is an arbitrary partition of the departments under invented
names.* Novelty is enforced mechanically rather than asserted: the names are
generated from a syllable inventory and rejected if they occur in the ticket
corpus, if any four consecutive characters of them occur inside a task term, or
if they share a first letter with each other. The partition is rejected unless
every category draws from at least two of three declared semantic families, so
no category can be a synonym for a familiar grouping.

*The inverted rubric swaps two teams' scopes and keeps the request otherwise
identical to its control.* The two arms differ in one sentence, so the
definition-following rate is not confounded by which team was asked about. Both
rubric conditions are built on routing, and the routing saturation is a help
there rather than a problem: the model can already route these tickets perfectly,
so a failure to follow an invented or inverted definition is a
definition-following failure and not a routing failure.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config, metrics, plots, predictions, tiers
from ..client import Call, CallResult, JevClient
from ..generators import semantic
from ..instances import choice as choice_question
from ..instances import noul as noul_question
from ..instances import rng_for
from ..instances import score as score_question
from ..instances import shuffled_options

EXPERIMENT = "E8"
LEVEL = "hard"

# 0 is the shared zero-shot cell of each arm; it is run once and serves as the
# baseline for both channels, because with no examples the two channels produce
# an identical request.
EXAMPLE_COUNTS: tuple[int, ...] = (0, 1, 3, 10)
MAX_EXAMPLES = max(EXAMPLE_COUNTS)
CHANNELS: tuple[str, ...] = ("instructions", "state")
POISON_FRACTION = 0.20
# The example count poisoning is applied at. Equal to MAX_EXAMPLES today and kept
# as its own name: every cell the headline, the channel choice and P22/P23 read
# is "the top of the sweep" and says MAX_EXAMPLES, while the poisoned cells and
# the clean control they are paired against say POISON_AT. Conflating the two
# would silently move the headline if poisoning were ever run at a smaller k.
POISON_AT = 10

# 32 distinct exemplars, drawn once and shared by every item. The generator
# assigns one template per (department, urgency) pair over blocks of 8, so 32
# covers every combination exactly once, which is what lets the urgency arm draw
# a rubric-balanced block. Each item selects its own ordered subset, so no single
# unlucky exemplar drives a whole cell.
EXEMPLAR_POOL = 32

# Two different floors, kept apart because one is the plan's and one is not.
#
# The plan's: "500 instances per condition minimum for anything where ECE is
# reported -- 10 bins needs enough mass per bin to be stable." A cell under it
# still reports its ECE, and the result says the number is under-sampled rather
# than printing it as though it meant something.
ECE_SAMPLE_FLOOR = config.SIZES.calibration
# The harness's: below this, a reliability curve is non-monotone for reasons
# that have nothing to do with the model, and non-monotone is a disqualifier
# that costs a condition every tier it has. So monotonicity is not assessed
# below it and cannot send a thin cell to the worst tier. The bootstrap interval
# on an ECE difference is gated the same way.
MIN_N_FOR_MONOTONICITY = 100

BOOTSTRAP_RESAMPLES = 2000

# P22's claim is a band and its falsifier is a direction, and the two do not
# cover the same ground: a +0.001 degradation is neither the cost the plan
# predicted nor the "unchanged or improved" it named as falsification. The band
# decides the verdict, which is how every other module here scores a prediction,
# and the falsifier is reported beside it. "Unchanged" is read as the harness's
# own EPS_FLAT -- the change it declines to resolve -- which is also about the
# floor that 2-decimal quantization puts under any ECE.
P22_BAND = (0.03, 0.10)
P22_UNCHANGED = tiers.EPS_FLAT
# P24 predicts "about 70%" with no band of its own, so the plan's falsification
# clause -- >0.95 or <0.30 -- is the only line there is, and it is the verdict.
P24_WINDOW = (0.30, 0.95)

KEY = "q"


@dataclass(frozen=True)
class Task:
    """One base task for the ICL sweep.

    `calibration` names which kind of ECE the arm yields, because the two are not
    interchangeable and the report must not print them in one column as though
    they were.
    """

    name: str
    kind: str  # "noul" or "score"
    key: str  # the generator's question key
    chance: float
    calibration: str
    role: str


ROUTING = Task(
    name="routing",
    kind="noul",
    key=semantic.KEY_NOUL,
    chance=0.5,
    calibration="probability against outcome, the plan's definition",
    role="calibration arm; accuracy is at ceiling zero-shot, so no ICL gain is possible here",
)
URGENCY = Task(
    name="urgency",
    kind="score",
    key=semantic.KEY_SCORE,
    chance=1.0 / len(semantic.SEVERITIES),
    calibration="top-label: probability on the chosen rubric point against whether it was right",
    role="steerability arm; 0.90 zero-shot against 0.25 chance leaves room for examples to move accuracy",
)
TASKS: tuple[Task, ...] = (ROUTING, URGENCY)

# What each department actually covers, in plain language. Used by the novel
# rubric (so a category is defined by the requests it covers rather than by a
# list of department names, which would make the task a name lookup) and by the
# inverted rubric (so a team's scope can be restated). Derived from
# semantic.DEPARTMENT_KEYWORDS; the self-check asserts each phrase carries its
# own department's vocabulary and no other department's.
DEPARTMENT_SCOPES: dict[str, str] = {
    "billing": "invoices, charges, refunds and payment problems",
    "shipping": "shipment tracking, delivery delays and courier problems",
    "technical": "software errors, crashes, failed syncs and API problems",
    "account_access": "passwords, sign-in failures, lockouts and two-factor problems",
    "sales": "pricing, quotes, contracts, seats and purchase orders",
    "privacy": "personal data requests, erasure, retention and data protection questions",
    "careers": "job applications, interviews and recruiting questions",
    "press": "journalists, media enquiries, statements and embargoes",
}

# Three declared families over the eight departments. They exist only as a
# rejection criterion: a novel category confined to one family, or an inverted
# pair drawn from one family, would be close enough to a familiar concept that a
# model pattern-matching rather than reading the definition could still score.
SEMANTIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "commerce": ("billing", "shipping", "sales"),
    "systems": ("technical", "account_access"),
    "external": ("privacy", "careers", "press"),
}
FAMILY_OF: dict[str, str] = {
    d: fam for fam, members in SEMANTIC_FAMILIES.items() for d in members
}

NOVEL_CATEGORY_SIZES = (3, 3, 2)

SEVERITY_LABEL: dict[str, str] = {r["id"]: r["label"] for r in semantic.SEVERITY_RUBRIC}


# --------------------------------------------------------------------------
# Exemplars
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Exemplar:
    """One labelled worked example.

    `shown` is the label the model is told; `true` is the correct one. They
    differ only in the poisoned cells, and keeping both means the JSONL log
    records which examples were mislabelled without the analysis having to
    re-derive the poisoning.

    `asked` is the department a routing exemplar asks about and is None for an
    urgency exemplar, which asks the same question of every ticket.
    """

    ticket: semantic.Ticket
    shown: Any
    true: Any
    asked: str | None = None

    def question(self, task: Task) -> str:
        if task.kind == "noul":
            return semantic.NOUL_QUESTION.format(
                label=semantic.DEPARTMENT_LABELS[self.asked]
            )
        return semantic.SCORE_QUESTION

    def answer(self, task: Task) -> str:
        if task.kind == "noul":
            return "YES" if self.shown else "NO"
        return SEVERITY_LABEL[self.shown]

    def as_text(self, task: Task, ordinal: int) -> str:
        t = self.ticket
        lines = [f"Example {ordinal}.", f"From: {t.customer} <{t.customer_email}>"]
        if t.subject is not None:
            lines.append(f"Subject: {t.subject}")
        lines += [
            "Email:",
            t.body,
            f"Question: {self.question(task)}",
            f"Correct answer: {self.answer(task)}",
        ]
        return "\n".join(lines)

    def as_state(self, task: Task, ordinal: int) -> dict:
        state: dict[str, Any] = {"example": ordinal}
        state.update(self.ticket.as_state())
        state["question"] = self.question(task)
        state["correct_answer"] = self.answer(task)
        return state

    def to_meta(self) -> dict:
        return {
            "template_id": self.ticket.template_id,
            "department": self.ticket.department,
            "severity": self.ticket.severity,
            "asked": self.asked,
            "shown_label": self.shown,
            "true_label": self.true,
            "poisoned": self.shown != self.true,
        }


def _label_plan(index: int, rng: random.Random) -> list[bool]:
    """The YES/NO labels for one routing item's ten exemplar slots.

    Constrained so that every prefix the sweep uses is as balanced as it can be,
    and so that what is unbalanced within an item cancels across items: the
    length-1 prefix alternates with the item index, the length-3 prefix is 2:1 in
    the direction the item index picks, and the full ten are 5:5. Without this
    the k=1 cell would carry a systematic prior toward one answer and would not
    be comparable to the k=10 cell.
    """
    first = index % 2 == 0
    head = [first, not first, first]
    remaining = MAX_EXAMPLES - len(head)
    need_true = MAX_EXAMPLES // 2 - sum(head)
    tail = [True] * need_true + [False] * (remaining - need_true)
    rng.shuffle(tail)
    return head + tail


def _severity_order(candidates: list[semantic.Ticket], index: int) -> list[semantic.Ticket]:
    """Order exemplars so that every prefix covers the rubric as evenly as it can.

    The urgency arm's analogue of `_label_plan`. Tickets are dealt round-robin
    over the four severities, starting from the one the item index rotates to, so
    a prefix of length k holds either floor(k/4) or ceil(k/4) of each severity
    and the single exemplar in the k=1 cell is a different severity on every
    fourth item. When the template exclusion leaves the pool short of one
    severity the round-robin simply skips it, so the imbalance lands at the tail
    of the order rather than inside the prefixes the sweep uses.
    """
    buckets: dict[str, list[semantic.Ticket]] = {s: [] for s in semantic.SEVERITIES}
    for t in candidates:
        buckets[t.severity].append(t)
    rotation = index % len(semantic.SEVERITIES)
    wheel = semantic.SEVERITIES[rotation:] + semantic.SEVERITIES[:rotation]
    out: list[semantic.Ticket] = []
    while any(buckets[s] for s in wheel):
        for s in wheel:
            if buckets[s]:
                out.append(buckets[s].pop(0))
    return out


def _poison(labels: list[Any], k: int, rng: random.Random) -> list[Any]:
    """Mislabel POISON_FRACTION of the first `k` slots, holding the marginal fixed.

    Poisoning swaps labels between pairs of examples that carry different labels.
    A swap changes two examples and changes no label count, so a measured drop
    cannot be the answer's prior moving. On the binary routing arm one swap is
    exactly one YES flipped to NO and one NO flipped to YES.
    """
    shown = list(labels)
    swaps = max(1, int(round(POISON_FRACTION * k / 2)))
    positions = list(range(k))
    for _ in range(swaps):
        rng.shuffle(positions)
        pair = next(
            (
                (i, j)
                for a, i in enumerate(positions)
                for j in positions[a + 1 :]
                if shown[i] != shown[j] and shown[i] == labels[i] and shown[j] == labels[j]
            ),
            None,
        )
        if pair is None:  # every distinct pair is already swapped
            break
        i, j = pair
        shown[i], shown[j] = shown[j], shown[i]
    return shown


def _exemplars_for(
    *,
    task: Task,
    item_index: int,
    item_template_id: str,
    pool: list[semantic.Ticket],
    seed: int,
    poisoned: bool,
    k: int,
) -> list[Exemplar]:
    """The ordered exemplar block for one item at one example count.

    The eligible pool excludes any exemplar rendered from the item's own
    template, which would otherwise be a labelled copy of the answer. Selection
    order is fixed per item and prefixes are taken from it, so the exemplar sets
    are nested across k; the block is then shuffled for presentation.

    Poisoning draws from a stream of its own rather than from `rng`, so a
    poisoned block is the clean block with two labels swapped and nothing else.
    Sharing the stream would have advanced it by however many draws the swap
    took, which changes the department each negative exemplar asks about and the
    order the block is presented in -- so the poisoned cell would have differed
    from its clean control in three ways at once and the measured drop could not
    be attributed to the mislabelling.
    """
    if k == 0:
        return []
    rng = rng_for("e8", {"purpose": "exemplars", "task": task.name}, seed, item_index)
    eligible = [t for t in pool if t.template_id != item_template_id]
    if len(eligible) < k:
        raise ValueError(
            f"item {item_index}: only {len(eligible)} exemplars are eligible, need {k}; "
            "enlarge EXEMPLAR_POOL"
        )

    if task.kind == "noul":
        order = list(eligible)
        rng.shuffle(order)
        chosen = order[:k]
        true_labels: list[Any] = _label_plan(item_index, rng)[:k]
    else:
        shuffled = list(eligible)
        rng.shuffle(shuffled)
        chosen = _severity_order(shuffled, item_index)[:k]
        true_labels = [t.severity for t in chosen]

    shown = list(true_labels)
    if poisoned:
        shown = _poison(
            true_labels,
            k,
            rng_for("e8", {"purpose": "poison", "task": task.name}, seed, item_index),
        )

    block: list[Exemplar] = []
    for slot, ticket in enumerate(chosen):
        asked = None
        if task.kind == "noul":
            asked = (
                ticket.department
                if true_labels[slot]
                else rng.choice([d for d in semantic.DEPARTMENTS if d != ticket.department])
            )
        block.append(
            Exemplar(ticket=ticket, shown=shown[slot], true=true_labels[slot], asked=asked)
        )
    rng.shuffle(block)
    return block


_INSTRUCTIONS_PREAMBLE = "Worked examples of this task, each one an email and the correct answer:"
_INSTRUCTIONS_CLOSER = "Now answer the same kind of question for the email in the state."
_STATE_ORIENTATION = (
    'The state holds worked examples under "examples" and, under "ticket", '
    "the one email you are being asked about."
)


def _icl_request(
    *,
    task: Task,
    channel: str,
    exemplars: list[Exemplar],
    item_state: Any,
    question: str,
) -> tuple[Any, str]:
    """(state, instructions) for one cell.

    The zero-shot form is the bare ticket state and the bare question. Each
    channel adds the examples to one field and one orienting sentence to the
    instructions. That sentence is not a hint: it is what tells the model which
    of several emails it is being asked about, and both channels carry one, so
    the comparison is about placement and not about one channel being told it has
    examples while the other is not.
    """
    if not exemplars:
        return item_state, question
    if channel == "instructions":
        blocks = [ex.as_text(task, i + 1) for i, ex in enumerate(exemplars)]
        return item_state, "\n\n".join(
            [_INSTRUCTIONS_PREAMBLE, *blocks, _INSTRUCTIONS_CLOSER, question]
        )
    if channel == "state":
        state = {
            "examples": [ex.as_state(task, i + 1) for i, ex in enumerate(exemplars)],
            "ticket": item_state,
        }
        return state, f"{_STATE_ORIENTATION}\n\n{question}"
    raise ValueError(f"unknown channel {channel!r}")


# --------------------------------------------------------------------------
# The novel rubric
# --------------------------------------------------------------------------

_ONSETS = ("k", "v", "z", "thr", "gl", "br", "sk", "dr", "pl", "tr", "kr", "vr")
_NUCLEI = ("a", "e", "i", "o", "u", "au", "ei")
_CODAS = ("rn", "lk", "th", "ndth", "sk", "rv", "ln", "mp", "rth")


def _novel_names(rng: random.Random, corpus: str, vocabulary: str, count: int) -> list[str]:
    """Pronounceable non-words that occur nowhere in the task's own vocabulary.

    Three rejections make the novelty claim checkable rather than asserted. A
    candidate is discarded if its lowercase form appears anywhere in the corpus
    -- the department labels, every department keyword, every scope phrase, and
    the rendered text of the whole exemplar pool. It is discarded if any four
    consecutive characters of it occur inside a task term, which is the test that
    rules out a candidate such as TROMPAUTH: "auth" sits inside "authenticator",
    so the name would point at the account-access department whatever its
    definition said. And it is discarded if it starts with the same letter as a
    name already accepted, so the model cannot tell two categories apart by a
    glance at the first character instead of by reading either definition.
    """
    names: list[str] = []
    for _ in range(20000):
        if len(names) == count:
            break
        word = (
            rng.choice(_ONSETS)
            + rng.choice(_NUCLEI)
            + rng.choice(_CODAS)
            + rng.choice(_NUCLEI)
            + rng.choice(_CODAS)
        )
        if len(word) < 7 or word in corpus:
            continue
        if any(word[i : i + 4] in vocabulary for i in range(len(word) - 3)):
            continue
        if any(n[0] == word[0] for n in names):
            continue
        names.append(word)
    if len(names) != count:
        raise RuntimeError(f"could not generate {count} novel names")
    return [n.upper() for n in names]


def _novel_partition(rng: random.Random) -> list[list[str]]:
    """An arbitrary partition of the departments, rejected unless every category
    spans at least two semantic families."""
    for _ in range(2000):
        order = list(semantic.DEPARTMENTS)
        rng.shuffle(order)
        groups: list[list[str]] = []
        at = 0
        for size in NOVEL_CATEGORY_SIZES:
            groups.append(sorted(order[at : at + size]))
            at += size
        if all(len({FAMILY_OF[d] for d in g}) >= 2 for g in groups):
            return groups
    raise RuntimeError("no cross-family partition found")


def _novel_scheme(seed: int, pool: list[semantic.Ticket]) -> dict:
    corpus = " ".join(
        [
            *semantic.DEPARTMENT_LABELS.values(),
            *(k for kws in semantic.DEPARTMENT_KEYWORDS.values() for k in kws),
            *DEPARTMENT_SCOPES.values(),
            *(t.text for t in pool),
        ]
    ).lower()
    # Every task word, as one string to test four-character runs against. Four is
    # the shortest run that reads as a hint rather than as noise.
    vocabulary = " ".join(
        sorted(
            {
                term
                for source in (
                    (k for kws in semantic.DEPARTMENT_KEYWORDS.values() for k in kws),
                    (w for label in semantic.DEPARTMENT_LABELS.values() for w in label.split()),
                    (w for phrase in DEPARTMENT_SCOPES.values() for w in phrase.split()),
                )
                for term in (re.sub(r"[^a-z]", "", t.lower()) for t in source)
                if len(term) >= 4
            }
        )
    )
    rng = random.Random(seed)
    groups = _novel_partition(rng)
    names = _novel_names(rng, corpus, vocabulary, len(groups))
    return {
        "names": names,
        "members": {names[i]: g for i, g in enumerate(groups)},
        "category_of": {d: names[i] for i, g in enumerate(groups) for d in g},
        "novelty_checks": {
            "names_absent_from_corpus": True,
            "names_share_no_four_character_run_with_task_vocabulary": True,
            "names_start_with_distinct_letters": True,
            "every_category_spans_two_families": True,
            "corpus_chars": len(corpus),
            "vocabulary_chars_checked": len(vocabulary),
            "families": {
                names[i]: sorted({FAMILY_OF[d] for d in g}) for i, g in enumerate(groups)
            },
        },
    }


_NOVEL_PREAMBLE = (
    "This queue sorts email into three local categories. The category names are "
    "internal codes with no meaning beyond the definitions given here, and they "
    "do not line up with any ordinary grouping. Read the definitions and assign "
    "this email to the one category whose definition covers it."
)


def _novel_options(scheme: dict) -> list[dict]:
    return [
        {
            "id": name,
            "label": f"{name} -- covers "
            + "; ".join(DEPARTMENT_SCOPES[d] for d in scheme["members"][name])
            + ".",
        }
        for name in scheme["names"]
    ]


# --------------------------------------------------------------------------
# The inverted rubric
# --------------------------------------------------------------------------


def _swap_pairs(seed: int) -> dict[str, str]:
    """A perfect matching on the departments in which no pair sits inside one
    semantic family, so each swap is a real conflict with the conventional
    meaning rather than a near-synonym."""
    rng = random.Random(seed)
    for _ in range(2000):
        order = list(semantic.DEPARTMENTS)
        rng.shuffle(order)
        pairs = [(order[i], order[i + 1]) for i in range(0, len(order), 2)]
        if all(FAMILY_OF[a] != FAMILY_OF[b] for a, b in pairs):
            out: dict[str, str] = {}
            for a, b in pairs:
                out[a] = b
                out[b] = a
            return out
    raise RuntimeError("no cross-family pairing found")


def _convention(a: str, b: str, *, inverted: bool) -> str:
    la, lb = semantic.DEPARTMENT_LABELS[a], semantic.DEPARTMENT_LABELS[b]
    sa, sb = DEPARTMENT_SCOPES[a], DEPARTMENT_SCOPES[b]
    if inverted:
        return (
            f'Local convention for this queue, which differs from the usual one: the team '
            f'named "{la}" handles {sb}. The team named "{lb}" handles {sa}. Every other '
            f"team keeps its usual scope. Apply this convention rather than the usual "
            f"meaning of the team names."
        )
    return (
        f'Local convention for this queue: the team named "{la}" handles {sa}. The team '
        f'named "{lb}" handles {sb}. Every other team keeps its usual scope. Apply this '
        f"convention."
    )


# --------------------------------------------------------------------------
# Call construction
# --------------------------------------------------------------------------


def _cell_key(task: Task, channel: str, k: int, poisoned: bool) -> str:
    if k == 0:
        return f"{task.name}/zeroshot"
    return f"{task.name}/{channel}-{k}" + ("-poisoned" if poisoned else "")


def _icl_calls(
    *,
    task: Task,
    items: list,
    pool: list[semantic.Ticket],
    exemplar_seed: int,
    channel: str,
    k: int,
    poisoned: bool,
) -> list[Call]:
    condition = _cell_key(task, channel, k, poisoned)
    calls: list[Call] = []
    for item in items:
        question = item.questions[task.key]["question"]
        exemplars = _exemplars_for(
            task=task,
            item_index=item.index,
            item_template_id=item.meta["template_id"],
            pool=pool,
            seed=exemplar_seed,
            poisoned=poisoned,
            k=k,
        )
        state, instructions = _icl_request(
            task=task,
            channel=channel,
            exemplars=exemplars,
            item_state=item.state,
            question=question,
        )
        wire_question = (
            noul_question(instructions)
            if task.kind == "noul"
            else score_question(instructions, semantic.SEVERITY_RUBRIC)
        )
        calls.append(
            Call(
                experiment=EXPERIMENT,
                condition=condition,
                instance_id=item.instance_id,
                state=state,
                questions={KEY: wire_question},
                meta={
                    "truth": {KEY: item.truth[task.key]},
                    "difficulty": {
                        "task": task.name,
                        "channel": channel,
                        "examples": k,
                        "poisoned": poisoned,
                    },
                    "index": item.index,
                    "task": task.name,
                    "question_kind": task.kind,
                    "level": LEVEL,
                    "template_id": item.meta["template_id"],
                    "department": item.meta["department"],
                    "severity": item.meta["severity"],
                    "asked_department": item.meta.get("asked_department"),
                    "baseline_keyword_prediction": item.meta["baseline_keyword_prediction"],
                    "n_rubric_points": len(semantic.SEVERITIES),
                    # Ground truth on the urgency arm is a rubric *id*, while the
                    # wire question carries the rubric *labels* -- so a rescore
                    # from the log alone reads back a label and would score every
                    # urgency answer wrong against an id. The ids in wire order
                    # are the missing half of that mapping: wire criterion i is
                    # rubric_ids[i]. Logged rather than left to whoever rescores
                    # to import this generator's constants.
                    "rubric_ids": (
                        list(semantic.SEVERITIES) if task.kind == "score" else None
                    ),
                    "exemplars": [ex.to_meta() for ex in exemplars],
                    "n_poisoned_exemplars": sum(1 for ex in exemplars if ex.shown != ex.true),
                },
            )
        )
    return calls


def _novel_calls(
    *, tickets: list[semantic.Ticket], scheme: dict, seed: int, arm: str
) -> list[Call]:
    """`arm` is "novel" (three invented categories) or "conventional" (the eight
    real departments on the same tickets, which is what separates a definition
    the model would not follow from a ticket it could not route)."""
    calls: list[Call] = []
    for index, ticket in enumerate(tickets):
        counts = semantic.keyword_counts(ticket.text)
        keyword = semantic.keyword_baseline(counts)
        rng = rng_for("e8", {"purpose": f"novel-{arm}"}, seed, index)
        if arm == "novel":
            q = shuffled_options(choice_question(_NOVEL_PREAMBLE, _novel_options(scheme)), rng)
            truth = scheme["category_of"][ticket.department]
            heuristic = scheme["category_of"].get(keyword) if keyword else None
        else:
            options = [
                {"id": d, "label": semantic.DEPARTMENT_LABELS[d]} for d in semantic.DEPARTMENTS
            ]
            q = shuffled_options(choice_question(semantic.CHOICE_QUESTION, options), rng)
            truth = ticket.department
            heuristic = keyword
        calls.append(
            Call(
                experiment=EXPERIMENT,
                condition=f"rubric-{arm}",
                instance_id=f"e8-{arm}-{index:05d}",
                state=ticket.as_state(),
                questions={KEY: q},
                meta={
                    "truth": {KEY: truth},
                    "difficulty": {"condition": f"rubric-{arm}"},
                    "index": index,
                    "level": LEVEL,
                    "template_id": ticket.template_id,
                    "department": ticket.department,
                    "n_options": len(q["options"]),
                    "baseline_keyword_prediction": keyword,
                    "baseline_heuristic_answer": heuristic,
                    "category_of": scheme["category_of"] if arm == "novel" else None,
                },
            )
        )
    return calls


def _override_calls(
    *, tickets: list[semantic.Ticket], partner: dict[str, str], inverted: bool
) -> list[Call]:
    """Both arms ask about the same team on the same ticket and differ in one
    sentence, so a difference between them cannot come from which team was named
    and the definition-following rate is not a yes-rate in disguise."""
    condition = "override-inverted" if inverted else "override-conventional"
    calls: list[Call] = []
    for index, ticket in enumerate(tickets):
        own = ticket.department
        other = partner[own]
        # Polarity alternates so the definition answer is yes on half the items
        # of each arm; the majority-class baseline is then exactly 0.5.
        asked = own if index % 2 == 0 else other
        prior_answer = asked == own
        definition_answer = (asked == own) if not inverted else (asked == other)
        instructions = (
            f"{_convention(own, other, inverted=inverted)}\n\n"
            f"{semantic.NOUL_QUESTION.format(label=semantic.DEPARTMENT_LABELS[asked])}"
        )
        calls.append(
            Call(
                experiment=EXPERIMENT,
                condition=condition,
                instance_id=f"e8-override-{index:05d}",
                state=ticket.as_state(),
                questions={KEY: noul_question(instructions)},
                meta={
                    "truth": {KEY: definition_answer},
                    "difficulty": {"condition": condition},
                    "index": index,
                    "level": LEVEL,
                    "inverted": inverted,
                    "template_id": ticket.template_id,
                    "department": own,
                    "partner_department": other,
                    "asked_department": asked,
                    "definition_answer": definition_answer,
                    "prior_answer": prior_answer,
                },
            )
        )
    return calls


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass
class Scored:
    """One cell's answers, keyed by item index.

    Keyed by index rather than accumulated into lists so that paired comparisons
    between cells intersect on index and never silently compare different items.
    """

    key: str
    kind: str
    difficulty: dict
    by_index: dict[int, dict] = field(default_factory=dict)
    failures: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.by_index)

    def note_failure(self, result: CallResult) -> None:
        self.failures += 1
        reason = (result.error or f"HTTP {result.http_status}")[:80]
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def _collect(results: list[CallResult], key: str, kind: str, difficulty: dict) -> Scored:
    """Extract one row per successful call. A failed call is counted and dropped;
    nothing is defaulted, so a failure can never reach a metric as an answer.

    Every row exposes `p`, the probability that feeds calibration, whatever the
    question type produced it: the noul probability for the routing arm and the
    probability on the chosen rubric point for the urgency arm. `p_means` records
    which, so the report cannot print the two in one column unlabelled.
    """
    cell = Scored(key=key, kind=kind, difficulty=difficulty)
    for r in results:
        if not r.ok:
            cell.note_failure(r)
            continue
        answer = r.answers[KEY]
        meta = r.call.meta
        truth = meta["truth"][KEY]
        row: dict[str, Any] = {
            "truth": truth,
            "latency_s": r.latency_s,
            "prior_answer": meta.get("prior_answer"),
        }
        if answer["type"] == "noul":
            keyword = meta.get("baseline_keyword_prediction")
            asked = meta.get("asked_department")
            row.update(
                {
                    "p": float(answer["p"]),
                    "p_means": "noul probability",
                    "predicted": bool(answer["predicted"]),
                    "correct": bool(answer["predicted"]) == bool(truth),
                    "confidence": None,
                    "heuristic_correct": (
                        ((keyword == asked) == bool(truth)) if asked is not None else None
                    ),
                }
            )
        elif answer["type"] == "score":
            chosen = answer["chosen"]
            row.update(
                {
                    "p": answer["max_probability"],
                    "p_means": "probability on the chosen rubric point",
                    "predicted": chosen,
                    "correct": chosen == truth,
                    "confidence": answer.get("confidence"),
                    "score": answer.get("score"),
                    "rank_error": (
                        abs(semantic.SEVERITIES.index(chosen) - semantic.SEVERITIES.index(truth))
                        if chosen in semantic.SEVERITIES and truth in semantic.SEVERITIES
                        else None
                    ),
                    # The urgency arm has no cheap keyword heuristic: the keyword
                    # vocabulary predicts departments, not severities, so there is
                    # nothing to report rather than a heuristic worth zero.
                    "heuristic_correct": None,
                }
            )
        else:
            heuristic = meta.get("baseline_heuristic_answer")
            row.update(
                {
                    "p": answer.get("max_probability"),
                    "p_means": "probability on the chosen option",
                    "predicted": answer["chosen"],
                    "correct": answer["chosen"] == truth,
                    "confidence": answer.get("confidence"),
                    "max_probability": answer.get("max_probability"),
                    "n_options": meta.get("n_options"),
                    "heuristic_correct": (heuristic == truth) if heuristic is not None else False,
                }
            )
        cell.by_index[int(meta["index"])] = row
    return cell


def _cell_metrics(cell: Scored, task: Task | None = None) -> dict:
    """Accuracy, calibration and discrimination for one cell.

    Every metric that needs data guards on having it: a cell whose calls all
    failed returns n=0 and no numbers, rather than a zero that would read as a
    measured floor.
    """
    rows = [cell.by_index[i] for i in sorted(cell.by_index)]
    out: dict[str, Any] = {
        "n": len(rows),
        "failures": cell.failures,
        "underpowered_for_ece": len(rows) < ECE_SAMPLE_FLOOR,
    }
    if task is not None:
        out["calibration_is"] = task.calibration
    if not rows:
        return out

    correct = [r["correct"] for r in rows]
    lo, hi = metrics.wilson_interval(sum(correct), len(rows))
    out.update(
        {
            "accuracy": metrics.accuracy(correct),
            "accuracy_ci": [lo, hi],
            "majority_baseline": metrics.majority_baseline([r["truth"] for r in rows]),
            "latency_p50_p95": metrics.percentiles([r["latency_s"] for r in rows]),
        }
    )
    if task is not None:
        out["chance_baseline"] = task.chance
    elif rows[0].get("n_options"):
        out["chance_baseline"] = metrics.random_baseline(rows[0]["n_options"])
    elif cell.kind == "noul":
        out["chance_baseline"] = 0.5
    heur = [r["heuristic_correct"] for r in rows if r.get("heuristic_correct") is not None]
    if heur:
        out["heuristic_baseline"] = sum(heur) / len(heur)

    # An answer that arrived without a distribution has no probability to
    # calibrate. Those rows are dropped from the calibration metrics only -- they
    # are real answers and still count toward accuracy -- and the count is
    # reported so a cell computing its ECE on a subset cannot do it quietly.
    scorable = [r for r in rows if r.get("p") is not None]
    out["n_with_probability"] = len(scorable)
    if scorable:
        # For the routing arm the outcome is the truth and `p` is P(yes); for the
        # urgency arm the outcome is whether the chosen point was right and `p` is
        # the probability it carried. Both are a probability against the event it
        # is a probability of, which is what ECE needs.
        ps = [r["p"] for r in scorable]
        ys = [
            bool(r["truth"]) if cell.kind == "noul" else bool(r["correct"]) for r in scorable
        ]
        bins = metrics.reliability_bins(ps, ys)
        out.update(
            {
                "ece": metrics.ece(ps, ys),
                "brier": metrics.brier(ps, ys),
                "auroc": metrics.auroc(ps, ys),
                "mean_p": sum(ps) / len(ps),
                "reliability_bins": bins,
                "reliability_monotone": (
                    metrics.is_monotone(bins)
                    if len(scorable) >= MIN_N_FOR_MONOTONICITY
                    else None
                ),
            }
        )
    ranks = [r["rank_error"] for r in rows if r.get("rank_error") is not None]
    if ranks:
        out["mean_abs_rank_error"] = sum(ranks) / len(ranks)
    confs = [r["confidence"] for r in rows if r.get("confidence") is not None]
    if confs:
        out["ece_from_confidence"] = metrics.ece(confs, [bool(r["correct"]) for r in rows])
    return out


def _paired(a: Scored, b: Scored) -> tuple[list[dict], list[dict]]:
    """Rows of two cells restricted to the items both answered, in index order."""
    shared = sorted(set(a.by_index) & set(b.by_index))
    return [a.by_index[i] for i in shared], [b.by_index[i] for i in shared]


def _ece_of_rows(rows: list) -> float:
    ys = [bool(r["truth"]) if r["p_means"] == "noul probability" else bool(r["correct"]) for r in rows]
    return metrics.ece([r["p"] for r in rows], ys)


def _accuracy_of_rows(rows: list) -> float:
    return sum(1 for r in rows if r["correct"]) / len(rows)


def _delta(cell: Scored, base: Scored) -> dict:
    """Paired accuracy and ECE differences of a cell against its arm's zero-shot cell."""
    a, b = _paired(cell, base)
    out: dict[str, Any] = {"n_paired": len(a)}
    if not a:
        return out
    out["accuracy_gain"] = _accuracy_of_rows(a) - _accuracy_of_rows(b)
    out["mcnemar_p"] = metrics.mcnemar([r["correct"] for r in a], [r["correct"] for r in b])
    both = [
        (x, y) for x, y in zip(a, b) if x.get("p") is not None and y.get("p") is not None
    ]
    if both:
        pa = [x for x, _ in both]
        pb = [y for _, y in both]
        out["n_paired_with_probability"] = len(both)
        # Rounded only to clear the floating-point residue two identical ECEs
        # leave behind: without it an unchanged cell reports -7.6e-17, which
        # prints as a number and reads as a direction.
        out["ece_delta"] = round(_ece_of_rows(pa) - _ece_of_rows(pb), 12)
        if len(both) >= MIN_N_FOR_MONOTONICITY:
            boot = metrics.paired_bootstrap(
                pa, pb, _ece_of_rows, n_boot=BOOTSTRAP_RESAMPLES, seed=config.MASTER_SEED
            )
            out["ece_delta_ci"] = [boot["ci_lo"], boot["ci_hi"]]
            out["ece_delta_p"] = boot["p_value"]
    return out


# --------------------------------------------------------------------------
# Anomaly probes
# --------------------------------------------------------------------------


def _quantization(ps: list[float]) -> dict:
    if not ps:
        return {"n": 0}
    two_dp = sum(1 for p in ps if abs(p * 100 - round(p * 100)) < 1e-9)
    counts: dict[float, int] = {}
    for p in ps:
        counts[p] = counts.get(p, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    return {
        "n": len(ps),
        "fraction_on_2dp_grid": two_dp / len(ps),
        "distinct_values": len(counts),
        "most_common": [[v, c, c / len(ps)] for v, c in top],
    }


def _confidence_vs_max(rows: list[dict]) -> dict:
    pairs = [
        (r["confidence"], r["p"])
        for r in rows
        if r.get("confidence") is not None and r.get("p") is not None
    ]
    if not pairs:
        return {"n": 0}
    diffs = [c - m for c, m in pairs]
    return {
        "n": len(pairs),
        "fraction_differing": sum(1 for d in diffs if abs(d) > 0.005) / len(diffs),
        "mean_difference": sum(diffs) / len(diffs),
        "max_abs_difference": max(abs(d) for d in diffs),
    }


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def _icl_cells(task: Task) -> list[tuple[str, Task, dict, str, int, bool]]:
    """Descriptors for every cell of one arm: zero-shot, both channels at each
    count, both channels poisoned at the poisoning count.

    Descriptors rather than built calls, because a k=10 state-channel call
    carries ten ticket bodies and holding every cell's calls at once is tens of
    megabytes for no reason. `run` builds a cell's calls when it is about to
    send them and lets them go afterwards.
    """

    def cell(channel: str, k: int, poisoned: bool):
        return (
            _cell_key(task, channel, k, poisoned),
            task,
            # The zero-shot cell belongs to no channel: with no examples the two
            # channels emit an identical request, so it is run once and labelled
            # "none" rather than attributed to whichever channel built it.
            {
                "task": task.name,
                "channel": channel if k else "none",
                "examples": k,
                "poisoned": poisoned,
            },
            channel,
            k,
            poisoned,
        )

    cells = [cell(CHANNELS[0], 0, False)]
    cells += [cell(c, k, False) for c in CHANNELS for k in EXAMPLE_COUNTS if k]
    cells += [cell(c, POISON_AT, True) for c in CHANNELS]
    return cells


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_cal = config.n(config.SIZES.calibration)
    n_acc = config.n(config.SIZES.accuracy)

    items_seed = config.seed_for(EXPERIMENT, "items")
    exemplar_seed = config.seed_for(EXPERIMENT, "exemplars")
    rubric_seed = config.seed_for(EXPERIMENT, "rubric")
    override_seed = config.seed_for(EXPERIMENT, "override")

    pool = semantic.tickets(seed=exemplar_seed, count=EXEMPLAR_POOL, level=LEVEL)
    scheme = _novel_scheme(rubric_seed, pool)
    partner = _swap_pairs(override_seed)

    # Both arms draw the same tickets, since `assignment_for` depends only on the
    # seed and the index. The two arms therefore ask two questions of one set of
    # states, which is what makes the calibration arm and the steerability arm
    # comparable to each other.
    items_by_task = {
        task.name: semantic.generate(
            difficulty={"level": LEVEL, "question_type": task.kind},
            seed=items_seed,
            count=n_cal,
        )
        for task in TASKS
    }

    novel_tickets = semantic.tickets(
        seed=config.seed_for(EXPERIMENT, "novel-items"), count=n_acc, level=LEVEL
    )
    override_tickets = semantic.tickets(
        seed=config.seed_for(EXPERIMENT, "override-items"), count=n_acc, level=LEVEL
    )
    rubric_plan = [
        ("rubric-novel", "choice",
         _novel_calls(tickets=novel_tickets, scheme=scheme, seed=rubric_seed, arm="novel")),
        ("rubric-conventional", "choice",
         _novel_calls(tickets=novel_tickets, scheme=scheme, seed=rubric_seed, arm="conventional")),
        ("override-inverted", "noul",
         _override_calls(tickets=override_tickets, partner=partner, inverted=True)),
        ("override-conventional", "noul",
         _override_calls(tickets=override_tickets, partner=partner, inverted=False)),
    ]

    cells: dict[str, Scored] = {}
    with JevClient(run_dir=run_dir, log_name="e8.jsonl") as client:
        for task in TASKS:
            for key, _, difficulty, channel, k, poisoned in _icl_cells(task):
                calls = _icl_calls(
                    task=task,
                    items=items_by_task[task.name],
                    pool=pool,
                    exemplar_seed=exemplar_seed,
                    channel=channel,
                    k=k,
                    poisoned=poisoned,
                )
                results = client.run(calls, progress_every=100, label=f"E8/{key}")
                cells[key] = _collect(results, key, task.kind, difficulty)
                del calls, results
        for key, kind, calls in rubric_plan:
            results = client.run(calls, progress_every=100, label=f"E8/{key}")
            cells[key] = _collect(results, key, kind, {"condition": key})
        summary = client.summary()

    return _analyse(
        cells=cells,
        plot_dir=plot_dir,
        scheme=scheme,
        partner=partner,
        summary=summary,
        n_cal=n_cal,
        n_acc=n_acc,
    )


def _analyse(
    *,
    cells: dict[str, Scored],
    plot_dir: Path,
    scheme: dict,
    partner: dict[str, str],
    summary: dict,
    n_cal: int,
    n_acc: int,
) -> dict:
    anomalies: list[str] = []
    notes: list[str] = [
        "There is no request-level instructions field on this endpoint -- sending "
        "one is a 400 -- so the 'examples in instructions' channel is the "
        "per-question instructions string. The plan's channel comparison is "
        "therefore narrower than written: per-question instructions against "
        "state, not a system prompt against state. P23 is scored on that.",
        "The semantic control is saturated for this model at its hard level: "
        "40/40 on the yes/no routing question and 40/40 on the 8-way routing "
        "choice in a pre-run probe, against keyword-heuristic baselines of 0.60 "
        "and 0.43. The ICL sweep therefore runs on two questions from that "
        "generator -- routing for calibration, where a noul answer gives the "
        "plan's own ECE, and 4-point urgency for steerability, where 0.90 "
        "zero-shot against a 0.25 chance baseline leaves accuracy room to move.",
        "The routing arm's rows are graded by the calibration rubric and the "
        "urgency arm's by the E8 rubric. The E8 ladder is written in terms of "
        "accuracy_gain, and on a task at ceiling its 'examples have no effect' "
        "disqualifier would fire on the ceiling rather than on a model ignoring "
        "its examples.",
        "Each arm's zero-shot cell is run once and is the shared baseline for "
        "both channels, because with no examples the two channels emit an "
        "identical request. Every gain and delta is paired against it on the "
        "same items.",
        "What a state-channel delta contains. Examples cannot be put in the "
        "state without changing the state's shape: the zero-shot state is the "
        "bare ticket, and a state-channel state at k>0 is {'examples': [...], "
        "'ticket': <that same bare ticket, unaltered>}. So a state-channel "
        "delta against zero-shot is the examples plus that wrapper, and cannot "
        "separate them. The instructions channel leaves the state untouched at "
        "every k, which makes it the cleaner arm for reading a delta against "
        "zero-shot; the two channels are only ever compared to each other at "
        "equal k, where both carry their examples.",
        "The two arms report different kinds of ECE and the numbers are not "
        "interchangeable: the routing arm's is a probability against the outcome "
        "it is a probability of, the urgency arm's is top-label calibration.",
        "The overall E8 tier is a composite: its accuracy_gain comes from the "
        "urgency arm, the only one where accuracy can move, and its ece, "
        "ece_delta and reliability_monotone from the routing arm, the only one "
        "with a clean probability ECE. The rubric's rungs are written as "
        "'examples improve accuracy and leave ECE unchanged', and neither arm on "
        "its own can answer both halves. The calibration half is taken from one "
        "arm entire because the rubric's non-monotone rule is a disqualifier: it "
        "sends an experiment to \"Doesn't work\" outright, so it has to read the "
        "curve the ece_delta was measured on rather than the urgency arm's "
        "weaker top-label one. The row records which cell each half came from.",
    ]

    task_of = {t.name: t for t in TASKS}
    cell_metrics: dict[str, dict] = {}
    for key, cell in cells.items():
        task = task_of.get(str(cell.difficulty.get("task")))
        cell_metrics[key] = _cell_metrics(cell, task)

    icl_keys: dict[str, list[str]] = {}
    poison_keys: dict[str, list[str]] = {}
    for task in TASKS:
        base_key = _cell_key(task, "none", 0, False)
        icl_keys[task.name] = [
            _cell_key(task, c, k, False) for c in CHANNELS for k in EXAMPLE_COUNTS if k
        ]
        poison_keys[task.name] = [_cell_key(task, c, POISON_AT, True) for c in CHANNELS]
        for key in icl_keys[task.name] + poison_keys[task.name]:
            cell_metrics[key].update(_delta(cells[key], cells[base_key]))

    # -- the channel an operator would deploy -----------------------------
    # Fixed before the numbers arrive: the channel is chosen on the arm where
    # accuracy can actually move, since ICL is used for accuracy. If the two
    # channels tie there, the tie is broken toward the channel with the worse
    # routing-arm ECE delta, because with nothing to choose on accuracy the
    # operator should be shown the calibration risk rather than the flattering
    # half of it.
    def accuracy_at_max(task: Task, channel: str) -> float | None:
        return cell_metrics[_cell_key(task, channel, MAX_EXAMPLES, False)].get("accuracy")

    scored_channels = [c for c in CHANNELS if accuracy_at_max(URGENCY, c) is not None]
    if not scored_channels:
        headline_channel = CHANNELS[0]
    else:
        best = max(accuracy_at_max(URGENCY, c) for c in scored_channels)
        tied = [c for c in scored_channels if accuracy_at_max(URGENCY, c) == best]
        headline_channel = max(
            tied,
            key=lambda c: cell_metrics[_cell_key(ROUTING, c, MAX_EXAMPLES, False)].get(
                "ece_delta"
            )
            or 0.0,
        )

    headline_key = _cell_key(ROUTING, headline_channel, MAX_EXAMPLES, False)
    hm = cell_metrics[headline_key]
    base_routing = cell_metrics[_cell_key(ROUTING, "none", 0, False)]
    base_urgency = cell_metrics[_cell_key(URGENCY, "none", 0, False)]
    steer_key = _cell_key(URGENCY, headline_channel, MAX_EXAMPLES, False)

    # -- poisoning, on both arms ------------------------------------------
    poisoning: dict[str, dict] = {}
    for task in TASKS:
        clean_key = _cell_key(task, headline_channel, POISON_AT, False)
        dirty_key = _cell_key(task, headline_channel, POISON_AT, True)
        clean, dirty = _paired(cells[clean_key], cells[dirty_key])
        entry: dict[str, Any] = {
            "channel": headline_channel,
            "fraction_mislabelled": POISON_FRACTION,
            "n_paired": len(clean),
        }
        if clean:
            entry["accuracy_drop"] = _accuracy_of_rows(clean) - _accuracy_of_rows(dirty)
            entry["mcnemar_p"] = metrics.mcnemar(
                [r["correct"] for r in clean], [r["correct"] for r in dirty]
            )
            entry["ece_change"] = _ece_of_rows(dirty) - _ece_of_rows(clean)
            entry["headroom"] = 1.0 - _accuracy_of_rows(clean)
        poisoning[task.name] = entry
    poison_sensitivity = poisoning[URGENCY.name].get("accuracy_drop")

    # -- rubric conditions -------------------------------------------------
    novel = _cell_metrics(cells["rubric-novel"])
    novel_control = _cell_metrics(cells["rubric-conventional"])
    novel_rows, control_rows = _paired(cells["rubric-novel"], cells["rubric-conventional"])
    if novel_rows:
        # Did it apply the definition to its own belief about the department, or
        # not apply the definition at all? This separates a definition failure
        # from a routing failure, which raw accuracy cannot.
        agree = sum(
            1
            for a, b in zip(novel_rows, control_rows)
            if scheme["category_of"].get(b["predicted"]) == a["predicted"]
        )
        novel["consistency_with_own_routing"] = agree / len(novel_rows)
        novel["n_paired_with_routing"] = len(novel_rows)
        novel["routing_accuracy"] = _accuracy_of_rows(control_rows)

    inverted = _cell_metrics(cells["override-inverted"])
    conventional = _cell_metrics(cells["override-conventional"])
    inv_rows = [cells["override-inverted"].by_index[i] for i in sorted(cells["override-inverted"].by_index)]
    if inv_rows:
        follow = sum(1 for r in inv_rows if r["correct"])
        lo, hi = metrics.wilson_interval(follow, len(inv_rows))
        inverted.update(
            {
                "following_rate": follow / len(inv_rows),
                "following_ci": [lo, hi],
                "prior_rate": 1.0 - follow / len(inv_rows),
                "yes_rate": sum(1 for r in inv_rows if r["predicted"]) / len(inv_rows),
            }
        )
        # Split by which answer the prior would give, so a model that simply
        # answers yes cannot look like a definition follower.
        for label, want in (("when_prior_says_yes", True), ("when_prior_says_no", False)):
            subset = [r for r in inv_rows if r["prior_answer"] is want]
            inverted[label] = (
                sum(1 for r in subset if r["correct"]) / len(subset) if subset else None
            )
            inverted[f"{label}_n"] = len(subset)

    # -- tiers by example count -------------------------------------------
    def tier_metrics(key: str, task: Task) -> dict:
        m = cell_metrics[key]
        base = cell_metrics[_cell_key(task, "none", 0, False)]
        out: dict[str, Any] = {
            "accuracy": m.get("accuracy"),
            "accuracy_gain": m.get("accuracy_gain"),
            "ece": m.get("ece"),
            "ece_delta": m.get("ece_delta"),
            "zeroshot_accuracy": base.get("accuracy"),
            "chance_baseline": m.get("chance_baseline"),
            "heuristic_baseline": m.get("heuristic_baseline"),
            "n": m.get("n"),
        }
        if m.get("reliability_monotone") is not None:
            out["reliability_monotone"] = m["reliability_monotone"]
        return {k: v for k, v in out.items() if v is not None}

    # The routing arm has no accuracy headroom, so its rows are graded on their
    # reliability curve alone; the urgency arm carries the E8 ladder.
    rubric_for = {ROUTING.name: "calibration", URGENCY.name: EXPERIMENT}
    by_difficulty: list[dict] = []
    crossings: dict[str, Any] = {}
    for task in TASKS:
        for channel in CHANNELS:
            sweep = [
                (
                    {"task": task.name, "channel": channel, "examples": k},
                    tier_metrics(_cell_key(task, channel, k, False), task),
                )
                for k in EXAMPLE_COUNTS
                if k
            ]
            results = tiers.tier_by_difficulty(rubric_for[task.name], sweep)
            by_difficulty.extend(r.to_json() for r in results)
            # A cell whose calls all failed still gets a row, so the gap is
            # visible, but it must not contribute a crossing: the residual tier
            # of a cell with no data would otherwise be reported as the
            # difficulty at which the model fell over.
            #
            # `is_reportable` is not sufficient on its own. It asks only whether
            # a baseline and an n are present, and a dead cell has n=0 -- present
            # -- while the calibration rubric supplies its baseline by default.
            # A dead routing cell therefore passed that test and contributed a
            # "Human->Bad" crossing built out of nothing. The n>0 test is what
            # actually distinguishes a measured tier from a residual one.
            graded = [r for r in results if r.is_reportable and (r.n or 0) > 0]
            if len(graded) < 2:
                continue
            for boundary, crossing in tiers.boundary_crossings(
                rubric_for[task.name], graded
            ).items():
                if crossing is None:
                    continue
                crossings[f"{task.name}/{channel}: {boundary}"] = crossing.difficulty
                if crossing.recovered_at is not None:
                    anomalies.append(
                        f"{task.name}/{channel} falls through {boundary} at "
                        f"{crossing.difficulty} and climbs back at {crossing.recovered_at}; "
                        "the knee is not clean, so treat the crossing point as approximate."
                    )
        for key in poison_keys[task.name]:
            by_difficulty.append(
                tiers.assign(
                    rubric_for[task.name],
                    tier_metrics(key, task),
                    difficulty=cells[key].difficulty,
                ).to_json()
            )

    # The calibration half of the composite comes from the routing arm entire,
    # not from its delta alone. Two reasons, and the second is the one that
    # could have changed a grade: an `ece` from the urgency arm printed beside an
    # `ece_delta` from the routing arm puts top-label calibration and
    # probability-against-outcome calibration in adjacent columns of one row,
    # where they read as the same quantity; and the rubric's non-monotone
    # disqualifier sends an experiment to "Doesn't work" outright, so it has to
    # read the curve the delta was measured on. A ragged top-label curve is a
    # much weaker finding than a ragged probability curve and must not spend the
    # whole experiment's grade.
    routing_headline = tier_metrics(headline_key, ROUTING)
    overall = tiers.assign(
        EXPERIMENT,
        {
            **tier_metrics(steer_key, URGENCY),
            "ece": routing_headline.get("ece"),
            "reliability_monotone": routing_headline.get("reliability_monotone"),
            "ece_delta": hm.get("ece_delta"),
            "novel_rubric_following": novel.get("accuracy"),
            "inverted_rubric_following": inverted.get("following_rate"),
            "poison_sensitivity": poison_sensitivity,
        },
        difficulty={
            "condition": "overall",
            "accuracy_from": steer_key,
            "ece_from": headline_key,
        },
    )
    routing_calibration_tier = (
        tiers.assign(
            "calibration",
            tier_metrics(headline_key, ROUTING),
            difficulty={"condition": headline_key},
        ).to_json()
        if hm.get("ece") is not None
        else None
    )

    anomalies.extend(
        _probe_anomalies(
            cells=cells,
            cell_metrics=cell_metrics,
            base_routing=base_routing,
            headline=hm,
        )
    )

    plot_paths = _plots(
        plot_dir=plot_dir,
        cell_metrics=cell_metrics,
        headline_channel=headline_channel,
        anomalies=anomalies,
    )

    preds = _score_predictions(
        cells=cells,
        cell_metrics=cell_metrics,
        base_routing=base_routing,
        base_urgency=base_urgency,
        headline_key=headline_key,
        steer_key=steer_key,
        headline_channel=headline_channel,
        inverted=inverted,
        conventional=conventional,
    )

    total_failures = sum(c.failures for c in cells.values())
    reasons: dict[str, int] = {}
    for c in cells.values():
        for reason, count in c.reasons.items():
            reasons[reason] = reasons.get(reason, 0) + count

    return {
        "experiment": EXPERIMENT,
        "question": (
            "Can worked examples steer the model, and does steering cost it the "
            "calibration that is the reason to use it?"
        ),
        "headline": {
            "metric": (
                f"ECE delta at {MAX_EXAMPLES} examples, routing arm, {headline_channel} "
                "channel "
                "(probability against outcome)"
            ),
            "value": hm.get("ece_delta"),
            "baseline": base_routing.get("ece"),
            "baseline_name": "zero-shot ECE on the same items",
            "n": hm.get("n"),
        },
        "by_difficulty": by_difficulty,
        "boundary_crossings": crossings,
        "plots": plot_paths,
        "predictions": preds,
        "anomalies": anomalies,
        "failures": {"calls": total_failures, "excluded": total_failures, "reasons": reasons},
        "notes": notes,
        "overall_tier": overall.to_json(),
        "routing_calibration_tier": routing_calibration_tier,
        "tier_rubrics": rubric_for,
        "headline_cell": headline_key,
        "steerability_cell": steer_key,
        "tasks": {
            t.name: {
                "question_type": t.kind,
                "chance_baseline": t.chance,
                "calibration_is": t.calibration,
                "role": t.role,
            }
            for t in TASKS
        },
        "sample_sizes": {
            "icl_cell": n_cal,
            "rubric_arm": n_acc,
            "scale": config.SCALE,
            "ece_sample_floor": ECE_SAMPLE_FLOOR,
            "min_n_for_monotonicity": MIN_N_FOR_MONOTONICITY,
            "calls_at_this_scale": len(TASKS) * 9 * n_cal + 4 * n_acc,
            "calls_at_scale_1": len(TASKS) * 9 * config.SIZES.calibration
            + 4 * config.SIZES.accuracy,
        },
        "conditions": cell_metrics,
        "calibration_under_icl": {
            task.name: {
                "calibration_is": task.calibration,
                "zeroshot_ece": cell_metrics[_cell_key(task, "none", 0, False)].get("ece"),
                "zeroshot_accuracy": cell_metrics[_cell_key(task, "none", 0, False)].get(
                    "accuracy"
                ),
                "by_cell": {
                    k: {
                        m: cell_metrics[k].get(m)
                        for m in (
                            "n",
                            "accuracy",
                            "accuracy_gain",
                            "ece",
                            "ece_delta",
                            "ece_delta_ci",
                            "brier",
                            "auroc",
                            "mcnemar_p",
                            "underpowered_for_ece",
                        )
                    }
                    for k in icl_keys[task.name] + poison_keys[task.name]
                },
            }
            for task in TASKS
        },
        "poisoning": {
            **poisoning,
            "note": (
                "Poisoning swaps the labels of two examples that carry different "
                "labels, so the block's label marginal is unchanged and the drop "
                "measures the example-to-label mapping rather than a shifted "
                "prior. On the routing arm, where accuracy is already at ceiling, "
                "only a drop is informative: no drop is consistent with the "
                "examples being ignored and with the task being too easy to "
                "perturb, and those cannot be told apart there."
            ),
        },
        "novel_rubric": {**novel, "scheme": scheme},
        "inverted_rubric": {
            **inverted,
            "swap_pairs": sorted({tuple(sorted((a, b))) for a, b in partner.items()}),
            "control": conventional,
        },
        "rubric_conventional_control": novel_control,
        "run": summary,
    }


# --------------------------------------------------------------------------
# Anomaly reporting
# --------------------------------------------------------------------------


def _probe_anomalies(
    *,
    cells: dict[str, Scored],
    cell_metrics: dict[str, dict],
    base_routing: dict,
    headline: dict,
) -> list[str]:
    """The things worth reporting that no condition of E8 was designed to test.

    The plan requires a separate "unexpected behaviors" section and names what to
    watch for; three of its items are visible from E8's own answers -- probability
    quantization, confidence diverging from the maximum option probability, and a
    condition whose numbers rest on too few samples to mean anything. Each probe
    reports what it found either way rather than only when it fires, because
    "confidence tracked max-probability on every answer" is as much a result as
    the divergence would have been.
    """
    found: list[str] = []

    noul_ps = [
        r["p"]
        for cell in cells.values()
        if cell.kind == "noul"
        for r in cell.by_index.values()
        if r.get("p") is not None
    ]
    quant = _quantization(noul_ps)
    if quant.get("n") and quant["fraction_on_2dp_grid"] > 0.999:
        found.append(
            f"Every noul probability E8 saw ({quant['n']} of them, "
            f"{quant['distinct_values']} distinct values) lies on a 2-decimal "
            "grid. Consistent with quantization to 2dp, which the plan flags as a "
            "thing to watch; it also puts a floor of roughly 0.005 under any ECE."
        )
    elif quant.get("n"):
        found.append(
            f"{1 - quant['fraction_on_2dp_grid']:.3f} of noul probabilities are off "
            "the 2-decimal grid, so the quantization seen elsewhere does not hold here."
        )

    conf_rows = [
        r
        for cell in cells.values()
        for r in cell.by_index.values()
        if r.get("confidence") is not None
    ]
    conf = _confidence_vs_max(conf_rows)
    if conf.get("n") and conf["fraction_differing"] > 0.02:
        found.append(
            f"On choice and score answers, confidence differs from the maximum "
            f"option probability on {conf['fraction_differing']:.3f} of "
            f"{conf['n']} answers (mean difference {conf['mean_difference']:+.4f}, "
            f"largest {conf['max_abs_difference']:.4f}). The plan lists this as a "
            "behaviour to watch for."
        )
    elif conf.get("n"):
        found.append(
            f"On {conf['n']} choice and score answers, confidence tracks the maximum "
            f"option probability on {1 - conf['fraction_differing']:.3f} of them."
        )

    if base_routing.get("accuracy") is not None and base_routing["accuracy"] >= 0.99:
        found.append(
            f"The routing arm is at {base_routing['accuracy']:.4f} accuracy with no "
            "examples at all, against a keyword heuristic of "
            + (
                f"{base_routing['heuristic_baseline']:.4f}"
                if base_routing.get("heuristic_baseline") is not None
                else "n/a"
            )
            + ". The semantic control's hardest level does not challenge this model, "
            "so accuracy there cannot move and the arm reports calibration only. "
            "E4 and E9 should not assume the hard level is hard."
        )

    thin = [
        f"{k} (n={cell_metrics[k]['n']})"
        for k in cell_metrics
        if cell_metrics[k].get("underpowered_for_ece")
        and cell_metrics[k].get("n")
        and cell_metrics[k].get("ece") is not None
    ]
    if thin:
        found.append(
            f"{len(thin)} cell(s) fall below the plan's {ECE_SAMPLE_FLOOR}-item "
            "minimum for a reported ECE: "
            + ", ".join(thin[:6])
            + (f" and {len(thin) - 6} more" if len(thin) > 6 else "")
            + f". Their ECE is reported but is under-sampled. Below "
            f"{MIN_N_FOR_MONOTONICITY} items monotonicity is not assessed either, "
            "so a thin cell cannot be sent to the worst tier by a curve that is "
            "ragged only because it is thin."
        )
    dead = sorted(k for k in cell_metrics if not cell_metrics[k].get("n"))
    if dead and len(dead) < len(cell_metrics):
        found.append(
            f"{len(dead)} condition(s) produced no scorable answer at all: "
            + ", ".join(dead[:6])
            + (f" and {len(dead) - 6} more" if len(dead) > 6 else "")
            + ". Their row carries the rubric's residual tier rather than a "
            "measurement, and they contribute no boundary crossing. What is "
            "missing there is data, not model performance."
        )
    if not any(cell_metrics[k].get("n") for k in cell_metrics):
        found.append(
            "No cell produced a single scorable answer. Every number below is "
            "absent rather than zero, and no tier here is a model result."
        )
    ci = headline.get("ece_delta_ci")
    if ci and ci[0] <= 0.0 <= ci[1]:
        found.append(
            f"The headline ECE delta ({headline.get('ece_delta'):+.4f}) has a bootstrap "
            f"95% interval of [{ci[0]:+.4f}, {ci[1]:+.4f}], which contains zero. "
            "The direction is not resolved by this sample."
        )

    return found


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _plots(
    *,
    plot_dir: Path,
    cell_metrics: dict[str, dict],
    headline_channel: str,
    anomalies: list[str],
) -> list[str]:
    paths: list[str] = []

    def emit(fn, name: str) -> None:
        try:
            fn(plot_dir / name)
        except Exception as exc:  # a plot must never cost the run its data
            anomalies.append(f"Plot {name} could not be drawn: {type(exc).__name__}: {exc}")
        else:
            paths.append(f"plots/{name}")

    xs = list(EXAMPLE_COUNTS)
    for task, metric, ylabel, baseline_of in (
        (ROUTING, "ece", "ECE (10 bins, probability against outcome)", "ece"),
        (ROUTING, "accuracy", "accuracy", None),
        (URGENCY, "ece", "ECE (10 bins, top-label)", "ece"),
        (URGENCY, "accuracy", "accuracy", None),
    ):
        keys = {
            c: [_cell_key(task, c if k else "none", k, False) for k in xs] for c in CHANNELS
        }
        if not all(
            cell_metrics[k].get(metric) is not None for ks in keys.values() for k in ks
        ):
            continue
        series = {c: [cell_metrics[k][metric] for k in ks] for c, ks in keys.items()}
        ns = [cell_metrics[k].get("n", 0) for ks in keys.values() for k in ks]
        base_key = _cell_key(task, "none", 0, False)
        baseline = (
            cell_metrics[base_key].get(baseline_of)
            if baseline_of
            else cell_metrics[base_key].get("chance_baseline")
        )
        emit(
            lambda p, series=series, ylabel=ylabel, task=task, baseline=baseline, ns=ns, metric=metric, baseline_of=baseline_of, heur=cell_metrics[base_key].get(
                "heuristic_baseline"
            ): plots.curve(
                xs,
                series,
                p,
                f"E8: {metric} under in-context learning, {task.name} arm",
                "worked examples in context",
                ylabel,
                n=min(ns),
                baseline=baseline,
                baseline_label=(
                    "zero-shot ECE" if baseline_of else f"chance ({task.chance:.3g})"
                ),
                note=(
                    f"semantic control at the {LEVEL} level; the zero-shot point is "
                    "shared by both channels"
                    + (f"; keyword heuristic {heur:.3f}" if heur is not None else "")
                ),
            ),
            f"e8_{metric}_vs_examples_{task.name}.png",
        )

    wanted = [
        (_cell_key(ROUTING, "none", 0, False), "E8: reliability, routing, zero-shot"),
        (
            _cell_key(ROUTING, headline_channel, MAX_EXAMPLES, False),
            f"E8: reliability, routing, {MAX_EXAMPLES} examples in {headline_channel}",
        ),
        (_cell_key(URGENCY, "none", 0, False), "E8: reliability, urgency, zero-shot"),
        (
            _cell_key(URGENCY, headline_channel, MAX_EXAMPLES, False),
            f"E8: reliability, urgency, {MAX_EXAMPLES} examples in {headline_channel}",
        ),
        (
            _cell_key(URGENCY, headline_channel, POISON_AT, True),
            f"E8: reliability, urgency, {POISON_AT} poisoned examples in {headline_channel}",
        ),
    ]
    for key, title in wanted:
        m = cell_metrics.get(key) or {}
        if not m.get("n") or m.get("reliability_bins") is None:
            continue
        emit(
            lambda p, m=m, title=title, key=key: plots.reliability_diagram(
                m["reliability_bins"],
                p,
                title,
                condition=key,
                ece=m["ece"],
                n=m["n"],
                monotone=m.get("reliability_monotone"),
                note=(
                    f"n={m['n']} is below the plan's {ECE_SAMPLE_FLOOR}-item minimum "
                    "for a reported ECE"
                    if m.get("underpowered_for_ece")
                    else m.get("calibration_is", "")
                ),
            ),
            f"e8_reliability_{key.replace('/', '_').replace('-', '_')}.png",
        )
    return paths


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


def _score_predictions(
    *,
    cells: dict[str, Scored],
    cell_metrics: dict[str, dict],
    base_routing: dict,
    base_urgency: dict,
    headline_key: str,
    steer_key: str,
    headline_channel: str,
    inverted: dict,
    conventional: dict,
) -> list[dict]:
    """Every E8 prediction, scored, with none omitted.

    The claim as written decides the verdict, which is how the rest of this
    harness scores a prediction. Where a claim and its falsification clause do
    not cover the same ground -- P22 predicts an ECE cost of +0.03 to +0.10 and
    is falsified by "unchanged or improved", leaving +0.001 in neither -- the
    outcome states both: what was measured, whether the claim holds, and whether
    the falsifier was met. A prediction can fail without being falsified, and a
    report that cannot tell those apart is worse than one that omits the
    distinction, because it reads as though the plan had been confirmed.
    """
    out: list[dict] = []
    table = {p.id: p for p in predictions.for_experiment(EXPERIMENT)}

    # -- P22: ICL degrades calibration ------------------------------------
    hm = cell_metrics[headline_key]
    sm = cell_metrics[steer_key]
    delta = hm.get("ece_delta")
    ci = hm.get("ece_delta_ci")
    if delta is None:
        verdict = "untestable"
        outcome = (
            "No scorable pairs in the routing arm's 10-example cell, so no ECE "
            "delta exists. That is a harness or availability failure, not a model "
            "result."
        )
    else:
        band = P22_BAND[0] <= delta <= P22_BAND[1]
        falsified = delta <= P22_UNCHANGED
        verdict = "right" if band else "wrong"
        outcome = (
            f"On the routing arm, where a noul answer gives the plan's own ECE, it "
            f"went from {_num(base_routing.get('ece'), '.4f', 0)} zero-shot to "
            f"{_num(hm.get('ece'), '.4f', 0)} with {MAX_EXAMPLES} examples in the "
            f"{headline_channel} channel, a change of {delta:+.4f}"
            + (f" (bootstrap 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}])" if ci else "")
            + f". Accuracy there is pinned at {_num(base_routing.get('accuracy'), '.4f', 0)} "
            "zero-shot, so whatever the examples cost in calibration on this arm "
            "they buy nothing. On the urgency arm, where accuracy can move, ECE "
            "(top-label) went "
            + (
                f"{sm.get('ece_delta'):+.4f} while accuracy went "
                f"{sm.get('accuracy_gain'):+.4f}"
                if sm.get("ece_delta") is not None
                else "unmeasured"
            )
            + ". "
            + (
                f"The direction and the predicted {P22_BAND[0]:.2f}-{P22_BAND[1]:.2f} "
                "magnitude both hold."
                if band
                else (
                    "Calibration was unchanged -- within the "
                    f"{P22_UNCHANGED} this harness treats as a change it cannot "
                    "resolve, and about the floor 2-decimal quantization puts under "
                    "any ECE -- or improved, which is the plan's stated "
                    "falsification condition."
                    if falsified
                    else "Calibration degraded, so the plan's falsification clause is "
                    f"not met, but by {delta:+.4f}, outside the predicted "
                    f"{P22_BAND[0]:.2f}-{P22_BAND[1]:.2f} band. The prediction as "
                    "written does not hold; it was not falsified either."
                )
            )
        )
        if hm.get("underpowered_for_ece"):
            outcome += (
                f" n={hm.get('n')} is below the plan's {ECE_SAMPLE_FLOOR}-item minimum "
                "for a reported ECE, so the verdict is not load-bearing at this scale."
            )
    out.append(
        {
            "id": "P22",
            "claim": table["P22"].claim,
            "outcome": outcome,
            "verdict": verdict,
            "evidence": {
                "routing_zeroshot_ece": base_routing.get("ece"),
                "routing_ece": hm.get("ece"),
                "routing_ece_delta": delta,
                "routing_ece_delta_ci": ci,
                "routing_ece_delta_p": hm.get("ece_delta_p"),
                "routing_accuracy_gain": hm.get("accuracy_gain"),
                # Kept apart because they are not the same test: the band is the
                # claim, the clause is what the plan said would refute it, and a
                # delta can miss both.
                "predicted_band": list(P22_BAND),
                "in_predicted_band": (None if delta is None else P22_BAND[0] <= delta <= P22_BAND[1]),
                "falsification_clause": table["P22"].falsified_if,
                "falsifier_met": (None if delta is None else delta <= P22_UNCHANGED),
                "urgency_zeroshot_ece": base_urgency.get("ece"),
                "urgency_ece_delta": sm.get("ece_delta"),
                "urgency_accuracy_gain": sm.get("accuracy_gain"),
                "headline_cell": headline_key,
                "per_cell": {
                    k: {
                        "ece_delta": cell_metrics[k].get("ece_delta"),
                        "accuracy_gain": cell_metrics[k].get("accuracy_gain"),
                    }
                    for k in cell_metrics
                    if cell_metrics[k].get("ece_delta") is not None
                },
            },
        }
    )

    # -- P23: state channel beats instructions channel ---------------------
    # Compared on the urgency arm, because the routing arm is at ceiling in both
    # channels and a tie at 1.0 would say nothing about the channels.
    st_key = _cell_key(URGENCY, "state", MAX_EXAMPLES, False)
    ins_key = _cell_key(URGENCY, "instructions", MAX_EXAMPLES, False)
    a, b = _paired(cells[st_key], cells[ins_key])
    if not a:
        verdict = "untestable"
        outcome = (
            f"No urgency item was answered in both channels at {MAX_EXAMPLES} examples, "
            "so the "
            "two channels cannot be compared."
        )
        evidence: dict[str, Any] = {}
    else:
        st_acc = _accuracy_of_rows(a)
        ins_acc = _accuracy_of_rows(b)
        p = metrics.mcnemar([r["correct"] for r in a], [r["correct"] for r in b])
        verdict = "right" if st_acc > ins_acc else "wrong"
        r_st, r_ins = _paired(
            cells[_cell_key(ROUTING, "state", MAX_EXAMPLES, False)],
            cells[_cell_key(ROUTING, "instructions", MAX_EXAMPLES, False)],
        )
        outcome = (
            "There is no request-level instructions field on this endpoint, so the "
            "instructions channel tested here is the per-question instructions "
            "string, not a system prompt; the plan's expectation was about a "
            "channel that does not exist, and this scores the narrower claim. The "
            "comparison is made on the urgency arm because the routing arm is at "
            f"ceiling in both channels. At {MAX_EXAMPLES} examples the state channel scored "
            f"{st_acc:.4f} and the instructions channel {ins_acc:.4f} on the same "
            f"{len(a)} items (McNemar p={p:.4f}); top-label ECE "
            f"{_num(cell_metrics[st_key].get('ece'), '.4f', 0)} against "
            f"{_num(cell_metrics[ins_key].get('ece'), '.4f', 0)}. "
            + (
                f"On the routing arm the same comparison is {_accuracy_of_rows(r_st):.4f} "
                f"against {_accuracy_of_rows(r_ins):.4f}, both at or near ceiling. "
                if r_st
                else ""
            )
            + (
                "State wins, as predicted."
                if verdict == "right"
                else (
                    "Instructions win, which is the stated falsification condition."
                    if ins_acc > st_acc
                    else "The two channels tie, so the prediction that state beats "
                    "instructions is not borne out, although the plan's "
                    "falsification clause is not met either."
                )
            )
            + (
                f" McNemar p={p:.3f} does not resolve the channels apart on this "
                "sample, so the verdict rests on a point estimate that could go "
                "either way."
                if p > 0.05
                else ""
            )
        )
        evidence = {
            "compared_on": "urgency arm (the routing arm is at ceiling in both channels)",
            "state_accuracy": st_acc,
            "instructions_accuracy": ins_acc,
            "n_paired": len(a),
            "mcnemar_p": p,
            "state_ece": cell_metrics[st_key].get("ece"),
            "instructions_ece": cell_metrics[ins_key].get("ece"),
            "by_example_count": {
                f"{task.name}/{k}": {
                    "state_accuracy": cell_metrics[_cell_key(task, "state", k, False)].get(
                        "accuracy"
                    ),
                    "instructions_accuracy": cell_metrics[
                        _cell_key(task, "instructions", k, False)
                    ].get("accuracy"),
                }
                for task in TASKS
                for k in EXAMPLE_COUNTS
                if k
            },
            "channel_caveat": "per-question instructions, not a request-level prompt",
        }
    out.append(
        {
            "id": "P23",
            "claim": table["P23"].claim,
            "outcome": outcome,
            "verdict": verdict,
            "evidence": evidence,
        }
    )

    # -- P24: inverted rubric followed about 70% ---------------------------
    rate = inverted.get("following_rate")
    if rate is None:
        verdict = "untestable"
        outcome = "No scorable answers in the inverted-rubric arm, so no following rate exists."
    else:
        verdict = "right" if P24_WINDOW[0] <= rate <= P24_WINDOW[1] else "wrong"
        ci24 = inverted.get("following_ci") or [float("nan"), float("nan")]
        control_acc = conventional.get("accuracy")
        outcome = (
            f"The inverted rubric was followed on {rate:.3f} of {inverted.get('n')} "
            f"items (95% CI {ci24[0]:.3f}-{ci24[1]:.3f}); the prior won on "
            f"{inverted.get('prior_rate'):.3f}. The control arm, an identical "
            "request differing only in that it states the conventional scopes, "
            "scored "
            + (f"{control_acc:.3f}" if control_acc is not None else "n/a")
            + ", so "
            + (
                "the task itself is doable and the shortfall is definition-following."
                if (control_acc or 0) > 0.7
                else "part of the shortfall may be the routing task rather than the "
                "definition, and the following rate should be read against that control."
            )
            + f" The yes-rate in the inverted arm was {inverted.get('yes_rate'):.3f}, so "
            "the result is not a yes-bias reading as definition-following. "
            + (
                f"Within the plan's {P24_WINDOW[0]:.2f}-{P24_WINDOW[1]:.2f} window."
                if verdict == "right"
                else "Outside the plan's window, which is its falsification condition."
            )
            + (
                " The interval also covers the falsification region, so this sample "
                "cannot separate the prediction from its own falsification condition."
                if ci24[0] < P24_WINDOW[0] or ci24[1] > P24_WINDOW[1]
                else ""
            )
        )
    out.append(
        {
            "id": "P24",
            "claim": table["P24"].claim,
            "outcome": outcome,
            "verdict": verdict,
            "evidence": {
                "following_rate": rate,
                "following_ci": inverted.get("following_ci"),
                "prior_rate": inverted.get("prior_rate"),
                "yes_rate": inverted.get("yes_rate"),
                "following_when_prior_says_yes": inverted.get("when_prior_says_yes"),
                "following_when_prior_says_no": inverted.get("when_prior_says_no"),
                "conventional_control_accuracy": conventional.get("accuracy"),
                "n": inverted.get("n"),
            },
        }
    )

    missing = set(table) - {p["id"] for p in out}
    if missing:
        raise AssertionError(f"E8 failed to score its own predictions: {sorted(missing)}")
    return out


# --------------------------------------------------------------------------
# Terminal summary
# --------------------------------------------------------------------------


def _num(v: Any, fmt: str = ".4f", width: int = 8) -> str:
    return (f"{v:{fmt}}" if isinstance(v, (int, float)) else "n/a").rjust(width)


def format_report(result: dict) -> str:
    lines = [f"E8 -- {result['question']}"]
    h = result["headline"]
    lines.append(
        f"  HEADLINE  {h['metric']}: {_num(h['value'], '+.4f')}  against "
        f"{_num(h['baseline'])} {h['baseline_name']}, n={h['n']}"
    )
    overall = result["overall_tier"]
    lines.append(
        f"  overall tier (E8 rubric, urgency arm): {overall['tier']}"
        + ("" if overall.get("reportable") else "  [not reportable: baseline or n missing]")
        + ("" if overall.get("fully_evaluated", True) else "  [tier from a partial rule]")
    )
    rct = result.get("routing_calibration_tier")
    if rct:
        lines.append(f"  routing-arm calibration tier: {rct['tier']}")

    for task_name, arm in result["calibration_under_icl"].items():
        lines.append("")
        lines.append(f"  {task_name} arm -- ECE is {arm['calibration_is']}")
        lines.append("    cell                              n  accuracy      gain       ECE     delta")
        lines.append(
            f"    {'zeroshot':<28}{result['conditions'][f'{task_name}/zeroshot'].get('n', 0):>4}"
            f"{_num(arm['zeroshot_accuracy'], '.4f', 10)}{'--':>10}"
            f"{_num(arm['zeroshot_ece'], '.4f', 10)}{'--':>10}"
        )
        for key, cell in arm["by_cell"].items():
            lines.append(
                f"    {key.split('/', 1)[1]:<28}{cell.get('n', 0):>4}"
                f"{_num(cell.get('accuracy'), '.4f', 10)}"
                f"{_num(cell.get('accuracy_gain'), '+.4f', 10)}"
                f"{_num(cell.get('ece'), '.4f', 10)}"
                f"{_num(cell.get('ece_delta'), '+.4f', 10)}"
            )
        po = result["poisoning"].get(task_name, {})
        if po.get("accuracy_drop") is not None:
            lines.append(
                f"    20% mislabelled examples moved accuracy by "
                f"{-po['accuracy_drop']:+.4f} (McNemar p={po['mcnemar_p']:.4f}, "
                f"n={po['n_paired']}, headroom {po['headroom']:.4f})"
            )

    nov = result["novel_rubric"]
    lines.append("")
    if nov.get("accuracy") is None:
        lines.append(f"  novel rubric: no scorable answers (n={nov.get('n', 0)})")
    else:
        lines.append(
            f"  novel rubric: accuracy {nov['accuracy']:.4f} vs chance "
            f"{nov['chance_baseline']:.4f}, majority {nov['majority_baseline']:.4f}, "
            f"keyword heuristic {nov['heuristic_baseline']:.4f}"
            + (
                "; applied to its own 8-way routing answer on "
                f"{nov['consistency_with_own_routing']:.4f}"
                if nov.get("consistency_with_own_routing") is not None
                else ""
            )
        )
    inv = result["inverted_rubric"]
    if inv.get("following_rate") is not None:
        lines.append(
            f"  inverted rubric followed {inv['following_rate']:.4f} "
            f"(prior {inv['prior_rate']:.4f}, yes-rate {inv['yes_rate']:.4f}), "
            f"control arm {_num(inv['control'].get('accuracy'), '.4f', 0)}"
        )

    lines.append("")
    for p in result["predictions"]:
        lines.append(f"  {p['id']} {p['verdict'].upper():<11}{p['outcome'][:160]}")
    f = result["failures"]
    lines.append("")
    lines.append(f"  failed calls: {f['calls']} (excluded, never defaulted)")
    for a in result["anomalies"]:
        lines.append(f"  ! {a}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Self-check -- python -m jeveval.experiments.e8_icl
# --------------------------------------------------------------------------


def _self_check() -> None:
    # The scope phrases carry their own department's vocabulary and no other's,
    # so a novel category's definition cannot be read as naming a department it
    # does not contain.
    for dept, phrase in DEPARTMENT_SCOPES.items():
        hits = {d: n for d, n in semantic.keyword_counts(phrase).items() if n}
        assert set(hits) == {dept}, f"{dept} scope phrase matches {sorted(hits)}"
    assert set(DEPARTMENT_SCOPES) == set(semantic.DEPARTMENTS)
    assert set(FAMILY_OF) == set(semantic.DEPARTMENTS)

    rng = random.Random(1)
    for index in range(8):
        labels = _label_plan(index, rng)
        assert len(labels) == MAX_EXAMPLES and sum(labels) == MAX_EXAMPLES // 2
        assert sum(labels[:3]) in (1, 2)
    assert all(_label_plan(i, random.Random(i))[0] for i in range(0, 8, 2))
    assert not any(_label_plan(i, random.Random(i))[0] for i in range(1, 8, 2))

    # Poisoning holds the label marginal fixed on both a binary and a 4-way
    # label, and mislabels the fraction the result claims it does. The second
    # assertion is the one that matters if POISON_AT ever moves: `_poison`
    # rounds to whole swaps, so at a small k the realised fraction would drift
    # away from POISON_FRACTION while the result went on reporting the nominal
    # figure.
    for labels in (
        [True] * 5 + [False] * 5,
        ["low", "normal", "high", "critical"] * 2 + ["low", "high"],
    ):
        shown = _poison(list(labels), POISON_AT, random.Random(3))
        assert sorted(map(str, shown)) == sorted(map(str, labels)), "poisoning moved the marginal"
        mislabelled = sum(1 for a, b in zip(shown, labels) if a != b)
        assert mislabelled / POISON_AT == POISON_FRACTION, (
            f"poisoning mislabelled {mislabelled}/{POISON_AT}, not {POISON_FRACTION:.0%}"
        )

    pool = semantic.tickets(seed=7, count=EXEMPLAR_POOL, level=LEVEL)
    assert len({(t.department, t.severity) for t in pool}) == EXEMPLAR_POOL, (
        "the exemplar pool does not cover every department/severity pair, so the "
        "urgency arm cannot draw a rubric-balanced block"
    )
    scheme = _novel_scheme(11, pool)
    corpus = " ".join(t.text for t in pool).lower()
    vocabulary = " ".join(
        re.sub(r"[^a-z]", "", k.lower())
        for kws in semantic.DEPARTMENT_KEYWORDS.values()
        for k in kws
    )
    for name in scheme["names"]:
        lowered = name.lower()
        assert lowered not in corpus, f"{name} occurs in the ticket corpus"
        assert lowered not in " ".join(DEPARTMENT_SCOPES.values()).lower()
        assert not any(
            lowered[i : i + 4] in vocabulary for i in range(len(lowered) - 3)
        ), f"{name} shares a four-character run with task vocabulary"
    assert len({n[0] for n in scheme["names"]}) == len(scheme["names"])
    assert sorted(len(v) for v in scheme["members"].values()) == sorted(NOVEL_CATEGORY_SIZES)
    assert set(scheme["category_of"]) == set(semantic.DEPARTMENTS)
    for name, members in scheme["members"].items():
        assert len({FAMILY_OF[d] for d in members}) >= 2, f"{name} sits in one family"

    partner = _swap_pairs(13)
    assert set(partner) == set(semantic.DEPARTMENTS)
    for a, b in partner.items():
        assert partner[b] == a and FAMILY_OF[a] != FAMILY_OF[b]

    for task in TASKS:
        items = semantic.generate(
            difficulty={"level": LEVEL, "question_type": task.kind}, seed=17, count=8
        )
        for item in items:
            sets = []
            for k in (1, 3, 10):
                block = _exemplars_for(
                    task=task,
                    item_index=item.index,
                    item_template_id=item.meta["template_id"],
                    pool=pool,
                    seed=7,
                    poisoned=False,
                    k=k,
                )
                assert len(block) == k
                for ex in block:
                    assert ex.ticket.template_id != item.meta["template_id"]
                    assert ex.shown == ex.true
                    if task.kind == "noul":
                        assert (ex.asked == ex.ticket.department) == ex.true
                    else:
                        assert ex.true == ex.ticket.severity
                sets.append({id(ex.ticket) for ex in block})
            assert sets[0] <= sets[1] <= sets[2], "exemplar sets are not nested across k"

            # A poisoned block is its clean block with two labels swapped and
            # nothing else: same tickets, same order, same question asked of
            # each. Poisoning draws from its own RNG stream to make that true,
            # and this is the assertion that says so -- sharing the stream left
            # the poisoned cell differing from its control in three ways at once
            # and the measured drop attributable to none of them.
            clean = _exemplars_for(
                task=task, item_index=item.index,
                item_template_id=item.meta["template_id"], pool=pool, seed=7,
                poisoned=False, k=POISON_AT,
            )
            dirty = _exemplars_for(
                task=task, item_index=item.index,
                item_template_id=item.meta["template_id"], pool=pool, seed=7,
                poisoned=True, k=POISON_AT,
            )
            assert [e.ticket.template_id for e in clean] == [
                e.ticket.template_id for e in dirty
            ], "poisoning moved the exemplars"
            assert [e.asked for e in clean] == [e.asked for e in dirty], (
                "poisoning moved the question the exemplars ask"
            )
            assert [e.true for e in clean] == [e.true for e in dirty]
            moved = sum(1 for a, b in zip(clean, dirty) if a.shown != b.shown)
            assert moved == round(POISON_FRACTION * POISON_AT), (
                f"poisoned {moved}/{POISON_AT} of the block, not {POISON_FRACTION:.0%}"
            )
        # The urgency arm's k=10 block covers the rubric evenly, and its k=1
        # exemplar rotates over the four severities across items.
        if task.kind == "score":
            ten = _exemplars_for(
                task=task, item_index=0, item_template_id="", pool=pool, seed=7,
                poisoned=False, k=10,
            )
            counts = {s: sum(1 for e in ten if e.true == s) for s in semantic.SEVERITIES}
            assert max(counts.values()) - min(counts.values()) <= 1, counts
            firsts = {
                _exemplars_for(
                    task=task, item_index=i, item_template_id="", pool=pool, seed=7,
                    poisoned=False, k=1,
                )[0].true
                for i in range(4)
            }
            assert len(firsts) == len(semantic.SEVERITIES), firsts

        # Both channels leave the item's own encoding alone.
        item = items[0]
        question = item.questions[task.key]["question"]
        block = _exemplars_for(
            task=task, item_index=item.index, item_template_id=item.meta["template_id"],
            pool=pool, seed=7, poisoned=False, k=3,
        )
        s_ins, i_ins = _icl_request(
            task=task, channel="instructions", exemplars=block,
            item_state=item.state, question=question,
        )
        s_st, i_st = _icl_request(
            task=task, channel="state", exemplars=block,
            item_state=item.state, question=question,
        )
        assert s_ins == item.state and s_st["ticket"] == item.state
        assert question in i_ins and question in i_st
        assert len(s_st["examples"]) == 3 and "Example 1." in i_ins

    # The override arms differ in one sentence and nothing else.
    tickets = semantic.tickets(seed=19, count=6, level=LEVEL)
    inv = _override_calls(tickets=tickets, partner=partner, inverted=True)
    con = _override_calls(tickets=tickets, partner=partner, inverted=False)
    for a, b in zip(inv, con):
        assert a.state == b.state
        assert a.meta["asked_department"] == b.meta["asked_department"]
        assert a.meta["definition_answer"] != b.meta["definition_answer"]
        assert b.meta["definition_answer"] == b.meta["prior_answer"]
        assert a.meta["definition_answer"] != a.meta["prior_answer"]
    assert sum(1 for c in inv if c.meta["definition_answer"]) == len(inv) // 2

    # The log has to be rescorable on its own. Truth rides in meta, and on the
    # urgency arm the rubric ids ride with it: the wire question carries the
    # rubric's labels, so a rescore reading the log alone gets a label back and
    # would score every urgency answer wrong against an id without this map.
    for task in TASKS:
        items = semantic.generate(
            difficulty={"level": LEVEL, "question_type": task.kind}, seed=17, count=4
        )
        calls = _icl_calls(
            task=task, items=items, pool=pool, exemplar_seed=7,
            channel="state", k=3, poisoned=False,
        )
        for call in calls:
            assert KEY in call.meta["truth"]
            if task.kind == "score":
                ids = call.meta["rubric_ids"]
                labels = call.questions[KEY]["rubric"]
                assert ids == semantic.SEVERITIES
                assert [r["id"] for r in labels] == ids, (
                    "rubric_ids must be in the same order as the rubric on the wire, "
                    "or the index-to-id mapping a rescore reads back is wrong"
                )
                assert call.meta["truth"][KEY] in ids
            else:
                assert call.meta["rubric_ids"] is None

    print("e8_icl self-check passed")
    print(f"  novel categories: {scheme['members']}")
    print(f"  swap pairs: {sorted({tuple(sorted((a, b))) for a, b in partner.items()})}")
    print(
        "  calls at scale 1.0: "
        f"{len(TASKS) * 9 * config.SIZES.calibration + 4 * config.SIZES.accuracy}"
    )


if __name__ == "__main__":
    _self_check()
