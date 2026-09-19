"""Generator-neutral instance and question types.

Every generator emits `Instance` objects. A question is described here in a
*neutral* form -- `{"type": "noul", "question": ...}` and friends -- rather than
in the exact wire format the API expects. `jeveval.client` owns the translation
to the wire. That indirection exists because the plan was written from docs and
press coverage; the real request schema is discovered by probing the endpoint,
and when it turns out to differ, only the adapter changes rather than all eight
generators.

Ground truth travels with the instance, keyed by the same question keys, so no
experiment ever has to re-derive an answer it already knows.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------
# Question construction
# --------------------------------------------------------------------------
#
# Three question types, per the plan:
#   noul   -> yes/no, the API returns a probability in [0, 1]
#   choice -> pick one of <=255 options; returns chosen id, a distribution, and
#             a confidence
#   score  -> a position on an ordered rubric supplied by the caller

MAX_CHOICE_OPTIONS = 255


def noul(question: str, **extra: Any) -> dict:
    """A yes/no question. Ground truth for one of these is a bool."""
    q = {"type": "noul", "question": question}
    q.update(extra)
    return q


def choice(question: str, options: list[Any], **extra: Any) -> dict:
    """A pick-one question.

    `options` may be a list of plain ids (strings) or a list of
    ``{"id": ..., "label": ...}`` dicts. Ground truth is the id of the correct
    option. The 255-option cap is enforced here rather than discovered as a
    server-side error halfway through a 300-instance condition.
    """
    if len(options) > MAX_CHOICE_OPTIONS:
        raise ValueError(
            f"choice has {len(options)} options; the API caps at {MAX_CHOICE_OPTIONS}"
        )
    if len(options) < 2:
        raise ValueError("choice needs at least 2 options")
    norm = [{"id": o, "label": str(o)} if not isinstance(o, dict) else o for o in options]
    ids = [o["id"] for o in norm]
    if len(set(ids)) != len(ids):
        raise ValueError("choice option ids must be unique")
    q = {"type": "choice", "question": question, "options": norm}
    q.update(extra)
    return q


def score(question: str, rubric: list[Any], **extra: Any) -> dict:
    """A position on an ordered rubric. `rubric` is ordered low to high."""
    if len(rubric) < 2:
        raise ValueError("score needs at least 2 rubric points")
    norm = [{"id": r, "label": str(r)} if not isinstance(r, dict) else r for r in rubric]
    q = {"type": "score", "question": question, "rubric": norm}
    q.update(extra)
    return q


# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------


@dataclass
class Instance:
    """One generated problem plus its ground truth.

    Attributes:
        generator:  short name of the producing generator, e.g. "sat3".
        difficulty: the generator's difficulty knobs for this instance, e.g.
                    ``{"n": 20, "ratio": 4.25}``. Every generator takes a
                    difficulty parameter; experiments group and plot by it.
        seed:       the seed this instance was drawn from. Together with
                    `generator`, `difficulty` and `index` it reproduces the
                    instance exactly.
        index:      position within the (generator, difficulty, seed) draw.
        state:      the `state` payload -- str, dict, or list.
        questions:  question key -> neutral question spec.
        truth:      question key -> ground truth. bool for noul, option id for
                    choice, rubric id for score.
        meta:       anything an experiment needs that is not ground truth:
                    baseline features (clause density, path length), the
                    encoding variant, solver timings, and so on.
    """

    generator: str
    difficulty: dict
    seed: int
    index: int
    state: Any
    questions: dict[str, dict]
    truth: dict[str, Any]
    meta: dict = field(default_factory=dict)

    @property
    def instance_id(self) -> str:
        """Stable id derived from the fields that determine the instance.

        Derived rather than random so that the same instance carries the same id
        across reruns, which is what makes the JSONL log joinable offline.
        """
        payload = json.dumps(
            {
                "g": self.generator,
                "d": self.difficulty,
                "s": self.seed,
                "i": self.index,
            },
            sort_keys=True,
        )
        return f"{self.generator}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"

    def validate(self) -> None:
        """Fail loudly at generation time rather than at scoring time.

        Every question must have ground truth, and that truth must be a legal
        value for the question's type. A generator that silently emits a choice
        whose correct answer is not among its options produces a whole condition
        of data that scores as 0% and looks like a model failure.
        """
        missing = set(self.questions) - set(self.truth)
        if missing:
            raise ValueError(f"{self.instance_id}: questions without truth: {sorted(missing)}")
        extra = set(self.truth) - set(self.questions)
        if extra:
            raise ValueError(f"{self.instance_id}: truth without questions: {sorted(extra)}")
        for key, q in self.questions.items():
            t = self.truth[key]
            if q["type"] == "noul":
                if not isinstance(t, bool):
                    raise ValueError(f"{self.instance_id}.{key}: noul truth must be bool, got {t!r}")
            elif q["type"] == "choice":
                ids = {o["id"] for o in q["options"]}
                if t not in ids:
                    raise ValueError(
                        f"{self.instance_id}.{key}: correct answer {t!r} is not among the options"
                    )
            elif q["type"] == "score":
                ids = {r["id"] for r in q["rubric"]}
                if t not in ids:
                    raise ValueError(
                        f"{self.instance_id}.{key}: truth {t!r} is not a rubric point"
                    )
            else:
                raise ValueError(f"{self.instance_id}.{key}: unknown question type {q['type']!r}")

    def to_json(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "generator": self.generator,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "index": self.index,
            "state": self.state,
            "questions": self.questions,
            "truth": self.truth,
            "meta": self.meta,
        }


def rng_for(generator: str, difficulty: dict, seed: int, index: int = 0) -> random.Random:
    """A `random.Random` determined by the instance coordinates.

    Deriving each instance's RNG from its own coordinates -- rather than drawing
    from one shared stream -- means instance 400 of a condition is reproducible
    without generating the 399 before it, and adding a difficulty level does not
    shift the instances at every other level.
    """
    payload = json.dumps(
        {"g": generator, "d": difficulty, "s": seed, "i": index}, sort_keys=True
    )
    digest = hashlib.sha256(payload.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def shuffled_options(q: dict, rng: random.Random) -> dict:
    """Return a copy of a choice question with its options shuffled.

    The plan requires option order to be randomized everywhere except where an
    experiment is specifically measuring order effects, because position bias
    would otherwise confound every choice-based result.
    """
    if q["type"] != "choice":
        return dict(q)
    out = dict(q)
    opts = list(q["options"])
    rng.shuffle(opts)
    out["options"] = opts
    return out


def shuffled_questions(questions: dict[str, dict], rng: random.Random) -> dict[str, dict]:
    """Return the question map with its key order shuffled.

    Python dicts preserve insertion order and that order reaches the wire as
    JSON key order, so this is a real manipulation and not a no-op.
    """
    keys = list(questions)
    rng.shuffle(keys)
    return {k: questions[k] for k in keys}
