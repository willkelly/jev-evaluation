"""Translation between the neutral question/answer types and the real wire format.

The plan was written from documentation and press coverage, and its paraphrase of
the request shape turned out to be wrong in ways that matter. What the endpoint
actually accepts, confirmed by probing it:

    POST /v1/systemone
    {
      "model": "jev-latest",              # required; the response echoes a
                                          # versioned string such as "jev-1.13.0"
      "state": <str | dict | list>,       # required
      "questions": {                      # required, at least one
        "<key>": {
          "type": "noul",
          "instructions": "<prompt>",     # the prompt channel for every type
          "criteria": {"true": "...", "false": "..."}   # optional alternative
        },
        "<key>": {
          "type": "choice",
          "instructions": "<prompt>",
          "criteria": {"<option_id>": "<description>", ...}   # 1..255 options
        },
        "<key>": {
          "type": "score",
          "instructions": "<prompt>",
          "criteria": ["low", "medium", "high"]   # ordered, or list of objects
        }
      }
    }

Three consequences worth stating, because experiments depend on them:

* There is no top-level `instructions` field -- sending one is a 400. E8's
  "examples in instructions" channel is therefore the *per-question*
  `instructions` string, not a request-level prompt.
* A choice question's options are the KEYS of its `criteria` object, so option
  order on the wire is JSON key order, which is Python dict insertion order.
  Shuffling the neutral option list really does shuffle what the model sees.
* A noul answer comes back as a bare probability with **no confidence field**.
  Only choice and score carry `confidence`. Anything in the plan that asks about
  confidence on a yes/no question -- E9's abstention test in particular -- has to
  use the probability's distance from 0.5 instead, and the report should say so.

Responses:

    noul   -> {"type": "noul",   "noul": 0.98}
    choice -> {"type": "choice", "choice": "<id>", "confidence": 1.0,
               "probabilities": {"<id>": 0.98, ...}}
    score  -> {"type": "score",  "score": 1.54, "confidence": 0.3,
               "legend": {"0": "low", ...}, "probabilities": {"0": 0.04, ...}}

    usage  -> {"input_tokens": int, "output_tokens": int}
"""

from __future__ import annotations

from typing import Any

MAX_CHOICE_OPTIONS = 255


class WireError(ValueError):
    """A request could not be built, or a response could not be parsed."""


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


def question_to_wire(q: dict) -> dict:
    """Convert one neutral question spec to its wire form.

    Any key in the neutral spec prefixed with "wire_" is merged verbatim into
    the wire question after translation. Experiments that need to manipulate the
    wire directly -- E1 renames question keys, E8 stuffs examples into
    `instructions` -- use that rather than reaching around this function.
    """
    qtype = q.get("type")
    prompt = q.get("question")
    passthrough = {k[5:]: v for k, v in q.items() if k.startswith("wire_")}

    if qtype == "noul":
        wire: dict[str, Any] = {"type": "noul"}
        if prompt:
            wire["instructions"] = prompt
        if "criteria" in q:
            wire["criteria"] = q["criteria"]
        if "instructions" not in wire and "criteria" not in wire:
            raise WireError("noul needs either a question or explicit criteria")

    elif qtype == "choice":
        options = q.get("options") or []
        if not 1 <= len(options) <= MAX_CHOICE_OPTIONS:
            raise WireError(
                f"choice has {len(options)} options; the API accepts 1..{MAX_CHOICE_OPTIONS}"
            )
        # Dict insertion order is JSON key order is wire option order.
        criteria: dict[str, str] = {}
        for o in options:
            oid = str(o["id"])
            if oid in criteria:
                raise WireError(f"duplicate choice option id {oid!r}")
            criteria[oid] = str(o.get("label", ""))
        wire = {"type": "choice", "criteria": criteria}
        if prompt:
            wire["instructions"] = prompt

    elif qtype == "score":
        rubric = q.get("rubric") or []
        if not rubric:
            raise WireError("score needs at least one rubric point")
        labels = [str(r.get("label", r["id"])) for r in rubric]
        if len(set(labels)) != len(labels):
            # The response identifies rubric points by index, and the legend
            # echoes labels. Duplicate labels make a returned point ambiguous
            # when mapping back to a rubric id.
            raise WireError("score rubric labels must be distinct")
        wire = {"type": "score", "criteria": labels}
        if prompt:
            wire["instructions"] = prompt

    else:
        raise WireError(f"unknown question type {qtype!r}")

    wire.update(passthrough)
    return wire


def build_request(
    *,
    state: Any,
    questions: dict[str, dict],
    model: str,
) -> dict:
    """Build the full request body. Question key order is preserved."""
    if not questions:
        raise WireError("at least one question is required")
    return {
        "model": model,
        "state": state,
        "questions": {k: question_to_wire(q) for k, q in questions.items()},
    }


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


def parse_answer(wire_answer: dict, question: dict) -> dict:
    """Convert one wire answer to a neutral answer.

    The neutral form is what metrics consume, so it always exposes the same
    names: `p` for a yes/no probability, `chosen` for a selected id, and
    `probabilities` keyed by the ids the caller supplied rather than by wire
    indices.
    """
    atype = wire_answer.get("type")

    if atype == "noul":
        p = wire_answer.get("noul")
        if not isinstance(p, (int, float)):
            raise WireError(f"noul answer has no probability: {wire_answer!r}")
        return {
            "type": "noul",
            "p": float(p),
            "predicted": bool(p >= 0.5),
            # No confidence field exists on noul; record that explicitly rather
            # than letting a downstream `.get("confidence")` silently yield None.
            "confidence": None,
        }

    if atype == "choice":
        chosen = wire_answer.get("choice")
        if chosen is None:
            raise WireError(f"choice answer has no selection: {wire_answer!r}")
        probs = {str(k): float(v) for k, v in (wire_answer.get("probabilities") or {}).items()}
        return {
            "type": "choice",
            "chosen": str(chosen),
            "confidence": _opt_float(wire_answer.get("confidence")),
            "probabilities": probs,
            "max_probability": max(probs.values()) if probs else None,
        }

    if atype == "score":
        raw = wire_answer.get("score")
        if not isinstance(raw, (int, float)):
            raise WireError(f"score answer has no score: {wire_answer!r}")
        rubric = question.get("rubric") or []
        # Wire probabilities are keyed by rubric INDEX as a string; map them back
        # onto the caller's rubric ids so no experiment has to know that.
        idx_probs = {int(k): float(v) for k, v in (wire_answer.get("probabilities") or {}).items()}
        by_id: dict[str, float] = {}
        for i, point in enumerate(rubric):
            if i in idx_probs:
                by_id[str(point["id"])] = idx_probs[i]
        chosen_id = None
        if idx_probs:
            best = max(idx_probs, key=lambda i: idx_probs[i])
            if 0 <= best < len(rubric):
                chosen_id = str(rubric[best]["id"])
        return {
            "type": "score",
            "score": float(raw),
            "chosen": chosen_id,
            "confidence": _opt_float(wire_answer.get("confidence")),
            "probabilities": by_id,
            "index_probabilities": idx_probs,
            "legend": wire_answer.get("legend"),
            "max_probability": max(idx_probs.values()) if idx_probs else None,
        }

    raise WireError(f"unknown answer type {atype!r}")


def parse_response(response: dict, questions: dict[str, dict]) -> dict[str, dict]:
    """Convert a whole response body to neutral answers, keyed as sent.

    A question that came back without an answer is an error, not an empty
    result: the plan forbids a failed call from silently becoming a data point.
    """
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise WireError(f"response has no answers object: {list(response)}")
    missing = set(questions) - set(answers)
    if missing:
        raise WireError(f"response omitted answers for {sorted(missing)}")
    return {k: parse_answer(answers[k], questions[k]) for k in questions}


def model_version(response: dict) -> str | None:
    """The versioned model string, e.g. "jev-1.13.0"."""
    v = response.get("model")
    return str(v) if v else None


def usage(response: dict) -> tuple[int, int]:
    """(input_tokens, output_tokens), defaulting to 0 when absent."""
    u = response.get("usage") or {}
    return int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)


def _opt_float(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None
