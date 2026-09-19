"""E5 -- enrollment and entanglement.

Four nouls give four marginals. One choice over the sixteen combinations gives a
joint. Marginals do not determine a joint, so the question is whether enrolling
the outcome space buys coherence that separate yes/no questions cannot express.

Seven conditions, in the plan's order: marginal consistency, exclusion,
implication, joint versus product, labelling, Sudoku, and reachable-successor
enumeration. The first five need states whose logical structure is known
exactly, and no generator produces those, so this module builds them. Three
things about that construction matter.

*The constraint is verified, not asserted.* Every constructed scene is defined
by an explicit set of underlying worlds plus a predicate per proposition. The
set of logically possible (A, B) cells is then computed by evaluating those
predicates over every world, and the exclusion scenes assert that the possible
set is exactly {(true, false), (false, true)} while the implication scenes
assert that (true, false) is absent and that both propositions are still
genuinely uncertain. A scene that fails those assertions raises at construction
time. Nothing in the analysis takes the exclusivity on trust.

*The reference joint is uniform over the worlds, and it is a reference, not
ground truth.* The states deliberately underdetermine their propositions: a
state that fixed all four bits would make the correct joint a point mass, whose
marginals are also a point mass, so KL(joint || product) would be zero for a
model that was simply right. That would read as "enrollment is decorative" when
it was nothing of the kind. Leaving genuine residual uncertainty, with
dependence between the bits, gives KL something real to measure and supplies a
baseline for it: the dependence actually present in the uniform-over-worlds
reference. Accuracy on those states is therefore "did the chosen cell land on a
logically possible combination", not "did it pick the one true combination".

*The four nouls share one call.* The plan says to ask them separately and to ask
the joint separately. Asked as four separate calls, any disagreement between the
standalone marginals and the joint-derived ones would mix the encoding
difference this experiment is about with the cross-call variance E6 is about.
Batching the nouls into one call holds the call count and the state presentation
fixed, so `marginal_disagreement` measures the two encodings and nothing else.
E6 measures the cross-call component; E3 measures whether batching four
questions costs anything.

Two quantities are reported that the plan asks for only implicitly. KL(joint ||
product of the standalone marginals) decomposes exactly into the joint's own
total correlation plus the disagreement between the two encodings' marginals,
and those are different findings, so both appear. And the plan names a likely
surprise -- the joint being sharp while the standalone marginals are mushy --
which is the entropy of the joint against the summed entropy of the marginals,
reported per condition as `entropy_gap_nats`.

Probabilities come back quantized to two decimal places, so a returned 0.00 means
"below 0.005", not "impossible". Marginals are therefore clamped to
[0.005, 0.995] before the outer product is formed. Without that, one marginal of
0.00 puts an exact zero in the product and sends KL to infinity on the strength
of a rounding step.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import config, instances, metrics, plots, tiers
from ..client import Call, CallResult, JevClient
from ..generators import dfa, semantic, sudoku
from ..instances import Instance
from ..predictions import for_experiment

EXPERIMENT = "E5"

#: Returned probabilities are quantized to two decimal places, so 0.00 is an
#: interval, not a zero. Marginals are clamped this far from the ends before
#: being multiplied out.
QUANTUM = 0.01
FLOOR = QUANTUM / 2.0

#: A returned distribution whose raw mass is further than this from 1 is
#: renormalized and counted as an anomaly rather than used as it stands.
SUM_TOLERANCE = 0.05

#: Below this many scored instances a condition's means are reported but not
#: read as a result. At JEV_SCALE=1.0 no condition comes close to it.
MIN_MEANINGFUL_N = 20

#: How much dependence a four-proposition scene is allowed to carry, measured as
#: KL(reference joint || product of its marginals) in nats. The band extends past
#: P13's predicted 0.2-0.5 at both ends, so the prediction can fail in either
#: direction because of the model rather than because of the construction.
REFERENCE_KL_BAND = (0.15, 0.80)


# --------------------------------------------------------------------------
# Propositions, cells and scenes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Prop:
    """One boolean proposition, with the two tokens its option id is built from.

    `true_token` and `false_token` are what makes a legible option id legible:
    a cell over four propositions becomes `pump_running_valve_shut_door_locked_
    generator_online`, which is the plan's `north_open_treasure` at four bits.
    """

    text: str
    true_token: str
    false_token: str

    def token(self, value: bool) -> str:
        return self.true_token if value else self.false_token


Cell = tuple[bool, ...]


def _cells(k: int) -> tuple[Cell, ...]:
    """Every assignment to k propositions, proposition 0 first in each tuple."""
    return tuple(itertools.product((False, True), repeat=k))


def _bits(cell: Cell) -> str:
    return "".join("1" if b else "0" for b in cell)


def _legible_id(props: Sequence[Prop], cell: Cell) -> str:
    return "_".join(p.token(b) for p, b in zip(props, cell))


@dataclass(frozen=True)
class Scene:
    """One constructed state, its propositions, and what the state permits.

    `consistent` is computed by enumerating the scene's underlying worlds, so it
    is a fact about the state rather than a claim about it. `reference` is the
    uniform-over-worlds joint: the distribution a reasoner with no information
    beyond the state should hold.
    """

    family: str
    difficulty: dict
    seed: int
    index: int
    state: Any
    props: tuple[Prop, ...]
    consistent: tuple[Cell, ...]
    reference: tuple[float, ...]
    forbidden: Cell | None
    question_preamble: str

    @property
    def cells(self) -> tuple[Cell, ...]:
        return _cells(len(self.props))

    @property
    def instance_id(self) -> str:
        payload = json.dumps(
            {"f": self.family, "d": self.difficulty, "s": self.seed, "i": self.index},
            sort_keys=True,
        )
        return f"e5{self.family[:4]}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"

    def reference_marginals(self) -> tuple[float, ...]:
        return tuple(
            sum(p for cell, p in zip(self.cells, self.reference) if cell[i])
            for i in range(len(self.props))
        )


def _reference_from_worlds(
    worlds: Sequence[Any], predicates: Sequence[Callable[[Any], bool]]
) -> tuple[tuple[Cell, ...], tuple[float, ...]]:
    """(consistent cells, uniform-over-worlds joint) for a scene's worlds.

    Every world is equally weighted because the states are written so that
    nothing distinguishes them. That assumption is stated in the result rather
    than left implicit here.
    """
    cells = _cells(len(predicates))
    index = {cell: i for i, cell in enumerate(cells)}
    mass = [0.0] * len(cells)
    for w in worlds:
        mass[index[tuple(bool(p(w)) for p in predicates)]] += 1.0
    total = float(len(worlds))
    joint = tuple(m / total for m in mass)
    consistent = tuple(cell for cell, m in zip(cells, joint) if m > 0.0)
    return consistent, joint


# --------------------------------------------------------------------------
# The two-proposition scenes: exclusion and implication
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairSpec:
    """A two-proposition scenario before the stated/structural split.

    `facts` always reach the state. `constraint` is the sentence that spells the
    logical relation out in so many words, and it reaches the state only in the
    `stated` mode -- which is the difficulty axis for these two conditions: does
    the joint respect a relation the state does not name?
    """

    scenario: str
    facts: tuple[str, ...]
    constraint: str
    prop_a: Prop
    prop_b: Prop
    worlds: tuple[Any, ...]
    a_of: Callable[[Any], bool]
    b_of: Callable[[Any], bool]


_ENGINEERS = ["Alvarez", "Boateng", "Chen", "Doherty", "Eriksen", "Farrow", "Gill", "Haddad"]
_DOCKS = ["north", "south", "east", "west"]
_AISLES = ["A4", "B2", "C9", "D6"]


def _exclusion_specs(rng: random.Random) -> PairSpec:
    """One randomly chosen mutually-exclusive-and-exhaustive scenario.

    Every template turns on a single-valued underlying fact -- who holds one
    slot, which side of one threshold a single reading fell, which of two docks
    one shipment went to, which of two relays carried a single-point failure --
    so the exclusivity is a property of the scenario rather than a stipulation.
    """
    which = rng.randrange(4)

    if which == 0:
        a, b = rng.sample(_ENGINEERS, 2)
        return PairSpec(
            scenario="Pumping station 7, night shift.",
            facts=(
                "The 02:00 escalation slot is held by exactly one engineer.",
                f"Tonight's roster lists only {a} and {b} as eligible for that slot.",
                "The slot is filled; the roster sheet itself is not available.",
            ),
            constraint=(
                f"Exactly one of these is true: {a} holds the 02:00 slot; "
                f"{b} holds the 02:00 slot."
            ),
            prop_a=Prop(f"{a} holds the 02:00 escalation slot", f"{a.lower()}_oncall", f"{a.lower()}_clear"),
            prop_b=Prop(f"{b} holds the 02:00 escalation slot", f"{b.lower()}_oncall", f"{b.lower()}_clear"),
            worlds=(a, b),
            a_of=lambda w, a=a: w == a,
            b_of=lambda w, b=b: w == b,
        )

    if which == 1:
        t = rng.choice([40, 45, 50, 55, 60])
        return PairSpec(
            scenario="Tank farm telemetry, overnight.",
            facts=(
                "Tank 3's level was recorded once at 06:00 as a whole percentage "
                "between 1 and 100 inclusive.",
                "The log page is torn and the recorded value cannot be read.",
                "No other reading of tank 3 was taken that night.",
            ),
            constraint=(
                f"Exactly one of these is true: the recorded level was below {t}%; "
                f"the recorded level was {t}% or above."
            ),
            prop_a=Prop(f"the recorded level of tank 3 was below {t}%", "level_below", "level_not_below"),
            prop_b=Prop(f"the recorded level of tank 3 was {t}% or above", "level_atleast", "level_not_atleast"),
            worlds=tuple(range(1, 101)),
            a_of=lambda w, t=t: w < t,
            b_of=lambda w, t=t: w >= t,
        )

    if which == 2:
        d1, d2 = rng.sample(_DOCKS, 2)
        return PairSpec(
            scenario="Regional depot, inbound shift.",
            facts=(
                "Consignment 4471 arrived as a single indivisible pallet.",
                f"The depot has two receiving docks in service tonight, {d1} and {d2}.",
                "The pallet was received; the dock log has not been reconciled yet.",
            ),
            constraint=(
                f"Exactly one of these is true: consignment 4471 was received at the "
                f"{d1} dock; consignment 4471 was received at the {d2} dock."
            ),
            prop_a=Prop(f"consignment 4471 was received at the {d1} dock", f"dock_{d1}", f"not_dock_{d1}"),
            prop_b=Prop(f"consignment 4471 was received at the {d2} dock", f"dock_{d2}", f"not_dock_{d2}"),
            worlds=(d1, d2),
            a_of=lambda w, d=d1: w == d,
            b_of=lambda w, d=d2: w == d,
        )

    r1, r2 = rng.sample(["K1", "K2", "K3", "K4"], 2)
    return PairSpec(
        scenario="Substation controller, post-incident.",
        facts=(
            "The controller logged a single-point failure: exactly one component failed.",
            f"Diagnostics narrowed the failure to relay {r1} or relay {r2}.",
            "The bench test that would distinguish them has not been run.",
        ),
        constraint=(
            f"Exactly one of these is true: relay {r1} is the failed component; "
            f"relay {r2} is the failed component."
        ),
        prop_a=Prop(f"relay {r1} is the failed component", f"relay_{r1.lower()}_failed", f"relay_{r1.lower()}_sound"),
        prop_b=Prop(f"relay {r2} is the failed component", f"relay_{r2.lower()}_failed", f"relay_{r2.lower()}_sound"),
        worlds=(r1, r2),
        a_of=lambda w, r=r1: w == r,
        b_of=lambda w, r=r2: w == r,
    )


def _implication_specs(rng: random.Random) -> PairSpec:
    """One randomly chosen scenario in which A entails B and both are uncertain.

    Entailment comes from containment in every case -- a narrower threshold
    inside a wider one, a bay inside an aisle, a later stage of a gated sequence,
    a tier of an escalation -- so (A true, B false) is impossible by the
    scenario's own arithmetic rather than by fiat.
    """
    which = rng.randrange(4)

    if which == 0:
        lo = rng.choice([20, 25, 30])
        hi = lo + rng.choice([20, 25, 30])
        return PairSpec(
            scenario="Compressor bay, unattended run.",
            facts=(
                "Compressor 2's peak discharge pressure during the run was a whole "
                "number of bar between 1 and 100 inclusive.",
                "The peak was recorded to the strip chart, which has not been read back.",
                "No other pressure figure is available.",
            ),
            constraint=(
                f"If the peak exceeded {hi} bar then it also exceeded {lo} bar."
            ),
            prop_a=Prop(f"compressor 2's peak discharge pressure exceeded {hi} bar", "peak_over_high", "peak_not_over_high"),
            prop_b=Prop(f"compressor 2's peak discharge pressure exceeded {lo} bar", "peak_over_low", "peak_not_over_low"),
            worlds=tuple(range(1, 101)),
            a_of=lambda w, hi=hi: w > hi,
            b_of=lambda w, lo=lo: w > lo,
        )

    if which == 1:
        a1, a2 = rng.sample(_AISLES, 2)
        bay = rng.choice([1, 2, 3])
        other = 4
        return PairSpec(
            scenario="Bonded warehouse, stock check.",
            facts=(
                f"Crate 88 is somewhere in aisle {a1} or aisle {a2}; no other aisle "
                "holds bonded stock.",
                f"Within whichever aisle holds it, the crate is in bay {bay} or bay {other}.",
                "The scanner trail for the crate is incomplete.",
            ),
            constraint=(
                f"If crate 88 is in aisle {a1} bay {bay} then it is in aisle {a1}."
            ),
            prop_a=Prop(f"crate 88 is in aisle {a1} bay {bay}", "in_bay", "not_in_bay"),
            prop_b=Prop(f"crate 88 is in aisle {a1}", "in_aisle", "not_in_aisle"),
            worlds=tuple((x, y) for x in (a1, a2) for y in (bay, other)),
            a_of=lambda w, a=a1, b=bay: w == (a, b),
            b_of=lambda w, a=a1: w[0] == a,
        )

    if which == 2:
        return PairSpec(
            scenario="Release pipeline, overnight window.",
            facts=(
                "The pipeline runs three stages in order: schema migration, "
                "migration verification, then deploy.",
                "The deploy gate opens only once verification has passed; a deploy "
                "cannot start before it.",
                "The run halted at some stage and the pipeline log was rotated away.",
            ),
            constraint="If the deploy ran then verification had passed.",
            prop_a=Prop("the deploy ran", "deployed", "not_deployed"),
            prop_b=Prop("migration verification passed", "verified", "not_verified"),
            worlds=("halted_before_verification", "verified_not_deployed", "verified_and_deployed"),
            a_of=lambda w: w == "verified_and_deployed",
            b_of=lambda w: w in ("verified_not_deployed", "verified_and_deployed"),
        )

    return PairSpec(
        scenario="Support desk, end of shift.",
        facts=(
            "Ticket 2190 is either not escalated at all, or escalated to exactly "
            "one of tiers 1, 2 or 3.",
            "Tier 3 is a level of escalation, as are tiers 1 and 2.",
            "The escalation field was not exported with tonight's batch.",
        ),
        constraint="If ticket 2190 is escalated to tier 3 then ticket 2190 is escalated.",
        prop_a=Prop("ticket 2190 is escalated to tier 3", "tier3", "not_tier3"),
        prop_b=Prop("ticket 2190 is escalated", "escalated", "not_escalated"),
        worlds=("none", "tier1", "tier2", "tier3"),
        a_of=lambda w: w == "tier3",
        b_of=lambda w: w != "none",
    )


def _pair_scene(
    relation: str, mode: str, seed: int, index: int
) -> Scene:
    """Build and verify one exclusion or implication scene."""
    difficulty = {"condition": relation, "constraint": mode}
    rng = instances.rng_for(f"e5-{relation}", difficulty, seed, index)
    spec = _exclusion_specs(rng) if relation == "exclusion" else _implication_specs(rng)

    consistent, reference = _reference_from_worlds(spec.worlds, (spec.a_of, spec.b_of))
    possible = set(consistent)
    if relation == "exclusion":
        if possible != {(True, False), (False, True)}:
            raise AssertionError(
                f"exclusion scene is not exactly-one-of: possible cells {sorted(possible)}"
            )
        forbidden: Cell = (True, True)
    else:
        if (True, False) in possible:
            raise AssertionError("implication scene permits (A true, B false)")
        if not {(True, True), (False, False)} <= possible:
            raise AssertionError(
                "implication scene leaves a proposition determined, so the "
                f"implication is vacuous: possible cells {sorted(possible)}"
            )
        forbidden = (True, False)

    facts = list(spec.facts)
    if mode == "stated":
        facts.append(spec.constraint)
    state = {
        "situation": spec.scenario,
        "what_is_known": facts,
        "note": "Nothing beyond the facts above is known about this case.",
    }
    return Scene(
        family=relation,
        difficulty=difficulty,
        seed=seed,
        index=index,
        state=state,
        props=(spec.prop_a, spec.prop_b),
        consistent=consistent,
        reference=reference,
        forbidden=forbidden,
        question_preamble="",
    )


# --------------------------------------------------------------------------
# The four-proposition scenes: marginal consistency, KL, labelling
# --------------------------------------------------------------------------

_WORLD_PROPS: tuple[Prop, ...] = (
    Prop("the main pump is running", "pump_running", "pump_stopped"),
    Prop("valve 2 is open", "valve_open", "valve_shut"),
    Prop("the north door is unlocked", "door_unlocked", "door_locked"),
    Prop("the backup generator is online", "generator_online", "generator_offline"),
    Prop("the intruder alarm is armed", "alarm_armed", "alarm_disarmed"),
    Prop("tank 3 is being filled", "tank_filling", "tank_idle"),
    Prop("the night crew has signed in", "crew_present", "crew_absent"),
    Prop("the site is drawing mains power", "on_mains", "on_battery"),
)

_CONSTRAINT_KINDS = ("implies", "not_both", "at_least_one", "exactly_one", "iff")


def _constraint_sentence(kind: str, a: Prop, b: Prop) -> str:
    if kind == "implies":
        return f"If {a.text} then {b.text}."
    if kind == "not_both":
        return f"It is never the case that {a.text} and {b.text} at the same time."
    if kind == "at_least_one":
        return f"At least one of the following holds: {a.text}; {b.text}."
    if kind == "exactly_one":
        return f"Exactly one of the following holds: {a.text}; {b.text}."
    return f"{a.text.capitalize()} if and only if {b.text}."


def _constraint_holds(kind: str, x: bool, y: bool) -> bool:
    if kind == "implies":
        return (not x) or y
    if kind == "not_both":
        return not (x and y)
    if kind == "at_least_one":
        return x or y
    if kind == "exactly_one":
        return x != y
    return x == y


def _world_scene(n_constraints: int, seed: int, index: int) -> Scene:
    """Four propositions and `n_constraints` relations among them.

    Rejection sampling enforces the three properties the analysis needs: at
    least two combinations survive (so the joint has something to be uncertain
    about), every proposition's reference marginal sits between 0.2 and 0.8 (so
    no bit is effectively decided), and the surviving set is not a product set
    (so a coherent joint must express dependence and KL from the product has a
    non-zero target).
    """
    difficulty = {"condition": "enrolled-joint", "n_constraints": n_constraints}
    rng = instances.rng_for("e5-world", difficulty, seed, index)
    cells = _cells(4)

    for _ in range(800):
        props = tuple(rng.sample(_WORLD_PROPS, 4))
        pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
        rng.shuffle(pairs)
        chosen: list[tuple[str, int, int]] = []
        used: set[frozenset[int]] = set()
        for i, j in pairs:
            if len(chosen) == n_constraints:
                break
            if frozenset((i, j)) in used:
                continue
            used.add(frozenset((i, j)))
            chosen.append((rng.choice(_CONSTRAINT_KINDS), i, j))
        if len(chosen) < n_constraints:
            continue

        keep = [
            cell
            for cell in cells
            if all(_constraint_holds(k, cell[i], cell[j]) for k, i, j in chosen)
        ]
        if not 2 <= len(keep) <= 8:
            continue
        reference = tuple(1.0 / len(keep) if cell in set(keep) else 0.0 for cell in cells)
        marginals = [
            sum(p for cell, p in zip(cells, reference) if cell[i]) for i in range(4)
        ]
        if any(not 0.2 <= m <= 0.8 for m in marginals):
            continue
        product = _outer_product(marginals, cells)
        dependence = metrics.kl_divergence(list(reference), product)
        if not REFERENCE_KL_BAND[0] <= dependence <= REFERENCE_KL_BAND[1]:
            # Below the band the surviving set is close to a product set, so a
            # joint that equalled the product of its marginals would be right and
            # KL would be near zero for the right reason -- not the finding this
            # condition is for. Above it the state is so entangled that a model
            # answering well would still miss P13's 0.2-0.5 band, which would make
            # the prediction fail on the construction rather than on the model.
            continue

        state = {
            "site": "pumping station 7, night shift",
            "what_is_known_about_the_current_configuration": [
                _constraint_sentence(k, props[i], props[j]) for k, i, j in chosen
            ],
            "note": (
                "Nothing else is known. Any configuration consistent with the "
                "statements above is possible."
            ),
        }
        return Scene(
            family="world",
            difficulty=difficulty,
            seed=seed,
            index=index,
            state=state,
            props=props,
            consistent=tuple(keep),
            reference=reference,
            forbidden=None,
            question_preamble="",
        )

    raise RuntimeError(
        f"no admissible 4-proposition scene at n_constraints={n_constraints} "
        f"(seed={seed}, index={index})"
    )


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------


def _clamp(p: float) -> float:
    return min(max(p, FLOOR), 1.0 - FLOOR)


def _outer_product(marginals: Sequence[float], cells: Sequence[Cell]) -> list[float]:
    """The joint independent marginals imply, normalized exactly."""
    raw = [
        math.prod(m if b else 1.0 - m for m, b in zip(marginals, cell)) for cell in cells
    ]
    total = sum(raw)
    if total <= 0.0:
        # Only reachable if every marginal underflowed to an exact 0 or 1 in a
        # contradictory pattern. Uniform is the only defensible fallback and it
        # is better than a ZeroDivisionError halfway through a condition.
        return [1.0 / len(cells)] * len(cells)
    return [v / total for v in raw]


def _normalize(v: Sequence[float]) -> list[float] | None:
    total = sum(v)
    if total <= 0.0:
        return None
    return [x / total for x in v]


def _joint_marginals(joint: Sequence[float], cells: Sequence[Cell], k: int) -> list[float]:
    return [sum(p for cell, p in zip(cells, joint) if cell[i]) for i in range(k)]


def _bernoulli_entropy_sum(marginals: Sequence[float]) -> float:
    return sum(metrics.entropy([m, 1.0 - m]) for m in (_clamp(x) for x in marginals))


# --------------------------------------------------------------------------
# Calls
# --------------------------------------------------------------------------


def _scene_meta(scene: Scene, kind: str, **extra: Any) -> dict:
    """What the JSONL needs to rescore this scene without the generator.

    Everything the metrics read comes from here: the propositions in bit order,
    which cells the state permits, the reference joint, and which cell the plan
    designates as forbidden.
    """
    meta = {
        "family": scene.family,
        "difficulty": scene.difficulty,
        "kind": kind,
        "index": scene.index,
        "seed": scene.seed,
        "props": [
            {"text": p.text, "true_token": p.true_token, "false_token": p.false_token}
            for p in scene.props
        ],
        "consistent_cells": [_bits(c) for c in scene.consistent],
        "reference_joint": {_bits(c): p for c, p in zip(scene.cells, scene.reference)},
        "reference_marginals": list(scene.reference_marginals()),
        "forbidden_cell": None if scene.forbidden is None else _bits(scene.forbidden),
        "n_consistent": len(scene.consistent),
    }
    meta.update(extra)
    return meta


def _noul_call(scene: Scene, condition: str) -> Call:
    rng = instances.rng_for("e5-nouls", scene.difficulty, scene.seed, scene.index)
    questions = {
        f"p{i}": instances.noul(
            f"In the current case, is this true: {p.text}?"
        )
        for i, p in enumerate(scene.props)
    }
    return Call(
        experiment=EXPERIMENT,
        condition=condition,
        instance_id=scene.instance_id,
        state=scene.state,
        questions=instances.shuffled_questions(questions, rng),
        meta=_scene_meta(scene, "nouls", bit_of={f"p{i}": i for i in range(len(scene.props))}),
    )


def _joint_call(scene: Scene, condition: str, encoding: str) -> Call:
    rng = instances.rng_for(f"e5-joint-{encoding}", scene.difficulty, scene.seed, scene.index)
    cells = scene.cells
    ids = {
        cell: (_bits(cell) if encoding == "bitstring" else _legible_id(scene.props, cell))
        for cell in cells
    }
    # The option id is the whole manipulation, so the label repeats it rather
    # than adding a description the bitstring arm would not have.
    options = [{"id": ids[cell], "label": ids[cell]} for cell in cells]

    listing = "; ".join(f"({i + 1}) {p.text}" for i, p in enumerate(scene.props))
    if encoding == "bitstring":
        how = (
            f"Each option is a {len(scene.props)}-character string of 1s and 0s, one "
            "character per proposition in the order listed: a 1 means that "
            "proposition is true and a 0 means it is false."
        )
    else:
        how = (
            "Each option names the state of every proposition in that same order, "
            "one term per proposition, joined by underscores."
        )
    question = instances.choice(
        f"These propositions are in play: {listing}. {how} Exactly one option "
        "describes the actual case. Which one is it?",
        options,
    )
    return Call(
        experiment=EXPERIMENT,
        condition=condition,
        instance_id=scene.instance_id,
        state=scene.state,
        questions={"joint": instances.shuffled_options(question, rng)},
        meta=_scene_meta(
            scene,
            "joint",
            encoding=encoding,
            option_ids={_bits(cell): ids[cell] for cell in cells},
        ),
    )


def _instance_call(inst: Instance, condition: str) -> Call:
    """A generator instance as a Call, with options and question order shuffled."""
    rng = random.Random(config.seed_for(EXPERIMENT, f"shuffle-{inst.instance_id}"))
    questions = {
        k: instances.shuffled_options(q, rng) if q["type"] == "choice" else dict(q)
        for k, q in inst.questions.items()
    }
    return Call(
        experiment=EXPERIMENT,
        condition=condition,
        instance_id=inst.instance_id,
        state=inst.state,
        questions=instances.shuffled_questions(questions, rng),
        meta={**inst.meta, "truth": inst.truth, "difficulty": inst.difficulty},
    )


# --------------------------------------------------------------------------
# Reading answers back
# --------------------------------------------------------------------------


@dataclass
class JointRead:
    joint: list[float]
    chosen: Cell | None
    chosen_off_menu: bool
    raw_sum: float
    missing_ids: int
    extra_ids: int
    extra_mass: float
    confidence: float | None
    max_probability: float | None


def _read_joint(answer: dict, scene: Scene, encoding: str) -> JointRead | None:
    """The returned distribution in canonical cell order, or None if unusable.

    Unusable means no mass on any id we sent. A selection that is not one of
    those ids is a different matter and does not make the distribution unusable,
    so it is flagged on the read rather than returned as a failure: the KL,
    marginal-disagreement and forbidden-mass figures this experiment exists to
    produce are all computed from the distribution and none of them touch
    `chosen`.
    """
    probs = answer.get("probabilities") or {}
    ids = {
        cell: (_bits(cell) if encoding == "bitstring" else _legible_id(scene.props, cell))
        for cell in scene.cells
    }
    ours = set(ids.values())
    # A negative probability would make `metrics.entropy` raise and take the whole
    # condition down; it has not been seen, but clamping costs nothing.
    raw = [max(0.0, float(probs.get(ids[cell], 0.0))) for cell in scene.cells]
    missing = sum(1 for cell in scene.cells if ids[cell] not in probs)
    # Mass on an id we never sent is dropped by the renormalization below, and
    # `raw_sum` alone would not show it: a map that puts 1.0 on our ids and 0.5
    # on an invented one sums to 1.0 over our ids and looks perfect.
    extra = {k: float(v) for k, v in probs.items() if str(k) not in ours}
    total = sum(raw)
    normalized = _normalize(raw)
    if normalized is None:
        return None
    by_id = {ids[cell]: cell for cell in scene.cells}
    selected = str(answer.get("chosen"))
    return JointRead(
        joint=normalized,
        chosen=by_id.get(selected),
        chosen_off_menu=selected not in by_id,
        raw_sum=total,
        missing_ids=missing,
        extra_ids=len(extra),
        extra_mass=sum(max(0.0, v) for v in extra.values()),
        confidence=answer.get("confidence"),
        max_probability=answer.get("max_probability"),
    )


# --------------------------------------------------------------------------
# Per-condition scoring
# --------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _col(rows: Sequence[dict], key: str) -> list[float]:
    """One scored column, skipping rows that do not carry the key.

    `forbidden_mass` exists only on the two-proposition rows and a metric can be
    absent on a row whose distribution was unreadable, so a missing key is a row
    to leave out rather than a zero to average in.
    """
    return [float(r[key]) for r in rows if r.get(key) is not None]


def _median(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def _score_joint_condition(
    scenes: Sequence[Scene],
    noul_results: Sequence[CallResult],
    joint_results: Sequence[CallResult],
    *,
    encoding: str,
    watch: "Watchlist",
) -> dict:
    """Everything conditions 1-4 measure, for one set of scenes.

    An instance contributes only if both of its calls succeeded and the returned
    distribution could be read; the two exclusion counts are kept apart because
    a failed call and an unreadable distribution are different problems.

    A selection that is not one of the ids we sent is a third thing again, and
    is neither of those. The distribution is still there, so the row is kept and
    every distribution metric on it is used; only `chosen_consistent` records
    the failure, as false, which is the same treatment `_accuracy_block` gives an
    off-menu selection on a plain choice. Excluding the row instead would drop
    the forbidden-mass and KL figures on exactly the instances where the model
    behaved oddly, which is the wrong subset to lose.
    """
    by_scene = {s.instance_id: s for s in scenes}
    nouls: dict[str, dict] = {}
    for r in noul_results:
        if r.ok:
            nouls[r.call.instance_id] = r.answers
    joints: dict[str, dict] = {}
    for r in joint_results:
        if r.ok:
            joints[r.call.instance_id] = r.answers

    rows: list[dict] = []
    excluded_failed = 0
    excluded_unreadable = 0
    off_menu = 0

    for scene in scenes:
        sid = scene.instance_id
        na, ja = nouls.get(sid), joints.get(sid)
        if na is None or ja is None:
            excluded_failed += 1
            continue
        read = _read_joint(ja["joint"], scene, encoding)
        if read is None:
            # No mass on any id we sent. There is no distribution to score and
            # defaulting it to "wrong" would put a malformed response into an
            # accuracy denominator.
            excluded_unreadable += 1
            continue
        if read.chosen_off_menu:
            off_menu += 1

        k = len(scene.props)
        marginals = [float(na[f"p{i}"]["p"]) for i in range(k)]
        for i, p in enumerate(marginals):
            # Keyed, because the four nouls of a 16-cell scene are read twice --
            # once for the legible joint and once for the bit-string one -- and
            # they are one observation, not two.
            watch.probability(p, key=(sid, i))
        watch.choice_answer(ja["joint"])
        watch.joint(read)

        cells = scene.cells
        derived = _joint_marginals(read.joint, cells, k)
        product = _outer_product([_clamp(m) for m in marginals], cells)
        own_product = _outer_product(derived, cells)

        consistent = set(scene.consistent)
        row = {
            "instance_id": sid,
            "marginal_disagreement": _mean([abs(a - b) for a, b in zip(marginals, derived)]),
            "max_marginal_disagreement": max(abs(a - b) for a, b in zip(marginals, derived)),
            "kl_joint_product": metrics.kl_divergence(read.joint, product),
            "total_correlation": metrics.kl_divergence(read.joint, own_product),
            "kl_reference": metrics.kl_divergence(
                list(scene.reference), _outer_product(list(scene.reference_marginals()), cells)
            ),
            "joint_entropy": metrics.entropy(read.joint),
            "marginal_implied_entropy": _bernoulli_entropy_sum(marginals),
            "reference_entropy": metrics.entropy(list(scene.reference)),
            "impossible_mass": sum(
                p for cell, p in zip(cells, read.joint) if cell not in consistent
            ),
            # An off-menu selection is not a consistent cell: `read.chosen` is
            # None there and None is not in `consistent`.
            "chosen_consistent": read.chosen in consistent,
            "chosen_off_menu": read.chosen_off_menu,
            "chance": len(consistent) / len(cells),
            "consistent_bits": [_bits(c) for c in scene.consistent],
            "raw_sum": read.raw_sum,
            "missing_ids": read.missing_ids,
            "extra_ids": read.extra_ids,
        }
        if scene.forbidden is not None:
            fi = cells.index(scene.forbidden)
            row["forbidden_mass"] = read.joint[fi]
            row["forbidden_product_baseline"] = product[fi]
            row["forbidden_reference_baseline"] = _outer_product(
                list(scene.reference_marginals()), cells
            )[fi]
        rows.append(row)

    return {
        "rows": rows,
        "n": len(rows),
        "excluded_failed_call": excluded_failed,
        "excluded_unreadable": excluded_unreadable,
        # Scored, not excluded -- see the docstring.
        "selections_not_among_the_options": off_menu,
        "cells_per_joint": len(scenes[0].cells) if scenes else 0,
    }


def _best_constant_baseline(rows: Sequence[dict]) -> float | None:
    """Accuracy of always naming one fixed cell, maximized over cells.

    The majority-class baseline the rubric demands, for a choice whose
    correctness is set-valued: correctness here is "the chosen cell is logically
    possible", so the best constant answer is the cell that is possible on the
    most instances.
    """
    if not rows:
        return None
    counts: dict[str, int] = {}
    for r in rows:
        for bits in r.get("consistent_bits") or ():
            counts[bits] = counts.get(bits, 0) + 1
    if not counts:
        return None
    return max(counts.values()) / len(rows)


def _aggregate(scored: dict, keys: Sequence[str]) -> dict:
    rows = scored["rows"]
    out: dict[str, Any] = {}
    for key in keys:
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        out[key] = _mean([float(v) for v in vals])
    return out


def _accuracy_block(
    results: Sequence[CallResult],
    *,
    question_key: str,
    truth_of: Callable[[Call], Any],
    chance_of: Callable[[Call], float],
    heuristic_of: Callable[[Call, dict], Any] | None,
    watch: "Watchlist",
) -> dict:
    """Accuracy with its three baselines for a plain choice condition."""
    correct: list[bool] = []
    labels: list[Any] = []
    chance: list[float] = []
    heuristic: list[bool] = []
    confidences: list[float] = []
    failures = 0
    off_menu = 0

    for r in results:
        if not r.ok:
            failures += 1
            continue
        answer = r.answers.get(question_key)
        if answer is None:
            failures += 1
            continue
        truth = truth_of(r.call)
        offered = {
            str(o["id"]) for o in (r.call.questions[question_key].get("options") or [])
        }
        if offered and str(answer.get("chosen")) not in offered:
            # Scored as wrong rather than excluded -- the call succeeded and the
            # model did answer -- but counted, because an id we never sent is a
            # response shape the plan asks to be told about.
            off_menu += 1
        correct.append(str(answer.get("chosen")) == str(truth))
        labels.append(str(truth))
        chance.append(chance_of(r.call))
        watch.choice_answer(answer)
        if answer.get("confidence") is not None:
            confidences.append(float(answer["confidence"]))
        if heuristic_of is not None:
            guess = heuristic_of(r.call, r.call.meta)
            if guess is not None:
                heuristic.append(str(guess) == str(truth))

    n = len(correct)
    lo, hi = metrics.wilson_interval(sum(correct), n) if n else (0.0, 1.0)
    return {
        "n": n,
        "failures": failures,
        "accuracy": (sum(correct) / n) if n else None,
        "wilson_95": [lo, hi],
        "chance_baseline": _mean(chance),
        "majority_baseline": metrics.majority_baseline(labels) if labels else None,
        "heuristic_baseline": (sum(heuristic) / len(heuristic)) if heuristic else None,
        "heuristic_n": len(heuristic),
        "mean_confidence": _mean(confidences),
        "selections_not_among_the_options": off_menu,
        "correct": correct,
    }


def _sudoku_heuristic(call: Call, meta: dict) -> str | None:
    """Take a one-step hidden single if there is exactly one; else the digit
    already placed most often elsewhere in the grid. Both features are in the
    generator's meta, so this is recomputable from the log alone."""
    singles = meta.get("hidden_single_candidates") or []
    if len(singles) == 1:
        return str(singles[0])
    counts = meta.get("candidate_placed_counts") or {}
    if not counts:
        return None
    # Lowest digit on a tie: `max` keeps the first maximal element it sees.
    return max(sorted(counts, key=int), key=lambda d: counts[d])


def _successor_heuristic(call: Call, meta: dict) -> str | None:
    """Score each option's label against the goal clauses and take the best.

    This is the "20 lines of code" the rubric's Bad tier is defined against, and
    for this construction it is exact -- which is the finding, not a flaw in the
    baseline.
    """
    state = call.state
    if not isinstance(state, dict):
        return None
    clauses: list[tuple[str, bool]] = []
    for clause in state.get("goal_condition") or []:
        text = str(clause)
        if text.endswith(" is true"):
            clauses.append((text[: -len(" is true")], True))
        elif text.endswith(" is false"):
            clauses.append((text[: -len(" is false")], False))
    if not clauses:
        return None
    question = call.questions.get("successor") or {}
    best_id, best_score = None, -1
    for option in question.get("options") or []:
        label = str(option.get("label", ""))
        true_props = {
            s.strip()
            for s in label.split(":", 1)[-1].split(",")
            if s.strip() and s.strip() != "(none)"
        }
        score = sum(1 for name, want in clauses if (name in true_props) == want)
        if score > best_score:
            best_id, best_score = str(option["id"]), score
    return best_id


# --------------------------------------------------------------------------
# The watch list
# --------------------------------------------------------------------------


class Watchlist:
    """Counters for the behaviours the plan's "unexpected behaviors" section
    names: probability quantization, and `confidence` diverging from
    max-probability. Both are on the plan's watch list and E5 sees a lot of
    both, so it counts them rather than leaving them to another experiment."""

    def __init__(self) -> None:
        self.probabilities = 0
        self.off_grid = 0
        self.conf_pairs = 0
        self.conf_gap_sum = 0.0
        self.conf_gap_max = 0.0
        self.conf_below_max = 0
        self.sum_off = 0
        self.sum_seen = 0
        self.missing_ids = 0
        self.extra_ids = 0
        self.extra_mass_max = 0.0
        self._seen: set[Any] = set()

    def probability(self, p: float, key: Any = None) -> None:
        """One returned probability. `key` identifies the answer it came from.

        A keyed probability is counted once however many times it is read. The
        four nouls of a 16-cell scene are read twice -- once against the legible
        joint and once against the bit-string one -- and counting them twice
        would inflate the denominator of the quantization anomaly by the whole
        marginal-consistency condition.
        """
        if key is not None:
            if key in self._seen:
                return
            self._seen.add(key)
        self.probabilities += 1
        if abs(p * 100.0 - round(p * 100.0)) > 1e-6:
            self.off_grid += 1

    def choice_answer(self, answer: dict) -> None:
        for v in (answer.get("probabilities") or {}).values():
            self.probability(float(v))
        conf, mx = answer.get("confidence"), answer.get("max_probability")
        if conf is not None and mx is not None:
            self.conf_pairs += 1
            gap = float(conf) - float(mx)
            self.conf_gap_sum += abs(gap)
            self.conf_gap_max = max(self.conf_gap_max, abs(gap))
            if gap < -1e-9:
                self.conf_below_max += 1

    def joint(self, read: JointRead) -> None:
        """Only the shape of the returned distribution. The probabilities and the
        confidence of the same answer go through `choice_answer`, so that a joint
        read does not count either of them twice."""
        self.sum_seen += 1
        if abs(read.raw_sum - 1.0) > SUM_TOLERANCE:
            self.sum_off += 1
        self.missing_ids += read.missing_ids
        self.extra_ids += read.extra_ids
        self.extra_mass_max = max(self.extra_mass_max, read.extra_mass)

    def report(self) -> list[str]:
        out: list[str] = []
        if self.probabilities:
            if self.off_grid == 0:
                out.append(
                    f"Every one of the {self.probabilities} probabilities E5 saw -- noul "
                    "answers and choice distributions alike -- was an exact multiple of "
                    "0.01. The plan's quantization watch item is confirmed here."
                )
            else:
                out.append(
                    f"{self.off_grid} of {self.probabilities} probabilities were not "
                    "multiples of 0.01, so the two-decimal quantization seen elsewhere "
                    "is not universal."
                )
        if self.conf_pairs:
            mean_gap = self.conf_gap_sum / self.conf_pairs
            out.append(
                f"`confidence` differed from max-probability by {mean_gap:.3f} on average "
                f"(worst {self.conf_gap_max:.3f}) over {self.conf_pairs} choice answers; "
                f"{self.conf_below_max} were below the max probability. The two are not "
                "the same quantity."
            )
        if self.sum_seen and self.sum_off:
            out.append(
                f"{self.sum_off} of {self.sum_seen} returned joint distributions had raw "
                f"mass further than {SUM_TOLERANCE} from 1 and were renormalized before "
                "any metric was computed."
            )
        if self.missing_ids:
            out.append(
                f"{self.missing_ids} option ids were absent from the returned "
                "probability maps and were read as zero mass."
            )
        if self.extra_ids:
            out.append(
                f"{self.extra_ids} probabilities came back keyed to ids that were "
                f"never sent (up to {self.extra_mass_max:.3f} of mass on one answer). "
                "That mass is not on any cell of the enrolled space, so it was "
                "dropped by the renormalization and is reported here instead."
            )
        return out


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


def _small(n: int | None) -> str:
    if n is None:
        return " No instance was scored, so this verdict carries no weight."
    if n < MIN_MEANINGFUL_N:
        return (
            f" Only {n} instances were scored (JEV_SCALE={config.SCALE}), which is "
            "below the point at which this mean means anything."
        )
    return ""


def _score_predictions(agg: dict) -> list[dict]:
    """P12 through P15, each with a verdict and the number it rests on."""
    claims = {p.id: p for p in for_experiment(EXPERIMENT)}
    out: list[dict] = []

    # -- P12: forbidden-cell mass 0.08-0.15 against the 0.25 product baseline.
    mass = agg["forbidden_mass"]
    base = agg["forbidden_mass_baseline"]
    n = agg["forbidden_n"]
    if mass is None:
        verdict, outcome = "untestable", (
            "No exclusion or implication instance produced a readable joint, so the "
            "forbidden-cell mass was never measured."
        )
    else:
        verdict = "right" if 0.08 <= mass <= 0.15 else "wrong"
        how = (
            "inside the predicted band"
            if verdict == "right"
            else ("below the predicted band -- enrollment helped more than predicted"
                  if mass < 0.08 else "above the predicted band")
        )
        falsified = " The plan's falsification condition (mass about 0.25) is met." if abs(mass - 0.25) <= 0.03 else ""
        def _f(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.4f}"

        outcome = (
            f"Forbidden-cell mass {mass:.4f}, {how}, against a product-of-marginals "
            f"baseline of {base:.4f} computed from the model's own standalone nouls "
            f"(the plan's nominal figure is 0.25). Pooled over both relations; "
            f"separately, exclusion {_f(agg['forbidden_mass_exclusion'])} and "
            f"implication {_f(agg['forbidden_mass_implication'])}. Counting every "
            "impossible cell rather than only the one the plan designates -- which for "
            "exclusion means (false, false) as well as (true, true) -- the mass is "
            f"{_f(agg['impossible_mass_exclusion'])} on exclusion and "
            f"{_f(agg['impossible_mass_implication'])} on implication. n={n}.{falsified}"
            + _small(n)
        )
    out.append({"id": "P12", "claim": claims["P12"].claim, "outcome": outcome,
                "verdict": verdict,
                "evidence": {
                    "forbidden_mass": mass, "product_baseline": base,
                    "nominal_baseline": 0.25, "n": n,
                    "forbidden_mass_exclusion": agg["forbidden_mass_exclusion"],
                    "forbidden_mass_implication": agg["forbidden_mass_implication"],
                    "impossible_mass_exclusion": agg["impossible_mass_exclusion"],
                    "impossible_mass_implication": agg["impossible_mass_implication"],
                }})

    # -- P13: KL(joint || product) in 0.2-0.5.
    kl = agg["kl_joint_product"]
    kl_ref = agg["kl_reference"]
    tc = agg["total_correlation"]
    n_kl = agg["world_n"]
    if kl is None:
        verdict, outcome = "untestable", (
            "No four-proposition instance produced both a noul call and a readable "
            "joint, so KL was never measured."
        )
    else:
        verdict = "right" if 0.2 <= kl <= 0.5 else "wrong"
        falsified = " The plan's falsification condition (KL about 0) is met." if kl <= 0.05 else ""
        outcome = (
            f"KL(joint || product of the standalone marginals) = {kl:.4f} nats, against "
            f"{kl_ref:.4f} nats of dependence actually present in the "
            f"uniform-over-consistent-worlds reference. Of the measured KL, {tc:.4f} is "
            "the joint's own total correlation and the remainder is disagreement "
            f"between the two encodings' marginals. n={n_kl}.{falsified}" + _small(n_kl)
        )
    out.append({"id": "P13", "claim": claims["P13"].claim, "outcome": outcome,
                "verdict": verdict,
                "evidence": {"kl_joint_product": kl, "kl_reference": kl_ref,
                             "total_correlation": tc, "n": n_kl}})

    # -- P14: legible ids beat bit strings by >=15 points.
    gap = agg["labeled_vs_bitstring_gap"]
    n_lab = agg["labelling_n"]
    if gap is None:
        verdict, outcome = "untestable", (
            "The labelling condition produced no paired instances, so the gap was "
            "never measured."
        )
    else:
        verdict = "right" if gap >= 0.15 else "wrong"
        falsified = " The plan's falsification condition (gap under 5 points) is met." if gap < 0.05 else ""
        outcome = (
            f"Legible option ids scored {agg['labelled_accuracy']:.4f} and bit strings "
            f"{agg['bitstring_accuracy']:.4f} on the same instances, a gap of "
            f"{gap * 100:.1f} percentage points (McNemar p={agg['labelling_p']}). The "
            "prediction's 15% is read as 15 percentage points of accuracy, not a 15% "
            "relative improvement. Accuracy here is the chosen cell being logically "
            "consistent with the state. n="
            f"{n_lab}.{falsified}" + _small(n_lab)
        )
    out.append({"id": "P14", "claim": claims["P14"].claim, "outcome": outcome,
                "verdict": verdict,
                "evidence": {"gap": gap, "labelled": agg["labelled_accuracy"],
                             "bitstring": agg["bitstring_accuracy"], "n": n_lab}})

    # -- P15: Sudoku near-perfect forced, near-random by five options.
    forced = agg["sudoku_forced_accuracy"]
    five = agg["sudoku_five"]
    five_chance = agg["sudoku_five_chance"]
    deviation = (
        "The generator cannot ask a literal one-option choice -- `instances.choice` "
        "requires two -- so a forced cell is asked as the legal digit plus one "
        "eliminated decoy, which puts chance at 0.50 rather than 1.00 and makes the "
        "forced condition an elimination test. P15 assumed the literal version."
    )
    if forced is None or five is None:
        verdict, outcome = "untestable", (
            "The Sudoku condition did not produce both a one-option and a five-option "
            "level, so the two halves of P15 could not be compared. " + deviation
        )
    else:
        near_perfect = forced >= 0.95
        near_random = five <= (five_chance or 0.2) + 0.10
        strong_at_five = five >= (five_chance or 0.2) + 0.25
        verdict = "right" if (near_perfect and near_random) else "wrong"
        falsified = " The plan's falsification condition (strong at 5+) is met." if strong_at_five else ""
        outcome = (
            f"Forced moves (one legal digit) scored {forced:.4f} against a 0.50 chance "
            f"baseline; five legal digits scored {five:.4f} against "
            f"{five_chance:.4f} chance. Near-perfect at one: {near_perfect}. "
            f"Near-random at five: {near_random}. n={agg['sudoku_forced_n']} forced, "
            f"{agg['sudoku_five_n']} at five.{falsified} " + deviation
            + _small(min(agg["sudoku_forced_n"], agg["sudoku_five_n"]))
        )
    out.append({"id": "P15", "claim": claims["P15"].claim, "outcome": outcome,
                "verdict": verdict,
                "evidence": {"forced_accuracy": forced, "five_accuracy": five,
                             "five_chance": five_chance,
                             "forced_n": agg["sudoku_forced_n"],
                             "five_n": agg["sudoku_five_n"]}})
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _make_plots(
    run_dir: Path,
    *,
    pair_scored: dict,
    pair_modes: Sequence[str],
    world_levels: Sequence[int],
    world_scored: dict,
    world_bits_scored: dict,
    sudoku_scored: dict,
) -> list[str]:
    """Every figure E5 emits, as paths relative to the run directory.

    Each one is skipped rather than drawn empty when its condition scored
    nothing: `plots.curve` refuses an empty axis, and a figure with a caption
    and no points is worse in a report than a missing figure.
    """
    plot_paths: list[str] = []
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    pair_labels = [f"{rel}\n{mode}" for rel in ("exclusion", "implication") for mode in pair_modes]
    pair_mass = [
        _mean(_col(pair_scored[(rel, mode)]["rows"], "forbidden_mass"))
        for rel in ("exclusion", "implication")
        for mode in pair_modes
    ]
    pair_base = [
        _mean(_col(pair_scored[(rel, mode)]["rows"], "forbidden_product_baseline"))
        for rel in ("exclusion", "implication")
        for mode in pair_modes
    ]
    pair_n = [pair_scored[(rel, mode)]["n"] for rel in ("exclusion", "implication") for mode in pair_modes]
    if all(v is not None for v in pair_mass) and sum(pair_n) > 0:
        path = plots.curve(
            pair_labels,
            {
                "forbidden-cell mass in the joint": [float(v) for v in pair_mass],
                "product of the standalone marginals": [float(v or 0.0) for v in pair_base],
            },
            plots_dir / "e5_forbidden_mass.png",
            "Mass the joint places on a logically forbidden cell",
            "condition",
            "probability mass",
            n=pair_n,
            baseline=0.25,
            baseline_label="plan's nominal product baseline",
            ylim=(0.0, 0.45),
            note="lower is better; a constraint checker would put zero here",
        )
        plot_paths.append(f"plots/{path.name}")

    world_n_list = [world_scored[k]["n"] for k in world_levels]
    if sum(world_n_list) > 0:
        kl_vals = [_mean(_col(world_scored[k]["rows"], "kl_joint_product")) for k in world_levels]
        ref_vals = [_mean(_col(world_scored[k]["rows"], "kl_reference")) for k in world_levels]
        tc_vals = [_mean(_col(world_scored[k]["rows"], "total_correlation")) for k in world_levels]
        if all(v is not None for v in kl_vals):
            path = plots.curve(
                world_levels,
                {
                    "KL(joint || product of standalone marginals)": [float(v) for v in kl_vals],
                    "total correlation within the joint": [float(v or 0.0) for v in tc_vals],
                    "dependence present in the reference": [float(v or 0.0) for v in ref_vals],
                },
                plots_dir / "e5_joint_vs_product.png",
                "Does the enrolled joint express more than its marginals?",
                "constraints linking the four propositions",
                "nats",
                n=world_n_list,
                xticks=world_levels,
                note="KL near zero would mean enrollment bought nothing",
            )
            plot_paths.append(f"plots/{path.name}")

        leg = [_mean([float(r["chosen_consistent"]) for r in world_scored[k]["rows"]]) for k in world_levels]
        bit = [_mean([float(r["chosen_consistent"]) for r in world_bits_scored[k]["rows"]]) for k in world_levels]
        chance = [_mean(_col(world_scored[k]["rows"], "chance")) for k in world_levels]
        if all(v is not None for v in leg) and all(v is not None for v in bit):
            path = plots.curve(
                world_levels,
                {
                    "legible option ids": [float(v) for v in leg],
                    "bit strings": [float(v) for v in bit],
                    "random option": [float(v or 0.0) for v in chance],
                },
                plots_dir / "e5_labelling.png",
                "Legible option ids against bit strings, same instances",
                "constraints linking the four propositions",
                "chosen cell is logically possible",
                n=world_n_list,
                xticks=world_levels,
                ylim=(0.0, 1.0),
            )
            plot_paths.append(f"plots/{path.name}")

    sud_keys = sorted(sudoku_scored)
    sud_n = [sudoku_scored[k]["n"] for k in sud_keys]
    if sud_keys and sum(sud_n) > 0 and all(sudoku_scored[k]["accuracy"] is not None for k in sud_keys):
        series = {
            "accuracy": [float(sudoku_scored[k]["accuracy"]) for k in sud_keys],
            "random option": [float(sudoku_scored[k]["chance_baseline"] or 0.0) for k in sud_keys],
        }
        if all(sudoku_scored[k]["heuristic_baseline"] is not None for k in sud_keys):
            series["cheap heuristic"] = [
                float(sudoku_scored[k]["heuristic_baseline"]) for k in sud_keys
            ]
        band_lo = [sudoku_scored[k]["wilson_95"][0] for k in sud_keys]
        band_hi = [sudoku_scored[k]["wilson_95"][1] for k in sud_keys]
        path = plots.curve(
            sud_keys,
            series,
            plots_dir / "e5_sudoku.png",
            "Sudoku cell choice by number of legal digits",
            "legal digits for the probed cell",
            "accuracy",
            n=sud_n,
            band={"accuracy": (band_lo, band_hi)},
            xticks=sud_keys,
            ylim=(0.0, 1.0),
            note="one legal digit is asked as that digit plus one eliminated decoy",
        )
        plot_paths.append(f"plots/{path.name}")

    return plot_paths


# --------------------------------------------------------------------------
# The experiment
# --------------------------------------------------------------------------

QUESTION = (
    "Four nouls give four marginals and one choice over the combinations gives a "
    "joint; does enrolling the outcome space buy coherence the separate questions "
    "cannot express?"
)


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    n_world = config.n(config.SIZES.enrollment)
    n_pair = config.n(config.SIZES.enrollment)
    n_applied = config.n(config.SIZES.accuracy)
    n_control = config.n(config.SIZES.accuracy)

    world_levels = [2, 3, 4]
    pair_modes = ["stated", "structural"]

    # -- build every instance before spending a call ----------------------
    world_scenes: dict[int, list[Scene]] = {
        k: [
            _world_scene(k, config.seed_for(EXPERIMENT, f"world-{k}"), i)
            for i in range(n_world)
        ]
        for k in world_levels
    }
    pair_scenes: dict[tuple[str, str], list[Scene]] = {}
    for relation in ("exclusion", "implication"):
        for mode in pair_modes:
            pair_scenes[(relation, mode)] = [
                _pair_scene(relation, mode, config.seed_for(EXPERIMENT, f"{relation}-{mode}"), i)
                for i in range(n_pair)
            ]

    sudoku_levels = sudoku.difficulty_sweep()
    sudoku_instances = {
        int(d["n_options"]): sudoku.generate(
            difficulty=d,
            seed=config.seed_for(EXPERIMENT, f"sudoku-{d['n_options']}"),
            count=n_applied,
        )
        for d in sudoku_levels
    }
    succ_levels = [d for d in dfa.difficulty_sweep() if d.get("mode") == "successors"]
    succ_instances = {
        int(d["n_props"]): dfa.generate(
            difficulty=d,
            seed=config.seed_for(EXPERIMENT, f"succ-{d['n_props']}"),
            count=n_applied,
        )
        for d in succ_levels
    }
    control_instances = semantic.generate(
        difficulty={"level": "hard", "question_type": "choice"},
        seed=config.seed_for(EXPERIMENT, "control"),
        count=n_control,
    )

    # -- spend the calls ---------------------------------------------------
    world_noul: dict[int, list[CallResult]] = {}
    world_joint: dict[int, list[CallResult]] = {}
    world_bits: dict[int, list[CallResult]] = {}
    pair_noul: dict[tuple[str, str], list[CallResult]] = {}
    pair_joint: dict[tuple[str, str], list[CallResult]] = {}
    sudoku_results: dict[int, list[CallResult]] = {}
    succ_results: dict[int, list[CallResult]] = {}

    with JevClient(run_dir=run_dir, log_name="e5.jsonl") as client:
        for k, scenes in world_scenes.items():
            cond = f"enrolled-joint/k{k}"
            world_noul[k] = client.run(
                [_noul_call(s, f"{cond}/nouls") for s in scenes], label=f"E5/world{k}/nouls"
            )
            world_joint[k] = client.run(
                [_joint_call(s, f"{cond}/legible", "legible") for s in scenes],
                label=f"E5/world{k}/legible",
            )
            world_bits[k] = client.run(
                [_joint_call(s, f"{cond}/bitstring", "bitstring") for s in scenes],
                label=f"E5/world{k}/bitstring",
            )
        for key, scenes in pair_scenes.items():
            relation, mode = key
            cond = f"{relation}/{mode}"
            pair_noul[key] = client.run(
                [_noul_call(s, f"{cond}/nouls") for s in scenes], label=f"E5/{cond}/nouls"
            )
            pair_joint[key] = client.run(
                [_joint_call(s, f"{cond}/joint", "legible") for s in scenes],
                label=f"E5/{cond}/joint",
            )
        for k, insts in sudoku_instances.items():
            sudoku_results[k] = client.run(
                [_instance_call(i, f"sudoku/options{k}") for i in insts], label=f"E5/sudoku{k}"
            )
        for k, insts in succ_instances.items():
            succ_results[k] = client.run(
                [_instance_call(i, f"successors/props{k}") for i in insts],
                label=f"E5/successors{k}",
            )
        control_results = client.run(
            [_instance_call(i, "positive-control") for i in control_instances],
            label="E5/control",
        )
        summary = client.summary()

    # -- score offline ----------------------------------------------------
    watch = Watchlist()
    notes: list[str] = []
    anomalies: list[str] = []
    # Kept apart: a call that failed and a response that could not be read are
    # different problems, and reporting them under one number hides which.
    excluded_failed = 0
    excluded_unreadable = 0
    # Scored as wrong rather than excluded, so it is counted here and reported
    # beside the exclusions rather than inside them.
    joint_off_menu = 0

    joint_keys = [
        "marginal_disagreement", "max_marginal_disagreement", "kl_joint_product",
        "total_correlation", "kl_reference", "joint_entropy",
        "marginal_implied_entropy", "reference_entropy", "impossible_mass", "chance",
    ]

    world_scored: dict[int, dict] = {}
    world_bits_scored: dict[int, dict] = {}
    for k, scenes in world_scenes.items():
        world_scored[k] = _score_joint_condition(
            scenes, world_noul[k], world_joint[k], encoding="legible", watch=watch
        )
        world_bits_scored[k] = _score_joint_condition(
            scenes, world_noul[k], world_bits[k], encoding="bitstring", watch=watch
        )
        excluded_failed += (
            world_scored[k]["excluded_failed_call"]
            + world_bits_scored[k]["excluded_failed_call"]
        )
        excluded_unreadable += (
            world_scored[k]["excluded_unreadable"]
            + world_bits_scored[k]["excluded_unreadable"]
        )
        joint_off_menu += (
            world_scored[k]["selections_not_among_the_options"]
            + world_bits_scored[k]["selections_not_among_the_options"]
        )

    pair_scored: dict[tuple[str, str], dict] = {}
    for key, scenes in pair_scenes.items():
        pair_scored[key] = _score_joint_condition(
            scenes, pair_noul[key], pair_joint[key], encoding="legible", watch=watch
        )
        excluded_failed += pair_scored[key]["excluded_failed_call"]
        excluded_unreadable += pair_scored[key]["excluded_unreadable"]
        joint_off_menu += pair_scored[key]["selections_not_among_the_options"]

    sudoku_scored = {
        k: _accuracy_block(
            res,
            question_key=sudoku.QUESTION_KEY,
            truth_of=lambda c: c.meta["truth"][sudoku.QUESTION_KEY],
            chance_of=lambda c: float(c.meta.get("baseline_random", 0.0)),
            heuristic_of=_sudoku_heuristic,
            watch=watch,
        )
        for k, res in sudoku_results.items()
    }
    succ_scored = {
        k: _accuracy_block(
            res,
            question_key="successor",
            truth_of=lambda c: c.meta["truth"]["successor"],
            chance_of=lambda c: float(c.meta.get("baseline_random", 0.0)),
            heuristic_of=_successor_heuristic,
            watch=watch,
        )
        for k, res in succ_results.items()
    }
    control_scored = _accuracy_block(
        control_results,
        question_key=semantic.KEY_CHOICE,
        truth_of=lambda c: c.meta["truth"][semantic.KEY_CHOICE],
        chance_of=lambda c: 1.0 / max(1, int(c.meta.get("n_departments", 8))),
        heuristic_of=lambda c, m: m.get("baseline_keyword_prediction"),
        watch=watch,
    )
    if control_scored["n"] == 0:
        notes.append("The positive control returned no scored instance.")
    elif (
        control_scored["accuracy"] is not None
        and control_scored["chance_baseline"] is not None
        and control_scored["accuracy"] <= control_scored["chance_baseline"]
    ):
        notes.append(
            f"The semantic positive control scored {control_scored['accuracy']:.3f} "
            f"against chance {control_scored['chance_baseline']:.3f}. Per the plan a "
            "failing control makes every off-distribution result here uninterpretable; "
            "treat E5's numbers as suspect until the harness is checked."
        )

    # -- aggregates -------------------------------------------------------
    world_rows = [r for k in world_levels for r in world_scored[k]["rows"]]
    pair_rows_all = [r for key in pair_scenes for r in pair_scored[key]["rows"]]
    forbidden_rows = [r for r in pair_rows_all if "forbidden_mass" in r]
    rows_by_relation = {
        rel: [r for mode in pair_modes for r in pair_scored[(rel, mode)]["rows"]]
        for rel in ("exclusion", "implication")
    }

    labelled_pairs: list[tuple[bool, bool]] = []
    for k in world_levels:
        legible = {r["instance_id"]: r["chosen_consistent"] for r in world_scored[k]["rows"]}
        bits = {r["instance_id"]: r["chosen_consistent"] for r in world_bits_scored[k]["rows"]}
        for sid in legible.keys() & bits.keys():
            labelled_pairs.append((legible[sid], bits[sid]))
    lab_a = [a for a, _ in labelled_pairs]
    lab_b = [b for _, b in labelled_pairs]
    labelling_p = metrics.mcnemar(lab_a, lab_b) if labelled_pairs else None

    sudoku_forced = sudoku_scored.get(1, {})
    sudoku_five = sudoku_scored.get(5, {})

    agg = {
        "forbidden_mass": _mean(_col(forbidden_rows, "forbidden_mass")),
        "forbidden_mass_baseline": _mean(_col(forbidden_rows, "forbidden_product_baseline")),
        "forbidden_reference_baseline": _mean(
            _col(forbidden_rows, "forbidden_reference_baseline")
        ),
        "forbidden_n": len(forbidden_rows),
        # Pooled is what P12 predicts, but the two relations behave very
        # differently and the pooled mean hides that, so both are kept.
        "forbidden_mass_exclusion": _mean(_col(rows_by_relation["exclusion"], "forbidden_mass")),
        "forbidden_mass_implication": _mean(
            _col(rows_by_relation["implication"], "forbidden_mass")
        ),
        "impossible_mass_exclusion": _mean(
            _col(rows_by_relation["exclusion"], "impossible_mass")
        ),
        "impossible_mass_implication": _mean(
            _col(rows_by_relation["implication"], "impossible_mass")
        ),
        "impossible_mass": _mean(_col(pair_rows_all, "impossible_mass")),
        "marginal_disagreement": _mean(_col(world_rows, "marginal_disagreement")),
        "kl_joint_product": _mean(_col(world_rows, "kl_joint_product")),
        "kl_joint_product_median": _median(_col(world_rows, "kl_joint_product")),
        "total_correlation": _mean(_col(world_rows, "total_correlation")),
        "kl_reference": _mean(_col(world_rows, "kl_reference")),
        "joint_entropy": _mean(_col(world_rows, "joint_entropy")),
        "marginal_implied_entropy": _mean(_col(world_rows, "marginal_implied_entropy")),
        "reference_entropy": _mean(_col(world_rows, "reference_entropy")),
        "world_n": len(world_rows),
        "labelled_accuracy": _mean([float(a) for a in lab_a]),
        "bitstring_accuracy": _mean([float(b) for b in lab_b]),
        "labelling_n": len(labelled_pairs),
        "labelling_p": labelling_p,
        "sudoku_forced_accuracy": sudoku_forced.get("accuracy"),
        "sudoku_forced_n": sudoku_forced.get("n", 0),
        "sudoku_five": sudoku_five.get("accuracy"),
        "sudoku_five_chance": sudoku_five.get("chance_baseline"),
        "sudoku_five_n": sudoku_five.get("n", 0),
    }
    gap = (
        agg["labelled_accuracy"] - agg["bitstring_accuracy"]
        if agg["labelled_accuracy"] is not None and agg["bitstring_accuracy"] is not None
        else None
    )
    agg["labeled_vs_bitstring_gap"] = gap
    if agg["joint_entropy"] is not None and agg["marginal_implied_entropy"] is not None:
        agg["entropy_gap_nats"] = agg["marginal_implied_entropy"] - agg["joint_entropy"]
    else:
        agg["entropy_gap_nats"] = None

    # -- tier by difficulty ------------------------------------------------
    sweep: list[tuple[dict, dict]] = []

    for relation in ("exclusion", "implication"):
        for mode in pair_modes:
            s = pair_scored[(relation, mode)]
            rows = s["rows"]
            row_metrics: dict[str, Any] = {
                "forbidden_mass": _mean(_col(rows, "forbidden_mass")),
                "forbidden_mass_baseline": _mean(_col(rows, "forbidden_product_baseline")),
                "marginal_disagreement": _mean(_col(rows, "marginal_disagreement")),
                "kl_joint_product": _mean(_col(rows, "kl_joint_product")),
                "impossible_mass": _mean(_col(rows, "impossible_mass")),
                "entropy_gap_nats": (
                    _mean(_col(rows, "marginal_implied_entropy"))
                    - _mean(_col(rows, "joint_entropy"))
                    if rows
                    else None
                ),
                "accuracy": _mean([float(r["chosen_consistent"]) for r in rows]),
                "chance_baseline": _mean(_col(rows, "chance")),
                # Filtering four cells by a constraint the state states is the
                # cheap deterministic alternative, and it is exact. In the
                # `structural` mode the state does not state it -- deriving it
                # needs the reasoning this condition is testing -- so there is no
                # cheap deterministic route and claiming 1.0 there would invent a
                # baseline nothing could reach.
                "heuristic_baseline": (1.0 if rows else None) if mode == "stated" else None,
                "majority_baseline": _best_constant_baseline(rows),
                "off_menu_selections": sum(
                    1 for r in rows if r.get("chosen_off_menu")
                ),
                "n": len(rows),
            }
            sweep.append(({"condition": relation, "constraint": mode}, row_metrics))

    for k in world_levels:
        rows = world_scored[k]["rows"]
        sweep.append((
            {"condition": "enrolled-joint-16", "n_constraints": k},
            {
                "marginal_disagreement": _mean(_col(rows, "marginal_disagreement")),
                "kl_joint_product": _mean(_col(rows, "kl_joint_product")),
                "kl_reference": _mean(_col(rows, "kl_reference")),
                "impossible_mass": _mean(_col(rows, "impossible_mass")),
                "entropy_gap_nats": (
                    _mean(_col(rows, "marginal_implied_entropy"))
                    - _mean(_col(rows, "joint_entropy"))
                    if rows
                    else None
                ),
                "accuracy": _mean([float(r["chosen_consistent"]) for r in rows]),
                "chance_baseline": _mean(_col(rows, "chance")),
                "heuristic_baseline": 1.0 if rows else None,
                "majority_baseline": _best_constant_baseline(rows),
                "n": len(rows),
            },
        ))

    for k in world_levels:
        rows = world_bits_scored[k]["rows"]
        sweep.append((
            {"condition": "labelling-bitstring", "n_constraints": k},
            {
                "marginal_disagreement": _mean(_col(rows, "marginal_disagreement")),
                "kl_joint_product": _mean(_col(rows, "kl_joint_product")),
                "impossible_mass": _mean(_col(rows, "impossible_mass")),
                "accuracy": _mean([float(r["chosen_consistent"]) for r in rows]),
                "chance_baseline": _mean(_col(rows, "chance")),
                "heuristic_baseline": 1.0 if rows else None,
                "majority_baseline": _best_constant_baseline(rows),
                "n": len(rows),
            },
        ))

    for k in sorted(sudoku_scored):
        s = sudoku_scored[k]
        row_metrics = {
            "accuracy": s["accuracy"],
            "chance_baseline": s["chance_baseline"],
            "heuristic_baseline": s["heuristic_baseline"],
            "majority_baseline": s["majority_baseline"],
            "n": s["n"],
        }
        if k == 1:
            row_metrics["sudoku_forced_accuracy"] = s["accuracy"]
        sweep.append(({"condition": "sudoku", "n_options": k}, row_metrics))

    for k in sorted(succ_scored):
        s = succ_scored[k]
        insts = succ_instances[k]
        sweep.append((
            {"condition": "reachable-successors", "n_props": k},
            {
                "accuracy": s["accuracy"],
                "chance_baseline": s["chance_baseline"],
                "heuristic_baseline": s["heuristic_baseline"],
                "majority_baseline": s["majority_baseline"],
                "n_reachable_successors": _mean(
                    [float(i.meta["n_reachable_successors"]) for i in insts]
                ),
                "n_invariant_satisfying": _mean(
                    [float(i.meta["n_invariant_satisfying"]) for i in insts]
                ),
                "n": s["n"],
            },
        ))

    graded = tiers.tier_by_difficulty(EXPERIMENT, sweep)
    crossings = tiers.boundary_crossings(EXPERIMENT, graded)
    by_condition: dict[str, Any] = {}
    for name in ("exclusion", "implication", "enrolled-joint-16", "labelling-bitstring",
                 "sudoku", "reachable-successors"):
        subset = [r for r in graded if r.difficulty.get("condition") == name]
        if subset:
            by_condition[name] = tiers.crossings_json(
                tiers.boundary_crossings(EXPERIMENT, subset)
            )

    overall_metrics = {
        "forbidden_mass": agg["forbidden_mass"],
        "forbidden_mass_baseline": agg["forbidden_mass_baseline"],
        "marginal_disagreement": agg["marginal_disagreement"],
        "kl_joint_product": agg["kl_joint_product"],
        "sudoku_forced_accuracy": agg["sudoku_forced_accuracy"],
        "labeled_vs_bitstring_gap": agg["labeled_vs_bitstring_gap"],
        "n": agg["forbidden_n"] + agg["world_n"],
    }
    overall = tiers.assign(EXPERIMENT, overall_metrics, difficulty={"condition": "all E5"})

    # -- anomalies ---------------------------------------------------------
    anomalies.extend(watch.report())
    if agg["entropy_gap_nats"] is not None:
        if agg["entropy_gap_nats"] > 0.1:
            anomalies.append(
                f"The plan's predicted surprise appears: the 16-way joint carries "
                f"{agg['joint_entropy']:.3f} nats of entropy while the four standalone "
                f"nouls imply {agg['marginal_implied_entropy']:.3f} nats, a gap of "
                f"{agg['entropy_gap_nats']:.3f}. The joint is sharper than the marginals, "
                f"against {agg['reference_entropy']:.3f} nats in the reference."
                + _small(agg["world_n"])
            )
        elif agg["entropy_gap_nats"] < -0.1:
            anomalies.append(
                f"The joint is mushier than the marginals, not sharper: "
                f"{agg['joint_entropy']:.3f} nats against {agg['marginal_implied_entropy']:.3f} "
                "implied by the standalone nouls. That is the opposite of the plan's "
                "predicted surprise." + _small(agg["world_n"])
            )
    applied_off_menu = (
        sum(s["selections_not_among_the_options"] for s in sudoku_scored.values())
        + sum(s["selections_not_among_the_options"] for s in succ_scored.values())
        + control_scored["selections_not_among_the_options"]
    )
    if joint_off_menu or applied_off_menu:
        anomalies.append(
            f"{joint_off_menu + applied_off_menu} answers named an option id that was "
            f"never sent ({joint_off_menu} on an enrolled joint, {applied_off_menu} on "
            "an applied condition). They are scored as wrong rather than excluded -- "
            "the call succeeded and the model did answer -- and on a joint the returned "
            "distribution is still used, since no distribution metric reads the "
            "selection."
        )
    if agg["impossible_mass"] is not None and agg["forbidden_n"]:
        anomalies.append(
            f"Across the two-proposition conditions the joint placed "
            f"{agg['impossible_mass']:.4f} of its mass on cells the state rules out "
            "altogether, counting every impossible cell rather than only the one the "
            "plan designates."
        )
    for k in sorted(succ_scored):
        insts = succ_instances[k]
        reach = _mean([float(i.meta["n_reachable_successors"]) for i in insts])
        inv = _mean([float(i.meta["n_invariant_satisfying"]) for i in insts])
        if reach is not None:
            anomalies.append(
                f"Reachable-successor enumeration at {k} propositions: 2^{k} = "
                f"{2 ** k} assignments in principle, {inv:.0f} satisfying the "
                f"invariants, {reach:.1f} reachable in one move. The 255-option cap "
                "binds on reachable states, not on propositions."
            )

    # -- plots -------------------------------------------------------------
    plot_paths = _make_plots(
        run_dir,
        pair_scored=pair_scored,
        pair_modes=pair_modes,
        world_levels=world_levels,
        world_scored=world_scored,
        world_bits_scored=world_bits_scored,
        sudoku_scored=sudoku_scored,
    )

    # -- assemble ----------------------------------------------------------
    notes.extend([
        "The four nouls of a state share one call. Asked as four calls, any "
        "disagreement with the joint would mix this experiment's question with E6's; "
        "batched, `marginal_disagreement` measures the two encodings only.",
        "Exclusion and implication states are verified at construction: the possible "
        "(A, B) cells are computed by evaluating both propositions over the scenario's "
        "enumerated worlds, and a scene whose possible set is not exactly "
        "{(true, false), (false, true)} -- or, for implication, that permits "
        "(true, false) or leaves a proposition determined -- raises.",
        "Accuracy on the constructed states is 'the chosen cell is logically possible "
        "given the state', not 'the chosen cell is the one true configuration': the "
        "states deliberately leave residual uncertainty so that KL from the product "
        "has something real to measure.",
        f"Marginals are clamped to [{FLOOR}, {1 - FLOOR}] before the outer product is "
        "formed, because a returned 0.00 is a two-decimal rounding of something below "
        "0.005 rather than an impossibility, and an exact zero in the product sends KL "
        "to infinity.",
        "The reference joint is uniform over the combinations the state permits. The "
        "states are written so that nothing distinguishes those combinations, but it "
        "is an assumption and every reference-derived number depends on it.",
    ])

    result: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "question": QUESTION,
        "headline": {
            "metric": "forbidden-cell mass in the enrolled joint",
            "value": agg["forbidden_mass"],
            "baseline": agg["forbidden_mass_baseline"],
            "baseline_name": (
                "outer product of the model's own standalone nouls "
                "(the plan's nominal figure is 0.25)"
            ),
            "n": agg["forbidden_n"],
        },
        "by_difficulty": [r.to_json() for r in graded],
        "boundary_crossings": tiers.crossings_json(crossings),
        "boundary_crossings_by_condition": by_condition,
        "boundary_crossings_axis": (
            "E5's conditions in order of the size and difficulty of the enrolled "
            "outcome space: 4-cell exclusion and implication, then the 16-cell joint "
            "and its bit-string relabelling, then Sudoku by number of legal digits, "
            "then reachable-successor enumeration. Only the Sudoku rungs form a single "
            "monotone ladder, so the per-condition crossings are the ones to read."
        ),
        "plots": plot_paths,
        "predictions": _score_predictions(agg),
        "anomalies": anomalies,
        "failures": {
            "calls": summary["failed_calls"],
            "excluded": excluded_failed
            + excluded_unreadable
            + sum(s["failures"] for s in sudoku_scored.values())
            + sum(s["failures"] for s in succ_scored.values())
            + control_scored["failures"],
            "reasons": {
                "http_and_transport": summary["status_counts"],
                "paired_call_failed": excluded_failed,
                "unreadable_distribution": excluded_unreadable,
                # Not excluded: scored as wrong, and on a joint the distribution
                # is still used. Counted here so the two are never confused.
                "selections_not_among_the_options": joint_off_menu + applied_off_menu,
                "applied_condition_failures": (
                    sum(s["failures"] for s in sudoku_scored.values())
                    + sum(s["failures"] for s in succ_scored.values())
                    + control_scored["failures"]
                ),
            },
        },
        "overall_tier": overall.to_json(),
        "aggregate": agg,
        "conditions": {
            "marginal-consistency": {
                str(k): {
                    "n": world_scored[k]["n"],
                    **_aggregate(world_scored[k], joint_keys),
                }
                for k in world_levels
            },
            "labelling-bitstring": {
                str(k): {
                    "n": world_bits_scored[k]["n"],
                    **_aggregate(world_bits_scored[k], joint_keys),
                }
                for k in world_levels
            },
            "exclusion": {
                mode: {
                    "n": pair_scored[("exclusion", mode)]["n"],
                    **_aggregate(pair_scored[("exclusion", mode)], joint_keys + ["forbidden_mass", "forbidden_product_baseline"]),
                }
                for mode in pair_modes
            },
            "implication": {
                mode: {
                    "n": pair_scored[("implication", mode)]["n"],
                    **_aggregate(pair_scored[("implication", mode)], joint_keys + ["forbidden_mass", "forbidden_product_baseline"]),
                }
                for mode in pair_modes
            },
            "sudoku": {
                str(k): {x: v for x, v in s.items() if x != "correct"}
                for k, s in sorted(sudoku_scored.items())
            },
            "reachable-successors": {
                str(k): {
                    **{x: v for x, v in s.items() if x != "correct"},
                    "n_props": k,
                    "n_assignments_total": 2 ** k,
                    "mean_invariant_satisfying": _mean(
                        [float(i.meta["n_invariant_satisfying"]) for i in succ_instances[k]]
                    ),
                    "mean_reachable_successors": _mean(
                        [float(i.meta["n_reachable_successors"]) for i in succ_instances[k]]
                    ),
                }
                for k, s in sorted(succ_scored.items())
            },
        },
        "positive_control": {x: v for x, v in control_scored.items() if x != "correct"},
        "notes": notes,
        "sample_sizes": {
            "scale": config.SCALE,
            "enrolled_joint_states_per_level": n_world,
            "pair_states_per_mode": n_pair,
            "applied_instances_per_level": n_applied,
            "positive_control": n_control,
        },
        "run": summary,
    }
    return result


def _tier_caveat(row: dict) -> str:
    """What has to be said beside a tier for it to be a reading rather than a label.

    Three things can make one hollow: nothing was scored, so the tier is the
    rubric's residual; the rung that matched rested on criteria that were never
    measured; or the sample is too small for the mean underneath it to mean
    anything. Printing the tier bare in any of those cases is the failure mode
    the plan's reporting rules exist to prevent.
    """
    n = row.get("n")
    if not n:
        return "  [UNMEASURED: nothing scored; the tier is the rubric's residual]"
    parts = []
    if row.get("unevaluated"):
        parts.append("tier from a partial rule: " + ", ".join(row["unevaluated"]) + " not measured")
    if not row.get("reportable", True):
        parts.append("not reportable: baseline or n missing")
    if n < MIN_MEANINGFUL_N:
        parts.append(f"n={n} is below {MIN_MEANINGFUL_N}, too few to read")
    return ("  [" + "; ".join(parts) + "]") if parts else ""


def format_report(result: dict) -> str:
    agg = result["aggregate"]
    lines = [f"E5 -- {result['question']}"]
    h = result["headline"]

    def f(v: Any, spec: str = ".4f") -> str:
        return "n/a" if v is None else format(v, spec)

    lines.append(
        f"  forbidden-cell mass {f(h['value'])} vs product-of-marginals "
        f"{f(h['baseline'])} (plan's nominal 0.25), n={h['n']}"
    )
    lines.append(
        f"  marginal disagreement {f(agg['marginal_disagreement'])}  "
        f"KL(joint||product) {f(agg['kl_joint_product'])} "
        f"(reference dependence {f(agg['kl_reference'])}, "
        f"total correlation {f(agg['total_correlation'])})"
    )
    lines.append(
        f"  joint entropy {f(agg['joint_entropy'])} nats vs "
        f"{f(agg['marginal_implied_entropy'])} implied by the marginals "
        f"(gap {f(agg['entropy_gap_nats'])})"
    )
    lines.append(
        f"  labelling: legible {f(agg['labelled_accuracy'])} vs bitstring "
        f"{f(agg['bitstring_accuracy'])}, gap {f(agg['labeled_vs_bitstring_gap'])}, "
        f"n={agg['labelling_n']}"
    )
    lines.append(
        f"  sudoku forced {f(agg['sudoku_forced_accuracy'])} (chance 0.50), "
        f"five options {f(agg['sudoku_five'])} (chance {f(agg['sudoku_five_chance'])})"
    )
    ctrl = result["positive_control"]
    lines.append(
        f"  positive control {f(ctrl.get('accuracy'))} vs chance "
        f"{f(ctrl.get('chance_baseline'))} / keyword heuristic "
        f"{f(ctrl.get('heuristic_baseline'))}, n={ctrl.get('n')}"
    )
    lines.append("  tier by difficulty:")
    for row in result["by_difficulty"]:
        d = row.get("difficulty") or {}
        where = ",".join(f"{k}={v}" for k, v in sorted(d.items()))
        lines.append(
            f"    {where:<48} {row.get('tier', '?'):<14} n={row.get('n')}"
            + _tier_caveat(row)
        )
    overall = result["overall_tier"]
    lines.append(f"  overall: {overall.get('tier')}" + _tier_caveat(overall))
    for p in result["predictions"]:
        lines.append(f"  {p['id']}: {p['verdict'].upper()} -- {p['outcome']}")
    fails = result["failures"]
    lines.append(f"  failed calls {fails['calls']}, excluded data points {fails['excluded']}")
    for a in result["anomalies"]:
        lines.append(f"  ! {a}")
    return "\n".join(lines)
