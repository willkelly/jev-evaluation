"""DFA acceptance, and reachable-successor enumeration for E5 item 7.

Two families of question live in this module because they share one substrate
-- a finite machine whose successors are fixed by construction -- while
answering different questions in the plan. `difficulty["mode"]` selects the
family.

`mode="accept"` is the difficulty ladder. A random DFA plus an input string,
and one noul asking whether the string is accepted. Nothing visible about the
automaton predicts the answer: the string has to be walked. Accept and reject
are exactly balanced by index parity, so the majority-class baseline is 0.50 at
every rung, and the automaton is rerolled until *both* answers are reachable at
the string's length, so the density of accepting states carries no information
about the label either. The ladder's main axis is string length, with state
count and alphabet growing alongside it -- a long walk on four states settles
into its recurrent states and stops being a length test.

`mode="successors"` is E5 item 7. A state carries 20 or more boolean
properties, so the assignment space is 2^20 or larger, but per-block invariants
cut that to a few thousand legal assignments and the one-move rule cuts it
again to 30-40 successors that are actually reachable. Those successors are
enumerated as a single choice, which is the point: the 255-option cap binds on
reachable states, not on propositions. All four counts go into meta --
propositions, total assignments, invariant-satisfying assignments, reachable
successors -- so the report can state the collapse rather than assert it.

Ground truth is computed here in both modes. Acceptance is decided by
simulating the automaton over the string. The successor question is decided by
evaluating the goal condition against every enumerated successor and asserting
that exactly one of them satisfies it.
"""

from __future__ import annotations

from typing import Any

from ..instances import Instance, choice, noul, rng_for, shuffled_options

NAME = "dfa"

_SYMBOLS = "abcd"

# Rerolling is needed when every state reachable at the target length happens to
# be accepting (or every one rejecting). That is common on small automata and
# rare on large ones; 200 attempts has never been approached in practice.
_MAX_DFA_ATTEMPTS = 200

# Successor mode: properties are grouped into blocks of four, and the invariant
# is a per-block whitelist of legal 4-bit patterns. Independent blocks make the
# number of invariant-satisfying assignments a closed form (patterns ** blocks)
# rather than something that has to be model-counted.
_BLOCK_SIZE = 4
_PATTERNS_PER_BLOCK_SPACE = 1 << _BLOCK_SIZE
_GOAL_LITERALS = 7

_PROPERTY_NAMES = [
    "pump_a_running", "pump_b_running", "valve_1_open", "valve_2_open",
    "tank_pressurised", "tank_vented", "heater_on", "chiller_on",
    "door_locked", "door_open", "alarm_armed", "alarm_muted",
    "battery_charging", "battery_isolated", "radio_linked", "radio_muted",
    "gps_locked", "imu_calibrated", "hatch_sealed", "hatch_latched",
    "fuel_crossfeed", "fuel_dumping", "gear_down", "gear_locked",
    "brake_engaged", "thrust_reversed", "autopilot_engaged", "manual_override",
    "cabin_lights_on", "beacon_flashing", "deice_active", "cargo_secured",
]


def difficulty_sweep() -> list[dict]:
    """Acceptance ladder first, then the E5 item 7 settings.

    The acceptance rungs are ordered easy to hard and are the ladder in the
    sense the generator contract means: four states and a four-symbol string is
    something a person traces without writing anything down, while sixteen
    states and a sixty-four symbol string is sixty-four table lookups with no
    shortcut. The knee is expected in between.

    The three successor rungs are a separate family, not a continuation of that
    ordering. They vary the proposition count while holding the reachable
    successor count in the 30-40 range the plan names, which is exactly the
    comparison E5 item 7 needs. An experiment that wants one family filters on
    ``d["mode"]``.
    """
    return [
        {"mode": "accept", "n_states": 4, "alphabet": 2, "string_len": 4},
        {"mode": "accept", "n_states": 6, "alphabet": 2, "string_len": 8},
        {"mode": "accept", "n_states": 8, "alphabet": 2, "string_len": 16},
        {"mode": "accept", "n_states": 10, "alphabet": 3, "string_len": 24},
        {"mode": "accept", "n_states": 12, "alphabet": 3, "string_len": 40},
        {"mode": "accept", "n_states": 16, "alphabet": 4, "string_len": 64},
        {"mode": "successors", "n_props": 20, "patterns_per_block": 7},
        {"mode": "successors", "n_props": 24, "patterns_per_block": 7},
        {"mode": "successors", "n_props": 28, "patterns_per_block": 6},
    ]


def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]:
    """`count` instances at one difficulty setting, instance `i` from `i` alone."""
    mode = difficulty.get("mode", "accept")
    if mode == "accept":
        return [_accept_instance(difficulty, seed, i) for i in range(count)]
    if mode == "successors":
        return [_successor_instance(difficulty, seed, i) for i in range(count)]
    raise ValueError(f"unknown dfa mode {mode!r}")


# --------------------------------------------------------------------------
# Acceptance
# --------------------------------------------------------------------------


def _weighted_choice(rng, items: list, weights: list[int]):
    """Exact weighted choice over integer weights.

    `random.choices` accumulates weights as floats, and the path counts here are
    k**string_len, which exceeds float precision long before it exceeds int.
    """
    total = sum(weights)
    r = rng.randrange(total)
    for item, w in zip(items, weights):
        r -= w
        if r < 0:
            return item
    return items[-1]


def _random_dfa(rng, n_states: int, symbols: list[str]) -> tuple[list[dict[str, int]], set[int]]:
    delta = [{a: rng.randrange(n_states) for a in symbols} for _ in range(n_states)]
    while True:
        accepting = {s for s in range(n_states) if rng.random() < 0.5}
        if 0 < len(accepting) < n_states:
            return delta, accepting


def _layer_counts(delta, symbols: list[str], start: int, length: int) -> list[dict[int, int]]:
    """counts[t][s] = number of length-t strings that drive start to s."""
    counts = [{start: 1}]
    for _ in range(length):
        nxt: dict[int, int] = {}
        for s, c in counts[-1].items():
            for a in symbols:
                t = delta[s][a]
                nxt[t] = nxt.get(t, 0) + c
        counts.append(nxt)
    return counts


def _sample_string(rng, delta, symbols, counts, length, accepting, target) -> str:
    """A string of the given length, drawn uniformly among those with the
    required acceptance outcome.

    Walking backwards with the layer counts as weights is what makes it uniform
    rather than merely valid; a forward rejection sampler would bias towards
    whichever outcome is easier to hit on this particular automaton.
    """
    rev: dict[int, list[tuple[int, str]]] = {}
    for s in range(len(delta)):
        for a in symbols:
            rev.setdefault(delta[s][a], []).append((s, a))

    finals = [s for s in counts[length] if (s in accepting) == target]
    state = _weighted_choice(rng, finals, [counts[length][s] for s in finals])
    out: list[str] = []
    for t in range(length, 0, -1):
        prev_layer = counts[t - 1]
        cands = [(s, a) for (s, a) in rev.get(state, []) if s in prev_layer]
        state, sym = _weighted_choice(rng, cands, [prev_layer[s] for (s, _) in cands])
        out.append(sym)
    out.reverse()
    return "".join(out)


def _trajectory(delta, start: int, string: str) -> list[int]:
    states = [start]
    for ch in string:
        states.append(delta[states[-1]][ch])
    return states


def _accept_instance(difficulty: dict, seed: int, index: int) -> Instance:
    rng = rng_for(NAME, difficulty, seed, index)
    n_states = difficulty["n_states"]
    k = difficulty["alphabet"]
    length = difficulty["string_len"]
    if not 2 <= k <= len(_SYMBOLS):
        raise ValueError(f"alphabet must be 2..{len(_SYMBOLS)}, got {k}")
    # Both bounds would otherwise fail obscurely: one accepting state out of one
    # state cannot leave a rejecting state to reach, and a zero-length string
    # has only the start state to end on.
    if n_states < 2:
        raise ValueError(f"n_states must be at least 2, got {n_states}")
    if length < 1:
        raise ValueError(f"string_len must be at least 1, got {length}")
    symbols = list(_SYMBOLS[:k])
    target = index % 2 == 0

    for attempt in range(1, _MAX_DFA_ATTEMPTS + 1):
        delta, accepting = _random_dfa(rng, n_states, symbols)
        counts = _layer_counts(delta, symbols, 0, length)
        final_layer = counts[length]
        # Require both outcomes reachable, not just the one wanted: rerolling on
        # the target alone would correlate the accepting-state density with the
        # label and hand a cheap baseline a real signal.
        if any(s in accepting for s in final_layer) and any(s not in accepting for s in final_layer):
            break
    else:
        raise RuntimeError(f"{NAME}: no usable DFA after {_MAX_DFA_ATTEMPTS} attempts at {difficulty}")

    string = _sample_string(rng, delta, symbols, counts, length, accepting, target)

    states = _trajectory(delta, 0, string)
    truth = states[-1] in accepting
    assert truth == target, "sampled string does not have its constructed outcome"

    accept_mass = sum(c for s, c in final_layer.items() if s in accepting)
    total_mass = sum(final_layer.values())

    state: dict[str, Any] = {
        "machine": "deterministic finite automaton",
        "alphabet": symbols,
        "states": [f"q{s}" for s in range(n_states)],
        "start_state": "q0",
        "accepting_states": [f"q{s}" for s in sorted(accepting)],
        "transitions": {
            f"q{s}": {a: f"q{delta[s][a]}" for a in symbols} for s in range(n_states)
        },
        "input_string": string,
    }
    questions = {
        "accepts": noul(
            "Reading the input string left to right from the start state, "
            "does the automaton end in an accepting state?"
        )
    }
    inst = Instance(
        generator=NAME,
        difficulty=dict(difficulty),
        seed=seed,
        index=index,
        state=state,
        questions=questions,
        truth={"accepts": truth},
        meta={
            "mode": "accept",
            "n_states": n_states,
            "alphabet": k,
            "string_len": length,
            "n_accepting": len(accepting),
            # Cheap-baseline features. accept_density is the fraction of states
            # that accept; layer_accept_share is the probability a *random*
            # string of this length is accepted on this automaton. Both are
            # uncorrelated with the label by construction, so a baseline built
            # on them should land at 0.50 -- which is the check that the label
            # really does require simulation.
            "accept_density": round(len(accepting) / n_states, 6),
            "layer_accept_share": round(accept_mass / total_mass, 6),
            "symbol_counts": {a: string.count(a) for a in symbols},
            "distinct_states_visited": len(set(states)),
            "states_reachable_at_end": len(final_layer),
            "dfa_attempts": attempt,
            "baseline_majority": 0.5,
        },
    )
    inst.validate()
    return inst


# --------------------------------------------------------------------------
# Reachable-successor enumeration (E5 item 7)
# --------------------------------------------------------------------------


def _prop_value(assignment: tuple[int, ...], prop: int) -> bool:
    return bool((assignment[prop // _BLOCK_SIZE] >> (prop % _BLOCK_SIZE)) & 1)


def _true_props(assignment: tuple[int, ...], names: list[str]) -> list[str]:
    return [n for i, n in enumerate(names) if _prop_value(assignment, i)]


def _satisfies(assignment: tuple[int, ...], goal: list[tuple[int, bool]]) -> bool:
    return all(_prop_value(assignment, p) == v for p, v in goal)


def _n_matched(assignment: tuple[int, ...], goal: list[tuple[int, bool]]) -> int:
    return sum(1 for p, v in goal if _prop_value(assignment, p) == v)


def _successor_instance(difficulty: dict, seed: int, index: int) -> Instance:
    rng = rng_for(NAME, difficulty, seed, index)
    n_props = difficulty["n_props"]
    ppb = difficulty["patterns_per_block"]
    if n_props % _BLOCK_SIZE or n_props < 2 * _BLOCK_SIZE or n_props > len(_PROPERTY_NAMES):
        raise ValueError(f"n_props must be a multiple of {_BLOCK_SIZE} in 8..{len(_PROPERTY_NAMES)}")
    if not 2 <= ppb <= _PATTERNS_PER_BLOCK_SPACE:
        raise ValueError(f"patterns_per_block must be 2..{_PATTERNS_PER_BLOCK_SPACE}")
    n_blocks = n_props // _BLOCK_SIZE

    names = rng.sample(_PROPERTY_NAMES, n_props)
    legal = [sorted(rng.sample(range(_PATTERNS_PER_BLOCK_SPACE), ppb)) for _ in range(n_blocks)]
    current = tuple(rng.choice(legal[b]) for b in range(n_blocks))

    # A move rewrites exactly one block to a different legal pattern of that
    # block. Every result satisfies the invariants and no two results coincide,
    # so this set is exactly the one-step reachable set.
    successors: list[tuple[int, ...]] = []
    for b in range(n_blocks):
        for pattern in legal[b]:
            if pattern != current[b]:
                succ = list(current)
                succ[b] = pattern
                successors.append(tuple(succ))
    if not 2 <= len(successors) <= 255:
        raise ValueError(f"{len(successors)} successors is outside the choice range")

    target_ix = rng.randrange(len(successors))
    target = successors[target_ix]
    changed = next(b for b in range(n_blocks) if target[b] != current[b])

    # The four literals of the changed block single out the target: every other
    # successor either left that block alone or set it to a different pattern.
    # Greedily shrink that set so the goal does not simply name the block, then
    # pad with literals from elsewhere that are true of the target. Padding can
    # only remove satisfiers, so uniqueness survives it.
    block_lits = [
        (changed * _BLOCK_SIZE + j, _prop_value(target, changed * _BLOCK_SIZE + j))
        for j in range(_BLOCK_SIZE)
    ]
    goal: list[tuple[int, bool]] = []
    remaining = [s for s in successors if s != target]
    while remaining:
        best = max(
            (lit for lit in block_lits if lit not in goal),
            key=lambda lit: sum(1 for s in remaining if _prop_value(s, lit[0]) != lit[1]),
        )
        goal.append(best)
        remaining = [s for s in remaining if _satisfies(s, [best])]
    distinguishing = len(goal)

    pad_pool = [
        i for i in range(n_props)
        if i // _BLOCK_SIZE != changed and (i, _prop_value(target, i)) not in goal
    ]
    rng.shuffle(pad_pool)
    for i in pad_pool[: max(0, _GOAL_LITERALS - len(goal))]:
        goal.append((i, _prop_value(target, i)))
    rng.shuffle(goal)

    matching = [s for s in successors if _satisfies(s, goal)]
    assert matching == [target], "goal condition does not pick out exactly one successor"
    runner_up = max(_n_matched(s, goal) for s in successors if s != target)

    options = [
        {
            "id": f"successor_{i:03d}",
            "label": "true: " + (", ".join(_true_props(s, names)) or "(none)"),
        }
        for i, s in enumerate(successors)
    ]

    state: dict[str, Any] = {
        "system": "state machine whose states are assignments to boolean properties",
        "properties": names,
        "note": "a property not listed as true in a state is false in that state",
        "invariants": {
            f"group_{b + 1}": {
                "properties": names[b * _BLOCK_SIZE:(b + 1) * _BLOCK_SIZE],
                "legal_combinations": [
                    [names[b * _BLOCK_SIZE + j] for j in range(_BLOCK_SIZE) if (p >> j) & 1] or ["(none true)"]
                    for p in legal[b]
                ],
            }
            for b in range(n_blocks)
        },
        "current_state": {"true_properties": _true_props(current, names)},
        "move_rule": (
            "exactly one group changes to a different legal combination for that "
            "group; every property outside that group keeps its current value"
        ),
        "goal_condition": [
            f"{names[p]} is {'true' if v else 'false'}" for p, v in goal
        ],
    }
    questions = {
        "successor": shuffled_options(
            choice(
                "The options are every state reachable in one move. Exactly one of "
                "them satisfies every clause of the goal condition. Which one is it?",
                options,
            ),
            rng,
        )
    }
    inst = Instance(
        generator=NAME,
        difficulty=dict(difficulty),
        seed=seed,
        index=index,
        state=state,
        questions=questions,
        truth={"successor": options[target_ix]["id"]},
        meta={
            "mode": "successors",
            # The four counts E5 item 7 exists to compare.
            "n_props": n_props,
            "n_assignments_total": 2 ** n_props,
            "n_invariant_satisfying": ppb ** n_blocks,
            "n_reachable_successors": len(successors),
            "n_blocks": n_blocks,
            "patterns_per_block": ppb,
            "goal_literals": len(goal),
            "distinguishing_literals": distinguishing,
            # How close the best wrong answer comes to satisfying the goal. A
            # runner-up one literal short is a much harder discrimination than
            # one that misses three.
            "runner_up_literals_matched": runner_up,
            "baseline_random": round(1 / len(successors), 6),
        },
    )
    inst.validate()
    return inst


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import itertools
    import json

    SEED = 4242
    N = 24

    def _check_accept(d: dict, batch: list[Instance]) -> dict:
        yes = 0
        for inst in batch:
            st = inst.state
            delta = {q: st["transitions"][q] for q in st["states"]}
            cur = st["start_state"]
            for ch in st["input_string"]:
                cur = delta[cur][ch]
            assert (cur in st["accepting_states"]) == inst.truth["accepts"], "simulation disagrees"
            assert len(st["input_string"]) == d["string_len"]
            assert set(st["input_string"]) <= set(st["alphabet"])
            assert 0 < len(st["accepting_states"]) < d["n_states"]
            yes += inst.truth["accepts"]
        assert yes * 2 == len(batch), f"accept/reject not balanced: {yes}/{len(batch)}"
        return {
            "accept_rate": yes / len(batch),
            "mean_density": sum(i.meta["accept_density"] for i in batch) / len(batch),
            "max_rerolls": max(i.meta["dfa_attempts"] for i in batch),
        }

    def _check_successors(d: dict, batch: list[Instance]) -> dict:
        for inst in batch:
            st = inst.state
            q = inst.questions["successor"]
            opts = {o["id"]: o["label"] for o in q["options"]}
            assert len(opts) == inst.meta["n_reachable_successors"]
            assert len(set(opts.values())) == len(opts), "duplicate successor assignments"
            goal = []
            for clause in st["goal_condition"]:
                prop, _, val = clause.rpartition(" is ")
                goal.append((prop, val == "true"))
            # Re-derive the answer from the rendered payload alone, not from the
            # tuples generate() worked with.
            winners = [
                oid for oid, label in opts.items()
                if all((p in _labelled(label)) == v for p, v in goal)
            ]
            assert winners == [inst.truth["successor"]], f"goal picks {winners}"
            cur = set(st["current_state"]["true_properties"])
            for label in opts.values():
                diff = _labelled(label) ^ cur
                groups = {
                    g for g, spec in st["invariants"].items()
                    if diff & set(spec["properties"])
                }
                assert len(groups) == 1, "successor changes other than one group"
            assert inst.meta["n_invariant_satisfying"] < inst.meta["n_assignments_total"]
            assert inst.meta["n_reachable_successors"] < inst.meta["n_invariant_satisfying"]
        return {
            "successors": batch[0].meta["n_reachable_successors"],
            "invariant_sat": batch[0].meta["n_invariant_satisfying"],
            "assignments": batch[0].meta["n_assignments_total"],
            "mean_runner_up": sum(i.meta["runner_up_literals_matched"] for i in batch) / len(batch),
        }

    def _labelled(label: str) -> set[str]:
        body = label.split("true: ", 1)[1]
        return set() if body == "(none)" else {p.strip() for p in body.split(",")}

    # The closed form for invariant-satisfying assignments is a product over
    # independent blocks. Verify the counting itself by brute force at a size
    # small enough to enumerate.
    small = {"mode": "successors", "n_props": 12, "patterns_per_block": 5}
    probe = generate(difficulty=small, seed=1, count=1)[0]
    legal_sets = []
    for spec in probe.state["invariants"].values():
        props = spec["properties"]
        legal_sets.append({
            frozenset(c) & set(props) for c in
            [set(x) for x in spec["legal_combinations"]]
        })
    brute = 0
    props_flat = probe.state["properties"]
    for bits in itertools.product([0, 1], repeat=len(props_flat)):
        assign = {p for p, b in zip(props_flat, bits) if b}
        ok = True
        for spec, legal_set in zip(probe.state["invariants"].values(), legal_sets):
            if frozenset(assign & set(spec["properties"])) not in legal_set:
                ok = False
                break
        brute += ok
    assert brute == probe.meta["n_invariant_satisfying"], (brute, probe.meta["n_invariant_satisfying"])

    rows = []
    for d in difficulty_sweep():
        batch = generate(difficulty=d, seed=SEED, count=N)
        assert len(batch) == N
        assert len({i.instance_id for i in batch}) == N
        again = generate(difficulty=d, seed=SEED, count=N)
        a = json.dumps([i.to_json() for i in batch], sort_keys=True)
        b = json.dumps([i.to_json() for i in again], sort_keys=True)
        assert a == b, f"not deterministic at {d}"
        half = generate(difficulty=d, seed=SEED, count=N // 2)
        assert [i.to_json() for i in half] == [i.to_json() for i in batch[: N // 2]], "count-dependent"
        extra = _check_accept(d, batch) if d["mode"] == "accept" else _check_successors(d, batch)
        rows.append((d, len(a), extra))

    print(f"{'difficulty':<58} {'bytes/inst':>10}  notes")
    for d, nbytes, extra in rows:
        desc = ", ".join(f"{k}={v}" for k, v in d.items())
        note = " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in extra.items())
        print(f"{desc:<58} {nbytes // N:>10}  {note}")
    print(f"\nbrute-force invariant count at n_props=12: {brute} (closed form agrees)")
    print("ok")
