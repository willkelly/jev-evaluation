"""Balanced bracket strings, for E9 (distribution edges).

The plan predicts counting degrades sharply past roughly twenty elements, so the
main control is string length and the sweep runs 12 to 128 -- dense through the
predicted knee and well past it -- with a separate nesting-depth ladder at a
fixed length so depth and length can be told apart.

Half of every batch is balanced. The unbalanced half is one minimal edit away
from a balanced string of the same difficulty, which is what keeps the task from
being solvable by counting characters. Three edits are used, in equal shares:

    swap    transpose two brackets, adjacent wherever that works
    flip    turn one bracket into its own partner, "(" into ")"
    delete  remove one bracket

They differ in what a counting baseline sees, and that difference is the point.
A swap changes neither the length nor any bracket count, so length and counts
say nothing and the string has to be matched. A flip leaves the length alone but
makes the open and close counts unequal. A delete makes the length odd, which is
by itself a proof of imbalance. So the cheap baseline named in the plan -- look
at the length -- scores 0.67 over a whole batch, a count-matching baseline 0.83,
and both score exactly 0.50 on the swap third. The edit is recorded in meta so
results can be split by it: a model that catches deletes and flips but not swaps
is counting rather than matching, and that is a specific claim the split can
support.

Those two are not the strongest cheap baseline, though, and quoting them alone
would overstate what the swap third proves. One running counter -- add one on an
opener, subtract one on a closer, reject if it ever goes negative or does not
end at zero -- decides a single-type string exactly, swaps included, because a
one-type alphabet has no way to be wrong that a counter cannot see. It therefore
scores 1.00 on the single-type half of every condition and about 0.84 on the
mixed half, so about 0.92 over a whole batch. That, not 0.50, is the number a
model has to beat here, and the mixed half is the only part of the batch that a
stack is needed for. The counter is recorded in meta["baseline"] alongside the
two weaker baselines and printed by the self-check rather than left out.

The three edits are used in exactly equal numbers, which is why the swap case
does not fall back to some other edit when an adjacent transposition happens not
to work. Falling back would have made flips more common in exactly the deeply
nested conditions where adjacent swaps fail, and the headline accuracy would
then have moved with the mix rather than with the difficulty.

Single-bracket and mixed-bracket alphabets alternate within each batch and are
recorded in meta rather than being separate difficulty levels, because the
comparison of interest is within a length, not across lengths. A swap in a mixed
alphabet usually produces a type mismatch rather than an ordering error, so the
mixed half is also where bracket-type tracking gets tested.

Ground truth comes from a stack: push on an opener, pop and compare on a closer,
and require an empty stack at the end. Every generated string is run through it,
including the ones the generator believes it built balanced, so a bug in the
construction shows up as a failed self-check rather than as a mislabelled batch.
Candidate edits are also checked before use, since not every edit unbalances a
string -- transposing the middle two characters of "((()))" gives "(()())",
which is still balanced. In a single-type string an adjacent transposition
unbalances it only where it happens at the top level, so a fully nested string
has none that works and the generator transposes a distant pair instead. The
distance between the two positions is recorded, because a distant transposition
is a different problem from an adjacent one.

Among the edits that do unbalance, one that leaves the peak nesting depth alone
is preferred. The balanced base hits `max_depth` exactly, so an edit that lowers
the peak would let "as deep as this condition allows" predict the label at about
0.70 -- an artifact of how the base is built, with nothing to say about whether
the brackets match. With the preference in place the peak is `max_depth` in both
classes and the feature is constant, which the self-check asserts.

Run the self-check with:

    python -m jeveval.generators.dyck
"""

from __future__ import annotations

import random

from ..instances import Instance, noul, rng_for

NAME = "dyck"

QUESTION_KEY = "balanced"

# Ordered, so bracket type i is the same everywhere. The single-type alphabet is
# the first entry; the mixed alphabet is all three.
_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"))
_OPENERS = {o for o, _ in _PAIRS}
_PARTNER = {o: c for o, c in _PAIRS} | {c: o for o, c in _PAIRS}

_EDITS = ("swap", "flip", "delete")

# Random distant transpositions tried before sweeping every pair instead. The
# sweep cannot come up empty: the first character opens and the last closes, so
# exchanging those two alone leaves a string that starts by closing nothing.
_DISTANT_SWAP_ATTEMPTS = 200

# The two-character windows that read as a matched pair, for the local baseline
# that settles the depth-1 condition on its own.
_MATCHED_WINDOWS = frozenset(opener + closer for opener, closer in _PAIRS)


def difficulty_sweep() -> list[dict]:
    """Length 12 to 128 at depth 4, plus a depth ladder at length 32.

    Lengths are dense from 12 to 32 because that is where the plan expects the
    knee, and reach 128 because a prediction of "degrades past 20" is only
    tested by showing the curve has flattened at the bottom. The depth ladder
    sits at a length in the middle of the range, so a fall-off can be attributed
    to depth rather than to length.

    Two ends of these ranges are left out because they admit too few strings to
    sample. Length 8 at depth 4 is fully nested, and depth 16 at length 32 is
    fully nested, and in each case there is exactly one single-type balanced
    string: a whole condition would be that string repeated. Depth 1 has the
    same property and is kept anyway, since a flat sequence of pairs is the
    pure-counting case the plan asks for and no generator can make it varied.
    It is a control condition and not a measurement: a balanced depth-1 string
    is a run of adjacent matched pairs, so a two-character window settles the
    whole level, and so does the counter scan: a type mismatch needs one pair
    nested inside another, which depth 1 has no room for. The self-check asserts
    both baselines at 1.00 there, against 0.50 and about 0.84 elsewhere. A model
    score at depth 1 is a floor and nothing more.
    """
    levels = [
        {"length": length, "max_depth": 4}
        for length in (12, 16, 20, 24, 32, 48, 64, 96, 128)
    ]
    levels += [{"length": 32, "max_depth": depth} for depth in (1, 2, 8, 12)]
    levels.sort(key=lambda d: (d["length"], d["max_depth"]))
    return levels


def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]:
    """`count` instances at one difficulty, alternating balanced and unbalanced.

    The class is `index % 2`, the alphabet is `(index // 2) % 2` and the edit is
    `_EDITS[(index // 2) % 3]`, all functions of the index alone, because the
    generator contract requires instance i to depend on i and not on `count`. The
    two cycles have coprime periods, so alphabet and edit are fully crossed
    rather than confounded: over any twelve consecutive indices each edit appears
    with each alphabet.
    """
    length, max_depth = _unpack(difficulty)
    out: list[Instance] = []
    for index in range(count):
        rng = rng_for(NAME, difficulty, seed, index)
        pair_index = index // 2
        out.append(
            _build_instance(
                difficulty=difficulty,
                seed=seed,
                index=index,
                rng=rng,
                length=length,
                max_depth=max_depth,
                balanced=(index % 2 == 0),
                n_types=3 if pair_index % 2 else 1,
                edit=_EDITS[pair_index % len(_EDITS)],
            )
        )
    return out


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def _unpack(difficulty: dict) -> tuple[int, int]:
    try:
        length = int(difficulty["length"])
        max_depth = int(difficulty["max_depth"])
    except KeyError as exc:
        raise ValueError(f"{NAME}: difficulty missing key {exc}") from exc
    if length < 2 or length % 2:
        raise ValueError(f"{NAME}: length must be even and >= 2, got {length}")
    if max_depth < 1 or max_depth > length // 2:
        raise ValueError(
            f"{NAME}: max_depth must be between 1 and {length // 2}, got {max_depth}"
        )
    return length, max_depth


def _random_balanced(
    rng: random.Random, length: int, max_depth: int, n_types: int
) -> str:
    """One balanced string of exactly `length` characters, nested exactly
    `max_depth` deep.

    Built as an ordered forest rather than as a left-to-right walk. A walk that
    is merely allowed to nest `max_depth` deep almost never does: at length 32
    and depth 16 the only string that reaches the cap is the fully nested one,
    so the depth ladder would compare strings that all sit at depth three or
    four. Here the forest starts as a chain of `max_depth` nodes, which fixes the
    depth exactly, and the remaining pairs attach at random positions under any
    node that still has room. Depth is then a real control rather than a ceiling
    the strings ignore.

    The result is not uniform over the balanced strings of that length and depth,
    and does not need to be. What it needs is to vary in shape and to hit the
    depth, and both classes are drawn from it, so nothing it does asymmetrically
    can separate them.
    """
    pairs = length // 2
    children: list[list[int]] = []
    depth: list[int] = []
    roots: list[int] = []

    def new_node(at_depth: int) -> int:
        children.append([])
        depth.append(at_depth)
        return len(children) - 1

    parent: int | None = None
    for level in range(1, max_depth + 1):
        node = new_node(level)
        (roots if parent is None else children[parent]).append(node)
        parent = node

    for _ in range(pairs - max_depth):
        # -1 stands for "start a new top-level group".
        options = [i for i, d in enumerate(depth) if d < max_depth] + [-1]
        chosen = rng.choice(options)
        node = new_node(1 if chosen == -1 else depth[chosen] + 1)
        siblings = roots if chosen == -1 else children[chosen]
        siblings.insert(rng.randrange(len(siblings) + 1), node)

    out: list[str] = []

    def emit(node: int) -> None:
        opener, closer = _PAIRS[rng.randrange(n_types)]
        out.append(opener)
        for child in children[node]:
            emit(child)
        out.append(closer)

    for root in roots:
        emit(root)
    return "".join(out)


def _edit_candidates(text: str, edit: str) -> list[tuple[int, str]]:
    """Every single edit of the given kind, as (position, result)."""
    if edit == "swap":
        return [
            (i, text[:i] + text[i + 1] + text[i] + text[i + 2 :])
            for i in range(len(text) - 1)
            if text[i] != text[i + 1]
        ]
    if edit == "flip":
        return [
            (i, text[:i] + _PARTNER[text[i]] + text[i + 1 :]) for i in range(len(text))
        ]
    if edit == "delete":
        return [(i, text[:i] + text[i + 1 :]) for i in range(len(text))]
    raise ValueError(f"{NAME}: unknown edit {edit!r}")


def _transpose(text: str, i: int, j: int) -> str:
    return text[:i] + text[j] + text[i + 1 : j] + text[i] + text[j + 1 :]


def _unbalance(
    rng: random.Random, text: str, edit: str
) -> tuple[str, int, int | None]:
    """Apply one edit of the given kind that actually unbalances `text`.

    Returns the edited string, the position edited, and for a swap the distance
    between the transposed characters. Flips and deletes always unbalance -- one
    changes the bracket counts and the other makes the length odd -- so only the
    swap needs to search. Every candidate is run through the stack before it can
    be chosen, including the ones reached by the exhaustive sweep, so no edit is
    ever returned on the strength of an argument about it.
    """
    peak = _depth(text)

    def pick(candidates: list[tuple]) -> tuple | None:
        """One unbalancing candidate, preferring those that keep the peak depth.

        Each candidate is a tuple whose last element is the edited string.
        Preferring equal peaks is what keeps peak depth from predicting the
        label: the base always sits at exactly `max_depth`, so an edit that
        lowers the peak would mark the unbalanced half.
        """
        broken = [c for c in candidates if not _check(c[-1])[0]]
        if not broken:
            return None
        return rng.choice([c for c in broken if _depth(c[-1]) == peak] or broken)

    chosen = pick(_edit_candidates(text, edit))
    if chosen is not None:
        position, result = chosen
        return result, position, (1 if edit == "swap" else None)
    if edit != "swap":
        raise AssertionError(f"{NAME}: no {edit} unbalances {text!r}")

    # A fully nested string has no adjacent transposition that unbalances it, so
    # look further afield. Random pairs first; every pair when those turn up
    # nothing, or nothing that holds the peak. The sweep cannot come up empty --
    # exchanging the first character with the last leaves a string that opens
    # with a closing bracket -- so the assertion below is a check on this
    # reasoning rather than a case that has to be handled.
    sampled = sorted(
        {
            pair
            for pair in (
                tuple(sorted(rng.sample(range(len(text)), 2)))
                for _ in range(_DISTANT_SWAP_ATTEMPTS)
            )
            if text[pair[0]] != text[pair[1]]
        }
    )
    chosen = pick([(i, j, _transpose(text, i, j)) for i, j in sampled])
    if chosen is None or _depth(chosen[-1]) != peak:
        chosen = (
            pick(
                [
                    (i, j, _transpose(text, i, j))
                    for i in range(len(text))
                    for j in range(i + 1, len(text))
                    if text[i] != text[j]
                ]
            )
            or chosen
        )
    if chosen is None:
        raise AssertionError(f"{NAME}: no transposition unbalances {text!r}")
    i, j, result = chosen
    return result, i, j - i


def _build_instance(
    *,
    difficulty: dict,
    seed: int,
    index: int,
    rng: random.Random,
    length: int,
    max_depth: int,
    balanced: bool,
    n_types: int,
    edit: str,
) -> Instance:
    base = _random_balanced(rng, length, max_depth, n_types)
    if not _check(base)[0]:
        raise AssertionError(f"{NAME}: built an unbalanced 'balanced' string {base!r}")

    if balanced:
        text, edit_used, edit_pos, edit_span = base, None, None, None
    else:
        text, edit_pos, edit_span = _unbalance(rng, base, edit)
        edit_used = edit

    ok, violation_at, violation_kind = _check(text)
    if ok is not balanced:
        raise AssertionError(f"{NAME}: label {balanced} disagrees with the stack")

    opens = sum(1 for ch in text if ch in _OPENERS)
    per_type_match = all(
        text.count(opener) == text.count(closer) for opener, closer in _PAIRS
    )
    peak, trough, final_depth = _scan(text)
    windows = sum(
        1 for i in range(len(text) - 1) if text[i : i + 2] in _MATCHED_WINDOWS
    )
    inst = Instance(
        generator=NAME,
        difficulty=dict(difficulty),
        seed=seed,
        index=index,
        state=text,
        questions={
            QUESTION_KEY: noul(
                "Is the bracket string in the state balanced? Every opening "
                "bracket must be closed by a matching closing bracket of the "
                "same type, in the correct order, with none left over."
            )
        },
        truth={QUESTION_KEY: bool(balanced)},
        meta={
            "length": len(text),
            "base_length": length,
            "max_depth_allowed": max_depth,
            "base_depth": _depth(base),
            "n_bracket_types": n_types,
            "mixed": n_types > 1,
            "perturbation": edit_used,
            "perturbation_pos": edit_pos,
            # 1 for an adjacent swap, larger when the string was too nested for
            # one to work, None for flips and deletes.
            "perturbation_span": edit_span,
            # Ground truth, for grouping only. A prefix scan finds the violation
            # at this index; a late or absent index means the whole string has to
            # be read, so this predicts which errors a partial read catches.
            "first_violation_index": violation_at,
            "violation_kind": violation_kind,
            "baseline": {
                "length": len(text),
                "length_is_even": len(text) % 2 == 0,
                "n_open": opens,
                "n_close": len(text) - opens,
                "counts_match": opens * 2 == len(text),
                "type_counts_match": per_type_match,
                # The one-counter scan, which is the strongest cheap baseline
                # here: exact on the single-type half of every condition and
                # about 0.84 on the mixed half. A model score has to be read
                # against this, not against the 0.50 majority class.
                "peak_depth": peak,
                "min_prefix_depth": trough,
                "scan_ok": trough >= 0 and final_depth == 0,
                # A two-character window. It settles nothing in general and
                # settles the depth-1 condition outright, which is the reason
                # depth 1 is a control rather than a measurement.
                "adjacent_matched_pairs": windows,
                "windows_tile": windows * 2 == len(text),
            },
        },
    )
    inst.validate()
    return inst


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def _check(text: str) -> tuple[bool, int | None, str | None]:
    """Stack check. Returns (balanced, first violation index, violation kind).

    The index is None when nothing is wrong at any single position and the only
    fault is brackets left open at the end, which is the case a reader scanning
    left to right has no local cue for.
    """
    stack: list[str] = []
    for i, ch in enumerate(text):
        if ch in _OPENERS:
            stack.append(ch)
        elif not stack:
            return False, i, "unmatched_close"
        elif _PARTNER[ch] != stack.pop():
            return False, i, "type_mismatch"
    if stack:
        return False, None, "unclosed_at_end"
    return True, None, None


def _scan(text: str) -> tuple[int, int, int]:
    """One running counter over the string: (peak, trough, final depth).

    This is the cheapest thing that is not just counting characters, and on a
    single-type alphabet it is not a heuristic but a complete decision
    procedure: such a string is balanced exactly when the trough is 0 or above
    and the final depth is 0. With more than one bracket type it is necessary
    and not sufficient, which is the whole of what a stack buys here.
    """
    depth = peak = trough = 0
    for ch in text:
        depth += 1 if ch in _OPENERS else -1
        peak = max(peak, depth)
        trough = min(trough, depth)
    return peak, trough, depth


def _depth(text: str) -> int:
    """Peak nesting depth reached by a left-to-right scan."""
    return _scan(text)[0]


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------


def _counts_say_balanced(inst: Instance) -> bool:
    """The count-matching baseline: equal opens and closes of every type."""
    base = inst.meta["baseline"]
    return base["counts_match"] and base["type_counts_match"]


def _scan_says_balanced(inst: Instance) -> bool:
    """The one-counter baseline: never negative, ends at zero."""
    return inst.meta["baseline"]["scan_ok"]


def _windows_say_balanced(inst: Instance) -> bool:
    """The two-character-window baseline: the string is a run of matched pairs."""
    return inst.meta["baseline"]["windows_tile"]


def _self_check() -> None:
    count = 120  # divisible by 12, so alphabet and edit are exactly crossed
    seed = 4242
    rows = []
    for difficulty in difficulty_sweep():
        length, max_depth = _unpack(difficulty)
        batch = generate(difficulty=difficulty, seed=seed, count=count)
        again = generate(difficulty=difficulty, seed=seed, count=count)
        assert [i.to_json() for i in batch] == [
            i.to_json() for i in again
        ], f"nondeterministic at {difficulty}"

        labels = [i.truth[QUESTION_KEY] for i in batch]
        assert sum(labels) == count // 2, f"class imbalance at {difficulty}"
        assert sum(i.meta["mixed"] for i in batch) == count // 2, (
            f"alphabet imbalance at {difficulty}"
        )

        edits: dict[str, int] = {}
        for inst in batch:
            text = inst.state
            ok, _, _ = _check(text)
            assert ok is inst.truth[QUESTION_KEY], "stack disagrees with the label"
            assert len(text) == inst.meta["length"]
            types = _PAIRS[: inst.meta["n_bracket_types"]]
            alphabet = {ch for pair in types for ch in pair}
            assert set(text) <= alphabet, "character outside the declared alphabet"
            # Held for both classes, which is the point: the base is built to
            # sit at exactly max_depth and the edit is chosen to keep it there,
            # so peak depth carries nothing about the label.
            assert inst.meta["baseline"]["peak_depth"] == max_depth, (
                f"peak depth {inst.meta['baseline']['peak_depth']} != {max_depth}, "
                f"so depth separates the classes at {difficulty}"
            )
            if inst.truth[QUESTION_KEY]:
                assert len(text) == length, "balanced string has the wrong length"
                assert inst.meta["perturbation"] is None
            else:
                edits[inst.meta["perturbation"]] = (
                    edits.get(inst.meta["perturbation"], 0) + 1
                )
                deleted = inst.meta["perturbation"] == "delete"
                expected = length - 1 if deleted else length
                assert len(text) == expected, "edited string has the wrong length"

        # The depth ladder only means something if the strings reach the depth.
        reached = sum(i.meta["base_depth"] == max_depth for i in batch) / count
        assert reached > 0.9, f"only {reached:.2f} of strings reach depth {max_depth}"

        # A condition made of a handful of repeated strings has an effective
        # sample size of a handful, whatever its nominal size says.
        distinct = len({i.state for i in batch}) / count
        assert distinct > 0.5, f"only {distinct:.2f} distinct states at {difficulty}"

        length_acc = sum(
            i.meta["baseline"]["length_is_even"] == i.truth[QUESTION_KEY] for i in batch
        ) / count
        count_acc = sum(
            _counts_say_balanced(i) == i.truth[QUESTION_KEY] for i in batch
        ) / count
        assert edits == {kind: count // 6 for kind in _EDITS}, (
            f"uneven edit shares {edits} at {difficulty}"
        )
        assert abs(length_acc - 2 / 3) < 1e-9, "the length baseline moved off 0.667"
        assert abs(count_acc - 5 / 6) < 1e-9, "the count baseline moved off 0.833"

        # The counter scan is exact on one bracket type -- a theorem about Dyck
        # words, so a failure here means the alphabet bookkeeping is broken, not
        # that the number drifted. On the mixed half it has to be beatable,
        # otherwise the condition never asks for a stack at all.
        single = [i for i in batch if not i.meta["mixed"]]
        mixed = [i for i in batch if i.meta["mixed"]]
        scan_single = sum(
            _scan_says_balanced(i) == i.truth[QUESTION_KEY] for i in single
        ) / len(single)
        scan_mixed = sum(
            _scan_says_balanced(i) == i.truth[QUESTION_KEY] for i in mixed
        ) / len(mixed)
        assert scan_single == 1.0, (
            f"the counter scan is {scan_single:.3f} on one bracket type at "
            f"{difficulty}; it is exact there by construction"
        )
        if max_depth == 1:
            # Flat strings have nowhere to hide a type mismatch: getting one
            # needs a pair nested inside another, which depth 1 forbids. So
            # every cheap baseline is exact here, and that is the whole reason
            # this level is carried as a control rather than as a measurement.
            assert scan_mixed == 1.0, (
                f"the counter scan is {scan_mixed:.3f} on the flat condition; it "
                "is exact there, so something has changed about depth 1"
            )
        else:
            assert scan_mixed < 0.95, (
                f"the counter scan reaches {scan_mixed:.3f} on the mixed half at "
                f"{difficulty}, leaving a stack almost nothing to add"
            )
        scan_acc = (scan_single * len(single) + scan_mixed * len(mixed)) / count

        # The local window decides the flat condition and nothing else. Stating
        # it as an assertion keeps depth 1 marked as a control: if this ever
        # reads 1.00 at a deeper level, that level has gone degenerate too.
        window_acc = sum(
            _windows_say_balanced(i) == i.truth[QUESTION_KEY] for i in batch
        ) / count
        if max_depth == 1:
            assert window_acc == 1.0, (
                f"depth 1 should be settled outright by a window, got {window_acc:.2f}"
            )
        else:
            assert window_acc == 0.5, (
                f"the window rule scores {window_acc:.2f} at {difficulty}; above "
                "depth 1 no balanced string tiles into adjacent pairs, so the "
                "rule should call everything unbalanced and land on 0.50"
            )

        swaps = [i for i in batch if i.meta["perturbation"] == "swap"]
        swap_caught = sum(not _counts_say_balanced(i) for i in swaps)
        assert swap_caught == 0, "the count baseline should never catch a swap"
        adjacent = sum(i.meta["perturbation_span"] == 1 for i in swaps) / len(swaps)
        early = [
            i.meta["first_violation_index"] / i.meta["length"]
            for i in batch
            if i.meta["first_violation_index"] is not None
        ]
        rows.append(
            (
                f"len={length} depth={max_depth}",
                length_acc,
                count_acc,
                scan_acc,
                scan_mixed,
                window_acc,
                sum(early) / len(early) if early else float("nan"),
                len(early) / (count // 2),
                adjacent,
                distinct,
            )
        )

    header = (
        f"{'difficulty':20} {'len':>5} {'cnt':>5} {'scan':>5} {'scan-mx':>7} "
        f"{'win':>5} {'viol@':>6} {'at-pos':>6} {'adj':>5} {'uniq':>5}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        name, length_acc, count_acc, scan, scan_mx, win, viol, at_pos, adj, uniq = row
        print(
            f"{name:20} {length_acc:5.2f} {count_acc:5.2f} {scan:5.2f} "
            f"{scan_mx:7.2f} {win:5.2f} {viol:6.2f} {at_pos:6.2f} "
            f"{adj:5.2f} {uniq:5.2f}"
        )
    print(
        "\nlen     : accuracy of 'balanced iff the length is even'."
        "\ncnt     : accuracy of 'balanced iff every bracket type's counts match'."
        "\nscan    : accuracy of the one-counter scan -- never negative, ends at"
        "\n          zero. THE BASELINE TO BEAT. Exact on the single-type half, so"
        "\n          it cannot go below about 0.9 whatever the length."
        "\nscan-mx : the same scan on the mixed half alone, which is the only part"
        "\n          of the batch a stack is needed for. Read model scores here."
        "\nwin     : accuracy of 'balanced iff the string is a run of adjacent"
        "\n          matched pairs'. 1.00 at depth 1, which is why that level is a"
        "\n          control condition and not a measurement; 0.50 everywhere else."
        "\n          The raw count is in meta[\"baseline\"] too, and a threshold"
        "\n          fitted on it reaches about 0.7 at the shortest lengths."
        "\nviol@   : mean position of the first violation, as a fraction of the string."
        "\nat-pos  : fraction of unbalanced strings whose fault is visible at some"
        "\n          single position rather than only from brackets left open."
        "\nadj     : fraction of swaps that were adjacent; the rest were too deeply"
        "\n          nested for an adjacent swap to unbalance them."
        "\nuniq    : fraction of the batch that is a distinct string. Short or very"
        "\n          deep conditions have few strings to draw from."
        f"\n{len(rows)} difficulty levels, {count} instances each, majority class 0.50."
        "\nEvery level splits into equal thirds of swap, flip and delete, so len and"
        "\ncnt are fixed at 0.667 and 0.833 by the design of the batch. Peak depth is"
        "\nheld at max_depth in both classes, so it is not a baseline at all."
    )


if __name__ == "__main__":
    _self_check()
