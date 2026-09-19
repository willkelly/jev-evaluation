"""Uniform random 3SAT, the instance source for E2.

E2 uses 3SAT because the random 3SAT phase transition hands you a target
probability curve for free: at clause-to-variable ratio well below the
threshold almost every formula is satisfiable, well above it almost none is,
and the true satisfiable fraction per ratio is a curve the model's mean
predicted P(SAT) can be plotted against. Nothing here is labelled by hand and
nothing is labelled by a heuristic. Ground truth is a Minisat22 decision on the
exact clause list that was serialised into the state.

Two things about that setup need care, and they are what most of this module
is for.

**An inverted or mis-wired solver is the dangerous failure.** It would not
crash; it would silently flip the target curve, and every downstream number --
ECE, AUROC, the crossover ratio -- would be computed against the wrong answer
and look merely surprising rather than wrong. So the satisfying assignment is
kept in `meta` for every SAT instance, and the self-check at the bottom
verifies it clause by clause, cross-checks every decision against a second
solver (Glucose3), brute-forces every n=10 instance over all 2**10 assignments
without using a solver at all, and asserts that the satisfiable fraction falls
from 1.0 at ratio 2.0 to near 0 at ratio 8.0. An inverted solver fails all four.

**Clause density alone predicts satisfiability well**, so accuracy on a plain
sweep is not evidence of reasoning. `generate_matched` is the control: at a
fixed ratio it returns equal numbers of SAT and UNSAT instances by rejection
sampling, so the majority-class baseline is exactly 0.5 and the density
heuristic -- which sees only n and m, identical across the whole matched set --
is also exactly 0.5. Anything above that came from somewhere else.

Rejection sampling only works where the rarer class is not vanishingly rare.
Measured frequencies of the rarer class, 40k/20k/6k draws per cell:

        n=10                    n=20                    n=50
    ratio  p(rare)          ratio  p(rare)          ratio  p(rare)
     2.75   0.0017           3.25   0.0047           3.50   0.0018
     3.00   0.0050           3.50   0.0208           3.75   0.0288
     3.25   0.0130           3.75   0.0683           4.00   0.1280
     3.50   0.0396           6.50   0.0143           5.25   0.0183
     7.50   0.0362           6.75   0.0083           5.50   0.0053
     7.75   0.0246           7.00   0.0050           5.75   0.0013
     8.00   0.0197           7.25   0.0015           6.00   0.0005

With the default 4000-draw-per-slot budget a slot fills with probability
1 - (1-p)**4000, so p >= 0.005 fills reliably and p >= 0.01 fills cheaply. That
gives the attainable ranges in `_MATCHED_RANGE` below, which `matched_sweep()`
returns: roughly 3.25-8.00 at n=10, 3.50-6.75 at n=20, 3.75-5.25 at n=50. The
plan asks for a matched control "at each ratio"; at the ends of the sweep that
is not possible, and `generate_matched` raises rather than quietly returning an
unbalanced set whose baseline is no longer 0.5.

**Two encodings, one formula.** `encoding` selects DIMACS text or the same
formula as a JSON object, for E4's structured-versus-stringified comparison.
The formula is drawn from the difficulty dict *with `encoding` removed*, so
instance `i` is the same formula either way; the encoding is part of the
instance's difficulty, so the two get different `instance_id`s and stay separate
rows in the log; and `meta["formula_id"]` pairs them for the paired test. The
question names the format the state is actually in and is otherwise identical
between them.

One measurement worth carrying into the report: the empirical crossover at
these sizes is not 4.26. At n=20 the satisfiable fraction passes 0.5 near ratio
4.55, at n=50 near 4.40, at n=10 near 5.00. 4.26 is the asymptotic threshold
and finite instances approach it from above. E2's "crossover within 0.5 of
4.26" tier criterion should be scored against the true curve measured from
these instances, not against the constant.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from pysat.solvers import Minisat22

from ..instances import Instance, noul, rng_for

NAME = "sat3"

# The asymptotic random-3SAT satisfiability threshold. Used only to define the
# cheap baseline; nothing about generation depends on it.
DENSITY_THRESHOLD = 4.26

QUESTION_KEY = "satisfiable"

# The question is one sentence naming the format the state is actually in,
# followed by a sentence that is word-for-word identical across encodings. Only
# the first sentence may differ, because E4 compares the two encodings against
# each other: if the question itself were also reworded, the comparison would
# be measuring the rewording as much as the encoding.
_STATE_SENTENCE = {
    "dimacs": (
        "The state is a Boolean formula in conjunctive normal form, written in "
        "DIMACS CNF format."
    ),
    "clauses": (
        "The state is a Boolean formula in conjunctive normal form, given as a "
        "JSON object whose `clauses` field lists the clauses, each one a list "
        "of signed integers naming its literals."
    ),
}
_QUESTION_SENTENCE = (
    "Is the formula satisfiable -- is there an assignment of true or false to "
    "each variable under which every clause contains at least one true literal?"
)

# The default encoding's question, kept as a module constant because E2 and the
# report quote it.
QUESTION_TEXT = f"{_STATE_SENTENCE['dimacs']} {_QUESTION_SENTENCE}"


def question_text(encoding: str) -> str:
    """The question for one encoding.

    Saying "written in DIMACS CNF format" over a JSON object would be a plain
    falsehood in the prompt, and the model's failure to reconcile it would be
    scored as a reasoning failure.
    """
    _check_encoding(encoding)
    return f"{_STATE_SENTENCE[encoding]} {_QUESTION_SENTENCE}"

# Attempt indices for a matched slot are laid out as slot * _ATTEMPT_STRIDE +
# attempt. The stride is a constant rather than the attempt budget so that
# raising the budget to fill a hard ratio extends each slot's draw sequence
# instead of reshuffling every instance that was already drawn.
_ATTEMPT_STRIDE = 1_000_000
MAX_ATTEMPTS_PER_SLOT = 4000

# Ratio ranges over which generate_matched can fill both halves at the default
# budget. Measured, not derived; see the module docstring.
_MATCHED_RANGE: dict[int, tuple[float, float]] = {
    10: (3.25, 8.00),
    20: (3.50, 6.75),
    50: (3.75, 5.25),
}


class MatchedSetUnfillable(RuntimeError):
    """Raised when one half of a density-matched set cannot be sampled.

    Carries the counts actually reached so the caller can report the gap rather
    than guess at it.
    """

    def __init__(self, message: str, *, n: int, ratio: float, n_sat: int, n_unsat: int, wanted: int):
        super().__init__(message)
        self.n = n
        self.ratio = ratio
        self.n_sat = n_sat
        self.n_unsat = n_unsat
        self.wanted = wanted


# --------------------------------------------------------------------------
# Formula construction, solving and encoding
# --------------------------------------------------------------------------


def clause_count(n: int, ratio: float) -> int:
    """Number of clauses at `n` variables and clause-to-variable `ratio`.

    `round` is half-to-even, so at n=10 and n=50 some quarter-step ratios land
    on a half-integer and the realised density m/n differs slightly from the
    nominal ratio. Both are recorded in meta; at n=20 they always agree.
    """
    return round(ratio * n)


def random_3sat(n: int, m: int, rng: random.Random) -> list[list[int]]:
    """`m` clauses of 3 distinct variables each, independent random polarity.

    This is the standard uniform random 3SAT model the 4.26 threshold refers
    to: clauses are drawn independently and with replacement, so a formula may
    contain the same clause twice. Requiring the three variables within a
    clause to be distinct rules out tautologies and repeated literals.
    """
    if n < 3:
        raise ValueError(f"3SAT needs at least 3 variables, got n={n}")
    if m < 0:
        raise ValueError(f"clause count must be non-negative, got m={m}")
    variables = range(1, n + 1)
    clauses = []
    for _ in range(m):
        chosen = rng.sample(variables, 3)
        clauses.append([v if rng.random() < 0.5 else -v for v in chosen])
    return clauses


def solve(clauses: list[list[int]], n: int) -> list[int] | None:
    """A satisfying assignment as `n` signed literals, or None if unsatisfiable.

    The returned list always covers variables 1..n in order, which the solver's
    own model does not: a variable that appears in no clause is absent from it.
    Such a variable is free; it is reported false so that the list is a total
    assignment `satisfies` can be checked against.
    """
    with Minisat22(bootstrap_with=clauses) as solver:
        if not solver.solve():
            return None
        model = solver.get_model() or []
    positive = {abs(lit) for lit in model if lit > 0}
    return [v if v in positive else -v for v in range(1, n + 1)]


def satisfies(clauses: list[list[int]], assignment: list[int]) -> bool:
    """Does `assignment` (a list of signed literals) satisfy every clause?"""
    true_literals = set(assignment)
    return all(any(lit in true_literals for lit in clause) for clause in clauses)


def to_dimacs(clauses: list[list[int]], n: int) -> str:
    """Canonical DIMACS CNF text. E4 and E9 reuse this encoding.

    No comment lines: a `c` header would be the easiest possible place to leak
    the answer into the state.

    The variable-range check is here rather than in the caller because E4 and
    E9 reuse this on clause lists this module did not build, and a header that
    understates `n` produces a file that is still parseable and no longer means
    what it says.
    """
    for clause in clauses:
        for lit in clause:
            if not 1 <= abs(lit) <= n:
                raise ValueError(f"literal {lit} is outside 1..{n}")
    lines = [f"p cnf {n} {len(clauses)}"]
    lines.extend(" ".join(str(lit) for lit in clause) + " 0" for clause in clauses)
    return "\n".join(lines) + "\n"


ENCODINGS = ("dimacs", "clauses")


def _check_encoding(encoding: str) -> None:
    # Checked before the loop, so `count=0` with a misspelled encoding fails
    # like `count=500` does instead of returning an empty list.
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")


def _resolve_encoding(difficulty: dict, encoding: str | None) -> str:
    """The encoding, from the keyword argument or from the difficulty dict.

    Both spellings are accepted because they serve different callers. E2 sweeps
    ratio at one encoding and passes the keyword; E4 needs `encoding` to be part
    of the difficulty so a paired sweep can be written as a list of difficulty
    dicts, the way `progreach.difficulty_sweep` does it. They may not disagree.
    """
    from_difficulty = difficulty.get("encoding")
    if encoding is not None and from_difficulty is not None and encoding != from_difficulty:
        raise ValueError(
            f"encoding={encoding!r} contradicts difficulty['encoding']={from_difficulty!r}"
        )
    resolved = encoding if encoding is not None else (from_difficulty or "dimacs")
    _check_encoding(resolved)
    return resolved


def _formula_key(difficulty: dict) -> dict:
    """The difficulty dict without `encoding`.

    The formula is drawn from an RNG seeded with this, so instance `i` is
    literally the same formula in both encodings -- which is what makes E4's
    paired comparison a paired comparison rather than two independent samples.
    """
    return {k: v for k, v in difficulty.items() if k != "encoding"}


def _formula_id(key: dict, seed: int, index: int) -> str:
    """Join key for the two encodings of one formula.

    `Instance.instance_id` deliberately separates them -- they are different
    states and must be different rows in the log -- so the analysis needs
    something else to pair them on. Mirrors `progreach.meta["program_id"]`.
    """
    payload = json.dumps({"g": NAME, "d": key, "s": seed, "i": index}, sort_keys=True)
    return f"cnf-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _encode_state(clauses: list[list[int]], n: int, encoding: str) -> Any:
    if encoding == "dimacs":
        return to_dimacs(clauses, n)
    if encoding == "clauses":
        # Structured equivalent of the same formula, for E4's structured-versus-
        # stringified comparison.
        return {"num_vars": n, "num_clauses": len(clauses), "clauses": [list(c) for c in clauses]}
    raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")


# --------------------------------------------------------------------------
# Difficulty
# --------------------------------------------------------------------------


def _ratio_ladder() -> list[float]:
    """2.0 to 8.0 in steps of 0.25.

    Built from quarters so the values are exact binary fractions: the
    difficulty dict is JSON-serialised inside `rng_for`, and a ratio whose repr
    drifted would silently change which instances a condition draws.
    """
    return [k / 4 for k in range(8, 33)]


def difficulty_sweep() -> list[dict]:
    """The full E2 ladder: ratio 2.0-8.0 at n=20, then the same at n=10 and n=50.

    Ordered by size then ratio rather than by difficulty, because difficulty is
    not monotone here. Instances are easy at both ends of the ratio range and
    hardest in the middle, near the transition, and the plotted curve is the
    point -- E2 reads the shape across the whole range, not a single hard cell.
    n=20 comes first because it is the primary sweep; n=10 and n=50 exist to
    separate instance size from ratio.
    """
    return [{"n": n, "ratio": r} for n in (20, 10, 50) for r in _ratio_ladder()]


def matched_sweep() -> list[dict]:
    """The subset of `difficulty_sweep()` where `generate_matched` can fill both halves.

    Filtering here rather than letting the caller discover it by exception means
    an E2 run does not spend a minute of solver time per ratio finding out that
    ratio 8.0 has no satisfiable instances left to find.
    """
    out = []
    for difficulty in difficulty_sweep():
        low, high = _MATCHED_RANGE[difficulty["n"]]
        if low <= difficulty["ratio"] <= high:
            out.append(difficulty)
    return out


def _unpack(difficulty: dict) -> tuple[int, float]:
    try:
        n = int(difficulty["n"])
        ratio = float(difficulty["ratio"])
    except KeyError as exc:
        raise ValueError(f"sat3 difficulty needs 'n' and 'ratio', got {difficulty!r}") from exc
    if n < 3:
        raise ValueError(f"sat3 needs n >= 3, got {n}")
    if ratio < 0:
        raise ValueError(f"sat3 needs ratio >= 0, got {ratio}")
    return n, ratio


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _build(
    *,
    difficulty: dict,
    key: dict,
    seed: int,
    index: int,
    clauses: list[list[int]],
    n: int,
    assignment: list[int] | None,
    encoding: str,
    extra_meta: dict | None = None,
) -> Instance:
    """One Instance. `difficulty` is what the instance carries and is identified
    by; `key` is the same dict without `encoding`, which is what the formula was
    drawn from and what pairs the encodings together."""
    m = len(clauses)
    satisfiable = assignment is not None
    meta = {
        "n": n,
        "m": m,
        "ratio": float(difficulty["ratio"]),
        "realized_ratio": m / n,
        "satisfiable": satisfiable,
        "assignment": assignment,
        # Deliberately redundant with the DIMACS state. Keeping the clause list
        # the solver actually decided lets an offline check confirm that the
        # state sent to the model encodes that same formula, without anyone
        # having to write a DIMACS parser to do it.
        "clauses": [list(c) for c in clauses],
        "solver": "minisat22",
        "encoding": encoding,
        # Equal across the encodings of one formula, where instance_id is not.
        "formula_id": _formula_id(key, seed, index),
        # Always present, on plain and matched instances alike, so the offline
        # analysis can split the two conditions with one key.
        "matched": False,
        # The cheap baseline for E2: parse the DIMACS header, compute m/n,
        # predict SAT below the threshold. Computed from the realised density
        # because that is all a header-reading baseline can see; on the
        # standard ladder it agrees with the nominal ratio in every cell.
        "baseline_density_threshold": DENSITY_THRESHOLD,
        "baseline_density_pred": (m / n) < DENSITY_THRESHOLD,
    }
    if extra_meta:
        meta.update(extra_meta)
    instance = Instance(
        generator=NAME,
        difficulty=dict(difficulty),
        seed=seed,
        index=index,
        state=_encode_state(clauses, n, encoding),
        questions={QUESTION_KEY: noul(question_text(encoding))},
        truth={QUESTION_KEY: satisfiable},
        meta=meta,
    )
    instance.validate()
    return instance


def generate(
    *, difficulty: dict, seed: int, count: int, encoding: str | None = None
) -> list[Instance]:
    """`count` uniform random 3SAT instances at one (n, ratio) setting.

    Instance `i` draws from `rng_for(NAME, key, seed, i)` alone, where `key` is
    the difficulty without `encoding`, so a shorter run is a prefix of a longer
    one, adding a ratio to the ladder does not disturb any other ratio, and the
    two encodings of instance `i` are the same formula.

    The encoding defaults to `"dimacs"` and may be given either here or as
    `difficulty["encoding"]`. It is recorded in the instance's difficulty, so
    the two encodings of one formula get different `instance_id`s -- they are
    different states and must be separable in the log -- and `meta["formula_id"]`
    pairs them back up.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    encoding = _resolve_encoding(difficulty, encoding)
    n, ratio = _unpack(difficulty)
    m = clause_count(n, ratio)
    key = _formula_key(difficulty)
    stamped = {**key, "encoding": encoding}
    out = []
    for i in range(count):
        rng = rng_for(NAME, key, seed, i)
        clauses = random_3sat(n, m, rng)
        out.append(
            _build(
                difficulty=stamped,
                key=key,
                seed=seed,
                index=i,
                clauses=clauses,
                n=n,
                assignment=solve(clauses, n),
                encoding=encoding,
            )
        )
    return out


def _fill_slot(
    *,
    difficulty: dict,
    key: dict,
    seed: int,
    slot: int,
    want_sat: bool,
    n: int,
    m: int,
    budget: int,
    encoding: str,
) -> tuple[Instance | None, int]:
    """Draw until one instance of the requested class appears, or give up."""
    for attempt in range(budget):
        rng = rng_for(NAME, key, seed, slot * _ATTEMPT_STRIDE + attempt)
        clauses = random_3sat(n, m, rng)
        assignment = solve(clauses, n)
        if (assignment is not None) is want_sat:
            return (
                _build(
                    difficulty=difficulty,
                    key=key,
                    seed=seed,
                    index=slot,
                    clauses=clauses,
                    n=n,
                    assignment=assignment,
                    encoding=encoding,
                    extra_meta={
                        "matched": True,
                        "matched_label": "sat" if want_sat else "unsat",
                        "matched_attempts": attempt + 1,
                    },
                ),
                attempt + 1,
            )
    return None, budget


def generate_matched(
    *,
    difficulty: dict,
    seed: int,
    count: int,
    encoding: str | None = None,
    max_attempts_per_slot: int = MAX_ATTEMPTS_PER_SLOT,
    abort_after_exhausted: int = 2,
    strict: bool = True,
) -> list[Instance]:
    """The density-matched control: `count` instances, exactly half satisfiable.

    Every instance in the returned list has the same n and the same m, so the
    density heuristic is constant across the set and the majority-class
    baseline is exactly 0.5. Accuracy above 0.5 here cannot have come from
    reading clause density, which is the whole reason E2 runs this alongside
    the plain sweep.

    Slots alternate SAT, UNSAT, and each slot is filled independently by
    rejection sampling from its own draw sequence, so the set is deterministic
    in (difficulty, seed) and slot `i` still depends only on `i`.

    Raises `MatchedSetUnfillable` when a half cannot be filled within the
    budget, which is what happens at the ends of the ratio range -- see
    `matched_sweep()` for the attainable range. With `strict=False` it instead
    returns a shorter but still balanced set, with `matched_short` set on every
    instance. It never returns an unbalanced set, because an unbalanced set
    looks exactly like a balanced one in the log while quietly restoring the
    density signal the control exists to remove. Where neither class can be
    reached at all, `strict=False` returns an empty list -- balanced, and with
    no instance left to carry the `matched_short` flag, so a caller that sets
    `strict=False` has to check the length.
    """
    encoding = _resolve_encoding(difficulty, encoding)
    n, ratio = _unpack(difficulty)
    m = clause_count(n, ratio)
    if count < 2 or count % 2:
        raise ValueError(f"matched count must be an even number >= 2, got {count}")
    if not 1 <= max_attempts_per_slot <= _ATTEMPT_STRIDE:
        raise ValueError(
            f"max_attempts_per_slot must be in 1..{_ATTEMPT_STRIDE}, got {max_attempts_per_slot}"
        )
    if abort_after_exhausted < 1:
        raise ValueError(f"abort_after_exhausted must be >= 1, got {abort_after_exhausted}")

    # The marker keeps the matched condition distinct from the plain sweep at
    # the same (n, ratio): different rng stream, different instance_id, and it
    # names the condition in every logged row.
    matched_key = {"n": n, "ratio": ratio, "matched": True}
    matched_difficulty = {**matched_key, "encoding": encoding}

    filled: dict[bool, list[tuple[int, Instance]]] = {True: [], False: []}
    attempts: dict[bool, int] = {True: 0, False: 0}
    consecutive_misses: dict[bool, int] = {True: 0, False: 0}
    abandoned: dict[bool, bool] = {True: False, False: False}

    for slot in range(count):
        want_sat = slot % 2 == 0
        other = not want_sat
        if abandoned[want_sat]:
            # This half will not grow, and the result is truncated to the
            # smaller half, so once the other half has caught up there is
            # nothing left to gain by drawing more.
            if abandoned[other] or len(filled[other]) >= len(filled[want_sat]):
                break
            continue
        instance, used = _fill_slot(
            difficulty=matched_difficulty,
            key=matched_key,
            seed=seed,
            slot=slot,
            want_sat=want_sat,
            n=n,
            m=m,
            budget=max_attempts_per_slot,
            encoding=encoding,
        )
        attempts[want_sat] += used
        if instance is None:
            # A slot that burns the whole budget means the class is close to
            # empty, not merely rare. Several in a row and it is not worth
            # spending the remaining slots to confirm it.
            consecutive_misses[want_sat] += 1
            if consecutive_misses[want_sat] >= abort_after_exhausted:
                abandoned[want_sat] = True
        else:
            consecutive_misses[want_sat] = 0
            filled[want_sat].append((slot, instance))

    half = min(len(filled[True]), len(filled[False]))
    short = 2 * half < count
    if short and strict:
        raise MatchedSetUnfillable(
            f"sat3 matched control at n={n} ratio={ratio} (m={m}): filled "
            f"{len(filled[True])} SAT and {len(filled[False])} UNSAT of {count // 2} "
            f"each after {attempts[True] + attempts[False]} draws at a budget of "
            f"{max_attempts_per_slot} per slot. One class is too rare here to "
            f"sample by rejection; matched_sweep() gives the attainable ratios.",
            n=n,
            ratio=ratio,
            n_sat=len(filled[True]),
            n_unsat=len(filled[False]),
            wanted=count // 2,
        )

    kept = sorted(filled[True][:half] + filled[False][:half], key=lambda pair: pair[0])
    out = []
    for _, instance in kept:
        instance.meta["matched_requested"] = count
        instance.meta["matched_returned"] = 2 * half
        instance.meta["matched_short"] = short
        instance.validate()
        out.append(instance)
    return out


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    import time

    import numpy as np
    from pysat.solvers import Glucose3

    SEED = 20260919
    SWEEP_COUNT = 30
    ANCHOR_COUNT = 120
    ANCHOR_RATIOS = (2.0, 3.0, 6.0, 8.0)

    _bit_cache: dict[int, dict[int, Any]] = {}

    def brute_force_satisfiable(clauses: list[list[int]], n: int) -> bool:
        """Decide satisfiability by enumerating all 2**n assignments.

        Deliberately shares no code and no library with `solve`. This is the
        check that a solver wired up backwards cannot pass.
        """
        if n not in _bit_cache:
            space = np.arange(1 << n, dtype=np.uint32)
            _bit_cache[n] = {v: (((space >> (v - 1)) & 1) == 1) for v in range(1, n + 1)}
        is_true = _bit_cache[n]
        alive = np.ones(1 << n, dtype=bool)
        for clause in clauses:
            hit = np.zeros(1 << n, dtype=bool)
            for lit in clause:
                hit |= is_true[abs(lit)] if lit > 0 else ~is_true[abs(lit)]
            alive &= hit
            if not alive.any():
                return False
        return bool(alive.any())

    def parse_dimacs(text: str) -> tuple[int, list[list[int]]]:
        """Read DIMACS back without calling to_dimacs, so the two can disagree.

        Verifying the state against meta["clauses"] is the second check this
        module needs after the solver check: ground truth is computed from the
        clause list, but the model only ever sees this text, and an encoder bug
        would mean the two are answers to different questions.
        """
        lines = text.splitlines()
        head = lines[0].split()
        assert head[0] == "p" and head[1] == "cnf", lines[0]
        declared_n, declared_m = int(head[2]), int(head[3])
        clauses = []
        for line in lines[1:]:
            literals = [int(token) for token in line.split()]
            assert literals[-1] == 0, f"clause line not 0-terminated: {line}"
            clauses.append(literals[:-1])
        assert len(clauses) == declared_m, f"header claims {declared_m}, found {len(clauses)}"
        return declared_n, clauses

    def second_opinion(clauses: list[list[int]]) -> bool:
        with Glucose3(bootstrap_with=clauses) as solver:
            return bool(solver.solve())

    def check_instance(inst: Instance, difficulty: dict) -> None:
        n, ratio = difficulty["n"], difficulty["ratio"]
        m = clause_count(n, ratio)
        clauses = inst.meta["clauses"]
        assert inst.generator == NAME
        assert inst.meta["n"] == n and inst.meta["m"] == m
        assert len(clauses) == m, f"{len(clauses)} clauses, expected {m}"
        for clause in clauses:
            assert len(clause) == 3, clause
            assert len({abs(lit) for lit in clause}) == 3, f"repeated variable: {clause}"
            assert all(1 <= abs(lit) <= n for lit in clause), clause

        truth = inst.truth[QUESTION_KEY]
        assert isinstance(truth, bool)
        assert truth == inst.meta["satisfiable"]
        assert inst.meta["baseline_density_pred"] == ((m / n) < DENSITY_THRESHOLD)

        assignment = inst.meta["assignment"]
        if truth:
            assert assignment is not None, "SAT instance with no recorded assignment"
            assert [abs(lit) for lit in assignment] == list(range(1, n + 1))
            assert satisfies(clauses, assignment), (
                f"recorded assignment does not satisfy {inst.instance_id}"
            )
        else:
            assert assignment is None, "UNSAT instance carries an assignment"

        assert second_opinion(clauses) is truth, f"glucose3 disagrees on {inst.instance_id}"

        parsed_n, parsed_clauses = parse_dimacs(inst.state)
        assert parsed_n == n, f"header declares {parsed_n} variables, expected {n}"
        assert parsed_clauses == clauses, (
            f"{inst.instance_id}: the DIMACS state is not the formula that was solved"
        )

    # ---- the full sweep -------------------------------------------------
    started = time.time()
    sweep = difficulty_sweep()
    ratios = _ratio_ladder()
    assert len(ratios) == 25 and ratios[0] == 2.0 and ratios[-1] == 8.0
    assert {d["n"] for d in sweep} == {10, 20, 50}
    assert len(sweep) == 75

    sat_fraction: dict[tuple[int, float], float] = {}
    baseline_accuracy: dict[tuple[int, float], float] = {}
    brute_forced = 0

    for difficulty in sweep:
        batch = generate(difficulty=difficulty, seed=SEED, count=SWEEP_COUNT)
        assert len(batch) == SWEEP_COUNT
        assert len({inst.instance_id for inst in batch}) == SWEEP_COUNT
        for inst in batch:
            check_instance(inst, difficulty)
            if difficulty["n"] == 10:
                assert brute_force_satisfiable(inst.meta["clauses"], 10) is inst.truth[
                    QUESTION_KEY
                ], f"exhaustive enumeration disagrees on {inst.instance_id}"
                brute_forced += 1
        key = (difficulty["n"], difficulty["ratio"])
        sat_fraction[key] = sum(i.truth[QUESTION_KEY] for i in batch) / SWEEP_COUNT
        baseline_accuracy[key] = sum(
            i.meta["baseline_density_pred"] == i.truth[QUESTION_KEY] for i in batch
        ) / SWEEP_COUNT

    # ---- the phase transition is the right way up -----------------------
    # An inverted solver produces a curve that rises with ratio instead of
    # falling, and every one of these assertions fails.
    anchors: dict[tuple[int, float], float] = {}
    for n in (10, 20, 50):
        for ratio in ANCHOR_RATIOS:
            batch = generate(difficulty={"n": n, "ratio": ratio}, seed=SEED + 1, count=ANCHOR_COUNT)
            anchors[(n, ratio)] = sum(i.truth[QUESTION_KEY] for i in batch) / ANCHOR_COUNT
        curve = [anchors[(n, r)] for r in ANCHOR_RATIOS]
        assert curve[0] == 1.0, f"n={n}: ratio 2.0 should be almost surely SAT, got {curve[0]}"
        assert curve[1] >= 0.85, f"n={n}: ratio 3.0 fraction {curve[1]}"
        assert curve[2] <= 0.35, f"n={n}: ratio 6.0 fraction {curve[2]}"
        assert curve[3] <= 0.15, f"n={n}: ratio 8.0 fraction {curve[3]}"
        assert curve == sorted(curve, reverse=True), f"n={n}: non-monotone anchors {curve}"

    # ---- determinism ----------------------------------------------------
    def dump(batch: list[Instance]) -> str:
        return json.dumps([i.to_json() for i in batch], sort_keys=True)

    for difficulty in ({"n": 20, "ratio": 4.25}, {"n": 10, "ratio": 2.0}, {"n": 50, "ratio": 5.5}):
        first = generate(difficulty=difficulty, seed=SEED, count=12)
        assert dump(first) == dump(generate(difficulty=difficulty, seed=SEED, count=12))
        # Instance i must not depend on how many instances were drawn with it.
        assert dump(first[:5]) == dump(generate(difficulty=difficulty, seed=SEED, count=5))
        assert dump(first) != dump(generate(difficulty=difficulty, seed=SEED + 1, count=12))

    # ---- encodings ------------------------------------------------------
    plain = generate(difficulty={"n": 20, "ratio": 4.25}, seed=SEED, count=3)
    structured = generate(
        difficulty={"n": 20, "ratio": 4.25}, seed=SEED, count=3, encoding="clauses"
    )
    for a, b in zip(plain, structured):
        assert isinstance(a.state, str) and isinstance(b.state, dict)
        assert a.meta["clauses"] == b.state["clauses"] == b.meta["clauses"]
        assert a.truth == b.truth
        assert a.state == to_dimacs(a.meta["clauses"], 20)
        # Same formula, so they pair; different states, so they are different
        # rows. Sharing an id would collapse them in any dict keyed on it.
        assert a.meta["formula_id"] == b.meta["formula_id"]
        assert a.instance_id != b.instance_id, "the two encodings collide on instance_id"
        assert a.difficulty["encoding"] == "dimacs" and b.difficulty["encoding"] == "clauses"
        # The prompt must not tell the model its JSON object is DIMACS text.
        qa = a.questions[QUESTION_KEY]["question"]
        qb = b.questions[QUESTION_KEY]["question"]
        assert "DIMACS" in qa and "DIMACS" not in qb, qb
        assert "JSON" in qb
        # Everything after the format sentence is word for word the same, so the
        # encoding comparison is not partly a comparison of two wordings.
        assert qa.endswith(_QUESTION_SENTENCE) and qb.endswith(_QUESTION_SENTENCE)
    # The two spellings of the encoding must agree, and must not disagree.
    assert dump(structured) == dump(
        generate(difficulty={"n": 20, "ratio": 4.25, "encoding": "clauses"}, seed=SEED, count=3)
    )
    try:
        generate(
            difficulty={"n": 20, "ratio": 4.25, "encoding": "clauses"},
            seed=SEED,
            count=1,
            encoding="dimacs",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("contradictory encoding spellings should have raised")
    # None is not an error: it means "unspecified", and resolves to dimacs.
    assert generate(difficulty={"n": 20, "ratio": 4.25}, seed=SEED, count=1, encoding=None)[
        0
    ].meta["encoding"] == "dimacs"
    for bad in ("DIMACS", "cnf", "", "json"):
        try:
            generate(difficulty={"n": 20, "ratio": 4.25}, seed=SEED, count=0, encoding=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"encoding={bad!r} should have raised")

    # ---- the density-matched control ------------------------------------
    matched_demo = {"n": 20, "ratio": 4.25}
    matched = generate_matched(difficulty=matched_demo, seed=SEED, count=20)
    assert len(matched) == 20
    labels = [i.truth[QUESTION_KEY] for i in matched]
    assert sum(labels) == 10, f"matched set is unbalanced: {sum(labels)} SAT of 20"
    assert len({i.instance_id for i in matched}) == 20
    assert {i.meta["m"] for i in matched} == {clause_count(20, 4.25)}
    assert {i.meta["baseline_density_pred"] for i in matched} == {True}
    assert not any(i.meta["matched_short"] for i in matched)
    for inst in matched:
        check_instance(inst, {"n": 20, "ratio": 4.25})
        assert inst.difficulty["matched"] is True
        assert inst.meta["matched_label"] == ("sat" if inst.truth[QUESTION_KEY] else "unsat")
    # Both baselines sit exactly at 0.5 on this set, which is the point of it.
    assert max(sum(labels), 20 - sum(labels)) / 20 == 0.5
    assert sum(i.meta["baseline_density_pred"] == i.truth[QUESTION_KEY] for i in matched) / 20 == 0.5
    assert dump(matched) == dump(generate_matched(difficulty=matched_demo, seed=SEED, count=20))
    # The matched condition must not collide with the plain sweep in the log.
    assert {i.instance_id for i in matched}.isdisjoint(
        {i.instance_id for i in generate(difficulty=matched_demo, seed=SEED, count=20)}
    )

    # Near the edge of the attainable range rejection still fills, just slowly.
    edge = generate_matched(difficulty={"n": 20, "ratio": 6.5}, seed=SEED, count=8)
    assert len(edge) == 8 and sum(i.truth[QUESTION_KEY] for i in edge) == 4
    edge_draws = max(i.meta["matched_attempts"] for i in edge)

    # Past the edge it must refuse rather than return something unbalanced.
    try:
        generate_matched(
            difficulty={"n": 20, "ratio": 8.0}, seed=SEED, count=10, max_attempts_per_slot=300
        )
    except MatchedSetUnfillable as exc:
        assert exc.n_sat < exc.wanted
        unfillable_message = str(exc)
        reachable = 2 * min(exc.n_sat, exc.n_unsat)
    else:
        raise AssertionError("ratio 8.0 has essentially no SAT instances; this should have raised")

    # Ratio 8.0 is past the point where either half is reachable, so the lenient
    # path returns nothing at all. Asserted explicitly: `all(...)` over an empty
    # list is true, so this case cannot be what tests the `matched_short` flag.
    lenient = generate_matched(
        difficulty={"n": 20, "ratio": 8.0},
        seed=SEED,
        count=10,
        max_attempts_per_slot=300,
        strict=False,
    )
    assert len(lenient) == reachable == 0, f"{len(lenient)} returned, expected none"

    # A cell where the rare class is reachable but not ten times over: short,
    # balanced and non-empty, which is the case that actually stamps the flag.
    partial = generate_matched(
        difficulty={"n": 20, "ratio": 7.0},
        seed=SEED,
        count=10,
        max_attempts_per_slot=150,
        strict=False,
    )
    assert 0 < len(partial) < 10, f"expected a short non-empty set, got {len(partial)}"
    assert len(partial) % 2 == 0
    assert sum(i.truth[QUESTION_KEY] for i in partial) * 2 == len(partial)
    assert all(i.meta["matched_short"] for i in partial)
    assert all(i.meta["matched_returned"] == len(partial) for i in partial)
    assert all(i.meta["matched_requested"] == 10 for i in partial)
    assert len({i.instance_id for i in partial}) == len(partial)
    for inst in partial:
        check_instance(inst, {"n": 20, "ratio": 7.0})
    # Strict refuses exactly where lenient comes up short.
    try:
        generate_matched(
            difficulty={"n": 20, "ratio": 7.0}, seed=SEED, count=10, max_attempts_per_slot=150
        )
    except MatchedSetUnfillable:
        pass
    else:
        raise AssertionError("strict should refuse the set lenient returned short")

    assert all(d["n"] in _MATCHED_RANGE for d in matched_sweep())
    assert {"n": 20, "ratio": 4.25} in matched_sweep()
    assert {"n": 20, "ratio": 8.0} not in matched_sweep()

    # ---- summary --------------------------------------------------------
    elapsed = time.time() - started
    print(f"sat3 self-check: {len(sweep)} difficulties x {SWEEP_COUNT} instances, {elapsed:.1f}s")
    print(f"  {brute_forced} n=10 instances re-decided by exhaustive enumeration")
    print("  every decision cross-checked against glucose3,"
          " every state re-parsed from DIMACS")
    print()
    print("  true P(SAT) by ratio                       density baseline")
    print(f"  {'ratio':>6} {'m@n=20':>7} {'n=10':>6} {'n=20':>6} {'n=50':>6}   {'acc@n=20':>9}")
    for ratio in ratios:
        marks = "".join(
            "*" if _MATCHED_RANGE[n][0] <= ratio <= _MATCHED_RANGE[n][1] else "." for n in (10, 20, 50)
        )
        print(
            f"  {ratio:>6.2f} {clause_count(20, ratio):>7}"
            f" {sat_fraction[(10, ratio)]:>6.2f}"
            f" {sat_fraction[(20, ratio)]:>6.2f}"
            f" {sat_fraction[(50, ratio)]:>6.2f}"
            f"   {baseline_accuracy[(20, ratio)]:>9.2f}   {marks}"
        )
    print("  (trailing marks: matched control attainable at n=10/20/50)")
    print()
    print(f"  anchors n=20: " + "  ".join(f"r={r}:{anchors[(20, r)]:.2f}" for r in ANCHOR_RATIOS))
    print("  matched at n=20 r=4.25: 20 instances, 10/10, baseline 0.50")
    print(f"  matched at n=20 r=6.5:  8 instances, worst slot took {edge_draws} draws")
    print(f"  matched at n=20 r=7.0:  strict refused; lenient returned {len(partial)} of 10, balanced")
    print(f"  matched at n=20 r=8.0:  refused -- {unfillable_message.splitlines()[0][:96]}...")
    print("  encodings: same formula, same formula_id, distinct instance_id,"
          " question names the format it is actually in")
    print("OK", file=sys.stderr)
