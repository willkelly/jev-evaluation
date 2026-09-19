"""Sudoku cell probes -- the applied enrollment condition for E5.

E5 item 6 asks whether enrolling an outcome space buys anything on a task where
the legal outcomes are known exactly. Sudoku supplies that: constraint
propagation yields the set of digits a cell may legally take, and the completed
grid says which of them is correct. A cell with one legal digit tests whether a
forced move is respected; a cell with five tests whether the returned preference
over legal-but-wrong digits means anything. The plan predicts near-perfect at
one option and near-random by five, so difficulty here is the number of legal
options for the probed cell and nothing else.

Ground truth. Each instance starts from a completed grid drawn by randomised
backtracking. Cells are then removed one at a time, each removal kept only if a
backtracking solver still finds exactly one solution; the counter stops at two,
because none, one and more-than-one is the whole distinction. The correct digit
for the probed cell is therefore the digit the original completed grid holds,
and it is the only digit any solver could return. A puzzle with two solutions
has no ground truth at all: it would score a legitimately correct answer as
wrong and corrupt the whole condition silently, which is why uniqueness is
verified rather than assumed.

Propagation. Three techniques, and deliberately no more:

  * peer elimination -- a cell cannot hold a digit already present in its row,
    column or box. This is what produces the candidate set, and the option list
    is exactly that set.
  * naked singles -- a cell left with one candidate is assigned it.
  * hidden singles -- a digit with one remaining placement in a row, column or
    box is assigned there.

Naked and hidden singles are applied in rounds: every assignment a round finds
is made at once, then candidates are recomputed. Locked candidates, naked pairs
and everything above them are left out on purpose. A stronger technique shrinks
the candidate sets, so including one would make the option list depend on how
clever this module is rather than on the grid, and `n_options` would stop being
comparable across a rerun that changed the technique set.

Which grid gets shown. Propagation turns the carved puzzle into a sequence of
grids: the puzzle, then the grid after each round. The instance shows the first
grid in that sequence holding a cell with exactly `n_options` candidates, and
only if that grid is no fuller than `MAX_PROBE_GIVENS`; past that the whole
puzzle is discarded and another is drawn. Both halves of that rule exist to
hold the number of filled cells constant over the sweep, so that accuracy
plotted against `n_options` is not also a plot against how much of the grid was
handed over. First-rather-than-most-propagated does most of the work, and the
cap closes the hole it leaves at `n_options` 1 -- see the constant. Measured at
n=300 per level with both in force, every difficulty runs 21 to 30 filled cells
with a median of 24 or 25 and a mean within 0.7 of every other difficulty.

One part of that confound cannot be removed: a cell has seven candidates only
if its twenty peers show two distinct digits between them, so the highest
difficulties come from emptier neighbourhoods. `meta["n_givens"]` records the
count on every instance so the analysis can check the residual.

State encoding. ``{"rows": [nine nine-character strings]}``, top row first, a
digit for a filled cell and "." for an empty one. This is the conventional
textual form for a sudoku, it is about half the characters of a nested array of
integers, and position is unambiguous: character c of row r is the cell at row
r+1, column c+1. The probed cell is named in the question text and kept out of
the state, so one grid can carry several cell questions in a batched E3 or E5
call without the state contradicting any of them.

Cells with one legal digit. `instances.choice` refuses fewer than two options,
so a cell with a single legal digit cannot be asked as a one-option choice. At
`n_options` 1 the option list is the legal digit plus one decoy drawn from the
digits peer elimination rules out, and `meta["padded"]` is set. That makes the
forced-move condition an actual elimination test instead of a question with one
possible answer, and it puts the random baseline at 0.5 rather than 1.0. The
plan's prediction P15 assumed a literal one-option choice, so the report must
state this deviation when scoring it.

Run the self-check with ``python -m jeveval.generators.sudoku``.
"""

from __future__ import annotations

import random
from typing import Iterator

from ..instances import Instance, choice, rng_for

NAME = "sudoku"

#: The single question every instance asks. Neutral by design -- E1 renames keys
#: to check they are not used in inference, and a key encoding the cell or the
#: answer would invalidate that test.
QUESTION_KEY = "cell_value"

#: A cell has eight candidates only when its twenty peers show one distinct digit
#: between them, which uniquely-solvable grids do not produce in any useful
#: quantity, so the sweep stops at seven.
MAX_OPTIONS = 7

#: Puzzles are redrawn until one contains a cell of the requested candidate
#: count. Seven candidates appears in about a third of minimal puzzles, so the
#: expected cost there is three draws and one draw everywhere else. The cap
#: turns a pathological case into an error instead of an unbounded loop.
MAX_ATTEMPTS = 40

#: A probe is taken only from a grid holding at most this many digits. Carved
#: puzzles run 21 to 28 givens, so the cap never rejects the puzzle itself; it
#: rejects only grids propagation has filled well past where any puzzle starts,
#: and those arise almost entirely at `n_options` 1, where two thirds of puzzles
#: offer a forced cell immediately and the rest need up to nine rounds to
#: produce one. Uncapped, 22% of `n_options` 1 instances came in above 30 givens
#: and the tail reached 52, against 21-28 at every other difficulty -- a density
#: difference sitting on exactly the level that `sudoku_forced_accuracy` scores
#: and E5's forced-move thresholds gate. Discarding the puzzle instead costs
#: about 0.4 extra draws per instance at `n_options` 1 and nothing elsewhere.
MAX_PROBE_GIVENS = 30

Grid = list[int]  # 81 cells, row-major, 0 for empty

FULL = 0b111111111

ROW_OF = [i // 9 for i in range(81)]
COL_OF = [i % 9 for i in range(81)]
BOX_OF = [(i // 27) * 3 + (i % 9) // 3 for i in range(81)]

UNITS: list[list[int]] = (
    [[r * 9 + c for c in range(9)] for r in range(9)]
    + [[r * 9 + c for r in range(9)] for c in range(9)]
    + [
        [(br * 3 + dr) * 9 + (bc * 3 + dc) for dr in range(3) for dc in range(3)]
        for br in range(3)
        for bc in range(3)
    ]
)

#: The three units containing each cell, used for the hidden-single baseline.
UNITS_OF: list[list[list[int]]] = [
    [UNITS[ROW_OF[i]], UNITS[9 + COL_OF[i]], UNITS[18 + BOX_OF[i]]] for i in range(81)
]


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


def _unit_masks(grid: Grid) -> tuple[list[int], list[int], list[int]]:
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    for i, v in enumerate(grid):
        if v:
            bit = 1 << (v - 1)
            rows[ROW_OF[i]] |= bit
            cols[COL_OF[i]] |= bit
            boxes[BOX_OF[i]] |= bit
    return rows, cols, boxes


def _count_solutions(grid: Grid, limit: int = 2) -> tuple[int, Grid | None]:
    """Count solutions up to `limit`, and return the first one found.

    Plain backtracking with most-constrained-cell ordering. It stops as soon as
    `limit` solutions exist: every caller here only needs to distinguish none,
    one, and more-than-one, and running to exhaustion on a sparse grid costs far
    more for an answer nobody reads.
    """
    g = list(grid)
    rows, cols, boxes = _unit_masks(g)
    found: list[Grid] = []

    def rec() -> bool:
        best = -1
        best_mask = 0
        best_count = 10
        for i in range(81):
            if g[i]:
                continue
            mask = FULL & ~(rows[ROW_OF[i]] | cols[COL_OF[i]] | boxes[BOX_OF[i]])
            n = mask.bit_count()
            if n < best_count:
                best_count, best, best_mask = n, i, mask
                if n <= 1:
                    break
        if best < 0:
            found.append(list(g))
            return len(found) >= limit
        r, c, b = ROW_OF[best], COL_OF[best], BOX_OF[best]
        mask = best_mask
        while mask:
            bit = mask & -mask
            mask ^= bit
            g[best] = bit.bit_length()
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            stop = rec()
            g[best] = 0
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            if stop:
                return True
        return False

    rec()
    return len(found), (found[0] if found else None)


def _random_solution(rng: random.Random) -> Grid:
    """A completed grid, filled cell by cell with the digits tried in rng order."""
    g = [0] * 81
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9

    def rec(i: int) -> bool:
        if i == 81:
            return True
        r, c, b = ROW_OF[i], COL_OF[i], BOX_OF[i]
        mask = FULL & ~(rows[r] | cols[c] | boxes[b])
        digits = [d for d in range(1, 10) if (mask >> (d - 1)) & 1]
        rng.shuffle(digits)
        for d in digits:
            bit = 1 << (d - 1)
            g[i] = d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            if rec(i + 1):
                return True
            g[i] = 0
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
        return False

    if not rec(0):
        raise RuntimeError("no completed grid exists, which cannot happen")
    return g


def _carve(solution: Grid, rng: random.Random) -> Grid:
    """Remove cells in rng order, keeping only removals that preserve uniqueness.

    The result is minimal under single-cell removal, which is what makes the
    higher difficulties reachable: a cell can only show six or seven candidates
    when its neighbourhood is nearly empty.
    """
    grid = list(solution)
    order = list(range(81))
    rng.shuffle(order)
    for i in order:
        v = grid[i]
        grid[i] = 0
        if _count_solutions(grid, limit=2)[0] != 1:
            grid[i] = v
    return grid


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------


def _candidates(grid: Grid) -> dict[int, int]:
    """Empty cell index -> bitmask of digits peer elimination leaves legal."""
    rows, cols, boxes = _unit_masks(grid)
    return {
        i: FULL & ~(rows[ROW_OF[i]] | cols[COL_OF[i]] | boxes[BOX_OF[i]])
        for i in range(81)
        if grid[i] == 0
    }


def _digits(mask: int) -> list[int]:
    return [d for d in range(1, 10) if (mask >> (d - 1)) & 1]


def _propagation_grids(puzzle: Grid, solution: Grid) -> Iterator[Grid]:
    """Yield the puzzle, then the grid after each round of singles.

    `solution` is not used to decide anything; it is checked against. Every
    assignment naked and hidden singles make is logically forced, so an
    assignment that disagrees with the unique solution is a bug in this module,
    and checking here catches it before it reaches an instance.
    """
    grid = list(puzzle)
    yield list(grid)
    while True:
        cands = _candidates(grid)
        assigns: dict[int, int] = {}
        for i, mask in cands.items():
            if mask.bit_count() == 1:
                assigns[i] = mask.bit_length()
        for unit in UNITS:
            spots: dict[int, int] = {}
            for i in unit:
                mask = cands.get(i, 0)
                while mask:
                    bit = mask & -mask
                    mask ^= bit
                    spots[bit] = i if bit not in spots else -1
            for bit, i in spots.items():
                if i >= 0:
                    assigns.setdefault(i, bit.bit_length())
        if not assigns:
            return
        for i in sorted(assigns):
            d = assigns[i]
            if solution[i] != d:
                raise AssertionError(
                    f"propagation assigned {d} to cell {i}, "
                    f"solution holds {solution[i]}"
                )
            grid[i] = d
        yield list(grid)


def _find_probe(
    puzzle: Grid, solution: Grid, n_options: int, rng: random.Random
) -> tuple[Grid, int, int, int] | None:
    """First (grid, cell, candidate mask, round) with exactly `n_options` candidates.

    None if the sequence runs out, and None as soon as a grid exceeds
    `MAX_PROBE_GIVENS`, since every later grid in the sequence is fuller still.
    Either way the caller draws a new puzzle.
    """
    for round_index, grid in enumerate(_propagation_grids(puzzle, solution)):
        if sum(1 for v in grid if v) > MAX_PROBE_GIVENS:
            return None
        cands = _candidates(grid)
        matches = [i for i, mask in cands.items() if mask.bit_count() == n_options]
        if matches:
            cell = rng.choice(matches)
            return grid, cell, cands[cell], round_index
    return None


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def _encode(grid: Grid) -> list[str]:
    return [
        "".join(str(v) if v else "." for v in grid[r * 9 : r * 9 + 9]) for r in range(9)
    ]


def _decode(rows: list[str]) -> Grid:
    return [0 if ch == "." else int(ch) for row in rows for ch in row]


# --------------------------------------------------------------------------
# Generator contract
# --------------------------------------------------------------------------


def difficulty_sweep() -> list[dict]:
    """Candidate counts 1 through 7, easy to hard.

    One option is the forced move the plan expects to be answered perfectly;
    five is where it predicts near-random. The sweep runs past that to seven so
    the curve has room on both sides of the predicted knee instead of ending on
    it. Eight is excluded because uniquely-solvable grids barely produce it.
    """
    return [{"n_options": k} for k in range(1, MAX_OPTIONS + 1)]


def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]:
    """`count` instances at one candidate count.

    Instance `i` depends only on `i`: its whole construction, including the
    retries that find a grid with a cell of the requested candidate count, runs
    off `rng_for(NAME, difficulty, seed, i)`.
    """
    n_options = int(difficulty["n_options"])
    if not 1 <= n_options <= MAX_OPTIONS:
        raise ValueError(
            f"n_options must be 1..{MAX_OPTIONS}, got {n_options}; see MAX_OPTIONS"
        )
    return [
        _one(difficulty, seed, i, n_options, rng_for(NAME, difficulty, seed, i))
        for i in range(count)
    ]


def _one(
    difficulty: dict, seed: int, index: int, n_options: int, rng: random.Random
) -> Instance:
    for _ in range(MAX_ATTEMPTS):
        solution = _random_solution(rng)
        puzzle = _carve(solution, rng)
        n_solutions, _ = _count_solutions(puzzle, limit=2)
        if n_solutions != 1:
            raise AssertionError(f"carved puzzle has {n_solutions} solutions, not 1")
        probe = _find_probe(puzzle, solution, n_options, rng)
        if probe is not None:
            break
    else:
        raise RuntimeError(
            f"no cell with exactly {n_options} candidates in {MAX_ATTEMPTS} puzzles "
            f"(generator={NAME}, seed={seed}, index={index})"
        )

    grid, cell, mask, round_index = probe
    digits = _digits(mask)
    truth = str(solution[cell])
    if truth not in {str(d) for d in digits}:
        raise AssertionError(
            f"solution digit {truth} is not a candidate for cell {cell}"
        )

    candidates = [str(d) for d in digits]
    decoys: list[str] = []
    if len(candidates) < 2:
        eliminated = sorted(set(range(1, 10)) - set(digits))
        decoys = [str(rng.choice(eliminated))]
    option_ids = candidates + decoys
    rng.shuffle(option_ids)

    row, col = ROW_OF[cell], COL_OF[cell]
    question = choice(
        "Each string in `rows` is one row of a 9x9 sudoku grid, listed top to "
        "bottom; a digit is a filled cell and '.' is an empty one. Exactly one "
        f"digit completes the grid at row {row + 1}, column {col + 1}. Which is it?",
        option_ids,
    )

    cands = _candidates(grid)
    hidden_singles = []
    for d in digits:
        bit = 1 << (d - 1)
        if any(
            sum(1 for j in unit if cands.get(j, 0) & bit) == 1
            for unit in UNITS_OF[cell]
        ):
            hidden_singles.append(str(d))

    inst = Instance(
        generator=NAME,
        difficulty=dict(difficulty),
        seed=seed,
        index=index,
        state={"rows": _encode(grid)},
        questions={QUESTION_KEY: question},
        truth={QUESTION_KEY: truth},
        meta={
            "n_options": n_options,
            "candidates": candidates,
            "cell": {
                "row": row + 1,
                "column": col + 1,
                "box": BOX_OF[cell] + 1,
                "index": cell,
            },
            "n_givens": sum(1 for v in grid if v),
            # 1/k, except at n_options 1 where the decoy makes a coin flip the
            # honest chance rate. Always 1/(number of options actually asked).
            "baseline_random": 1.0 / len(option_ids),
            "n_emitted_options": len(option_ids),
            "padded": bool(decoys),
            "decoys": decoys,
            "propagation_round": round_index,
            "carved_givens": sum(1 for v in puzzle if v),
            # Two cheap baselines an offline scorer can run from meta alone: is
            # the true digit forced by a one-step hidden single, and is it the
            # candidate already placed most often elsewhere in the grid?
            "hidden_single_candidates": hidden_singles,
            "candidate_placed_counts": {
                str(d): sum(1 for v in grid if v == d) for d in digits
            },
        },
    )
    inst.validate()
    return inst


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import statistics
    import time

    SEED = 20260919
    COUNT = 15

    print(f"{NAME}: {COUNT} instances per difficulty, seed {SEED}")
    print(
        f"{'k':>2} {'opts':>4} {'base':>5} {'givens':>10} {'round':>7} "
        f"{'hs-hit':>6} {'ms/inst':>8}"
    )

    for difficulty in difficulty_sweep():
        k = difficulty["n_options"]
        t0 = time.perf_counter()
        batch = generate(difficulty=difficulty, seed=SEED, count=COUNT)
        elapsed = time.perf_counter() - t0
        assert len(batch) == COUNT, f"asked {COUNT}, got {len(batch)}"
        assert len({i.instance_id for i in batch}) == COUNT, "duplicate instance ids"

        for inst in batch:
            grid = _decode(inst.state["rows"])
            n_solutions, solution = _count_solutions(grid, limit=2)
            assert n_solutions == 1, f"{inst.instance_id}: {n_solutions} solutions"
            assert solution is not None

            cell = inst.meta["cell"]["index"]
            assert inst.truth[QUESTION_KEY] == str(solution[cell])
            assert inst.truth[QUESTION_KEY] in inst.meta["candidates"]
            assert inst.meta["n_options"] == k == len(inst.meta["candidates"])

            # Propagation never eliminates the true value, and never contradicts
            # the solution on a cell it fills.
            for stage in _propagation_grids(grid, solution):
                for i, mask in _candidates(stage).items():
                    assert (mask >> (solution[i] - 1)) & 1, (
                        f"{inst.instance_id}: propagation eliminated {solution[i]} "
                        f"at cell {i}"
                    )
                for i, v in enumerate(stage):
                    assert v in (0, solution[i])

            options = [o["id"] for o in inst.questions[QUESTION_KEY]["options"]]
            assert sorted(options) == sorted(
                inst.meta["candidates"] + inst.meta["decoys"]
            )
            assert len(options) == inst.meta["n_emitted_options"] >= 2
            assert inst.meta["baseline_random"] == 1.0 / len(options)
            assert inst.meta["padded"] == (k == 1)
            # The density control, asserted rather than eyeballed off the table
            # below: uncapped, n_options 1 reached 52 givens.
            assert inst.meta["n_givens"] <= MAX_PROBE_GIVENS, (
                f"{inst.instance_id}: {inst.meta['n_givens']} givens exceeds "
                f"MAX_PROBE_GIVENS={MAX_PROBE_GIVENS}"
            )

        repeat = generate(difficulty=difficulty, seed=SEED, count=COUNT)
        assert [json.dumps(x.to_json()) for x in batch] == [
            json.dumps(y.to_json()) for y in repeat
        ], f"n_options={k} is not deterministic in (difficulty, seed)"

        givens = [i.meta["n_givens"] for i in batch]
        rounds = [i.meta["propagation_round"] for i in batch]
        hs = sum(
            1
            for i in batch
            if i.truth[QUESTION_KEY] in i.meta["hidden_single_candidates"]
        )
        print(
            f"{k:>2} {batch[0].meta['n_emitted_options']:>4} "
            f"{batch[0].meta['baseline_random']:>5.2f} "
            f"{min(givens):>3}-{max(givens):<3}({statistics.median(givens):>2.0f}) "
            f"{min(rounds)}-{max(rounds):<5} "
            f"{hs / COUNT:>6.2f} {elapsed / COUNT * 1000:>8.1f}"
        )

    print("ok: unique solutions, truth always a candidate, deterministic")
