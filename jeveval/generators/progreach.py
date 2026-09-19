"""Program reachability, one program rendered three ways.

E4.3 asks whether the shape of the state changes the answer: source text, AST
as nested JSON, or a flat CFG edge list. That comparison is only worth running
as a paired test, so `encoding` is a difficulty knob rather than a separate
generator, and the program is drawn from an RNG seeded from the difficulty dict
*with `encoding` removed*. Instance `i` at difficulty depth=4 is therefore
literally the same program in all three conditions -- same target line, same
ground truth, same question text -- and `meta["program_id"]` is the join key
that lets the analysis pair them.

The other two knobs, `depth` and `live_conditions`, control how much of the
program a reader has to traverse. `constraint_required` controls *why* an
unreachable line is unreachable, and it is the bridge to E2:

  - constraint_required=False: every branch condition tests a different
    variable, so no conjunction of guards is ever contradictory and
    reachability is decided by control flow alone -- a line is dead because
    a `return` precedes it, or because both arms of an `if/else` above it
    return. Walking the graph is sufficient.
  - constraint_required=True: control flow always reaches the line, and
    reachability hinges on whether the conjunction of enclosing integer
    conditions is satisfiable. The plan's example (`if x > 5` inside
    `if x < 3`) is the depth-2 case of the `squeeze` family below. This is a
    finite-domain CSP, so it is the same kind of question E2 asks with 3SAT,
    asked in a syntax the model is far more likely to have seen.

A model that scores well on the first and poorly on the second has a
constraint-solving weakness, not a graph-traversal weakness. Keeping the two
in separate conditions, with `constraint_required` in meta, is what makes that
inference available.

Ground truth is not taken from the construction. The program is built to a
target answer, then the answer is recomputed from scratch: enumerate every
entry-to-target path in the CFG, and ask a finite-domain solver whether any
path's guard conjunction is satisfiable. `generate` uses the solver's verdict
and raises if it disagrees with what was intended, so a construction bug shows
up as a crash rather than as a condition of mislabelled data. The self-check
cross-checks the solver against minisat via a one-hot encoding, and against
brute-force enumeration on small cases.

Confound control, in the same spirit as E2's density-matched pairs:

  - In the structural conditions the two answers are assembled from exactly the
    same statements in a different order; in the constraint conditions they
    differ by one constant or one relation direction. So an instance and the
    opposite-answer instance built from the same draw have the same line count,
    statement count and return count. The self-check verifies that pair by
    pair rather than leaving it as a claim, because statement count is the
    cheap baseline the plan requires this generator to report against.
  - All structural decisions are drawn before the intended answer is consulted,
    so program shape has the same distribution under both answers.
  - Variable names are drawn from a per-instance shuffle of the pool, so no
    name correlates with a role.
  - The target line is a `record(total)` call and so are several filler lines,
    so the question cannot be answered by finding the one odd-looking
    statement.
  - `_check_shortcuts` measures the cheap rules an adversarial reader reaches
    for first, because "nothing on the surface says which answer it is" is a
    measurement and not an argument. One of them used to score 1.000: the
    relation directions gave away every `cycle` instance at every depth, which
    is half the constraint instances at depths 2 and 4. It is at chance now and
    the self-check fails if it comes back. Two others still score 1.000 and are
    printed rather than fixed, because neither is a spelling accident. At depth
    1 the unreachable answer is a single condition no input can satisfy, which
    is what makes depth 1 the floor. In the `alldiff` family the width of the
    window is the answer in one step at any depth -- see `_core_alldiff` -- and
    it is named in the meta comment so the E4 writer splits on `core_family`
    rather than discovering it in the results.

The CFG encoding makes structural unreachability nearly free -- a dead block has
no incoming edge -- while leaving the constraint cases as hard as they were.
That asymmetry is the finding E4.3 is looking for, not a defect in the
generator.
"""

from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..instances import Instance, noul, rng_for

__all__ = ["NAME", "difficulty_sweep", "generate"]

NAME = "progreach"

ENCODINGS = ("source", "ast", "cfg")

# Every input variable ranges over this closed interval. A bounded domain is
# what makes `v > 11` an unsatisfiable condition rather than an ordinary one,
# and it makes the solver total: there is always a finite search to finish.
DOMAIN_LO = 0
DOMAIN_HI = 11

VAR_POOL = ("a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "m", "n", "p", "q", "r", "s")

# (nesting depth of the target, number of conditions on the route to it).
# Depth 1 is the floor: one guard, an answer any working model should get, so a
# bad number there means the harness and not the model. It is a floor in a
# second sense too -- with one guard, "this condition cannot hold anywhere in
# [0, 11]" is the whole question, and `_check_shortcuts` confirms that rule
# answers every unreachable depth-1 instance. Depth 6 with 10 conditions is a
# ~35-line program: measured, 32 to 35 lines and at most 16 entry-to-target
# paths, the bound being 2 ** (live_conditions - depth) since each distractor
# on the route either rejoins or returns. That is past where a reader holds the
# whole thing at once. The knee, if there is one, is bracketed.
_LADDER = ((1, 2), (2, 4), (4, 7), (6, 10))


# --------------------------------------------------------------------------
# Conditions and the finite-domain solver
# --------------------------------------------------------------------------

_OPS: dict[str, Callable[[int, int], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

_NEGATION = {"<": ">=", ">=": "<", ">": "<=", "<=": ">", "==": "!=", "!=": "=="}

# `a < b` and `b > a` are the same constraint written from opposite ends. The
# cores use this to choose which end to write from, independently of anything
# the answer depends on.
_MIRROR = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}


@dataclass(frozen=True)
class _Cond:
    """`left op right`, where `right` is a constant or another variable."""

    op: str
    left: str
    right: int | str

    @property
    def binary(self) -> bool:
        return isinstance(self.right, str)

    def text(self) -> str:
        return f"{self.left} {self.op} {self.right}"

    def holds(self, env: dict[str, int]) -> bool:
        right = env[self.right] if self.binary else self.right
        return _OPS[self.op](env[self.left], right)


def _negate(c: _Cond) -> _Cond:
    return _Cond(_NEGATION[c.op], c.left, c.right)


def _vars_of(conds: Iterable[_Cond]) -> list[str]:
    seen: set[str] = set()
    for c in conds:
        seen.add(c.left)
        if c.binary:
            seen.add(c.right)  # type: ignore[arg-type]
    return sorted(seen)


def _satisfy(conds: list[_Cond]) -> dict[str, int] | None:
    """A satisfying assignment for the conjunction, or None if there is none.

    Unary conditions prune each variable's domain outright. The variables that
    remain are split into connected components of the binary-constraint graph
    and searched one component at a time -- without that split, proving a
    six-variable core unsatisfiable is repeated once for every assignment of the
    unrelated variables the distractor branches contribute. Measured on the 69
    path conjunctions of twelve depth-6 constraint instances: 0.12 seconds with
    the split, and without it the same 69 were still unfinished after 120
    seconds, 11 of them done. The split is not an optimization, it is what makes
    the top of the ladder reachable at all.

    Within a component the search visits variables breadth-first from the
    most-constrained one, so every variable after the first shares a constraint
    with one already bound and a violation is caught on assignment rather than
    at the end. The search is complete -- it only ever skips an assignment that
    violates a condition -- so None is a proof of unsatisfiability, not a
    timeout.
    """
    names = _vars_of(conds)
    if not names:
        return {}
    domain: dict[str, list[int]] = {
        v: list(range(DOMAIN_LO, DOMAIN_HI + 1)) for v in names
    }
    binary: dict[str, list[_Cond]] = {v: [] for v in names}
    adjacent: dict[str, set[str]] = {v: set() for v in names}
    for c in conds:
        if c.binary:
            right: str = c.right  # type: ignore[assignment]
            binary[c.left].append(c)
            binary[right].append(c)
            if c.left != right:
                adjacent[c.left].add(right)
                adjacent[right].add(c.left)
        else:
            domain[c.left] = [d for d in domain[c.left] if _OPS[c.op](d, c.right)]
            if not domain[c.left]:
                return None

    env: dict[str, int] = {}

    def search(order: list[str]) -> bool:
        if not order:
            return True
        v = order[0]
        for d in domain[v]:
            env[v] = d
            if all(
                c.holds(env) for c in binary[v] if c.left in env and c.right in env
            ) and search(order[1:]):
                return True
        env.pop(v, None)
        return False

    placed: set[str] = set()
    for start in sorted(names, key=lambda v: (len(domain[v]), v)):
        if start in placed:
            continue
        component: list[str] = []
        frontier = [start]
        reached = {start}
        while frontier:
            frontier.sort(key=lambda v: (len(domain[v]), v))
            v = frontier.pop(0)
            component.append(v)
            for w in sorted(adjacent[v]):
                if w not in reached:
                    reached.add(w)
                    frontier.append(w)
        placed |= reached
        if not search(component):
            return None
    return dict(env)


# --------------------------------------------------------------------------
# Constraint cores
# --------------------------------------------------------------------------
#
# A core is `k` conditions whose conjunction is satisfiable or not, in matched
# pairs: the two variants have the same shape and the same variables, and
# differ by one constant or one relation direction.
#
# Matching the shape is not enough on its own, because how a condition is
# *spelled* can still give the answer away: `>= 12` announces itself against a
# domain of [0, 11], and a chain of relations all written in the same direction
# announces a cycle. Each family below says which spellings it randomizes and
# why, and `_check_shortcuts` measures whether the randomization worked.


def _at_least(var: str, bound: int, rng) -> _Cond:
    """`var >= bound`, spelled `>= bound` or `> bound - 1`, constant in domain.

    Which spelling is a coin flip while both literals sit inside the declared
    input range, and forced when one of them does not. Only the depth-1 squeeze
    core reaches that second case, where the unsatisfiable variant asks for a
    lower bound of 12: `>= 12` puts a literal in the program that no input can
    ever equal, while `> 11` says the same thing in the program's own
    vocabulary. Before this, 26% of depth-1 constraint instances carried a `12`
    or a `-1` and every one of them was unreachable.

    This buys less than it looks like. It removes a lexical tell -- a number
    outside the stated range, visible without reading the comparison at all --
    but not the semantic one: `x > 11` is still unsatisfiable on sight against
    inputs declared over [0, 11], and `_check_shortcuts` measures that rule
    answering every unreachable depth-1 instance. That is the floor doing its
    job, not a leak; with one guard there is nothing else for the question to
    be. Past depth 1 no single condition is unsatisfiable on its own and the
    rule never fires, which the self-check asserts.
    """
    if DOMAIN_LO <= bound - 1 and bound <= DOMAIN_HI:
        return _Cond(">=", var, bound) if rng.random() < 0.5 else _Cond(">", var, bound - 1)
    rng.random()  # spend the draw either way, so the two answers stay aligned
    return _Cond(">=", var, bound) if bound <= DOMAIN_HI else _Cond(">", var, bound - 1)


def _at_most(var: str, bound: int, rng) -> _Cond:
    """`var <= bound`, spelled `<= bound` or `< bound + 1`, constant in domain.

    The mirror of `_at_least`; the forced case there is a bound of -1 here.
    """
    if DOMAIN_LO <= bound and bound + 1 <= DOMAIN_HI:
        return _Cond("<=", var, bound) if rng.random() < 0.5 else _Cond("<", var, bound + 1)
    rng.random()
    return _Cond("<=", var, bound) if bound >= DOMAIN_LO else _Cond("<", var, bound + 1)


def _core_squeeze(k: int, sat: bool, rng, var: str) -> list[_Cond]:
    """k conditions narrowing one variable's range, alternating from each side.

    The last one leaves exactly one value (satisfiable) or none. Depth 2 of
    this family is the plan's `if x > 5` inside `if x < 3`.
    """
    lo, hi = DOMAIN_LO, DOMAIN_HI
    conds: list[_Cond] = []
    from_low = rng.random() < 0.5
    for j in range(k - 1):
        # Leave room for the remaining steps plus two values for the closer.
        remaining = k - 2 - j
        step = rng.randint(1, max(1, hi - lo - 1 - remaining))
        if from_low:
            lo += step
            conds.append(_at_least(var, lo, rng))
        else:
            hi -= step
            conds.append(_at_most(var, hi, rng))
        from_low = not from_low
    assert hi >= lo, "squeeze core narrowed to nothing before the closing condition"
    # Close against a bound that an earlier condition actually moved, so
    # deciding the core needs the conjunction and not just the closing
    # condition. At k=1 there is no earlier condition, so the single condition
    # is the one that empties the domain by itself -- `x < 0` or `x > 11`,
    # spelled by the helpers above so the literal still lies inside [0, 11].
    sides = []
    if hi < DOMAIN_HI:
        sides.append("lower")
    if lo > DOMAIN_LO:
        sides.append("upper")
    if not sides:
        sides = ["lower", "upper"]
    if rng.choice(sides) == "lower":
        conds.append(_at_least(var, hi if sat else hi + 1, rng))
    else:
        conds.append(_at_most(var, lo if sat else lo - 1, rng))
    return conds


def _core_cycle(k: int, sat: bool, rng, names: list[str]) -> list[_Cond]:
    """A chain of k-1 strict inequalities plus one closing relation.

    Closing the chain back on itself is unsatisfiable; closing it in the same
    direction as the chain is not. Detecting the first requires following the
    whole chain, which is why this family gets harder with depth while the
    squeeze family does not.

    Which end of each relation is written first is then drawn independently,
    and that is not cosmetic. Written all one way, the unsatisfiable variant is
    `a < b < c < ... < a`: every operator points the same direction, while the
    satisfiable twin has exactly one pointing the other way. "Are the operators
    all the same?" then answers the question perfectly without following the
    chain at all -- measured at 100% accuracy on every cycle instance at every
    depth, which is 50% of the constraint instances at depths 2 and 4. Mirroring
    each relation on a coin flip leaves both variants with the same distribution
    of operators, so the chain has to be oriented and traversed.
    """
    rel = "<" if rng.random() < 0.5 else ">"
    # Closing the cycle with the chain's own relation is the contradiction;
    # closing it the other way restates what the chain already implies.
    closing = rel if not sat else _MIRROR[rel]
    links = [(names[j], names[j + 1], rel) for j in range(k - 1)]
    links.append((names[k - 1], names[0], closing))
    conds = []
    for left, right, op in links:
        if rng.random() < 0.5:
            conds.append(_Cond(op, left, right))
        else:
            conds.append(_Cond(_MIRROR[op], right, left))
    return conds


def _core_alldiff(sat: bool, rng, names: list[str]) -> list[_Cond]:
    """Three variables confined to a short range and required to differ.

    Three values available is satisfiable, two is not. Unlike the other two
    families this one has no contradictory pair: every condition is consistent
    with every other, and only the three together are unsatisfiable.

    The window is anchored at either end of the domain on a coin flip, so the
    bound is `< 3` / `> 8` when satisfiable and `< 2` / `> 9` when not, and the
    two answers draw their constants from overlapping sets. Read the honest
    limit of this in `_check_shortcuts`: with one bound per variable, the width
    of the window *is* the answer, and a reader who converts the bound to a
    width has the answer without doing the counting the family is there to
    require. That is not true of squeeze or cycle, it is why `core_family`
    travels in meta, and it is why the audit reports this family separately.
    """
    room = 3 if sat else 2
    if rng.random() < 0.5:
        bounds = [_at_most(v, DOMAIN_LO + room - 1, rng) for v in names]
    else:
        bounds = [_at_least(v, DOMAIN_HI - room + 1, rng) for v in names]
    return bounds + [
        _Cond("!=", names[0], names[1]),
        _Cond("!=", names[0], names[2]),
        _Cond("!=", names[1], names[2]),
    ]


def _core_families(k: int) -> list[str]:
    families = ["squeeze"]
    if k >= 2:
        families.append("cycle")
    if k == 6:
        families.append("alldiff")
    return families


def _independent_cond(var: str, rng) -> _Cond:
    """A condition whose truth and whose negation are both satisfiable.

    Used where reachability must not depend on the constraints: each such
    condition sits on its own variable, so any conjunction of them and their
    negations is satisfiable and control flow decides the question by itself.
    """
    op = rng.choice(["<", "<=", ">", ">=", "==", "!="])
    if op == "<":
        const = rng.randint(DOMAIN_LO + 1, DOMAIN_HI)
    elif op == ">=":
        const = rng.randint(DOMAIN_LO + 1, DOMAIN_HI)
    elif op == "<=":
        const = rng.randint(DOMAIN_LO, DOMAIN_HI - 1)
    elif op == ">":
        const = rng.randint(DOMAIN_LO, DOMAIN_HI - 1)
    else:
        const = rng.randint(DOMAIN_LO, DOMAIN_HI)
    return _Cond(op, var, const)


# --------------------------------------------------------------------------
# Program representation
# --------------------------------------------------------------------------


@dataclass
class _Node:
    """One statement. `line` is filled in by the renderer, before anything
    else looks at it, because all three encodings quote the same line numbers."""

    kind: str  # "simple" | "return" | "if"
    text: str = ""
    cond: _Cond | None = None
    body: list["_Node"] = field(default_factory=list)
    orelse: list["_Node"] = field(default_factory=list)
    line: int = 0
    is_target: bool = False


def _return() -> _Node:
    return _Node("return", text="return total")


def _filler(rng) -> _Node:
    pick = rng.randrange(6)
    if pick == 0:
        return _Node("simple", text=f"total = total + {rng.randint(1, 9)}")
    if pick == 1:
        return _Node("simple", text=f"total = total - {rng.randint(1, 9)}")
    if pick == 2:
        return _Node("simple", text=f"scratch = total * {rng.randint(2, 4)}")
    if pick == 3:
        return _Node("simple", text=f"scratch = total + {rng.randint(1, 9)}")
    return _Node("simple", text="record(total)")


def _statement_count(stmts: Iterable[_Node]) -> int:
    n = 0
    for s in stmts:
        n += 1
        if s.kind == "if":
            n += _statement_count(s.body) + _statement_count(s.orelse)
    return n


def _return_count(stmts: Iterable[_Node]) -> int:
    n = 0
    for s in stmts:
        if s.kind == "return":
            n += 1
        elif s.kind == "if":
            n += _return_count(s.body) + _return_count(s.orelse)
    return n


def _all_texts(stmts: Iterable[_Node]) -> list[str]:
    """Every rendered statement, for checking that a matched pair really is
    matched: the reachable and unreachable variants of a structural instance
    should hold the same statements in a different order."""
    out: list[str] = []
    for s in stmts:
        if s.kind == "if":
            assert s.cond is not None
            out.append(f"if {s.cond.text()}:")
            out += _all_texts(s.body) + _all_texts(s.orelse)
        else:
            out.append(s.text)
    return out


def _count_live(stmts: Iterable[_Node]) -> int | None:
    """Conditions on the syntactic route to the target, or None if not here.

    That is every `if` enclosing the target plus every `if` that precedes it in
    an enclosing suite. Defined this way rather than as "conditions on some
    entry-to-target CFG path" because the latter is zero whenever the target is
    structurally dead, which would make the knob mean different things for the
    two answers.
    """
    seen = 0
    for s in stmts:
        if s.kind == "if":
            for arm in (s.body, s.orelse):
                inner = _count_live(arm)
                if inner is not None:
                    return seen + 1 + inner
            seen += 1
        elif s.is_target:
            return seen
    return None


def _find_target(stmts: Iterable[_Node]) -> _Node | None:
    for s in stmts:
        if s.is_target:
            return s
        if s.kind == "if":
            for arm in (s.body, s.orelse):
                found = _find_target(arm)
                if found is not None:
                    return found
    return None


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def _distractor(cond: _Cond, rng) -> _Node:
    """A branch on the route to the target that cannot cut it off.

    Single-armed, or two-armed with a falling-through else, so control always
    rejoins. The body may return; that is safe for exactly the same reason, and
    it is what stops "the program contains a return" from predicting the
    answer.
    """
    body = [_filler(rng)]
    if rng.random() < 0.4:
        body.append(_return())
    orelse = [_filler(rng)] if rng.random() < 0.5 else []
    return _Node("if", cond=cond, body=body, orelse=orelse)


def _build_program(
    *, depth: int, live_conditions: int, constraint_required: bool, reachable: bool, rng
) -> dict[str, Any]:
    """Assemble the program. Everything that does not depend on the intended
    answer is drawn first, so the distribution of program shapes is identical
    for reachable and unreachable instances."""

    names = list(VAR_POOL)
    rng.shuffle(names)
    supply = iter(names)

    def take() -> str:
        try:
            return next(supply)
        except StopIteration:
            raise ValueError(
                f"depth={depth} with live_conditions={live_conditions} needs more "
                f"than the {len(VAR_POOL)} variables in the pool"
            ) from None

    device = None
    device_conds = 0
    if not constraint_required:
        devices = ["return_before"]
        if live_conditions - depth >= 1:
            devices.append("if_else_return")
        device = rng.choice(devices)
        device_conds = 1 if device == "if_else_return" else 0

    n_distract = live_conditions - depth - device_conds
    if n_distract < 0:
        raise ValueError(
            f"live_conditions={live_conditions} is too small for depth={depth}"
            f" with device={device!r}"
        )

    # Where the distractors sit: level 0 is the function body, level j is inside
    # the j-th enclosing `if`. Level `depth` is the target's own suite.
    levels: list[list[_Node]] = [[] for _ in range(depth + 1)]
    placements = [rng.randrange(depth + 1) for _ in range(n_distract)]
    device_var = take() if device == "if_else_return" else None
    distract_vars = [take() for _ in range(n_distract)]
    trailing = [_filler(rng) for _ in range(depth + 1)]
    lead = _filler(rng)

    core_family = None
    if constraint_required:
        core_family = rng.choice(_core_families(depth))
        if core_family == "squeeze":
            core_vars = [take()]
            chain = _core_squeeze(depth, reachable, rng, core_vars[0])
        elif core_family == "cycle":
            core_vars = [take() for _ in range(depth)]
            chain = _core_cycle(depth, reachable, rng, core_vars)
        else:
            core_vars = [take() for _ in range(3)]
            chain = _core_alldiff(reachable, rng, core_vars)
        # Order along the nest does not change the conjunction, and shuffling
        # keeps the deciding condition from always being the innermost one.
        rng.shuffle(chain)
    else:
        chain = [_independent_cond(take(), rng) for _ in range(depth)]

    if len(chain) != depth:
        raise AssertionError(f"core produced {len(chain)} conditions, need {depth}")

    for level, var in zip(placements, distract_vars):
        levels[level].append(_distractor(_independent_cond(var, rng), rng))

    target = _Node("simple", text="record(total)", is_target=True)

    inner: list[_Node] = list(levels[depth])
    tail = trailing[depth]
    if device == "return_before":
        # The same two statements in either order: with the return first the
        # target is dead, with it second the target is live.
        inner += [_return(), target] if not reachable else [target, _return()]
    elif device == "if_else_return":
        # Both arms returning leaves the join -- and the target after it -- with
        # no incoming edge. The reachable twin does not drop that second return:
        # it swaps it with the statement after the target, so the two variants
        # are assembled from exactly the same statements.
        assert device_var is not None
        device_cond = _independent_cond(device_var, rng)
        arm_true = [_filler(rng), _return()]
        arm_false_head = _filler(rng)
        if reachable:
            arm_false = [arm_false_head, trailing[depth]]
            tail = _return()
        else:
            arm_false = [arm_false_head, _return()]
            tail = trailing[depth]
        inner += [
            _Node("if", cond=device_cond, body=arm_true, orelse=arm_false),
            target,
        ]
    else:
        inner.append(target)
    inner.append(tail)

    suite = inner
    for j in range(depth - 1, -1, -1):
        suite = list(levels[j]) + [_Node("if", cond=chain[j], body=suite)] + [trailing[j]]

    body = [_Node("simple", text="total = 0"), lead] + suite
    body.append(_return())

    used = sorted(_vars_of(_all_conds(body)))
    return {
        "body": body,
        "inputs": used,
        "device": device,
        "core_family": core_family,
        "distractors": n_distract,
    }


def _all_conds(stmts: Iterable[_Node]) -> list[_Cond]:
    out: list[_Cond] = []
    for s in stmts:
        if s.kind == "if":
            assert s.cond is not None
            out.append(s.cond)
            out += _all_conds(s.body)
            out += _all_conds(s.orelse)
    return out


# --------------------------------------------------------------------------
# Rendering: source
# --------------------------------------------------------------------------


def _render_source(body: list[_Node], inputs: list[str]) -> list[str]:
    """Render the program and stamp every node with its line number.

    Lines are numbered in the text itself. Without that, the `source` condition
    would partly be measuring whether the model can count lines, while the AST
    and CFG conditions hand it line numbers as fields -- and the encoding
    comparison would be confounded by exactly the thing it is trying to isolate.
    """
    lines: list[str] = []

    def put(text: str, indent: int) -> int:
        lines.append("    " * indent + text)
        return len(lines)

    put(
        f"# inputs: {', '.join(inputs)} are integers in "
        f"[{DOMAIN_LO}, {DOMAIN_HI}]",
        0,
    )
    put(f"def check({', '.join(inputs)}):", 0)

    def walk(stmts: list[_Node], indent: int) -> None:
        for s in stmts:
            if s.kind == "if":
                assert s.cond is not None
                s.line = put(f"if {s.cond.text()}:", indent)
                walk(s.body, indent + 1)
                if s.orelse:
                    put("else:", indent)
                    walk(s.orelse, indent + 1)
            else:
                s.line = put(s.text, indent)

    walk(body, 1)
    return lines


def _numbered(lines: list[str]) -> str:
    width = max(2, len(str(len(lines))))
    return "\n".join(f"{i + 1:>{width}}| {t}" for i, t in enumerate(lines))


# --------------------------------------------------------------------------
# Rendering: AST
# --------------------------------------------------------------------------


def _operand_json(value: int | str) -> dict:
    if isinstance(value, str):
        return {"kind": "var", "name": value}
    return {"kind": "const", "value": value}


def _cond_json(c: _Cond) -> dict:
    # `text` is the same string the source encoding shows, so the three
    # conditions differ in what structure they add, not in how they spell the
    # comparison.
    return {
        "op": c.op,
        "left": _operand_json(c.left),
        "right": _operand_json(c.right),
        "text": c.text(),
    }


def _ast_json(body: list[_Node], inputs: list[str]) -> dict:
    def node(s: _Node) -> dict:
        if s.kind == "if":
            assert s.cond is not None
            out = {
                "kind": "if",
                "line": s.line,
                "test": _cond_json(s.cond),
                "body": [node(x) for x in s.body],
            }
            if s.orelse:
                out["orelse"] = [node(x) for x in s.orelse]
            return out
        if s.kind == "return":
            return {"kind": "return", "line": s.line, "text": s.text}
        return {"kind": "statement", "line": s.line, "text": s.text}

    return {
        "function": "check",
        "inputs": {v: {"min": DOMAIN_LO, "max": DOMAIN_HI} for v in inputs},
        "body": [node(s) for s in body],
    }


# --------------------------------------------------------------------------
# Rendering: CFG
# --------------------------------------------------------------------------


@dataclass
class _Block:
    bid: str
    stmts: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class _Edge:
    src: str
    dst: str
    cond: _Cond | None


def _build_cfg(body: list[_Node]) -> tuple[list[_Block], list[_Edge], str]:
    """Basic blocks and edges. Guards on edges are already negated where the
    edge is a false branch, which is the normal form for a CFG.

    A `return` ends its block and leaves the current point unreachable; the next
    statement in that suite then starts a block with no incoming edge. An
    `if/else` whose arms both return leaves its join block in the same state.
    Those two are the only sources of structural unreachability, and they fall
    out of ordinary CFG construction rather than being special-cased.
    """
    blocks: list[_Block] = []
    edges: list[_Edge] = []
    index: dict[str, _Block] = {}

    def new_block() -> str:
        blk = _Block(f"b{len(blocks)}")
        blocks.append(blk)
        index[blk.bid] = blk
        return blk.bid

    def emit(stmts: list[_Node], cur: str | None) -> str | None:
        for s in stmts:
            if cur is None:
                cur = new_block()
            if s.kind == "if":
                assert s.cond is not None
                index[cur].stmts.append((s.line, f"if {s.cond.text()}:"))
                taken = new_block()
                edges.append(_Edge(cur, taken, s.cond))
                taken_out = emit(s.body, taken)
                incoming: list[tuple[str, _Cond | None]] = []
                if taken_out is not None:
                    incoming.append((taken_out, None))
                if s.orelse:
                    other = new_block()
                    edges.append(_Edge(cur, other, _negate(s.cond)))
                    other_out = emit(s.orelse, other)
                    if other_out is not None:
                        incoming.append((other_out, None))
                else:
                    incoming.append((cur, _negate(s.cond)))
                if incoming:
                    join = new_block()
                    for src, guard in incoming:
                        edges.append(_Edge(src, join, guard))
                    cur = join
                else:
                    cur = None
            elif s.kind == "return":
                index[cur].stmts.append((s.line, s.text))
                cur = None
            else:
                index[cur].stmts.append((s.line, s.text))
        return cur

    entry = new_block()
    emit(body, entry)
    return blocks, edges, entry


def _splice_empty(
    blocks: list[_Block], edges: list[_Edge], entry: str
) -> tuple[list[_Block], list[_Edge]]:
    """Drop empty join blocks, keeping every guard.

    Construction leaves one empty block after each `if` that has nothing after
    it. Those are real but uninformative, and leaving them in would make the
    CFG encoding longer than it needs to be for reasons that have nothing to do
    with the question. A block is only removed when its single outgoing edge is
    unconditional, so no conjunction of guards ever has to be collapsed onto one
    edge.
    """
    blocks = list(blocks)
    edges = list(edges)
    changed = True
    while changed:
        changed = False
        for blk in blocks:
            if blk.stmts or blk.bid == entry:
                continue
            outs = [e for e in edges if e.src == blk.bid]
            ins = [e for e in edges if e.dst == blk.bid]
            if len(outs) == 1 and outs[0].cond is None:
                target = outs[0].dst
                edges = [e for e in edges if e is not outs[0]]
                for e in ins:
                    e.dst = target
                blocks = [b for b in blocks if b.bid != blk.bid]
                changed = True
                break
            if not outs and not ins:
                blocks = [b for b in blocks if b.bid != blk.bid]
                changed = True
                break
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[_Edge] = []
    for e in edges:
        key = (e.src, e.dst, e.cond.text() if e.cond else None)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return blocks, unique


def _cfg_json(
    blocks: list[_Block], edges: list[_Edge], entry: str, inputs: list[str], rng
) -> dict:
    """The flat edge list, with block ids relabelled into a shuffled space.

    Construction numbers blocks in execution order, which would let the id alone
    hint at where control reaches. Relabelling and then emitting in id order
    removes that without changing the graph.
    """
    order = list(range(len(blocks)))
    rng.shuffle(order)
    relabel = {blk.bid: f"n{order[i]}" for i, blk in enumerate(blocks)}
    out_blocks = sorted(
        (
            {
                "id": relabel[blk.bid],
                "statements": [{"line": ln, "text": tx} for ln, tx in blk.stmts],
            }
            for blk in blocks
        ),
        key=lambda b: int(b["id"][1:]),
    )
    out_edges = [
        {
            "from": relabel[e.src],
            "to": relabel[e.dst],
            "when": _cond_json(e.cond) if e.cond is not None else None,
        }
        for e in edges
    ]
    rng.shuffle(out_edges)
    return {
        "function": "check",
        "inputs": {v: {"min": DOMAIN_LO, "max": DOMAIN_HI} for v in inputs},
        "entry": relabel[entry],
        "blocks": out_blocks,
        "edges": out_edges,
    }


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def _paths_to(
    blocks: list[_Block], edges: list[_Edge], entry: str, target_line: int
) -> tuple[list[list[_Cond]], str | None]:
    """Every entry-to-target guard conjunction, and the target's block id.

    The graph is acyclic -- the language has no loops -- so plain depth-first
    enumeration terminates and is exhaustive. It is deliberately not the same
    computation as the construction: the construction knows which guards it
    nested around the target, while this walks the graph and collects whatever
    it finds, including the guards of sibling branches completed before the
    target is reached.
    """
    holding = [blk.bid for blk in blocks if any(ln == target_line for ln, _ in blk.stmts)]
    if len(holding) > 1:
        raise AssertionError(f"line {target_line} appears in blocks {holding}")
    if not holding:
        return [], None
    holder = holding[0]

    out_edges: dict[str, list[_Edge]] = {}
    for e in edges:
        out_edges.setdefault(e.src, []).append(e)

    found: list[list[_Cond]] = []

    def walk(bid: str, guards: list[_Cond]) -> None:
        if bid == holder:
            found.append(guards)
            return
        for e in out_edges.get(bid, ()):
            walk(e.dst, (guards + [e.cond]) if e.cond is not None else guards)

    walk(entry, [])
    return found, holder


def _decide(
    blocks: list[_Block], edges: list[_Edge], entry: str, target_line: int
) -> tuple[bool, int, dict[str, int] | None]:
    """Reachability, recomputed from the CFG and the solver alone."""
    paths, holder = _paths_to(blocks, edges, entry, target_line)
    if holder is None:
        raise AssertionError(f"line {target_line} is in no basic block")
    for guards in paths:
        witness = _satisfy(guards)
        if witness is not None:
            return True, len(paths), witness
    return False, len(paths), None


# --------------------------------------------------------------------------
# Generator contract
# --------------------------------------------------------------------------


def difficulty_sweep() -> list[dict]:
    """Depth ladder x constraint_required x encoding, easy to hard.

    `encoding` is not a difficulty axis; it is the paired condition. The three
    entries that share a (depth, live_conditions, constraint_required) triple
    produce the same programs, index for index, which is what makes the paired
    significance test E4 asks for valid. 24 settings at the plan's 500 per
    encoding is 12k calls, well inside the budget.
    """
    return [
        {
            "depth": depth,
            "live_conditions": live,
            "constraint_required": constraint_required,
            "encoding": encoding,
        }
        for depth, live in _LADDER
        for constraint_required in (False, True)
        for encoding in ENCODINGS
    ]


def _program_key(difficulty: dict) -> dict:
    """The difficulty dict without `encoding`.

    Everything about the program is drawn from an RNG seeded with this, so the
    three encodings of instance `i` are the same program.
    """
    return {k: v for k, v in difficulty.items() if k != "encoding"}


def _program_id(key: dict, seed: int, index: int) -> str:
    payload = json.dumps({"g": NAME, "d": key, "s": seed, "i": index}, sort_keys=True)
    return f"prog-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]:
    depth = int(difficulty["depth"])
    live = int(difficulty["live_conditions"])
    constraint_required = bool(difficulty["constraint_required"])
    encoding = difficulty.get("encoding", "source")
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if live < depth:
        raise ValueError(f"live_conditions={live} must be at least depth={depth}")
    # The cycle core needs `depth` strictly increasing values and the squeeze
    # core needs `depth - 1` narrowings that each leave the range non-empty, so
    # both bottom out at the number of values in the domain. Measured cost per
    # instance, constraint_required=True: 0.06ms at depth 1, 0.13ms at depth 2,
    # 1.6ms at depth 4, 7.3ms at depth 6 -- so the shipped ladder is about 18
    # seconds for its 2000 constraint instances. Past the ladder it climbs with
    # the cost of proving a wider cycle unsatisfiable: 23ms at depth 8, 98ms at
    # depth 10, 292ms at depth 11. The structural conditions stay under 0.15ms
    # throughout.
    if depth > DOMAIN_HI - DOMAIN_LO:
        raise ValueError(
            f"depth={depth} exceeds what a domain of "
            f"{DOMAIN_HI - DOMAIN_LO + 1} values can constrain"
        )
    # Worst case: a cycle core takes `depth` variables, the distractors take
    # `live - depth`, and the if/else device takes one more.
    if live + 1 > len(VAR_POOL):
        raise ValueError(
            f"depth={depth} with live_conditions={live} needs more than the "
            f"{len(VAR_POOL)} variables in the pool"
        )

    key = _program_key(difficulty)
    out: list[Instance] = []
    for i in range(count):
        # Alternating rather than drawn keeps the split exactly 50/50 for any
        # even `count`, so the majority-class baseline is exactly 0.5 in every
        # condition. The index never reaches the model, so the parity cannot
        # leak.
        intended = bool(i % 2)
        rng = rng_for(NAME, key, seed, i)
        built = _build_program(
            depth=depth,
            live_conditions=live,
            constraint_required=constraint_required,
            reachable=intended,
            rng=rng,
        )
        body: list[_Node] = built["body"]

        # Always render first, whatever the encoding: this is what assigns the
        # line numbers that all three encodings quote and the question names.
        source_lines = _render_source(body, built["inputs"])
        target_node = _find_target(body)
        if target_node is None:
            raise AssertionError("program has no target statement")
        target_line = target_node.line

        blocks, edges, entry = _build_cfg(body)
        blocks, edges = _splice_empty(blocks, edges, entry)
        truth, path_count, witness = _decide(blocks, edges, entry, target_line)
        if truth != intended:
            raise AssertionError(
                f"{NAME}: construction asked for reachable={intended} at "
                f"{difficulty}, index {i}, but the solver says {truth}"
            )

        live_actual = _count_live(body)
        if live_actual != live:
            raise AssertionError(
                f"{NAME}: asked for {live} live conditions, built {live_actual}"
            )

        if encoding == "source":
            state: Any = _numbered(source_lines)
        elif encoding == "ast":
            state = _ast_json(body, built["inputs"])
        else:
            order_rng = rng_for(f"{NAME}:cfg-order", key, seed, i)
            state = _cfg_json(blocks, edges, entry, built["inputs"], order_rng)

        question = noul(
            f"Is line {target_line} reachable? Answer yes if some assignment of "
            f"the input variables causes line {target_line} to execute, and no "
            f"if no assignment does."
        )

        state_chars = len(state if isinstance(state, str) else json.dumps(state, sort_keys=True))
        cause = None
        if not truth:
            cause = "constraint" if constraint_required else "structural"

        inst = Instance(
            generator=NAME,
            difficulty=dict(difficulty),
            seed=seed,
            index=i,
            state=state,
            questions={"reachable": question},
            truth={"reachable": truth},
            meta={
                "encoding": encoding,
                "depth": depth,
                # Conditions on the route to the target: enclosing guards plus
                # branches completed before it.
                "live_conditions": live,
                "constraint_required": constraint_required,
                "reachable": truth,
                "line_count": len(source_lines),
                # The cheap baseline's feature. It is uninformative by
                # construction, and the plan requires reporting it anyway.
                "statement_count": _statement_count(body),
                # Also a candidate cheap feature, and also balanced across the
                # two answers; carried so the analysis can confirm that offline.
                "return_count": _return_count(body),
                "target_line": target_line,
                # Join key for the paired test across the three encodings.
                "program_id": _program_id(key, seed, i),
                # Everything below is solver output kept for analysis, not a
                # feature any baseline is allowed to use: path_count and
                # unreachable_cause would each be a perfect oracle.
                #
                # `core_family` is not a solver output, but read the warning in
                # `_core_alldiff`: on the instances where it reads "alldiff" the
                # answer is legible from the window width in one step, with no
                # dependence on depth. That is 1/3 of the depth-6 constraint
                # instances and nothing at the other depths, so a per-depth
                # curve that pools the families understates depth 6. Split by
                # `core_family`, or read the accuracy `_check_shortcuts` prints
                # for that rule and subtract it.
                "unreachable_cause": cause,
                "core_family": built["core_family"],
                "device": built["device"],
                "distractor_conditions": built["distractors"],
                "block_count": len(blocks),
                "edge_count": len(edges),
                "path_count": path_count,
                "variable_count": len(built["inputs"]),
                "state_chars": state_chars,
                "witness": witness,
            },
        )
        inst.validate()
        out.append(inst)
    return out


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------


def _run(stmts: Iterable[_Node], env: dict[str, int]) -> tuple[bool, bool]:
    """Interpret the program under one input assignment.

    Returns (the target executed, control returned). This is an operational
    semantics for the little language, written against the AST and owing nothing
    to the CFG or the solver. It turns a satisfying assignment into a direct
    proof of reachability: if the interpreter reaches the target statement under
    the witness, the line is reachable no matter what the rest of this module
    believes. Assignments to `total` and `scratch` are skipped because nothing
    branches on them.
    """
    for s in stmts:
        if s.kind == "return":
            return False, True
        if s.kind == "if":
            assert s.cond is not None
            hit, returned = _run(s.body if s.cond.holds(env) else s.orelse, env)
            if hit:
                return True, returned
            if returned:
                return False, True
        elif s.is_target:
            return True, False
    return False, False


def _satisfy_brute(conds: list[_Cond]) -> dict[str, int] | None:
    """Reference decision by enumeration. Exponential, so small cases only."""
    import itertools

    names = _vars_of(conds)
    values = range(DOMAIN_LO, DOMAIN_HI + 1)
    for combo in itertools.product(values, repeat=len(names)):
        env = dict(zip(names, combo))
        if all(c.holds(env) for c in conds):
            return env
    return None


def _satisfy_minisat(conds: list[_Cond]) -> bool:
    """Independent decision via minisat over a one-hot encoding of the domains."""
    from pysat.formula import IDPool
    from pysat.solvers import Minisat22

    pool = IDPool()
    names = _vars_of(conds)
    values = list(range(DOMAIN_LO, DOMAIN_HI + 1))
    cnf: list[list[int]] = []
    for v in names:
        lits = [pool.id((v, d)) for d in values]
        cnf.append(lits)
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                cnf.append([-lits[i], -lits[j]])
    for c in conds:
        if c.binary:
            for d1 in values:
                for d2 in values:
                    if not _OPS[c.op](d1, d2):
                        cnf.append([-pool.id((c.left, d1)), -pool.id((c.right, d2))])
        else:
            for d in values:
                if not _OPS[c.op](d, c.right):
                    cnf.append([-pool.id((c.left, d))])
    with Minisat22(bootstrap_with=cnf) as solver:
        return bool(solver.solve())


def _check_solver() -> None:
    hand = [
        ([], True),
        ([_Cond("<", "x", 3), _Cond(">", "x", 5)], False),
        ([_Cond("<", "x", 6), _Cond(">", "x", 3)], True),
        ([_Cond(">", "x", DOMAIN_HI)], False),
        ([_Cond(">", "x", DOMAIN_HI - 1)], True),
        ([_Cond("<", "x", "y"), _Cond("<", "y", "x")], False),
        ([_Cond("<", "x", "y"), _Cond("<", "y", "z"), _Cond("<", "z", "x")], False),
        ([_Cond("<", "x", "y"), _Cond("<", "y", "z"), _Cond(">", "z", "x")], True),
        (
            [_Cond("<", v, 2) for v in "xyz"]
            + [_Cond("!=", "x", "y"), _Cond("!=", "x", "z"), _Cond("!=", "y", "z")],
            False,
        ),
        (
            [_Cond("<", v, 3) for v in "xyz"]
            + [_Cond("!=", "x", "y"), _Cond("!=", "x", "z"), _Cond("!=", "y", "z")],
            True,
        ),
        ([_Cond("==", "x", 4), _Cond("!=", "x", 4)], False),
        ([_Cond(">=", "x", 11), _Cond("<=", "x", 11)], True),
    ]
    for conds, expected in hand:
        got = _satisfy(conds)
        assert (got is not None) == expected, f"solver wrong on {[c.text() for c in conds]}"
        if got is not None:
            assert all(c.holds(got) for c in conds), f"bad witness {got}"
        assert _satisfy_brute(conds) is not None or not expected
        assert _satisfy_minisat(conds) == expected, "minisat disagrees on a hand case"
    print(f"solver: {len(hand)} hand-written cases agree with brute force and minisat")


# --------------------------------------------------------------------------
# Shortcut audit
# --------------------------------------------------------------------------
#
# Rules that try to answer the question without doing the work. They are kept
# next to the generator rather than in the analysis because they are a property
# of the construction: if one of them scores well, the condition is not
# measuring what its name says, and that has to surface at generation time.


def _conds_in_ast(nodes: list[dict]) -> list[_Cond]:
    """Pull the conditions back out of the published AST encoding.

    Read from the published state rather than from the node tree, because a
    shortcut probe is only interesting if it sees exactly what the model sees.
    """
    out: list[_Cond] = []
    for n in nodes:
        if n["kind"] != "if":
            continue
        test = n["test"]
        right = test["right"]
        out.append(
            _Cond(
                test["op"],
                test["left"]["name"],
                right["name"] if right["kind"] == "var" else right["value"],
            )
        )
        out += _conds_in_ast(n["body"]) + _conds_in_ast(n.get("orelse", []))
    return out


def _probe_binary_direction(conds: list[_Cond]) -> bool | None:
    """Do the order comparisons between variables point every which way?

    `a < b < c < ... < a` closes a cycle and is unsatisfiable; flipping the last
    relation leaves it satisfiable. Written all from the same end, that makes
    the operators alone the answer, and counting them beats following the chain.
    `_core_cycle` mirrors each relation on a coin flip so this lands at chance.
    """
    order = [c.op for c in conds if c.binary and c.op in ("<", "<=", ">", ">=")]
    if not order:
        return None
    return len({op[0] for op in order}) > 1


def _probe_domain_edge(conds: list[_Cond]) -> bool | None:
    """Is there a condition no value in the declared range satisfies?

    Fires only when it finds one, and then the target is dead. At depth 1 this
    is the whole question and the rule is expected to answer it; past depth 1 no
    single condition may be unsatisfiable on its own, so it must never fire.
    """
    for c in conds:
        if c.binary:
            continue
        if not any(_OPS[c.op](d, c.right) for d in range(DOMAIN_LO, DOMAIN_HI + 1)):
            return False
    return None


def _probe_alldiff_width(conds: list[_Cond]) -> bool | None:
    """How many values does the tightest bound leave the variables that differ?

    Three variables required to be pairwise distinct fit in a window of three
    and not in a window of two, so this answers every `alldiff` instance in one
    step, at any depth. It is the family's irreducible limit rather than a
    spelling accident -- see `_core_alldiff` -- and it is measured here so the
    number is on the record instead of being discovered during analysis.
    """
    differ = {c.left for c in conds if c.binary and c.op == "!="}
    differ |= {c.right for c in conds if c.binary and c.op == "!="}  # type: ignore[misc]
    if len(differ) < 3:
        return None
    widths = [
        sum(1 for d in range(DOMAIN_LO, DOMAIN_HI + 1) if _OPS[c.op](d, c.right))
        for c in conds
        if not c.binary and c.left in differ
    ]
    if not widths:
        return None
    return min(widths) >= 3


_PROBES = {
    "binary-direction": _probe_binary_direction,
    "domain-edge": _probe_domain_edge,
    "alldiff-width": _probe_alldiff_width,
}

# Chance is 0.5 by construction. `binary-direction` fires on the cycle
# instances, which is around 220 of the 640 constraint instances drawn below,
# so the standard error on its accuracy is about 0.033 and this tolerance is
# four and a half of them: wide enough never to flake, narrow enough that a
# rule returning to the 1.000 it once scored fails the check.
_CHANCE_TOLERANCE = 0.15

_AUDIT_COUNT = 160


def _check_shortcuts(count: int = _AUDIT_COUNT) -> None:
    """Score the cheap rules that answer without doing the work.

    Two of these three scored 1.000 before they were measured. The
    operator-direction rule answered every `cycle` instance at every depth --
    half the constraint instances at depths 2 and 4 -- which would have
    flattened the difficulty curve this generator exists to produce, and every
    other assertion in the self-check passed while it did. So the rules are
    scored here rather than argued about in a docstring.

    What is asserted: the two spelling leaks are back at chance, no rule fires
    on a structural instance, and nothing but `alldiff` answers to the width
    rule. What is only printed: the width rule itself, which is the `alldiff`
    family's irreducible limit.
    """
    rows: list[tuple] = []
    pooled: dict[str, list[tuple[bool, bool]]] = {name: [] for name in _PROBES}

    for depth, live in _LADDER:
        for constraint_required in (True, False):
            difficulty = {
                "depth": depth,
                "live_conditions": live,
                "constraint_required": constraint_required,
                "encoding": "ast",
            }
            batch = generate(difficulty=difficulty, seed=20260919, count=count)

            # Not a scored rule but an invariant, and the reason `_at_least`
            # and `_at_most` exist: no condition may compare against a number
            # the inputs cannot take. Such a literal is legible as impossible
            # without reading the comparison, and only the unreachable variant
            # ever needs one.
            for inst in batch:
                for cond in _conds_in_ast(inst.state["body"]):
                    if cond.binary:
                        continue
                    assert DOMAIN_LO <= cond.right <= DOMAIN_HI, (  # type: ignore[operator]
                        f"`{cond.text()}` at {difficulty} puts a literal outside "
                        f"the declared input range [{DOMAIN_LO}, {DOMAIN_HI}]"
                    )

            for name, probe in _PROBES.items():
                hits: list[tuple[bool, bool]] = []
                families: set[str | None] = set()
                for inst in batch:
                    pred = probe(_conds_in_ast(inst.state["body"]))
                    if pred is None:
                        continue
                    hits.append((pred, inst.truth["reachable"]))
                    families.add(inst.meta["core_family"])

                if not constraint_required:
                    # Structural programs branch on `_independent_cond`s only:
                    # no relation between two variables, and none of them dead
                    # on its own. Every rule here should find nothing to read.
                    assert not hits, f"{name} fired on a structural instance at {difficulty}"
                    continue

                if name == "domain-edge":
                    # The rule claims it found a condition that cannot hold
                    # anywhere in the declared range. Wherever it fires the
                    # target must in fact be dead -- otherwise the core is
                    # broken, not merely legible.
                    assert all(not truth for _, truth in hits), (
                        f"domain-edge fired on a reachable instance at {difficulty}"
                    )
                    assert depth == 1 or not hits, (
                        f"a condition at {difficulty} is unsatisfiable on its own; "
                        f"past depth 1 the core must need the whole conjunction"
                    )
                if name == "alldiff-width":
                    assert families <= {"alldiff"}, (
                        f"alldiff-width fired outside the alldiff family: {families}"
                    )

                pooled[name] += hits
                if hits:
                    rows.append(
                        (name, depth, len(hits), len(hits) / len(batch),
                         sum(p == t for p, t in hits) / len(hits))
                    )

    direction = pooled["binary-direction"]
    assert len(direction) >= 100, (
        f"binary-direction fired only {len(direction)} times; too few to score"
    )
    accuracy = sum(p == t for p, t in direction) / len(direction)
    assert abs(accuracy - 0.5) <= _CHANCE_TOLERANCE, (
        f"binary-direction scores {accuracy:.3f} over {len(direction)} cycle "
        f"instances -- it is reading the answer off the operators rather than "
        f"following the chain"
    )

    verdicts = {
        "binary-direction": "at chance",
        "domain-edge": "the depth-1 floor, by design",
        "alldiff-width": "irreducible, see _core_alldiff",
    }
    head = (f"{'shortcut rule':<17} {'depth':>5} {'fires':>6} {'of batch':>9} "
            f"{'accuracy':>9}  verdict")
    print(head)
    print("-" * len(head))
    for name, depth, n, coverage, acc in rows:
        print(f"{name:<17} {depth:>5} {n:>6} {coverage:>8.0%} {acc:>9.3f}  "
              f"{verdicts[name]}")
    print(f"{'binary-direction':<17} {'all':>5} {len(direction):>6} "
          f"{'':>8} {accuracy:>9.3f}  pooled, the asserted number")


def _self_check() -> None:
    import itertools
    import random
    import statistics

    _check_solver()
    _check_shortcuts()
    print()

    per_program: dict[str, dict] = {}
    rows: list[tuple] = []
    checked_conjunctions = 0
    ran_witness = 0
    ran_exhaustive = 0
    ran_sampled = 0
    matched_pairs = 0
    sampler = random.Random(0)
    count = 12

    for difficulty in difficulty_sweep():
        seed = 20260919
        batch = generate(difficulty=difficulty, seed=seed, count=count)
        again = generate(difficulty=difficulty, seed=seed, count=count)
        assert json.dumps([x.to_json() for x in batch], sort_keys=True) == json.dumps(
            [x.to_json() for x in again], sort_keys=True
        ), f"not deterministic at {difficulty}"

        truths = [inst.truth["reachable"] for inst in batch]
        assert sum(truths) * 2 == len(truths), f"not balanced at {difficulty}"

        for inst in batch:
            meta = inst.meta
            assert meta["live_conditions"] == difficulty["live_conditions"]
            assert meta["depth"] == difficulty["depth"]
            assert meta["target_line"] <= meta["line_count"]
            assert meta["path_count"] <= 4096, "path enumeration is too wide"

            if meta["reachable"]:
                assert meta["witness"] is not None
                assert meta["unreachable_cause"] is None
            else:
                assert meta["witness"] is None
                expected = "constraint" if difficulty["constraint_required"] else "structural"
                assert meta["unreachable_cause"] == expected

            # The two failure modes must stay separable: a structural instance
            # is dead because no path exists, a constraint instance because
            # every path's guards contradict.
            if difficulty["constraint_required"]:
                assert meta["path_count"] > 0
            elif not meta["reachable"]:
                assert meta["path_count"] == 0

            if difficulty["encoding"] == "source":
                assert isinstance(inst.state, str)
                numbered = inst.state.splitlines()[meta["target_line"] - 1]
                assert numbered.lstrip().startswith(f"{meta['target_line']}|")
            else:
                assert isinstance(inst.state, dict)
                assert inst.state["inputs"]

            # Paired across encodings: same program, same question, same truth.
            slot = per_program.setdefault(meta["program_id"], {})
            slot[difficulty["encoding"]] = (
                inst.truth["reachable"],
                meta["target_line"],
                meta["line_count"],
                meta["statement_count"],
                list(inst.questions),
                inst.questions["reachable"]["question"],
            )
            slot.setdefault("_chars", {})[difficulty["encoding"]] = meta["state_chars"]

        # Re-decide from the unspliced CFG, so the empty-block cleanup cannot
        # be what makes a line look dead.
        for inst in batch:
            if inst.meta["encoding"] != "source":
                continue
            rng = rng_for(NAME, _program_key(difficulty), seed, inst.index)
            built = _build_program(
                depth=difficulty["depth"],
                live_conditions=difficulty["live_conditions"],
                constraint_required=difficulty["constraint_required"],
                reachable=inst.truth["reachable"],
                rng=rng,
            )
            _render_source(built["body"], built["inputs"])
            blocks, edges, entry = _build_cfg(built["body"])
            raw, _, _ = _decide(blocks, edges, entry, inst.meta["target_line"])
            assert raw == inst.truth["reachable"], "splicing changed reachability"

            paths, _ = _paths_to(blocks, edges, entry, inst.meta["target_line"])
            for guards in paths[:4]:
                mine = _satisfy(guards) is not None
                assert mine == _satisfy_minisat(guards), "minisat disagrees on a path"
                checked_conjunctions += 1

            # Past depth 1, a constraint core has to be a real conjunction: no
            # single condition in the program may be unsatisfiable on its own,
            # or the instance is really a depth-1 instance with extra
            # conditions nested around it.
            if difficulty["constraint_required"] and difficulty["depth"] >= 2:
                for c in _all_conds(built["body"]):
                    assert _satisfy([c]) is not None, (
                        f"{c.text()} is unsatisfiable alone at {difficulty}"
                    )

            # The matched-pair claim, checked rather than asserted in prose:
            # build the same instance with the opposite answer and confirm it is
            # the same size. In the structural conditions nothing about the
            # conditions depends on the answer, so the two variants must hold
            # exactly the same statements, only in a different order.
            twin = _build_program(
                depth=difficulty["depth"],
                live_conditions=difficulty["live_conditions"],
                constraint_required=difficulty["constraint_required"],
                reachable=not inst.truth["reachable"],
                rng=rng_for(NAME, _program_key(difficulty), seed, inst.index),
            )
            twin_lines = _render_source(twin["body"], twin["inputs"])
            assert len(twin_lines) == inst.meta["line_count"], "twin has a different length"
            assert _statement_count(twin["body"]) == inst.meta["statement_count"]
            assert _return_count(twin["body"]) == inst.meta["return_count"]
            assert twin["device"] == inst.meta["device"]
            assert twin["core_family"] == inst.meta["core_family"]
            if not difficulty["constraint_required"]:
                assert sorted(_all_texts(twin["body"])) == sorted(
                    _all_texts(built["body"])
                ), "the two answers are not built from the same statements"
            matched_pairs += 1

            # Run the program. A reachable instance's witness has to actually
            # execute the target under the interpreter, which proves the answer
            # without reference to the CFG or the solver. An unreachable one is
            # checked exhaustively where the input space is small enough and
            # sampled otherwise -- sampling is not a proof, but it would catch a
            # systematic error in either the construction or the decision.
            inputs = built["inputs"]
            values = range(DOMAIN_LO, DOMAIN_HI + 1)
            if inst.truth["reachable"]:
                env = dict(inst.meta["witness"] or {})
                missing = [v for v in inputs if v not in env]
                assert not missing, f"witness leaves {missing} unassigned"
                hit, _ = _run(built["body"], env)
                assert hit, f"witness {env} does not execute line {inst.meta['target_line']}"
                ran_witness += 1
            elif len(inputs) <= 3:
                for combo in itertools.product(values, repeat=len(inputs)):
                    hit, _ = _run(built["body"], dict(zip(inputs, combo)))
                    assert not hit, f"{dict(zip(inputs, combo))} executes a dead line"
                ran_exhaustive += 1
            else:
                for _ in range(200):
                    env = {v: sampler.choice(values) for v in inputs}
                    hit, _ = _run(built["body"], env)
                    assert not hit, f"{env} executes a dead line"
                ran_sampled += 1

        rows.append(
            (
                difficulty["depth"],
                difficulty["live_conditions"],
                difficulty["constraint_required"],
                difficulty["encoding"],
                len(batch),
                sum(truths) / len(truths),
                statistics.mean(i.meta["line_count"] for i in batch),
                statistics.mean(i.meta["statement_count"] for i in batch),
                statistics.mean(i.meta["path_count"] for i in batch),
                round(statistics.mean(i.meta["state_chars"] for i in batch)),
            )
        )

    paired = 0
    for pid, slot in per_program.items():
        present = {k: v for k, v in slot.items() if k in ENCODINGS}
        assert len(present) == 3, f"{pid} appeared in {sorted(present)} only"
        values = list(present.values())
        assert all(v == values[0] for v in values), f"{pid} differs across encodings"
        chars = slot["_chars"]
        assert len({round(c) for c in chars.values()}) > 1, "encodings are the same size"
        paired += 1

    print(f"paired: {paired} programs appear identically in all three encodings")
    print(f"solver: {checked_conjunctions} generated path conjunctions agree with minisat")
    print(f"matched: {matched_pairs} instances are the same size as their opposite-answer twin")
    print(
        f"interpreter: {ran_witness} witnesses execute the target; "
        f"{ran_exhaustive} dead lines proved dead over the whole input space, "
        f"{ran_sampled} over 200 sampled inputs each"
    )
    print()
    head = (
        f"{'depth':>5} {'live':>5} {'constr':>7} {'encoding':>9} {'n':>4} "
        f"{'P(reach)':>9} {'lines':>7} {'stmts':>7} {'paths':>8} {'chars':>7}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r[0]:>5} {r[1]:>5} {str(r[2]):>7} {r[3]:>9} {r[4]:>4} "
            f"{r[5]:>9.2f} {r[6]:>7.1f} {r[7]:>7.1f} {r[8]:>8.1f} {r[9]:>7}"
        )


if __name__ == "__main__":
    _self_check()
