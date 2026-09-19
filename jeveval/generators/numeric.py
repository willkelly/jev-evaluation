"""Small formal tasks with exact answers: numeric, spatial and symbolic.

E9 asks two of its seven questions here. The first is numeric and spatial --
coordinate arithmetic, relative position, simple geometry, counting objects in a
structured scene. The plan flags that one as genuinely open: it previously
asserted weakness at arithmetic from the marketing framing, had no evidence, and
retracted. Training exclusively on synthetic data is exactly how you would
cheaply produce large amounts of structured numeric reasoning, so the sweep here
is built to discriminate rather than to confirm. The second is symbolic input --
s-expressions, ASTs, hexdumps, bit patterns, DIMACS -- with questions that have
formal answers.

Both live in one module because they are the same kind of object: a small state
with an answer that is exact by construction, a size knob, and a set of wrong
answers chosen to name a specific mistake. E9 sweeps them as one ladder of
formal micro-domains, and `DOMAIN` says which of the two a family belongs to.

How these are built to discriminate
-----------------------------------
Every choice family offers the true answer alongside distractors that are each
the result of one named slip: reading a hexdump field big-endian instead of
little-endian, evaluating an s-expression left to right instead of by its
nesting, counting a colour while ignoring a shape, dropping the last move of a
walk. The slip that produced each option is recorded in
`meta["slips"]`, so a wrong answer says *which* step failed rather than only
that the instance was missed. A model that lands on `flat_left_to_right` on
every deep s-expression is not bad at arithmetic; it is ignoring structure, and
that is a different finding with a different fix.

The cheap deterministic baseline the plan requires is recorded per instance as
`meta["baseline"]["prediction"]` -- the answer a surface heuristic gives, in the
same form as the truth, so heuristic accuracy is `prediction == truth` counted
offline from the log and nothing else. The heuristics are deliberately surface
ones (nearest option to the start, one-axis rectangle overlap, colour count
ignoring shape, single byte instead of a 16-bit field). Several of these domains
also have an exact decision procedure in twenty lines -- rectangle overlap is
four comparisons -- so a model below 1.00 on them sits in the plan's Bad tier by
definition. That is not the interesting reading. The interesting reading is
where accuracy falls off as the size knob turns, which is why every family is
swept rather than sampled at one point.

Where content is deliberately shared
------------------------------------
Two pairs of conditions must see the *same* content through different surfaces,
or the comparison between them measures the generator rather than the model:

  * `sexpr` and `ast` at the same size and index are the same expression, once
    as `(+ (* 3 4) 7)` and once as a JSON tree.
  * `count_scene` at the same size and index is the same scene in every
    language; only the lexicon and the field labels change.

So the per-instance RNG is keyed on a *content* difficulty that drops `family`
for the expression pair and drops `language` for the scene, rather than on the
full difficulty dict. `Instance.instance_id` still uses the full dict, so the
variants remain separately addressable in the log.

The scene is rendered as labelled fields (`color=red, shape=triangle`) rather
than as a sentence, because a sentence would need gender and number agreement in
Spanish, French and German and the comparison would then partly measure how well
the harness inflects. Fields keep the change to the lexicon, which is what the
language arm is about.

Run the self-check with:

    python -m jeveval.generators.numeric
"""

from __future__ import annotations

import random
from typing import Any

from ..instances import Instance, choice, rng_for, shuffled_options
from . import sat3

NAME = "numeric"

# One question per instance; experiments join on this key.
QUESTION_KEY = "answer"

NUMERIC_FAMILIES: tuple[str, ...] = ("coord_walk", "nearest", "overlap", "count_scene")
SYMBOLIC_FAMILIES: tuple[str, ...] = ("sexpr", "ast", "hexdump", "bits", "dimacs")
FAMILIES: tuple[str, ...] = NUMERIC_FAMILIES + SYMBOLIC_FAMILIES

DOMAIN: dict[str, str] = {
    **{f: "numeric" for f in NUMERIC_FAMILIES},
    **{f: "symbolic" for f in SYMBOLIC_FAMILIES},
}

# The size knob per family, easy to hard. Four rungs for the numeric families,
# which carry the plan's open question and need the finer curve; three for the
# symbolic ones.
SIZES: dict[str, tuple[int, ...]] = {
    "coord_walk": (2, 4, 8, 16),        # moves in the walk
    "nearest": (4, 8, 16, 32),          # objects in the scene
    "overlap": (4, 8, 16, 32),          # rectangles in the scene
    "count_scene": (4, 8, 16, 32),      # objects in the scene
    "sexpr": (2, 5, 9),                 # operators in the expression
    "ast": (2, 5, 9),                   # the same expressions, as JSON
    "hexdump": (16, 64, 256),           # bytes in the record
    "bits": (8, 24, 64),                # bits in the word
    "dimacs": (10, 30, 80),             # clauses in the formula
}

# Languages for the `count_scene` robustness arm. English is the control.
LANGUAGES: tuple[str, ...] = ("en", "es", "fr", "de")

# Five options on the families whose answer is an unbounded value, so random is
# 0.20 there and stated rather than inferred.
N_VALUE_OPTIONS = 5

_DIMACS_VARS = 20


def difficulty_sweep() -> list[dict]:
    """Every family at every size, numeric families first, easy to hard.

    The language arm is not in the default sweep: it is the same `count_scene`
    conditions in another lexicon, and putting it here would make the numeric
    curve four times as long for a comparison that belongs to E9's format
    condition. `language_sweep` returns it.
    """
    return [
        {"family": family, "size": size}
        for family in FAMILIES
        for size in SIZES[family]
    ]


def language_sweep(size: int = 16) -> list[dict]:
    """One `count_scene` condition per language, at one size.

    Held at a single size because the question is whether the same scene in
    another language gets the same answer, not how counting scales -- the sweep
    above already answers that.
    """
    if size not in SIZES["count_scene"]:
        raise ValueError(f"size {size} is not a count_scene size; expected {SIZES['count_scene']}")
    return [{"family": "count_scene", "size": size, "language": lang} for lang in LANGUAGES]


def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]:
    """`count` instances at one difficulty setting."""
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    d = _normalize(difficulty)
    return [_instance(d, seed, index) for index in range(count)]


# --------------------------------------------------------------------------
# Difficulty handling
# --------------------------------------------------------------------------


def _normalize(difficulty: dict) -> dict:
    d = dict(difficulty)
    family = d.pop("family", None)
    size = d.pop("size", None)
    language = d.pop("language", "en")
    if d:
        raise ValueError(f"{NAME}: unknown difficulty keys {sorted(d)}")
    if family not in FAMILIES:
        raise ValueError(f"{NAME}: unknown family {family!r}; expected one of {FAMILIES}")
    if size is None:
        raise ValueError(f"{NAME}: difficulty needs a size")
    size = int(size)
    if size < 2:
        raise ValueError(f"{NAME}: size must be >= 2, got {size}")
    if language not in LANGUAGES:
        raise ValueError(f"{NAME}: unknown language {language!r}; expected one of {LANGUAGES}")
    if language != "en" and family != "count_scene":
        raise ValueError(f"{NAME}: only count_scene is translated; {family} is English only")
    out = {"family": family, "size": size}
    if language != "en":
        out["language"] = language
    return out


# Families whose content is shared with another family, keyed to the name the
# RNG uses instead of their own.
_CONTENT_FAMILY = {"sexpr": "expr", "ast": "expr"}


def _content_rng(difficulty: dict, seed: int, index: int) -> random.Random:
    """The RNG for the instance's *content*, ignoring the surface it is shown in.

    Dropping `language`, and collapsing `sexpr` and `ast` onto one content name,
    is what makes those variants the same problem in two surfaces. Without it the
    format comparison would be over different instances and would measure noise.
    """
    key = {
        "family": _CONTENT_FAMILY.get(difficulty["family"], difficulty["family"]),
        "size": difficulty["size"],
    }
    return rng_for(NAME, key, seed, index)


def _instance(difficulty: dict, seed: int, index: int) -> Instance:
    family = difficulty["family"]
    rng = _content_rng(difficulty, seed, index)
    builder = _BUILDERS[family]
    state, question, truth, meta = builder(rng, difficulty, index)
    # Shuffled here rather than in each builder, which construct their options
    # truth-first. The builders leave the RNG in the same state across the
    # surfaces that share content, so `sexpr` and `ast` -- and every language of
    # one scene -- also share the presented order.
    question = shuffled_options(question, rng)

    meta = {
        "family": family,
        "domain": DOMAIN[family],
        "size": difficulty["size"],
        "language": difficulty.get("language", "en"),
        "option_order": [o["id"] for o in question["options"]],
        **meta,
    }
    inst = Instance(
        generator=NAME,
        difficulty=dict(difficulty),
        seed=seed,
        index=index,
        state=state,
        questions={QUESTION_KEY: question},
        truth={QUESTION_KEY: truth},
        meta=meta,
    )
    inst.validate()
    return inst


# --------------------------------------------------------------------------
# Option helpers
# --------------------------------------------------------------------------


def _value_options(
    truth: str, slips: list[tuple[str, str]], rng: random.Random, filler: Any
) -> tuple[list[dict], dict[str, str]]:
    """Build an option list from the truth plus named slips.

    `slips` is (slip name, rendered value) in preference order. Duplicates and
    collisions with the truth are dropped -- a slip that happens to land on the
    right answer is not a distractor -- and `filler` supplies further distinct
    values until there are `N_VALUE_OPTIONS` of them, so the random baseline is
    the same at every size.

    Returns the options truth-first and a map from option id to the slip that
    produced it. `_instance` shuffles them before they reach a question, so the
    truth-first order never leaves this module.
    """
    ids = [truth]
    origin = {truth: "truth"}
    for name, value in slips:
        if value in origin:
            continue
        ids.append(value)
        origin[value] = name
    guard = 0
    while len(ids) < N_VALUE_OPTIONS:
        guard += 1
        if guard > 1000:
            raise AssertionError(f"{NAME}: could not find {N_VALUE_OPTIONS} distinct options")
        value = filler(rng)
        if value in origin:
            continue
        ids.append(value)
        origin[value] = "filler"
    return [{"id": i, "label": i} for i in ids], origin


def _count_options(k: int) -> list[dict]:
    """Options 0..k, for the families whose answer is a small count."""
    return [{"id": str(i), "label": str(i)} for i in range(k + 1)]


# --------------------------------------------------------------------------
# Numeric and spatial families
# --------------------------------------------------------------------------

_DIRECTIONS = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}
_OPPOSITE = {"east": "west", "west": "east", "north": "south", "south": "north"}


def _point(x: int, y: int) -> str:
    return f"({x}, {y})"


def _walk(start: tuple[int, int], moves: list[tuple[str, int]]) -> tuple[int, int]:
    x, y = start
    for direction, distance in moves:
        dx, dy = _DIRECTIONS[direction]
        x += dx * distance
        y += dy * distance
    return x, y


def _build_coord_walk(rng: random.Random, difficulty: dict, index: int):
    """Coordinate arithmetic: apply a sequence of moves, say where you end up.

    The distractors are the results of four specific arithmetic slips, so a wrong
    answer names the slip. The cheap baseline is "pick the option nearest the
    starting point", which needs no arithmetic at all.
    """
    size = difficulty["size"]
    start = (rng.randint(-20, 20), rng.randint(-20, 20))
    moves = [
        (rng.choice(list(_DIRECTIONS)), rng.randint(1, 9)) for _ in range(size)
    ]
    end = _walk(start, moves)

    drop_last = _walk(start, moves[:-1])
    flip_at = rng.randrange(size)
    flipped = list(moves)
    flipped[flip_at] = (_OPPOSITE[moves[flip_at][0]], moves[flip_at][1])
    sign_flip = _walk(start, flipped)
    axis_swap = (end[1], end[0])
    off_by_one = (end[0] + rng.choice((-1, 1)), end[1])

    truth = _point(*end)
    slips = [
        ("dropped_last_move", _point(*drop_last)),
        ("one_move_reversed", _point(*sign_flip)),
        ("axes_transposed", _point(*axis_swap)),
        ("off_by_one", _point(*off_by_one)),
    ]

    def filler(r: random.Random) -> str:
        return _point(end[0] + r.randint(-9, 9), end[1] + r.randint(-9, 9))

    options, origin = _value_options(truth, slips, rng, filler)

    lines = [
        f"A robot starts at {_point(*start)} on a grid where east increases x, "
        "west decreases x, north increases y and south decreases y.",
        "It then makes these moves, in order:",
    ]
    lines += [f"{i + 1}. move {d} {direction}" for i, (direction, d) in enumerate(moves)]
    state = "\n".join(lines)

    # The cheap baseline: no arithmetic, just pick whichever offered endpoint is
    # closest to where the robot started.
    nearest_to_start = min(
        (o["id"] for o in options),
        key=lambda oid: _sq_distance(_parse_point(oid), start),
    )

    meta = {
        "start": list(start),
        "moves": [[d, n] for d, n in moves],
        "true_end": list(end),
        "n_options": len(options),
        "slips": origin,
        "baseline": {
            "prediction": nearest_to_start,
            "total_distance": sum(n for _, n in moves),
        },
    }
    question = choice(
        "Where does the robot end up? Give its final coordinates.", options
    )
    return state, question, truth, meta


def _parse_point(text: str) -> tuple[int, int]:
    x, y = text.strip("()").split(",")
    return int(x), int(y)


def _sq_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _build_nearest(rng: random.Random, difficulty: dict, index: int):
    """Relative position: which of `size` objects is nearest a probe point.

    Points are drawn until the nearest is unique by a clear margin, so the answer
    does not turn on a tie the state cannot settle. The cheap baseline projects
    onto one axis and picks the nearest by x alone, which is a real partial
    predictor and not a solver.
    """
    size = difficulty["size"]
    # Bounded rather than `while True`: the margin condition is easy to satisfy
    # at every size in the sweep, so exhausting this means the condition itself
    # became unsatisfiable and the generator should say so instead of hanging.
    for _attempt in range(200):
        probe = (rng.randint(-40, 40), rng.randint(-40, 40))
        points: dict[tuple[int, int], str] = {}
        while len(points) < size:
            p = (rng.randint(-50, 50), rng.randint(-50, 50))
            # Both guards matter: a repeat draw would otherwise be renamed in
            # place and leave two points sharing an option id.
            if p != probe and p not in points:
                points[p] = f"p{len(points) + 1}"
        coords = list(points)
        ranked = sorted(coords, key=lambda p: _sq_distance(p, probe))
        if _sq_distance(ranked[0], probe) * 1.2 < _sq_distance(ranked[1], probe):
            break
    else:
        raise AssertionError(
            f"{NAME}: no scene of {size} points left a clear nearest point"
        )

    truth = points[ranked[0]]
    options = [{"id": points[p], "label": f"{points[p]} at {_point(*p)}"} for p in coords]
    by_x = min(coords, key=lambda p: (abs(p[0] - probe[0]), points[p]))
    by_manhattan = min(
        coords, key=lambda p: (abs(p[0] - probe[0]) + abs(p[1] - probe[1]), points[p])
    )

    lines = [f"Points on a plane, and a probe at {_point(*probe)}:"]
    lines += [f"{points[p]} at {_point(*p)}" for p in coords]
    state = "\n".join(lines)

    meta = {
        "probe": list(probe),
        "n_options": len(options),
        "true_squared_distance": _sq_distance(ranked[0], probe),
        "margin_ratio": _sq_distance(ranked[1], probe) / max(1, _sq_distance(ranked[0], probe)),
        "slips": {points[by_x]: "nearest_by_x_only", points[by_manhattan]: "nearest_by_manhattan"},
        "baseline": {
            # One axis only: cheap, deterministic, and wrong often enough to be
            # worth reporting against.
            "prediction": points[by_x],
            "manhattan_agrees": points[by_manhattan] == truth,
        },
    }
    question = choice(
        f"Which point is closest to the probe at {_point(*probe)} in straight-line "
        "distance?",
        options,
    )
    return state, question, truth, meta


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Do two axis-aligned rectangles share area? (x0, y0, x1, y1), x0<x1."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _build_overlap(rng: random.Random, difficulty: dict, index: int):
    """Geometry: how many of the other rectangles overlap r1.

    The count is drawn first and the scene is built to it, so the answer is
    spread over the option set instead of piling up at 0 -- otherwise the
    majority-class baseline would carry the condition. Every non-overlapping
    rectangle is a near miss that overlaps r1 in exactly one axis, so the
    one-axis heuristic recorded as the cheap baseline is wrong on all of them.
    """
    size = difficulty["size"]
    others = size - 1
    k = min(others, 6)
    target = rng.randint(0, k)

    r1 = (0, 0, rng.randint(8, 16), rng.randint(8, 16))
    rects = [("r1", r1)]

    def overlapping() -> tuple[int, int, int, int]:
        x0 = rng.randint(r1[0] - 4, r1[2] - 2)
        y0 = rng.randint(r1[1] - 4, r1[3] - 2)
        return (x0, y0, x0 + rng.randint(3, 12), y0 + rng.randint(3, 12))

    def near_miss() -> tuple[int, int, int, int]:
        # Overlaps r1 on one axis and clears it on the other by a small gap.
        if rng.random() < 0.5:
            x0 = rng.randint(r1[0] - 4, r1[2] - 2)
            w = rng.randint(3, 12)
            gap = rng.randint(1, 3)
            if rng.random() < 0.5:
                y0 = r1[3] + gap
            else:
                y0 = r1[1] - gap - rng.randint(3, 12)
            return (x0, y0, x0 + w, y0 + rng.randint(3, 12))
        y0 = rng.randint(r1[1] - 4, r1[3] - 2)
        h = rng.randint(3, 12)
        gap = rng.randint(1, 3)
        if rng.random() < 0.5:
            x0 = r1[2] + gap
        else:
            x0 = r1[0] - gap - rng.randint(3, 12)
        return (x0, y0, x0 + rng.randint(3, 12), y0 + h)

    made = 0
    while made < others:
        want_overlap = made < target
        for _ in range(200):
            candidate = overlapping() if want_overlap else near_miss()
            if _rects_overlap(r1, candidate) is want_overlap:
                break
        else:
            raise AssertionError(f"{NAME}: could not place a rectangle for overlap={want_overlap}")
        made += 1
        rects.append((f"r{made + 1}", candidate))

    true_count = sum(1 for name, r in rects[1:] if _rects_overlap(r1, r))
    if true_count != target:
        raise AssertionError(f"{NAME}: built {true_count} overlaps, wanted {target}")

    x_only = sum(1 for _, r in rects[1:] if r[0] < r1[2] and r1[0] < r[2])

    lines = ["Axis-aligned rectangles, given as (x0, y0) to (x1, y1):"]
    lines += [f"{name}: ({r[0]}, {r[1]}) to ({r[2]}, {r[3]})" for name, r in rects]
    state = "\n".join(lines)

    meta = {
        "n_rectangles": size,
        "true_count": true_count,
        "n_options": k + 1,
        "baseline": {
            # Checks the x extents and ignores y. Every non-overlapping
            # rectangle here overlaps r1 in one axis, so this over-counts by
            # construction and is a genuine surface heuristic rather than the
            # four-comparison decision procedure.
            "prediction": str(min(x_only, k)),
            "x_overlap_count": x_only,
        },
    }
    question = choice(
        "How many of the other rectangles share area with r1? Touching along an "
        "edge or a corner does not count.",
        _count_options(k),
    )
    return state, question, str(true_count), meta


# Field labels and lexicon per language. Only these words change between the
# language arms; the scene, the query and the answer are identical.
_SCENE_WORDS: dict[str, dict[str, Any]] = {
    "en": {
        "object": "object",
        "color": "color",
        "shape": "shape",
        "position": "position",
        "header": "A scene containing these objects:",
        "question": "How many objects have color={color} and shape={shape}?",
        "colors": {"red": "red", "green": "green", "blue": "blue"},
        "shapes": {"triangle": "triangle", "square": "square", "circle": "circle"},
    },
    "es": {
        "object": "objeto",
        "color": "color",
        "shape": "forma",
        "position": "posicion",
        "header": "Una escena que contiene estos objetos:",
        "question": "¿Cuántos objetos tienen color={color} y forma={shape}?",
        "colors": {"red": "rojo", "green": "verde", "blue": "azul"},
        "shapes": {"triangle": "triangulo", "square": "cuadrado", "circle": "circulo"},
    },
    "fr": {
        "object": "objet",
        "color": "couleur",
        "shape": "forme",
        "position": "position",
        "header": "Une scène contenant ces objets :",
        "question": "Combien d'objets ont couleur={color} et forme={shape} ?",
        "colors": {"red": "rouge", "green": "vert", "blue": "bleu"},
        "shapes": {"triangle": "triangle", "square": "carre", "circle": "cercle"},
    },
    "de": {
        "object": "Objekt",
        "color": "Farbe",
        "shape": "Form",
        "position": "Position",
        "header": "Eine Szene mit diesen Objekten:",
        "question": "Wie viele Objekte haben Farbe={color} und Form={shape}?",
        "colors": {"red": "rot", "green": "gruen", "blue": "blau"},
        "shapes": {"triangle": "Dreieck", "square": "Quadrat", "circle": "Kreis"},
    },
}

_COLORS = ("red", "green", "blue")
_SHAPES = ("triangle", "square", "circle")


def _build_count_scene(rng: random.Random, difficulty: dict, index: int):
    """Counting a conjunction of two attributes over a structured scene.

    The target count is drawn first and the scene built to it, so counts are
    spread over the option set. Distractor objects deliberately share the
    queried colour or the queried shape but not both, which is what makes the
    two disjunctive baselines -- count the colour, count the shape -- wrong
    while leaving them plausible.
    """
    size = difficulty["size"]
    language = difficulty.get("language", "en")
    words = _SCENE_WORDS[language]

    k = min(size, 8)
    target = rng.randint(0, k)
    want_color = rng.choice(_COLORS)
    want_shape = rng.choice(_SHAPES)

    objects: list[tuple[str, str, tuple[int, int]]] = []
    for _ in range(target):
        objects.append((want_color, want_shape, (rng.randint(-30, 30), rng.randint(-30, 30))))
    for _ in range(size - target):
        while True:
            color = rng.choice(_COLORS)
            shape = rng.choice(_SHAPES)
            if (color, shape) != (want_color, want_shape):
                break
        objects.append((color, shape, (rng.randint(-30, 30), rng.randint(-30, 30))))
    rng.shuffle(objects)

    true_count = sum(1 for c, s, _ in objects if c == want_color and s == want_shape)
    if true_count != target:
        raise AssertionError(f"{NAME}: built {true_count} matches, wanted {target}")
    n_color = sum(1 for c, _, _ in objects if c == want_color)
    n_shape = sum(1 for _, s, _ in objects if s == want_shape)

    lines = [words["header"]]
    for i, (color, shape, pos) in enumerate(objects, start=1):
        lines.append(
            f"{words['object']} {i}: {words['color']}={words['colors'][color]}, "
            f"{words['shape']}={words['shapes'][shape]}, "
            f"{words['position']}={_point(*pos)}"
        )
    state = "\n".join(lines)

    meta = {
        "n_objects": size,
        "true_count": true_count,
        "queried_color": want_color,
        "queried_shape": want_shape,
        "n_options": k + 1,
        "baseline": {
            # Counts the colour and forgets the shape: the disjunctive slip.
            "prediction": str(min(n_color, k)),
            "n_same_color": n_color,
            "n_same_shape": n_shape,
        },
    }
    question = choice(
        words["question"].format(
            color=words["colors"][want_color], shape=words["shapes"][want_shape]
        ),
        _count_options(k),
    )
    return state, question, str(true_count), meta


# --------------------------------------------------------------------------
# Symbolic families
# --------------------------------------------------------------------------

_OPS = ("+", "-", "*")


def _random_expr(rng: random.Random, operators: int) -> Any:
    """A binary expression tree with exactly `operators` internal nodes."""
    if operators == 0:
        return rng.randint(1, 9)
    left_ops = rng.randint(0, operators - 1)
    return {
        "op": rng.choice(_OPS),
        "left": _random_expr(rng, left_ops),
        "right": _random_expr(rng, operators - 1 - left_ops),
    }


def _eval_expr(node: Any) -> int:
    if isinstance(node, int):
        return node
    a, b = _eval_expr(node["left"]), _eval_expr(node["right"])
    return a + b if node["op"] == "+" else a - b if node["op"] == "-" else a * b


def _to_sexpr(node: Any) -> str:
    if isinstance(node, int):
        return str(node)
    return f"({node['op']} {_to_sexpr(node['left'])} {_to_sexpr(node['right'])})"


def _to_ast(node: Any) -> Any:
    if isinstance(node, int):
        return {"kind": "literal", "value": node}
    return {
        "kind": "binary",
        "operator": node["op"],
        "left": _to_ast(node["left"]),
        "right": _to_ast(node["right"]),
    }


def _flatten(node: Any) -> list[Any]:
    """Tokens in prefix order, which is what the flat baseline reads."""
    if isinstance(node, int):
        return [node]
    return [node["op"], *_flatten(node["left"]), *_flatten(node["right"])]


def _flat_left_to_right(node: Any) -> int:
    """Evaluate the token stream left to right, ignoring the nesting.

    This is the cheap heuristic for both expression families: take the literals
    in the order they appear and apply the operators in the order they appear.
    On a right-leaning tree it is right; on anything else it is not.
    """
    tokens = _flatten(node)
    literals = [t for t in tokens if isinstance(t, int)]
    ops = [t for t in tokens if isinstance(t, str)]
    value = literals[0]
    for op, operand in zip(ops, literals[1:]):
        value = value + operand if op == "+" else value - operand if op == "-" else value * operand
    return value


def _mutate_one_op(node: Any, rng: random.Random) -> Any:
    """A copy with exactly one operator replaced by a different one."""
    positions: list[list[Any]] = []

    def walk(n: Any) -> Any:
        if isinstance(n, int):
            return n
        copy = {"op": n["op"], "left": walk(n["left"]), "right": walk(n["right"])}
        positions.append(copy)
        return copy

    root = walk(node)
    if positions:
        victim = rng.choice(positions)
        victim["op"] = rng.choice([o for o in _OPS if o != victim["op"]])
    return root


def _mutate_one_literal(node: Any, rng: random.Random) -> Any:
    literals: list[tuple[dict, str]] = []

    def walk(n: Any) -> Any:
        if isinstance(n, int):
            return n
        copy = {"op": n["op"], "left": walk(n["left"]), "right": walk(n["right"])}
        for side in ("left", "right"):
            if isinstance(copy[side], int):
                literals.append((copy, side))
        return copy

    root = walk(node)
    if literals:
        holder, side = rng.choice(literals)
        holder[side] = holder[side] + rng.choice((-1, 1))
    return root


def _build_expression(rng: random.Random, difficulty: dict, index: int):
    """One expression, rendered as an s-expression or as a JSON AST.

    `sexpr` and `ast` share this builder and, because the content RNG collapses
    them onto one name, share the expression too -- so the pair is a clean
    surface comparison at every size.
    """
    family = difficulty["family"]
    size = difficulty["size"]
    tree = _random_expr(rng, size)
    value = _eval_expr(tree)

    flat = _flat_left_to_right(tree)
    op_slip = _eval_expr(_mutate_one_op(tree, rng))
    literal_slip = _eval_expr(_mutate_one_literal(tree, rng))

    truth = str(value)
    slips = [
        ("flat_left_to_right", str(flat)),
        ("one_operator_misread", str(op_slip)),
        ("one_literal_off_by_one", str(literal_slip)),
        ("sign_error", str(-value)),
    ]

    def filler(r: random.Random) -> str:
        return str(value + r.choice([n for n in range(-20, 21) if n != 0]))

    options, origin = _value_options(truth, slips, rng, filler)

    if family == "sexpr":
        state = _to_sexpr(tree)
        prompt = (
            "Evaluate this s-expression. Each list is (operator left right) and "
            "the operators are ordinary integer addition, subtraction and "
            "multiplication. What is its value?"
        )
    else:
        state = _to_ast(tree)
        prompt = (
            "Evaluate the expression this abstract syntax tree describes. A "
            "binary node applies its operator to the values of its left and "
            "right children. What is its value?"
        )

    meta = {
        "n_operators": size,
        "true_value": value,
        "n_options": len(options),
        "slips": origin,
        "sexpr": _to_sexpr(tree),
        "baseline": {"prediction": str(flat), "flat_value": flat},
    }
    return state, choice(prompt, options), truth, meta


def _build_hexdump(rng: random.Random, difficulty: dict, index: int):
    """Read a little-endian 16-bit field out of a hexdump at a stated offset.

    The distractors are the classic misreadings: big-endian, the single byte at
    the offset, and the fields one byte either side. The cheap baseline reads one
    byte, which is what a program that ignores the width does.
    """
    size = difficulty["size"]
    data = bytes(rng.randrange(256) for _ in range(size))
    offset = rng.randrange(0, size - 2)

    little = data[offset] | (data[offset + 1] << 8)
    big = (data[offset] << 8) | data[offset + 1]
    one_byte = data[offset]
    before = (data[offset - 1] | (data[offset] << 8)) if offset >= 1 else little + 257
    after = (
        (data[offset + 1] | (data[offset + 2] << 8))
        if offset + 2 < size
        else little + 513
    )

    truth = str(little)
    slips = [
        ("big_endian", str(big)),
        ("single_byte", str(one_byte)),
        ("offset_minus_one", str(before)),
        ("offset_plus_one", str(after)),
    ]

    def filler(r: random.Random) -> str:
        return str(r.randrange(65536))

    options, origin = _value_options(truth, slips, rng, filler)

    lines = []
    for base in range(0, size, 16):
        chunk = data[base : base + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{base:08x}  {hex_part:<47}  |{text}|")
    state = "\n".join(lines)

    meta = {
        "n_bytes": size,
        "offset": offset,
        "true_value": little,
        "n_options": len(options),
        "slips": origin,
        "baseline": {
            # Reading the two printed bytes in the order they appear, which is
            # what a reader who has not noticed the endianness does. It is right
            # only when the two bytes are equal, so this domain has no cheap
            # heuristic that beats the 0.2 random baseline -- which is itself
            # worth stating rather than leaving the field empty.
            "prediction": str(big),
            "single_byte": one_byte,
            "big_endian": big,
        },
    }
    question = choice(
        f"The dump in the state is a byte record. Reading the two bytes at "
        f"offset 0x{offset:04x} as one unsigned 16-bit little-endian integer "
        "(the first byte is the low half), what is its value?",
        options,
    )
    return state, question, truth, meta


def _build_bits(rng: random.Random, difficulty: dict, index: int):
    """Population count over a bit pattern of `size` bits.

    Counting on a different surface from the Dyck condition: the same underlying
    ability, no nesting. The cheap baseline guesses half the width, which is the
    expected value and therefore right about as often as chance allows.
    """
    size = difficulty["size"]
    # The count is drawn uniformly and the bits laid out to it. Drawing each bit
    # independently would make the popcount binomial, and the majority-class
    # baseline would then beat random by a wide margin at the small widths.
    ones = rng.randint(0, size)
    bits = [1] * ones + [0] * (size - ones)
    rng.shuffle(bits)
    zeros = size - ones

    truth = str(ones)
    slips = [
        ("counted_zeros", str(zeros)),
        ("off_by_one_high", str(ones + 1)),
        ("off_by_one_low", str(max(0, ones - 1))),
        ("half_the_width", str(size // 2)),
    ]

    def filler(r: random.Random) -> str:
        return str(r.randint(0, size))

    options, origin = _value_options(truth, slips, rng, filler)

    grouped = " ".join(
        "".join(str(b) for b in bits[i : i + 8]) for i in range(0, size, 8)
    )
    state = f"A {size}-bit pattern, most significant bit first, in groups of eight:\n{grouped}"

    meta = {
        "n_bits": size,
        "true_count": ones,
        "n_options": len(options),
        "slips": origin,
        "baseline": {"prediction": str(size // 2), "n_zeros": zeros},
    }
    question = choice("How many bits in this pattern are set to 1?", options)
    return state, question, truth, meta


def _build_dimacs(rng: random.Random, difficulty: dict, index: int):
    """Count occurrences of one literal in a DIMACS CNF.

    Deliberately not satisfiability -- E2 owns that question and it is a search
    problem. This is symbolic reading: parse the format, respect the sign, count.
    The cheap baseline ignores the sign, which is the slip the format invites.
    """
    m = difficulty["size"]
    clauses = sat3.random_3sat(_DIMACS_VARS, m, rng)
    state = sat3.to_dimacs(clauses, _DIMACS_VARS)

    var = rng.randint(1, _DIMACS_VARS)
    negative = rng.random() < 0.5
    literal = -var if negative else var

    signed = sum(1 for c in clauses if literal in c)
    opposite = sum(1 for c in clauses if -literal in c)
    either = signed + opposite

    truth = str(signed)
    slips = [
        ("ignored_the_sign", str(either)),
        ("counted_the_other_polarity", str(opposite)),
        ("off_by_one_high", str(signed + 1)),
        ("off_by_one_low", str(max(0, signed - 1))),
    ]

    def filler(r: random.Random) -> str:
        return str(r.randint(0, either + 10))

    options, origin = _value_options(truth, slips, rng, filler)

    sign_text = "negated" if negative else "positive"
    meta = {
        "n_clauses": m,
        "n_vars": _DIMACS_VARS,
        "literal": literal,
        "true_count": signed,
        "n_options": len(options),
        "slips": origin,
        "baseline": {"prediction": str(either), "either_polarity_count": either},
    }
    question = choice(
        f"This is a DIMACS CNF file. In how many clauses does variable {var} "
        f"appear {sign_text} (that is, as the literal {literal})?",
        options,
    )
    return state, question, truth, meta


_BUILDERS = {
    "coord_walk": _build_coord_walk,
    "nearest": _build_nearest,
    "overlap": _build_overlap,
    "count_scene": _build_count_scene,
    "sexpr": _build_expression,
    "ast": _build_expression,
    "hexdump": _build_hexdump,
    "bits": _build_bits,
    "dimacs": _build_dimacs,
}


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------


def _self_check() -> None:
    count = 60
    seed = 4242

    print(f"{'family':<12} {'size':>5} {'random':>7} {'majority':>9} {'cheap':>7}")
    for level in difficulty_sweep():
        batch = generate(difficulty=level, seed=seed, count=count)
        truths = [i.truth[QUESTION_KEY] for i in batch]
        n_options = batch[0].meta["n_options"]
        assert all(i.meta["n_options"] == n_options for i in batch), level
        majority = max(truths.count(t) for t in set(truths)) / len(truths)
        cheap = sum(
            1 for i in batch if i.meta["baseline"]["prediction"] == i.truth[QUESTION_KEY]
        ) / len(batch)
        print(
            f"{level['family']:<12} {level['size']:>5} {1 / n_options:>7.3f} "
            f"{majority:>9.3f} {cheap:>7.3f}"
        )
        # The cheap baseline must be a baseline, not the answer. If one of these
        # ever reaches 1.00 the condition measures nothing.
        assert cheap < 0.95, f"cheap baseline solves {level}: {cheap}"
        for inst in batch:
            q = inst.questions[QUESTION_KEY]
            assert q["type"] == "choice"
            assert inst.truth[QUESTION_KEY] in {o["id"] for o in q["options"]}

    # Determinism: the same coordinates give the same instance.
    a = generate(difficulty={"family": "sexpr", "size": 5}, seed=seed, count=5)
    b = generate(difficulty={"family": "sexpr", "size": 5}, seed=seed, count=5)
    assert [i.state for i in a] == [i.state for i in b]
    assert [i.instance_id for i in a] == [i.instance_id for i in b]

    # sexpr and ast are the same expression through two surfaces.
    asts = generate(difficulty={"family": "ast", "size": 5}, seed=seed, count=5)
    assert [i.meta["true_value"] for i in a] == [i.meta["true_value"] for i in asts]
    assert [i.meta["sexpr"] for i in a] == [i.meta["sexpr"] for i in asts]
    assert [i.state for i in a] != [i.state for i in asts]

    # Every language shows the same scene with the same answer.
    scenes = {
        level["language"]: generate(difficulty=level, seed=seed, count=20)
        for level in language_sweep()
    }
    english = scenes["en"]
    for lang, batch in scenes.items():
        assert [i.truth[QUESTION_KEY] for i in batch] == [
            i.truth[QUESTION_KEY] for i in english
        ], lang
        assert [i.meta["queried_color"] for i in batch] == [
            i.meta["queried_color"] for i in english
        ], lang
        if lang != "en":
            assert [i.state for i in batch] != [i.state for i in english], lang

    # A slip label for every non-truth option, so a wrong answer is diagnosable.
    for family in ("coord_walk", "sexpr", "hexdump", "bits", "dimacs"):
        batch = generate(
            difficulty={"family": family, "size": SIZES[family][-1]}, seed=seed, count=10
        )
        for inst in batch:
            ids = {o["id"] for o in inst.questions[QUESTION_KEY]["options"]}
            assert set(inst.meta["slips"]) == ids, (family, inst.instance_id)

    print("\nnumeric: self-check passed")


if __name__ == "__main__":
    _self_check()
