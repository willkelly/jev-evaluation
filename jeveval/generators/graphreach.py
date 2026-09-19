"""Directed graph reachability, for E9 (distribution edges).

The plan predicts reachability holds at one or two hops and falls off fast past
four, so the control here is the true shortest path length and the sweep runs 1
through 8 to bracket that prediction on both sides.

Half of every batch is reachable and half is not, so the majority-class baseline
is exactly 0.5.

The two halves are built by one procedure and differ by a single edge. Both
start from a cut of the node set into a source side S and a target side T, with
every edge from S to T forbidden. Both lay down a chain of `path_len` + 1 nodes:
the first k edges walk forward from the source inside S, the remaining ones
arrive at the target from inside T. The reachable half adds the one edge that
joins those two pieces across the cut; the unreachable half does not, and spends
that edge on one more distractor instead. Node count, edge count, the distractor
distribution, the source out-degree and the target in-degree are therefore the
same in both classes, and a search outward from the source sees the same graph
in both until it reaches the crossing point.

Building the unreachable class this way rather than by deleting the path matters:
deleting the path leaves the source with a short dead end and the target with no
in-edges at all, and both are visible without doing any search.

Shortest path length is exact, not approximate. Every node carries a level, the
source is at level 0 and the target at level `path_len`, and no edge may raise
the level by more than one, so no path can be shorter than the chain. The
crossing edge is the only S-to-T edge, so every source-to-target path must use
it, which pins the distance at exactly `path_len`. The distance and the
non-reachability are then re-checked with a BFS over the finished edge list
before the instance is returned; the construction argument is not trusted alone.

What does still separate the classes cheaply is depth-limited search: past the
crossing depth the reachable frontier keeps growing and the unreachable one
stops at the cut. That is the quantity the question is about rather than an
artifact of the generator, but it is a real shortcut, so the crossing depth is
randomized per instance, the frontier sizes go into meta, and the self-check
prints how well the best threshold on them does when it is fitted on one batch
and scored on another. That number runs about 0.80 at path_len 3 and decays to
about 0.51 at path_len 8, and it, not 0.5, is what a model score has to be read
against at the short end of the sweep. The counting features -- edge count,
density, endpoint degrees -- sit at 0.50 across the whole sweep, which the
self-check asserts.

Run the self-check with:

    python -m jeveval.generators.graphreach
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable

from ..instances import Instance, noul, rng_for

NAME = "graphreach"

# One question per instance; experiments join on this key.
QUESTION_KEY = "reachable"

# Source out-degree and target in-degree are drawn from this range in both
# classes and then forced to that exact value, so neither degree carries any
# information about the answer.
_ENDPOINT_DEGREE_RANGE = (1, 3)

# Fraction of nodes placed on the source side of the cut. Randomized per
# instance so that no single threshold on the size of a depth-limited frontier
# works across a whole condition.
_CUT_FRACTION_RANGE = (0.25, 0.75)

# Spare nodes held next to each endpoint. The level rule and the cut together
# can leave a source with no legal out-neighbour at all -- levels near 0 are
# scarce when path_len is large -- and then the forced endpoint degree is
# unsatisfiable. Reserving this many nodes at level <= 1 on the source side and
# level >= path_len - 1 on the target side makes it always satisfiable, in both
# classes identically.
_RESERVE = 3

# Depths at which the cheap depth-limited baseline is recorded. Three is already
# past the plan's predicted knee; anything deeper is just BFS.
_BASELINE_DEPTHS = (1, 2, 3)


def difficulty_sweep() -> list[dict]:
    """Hop depth 1..8 at fixed size and density, plus dilution and size arms.

    The hop ladder is the point of the generator and holds node count and
    density constant so the only thing moving is the number of hops. The three
    extra densities and the one larger graph sit at path_len 4 -- next to where
    the plan expects the knee -- so that a fall-off there can be attributed to
    hops rather than to dilution or to input size.
    """
    levels = [
        {"n_nodes": 24, "path_len": length, "distractor_ratio": 1.5}
        for length in range(1, 9)
    ]
    levels += [
        {"n_nodes": 24, "path_len": 4, "distractor_ratio": ratio}
        for ratio in (0.5, 3.0, 6.0)
    ]
    levels += [{"n_nodes": 48, "path_len": 4, "distractor_ratio": 1.5}]
    levels.sort(key=lambda d: (d["path_len"], d["distractor_ratio"], d["n_nodes"]))
    return levels


def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]:
    """`count` reachability instances, alternating reachable and unreachable.

    The class is `index % 2` rather than a shuffled balanced list because the
    generator contract requires instance i to depend on i alone: a shuffle would
    make every instance's label depend on `count`. Alternating gives an exactly
    balanced batch for even `count` and keeps per-index determinism.
    """
    n_nodes, path_len, ratio = _unpack(difficulty)
    out: list[Instance] = []
    for index in range(count):
        rng = rng_for(NAME, difficulty, seed, index)
        out.append(
            _build_instance(
                difficulty=difficulty,
                seed=seed,
                index=index,
                rng=rng,
                n_nodes=n_nodes,
                path_len=path_len,
                ratio=ratio,
                reachable=(index % 2 == 0),
            )
        )
    return out


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def _unpack(difficulty: dict) -> tuple[int, int, float]:
    try:
        n_nodes = int(difficulty["n_nodes"])
        path_len = int(difficulty["path_len"])
        ratio = float(difficulty["distractor_ratio"])
    except KeyError as exc:
        raise ValueError(f"{NAME}: difficulty missing key {exc}") from exc
    if path_len < 1:
        raise ValueError(f"{NAME}: path_len must be >= 1, got {path_len}")
    if n_nodes < path_len + 1 + 2 * _RESERVE:
        raise ValueError(
            f"{NAME}: n_nodes {n_nodes} leaves no room beside a {path_len}-hop chain "
            f"plus {_RESERVE} reserved nodes per side"
        )
    if ratio < 0:
        raise ValueError(f"{NAME}: distractor_ratio must be >= 0, got {ratio}")
    return n_nodes, path_len, ratio


def _build_edges(
    rng: random.Random, n_nodes: int, path_len: int, ratio: float, reachable: bool
) -> dict:
    """The shared skeleton-plus-cut construction. Returns the pieces meta needs."""
    n_edges = path_len + round(ratio * n_nodes)

    # k edges of the chain sit on the source side, the rest on the target side.
    # For path_len 1 both pieces are empty and the chain is the crossing edge.
    k = rng.randrange(path_len)
    tail = path_len - 1 - k
    chain = rng.sample(range(n_nodes), path_len + 1)
    head_nodes, tail_nodes = chain[: k + 1], chain[k + 1 :]
    src, dst = head_nodes[0], tail_nodes[-1]

    level = {v: i for i, v in enumerate(head_nodes)}
    level.update({v: k + 1 + j for j, v in enumerate(tail_nodes)})
    side = {v: 0 for v in head_nodes}
    side.update({v: 1 for v in tail_nodes})

    # Split the nodes that are not on the chain between the two sides. The size
    # of the split is randomized but clamped so each side keeps its reserve.
    free = [v for v in range(n_nodes) if v not in level]
    rng.shuffle(free)
    want_source_side = round(rng.uniform(*_CUT_FRACTION_RANGE) * n_nodes)
    want_source_side = max(
        len(head_nodes) + _RESERVE,
        min(n_nodes - len(tail_nodes) - _RESERVE, want_source_side),
    )
    split = want_source_side - len(head_nodes)
    for i, v in enumerate(free):
        on_source_side = i < split
        side[v] = 0 if on_source_side else 1
        reserved = i < _RESERVE if on_source_side else i - split < _RESERVE
        if not reserved:
            level[v] = rng.randrange(path_len + 1)
        elif on_source_side:
            level[v] = rng.randrange(0, 2)
        else:
            level[v] = rng.randrange(max(0, path_len - 1), path_len + 1)

    def allowed(u: int, v: int) -> bool:
        return level[v] <= level[u] + 1 and not (side[u] == 0 and side[v] == 1)

    edges: set[tuple[int, int]] = set()
    edges.update(zip(head_nodes, head_nodes[1:]))
    edges.update(zip(tail_nodes, tail_nodes[1:]))
    if reachable:
        # The only edge in the graph that runs from the source side to the
        # target side, added directly because `allowed` forbids exactly this.
        edges.add((head_nodes[-1], tail_nodes[0]))

    _force_endpoint_degree(rng, n_nodes, edges, allowed, src, dst)

    pool = [
        (u, v)
        for u in range(n_nodes)
        for v in range(n_nodes)
        if u != v and u != src and v != dst and (u, v) not in edges and allowed(u, v)
    ]
    need = n_edges - len(edges)
    if need < 0:
        raise ValueError(
            f"{NAME}: distractor_ratio too small for path_len {path_len}; "
            f"the skeleton alone needs {len(edges)} of {n_edges} edges"
        )
    if need > len(pool):
        raise ValueError(
            f"{NAME}: distractor_ratio too large for n_nodes {n_nodes}; "
            f"need {need} more edges but only {len(pool)} legal pairs remain"
        )
    edges.update(rng.sample(pool, need))

    # Sorted before shuffling so the wire order does not depend on set iteration.
    edge_list = sorted(edges)
    rng.shuffle(edge_list)
    return {
        "edges": edge_list,
        "src": src,
        "dst": dst,
        "crossing_depth": k + 1,
        "tail_len": tail,
        "n_edges": n_edges,
    }


def _force_endpoint_degree(
    rng: random.Random,
    n_nodes: int,
    edges: set[tuple[int, int]],
    allowed: Callable[[int, int], bool],
    src: int,
    dst: int,
) -> None:
    """Pin the source out-degree and target in-degree to values drawn the same
    way in both classes.

    Without this the unreachable class would have systematically fewer out-edges
    at the source, because the cut removes candidates there but not in the
    reachable class, and counting edges at one node would beat chance.
    """
    d_out = rng.randint(*_ENDPOINT_DEGREE_RANGE)
    d_in = rng.randint(*_ENDPOINT_DEGREE_RANGE)

    have_out = sum(1 for u, _ in edges if u == src)
    candidates = [
        v
        for v in range(n_nodes)
        if v != src and (src, v) not in edges and allowed(src, v)
    ]
    need = max(0, d_out - have_out)
    if need > len(candidates):
        raise ValueError(f"{NAME}: cannot give the source out-degree {d_out}")
    edges.update((src, v) for v in rng.sample(candidates, need))

    have_in = sum(1 for _, v in edges if v == dst)
    candidates = [
        u
        for u in range(n_nodes)
        if u != dst and (u, dst) not in edges and allowed(u, dst)
    ]
    need = max(0, d_in - have_in)
    if need > len(candidates):
        raise ValueError(f"{NAME}: cannot give the target in-degree {d_in}")
    edges.update((u, dst) for u in rng.sample(candidates, need))


def _build_instance(
    *,
    difficulty: dict,
    seed: int,
    index: int,
    rng: random.Random,
    n_nodes: int,
    path_len: int,
    ratio: float,
    reachable: bool,
) -> Instance:
    built = _build_edges(rng, n_nodes, path_len, ratio, reachable)
    edges, src, dst = built["edges"], built["src"], built["dst"]

    forward = _distances(edges, n_nodes, src, reverse=False)
    backward = _distances(edges, n_nodes, dst, reverse=True)
    observed = forward[dst]

    # The construction argument is checked against a real BFS, every instance.
    if reachable:
        if observed != path_len:
            raise AssertionError(
                f"{NAME}: built a {path_len}-hop instance whose BFS "
                f"distance is {observed}"
            )
    elif observed is not None:
        raise AssertionError(
            f"{NAME}: built an unreachable instance that BFS reaches in {observed}"
        )

    state = _render(edges, n_nodes)
    question = noul(
        f"Is there a directed path from {_name(src)} to {_name(dst)} in the graph "
        "in the state? Follow each edge only in the direction it is written."
    )
    baseline = {
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "edge_density": len(edges) / (n_nodes * (n_nodes - 1)),
        "out_degree_source": sum(1 for u, _ in edges if u == src),
        "in_degree_target": sum(1 for _, v in edges if v == dst),
    }
    for depth in _BASELINE_DEPTHS:
        baseline[f"reach_within_{depth}"] = observed is not None and observed <= depth
        baseline[f"frontier_out_{depth}"] = sum(
            1 for d in forward if d is not None and d <= depth
        )
        baseline[f"frontier_in_{depth}"] = sum(
            1 for d in backward if d is not None and d <= depth
        )

    inst = Instance(
        generator=NAME,
        difficulty=dict(difficulty),
        seed=seed,
        index=index,
        state=state,
        questions={QUESTION_KEY: question},
        truth={QUESTION_KEY: bool(reachable)},
        meta={
            # Ground truth, kept for grouping and for the per-hop curve. An
            # offline baseline must read meta["baseline"] and nothing else.
            "true_path_len": observed,
            "n_nodes": n_nodes,
            "n_edges": len(edges),
            # Nominal, and the same in both classes. The unreachable half
            # actually carries one more distractor, because it spends the
            # crossing edge on one; recording that would hand a baseline the
            # answer, so this is the setting rather than the realized count.
            "nominal_distractors": built["n_edges"] - path_len,
            "distractor_ratio": ratio,
            "source": _name(src),
            "target": _name(dst),
            # Hop at which the reachable frontier first crosses the cut. Present
            # in both classes; the two graphs are identical to a searcher below
            # this depth, so it predicts where a shallow search stops working.
            "crossing_depth": built["crossing_depth"],
            "tail_len": built["tail_len"],
            "baseline": baseline,
        },
    )
    inst.validate()
    return inst


# --------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------


def _name(node: int) -> str:
    return f"n{node}"


def _distances(
    edges: list[tuple[int, int]], n_nodes: int, start: int, *, reverse: bool
) -> list[int | None]:
    """BFS hop counts from `start`, or to `start` when `reverse`."""
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        if reverse:
            adj[v].append(u)
        else:
            adj[u].append(v)
    dist: list[int | None] = [None] * n_nodes
    dist[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] is None:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _render(edges: list[tuple[int, int]], n_nodes: int) -> str:
    nodes = ", ".join(_name(v) for v in range(n_nodes))
    lines = "\n".join(f"{_name(u)} -> {_name(v)}" for u, v in edges)
    return (
        f"Directed graph with {n_nodes} nodes and {len(edges)} edges.\n"
        f"Nodes: {nodes}\n"
        f"Edges (a -> b means an edge from a to b):\n{lines}"
    )


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------


def _best_threshold(values: list[float], labels: list[bool]) -> tuple[float, int]:
    """The single threshold on one feature that separates the classes best."""
    total = len(values)
    positives = sum(labels)
    pairs = sorted(zip(values, labels))
    # Seed with the better of the two trivial rules -- `cut = -inf` with
    # direction 1 calls everything positive, with direction -1 everything
    # negative -- and seed it with a score that the returned rule actually
    # achieves. Seeding with max(positives, total - positives) while always
    # returning direction 1 is wrong on an unbalanced batch: when no real
    # threshold beats the majority rate the caller gets the *losing* trivial
    # rule, and the leak comes back below chance instead of at the majority
    # rate. Understating is the dangerous direction for a leak detector.
    if positives >= total - positives:
        best, cut, direction = positives / total, float("-inf"), 1
    else:
        best, cut, direction = (total - positives) / total, float("-inf"), -1
    above_pos, above_neg = positives, total - positives
    for i, (value, label) in enumerate(pairs):
        if label:
            above_pos -= 1
        else:
            above_neg -= 1
        if i + 1 < total and pairs[i + 1][0] == value:
            continue
        acc = (above_pos + (total - positives) - above_neg) / total
        for candidate, sign in ((acc, 1), (1.0 - acc, -1)):
            if candidate > best:
                best, cut, direction = candidate, value, sign
    return cut, direction


def _threshold_accuracy(
    values: list[float], labels: list[bool], cut: float, direction: int
) -> float:
    hits = sum(
        ((value > cut) if direction == 1 else (value <= cut)) == label
        for value, label in zip(values, labels)
    )
    return hits / len(values)


def _held_out_leak(
    fit_batch: list[Instance], test_batch: list[Instance], feature: str
) -> float:
    """Accuracy of the best threshold on `feature`, fitted and scored on
    different batches.

    Fitting and scoring on the same batch inflates this badly at these sample
    sizes -- a free threshold gets about 0.60 on pure noise -- and an inflated
    baseline is worse than none, because the report would subtract it from the
    model's score.
    """
    key = lambda inst: float(inst.meta["baseline"][feature])
    cut, direction = _best_threshold(
        [key(i) for i in fit_batch], [i.truth[QUESTION_KEY] for i in fit_batch]
    )
    return _threshold_accuracy(
        [key(i) for i in test_batch],
        [i.truth[QUESTION_KEY] for i in test_batch],
        cut,
        direction,
    )


def _self_check() -> None:
    count = 120
    seed = 4242
    rows = []
    for difficulty in difficulty_sweep():
        batch = generate(difficulty=difficulty, seed=seed, count=count)
        again = generate(difficulty=difficulty, seed=seed, count=count)
        assert [i.to_json() for i in batch] == [
            i.to_json() for i in again
        ], f"nondeterministic at {difficulty}"
        held_out = generate(difficulty=difficulty, seed=seed + 1, count=count)

        labels = [i.truth[QUESTION_KEY] for i in batch]
        assert sum(labels) == count // 2, f"class imbalance at {difficulty}"
        assert len({i.meta["n_edges"] for i in batch}) == 1, (
            f"edge count leaks the class at {difficulty}"
        )
        assert len({i.meta["n_nodes"] for i in batch}) == 1, (
            f"node count leaks the class at {difficulty}"
        )

        marker = "Edges (a -> b means an edge from a to b):\n"
        for inst in batch:
            base = inst.meta["baseline"]
            assert base["out_degree_source"] >= 1
            assert base["in_degree_target"] >= 1
            assert inst.meta["source"] != inst.meta["target"]
            if inst.truth[QUESTION_KEY]:
                assert inst.meta["true_path_len"] == difficulty["path_len"]
            else:
                assert inst.meta["true_path_len"] is None
            edges = [
                tuple(line.split(" -> "))
                for line in inst.state.split(marker)[1].split("\n")
            ]
            assert len(edges) == inst.meta["n_edges"], "rendered edge count"
            assert len(set(edges)) == len(edges), "duplicate edge"
            assert all(u != v for u, v in edges), "self loop"
            assert inst.meta["source"] in {u for u, _ in edges}
            assert inst.meta["target"] in {v for _, v in edges}

        # Counting features must be at chance. The frontier features are not
        # expected to be -- a depth-limited search is a real baseline -- so they
        # are reported rather than asserted.
        for feature in (
            "n_edges",
            "edge_density",
            "out_degree_source",
            "in_degree_target",
        ):
            leak = _held_out_leak(batch, held_out, feature)
            assert leak < 0.60, (
                f"{feature} separates the classes ({leak:.2f}) at {difficulty}"
            )

        def mean(feature: str, want: bool) -> float:
            vals = [
                i.meta["baseline"][feature]
                for i in batch
                if i.truth[QUESTION_KEY] is want
            ]
            return sum(vals) / len(vals)

        depth2 = sum(
            i.meta["baseline"]["reach_within_2"] == i.truth[QUESTION_KEY] for i in batch
        ) / count
        rows.append(
            (
                f"n={difficulty['n_nodes']} L={difficulty['path_len']} "
                f"r={difficulty['distractor_ratio']}",
                batch[0].meta["n_edges"],
                mean("frontier_out_2", True),
                mean("frontier_out_2", False),
                depth2,
                _held_out_leak(batch, held_out, "frontier_out_3"),
                _held_out_leak(batch, held_out, "out_degree_source"),
            )
        )

    header = (
        f"{'difficulty':24} {'edges':>5} {'front2+':>7} {'front2-':>7} "
        f"{'d2 acc':>6} {'front3':>6} {'degree':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, edges, pos, neg, depth2, front, degree in rows:
        print(
            f"{name:24} {edges:5d} {pos:7.1f} {neg:7.1f} "
            f"{depth2:6.2f} {front:6.2f} {degree:6.2f}"
        )
    print(
        "\nfront2+/- : mean nodes within 2 hops of the source, reachable / unreachable."
        "\nd2 acc    : accuracy of 'reachable iff found within 2 hops'."
        "\nfront3    : held-out accuracy of the best threshold on the 3-hop"
        "\n            frontier size."
        "\ndegree    : the same on source out-degree, which should sit at 0.50."
        f"\n{len(rows)} difficulty levels, {count} instances each, majority class 0.50."
    )


if __name__ == "__main__":
    _self_check()
