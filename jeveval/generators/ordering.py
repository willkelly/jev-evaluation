"""Pairwise ordering against a known total order, for E6 transitivity.

The domain is the chemical elements ordered by atomic number. The plan offers
word frequencies, file sizes or dates; atomic number wins on the one criterion
that matters for a generator with no network access and no data files: the
whole table can be written down from memory and checked by anyone against a
periodic table. It is a total order by definition, it has no ties, the rank
*is* the attribute so there are no corpus-dependent near-ties to argue about,
and rank distance is a meaningful difficulty axis -- hydrogen against lead is
free, terbium against dysprosium is not. Word frequency would have given a
smoother difficulty gradient, but its close pairs are only defined relative to
a corpus, and a generator that ships ground truth it cannot defend is worse
than one with a coarser gradient.

Instances come in groups of four, which is what E6 items 1 and 2 need:

    index 4t+0  noul  a vs b        \\
    index 4t+1  noul  b vs c         >  the same three items, three calls
    index 4t+2  noul  a vs c        /
    index 4t+3  choice over the 6 orderings of a, b, c  -- the enrolled form

The three nouls are separate instances, not three questions on one instance,
because the experiment counts cycles across separate *calls*; batching them
would measure within-call coherence, which is E5's job. The fourth instance
asks the same question of the same items in one call, so enrolled and
unenrolled are compared on identical material rather than on matched samples.

`count` is a count of instances, as the generator contract requires, so an
experiment that wants T triples asks for 4*T. Grouping offline is a join on
``meta["triple_id"]``, and ``meta["role"]`` says which of the four an instance
is.

Which item is named first in a noul is randomized per instance, so the answer
is yes half the time and "always say yes" scores 0.50. That flip stays random
rather than being forced to an exact 50/50 by index parity, the way `dfa` does
it: parity would fix each triple's three orientations together, and a constant
answerer would then land on a cyclic tournament 34% of the time instead of the
25% a random answerer gets. E6's "doesn't work" tier is defined as cycles near
that 25% rate, so the null has to stay where the plan puts it. The cost is that
the realized yes-rate is 0.50 only to within sampling error -- about +/-0.013
over 1500 nouls -- so accuracy is reported against `metrics.majority_baseline`
of the labels actually drawn, not against the 0.5 in meta.

Ground truth is the element's position in the table below; the comparison
direction asked and the true ranks both travel in meta, so cycle detection
offline never has to re-derive anything.

Triples are drawn without replacement, but 118 elements is a hard ceiling on
how many exist: 116 at ``min_gap=max_gap=1``, about 1,000 at 2..4, and
thousands above that. ``meta["n_distinct_triples"]`` carries the ceiling for
the rung, so an experiment asking for more triples than that can either cap the
draw or report the effective sample size honestly.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from functools import lru_cache
from typing import Any

from ..instances import Instance, choice, noul, rng_for, shuffled_options

NAME = "ordering"

# IUPAC names, index + 1 == atomic number. This list is the ground truth.
ELEMENTS: list[str] = [
    "hydrogen", "helium", "lithium", "beryllium", "boron", "carbon",
    "nitrogen", "oxygen", "fluorine", "neon", "sodium", "magnesium",
    "aluminium", "silicon", "phosphorus", "sulfur", "chlorine", "argon",
    "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium",
    "manganese", "iron", "cobalt", "nickel", "copper", "zinc",
    "gallium", "germanium", "arsenic", "selenium", "bromine", "krypton",
    "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum",
    "technetium", "ruthenium", "rhodium", "palladium", "silver", "cadmium",
    "indium", "tin", "antimony", "tellurium", "iodine", "xenon",
    "caesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
    "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium",
    "holmium", "erbium", "thulium", "ytterbium", "lutetium", "hafnium",
    "tantalum", "tungsten", "rhenium", "osmium", "iridium", "platinum",
    "gold", "mercury", "thallium", "lead", "bismuth", "polonium",
    "astatine", "radon", "francium", "radium", "actinium", "thorium",
    "protactinium", "uranium", "neptunium", "plutonium", "americium", "curium",
    "berkelium", "californium", "einsteinium", "fermium", "mendelevium", "nobelium",
    "lawrencium", "rutherfordium", "dubnium", "seaborgium", "bohrium", "hassium",
    "meitnerium", "darmstadtium", "roentgenium", "copernicium", "nihonium", "flerovium",
    "moscovium", "livermorium", "tennessine", "oganesson",
]

RANK: dict[str, int] = {name: i + 1 for i, name in enumerate(ELEMENTS)}

ROLES = ("ab", "bc", "ac", "enrolled")

_ATTRIBUTE = "atomic number"
_DOMAIN = "chemical elements"


def difficulty_sweep() -> list[dict]:
    """Easy to hard, which here means far apart to adjacent.

    ``min_gap`` and ``max_gap`` bound the two *adjacent* gaps of the triple: the
    three ranks are r, r+g1, r+g1+g2 with each g drawn from that range, so the
    outer pair sits at g1+g2 and is recorded separately. Three items cannot all
    be one rank apart, which is why the constraint is on adjacent gaps rather
    than on every pair.

    The ends bracket the interesting range. At a gap of 40 or more every
    comparison is between elements a schoolchild can place, so any violation
    there is a coherence failure and nothing else; at a gap of 1 the pairs are
    neighbours, often in the lanthanide or actinide block, where the order is a
    genuine recall problem and where any transitivity violation that exists at
    all will show up. `distance` is carried as its own key because the plan
    requires violations reported split by distance and a categorical key makes
    that split a group-by rather than a threshold argument.

    The rungs differ in how much material they have. Distinct triples available
    per rung, in the order returned: 7,220, 23,600, 3,708, 1,008 and 116. The
    plan asks for 500 triples split by distance; the last rung cannot supply
    500 distinct ones, because only 116 exist. Widening it to ``max_gap=3``
    would clear 500 but would stop being the adjacent-pair test it exists to
    be, so the rung keeps its gap and reports its ceiling instead. Spending the
    close-pair budget across the three close rungs gets 500 distinct triples
    without asking any single rung for more than it has.
    """
    return [
        {"distance": "far", "min_gap": 40, "max_gap": 58},
        {"distance": "far", "min_gap": 20, "max_gap": 39},
        {"distance": "close", "min_gap": 5, "max_gap": 10},
        {"distance": "close", "min_gap": 2, "max_gap": 4},
        {"distance": "close", "min_gap": 1, "max_gap": 1},
    ]


def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]:
    """`count` instances at one difficulty setting, instance `i` from `i` alone.

    Instances arrive in triple order, four per triple; ``count`` need not be a
    multiple of four, in which case the final triple is truncated and an
    experiment counting cycles should discard it.
    """
    return [_instance(difficulty, seed, i) for i in range(count)]


# --------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _triple_pool(lo: int, hi: int) -> tuple[tuple[int, int, int], ...]:
    """Every `(start_rank, g1, g2)` a gap range admits, in a fixed order.

    The pool is small enough to enumerate -- 118 elements times a bounded gap
    range -- and enumerating it is what lets a draw take triples *without*
    replacement. Sampling gaps and a start rank independently, as this did
    originally, draws with replacement from the same pool: at ``min_gap=1,
    max_gap=1`` only 116 triples exist, so a 500-triple request returned about
    114 distinct ones, each repeated four or five times. The cycle rate would
    then have been reported over 500 nominally independent triples whose
    effective count was 114, and its confidence interval would have been about
    2.1x tighter than the data supports -- on the one rung E6 cares most about.
    """
    n = len(ELEMENTS)
    # A gap above n - 1 - lo can never appear, since the other gap is at least
    # lo and the two together must fit inside the table. Clamping here keeps
    # enumeration proportional to the table rather than to `hi`, which an
    # unclamped double loop would make quadratic in a caller-supplied number.
    hi = min(hi, n - 1 - lo)
    return tuple(
        (r, g1, g2)
        for g1 in range(lo, hi + 1)
        for g2 in range(lo, hi + 1)
        for r in range(1, n - g1 - g2 + 1)
    )


@lru_cache(maxsize=16)
def _triple_order(difficulty_key: str, seed: int) -> tuple[int, ...]:
    """A seeded permutation of the pool's indices.

    Memoised, but a pure function of its arguments, so it is not hidden state:
    a fresh process with the same key produces the same permutation.
    """
    difficulty = json.loads(difficulty_key)
    lo, hi = difficulty["min_gap"], difficulty["max_gap"]
    if not 1 <= lo <= hi:
        raise ValueError(f"bad gap range {lo}..{hi}")
    order = list(range(len(_triple_pool(lo, hi))))
    if not order:
        raise ValueError(
            f"min_gap {lo} leaves no room for a triple in {len(ELEMENTS)} items"
        )
    rng_for(f"{NAME}:pool", difficulty, seed, 0).shuffle(order)
    return tuple(order)


def _triple(difficulty: dict, seed: int, t: int) -> dict:
    """The three items of triple `t`, identical across its four instances.

    Triple `t` takes the `t`-th entry of a seeded permutation of the pool, so
    the triples of one draw are distinct until the pool runs out, and then
    repeat in the same order rather than colliding at random. `pool_index` is
    the identity of the *material*: two instances sharing it are the same three
    elements, where `triple_id` identifies the draw.

    The label shuffle is drawn from a stream namespaced away from the
    per-instance stream, so that instance 4t -- which shares an index with
    nothing else -- cannot end up with its presentation flip correlated with
    its own item draw.
    """
    order = _triple_order(json.dumps(difficulty, sort_keys=True), seed)
    pool_index = order[t % len(order)]
    r1, g1, g2 = _triple_pool(difficulty["min_gap"], difficulty["max_gap"])[pool_index]
    ranks = [r1, r1 + g1, r1 + g1 + g2]
    labels = ["a", "b", "c"]
    rng = rng_for(f"{NAME}:triple", difficulty, seed, t)
    rng.shuffle(labels)  # so the label order carries no information about rank
    items = {lab: ELEMENTS[rank - 1] for lab, rank in zip(labels, ranks)}
    payload = json.dumps({"g": NAME, "d": difficulty, "s": seed, "t": t}, sort_keys=True)
    return {
        "items": items,
        "adjacent_gaps": [g1, g2],
        "pool_index": pool_index,
        "pool_size": len(order),
        "triple_id": f"{NAME}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}",
    }


def _ordering_id(names: tuple[str, ...] | list[str]) -> str:
    return "|".join(names)


def _instance(difficulty: dict, seed: int, index: int) -> Instance:
    t, role_ix = divmod(index, len(ROLES))
    role = ROLES[role_ix]
    tri = _triple(difficulty, seed, t)
    items = tri["items"]
    ranks = {name: RANK[name] for name in items.values()}
    correct = tuple(sorted(items.values(), key=lambda nm: -ranks[nm]))
    distances = {
        "ab": abs(ranks[items["a"]] - ranks[items["b"]]),
        "bc": abs(ranks[items["b"]] - ranks[items["c"]]),
        "ac": abs(ranks[items["a"]] - ranks[items["c"]]),
    }

    rng = rng_for(NAME, difficulty, seed, index)
    meta: dict[str, Any] = {
        "role": role,
        "triple_id": tri["triple_id"],
        "triple_index": t,
        "items": dict(items),
        "true_ranks": ranks,
        "correct_ordering_id": _ordering_id(correct),
        "correct_ordering": list(correct),
        "adjacent_gaps": tri["adjacent_gaps"],
        # Identity of the material, and how much material this rung has. Equal
        # pool_index means literally the same three elements; a draw of more
        # than n_distinct_triples has begun reusing them, and its effective
        # sample size for a per-triple rate is n_distinct_triples, not the
        # number of triples asked for.
        "triple_pool_index": tri["pool_index"],
        "n_distinct_triples": tri["pool_size"],
        "rank_distances": distances,
        "distance_class": difficulty["distance"],
        "attribute": _ATTRIBUTE,
    }

    if role == "enrolled":
        shown = list(items.values())
        rng.shuffle(shown)
        options = [
            {
                "id": _ordering_id(perm),
                "label": f"{perm[0]}, then {perm[1]}, then {perm[2]}",
            }
            for perm in itertools.permutations(sorted(items.values()))
        ]
        state = {"domain": _DOMAIN, "attribute": _ATTRIBUTE, "items": shown}
        questions = {
            "ordering": shuffled_options(
                choice(
                    f"Which ordering lists these elements from highest {_ATTRIBUTE} "
                    f"to lowest?",
                    options,
                ),
                rng,
            )
        }
        truth: dict[str, Any] = {"ordering": _ordering_id(correct)}
        # The binding constraint on an ordering is its closest adjacent pair.
        meta["rank_distance"] = min(distances["ab"], distances["bc"], distances["ac"])
        meta["n_options"] = len(options)
        meta["baseline_random"] = 1 / 6
    else:
        first, second = items[role[0]], items[role[1]]
        # Randomize which item is named first so that "yes" is correct half the
        # time; without this, truth would track the label order.
        if rng.random() < 0.5:
            first, second = second, first
        state = {"domain": _DOMAIN, "attribute": _ATTRIBUTE, "items": [first, second]}
        questions = {
            "greater": noul(f"Does {first} have a higher {_ATTRIBUTE} than {second}?")
        }
        truth = {"greater": ranks[first] > ranks[second]}
        meta["pair"] = [role[0], role[1]]
        meta["left"] = first
        meta["right"] = second
        meta["rank_distance"] = distances[role]
        meta["baseline_random"] = 0.5
        meta["baseline_majority"] = 0.5

    inst = Instance(
        generator=NAME,
        difficulty=dict(difficulty),
        seed=seed,
        index=index,
        state=state,
        questions=questions,
        truth=truth,
        meta=meta,
    )
    inst.validate()
    return inst


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    SEED = 909
    N = 200  # 50 triples per difficulty

    assert len(ELEMENTS) == 118, len(ELEMENTS)
    assert len(set(ELEMENTS)) == 118, "duplicate element"
    assert all(e == e.lower() and e.isalpha() for e in ELEMENTS)
    for spot, z in [("hydrogen", 1), ("carbon", 6), ("iron", 26), ("silver", 47),
                    ("terbium", 65), ("gold", 79), ("uranium", 92), ("oganesson", 118)]:
        assert RANK[spot] == z, (spot, RANK[spot])

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

        # Triples must be distinct until the pool runs out: this is the check
        # that the draw is without replacement rather than merely seeded.
        pool = batch[0].meta["n_distinct_triples"]
        seen_ix = [i.meta["triple_pool_index"] for i in batch[::4]]
        assert len(set(seen_ix[:min(len(seen_ix), pool)])) == min(len(seen_ix), pool), (
            f"repeated triple before the pool of {pool} was exhausted at {d}")
        materials = {
            tuple(sorted(batch[4 * t].meta["items"].values())) for t in range(N // 4)
        }
        assert len(materials) == len(set(seen_ix)), "pool_index does not identify the material"
        # And a draw that overruns the pool must wrap cleanly, not collide.
        over = generate(difficulty=d, seed=SEED, count=4 * (pool + 3))
        ix = [i.meta["triple_pool_index"] for i in over[::4]]
        assert len(set(ix)) == pool, f"wrapped draw covered {len(set(ix))} of {pool}"
        assert ix[:3] == ix[pool:pool + 3], "the wrap is not a clean repeat"

        yes = 0
        gaps = []
        for t in range(N // 4):
            group = batch[4 * t: 4 * t + 4]
            assert [i.meta["role"] for i in group] == list(ROLES)
            assert len({i.meta["triple_id"] for i in group}) == 1
            assert len({json.dumps(i.meta["items"], sort_keys=True) for i in group}) == 1

            items = group[0].meta["items"]
            ranks = group[0].meta["true_ranks"]
            assert len(set(ranks.values())) == 3, "tied items"
            g1, g2 = group[0].meta["adjacent_gaps"]
            assert d["min_gap"] <= g1 <= d["max_gap"] and d["min_gap"] <= g2 <= d["max_gap"]
            gaps.extend([g1, g2])

            # Pairwise truth is the table, read back off the rendered question.
            direction = {}
            for inst in group[:3]:
                left, right = inst.state["items"]
                assert f"Does {left} have a higher" in inst.questions["greater"]["question"]
                assert inst.truth["greater"] == (RANK[left] > RANK[right])
                assert inst.meta["rank_distance"] == abs(RANK[left] - RANK[right])
                hi = left if inst.truth["greater"] else right
                lo = right if inst.truth["greater"] else left
                direction[frozenset((left, right))] = (hi, lo)
                yes += inst.truth["greater"]
            assert len(direction) == 3, "a triple asked the same pair twice"

            # Ground truth is transitive: the three pairwise answers induce a
            # total order with one source, one sink and no cycle. This is the
            # invariant the experiment tests the model against.
            wins: dict[str, int] = {nm: 0 for nm in items.values()}
            for hi, _lo in direction.values():
                wins[hi] += 1
            assert sorted(wins.values()) == [0, 1, 2], wins

            enrolled = group[3]
            order = sorted(wins, key=lambda nm: -wins[nm])
            assert enrolled.truth["ordering"] == "|".join(order), "enrolled disagrees with pairwise"
            assert enrolled.truth["ordering"] == enrolled.meta["correct_ordering_id"]
            assert order == sorted(items.values(), key=lambda nm: -RANK[nm])
            opts = enrolled.questions["ordering"]["options"]
            assert len(opts) == 6 and len({o["id"] for o in opts}) == 6
            assert enrolled.meta["baseline_random"] == 1 / 6

        rate = yes / (3 * (N // 4))
        assert abs(rate - 0.5) < 0.1, f"yes-rate {rate} at {d}"
        example = batch[0].state["items"]
        rows.append((d, rate, sum(gaps) / len(gaps), pool, example))

    print(f"{'difficulty':<48} {'yes-rate':>8} {'mean gap':>9} {'triples':>8}  example pair")
    for d, rate, mean_gap, pool, example in rows:
        desc = ", ".join(f"{k}={v}" for k, v in d.items())
        print(f"{desc:<48} {rate:>8.3f} {mean_gap:>9.2f} {pool:>8}  {example[0]} vs {example[1]}")
    print("\n'triples' is how many distinct triples the rung has; a draw larger "
          "than that reuses them.")
    print("ok")
