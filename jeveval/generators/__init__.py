"""Instance generators.

Every module in this package exposes the same three names, so experiments can
treat generators interchangeably and sweep difficulty without special-casing:

    NAME: str
        Short identifier, matching the module name, e.g. "sat3".

    def difficulty_sweep() -> list[dict]
        The default ladder of difficulty settings for this generator, ordered
        easy to hard. The plan requires curves rather than point estimates, so
        each generator is responsible for knowing its own interesting range --
        the range that brackets the knee, not just the comfortable part of it.

    def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]
        `count` instances at one difficulty setting. Must be deterministic in
        (difficulty, seed) and must not depend on how many instances were drawn
        before it: instance `i` is a function of `i` alone, via
        `instances.rng_for`. Every returned Instance carries ground truth and
        has passed `Instance.validate()`.

Ground truth is computed here, offline, by an actual solver or by construction
-- never by the model under test, and never by a heuristic that is itself one of
the things being measured.

Generators put the features a baseline would use into `Instance.meta`, because
the plan requires every result to be reported next to its baseline: clause
density for 3SAT, true path length for graph reachability, string length for
Dyck. A result without its baseline is not a result.
"""

from __future__ import annotations

from typing import Protocol

from ..instances import Instance


class Generator(Protocol):
    NAME: str

    def difficulty_sweep(self) -> list[dict]: ...

    def generate(self, *, difficulty: dict, seed: int, count: int) -> list[Instance]: ...


def load(name: str):
    """Import a generator module by its short name."""
    import importlib

    return importlib.import_module(f"{__name__}.{name}")


ALL = [
    "sat3",
    "graphreach",
    "progreach",
    "sudoku",
    "ordering",
    "dfa",
    "dyck",
    "semantic",
    "filler",
]
