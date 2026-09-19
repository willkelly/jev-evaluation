"""A faceted museum catalogue with 10,240 leaves, for E7's cardinality sweep.

E7 varies the number of options in a `choice` and asks where quality falls off.
That only measures cardinality if the per-item task is held constant, so this
generator is built backwards from that requirement rather than from realism.

*The label set is a product, not a hand-written list.* Four facets --
8 departments x 8 object classes x 10 period bands x 16 regional schools -- give
10,240 leaves from a vocabulary of 98 facet values, which is small enough to
read in full and audit. The plan asks for "a taxonomy with ~10,000 leaves"; a
hand-written one that size cannot be checked by anyone, and an unauditable
ground truth is worse than a coarser one. Ordering the facets department ->
object -> period -> region turns the faceted classification into a strict tree,
which is what E7's hierarchical descent needs: every leaf has exactly one path,
so a wrong turn puts the correct leaf outside every later option set.

*The item task is a semantic mapping, not a string match.* Each state describes
one object without using the head word of any of its four labels: the label says
"carpet" and the state says "a hand-knotted wool floor covering"; the label says
"Persian" and the state says "Isfahan"; the label says "later nineteenth
century" and the state says "1873". A person does all three instantly. A token
overlap between the state and the option labels does not, which is deliberate:
a lexical task would be invariant to option count by construction and would show
no knee for reasons having nothing to do with the model. The consequence is that
the cheap deterministic baseline this generator computes -- the token overlap,
in `meta["baseline_overlap_expectation"]` -- sits near chance, and E7 reports it
that way rather than pretending a stronger cheap baseline exists.

*Distractors are drawn uniformly by default, and the profile is a knob.* Uniform
sampling keeps the per-option similarity distribution identical at every
cardinality, so the only thing that changes between the 2-option and the
255-option condition is the count. The chance that an option set *contains* a
near miss still rises with N, but that is what having more options means, not a
design confound. E7's second condition holds N at 64 and varies the pool --
`far`, `uniform`, `same_domain`, `same_object` -- which is what separates "too
many options" from "options too similar".

The item is a function of (seed, index) alone and does not depend on the
difficulty, so the same 300 objects can be presented at nine cardinalities and
four distractor profiles and compared within item.

Sizes: the nearest pool, `same_object`, holds 159 leaves, so a `same_object`
condition cannot exceed 160 options; `generate` raises at generation time rather
than letting a condition silently repeat distractors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence

from ..instances import Instance, choice, rng_for, score, shuffled_options

NAME = "taxonomy"

# The four levels, outermost first. Their order is what makes the faceted
# classification a tree, and `children()` and the descent in E7 both read it.
LEVELS: tuple[str, ...] = ("department", "object class", "period", "regional school")

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
#
# Every object's `cue` describes the thing without using the head word of its
# own label, so the state never contains the answer as a string. Each cue also
# names a material or technique that places the object in its department, so
# level 1 of a descent is answerable from the cue alone.


@dataclass(frozen=True)
class ObjectClass:
    id: str
    label: str
    cue: str


@dataclass(frozen=True)
class Department:
    id: str
    label: str
    objects: tuple[ObjectClass, ...]


def _oc(id_: str, label: str, cue: str) -> ObjectClass:
    return ObjectClass(id_, label, cue)


DEPARTMENTS: tuple[Department, ...] = (
    Department("textiles", "Textiles", (
        _oc("tapestry", "tapestry",
            "a large wool and silk hanging woven on a loom with a figural scene"),
        _oc("quilt", "quilt",
            "a layered and stitched bedcover made of pieced cotton patches"),
        _oc("sampler", "sampler",
            "a small embroidered linen practice cloth worked with an alphabet and motifs"),
        _oc("carpet", "carpet",
            "a hand-knotted wool floor covering with a deep pile and a medallion field"),
        _oc("shawl", "shawl",
            "a large woven wrap for the shoulders in fine twill wool"),
        _oc("lace", "lace",
            "openwork needle-made linen trimming worked in a floral repeat"),
        _oc("banner", "banner",
            "a stitched and painted silk processional flag with a pole sleeve"),
        _oc("purse", "purse",
            "a small drawstring bag worked in coloured silks"),
    )),
    Department("ceramics", "Ceramics", (
        _oc("vase", "vase",
            "a tall footed earthenware vessel for cut flowers"),
        _oc("jug", "jug",
            "a glazed pouring vessel with a loop handle and a pinched lip"),
        _oc("bowl", "bowl",
            "a shallow open dish thrown in stoneware and glazed inside"),
        _oc("tile", "tile",
            "a flat square glazed facing slab for a wall"),
        _oc("figurine", "figurine",
            "a small modelled and fired clay figure of a standing woman"),
        _oc("teapot", "teapot",
            "a lidded ceramic brewing vessel with a spout and a strainer"),
        _oc("charger", "charger",
            "an oversized decorated display plate made to hang on a wall"),
        _oc("drug_jar", "drug jar",
            "a waisted apothecary storage vessel with a tin glaze"),
    )),
    Department("metalwork", "Metalwork", (
        _oc("candlestick", "candlestick",
            "a footed brass stand made to hold a single taper"),
        _oc("chalice", "chalice",
            "a stemmed silver communion cup with a knopped stem"),
        _oc("casket", "casket",
            "a small hinged metal box with a lock plate and chased sides"),
        _oc("ewer", "ewer",
            "a tall narrow-necked pouring vessel raised in copper"),
        _oc("salver", "salver",
            "a flat footed serving tray with a moulded rim in silver"),
        _oc("censer", "censer",
            "a pierced bronze vessel on chains for burning incense"),
        _oc("buckle", "buckle",
            "a cast belt fastening with a pin and a decorated plate"),
        _oc("bell", "bell",
            "a cast bronze sounding vessel with a suspended clapper"),
    )),
    Department("furniture", "Furniture", (
        _oc("cabinet", "cabinet",
            "a fronted storage case on a stand, fitted inside with small drawers"),
        _oc("chair", "chair",
            "a joined wooden seat with a shaped back, scroll elbows and a drop-in squab"),
        _oc("table", "table",
            "a four-legged wooden surface on turned and stretchered supports"),
        _oc("chest", "chest",
            "a lidded storage box on a plinth with iron carrying handles"),
        _oc("settle", "settle",
            "a long joined wooden bench with a high panelled back and a lidded base"),
        _oc("bookcase", "bookcase",
            "an open shelved wooden case with adjustable shelves for books"),
        _oc("desk", "desk",
            "a writing surface with a sloped fall front and pigeonholes"),
        _oc("stool", "stool",
            "a low backless wooden seat on four splayed legs"),
    )),
    Department("glass", "Glass", (
        _oc("goblet", "goblet",
            "a blown and stemmed drinking cup with a conical body and a folded foot"),
        _oc("decanter", "decanter",
            "a stoppered clear vessel for serving wine, blown and polished"),
        _oc("window_panel", "window panel",
            "a leaded rectangle of coloured window glazing with painted detail"),
        _oc("paperweight", "paperweight",
            "a solid clear dome enclosing a coloured cane design"),
        _oc("hip_flask", "hip flask",
            "a flattened clear vessel with a screw cap for carrying spirits"),
        _oc("bead_string", "bead string",
            "a strung row of wound coloured beads, each lampworked at the flame"),
        _oc("cut_bowl", "cut bowl",
            "a wheel-cut lead crystal footed dish for serving fruit"),
        _oc("scent_bottle", "scent bottle",
            "a small clear container with a ground stopper for perfume"),
    )),
    Department("prints", "Prints and Drawings", (
        _oc("engraving", "engraving",
            "a line image cut into copper with a burin and printed on paper"),
        _oc("etching", "etching",
            "an image bitten into a plate with acid and printed on laid paper"),
        _oc("woodcut", "woodcut",
            "an image printed from a carved wooden block"),
        _oc("lithograph", "lithograph",
            "an image drawn on stone with greasy crayon and printed flat"),
        _oc("chalk_drawing", "chalk drawing",
            "a red study of a seated figure worked on toned paper"),
        _oc("watercolour", "watercolour",
            "a transparent wash view painted over a pencil underdrawing"),
        _oc("poster", "poster",
            "a colour-printed advertising bill for public display"),
        _oc("map_sheet", "map sheet",
            "an engraved leaf showing a coastline with a decorative cartouche"),
    )),
    Department("instruments", "Scientific Instruments", (
        _oc("astrolabe", "astrolabe",
            "a brass disc with engraved plates and a rotating rete for star positions"),
        _oc("microscope", "microscope",
            "a compound optical device on a brass pillar with a stage and mirror"),
        _oc("sundial", "sundial",
            "a portable engraved plate that tells the hour from a shadow"),
        _oc("theodolite", "theodolite",
            "a surveying device with a telescope over a graduated horizontal circle"),
        _oc("orrery", "orrery",
            "a geared brass model of the planets turning about a central sun"),
        _oc("barometer", "barometer",
            "a mercury column in a glazed case for reading air pressure"),
        _oc("sextant", "sextant",
            "a navigator's frame with a graduated arc and an index mirror for angles"),
        _oc("globe", "globe",
            "a printed and varnished sphere of the earth mounted in a meridian ring"),
    )),
    Department("arms", "Arms and Armour", (
        _oc("sword", "sword",
            "a straight double-edged blade with a cross guard and a wire-bound grip"),
        _oc("helmet", "helmet",
            "a plate head defence with a hinged visor and a neck guard"),
        _oc("breastplate", "breastplate",
            "a shaped steel defence for the torso with arm cut-outs"),
        _oc("dagger", "dagger",
            "a short single-handed thrusting blade with a shell guard"),
        _oc("pistol", "pistol",
            "a flintlock handgun with a chased lock plate and a walnut stock"),
        _oc("shield", "shield",
            "a faced and bordered hand-held defence carried on the arm"),
        _oc("crossbow", "crossbow",
            "a stocked bow with a nut and a trigger lever"),
        _oc("gauntlet", "gauntlet",
            "an articulated plate defence for the hand and wrist"),
    )),
)


@dataclass(frozen=True)
class Period:
    id: str
    label: str
    lo: int
    hi: int
    century: str


PERIODS: tuple[Period, ...] = (
    Period("p1500", "first half of the sixteenth century", 1500, 1549, "sixteenth century"),
    Period("p1550", "later sixteenth century", 1550, 1599, "sixteenth century"),
    Period("p1600", "early seventeenth century", 1600, 1649, "seventeenth century"),
    Period("p1650", "later seventeenth century", 1650, 1699, "seventeenth century"),
    Period("p1700", "early eighteenth century", 1700, 1749, "eighteenth century"),
    Period("p1750", "later eighteenth century", 1750, 1799, "eighteenth century"),
    Period("p1800", "early nineteenth century", 1800, 1849, "nineteenth century"),
    Period("p1850", "later nineteenth century", 1850, 1899, "nineteenth century"),
    Period("p1900", "first half of the twentieth century", 1900, 1949, "twentieth century"),
    Period("p1950", "later twentieth century", 1950, 1999, "twentieth century"),
)

YEAR_LO = PERIODS[0].lo
YEAR_HI = PERIODS[-1].hi


@dataclass(frozen=True)
class Region:
    id: str
    label: str
    cities: tuple[str, ...]


REGIONS: tuple[Region, ...] = (
    Region("english", "English", ("London", "Birmingham")),
    Region("french", "French", ("Paris", "Lyon")),
    Region("italian", "Italian", ("Florence", "Venice")),
    Region("german", "German", ("Nuremberg", "Dresden")),
    Region("dutch", "Dutch", ("Amsterdam", "Haarlem")),
    Region("spanish", "Spanish", ("Seville", "Valencia")),
    Region("swedish", "Swedish", ("Stockholm", "Gothenburg")),
    Region("russian", "Russian", ("St Petersburg", "Moscow")),
    Region("ottoman", "Ottoman", ("Istanbul", "Bursa")),
    Region("persian", "Persian", ("Isfahan", "Tabriz")),
    Region("indian", "Indian", ("Jaipur", "Lucknow")),
    Region("chinese", "Chinese", ("Guangzhou", "Suzhou")),
    Region("japanese", "Japanese", ("Kyoto", "Osaka")),
    Region("korean", "Korean", ("Seoul", "Gyeongju")),
    Region("mexican", "Mexican", ("Puebla", "Oaxaca")),
    Region("american", "American", ("Philadelphia", "Boston")),
)

# Register remarks that carry no facet information. No years and no place names:
# either would contaminate the period or region the state is meant to signal
# exactly once.
NEUTRAL_NOTES: tuple[str, ...] = (
    "Acquired by bequest; the register gives no vendor.",
    "Minor losses along one edge, otherwise sound.",
    "An old inventory number is inked on the underside.",
    "Stored flat in the reserve collection.",
    "A conservation report is on file; no treatment proposed.",
    "Labelled in an early hand, now largely illegible.",
    "Displayed once and returned to store.",
    "Measurements taken on accession and not since checked.",
)

N_DEPARTMENTS = len(DEPARTMENTS)
N_OBJECTS = len(DEPARTMENTS[0].objects)
N_PERIODS = len(PERIODS)
N_REGIONS = len(REGIONS)
LEVEL_WIDTHS: tuple[int, ...] = (N_DEPARTMENTS, N_OBJECTS, N_PERIODS, N_REGIONS)
N_LEAVES = N_DEPARTMENTS * N_OBJECTS * N_PERIODS * N_REGIONS

Path4 = tuple[int, int, int, int]

for _d in DEPARTMENTS:
    if len(_d.objects) != N_OBJECTS:
        raise AssertionError(
            f"{_d.id} has {len(_d.objects)} object classes; the tree is a product "
            f"and every department must have {N_OBJECTS}"
        )


# --------------------------------------------------------------------------
# Paths, ids and labels
# --------------------------------------------------------------------------
#
# A node is a prefix of a leaf's (department, object, period, region) index
# tuple. Its id spells the prefix out -- "d2.o4.p8.r10" -- so that an offline
# analysis reading only the JSONL log can recover the path, and hence the tree
# distance between what the model chose and what was correct, without importing
# this module. The separator matters: "d2.o4" is a prefix of "d2.o4.p8.r1" and
# also, as a bare string, of "d20.o4...", so prefix tests go through `path_of`
# and tuple comparison rather than `str.startswith`.

_PREFIX_KEYS = ("d", "o", "p", "r")


def node_id(prefix: Sequence[int]) -> str:
    return ".".join(f"{k}{v}" for k, v in zip(_PREFIX_KEYS, prefix))


def leaf_id(path: Path4) -> str:
    return node_id(path)


def path_of(node: str) -> tuple[int, ...]:
    """The index tuple a node id spells. Raises on anything it did not write."""
    parts = node.split(".")
    if not 1 <= len(parts) <= len(_PREFIX_KEYS):
        raise ValueError(f"not a taxonomy node id: {node!r}")
    out = []
    for key, part in zip(_PREFIX_KEYS, parts):
        if not part.startswith(key) or not part[1:].isdigit():
            raise ValueError(f"not a taxonomy node id: {node!r}")
        out.append(int(part[1:]))
    return tuple(out)


def facet_label(level: int, prefix: Sequence[int]) -> str:
    """The label of one level of a path, e.g. level 1 -> the object class."""
    if level == 0:
        return DEPARTMENTS[prefix[0]].label
    if level == 1:
        return DEPARTMENTS[prefix[0]].objects[prefix[1]].label
    if level == 2:
        return PERIODS[prefix[2]].label
    if level == 3:
        return REGIONS[prefix[3]].label
    raise ValueError(f"no level {level}; the tree has {len(LEVELS)}")


def leaf_label(path: Path4) -> str:
    return " > ".join(facet_label(i, path) for i in range(len(LEVELS)))


def children(prefix: Sequence[int]) -> list[dict]:
    """The option list for the level below `prefix`, as id/label dicts."""
    level = len(prefix)
    if not 0 <= level < len(LEVELS):
        raise ValueError(f"{node_id(prefix)!r} has no children")
    out = []
    for i in range(LEVEL_WIDTHS[level]):
        child = (*prefix, i)
        out.append({"id": node_id(child), "label": facet_label(level, child)})
    return out


def tree_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Levels from the leaf at which two paths diverge: 0 same, 4 different
    departments. This is the similarity scale E7's distractor conditions use."""
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return len(LEVELS) - common


def level_question(level: int, parent_label: str | None = None) -> str:
    """The wording for one level of a hierarchical descent."""
    if level == 0:
        return "Which collection department holds objects of this kind?"
    if level == 1:
        return (
            f"Within the {parent_label} department, which object class does this "
            "record describe?"
        )
    if level == 2:
        return "Which period band does this object date from?"
    if level == 3:
        return "Which regional school produced this object?"
    raise ValueError(f"no level {level}")


LEAF_QUESTION = "Which catalogue category should this accession record be filed under?"
CANDIDATE_QUESTION = "Could this accession record be filed under {label}?"
BAND_QUESTION = "Which date band does this accession record fall in?"


# --------------------------------------------------------------------------
# Tokens, for the cheap deterministic baseline
# --------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "the and with for from made this that into over under its was are has "
    "one two four his her but not all any".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def tokens_of(text: str) -> frozenset[str]:
    return frozenset(
        t for t in _WORD.findall(text.lower()) if len(t) >= 3 and t not in _STOPWORDS
    )


@lru_cache(maxsize=None)
def _facet_tokens(level: int, value: int, parent: int = 0) -> frozenset[str]:
    if level == 0:
        return tokens_of(DEPARTMENTS[value].label)
    if level == 1:
        return tokens_of(DEPARTMENTS[parent].objects[value].label)
    if level == 2:
        return tokens_of(PERIODS[value].label)
    return tokens_of(REGIONS[value].label)


@lru_cache(maxsize=None)
def label_tokens(path: Path4) -> frozenset[str]:
    d, o, p, r = path
    return (
        _facet_tokens(0, d)
        | _facet_tokens(1, o, d)
        | _facet_tokens(2, p)
        | _facet_tokens(3, r)
    )


def lookup_baseline(truth: Path4, paths: Sequence[Path4]) -> dict:
    """The stronger cheap baseline: a date parser and a gazetteer.

    Two of the four facets are lookups. The state gives a year, and the period
    band a year falls in is arithmetic; it gives a city, and the region a city
    belongs to is a table anyone would already have. Twenty lines of code
    therefore narrows the option set to those sharing the true period and
    region, and has nothing to say about the object class -- so it guesses
    uniformly among what is left.

    This is the baseline the rubric's Bad tier is about, and it is deliberately
    the strong one: reporting only the token overlap, which on these states is
    exactly chance because nothing ever breaks the tie, would let any result
    clear its cheap baseline by default. It is worth stating that the lookups
    are exact against this generator's own vocabulary, which a real gazetteer
    would not be, so the number is an upper bound on what the cheap route buys.
    """
    consistent = [p for p in paths if p[2] == truth[2] and p[3] == truth[3]]
    if not consistent:
        raise ValueError("the truth is always consistent with itself; option set is wrong")
    return {"expectation": 1.0 / len(consistent), "consistent": len(consistent)}


def overlap_baseline(state_tokens: frozenset[str], paths: Sequence[Path4],
                     truth: Path4) -> dict:
    """The 20-lines-of-code baseline: pick the option label sharing the most
    content words with the state.

    Ties are the normal case here rather than an edge case, because the state is
    written to avoid the label head words, so the honest score of a tie is
    1/len(tie) rather than the credit a fixed tie-break would hand it. `pick` is
    what a real implementation would output, taking the first in presentation
    order; `expectation` is what E7 averages.
    """
    if not paths:
        raise ValueError("overlap_baseline needs at least one option")
    scores = [len(state_tokens & label_tokens(p)) for p in paths]
    best = max(scores)
    tied = [p for p, s in zip(paths, scores) if s == best]
    return {
        "pick": leaf_id(tied[0]),
        "top_score": best,
        "tied": len(tied),
        "expectation": (1.0 / len(tied)) if truth in tied else 0.0,
    }


# --------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------

DATE_CUES: tuple[str, ...] = ("exact", "vague", "mixed")


@dataclass(frozen=True)
class Item:
    """One catalogued object: the state, its true leaf, and the facts a metric
    or a baseline needs. Independent of how many options it will be shown."""

    index: int
    seed: int
    path: Path4
    year: int
    date_cue: str
    state: dict
    tokens: frozenset[str] = field(repr=False)

    @property
    def leaf_id(self) -> str:
        return leaf_id(self.path)

    @property
    def leaf_label(self) -> str:
        return leaf_label(self.path)

    def facets(self) -> dict:
        d, o, p, r = self.path
        return {
            "department": DEPARTMENTS[d].id,
            "object_class": DEPARTMENTS[d].objects[o].id,
            "period": PERIODS[p].id,
            "region": REGIONS[r].id,
        }

    def meta(self) -> dict:
        return {
            "leaf_id": self.leaf_id,
            "leaf_label": self.leaf_label,
            "path": list(self.path),
            "facets": self.facets(),
            "year": self.year,
            "date_cue": self.date_cue,
            "n_leaves": N_LEAVES,
            "level_widths": list(LEVEL_WIDTHS),
        }


def _date_cue_for(index: int, mode: str) -> str:
    if mode == "mixed":
        # Alternating rather than random so that every cardinality and every
        # rubric size sees the same 50/50 split of easy and uncertain dates.
        return "exact" if index % 2 == 0 else "vague"
    return mode


def item_for(*, seed: int, index: int, date_cue: str = "exact") -> Item:
    """Item `index` of a draw. A function of (seed, index) and the date cue
    only: the same object is presented at every cardinality and every distractor
    profile, so those conditions are compared within item."""
    if date_cue not in DATE_CUES:
        raise ValueError(f"date_cue must be one of {DATE_CUES}, got {date_cue!r}")
    cue = _date_cue_for(index, date_cue)
    rng = rng_for(f"{NAME}:item", {}, seed, index)
    path: Path4 = (
        rng.randrange(N_DEPARTMENTS),
        rng.randrange(N_OBJECTS),
        rng.randrange(N_PERIODS),
        rng.randrange(N_REGIONS),
    )
    d, o, p, r = path
    period = PERIODS[p]
    year = rng.randint(period.lo, period.hi)
    city = rng.choice(REGIONS[r].cities)
    note = rng.choice(NEUTRAL_NOTES)
    dated = (
        str(year)
        if cue == "exact"
        else f"undated; the register places it broadly in the {period.century}"
    )
    state = {
        "record": "museum accession record",
        "description": DEPARTMENTS[d].objects[o].cue,
        "place_of_manufacture": city,
        "dated": dated,
        "register_note": note,
    }
    text = " ".join(str(v) for v in state.values())
    return Item(
        index=index,
        seed=seed,
        path=path,
        year=year,
        date_cue=cue,
        state=state,
        tokens=tokens_of(text),
    )


def items(*, seed: int, count: int, date_cue: str = "exact") -> list[Item]:
    return [item_for(seed=seed, index=i, date_cue=date_cue) for i in range(count)]


# --------------------------------------------------------------------------
# Distractor pools
# --------------------------------------------------------------------------
#
# Four named profiles plus the four tree distances. A profile names a pool; the
# draw is uniform without replacement from it. `uniform` is the default because
# it keeps the per-option similarity distribution identical at every
# cardinality, which is what makes the knee sweep a measurement of count alone.

DISTRACTOR_PROFILES: tuple[str, ...] = (
    "uniform", "far", "same_domain", "same_object", "same_period_region",
)

# Pools at or below this size are enumerated and sampled from directly; larger
# ones are drawn by rejection. Enumerating 10,239 leaves per instance would cost
# more than the call it feeds, and rejection from a pool that large collides
# about once in forty draws.
_ENUMERATE_LIMIT = 2048


def pool_size(profile: str) -> int:
    """How many distractors a profile has available, truth excluded."""
    if profile in ("uniform",):
        return N_LEAVES - 1
    if profile in ("far", "d4"):
        return (N_DEPARTMENTS - 1) * N_OBJECTS * N_PERIODS * N_REGIONS
    if profile in ("same_domain", "d3_plus"):
        return N_OBJECTS * N_PERIODS * N_REGIONS - 1
    if profile in ("same_object",):
        return N_PERIODS * N_REGIONS - 1
    if profile == "same_period_region":
        # Every other object class, at the truth's own period and region. The
        # date parser and the gazetteer buy nothing here, so this is the pool
        # that asks whether the description was read at all. 63 distractors
        # exist, which is exactly the 64-option condition and no more.
        return N_DEPARTMENTS * N_OBJECTS - 1
    if profile == "d1":
        return N_REGIONS - 1
    if profile == "d2":
        return (N_PERIODS - 1) * N_REGIONS
    if profile == "d3":
        return (N_OBJECTS - 1) * N_PERIODS * N_REGIONS
    raise ValueError(f"unknown distractor profile {profile!r}")


def _enumerate(profile: str, truth: Path4) -> list[Path4]:
    d0, o0, p0, r0 = truth
    if profile == "same_domain":
        return [
            (d0, o, p, r)
            for o in range(N_OBJECTS)
            for p in range(N_PERIODS)
            for r in range(N_REGIONS)
            if (o, p, r) != (o0, p0, r0)
        ]
    if profile == "same_object":
        return [
            (d0, o0, p, r)
            for p in range(N_PERIODS)
            for r in range(N_REGIONS)
            if (p, r) != (p0, r0)
        ]
    if profile == "same_period_region":
        return [
            (d, o, p0, r0)
            for d in range(N_DEPARTMENTS)
            for o in range(N_OBJECTS)
            if (d, o) != (d0, o0)
        ]
    if profile == "d1":
        return [(d0, o0, p0, r) for r in range(N_REGIONS) if r != r0]
    if profile == "d2":
        return [
            (d0, o0, p, r)
            for p in range(N_PERIODS)
            for r in range(N_REGIONS)
            if p != p0
        ]
    if profile == "d3":
        return [
            (d0, o, p, r)
            for o in range(N_OBJECTS)
            for p in range(N_PERIODS)
            for r in range(N_REGIONS)
            if o != o0
        ]
    raise ValueError(f"{profile!r} is not enumerated")


def _draw(profile: str, truth: Path4, rng) -> Path4:
    d0 = truth[0]
    if profile == "uniform":
        return (
            rng.randrange(N_DEPARTMENTS),
            rng.randrange(N_OBJECTS),
            rng.randrange(N_PERIODS),
            rng.randrange(N_REGIONS),
        )
    if profile in ("far", "d4"):
        d = rng.randrange(N_DEPARTMENTS - 1)
        if d >= d0:
            d += 1
        return (d, rng.randrange(N_OBJECTS), rng.randrange(N_PERIODS),
                rng.randrange(N_REGIONS))
    raise ValueError(f"{profile!r} is not drawn by rejection")


def sample_distractors(truth: Path4, *, profile: str, count: int, rng) -> list[Path4]:
    """`count` distinct leaves from `profile`'s pool, never the truth.

    Raises when the pool is too small rather than repeating a distractor: a
    condition with duplicate options would be rejected on the wire for a
    duplicate criteria key, halfway through, after the earlier calls were paid
    for.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []
    size = pool_size(profile)
    if count > size:
        raise ValueError(
            f"profile {profile!r} holds {size} distractors, {count} requested; "
            "the nearest pools are small and a condition cannot exceed them"
        )
    if size <= _ENUMERATE_LIMIT:
        return rng.sample(_enumerate(profile, truth), count)
    seen = {truth}
    out: list[Path4] = []
    while len(out) < count:
        cand = _draw(profile, truth, rng)
        if cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
    return out


def option_paths(truth: Path4, *, n_options: int, profile: str, rng) -> list[Path4]:
    """The truth plus `n_options - 1` distractors, in draw order.

    Not shuffled here: the caller shuffles through `instances.shuffled_options`,
    which is the one place in the harness that randomises option order, so a
    condition that deliberately fixes order (there is none in E7) is visible as
    the absence of that call rather than as a missing argument here.
    """
    if not 2 <= n_options <= 255:
        raise ValueError(f"choice takes 2..255 options, got {n_options}")
    return [truth, *sample_distractors(truth, profile=profile, count=n_options - 1, rng=rng)]


def options_for(paths: Iterable[Path4]) -> list[dict]:
    return [{"id": leaf_id(p), "label": leaf_label(p)} for p in paths]


# --------------------------------------------------------------------------
# The simulated retriever behind E7's "pre-filtered shortlist"
# --------------------------------------------------------------------------


def shortlist_paths(truth: Path4, *, size: int, rng) -> list[Path4]:
    """A 255-item shortlist enriched toward the true subtree.

    E7 compares hierarchical descent against "a flat choice over a 255-item
    pre-filtered shortlist". The pre-filter in a real system is an embedding
    retriever, which this harness does not have, and the token-overlap
    retriever it does have is near-random on these states by design -- which
    would make the shortlist a uniform draw and the comparison meaningless.

    So the retriever is simulated with a declared composition: every one of the
    15 leaves that differ from the truth only in region, then a quarter of the
    remainder from the same object class, a bit over a third from the same
    department, and the rest from elsewhere. That is a retriever with good but
    not perfect recall of the right neighbourhood. It is stated as simulated
    everywhere it is reported, and E7 prints the uniform 255-option condition
    from its knee sweep beside it, so the cost of the enrichment is visible
    rather than assumed.
    """
    if not 2 <= size <= 255:
        raise ValueError(f"shortlist size must be 2..255, got {size}")
    n1 = min(pool_size("d1"), size - 1)
    rest = size - 1 - n1
    n2 = min(int(round(0.25 * rest)), pool_size("d2"))
    n3 = min(int(round(0.37 * rest)), pool_size("d3"))
    n4 = rest - n2 - n3
    if n4 < 0:  # only reachable if a pool cap pushed the earlier buckets over
        n3 += n4
        n4 = 0
    out = [truth]
    for kind, count in (("d1", n1), ("d2", n2), ("d3", n3), ("d4", n4)):
        out.extend(sample_distractors(truth, profile=kind, count=count, rng=rng))
    return out


# --------------------------------------------------------------------------
# Ordered date bands, for the score question
# --------------------------------------------------------------------------

YEAR_SPAN = YEAR_HI + 1 - YEAR_LO

# The endpoint rejects a score question with more than ten rubric points --
# "Too many score levels. Must have at most 10 levels.", HTTP 400. That is
# undocumented and it is a much lower ceiling than choice's 255, so the sizes
# below stop at 10 and E7 probes the cap once rather than spending a whole
# condition discovering it. They are the divisors of the 500-year range that
# fit under it, which is what keeps the bands equal and their labels exact.
MAX_SCORE_LEVELS = 10
RUBRIC_SIZES: tuple[int, ...] = (2, 4, 5, 10)
RUBRIC_CAP_PROBE = 20


def period_rubric(k: int) -> list[dict]:
    """`k` contiguous date bands covering the whole range, ordered low to high.

    A genuinely ordered rubric, which is what `score` is for, and one whose
    length can be set to any divisor of the range -- giving E7 a cardinality
    axis for score questions to match the option-count axis for choice.
    """
    if k < 2:
        raise ValueError(f"a rubric needs at least 2 points, got {k}")
    if YEAR_SPAN % k:
        raise ValueError(
            f"{k} does not divide the {YEAR_SPAN}-year range; bands would not be "
            "equal and the labels would round into each other"
        )
    width = YEAR_SPAN // k
    return [
        {"id": f"b{i}", "label": f"{YEAR_LO + i * width}-{YEAR_LO + (i + 1) * width - 1}"}
        for i in range(k)
    ]


def band_for_year(year: int, k: int) -> str:
    if not YEAR_LO <= year <= YEAR_HI:
        raise ValueError(f"{year} is outside {YEAR_LO}-{YEAR_HI}")
    width = YEAR_SPAN // k
    return f"b{(year - YEAR_LO) // width}"


# --------------------------------------------------------------------------
# Generator contract
# --------------------------------------------------------------------------

KEY_CHOICE = "q_category"
KEY_SCORE = "q_date_band"

DEFAULT_DIFFICULTY: dict[str, Any] = {
    "n_options": 64,
    "distractor_profile": "uniform",
    "date_cue": "exact",
    "question_type": "choice",
}

CARDINALITIES: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 200, 255)


def difficulty_sweep() -> list[dict]:
    """Easy to hard is few options to many.

    The ladder brackets the knee the plan predicts at 30-60 rather than
    sampling only the comfortable end: 2 and 4 establish the ceiling this task
    has at all, and 200 and 255 sit either side of the documented cap so the
    last step is a real one rather than an extrapolation.
    """
    return [
        {**DEFAULT_DIFFICULTY, "n_options": k} for k in CARDINALITIES
    ]


def _normalize(difficulty: dict) -> dict:
    d = {**DEFAULT_DIFFICULTY, **difficulty}
    qtype = d["question_type"]
    if qtype not in ("choice", "score"):
        raise ValueError(f"question_type must be choice or score, got {qtype!r}")
    if d["date_cue"] not in DATE_CUES:
        raise ValueError(f"date_cue must be one of {DATE_CUES}, got {d['date_cue']!r}")
    if qtype == "choice":
        if d["distractor_profile"] not in DISTRACTOR_PROFILES:
            raise ValueError(
                f"distractor_profile must be one of {DISTRACTOR_PROFILES}, "
                f"got {d['distractor_profile']!r}"
            )
        n = int(d["n_options"])
        if not 2 <= n <= 255:
            raise ValueError(f"n_options must be 2..255, got {n}")
        d["n_options"] = n
    else:
        k = int(d.get("rubric_size") or 0)
        if k < 2:
            raise ValueError("a score difficulty needs rubric_size >= 2")
        d["rubric_size"] = k
        # Not part of a score condition; carried in the difficulty dict it would
        # make two otherwise identical conditions look different.
        d.pop("n_options", None)
        d.pop("distractor_profile", None)
    return d


def _instance(difficulty: dict, seed: int, index: int) -> Instance:
    item = item_for(seed=seed, index=index, date_cue=difficulty["date_cue"])
    rng = rng_for(NAME, difficulty, seed, index)
    meta: dict[str, Any] = item.meta()
    meta["random_baseline"] = None

    if difficulty["question_type"] == "score":
        k = difficulty["rubric_size"]
        rubric = period_rubric(k)
        truth_id = band_for_year(item.year, k)
        questions = {KEY_SCORE: score(BAND_QUESTION, rubric)}
        truth: dict[str, Any] = {KEY_SCORE: truth_id}
        meta.update(
            {
                "rubric_size": k,
                "band_width_years": YEAR_SPAN // k,
                "random_baseline": 1.0 / k,
                "truth_band": truth_id,
            }
        )
    else:
        n = difficulty["n_options"]
        profile = difficulty["distractor_profile"]
        paths = option_paths(item.path, n_options=n, profile=profile, rng=rng)
        q = shuffled_options(choice(LEAF_QUESTION, options_for(paths)), rng)
        shown = [path_of(o["id"]) for o in q["options"]]
        questions = {KEY_CHOICE: q}
        truth = {KEY_CHOICE: item.leaf_id}
        distances: dict[str, int] = {}
        for p in shown:
            if p == item.path:
                continue
            distances[str(tree_distance(p, item.path))] = (
                distances.get(str(tree_distance(p, item.path)), 0) + 1
            )
        base = overlap_baseline(item.tokens, shown, item.path)
        lookup = lookup_baseline(item.path, shown)
        meta.update(
            {
                "n_options": n,
                "distractor_profile": profile,
                "random_baseline": 1.0 / n,
                "option_ids": [o["id"] for o in q["options"]],
                # Position of the correct option as presented. The plan lists
                # position bias as a behaviour to watch for, and it can only be
                # checked if where the answer sat is in the log.
                "correct_position": shown.index(item.path),
                "distractor_distance_counts": distances,
                "baseline_overlap_pick": base["pick"],
                "baseline_overlap_expectation": base["expectation"],
                "baseline_overlap_tied": base["tied"],
                "baseline_lookup_expectation": lookup["expectation"],
                "baseline_lookup_consistent": lookup["consistent"],
            }
        )

    inst = Instance(
        generator=NAME,
        difficulty=difficulty,
        seed=seed,
        index=index,
        state=item.state,
        questions=questions,
        truth=truth,
        meta=meta,
    )
    inst.validate()
    return inst


def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]:
    d = _normalize(difficulty)
    return [_instance(d, seed, i) for i in range(count)]


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    assert N_LEAVES == 10240, N_LEAVES
    # Every leaf label is distinct, which the wire requires of choice ids and
    # which a compositional label set does not give for free.
    seen_labels = set()
    for d in range(N_DEPARTMENTS):
        for o in range(N_OBJECTS):
            lab = f"{DEPARTMENTS[d].label}>{DEPARTMENTS[d].objects[o].label}"
            assert lab not in seen_labels, lab
            seen_labels.add(lab)
    assert len({o.label for dep in DEPARTMENTS for o in dep.objects}) == 64
    assert len({p.label for p in PERIODS}) == N_PERIODS
    assert len({r.label for r in REGIONS}) == N_REGIONS

    # No state ever contains the head word of any of its own labels.
    leaks = 0
    for i in range(400):
        it = item_for(seed=7, index=i)
        for lvl in range(len(LEVELS)):
            head = facet_label(lvl, it.path).split()[-1].lower()
            if head in it.tokens:
                leaks += 1
                print("leak", it.leaf_label, head)
    assert leaks == 0, f"{leaks} label head words appeared in their own state"

    # A cue must not borrow another object class's head word either: the state
    # is meant to name one leaf, and "a panel of lace" pulling in the glass
    # department's window panel makes the correct answer arguable.
    heads: dict[str, tuple[str, str]] = {}
    for dep in DEPARTMENTS:
        for obj in dep.objects:
            heads.setdefault(obj.label.split()[-1].lower(), (dep.id, obj.id))
    borrowed = 0
    for dep in DEPARTMENTS:
        for obj in dep.objects:
            for tok in tokens_of(obj.cue):
                owner = heads.get(tok)
                if owner is not None and owner != (dep.id, obj.id):
                    borrowed += 1
                    print("cue borrows", tok, "from", owner, "in", dep.id, obj.id)
    assert borrowed == 0, f"{borrowed} cues borrow another object class's head word"

    # No cue may name a department other than its own. Naming its own is the
    # point: the material in the cue is what makes level 1 of a descent
    # answerable.
    for dep in DEPARTMENTS:
        for other in DEPARTMENTS:
            if other.id == dep.id:
                continue
            for obj in dep.objects:
                assert not (tokens_of(obj.cue) & tokens_of(other.label)), obj.id

    for node in ("d0", "d0.o1", "d7.o7.p9.r15"):
        assert node_id(path_of(node)) == node

    rng = random.Random(1)
    truth = (3, 4, 5, 6)
    for prof in DISTRACTOR_PROFILES:
        n = min(64, pool_size(prof) + 1)
        paths = option_paths(truth, n_options=n, profile=prof, rng=rng)
        assert len(set(paths)) == n
        assert paths[0] == truth
    sl = shortlist_paths(truth, size=255, rng=rng)
    assert len(set(sl)) == 255 and sl[0] == truth
    from collections import Counter

    comp = Counter(tree_distance(p, truth) for p in sl[1:])
    print("shortlist composition by tree distance:", dict(sorted(comp.items())))

    for k in RUBRIC_SIZES:
        rub = period_rubric(k)
        assert len(rub) == k and len({r["label"] for r in rub}) == k
        assert band_for_year(YEAR_LO, k) == "b0"
        assert band_for_year(YEAR_HI, k) == f"b{k - 1}"

    for diff in difficulty_sweep():
        insts = generate(difficulty=diff, seed=11, count=5)
        assert len(insts) == 5
    insts = generate(
        difficulty={"question_type": "score", "rubric_size": 20, "date_cue": "mixed"},
        seed=11,
        count=6,
    )
    assert {i.meta["date_cue"] for i in insts} == {"exact", "vague"}

    sample = generate(difficulty={"n_options": 64}, seed=3, count=200)
    overlap = sum(i.meta["baseline_overlap_expectation"] for i in sample) / len(sample)
    lookup = sum(i.meta["baseline_lookup_expectation"] for i in sample) / len(sample)
    print(
        f"cheap baselines at 64 options: token overlap {overlap:.3f}, "
        f"date+gazetteer lookup {lookup:.3f} (chance {1 / 64:.3f})"
    )
    print("taxonomy self-check ok")
