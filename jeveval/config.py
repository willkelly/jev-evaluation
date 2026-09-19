"""Run-wide configuration: seeds, sample sizes, pricing, paths.

Every number the plan fixes lives here, in one place, so the report can state
what was actually run and a reviewer can see at a glance whether a sample size
was quietly shrunk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Endpoint and model
# --------------------------------------------------------------------------

API_URL = os.environ.get("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone")

# The alias we request. The *versioned* string the server returns (e.g.
# "jev-1.13.0") is recorded on every logged call and reproduced in the report
# header; results are only meaningful against a pinned version.
MODEL_ALIAS = os.environ.get("JEV_MODEL", "jev-latest")

# $42 per 10^9 input tokens, output free. Used for cost-per-decision.
USD_PER_INPUT_TOKEN = 42.0 / 1e9
USD_PER_OUTPUT_TOKEN = 0.0

# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------

# One master seed. Each experiment derives its own stream from it by name, so
# adding an experiment does not perturb the instances of any other.
MASTER_SEED = int(os.environ.get("JEV_SEED", "20260919"))


def seed_for(experiment: str, condition: str = "") -> int:
    """A stable per-experiment, per-condition seed derived from MASTER_SEED."""
    import hashlib

    h = hashlib.sha256(f"{MASTER_SEED}:{experiment}:{condition}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# --------------------------------------------------------------------------
# Sample sizes
# --------------------------------------------------------------------------


@dataclass
class SampleSizes:
    """Sizes the plan fixes. See `scale` for how a dry run shrinks them."""

    # 10 equal-width bins need enough mass per bin for ECE to be stable.
    calibration: int = 500
    # Enough for a pure accuracy comparison, not enough for ECE.
    accuracy: int = 200
    # E3 and E7 conditions.
    batching: int = 300
    cardinality: int = 300
    # E1 determinism: 50 states x 30 repetitions.
    determinism_states: int = 50
    determinism_reps: int = 30
    # E5 enrollment.
    enrollment: int = 500
    # E6 coherence triples.
    coherence_triples: int = 500


SIZES = SampleSizes()

# A global multiplier for dry runs. 1.0 is the plan as written. Anything below
# 1.0 is recorded in the report header and beside every affected number, because
# a shrunk sample size that is not stated is how a noisy result gets reported as
# a finding.
SCALE = float(os.environ.get("JEV_SCALE", "1.0"))


def n(size: int) -> int:
    """Apply the dry-run scale factor, never going below 2."""
    return max(2, int(round(size * SCALE)))


# --------------------------------------------------------------------------
# Concurrency and retry
# --------------------------------------------------------------------------


@dataclass
class RateConfig:
    """Adaptive concurrency. Rate limits during early access are undocumented
    and, per the plan, more likely to bind than cost -- so the client starts
    conservative, climbs while calls succeed, and backs off hard on 429."""

    start_concurrency: int = 5
    # The ceiling has to be high enough for the limiter to actually find the
    # endpoint's limit rather than sit at a constant this file chose. Across
    # 52,200 consecutive calls the limiter backed off zero times at a cap of 24,
    # so the "sustained rate achieved" the plan asks to be reported as a finding
    # was a property of the cap, not of the endpoint. Additive increase and the
    # halving on the first 429 still bound how fast it climbs and how hard it
    # retreats.
    max_concurrency: int = 128
    min_concurrency: int = 1
    # Multiplicative-decrease on 429 / 5xx, additive-increase on sustained success.
    backoff_factor: float = 0.5
    # Kept for compatibility; the limiter now climbs after roughly `limit`
    # successes, with this as the floor, so recovery costs a constant number of
    # round trips instead of a constant number of calls.
    increase_after_successes: int = 50
    min_successes_to_increase: int = 12
    # A burst of rejections from requests already in flight is one throttling
    # event seen many times. Only the first decrease within this window counts.
    throttle_cooldown: float = 2.0
    max_retries: int = 6
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    request_timeout: float = 120.0


RATE = RateConfig(
    # Overridable so several processes sharing the endpoint -- parallel agents
    # during development, or a rehearsal running next to a real run -- can be
    # held well below the sustained rate a single full run is allowed to climb
    # to.
    start_concurrency=int(os.environ.get("JEV_START_CONCURRENCY", "5")),
    max_concurrency=int(
        os.environ.get("JEV_MAX_CONCURRENCY", str(RateConfig.max_concurrency))
    ),
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("JEV_RUNS", ROOT / "runs"))


def run_dir(run_id: str) -> Path:
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "plots").mkdir(exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Tier thresholds (the plan's scoring rubric, in one machine-readable place)
# --------------------------------------------------------------------------

# Calibration tiers, from "On calibration tiers specifically" in the plan.
ECE_TIERS = [
    (0.02, "Perfect"),
    (0.05, "Superhuman"),
    (0.15, "Human"),
    (0.30, "Bad"),
]
ECE_WORST = "Doesn't work"
