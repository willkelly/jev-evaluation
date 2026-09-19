"""Unit tests for the hand-written core: wire, instances, logstore, config,
predictions, and the smoke baselines.

These modules were not produced by the same build-and-adversarially-verify pass
the generators and metrics went through, so they are the least reviewed code in
the harness while also being the code every experiment depends on. A wire
translation bug would silently change what the model is asked; a logstore bug
would silently change what is scored.

Standard library unittest, so this runs with no extra dependency:

    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jeveval import config, logstore, predictions, wire
from jeveval.instances import (
    Instance,
    choice,
    noul,
    rng_for,
    score,
    shuffled_options,
    shuffled_questions,
)


class TestWireRequests(unittest.TestCase):
    def test_noul_uses_instructions_not_question(self):
        w = wire.question_to_wire(noul("Is this billing?"))
        self.assertEqual(w["type"], "noul")
        self.assertEqual(w["instructions"], "Is this billing?")
        # The endpoint rejects "question"; sending it would be silently ignored
        # at best and a 400 at worst.
        self.assertNotIn("question", w)

    def test_noul_requires_a_prompt_or_criteria(self):
        with self.assertRaises(wire.WireError):
            wire.question_to_wire({"type": "noul"})
        # criteria alone is legal -- the server accepts {"true":..,"false":..}
        w = wire.question_to_wire({"type": "noul", "criteria": {"true": "a", "false": "b"}})
        self.assertEqual(w["criteria"], {"true": "a", "false": "b"})

    def test_choice_options_become_criteria_keys_in_order(self):
        opts = [{"id": f"o{i}", "label": f"L{i}"} for i in range(5)]
        w = wire.question_to_wire(choice("pick", opts))
        # Option order on the wire is JSON key order is dict insertion order.
        # If this stops holding, every order-effect result in the plan is void.
        self.assertEqual(list(w["criteria"].keys()), ["o0", "o1", "o2", "o3", "o4"])
        self.assertEqual(w["criteria"]["o3"], "L3")

    def test_choice_cap_is_255(self):
        ok = [{"id": f"o{i}", "label": ""} for i in range(255)]
        wire.question_to_wire(choice("pick", ok))
        with self.assertRaises(ValueError):
            choice("pick", [{"id": f"o{i}", "label": ""} for i in range(256)])

    def test_choice_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            choice("pick", [{"id": "a", "label": "x"}, {"id": "a", "label": "y"}])

    def test_score_rubric_becomes_ordered_label_list(self):
        w = wire.question_to_wire(
            score("how urgent", [{"id": "lo", "label": "low"}, {"id": "hi", "label": "high"}])
        )
        self.assertEqual(w["criteria"], ["low", "high"])

    def test_score_rejects_duplicate_labels(self):
        # The response identifies rubric points by index and the legend echoes
        # labels, so duplicates make a returned point ambiguous.
        with self.assertRaises(wire.WireError):
            wire.question_to_wire(
                {"type": "score", "question": "q",
                 "rubric": [{"id": "a", "label": "same"}, {"id": "b", "label": "same"}]}
            )

    def test_wire_passthrough_overrides(self):
        w = wire.question_to_wire(
            {"type": "noul", "question": "q", "wire_instructions": "replaced"}
        )
        self.assertEqual(w["instructions"], "replaced")

    def test_build_request_preserves_question_key_order(self):
        qs = {"zzz": noul("a"), "aaa": noul("b"), "mmm": noul("c")}
        body = wire.build_request(state="s", questions=qs, model="jev-latest")
        self.assertEqual(list(body["questions"].keys()), ["zzz", "aaa", "mmm"])
        self.assertEqual(body["model"], "jev-latest")

    def test_build_request_rejects_empty_questions(self):
        with self.assertRaises(wire.WireError):
            wire.build_request(state="s", questions={}, model="m")


class TestWireResponses(unittest.TestCase):
    def test_noul_answer_has_no_confidence(self):
        a = wire.parse_answer({"type": "noul", "noul": 0.98}, noul("q"))
        self.assertAlmostEqual(a["p"], 0.98)
        self.assertTrue(a["predicted"])
        # Explicitly None rather than absent, so a caller cannot mistake a
        # missing key for a real confidence of 0.
        self.assertIsNone(a["confidence"])

    def test_noul_boundary_is_inclusive_at_half(self):
        self.assertTrue(wire.parse_answer({"type": "noul", "noul": 0.5}, noul("q"))["predicted"])
        self.assertFalse(wire.parse_answer({"type": "noul", "noul": 0.49}, noul("q"))["predicted"])

    def test_choice_answer(self):
        q = choice("pick", [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}])
        a = wire.parse_answer(
            {"type": "choice", "choice": "a", "confidence": 1.0,
             "probabilities": {"a": 0.9, "b": 0.1}}, q
        )
        self.assertEqual(a["chosen"], "a")
        self.assertAlmostEqual(a["max_probability"], 0.9)

    def test_score_maps_wire_indices_back_to_rubric_ids(self):
        q = score("urgency", [{"id": "lo", "label": "low"},
                              {"id": "me", "label": "medium"},
                              {"id": "hi", "label": "high"}])
        a = wire.parse_answer(
            {"type": "score", "score": 1.6, "confidence": 0.4,
             "legend": {"0": "low", "1": "medium", "2": "high"},
             "probabilities": {"0": 0.0, "1": 0.39, "2": 0.61}}, q
        )
        # The caller supplied ids lo/me/hi; the wire speaks indices. Getting this
        # mapping wrong would score every score question against the wrong point.
        self.assertEqual(a["probabilities"], {"lo": 0.0, "me": 0.39, "hi": 0.61})
        self.assertEqual(a["chosen"], "hi")

    def test_missing_answer_is_an_error_not_an_empty_result(self):
        qs = {"a": noul("x"), "b": noul("y")}
        with self.assertRaises(wire.WireError):
            wire.parse_response({"answers": {"a": {"type": "noul", "noul": 0.5}}}, qs)

    def test_usage_and_version(self):
        r = {"model": "jev-1.13.0", "usage": {"input_tokens": 297, "output_tokens": 21}}
        self.assertEqual(wire.model_version(r), "jev-1.13.0")
        self.assertEqual(wire.usage(r), (297, 21))
        self.assertEqual(wire.usage({}), (0, 0))


class TestInstances(unittest.TestCase):
    def _inst(self, **kw):
        base = dict(
            generator="t", difficulty={"d": 1}, seed=1, index=0, state="s",
            questions={"q": noul("x")}, truth={"q": True},
        )
        base.update(kw)
        return Instance(**base)

    def test_validate_catches_missing_truth(self):
        with self.assertRaises(ValueError):
            self._inst(truth={}).validate()

    def test_validate_catches_wrong_truth_type(self):
        with self.assertRaises(ValueError):
            self._inst(truth={"q": "yes"}).validate()

    def test_validate_catches_answer_not_among_options(self):
        q = choice("pick", [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}])
        with self.assertRaises(ValueError):
            self._inst(questions={"q": q}, truth={"q": "c"}).validate()

    def test_instance_id_is_stable_and_distinct(self):
        a = self._inst()
        b = self._inst()
        c = self._inst(index=1)
        self.assertEqual(a.instance_id, b.instance_id)
        self.assertNotEqual(a.instance_id, c.instance_id)

    def test_rng_is_determined_by_coordinates_not_by_draw_order(self):
        # Instance 5 must be reproducible without drawing 0..4 first.
        direct = rng_for("g", {"x": 1}, 7, 5).random()
        after_others = None
        for i in range(6):
            v = rng_for("g", {"x": 1}, 7, i).random()
            if i == 5:
                after_others = v
        self.assertEqual(direct, after_others)

    def test_shuffles_do_not_lose_or_duplicate(self):
        opts = [{"id": f"o{i}", "label": str(i)} for i in range(20)]
        q = choice("pick", opts)
        s = shuffled_options(q, rng_for("g", {}, 1, 0))
        self.assertEqual({o["id"] for o in s["options"]}, {o["id"] for o in opts})
        self.assertEqual(len(s["options"]), 20)
        qs = {f"q{i}": noul(str(i)) for i in range(10)}
        sq = shuffled_questions(qs, rng_for("g", {}, 2, 0))
        self.assertEqual(set(sq), set(qs))

    def test_shuffle_does_not_mutate_the_original(self):
        opts = [{"id": f"o{i}", "label": str(i)} for i in range(10)]
        q = choice("pick", opts)
        before = [o["id"] for o in q["options"]]
        shuffled_options(q, rng_for("g", {}, 3, 0))
        self.assertEqual([o["id"] for o in q["options"]], before)


class TestLogstore(unittest.TestCase):
    def _record(self, outcome="ok", **kw):
        rec = {
            "ts": 1.0, "experiment": "E1", "condition": "c", "instance_id": "i",
            "outcome": outcome, "http_status": 200, "latency_s": 0.2,
            "input_tokens": 100, "output_tokens": 10, "model_version": "jev-1.13.0",
            "request": {"questions": {"q": {"type": "noul", "instructions": "x"}}},
            "response": {"answers": {"q": {"type": "noul", "noul": 0.8}}},
            "meta": {"truth": {"q": True}},
        }
        rec.update(kw)
        return rec

    def test_terminal_excludes_retry_history(self):
        recs = [self._record("retry"), self._record("retry"), self._record("ok")]
        self.assertEqual(len(list(logstore.terminal(recs))), 1)
        self.assertEqual(len(list(logstore.successful(recs))), 1)

    def test_answers_reparse_from_the_log(self):
        a = logstore.answers_of(self._record())
        self.assertAlmostEqual(a["q"]["p"], 0.8)
        self.assertTrue(a["q"]["predicted"])

    def test_score_answers_reparse_via_wire_criteria(self):
        rec = self._record(
            request={"questions": {"q": {"type": "score", "criteria": ["low", "mid", "high"]}}},
            response={"answers": {"q": {"type": "score", "score": 2.0,
                                        "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}}},
        )
        a = logstore.answers_of(rec)["q"]
        self.assertEqual(a["chosen"], "high")
        self.assertAlmostEqual(a["probabilities"]["mid"], 0.2)

    def test_choice_option_order_recoverable_for_position_analysis(self):
        rec = self._record(
            request={"questions": {"q": {"type": "choice",
                                         "criteria": {"b": "", "a": "", "c": ""}}}},
            response={"answers": {"q": {"type": "choice", "choice": "a",
                                        "probabilities": {"a": 1.0}}}},
        )
        self.assertEqual(logstore.answers_of(rec)["q"]["option_order"], ["b", "a", "c"])

    def test_truth_survives_the_round_trip_to_disk(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calls.jsonl"
            p.write_text(json.dumps(self._record()) + "\n")
            recs = list(logstore.read(p))
            self.assertEqual(recs[0]["meta"]["truth"], {"q": True})

    def test_truncated_final_line_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calls.jsonl"
            p.write_text(json.dumps(self._record()) + "\n" + '{"partial": ')
            recs, stats = logstore.read_with_stats(p)
            self.assertEqual(len(recs), 1)
            self.assertEqual(stats["unparseable_lines"], 1)

    def test_run_stats_counts_failures_separately(self):
        recs = [self._record("ok"), self._record("error", http_status=400),
                self._record("retry", http_status=429)]
        st = logstore.run_stats(recs)
        self.assertEqual(st["calls"], 2)
        self.assertEqual(st["successful"], 1)
        self.assertEqual(st["failed"], 1)
        self.assertEqual(st["retries"], 1)


class TestConfigAndPredictions(unittest.TestCase):
    def test_seed_for_is_stable_and_separates_conditions(self):
        self.assertEqual(config.seed_for("E1", "a"), config.seed_for("E1", "a"))
        self.assertNotEqual(config.seed_for("E1", "a"), config.seed_for("E1", "b"))
        self.assertNotEqual(config.seed_for("E1", "a"), config.seed_for("E2", "a"))

    def test_n_never_collapses_below_two(self):
        self.assertGreaterEqual(config.n(500), 2)
        self.assertGreaterEqual(config.n(1), 2)

    def test_every_prediction_is_present_and_attributed(self):
        self.assertEqual(len(predictions.TABLE), 28)
        self.assertEqual([p.id for p in predictions.TABLE][:3], ["P1", "P2", "P3"])
        for e, count in [("E1", 3), ("E2", 3), ("E3", 3), ("E9", 4)]:
            self.assertEqual(len(predictions.for_experiment(e)), count, e)

    def test_untestable_is_excluded_from_hit_rate_not_counted_as_a_miss(self):
        s = predictions.score([
            {"id": "P1", "verdict": "right"},
            {"id": "P2", "verdict": "wrong"},
            {"id": "P25", "verdict": "untestable"},
        ])
        self.assertEqual(s["n_testable"], 2)
        self.assertAlmostEqual(s["hit_rate"], 0.5)
        self.assertEqual(s["untestable"], ["P25"])
        self.assertIn("P28", s["unscored"])


class TestSmokeBaselines(unittest.TestCase):
    def test_baseline_derives_from_question_structure(self):
        from jeveval.smoke import _baselines

        opts = [{"id": f"d{i}", "label": ""} for i in range(8)]
        insts = [
            Instance(generator="s", difficulty={}, seed=1, index=i, state="x",
                     questions={"q": choice("pick", opts)}, truth={"q": "d0"})
            for i in range(10)
        ]
        b = _baselines(insts)
        # 8 options -> random baseline 0.125, and all-same truth -> majority 1.0
        self.assertAlmostEqual(b["random"], 0.125)
        self.assertAlmostEqual(b["majority_class"], 1.0)
        self.assertAlmostEqual(b["chance"], 1.0)

    def test_chance_excludes_the_cheap_heuristic(self):
        from jeveval.smoke import _baselines

        opts = [{"id": "a", "label": ""}, {"id": "b", "label": ""}]
        insts = [
            Instance(generator="s", difficulty={}, seed=1, index=i, state="x",
                     questions={"q": choice("pick", opts)},
                     truth={"q": "a" if i % 2 else "b"},
                     meta={"baseline_keyword_prediction": "a" if i % 2 else "b"})
            for i in range(10)
        ]
        b = _baselines(insts)
        # A perfect keyword heuristic must be reported but must not become the
        # gate threshold, or Phase 0 could never pass.
        self.assertAlmostEqual(b["cheap_heuristic"], 1.0)
        self.assertAlmostEqual(b["chance"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
