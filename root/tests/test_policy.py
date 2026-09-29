from pathlib import Path
import unittest

from beads_bot import policy as policy_module
from beads_bot.board import FeedObservation, GameState, LockMarker, Tile
from beads_bot.main import run_offline
from beads_bot.policy import choose_move
from beads_bot.vision import analyze_frame


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "img"


def _tile(tile_id, y, *, color="green", legality="RH", kind="tile", locked=False,
          mechanic_overlays=()):
    return Tile(
        id=tile_id,
        bbox=(20, y, 80, y + 60),
        color=color,
        legality=legality,
        confidence=0.9 if legality == "RH" else 0.45,
        color_confidence=0.9 if color else 0.4,
        kind=kind,
        locked=locked,
        mechanic_overlays=mechanic_overlays,
    )


def _state(tiles, *, feed=None, locks=()):
    return GameState(
        screen="game",
        width=300,
        height=700,
        tiles=tuple(tiles),
        board_region=(20, min(tile.bbox[1] for tile in tiles), 280, max(tile.bbox[3] for tile in tiles)),
        feed=feed or FeedObservation(None, (), 0.0, "unresolved"),
        locks=tuple(locks),
    )


class PolicySampleTests(unittest.TestCase):
    def test_key_overlay_is_explicit_in_context_and_adds_uncertainty(self):
        ordinary = _tile("ordinary", 100, color="orange")
        key = _tile("key", 100, color="orange", mechanic_overlays=("key",))
        state = _state([ordinary, key])

        candidates = policy_module.rank_candidates(state)
        by_id = {candidate.tile.id: candidate for candidate in candidates}
        key_candidate = by_id["key"]
        ordinary_candidate = by_id["ordinary"]
        key_mechanics = next(item for item in key_candidate.requirements if item.name == "other_mechanics")
        ordinary_mechanics = next(item for item in ordinary_candidate.requirements if item.name == "other_mechanics")

        self.assertEqual(key_candidate.context.features["mechanic_overlay"], "key")
        self.assertEqual(key_candidate.context.features["mechanic_overlays"], ("key",))
        self.assertEqual(key_candidate.context.features["mechanic_overlay_status"], "KNOWN")
        self.assertEqual(key_candidate.context.features["mechanic_rule_status"], "UNKNOWN")
        self.assertEqual(key_mechanics.status, "UNCERTAIN")
        self.assertIn("key overlay", key_mechanics.source)
        self.assertEqual(ordinary_candidate.context.features["mechanic_overlay"], None)
        self.assertEqual(ordinary_mechanics.status, "KNOWN")
        self.assertEqual(candidates[0].tile.id, "ordinary")
        self.assertIn(key_candidate, candidates)

    def test_only_plausible_key_candidate_remains_executable(self):
        key = _tile("only-key", 100, mechanic_overlays=("key",))

        decision = choose_move(_state([key]))

        self.assertEqual(decision.tile, key)
        self.assertTrue(decision.candidates)

    def test_key_overlay_change_reenables_suppressed_physical_tile_after_jitter(self):
        key = _tile("key", 100, mechanic_overlays=("key",))
        attempted = policy_module.rank_candidates(_state([key]))[0]
        jittered_same_key = Tile(
            id="fresh-id", bbox=(23, 97, 83, 157), color=key.color,
            legality=key.legality, confidence=key.confidence,
            color_confidence=key.color_confidence, kind=key.kind,
            mechanic_overlays=("key",),
        )
        removed_key = Tile(
            id="another-id", bbox=(23, 97, 83, 157), color=key.color,
            legality=key.legality, confidence=key.confidence,
            color_confidence=key.color_confidence, kind=key.kind,
        )

        self.assertEqual(policy_module.rank_candidates(_state([jittered_same_key]), attempted=(attempted,)), ())
        reenabled = policy_module.rank_candidates(_state([removed_key]), attempted=(attempted,))
        self.assertEqual(len(reenabled), 1)
        self.assertEqual(reenabled[0].tile.mechanic_overlays, ())

    def test_attempted_candidate_identity_survives_tiny_bbox_jitter_and_new_tile_ids(self):
        original_tile = _tile("tile-1", 120)
        original_state = _state([original_tile])
        original = policy_module.rank_candidates(original_state)[0]
        jittered_tile = Tile(
            id="regenerated-id",
            bbox=(23, 117, 83, 177),
            color=original_tile.color,
            legality=original_tile.legality,
            confidence=original_tile.confidence,
            color_confidence=original_tile.color_confidence,
            kind=original_tile.kind,
            locked=original_tile.locked,
        )
        jittered_state = _state([jittered_tile])
        retried = policy_module.rank_candidates(
            jittered_state,
            attempted=(original,),
        )

        self.assertEqual(original.key, policy_module.rank_candidates(jittered_state)[0].key)
        self.assertEqual(retried, ())

    def test_attempted_tile_is_reenabled_after_its_local_semantics_change(self):
        tile = _tile("tile-1", 120)
        state = _state([tile])
        attempted = policy_module.rank_candidates(state)[0]
        changed = _state([Tile(
            id="new-id",
            bbox=tile.bbox,
            color=tile.color,
            legality="RH",
            confidence=tile.confidence,
            color_confidence=tile.color_confidence,
            kind="special",
            locked=tile.locked,
        )])

        candidates = policy_module.rank_candidates(changed, attempted=(attempted,))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tile.kind, "special")

    def test_attempted_tile_stays_suppressed_after_color_classifier_change(self):
        tile = _tile("tile-1", 120, color="blue")
        state = _state([tile])
        attempted = policy_module.rank_candidates(state)[0]
        color_jitter = _state([Tile(
            id="new-id",
            bbox=tile.bbox,
            color="cyan",
            legality=tile.legality,
            confidence=tile.confidence,
            color_confidence=tile.color_confidence,
            kind=tile.kind,
            locked=tile.locked,
        )])

        candidates = policy_module.rank_candidates(color_jitter, attempted=(attempted,))

        self.assertEqual(candidates, ())

    def test_attempted_tile_is_reenabled_when_its_own_lock_overlay_changes(self):
        tile = _tile("tile-1", 120)
        state = _state([tile])
        attempted = policy_module.rank_candidates(state)[0]
        lock = LockMarker("new-lock", tile.bbox, tile.center, 0.9)
        changed = _state([tile], locks=(lock,))

        candidates = policy_module.rank_candidates(changed, attempted=(attempted,))

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].context.features["lock_overlap"])

    def test_unknown_feed_still_returns_plausible_candidates(self):
        state = analyze_frame(SAMPLES / "level56_gameplay_large_grid.jpg")

        decision = choose_move(state)

        self.assertIsNotNone(decision.tile)
        self.assertEqual(state.feed.current, "green")
        candidates = getattr(decision, "candidates", ())
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].tile, decision.tile)

    def test_unknown_feed_does_not_invent_a_color(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        decision = choose_move(state)

        self.assertIsNotNone(decision.tile)
        self.assertIsNone(state.feed.current)
        self.assertEqual(state.feed.upcoming, (None,) * len(state.feed.upcoming))

    def test_understood_unlocked_tile_ranks_above_uncertain_candidates(self):
        ordinary = _tile("ordinary", 10)
        special = _tile("special", 110, kind="special")
        lock_affected = _tile("lock-affected", 210)
        uncertain = _tile("uncertain", 310, legality="UNKNOWN")
        lock = LockMarker("lock-1", (35, 190, 65, 250), (50, 205), 0.9)
        state = _state(
            [ordinary, special, lock_affected, uncertain],
            feed=FeedObservation("green", (), 0.95, "visible center"),
            locks=(lock,),
        )

        decision = choose_move(state)
        candidates = getattr(decision, "candidates", ())
        ranked_ids = [candidate.tile.id for candidate in candidates]

        self.assertTrue(candidates)
        self.assertEqual(ranked_ids[0], "ordinary")
        self.assertLess(ranked_ids.index("ordinary"), ranked_ids.index("special"))
        self.assertLess(ranked_ids.index("ordinary"), ranked_ids.index("lock-affected"))
        self.assertLess(ranked_ids.index("ordinary"), ranked_ids.index("uncertain"))

    def test_current_and_imminent_feed_matches_raise_candidate_rank(self):
        current = _tile("current-match", 10, color="blue")
        imminent = _tile("imminent-match", 110, color="red")
        unrelated = _tile("unrelated", 210, color="green")
        state = _state(
            [current, imminent, unrelated],
            feed=FeedObservation("blue", ("red", None), 0.9, "visible outlet"),
        )

        candidates = policy_module.rank_candidates(state)
        ranked_ids = [candidate.tile.id for candidate in candidates]

        self.assertLess(ranked_ids.index("current-match"), ranked_ids.index("unrelated"))
        self.assertLess(ranked_ids.index("imminent-match"), ranked_ids.index("unrelated"))

    def test_unknown_feed_still_keeps_optimistic_candidates(self):
        state = _state(
            [_tile("first", 10, color="blue"), _tile("second", 110, color="red")],
            feed=FeedObservation(None, (None, None), 0.0, "hidden feed"),
        )

        candidates = policy_module.rank_candidates(state)

        self.assertEqual({candidate.tile.id for candidate in candidates}, {"first", "second"})

    def test_uncertain_special_or_locked_tile_remains_as_fallback(self):
        special = _tile("special", 10, kind="special")
        locked = _tile("locked", 110, locked=True)
        lock_affected = _tile("overlapped", 210)
        uncertain = _tile("unknown-relief", 310, legality="UNKNOWN")
        lock = LockMarker("lock-1", (35, 190, 65, 250), (50, 205), 0.9)
        state = _state([special, locked, lock_affected, uncertain], locks=(lock,))

        decision = choose_move(state)
        candidates = getattr(decision, "candidates", ())

        self.assertEqual({candidate.tile.id for candidate in candidates}, {
            "special", "locked", "overlapped", "unknown-relief"
        })

    def test_known_dnh_tiles_are_not_candidates(self):
        state = _state([_tile("depressed", 10, legality="DNH")])

        decision = choose_move(state)

        self.assertIsNone(decision.tile)
        self.assertEqual(getattr(decision, "candidates", ()), ())

    def test_contextual_evidence_adjusts_candidate_rank_without_removing_it(self):
        rank_candidates = getattr(policy_module, "rank_candidates", None)
        evidence_tally = getattr(policy_module, "EvidenceTally", None)
        self.assertTrue(callable(rank_candidates))
        self.assertIsNotNone(evidence_tally)

        state = _state([_tile("yellow", 10, color="yellow"), _tile("blue", 110, color="blue")])
        baseline = rank_candidates(state)

        def accepted_evidence(context):
            if context.features.get("tile_color") == "yellow":
                return evidence_tally(action_accepted=4)
            return evidence_tally()

        def success_evidence(context):
            if context.features.get("tile_color") == "yellow":
                return evidence_tally(level_success=1)
            return evidence_tally()

        accepted = rank_candidates(state, evidence_for=accepted_evidence)
        strategic = rank_candidates(state, evidence_for=success_evidence)
        baseline_yellow = next(item for item in baseline if item.tile.id == "yellow")
        accepted_yellow = next(item for item in accepted if item.tile.id == "yellow")
        strategic_yellow = next(item for item in strategic if item.tile.id == "yellow")

        self.assertGreater(accepted_yellow.score, baseline_yellow.score)
        self.assertGreater(strategic_yellow.score, accepted_yellow.score)
        self.assertIn(accepted_yellow, accepted)
        self.assertIn(strategic_yellow, strategic)

    def test_report_serializes_ranked_candidate_evidence_and_outcomes(self):
        report = run_offline(SAMPLES / "level56_gameplay_large_grid.jpg", None)
        candidates = report.get("candidates", [])

        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["tile_id"], report["decision"]["tile_id"])
        self.assertIn("score", candidates[0])
        self.assertIn("context", candidates[0])
        requirement_names = {item["name"] for item in candidates[0]["requirements"]}
        self.assertEqual(requirement_names, {
            "tile_legality", "tile_color", "visible_current_state", "feed_state",
            "special_rule", "lock_rule", "other_mechanics",
        })
        self.assertEqual(report["outcomes"], {"interaction": "NOT_ATTEMPTED", "strategic": "UNKNOWN"})
        self.assertEqual(report["feed"]["current"], "green")
        self.assertEqual(report["execution"], "offline: no input sent")

    def test_offline_report_serializes_key_overlay_in_tile_and_candidate_context(self):
        report = run_offline(SAMPLES / "level80_feed_before_action_01.jpg", None)

        keyed_tiles = [tile for tile in report["tiles"] if "key" in tile["mechanic_overlays"]]
        keyed_candidates = [
            candidate for candidate in report["candidates"]
            if candidate["context"]["features"].get("mechanic_overlay") == "key"
        ]
        self.assertEqual(len(keyed_tiles), 3)
        self.assertTrue(keyed_candidates)
        self.assertTrue(all(item["context"]["features"]["mechanic_rule_status"] == "UNKNOWN"
                            for item in keyed_candidates))


if __name__ == "__main__":
    unittest.main()
