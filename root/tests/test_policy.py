from pathlib import Path
import unittest

from beads_bot import policy as policy_module
from beads_bot.board import FeedObservation, GameState, LockMarker, Tile
from beads_bot.main import run_offline
from beads_bot.policy import choose_move
from beads_bot.vision import analyze_frame


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "img"


def _tile(tile_id, y, *, color="green", legality="RH", kind="tile", locked=False):
    return Tile(
        id=tile_id,
        bbox=(20, y, 80, y + 60),
        color=color,
        legality=legality,
        confidence=0.9 if legality == "RH" else 0.45,
        color_confidence=0.9 if color else 0.4,
        kind=kind,
        locked=locked,
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
    def test_unknown_feed_still_returns_plausible_candidates(self):
        state = analyze_frame(SAMPLES / "level56_gameplay_large_grid.jpg")

        decision = choose_move(state)

        self.assertIsNotNone(decision.tile)
        self.assertIsNone(state.feed.current)
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
        self.assertIsNone(report["feed"]["current"])
        self.assertEqual(report["execution"], "offline: no input sent")


if __name__ == "__main__":
    unittest.main()
