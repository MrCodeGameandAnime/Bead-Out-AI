from pathlib import Path
import unittest

from beads_bot.board import FeedObservation, GameState, LockMarker, Tile
from beads_bot.policy import choose_move
from beads_bot.vision import analyze_frame


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "img"


class PolicySampleTests(unittest.TestCase):
    def test_abstains_when_feed_order_has_not_been_verified(self):
        state = analyze_frame(SAMPLES / "level56_gameplay_large_grid.jpg")

        decision = choose_move(state)

        self.assertIsNone(decision.tile)
        self.assertIn("unknown", decision.reason)

    def test_abstains_when_the_right_feed_is_hidden(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        decision = choose_move(state)

        self.assertIsNone(decision.tile)
        self.assertIn("unknown", decision.reason)

    def test_abstains_when_feed_order_has_not_been_verified_on_normal_board(self):
        state = analyze_frame(SAMPLES / "level52_gameplay_normal.png")

        decision = choose_move(state)

        self.assertIsNone(decision.tile)
        self.assertIn("unknown", decision.reason)

    def test_never_selects_a_locked_tile_even_if_it_is_raised(self):
        locked = Tile(
            id="locked-pink",
            bbox=(0, 0, 100, 100),
            color="pink",
            legality="RH",
            confidence=0.95,
            color_confidence=0.9,
            locked=True,
        )
        state = GameState(
            screen="game",
            width=200,
            height=300,
            tiles=(locked,),
            board_region=locked.bbox,
            feed=FeedObservation("pink", (), 0.95, "right conveyor"),
        )

        decision = choose_move(state)

        self.assertIsNone(decision.tile)

    def test_never_selects_a_tile_overlapped_by_a_separate_lock_marker(self):
        tile = Tile(
            id="under-lock",
            bbox=(70, 30, 130, 90),
            color="pink",
            legality="RH",
            confidence=0.95,
            color_confidence=0.9,
        )
        lock = LockMarker("lock-1", (90, 0, 110, 80), (100, 20), 0.9)
        state = GameState(
            screen="game",
            width=200,
            height=300,
            tiles=(tile,),
            board_region=tile.bbox,
            feed=FeedObservation("pink", (), 0.95, "right conveyor"),
            locks=(lock,),
        )

        decision = choose_move(state)

        self.assertIsNone(decision.tile)


if __name__ == "__main__":
    unittest.main()
