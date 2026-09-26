from pathlib import Path
import unittest

from beads_bot.board import FeedObservation, GameState, Tile
from beads_bot.policy import choose_move
from beads_bot.vision import analyze_frame


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "img"


class PolicySampleTests(unittest.TestCase):
    def test_selects_the_raised_tile_matching_the_visible_current_color(self):
        state = analyze_frame(SAMPLES / "level56_gameplay_large_grid.jpg")

        decision = choose_move(state)

        self.assertIsNotNone(decision.tile)
        self.assertEqual(decision.tile.color, "purple")
        self.assertEqual(decision.tile.legality, "RH")
        self.assertEqual(decision.tile.center, (671, 1245))

    def test_abstains_when_the_right_feed_is_hidden(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        decision = choose_move(state)

        self.assertIsNone(decision.tile)
        self.assertIn("unknown", decision.reason)

    def test_abstains_when_no_raised_tile_matches_the_current_color(self):
        state = analyze_frame(SAMPLES / "level52_gameplay_normal.png")

        decision = choose_move(state)

        self.assertIsNone(decision.tile)
        self.assertIn("no visibly raised red", decision.reason)

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


if __name__ == "__main__":
    unittest.main()
