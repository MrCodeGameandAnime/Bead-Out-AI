from pathlib import Path
import unittest

from beads_bot.vision import analyze_frame


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "img"


class VisionSampleTests(unittest.TestCase):
    def test_large_grid_finds_four_raised_tiles_and_depressed_tiles(self):
        state = analyze_frame(SAMPLES / "level56_gameplay_large_grid.jpg")

        self.assertEqual(state.screen, "game")
        raised = [tile for tile in state.tiles if tile.legality == "RH"]
        depressed = [tile for tile in state.tiles if tile.legality == "DNH"]
        self.assertEqual(len(raised), 4)
        self.assertGreaterEqual(len(depressed), 8)

    def test_hidden_tile_level_keeps_only_visibly_raised_tiles_legal(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        raised = [tile for tile in state.tiles if tile.legality == "RH"]
        self.assertEqual(len(raised), 2)
        self.assertTrue(all(tile.confidence >= 0.65 for tile in raised))

    def test_hidden_tiles_are_not_misreported_as_gray_cubes(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        hidden = [tile for tile in state.tiles if tile.kind == "hidden"]
        self.assertEqual(len(hidden), 4)
        self.assertTrue(all(tile.color is None for tile in hidden))

    def test_right_conveyor_reports_its_visible_leading_color(self):
        state = analyze_frame(SAMPLES / "level56_gameplay_large_grid.jpg")

        self.assertEqual(state.feed.current, "purple")
        self.assertGreaterEqual(state.feed.confidence, 0.65)

    def test_hidden_right_conveyor_is_unknown_not_guessed(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        self.assertIsNone(state.feed.current)
        self.assertLess(state.feed.confidence, 0.65)

    def test_ice_special_tiles_keep_their_visible_200_value(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        numbered = [tile.number for tile in state.tiles if tile.number is not None]
        self.assertEqual(numbered, [200, 200, 200, 200])

    def test_very_hard_board_finds_all_raised_tiles_locks_and_numbered_tiles(self):
        state = analyze_frame(SAMPLES / "level59_very_hard_locked_special_mechanics.png")

        self.assertEqual(state.difficulty, "Very Hard")
        self.assertEqual(len(state.tiles), 20)
        self.assertEqual(sum(tile.legality == "RH" for tile in state.tiles), 6)
        self.assertEqual(sum(tile.locked for tile in state.tiles), 2)
        self.assertEqual(
            sorted(tile.number for tile in state.tiles if tile.number is not None),
            [2, 2],
        )

    def test_home_screen_is_not_reported_as_a_game_board(self):
        state = analyze_frame(SAMPLES / "level51_home_screen.png")

        self.assertNotEqual(state.screen, "game")

    def test_completion_screen_is_not_reported_as_a_game_board(self):
        state = analyze_frame(SAMPLES / "level_complete_congrats.png")

        self.assertNotEqual(state.screen, "game")


if __name__ == "__main__":
    unittest.main()
