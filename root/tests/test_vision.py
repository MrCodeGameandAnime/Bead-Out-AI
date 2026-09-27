from pathlib import Path
import unittest

from beads_bot.vision import analyze_frame


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "img"


class VisionSampleTests(unittest.TestCase):
    def _tile_near(self, state, center):
        tile = min(state.tiles, key=lambda item: abs(item.center[0] - center[0]) + abs(item.center[1] - center[1]))
        self.assertLessEqual(abs(tile.center[0] - center[0]), 24)
        self.assertLessEqual(abs(tile.center[1] - center[1]), 24)
        return tile

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

    def test_right_conveyor_stays_unknown_until_its_outlet_is_calibrated(self):
        state = analyze_frame(SAMPLES / "level56_gameplay_large_grid.jpg")

        self.assertIsNone(state.feed.current)
        self.assertEqual(state.feed.confidence, 0.0)
        self.assertIn("unresolved", state.feed.source)

    def test_hidden_right_conveyor_is_unknown_not_guessed(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        self.assertIsNone(state.feed.current)
        self.assertLess(state.feed.confidence, 0.65)
        self.assertNotIn("top edge", state.feed.source)

    def test_ice_special_tiles_remain_unknown_without_digit_ocr(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        special = [tile for tile in state.tiles if tile.kind == "special"]
        self.assertEqual(len(special), 4)
        self.assertTrue(all(tile.number is None for tile in special))

    def test_very_hard_board_finds_all_raised_tiles_locks_and_numbered_tiles(self):
        state = analyze_frame(SAMPLES / "level59_very_hard_locked_special_mechanics.png")

        self.assertEqual(state.difficulty, "Very Hard")
        self.assertEqual(len(state.tiles), 20)
        self.assertEqual(sum(tile.legality == "RH" for tile in state.tiles), 6)
        self.assertEqual(len(state.locks), 2)
        special = [tile for tile in state.tiles if tile.kind == "special"]
        self.assertEqual(len(special), 2)
        self.assertTrue(all(tile.number is None for tile in special))

    def test_live_level60_recovers_every_cell_even_when_locks_join_masks(self):
        state = analyze_frame(ROOT / "debug" / "before_001.png")

        expected_centers = [
            (342, 1347), (474, 1347), (606, 1347), (738, 1347),
            (342, 1506), (474, 1506), (606, 1506), (738, 1506),
            (342, 1653), (474, 1653), (606, 1653), (738, 1653),
            (342, 1800), (474, 1800), (606, 1800), (738, 1800),
            (342, 1947), (474, 1947), (606, 1947),
        ]

        self.assertEqual(len(state.tiles), 19)
        for center in expected_centers:
            self._tile_near(state, center)

    def test_live_level60_top_row_blue_and_pink_remain_raised(self):
        state = analyze_frame(ROOT / "debug" / "before_001.png")

        blue = self._tile_near(state, (342, 1347))
        pink = self._tile_near(state, (474, 1347))
        cyan = self._tile_near(state, (738, 1347))
        self.assertEqual((blue.color, blue.legality), ("blue", "RH"))
        self.assertEqual((pink.color, pink.legality), ("pink", "RH"))
        self.assertEqual((cyan.color, cyan.legality), ("cyan", "RH"))

    def test_live_level60_bright_yellow_cubes_are_depressed(self):
        state = analyze_frame(ROOT / "debug" / "before_001.png")

        for center in ((342, 1653), (474, 1653), (606, 1653)):
            self.assertEqual(self._tile_near(state, center).legality, "DNH")

    def test_live_level60_special_values_are_correct_or_left_unknown(self):
        state = analyze_frame(ROOT / "debug" / "before_001.png")
        expected = {
            (342, 1800): 600,
            (474, 1800): 3,
            (606, 1800): 600,
            (738, 1800): 2,
            (474, 1947): 5,
        }

        for center, value in expected.items():
            tile = self._tile_near(state, center)
            self.assertEqual(tile.kind, "special")
            self.assertIn(tile.number, (value, None))

    def test_live_level60_reports_lock_markers_separately_from_tiles(self):
        state = analyze_frame(ROOT / "debug" / "before_001.png")

        self.assertEqual(len(state.locks), 2)
        self.assertEqual(len(state.tiles), 19)

    def test_live_level60_feed_stays_unknown_without_verified_outlet_read(self):
        state = analyze_frame(ROOT / "debug" / "before_001.png")

        self.assertIsNone(state.feed.current)
        self.assertNotIn("top edge", state.feed.source)

    def test_home_screen_is_not_reported_as_a_game_board(self):
        state = analyze_frame(SAMPLES / "level51_home_screen.png")

        self.assertNotEqual(state.screen, "game")

    def test_completion_screen_is_not_reported_as_a_game_board(self):
        state = analyze_frame(SAMPLES / "level_complete_congrats.png")

        self.assertNotEqual(state.screen, "game")


if __name__ == "__main__":
    unittest.main()
