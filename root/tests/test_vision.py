from pathlib import Path
import unittest
from PIL import Image

from beads_bot.vision import FeedTracker, analyze_frame


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "img"
LEVEL60_FRAME = SAMPLES / "before_001.png"


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

    def test_visible_level56_outlet_bead_is_read_directly(self):
        state = analyze_frame(SAMPLES / "level56_gameplay_large_grid.jpg")

        self.assertEqual(state.feed.current, "green")
        self.assertEqual(state.feed.upcoming[0], "yellow")
        self.assertGreaterEqual(state.feed.confidence, 0.65)

    def test_hidden_right_conveyor_is_unknown_not_guessed(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        self.assertIsNone(state.feed.current)
        self.assertLess(state.feed.confidence, 0.65)
        self.assertNotIn("top edge", state.feed.source)

    def test_hidden_conveyor_lookahead_is_an_explicit_unknown_position(self):
        state = analyze_frame(SAMPLES / "level57_hidden_beads_special_tiles.png")

        self.assertIsNone(state.feed.current)
        self.assertEqual(state.feed.upcoming, (None,))
        self.assertIn("concealed", state.feed.source)

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
        state = analyze_frame(LEVEL60_FRAME)

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
        state = analyze_frame(LEVEL60_FRAME)

        blue = self._tile_near(state, (342, 1347))
        pink = self._tile_near(state, (474, 1347))
        cyan = self._tile_near(state, (738, 1347))
        self.assertEqual((blue.color, blue.legality), ("blue", "RH"))
        self.assertEqual((pink.color, pink.legality), ("pink", "RH"))
        self.assertEqual((cyan.color, cyan.legality), ("cyan", "RH"))

    def test_live_level60_bright_yellow_cubes_are_depressed(self):
        state = analyze_frame(LEVEL60_FRAME)

        for center in ((342, 1653), (474, 1653), (606, 1653)):
            self.assertEqual(self._tile_near(state, center).legality, "DNH")

    def test_live_level60_special_values_are_correct_or_left_unknown(self):
        state = analyze_frame(LEVEL60_FRAME)
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
        state = analyze_frame(LEVEL60_FRAME)

        self.assertEqual(len(state.locks), 2)
        self.assertEqual(len(state.tiles), 19)

    def test_live_level60_feed_reads_visible_outlet_bead(self):
        state = analyze_frame(LEVEL60_FRAME)

        self.assertEqual(state.feed.current, "pink")
        self.assertEqual(state.feed.upcoming[:3], ("purple", "yellow", "orange"))
        self.assertGreaterEqual(state.feed.confidence, 0.65)
        self.assertNotIn("top edge", state.feed.source)

    def test_level79_live_frame_matches_manual_rh_dnh_ground_truth(self):
        state = analyze_frame(SAMPLES / "level79_very_hard_live_rh_regression_feed_unknown.png")

        expected_raised = {
            (174, 1457), (321, 1457), (467, 1457),
            (614, 1457), (760, 1457), (901, 1457),
        }
        expected_depressed = {
            (174, 1634), (321, 1634), (614, 1634), (760, 1634), (901, 1634),
            (174, 1797), (321, 1797), (467, 1797), (614, 1797), (760, 1797), (901, 1797),
            (168, 1954), (321, 1961), (467, 1961), (614, 1961), (760, 1961),
        }

        self.assertEqual(len(state.tiles), 22)
        actual_raised = {tile.center for tile in state.tiles if tile.legality == "RH"}
        actual_depressed = {tile.center for tile in state.tiles if tile.legality == "DNH"}
        self.assertEqual(actual_raised, expected_raised)
        self.assertEqual(actual_depressed, expected_depressed)

    def test_annotated_home_reference_exposes_only_its_marked_play_target(self):
        state = analyze_frame(SAMPLES / "home_screen.png")

        self.assertEqual(state.screen, "home")
        self.assertEqual(
            [(control.kind, control.center) for control in state.controls],
            [("start_level", (219, 738))],
        )

    def test_annotated_out_of_space_offer_exposes_only_its_marked_close_target(self):
        state = analyze_frame(SAMPLES / "failure_01.jpg")

        self.assertEqual(state.screen, "out_of_space")
        self.assertEqual(state.modal_substate, "space_offer")
        self.assertEqual(
            [(control.kind, control.center) for control in state.controls],
            [("close", (639, 268))],
        )

    def test_annotated_life_loss_warning_is_not_level_failure(self):
        state = analyze_frame(SAMPLES / "failure_02.jpg")

        self.assertEqual(state.screen, "out_of_space")
        self.assertEqual(state.modal_substate, "life_warning")
        self.assertEqual(
            [(control.kind, control.center) for control in state.controls],
            [("close", (639, 268))],
        )

    def test_raw_level80_out_of_space_frame_exposes_only_its_safe_close_target(self):
        state = analyze_frame(SAMPLES / "level80_live_out_of_space.jpg")

        self.assertEqual(state.screen, "out_of_space")
        self.assertEqual(state.modal_substate, "space_offer")
        self.assertEqual(
            [(control.kind, control.center) for control in state.controls],
            [("close", (540, 2273))],
        )

    def test_raw_level80_offer_and_life_warning_pages_keep_distinct_substates(self):
        offer = analyze_frame(SAMPLES / "level80_live_space_offer_page.jpg")
        warning = analyze_frame(SAMPLES / "level80_live_life_warning_page.jpg")

        self.assertEqual((offer.screen, offer.modal_substate), ("out_of_space", "space_offer"))
        self.assertEqual((warning.screen, warning.modal_substate), ("out_of_space", "life_warning"))
        self.assertEqual(offer.controls[0].kind, "close")
        self.assertEqual(warning.controls[0].kind, "close")
        self.assertEqual(offer.controls[0].center, warning.controls[0].center)

    def test_level80_feed_reads_outlet_and_keeps_hidden_lookahead_unknown(self):
        state = analyze_frame(SAMPLES / "level80_feed_before_action_01.jpg")

        self.assertEqual(state.feed.current, "white")
        self.assertEqual(state.feed.upcoming[:3], ("indigo", "pink", "orange"))
        self.assertIn(None, state.feed.upcoming)
        self.assertGreaterEqual(state.feed.confidence, 0.65)

    def test_level80_temporal_feed_tracker_confirms_advancement_toward_outlet(self):
        tracker = FeedTracker()
        first_image = Image.open(SAMPLES / "level80_feed_before_action_01.jpg")
        second_image = Image.open(SAMPLES / "level80_feed_before_action_02.jpg")
        first_state = tracker.observe(first_image, analyze_frame(first_image))
        second_state = tracker.observe(second_image, analyze_frame(second_image))

        self.assertEqual(first_state.feed.current, "white")
        self.assertEqual(second_state.feed.current, "indigo")
        self.assertEqual(second_state.feed.direction, "from_left_toward_outlet")
        self.assertGreater(second_state.feed.direction_confidence, 0.65)

    def test_annotated_level_failed_screen_exposes_only_its_marked_close_target(self):
        state = analyze_frame(SAMPLES / "failure_03.jpg")

        self.assertEqual(state.screen, "failure")
        self.assertEqual(
            [(control.kind, control.center) for control in state.controls],
            [("dismiss_failure", (599, 466))],
        )

    def test_completion_screen_is_not_reported_as_a_game_board(self):
        state = analyze_frame(SAMPLES / "level_complete_congrats.png")

        self.assertNotEqual(state.screen, "game")


if __name__ == "__main__":
    unittest.main()
