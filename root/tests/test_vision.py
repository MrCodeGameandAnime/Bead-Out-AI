from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from beads_bot.board import FeedObservation, GameState
from beads_bot.vision import FeedTracker, analyze_frame


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "img"
LEVEL60_FRAME = SAMPLES / "before_001.png"
LEVEL80_WORKING_FEED = SAMPLES / "level80_feed_dropout_reference_working.jpg"
LEVEL80_OUTLET_DROPOUT = SAMPLES / "level80_feed_dropout_reference_occluded.jpg"
LEVEL80_OVERSIZED_OUTLET = SAMPLES / "level80_feed_oversized_false_outlet_reference.jpg"


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

    def test_level80_key_overlay_is_attached_to_the_visually_marked_tiles(self):
        state = analyze_frame(SAMPLES / "level80_feed_before_action_01.jpg")

        keyed = {tile.center for tile in state.tiles if "key" in tile.mechanic_overlays}
        self.assertEqual(keyed, {(804, 1468), (672, 1616), (408, 1753)})
        # The ordinary orange tile, red/green/blue padlocks, and hidden tiles
        # are useful nearby negatives in this same raw live frame.
        for center in ((540, 1610), (276, 1474), (276, 1883), (672, 1916), (805, 1623)):
            tile = self._tile_near(state, center)
            self.assertNotIn("key", tile.mechanic_overlays, center)

    def test_level80_key_detector_survives_small_tile_bbox_jitter(self):
        from beads_bot.vision import _detect_tile_mechanic_overlays

        image = Image.open(SAMPLES / "level80_feed_before_action_01.jpg").convert("RGB")
        hsv = np.asarray(image.convert("HSV"))
        expected = ((745, 1407, 863, 1529), (613, 1555, 731, 1677), (349, 1683, 467, 1824))
        for box in expected:
            for dx, dy in ((1, -1), (-2, 3), (3, 2)):
                shifted = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
                with self.subTest(box=box, shift=(dx, dy)):
                    self.assertIn("key", _detect_tile_mechanic_overlays(hsv, shifted))

    def test_level80_second_live_frame_keeps_key_association_after_board_evolution(self):
        state = analyze_frame(SAMPLES / "level80_feed_before_action_02.jpg")

        keyed = [tile for tile in state.tiles if "key" in tile.mechanic_overlays]
        self.assertEqual(len(keyed), 3)
        self.assertEqual({tile.color for tile in keyed}, {"blue", "orange", "cyan"})
        for expected_center in ((804, 1468), (672, 1616), (408, 1753)):
            tile = min(keyed, key=lambda item: abs(item.center[0] - expected_center[0])
                       + abs(item.center[1] - expected_center[1]))
            self.assertLessEqual(abs(tile.center[0] - expected_center[0]), 8)
            self.assertLessEqual(abs(tile.center[1] - expected_center[1]), 16)

    def test_level80_evolving_frame_reads_visible_bead_at_tracked_outlet_geometry(self):
        tracker = FeedTracker()
        working_image = Image.open(LEVEL80_WORKING_FEED)
        dropout_image = Image.open(LEVEL80_OUTLET_DROPOUT)
        working_state = tracker.observe(working_image, analyze_frame(working_image))
        raw_dropout_state = analyze_frame(dropout_image)
        dropout_state = tracker.observe(dropout_image, raw_dropout_state)

        self.assertEqual(working_state.feed.current, "indigo")
        self.assertEqual(working_state.feed.upcoming[:3], ("red", "orange", "blue"))
        self.assertIsNone(raw_dropout_state.feed.current)
        self.assertEqual(dropout_state.feed.current, "orange")
        self.assertEqual(dropout_state.feed.current_status, "direct")
        self.assertEqual(dropout_state.feed.outlet_status, "tracked")
        self.assertEqual(dropout_state.feed.direction, "from_left_toward_outlet")
        self.assertEqual(dropout_state.feed.upcoming[:2], ("blue", "indigo"))

    def test_level80_oversized_track_component_is_not_mistaken_for_the_outlet(self):
        state = analyze_frame(LEVEL80_OVERSIZED_OUTLET)

        self.assertIsNone(state.feed.current)
        self.assertEqual(state.feed.outlet_status, "unknown")
        self.assertIn("outlet geometry not found", state.feed.source)

    def test_single_feed_dropout_retains_recent_belief_with_one_step_decay(self):
        image = Image.new("RGB", (400, 800), "beige")
        state = GameState("game", 400, 800, (), None, FeedObservation(None, (), 0.0, "unknown"))
        direct = FeedObservation(
            "blue", ("white", "pink", None), 0.9, "visible outlet",
            current_status="direct", upcoming_status="direct", outlet_status="direct",
        )
        unresolved = FeedObservation(None, (), 0.0, "unresolved: outlet geometry not found")
        tracker = FeedTracker()
        with patch("beads_bot.vision._read_feed", side_effect=(direct, unresolved)):
            tracker.observe(image, state)
            observed = tracker.observe(image, state).feed

        self.assertEqual(observed.current, "blue")
        self.assertEqual(observed.upcoming, ("white", "pink", None))
        self.assertEqual((observed.current_status, observed.upcoming_status, observed.age_frames),
                         ("tracked", "tracked", 1))
        self.assertAlmostEqual(observed.confidence, 0.45)

    def test_repeated_feed_dropout_decays_then_expires(self):
        image = Image.new("RGB", (400, 800), "beige")
        state = GameState("game", 400, 800, (), None, FeedObservation(None, (), 0.0, "unknown"))
        direct = FeedObservation("blue", ("white", None), 0.8, "visible outlet", current_status="direct")
        unresolved = FeedObservation(None, (), 0.0, "unresolved: outlet geometry not found")
        tracker = FeedTracker()
        with patch("beads_bot.vision._read_feed", side_effect=(direct, unresolved, unresolved, unresolved)):
            tracker.observe(image, state)
            first = tracker.observe(image, state).feed
            second = tracker.observe(image, state).feed
            expired = tracker.observe(image, state).feed

        self.assertEqual((first.current, first.age_frames, first.confidence), ("blue", 1, 0.4))
        self.assertEqual((second.current, second.age_frames, second.confidence), ("blue", 2, 0.2))
        self.assertIsNone(expired.current)
        self.assertEqual(expired.current_status, "unknown")
        self.assertEqual(expired.age_frames, 3)

    def test_contradictory_direct_feed_replaces_tracked_belief_immediately(self):
        image = Image.new("RGB", (400, 800), "beige")
        state = GameState("game", 400, 800, (), None, FeedObservation(None, (), 0.0, "unknown"))
        blue = FeedObservation("blue", ("white", "pink", None), 0.9, "direct blue", current_status="direct")
        unresolved = FeedObservation(None, (), 0.0, "unresolved")
        pink = FeedObservation("pink", ("orange", None), 0.8, "direct pink", current_status="direct")
        tracker = FeedTracker()
        with patch("beads_bot.vision._read_feed", side_effect=(blue, unresolved, pink)):
            tracker.observe(image, state)
            tracked = tracker.observe(image, state).feed
            direct = tracker.observe(image, state).feed

        self.assertEqual(tracked.current_status, "tracked")
        self.assertEqual((direct.current, direct.current_status, direct.age_frames), ("pink", "direct", 0))
        self.assertEqual(direct.upcoming, ("orange", None))

    def test_concealed_feed_invalidates_a_previously_tracked_color(self):
        tracker = FeedTracker()
        visible_image = Image.open(LEVEL80_WORKING_FEED)
        tracker.observe(visible_image, analyze_frame(visible_image))
        hidden_image = Image.open(SAMPLES / "level57_hidden_beads_special_tiles.png")
        hidden_state = tracker.observe(hidden_image, analyze_frame(hidden_image))

        self.assertIsNone(hidden_state.feed.current)
        self.assertEqual(hidden_state.feed.upcoming, (None,))
        self.assertEqual(hidden_state.feed.upcoming_status, "concealed")
        self.assertEqual(hidden_state.feed.current_status, "unknown")

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
