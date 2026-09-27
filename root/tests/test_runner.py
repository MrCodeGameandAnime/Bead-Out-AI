import argparse
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from beads_bot.board import FeedObservation, GameState, ScreenControl, Tile
from beads_bot.capture import CapturedFrame
from beads_bot import main as main_module


def _tile(tile_id, y, *, color="green", legality="RH", confidence=0.9):
    return Tile(tile_id, (20, y, 80, y + 60), color, legality, confidence, 0.9)


def _state(tiles=(), *, screen="game", controls=()):
    tiles = tuple(tiles)
    board = None if not tiles else (
        min(tile.bbox[0] for tile in tiles),
        min(tile.bbox[1] for tile in tiles),
        max(tile.bbox[2] for tile in tiles),
        max(tile.bbox[3] for tile in tiles),
    )
    return GameState(
        screen=screen,
        width=240,
        height=500,
        tiles=tiles,
        board_region=board,
        feed=FeedObservation(None, (), 0.0, "unresolved"),
        controls=tuple(controls),
    )


class ContinuousRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, states, *, max_moves=1, execute=True):
        states = list(states)
        images = []
        state_by_pixel = {}
        for index, state in enumerate(states, start=1):
            pixel = (index, 0, 0)
            state_by_pixel[pixel] = state
            images.append(Image.new("RGB", (1, 1), pixel))
        cursor = 0
        taps = []

        def capture_fn(_adb=None, _serial=None):
            nonlocal cursor
            index = min(cursor, len(images) - 1)
            cursor += 1
            return CapturedFrame(images[index], 1.0)

        def analyze_fn(image):
            return state_by_pixel[image.getpixel((0, 0))]

        def tap_fn(x, y, _adb=None, _serial=None):
            taps.append((x, y))
            return 0.5

        args = argparse.Namespace(
            adb=None,
            serial=None,
            debug_dir=Path(self.temp.name) / "debug",
            execute=execute,
            max_moves=max_moves,
            loop=False,
            settle_seconds=0.0,
        )
        result = main_module.run_live(
            args,
            capture_fn=capture_fn,
            tap_fn=tap_fn,
            analyze_fn=analyze_fn,
            sleep_fn=lambda _seconds: None,
        )
        return result, taps, cursor

    def test_live_summary_links_run_journal_and_terminal_status(self):
        state = _state([_tile("first", 20)])

        result, taps, captures = self._run([state], execute=False)
        event = json.loads(Path(result["events_path"]).read_text(encoding="utf-8").splitlines()[0])
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

        self.assertEqual(taps, [])
        self.assertEqual(captures, 1)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(manifest["status"], result["status"])
        self.assertEqual(event["chosen"]["tile"]["id"], "first")
        self.assertTrue(Path(result["events_path"]).is_file())
        self.assertTrue(Path(result["manifest_path"]).is_file())

    def test_unknown_feed_executes_best_candidate_and_observes_again(self):
        state = _state([_tile("first", 20), _tile("second", 120)])

        result, taps, captures = self._run([state, state], max_moves=1)

        self.assertEqual(len(taps), 1)
        self.assertEqual(taps[0], (50, 50))
        self.assertIsNone(state.feed.current)
        self.assertEqual(captures, 2)
        self.assertEqual(result["status"], "move_limit")

    def test_low_confidence_candidate_does_not_stop_execution(self):
        uncertain = _state([_tile("uncertain", 20, legality="UNKNOWN", confidence=0.2)])

        result, taps, _ = self._run([uncertain, uncertain], max_moves=1)

        self.assertEqual(taps, [(50, 50)])
        self.assertEqual(result["action_count"], 1)

    def test_accepted_action_continues_to_next_decision(self):
        initial = _state([_tile("first", 20), _tile("second", 120)])
        after_first = _state([_tile("second", 120)])
        after_second = _state([])

        result, taps, captures = self._run([initial, after_first, after_second], max_moves=2)

        self.assertEqual(len(taps), 2)
        self.assertEqual(captures, 3)
        self.assertEqual(result["action_count"], 2)

    def test_unchanged_action_tries_next_candidate_without_repeating(self):
        state = _state([_tile("first", 20), _tile("second", 120)])

        result, taps, captures = self._run([state, state, state], max_moves=2)
        events = [
            json.loads(line) for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
        ]
        actions = [event for event in events if event["chosen"] is not None]

        self.assertEqual(len(taps), 2)
        self.assertEqual(captures, 3)
        self.assertEqual([event["chosen"]["tile"]["id"] for event in actions], ["first", "second"])
        self.assertEqual([event["outcomes"]["interaction"] for event in actions], ["NO_CHANGE", "NO_CHANGE"])

    def test_progress_resets_attempts_for_a_changed_state(self):
        before = _state([_tile("same", 20, color="green")])
        changed = _state([_tile("same", 20, color="blue")])
        after_second = _state([])

        result, taps, _ = self._run([before, changed, after_second], max_moves=2)
        events = [json.loads(line) for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(taps), 2)
        self.assertEqual(events[0]["chosen"]["key"], events[1]["chosen"]["key"])
        self.assertEqual(events[0]["chosen"]["tile"]["color"], "green")
        self.assertEqual(events[1]["chosen"]["tile"]["color"], "blue")

    def test_exhausted_unchanged_candidates_stop_without_retry(self):
        state = _state([_tile("first", 20), _tile("second", 120)])

        result, taps, captures = self._run([state, state, state], max_moves=None)

        self.assertEqual(len(taps), 2)
        self.assertEqual(captures, 3)
        self.assertEqual(result["status"], "no_progress")

    def test_confirmed_failure_stops_input_and_finalizes_failure_bundle(self):
        game = _state([_tile("first", 20), _tile("second", 120)])
        failure = _state(screen="failure")

        result, taps, captures = self._run([game, failure, game], max_moves=None)
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

        self.assertEqual(taps, [(50, 50)])
        self.assertEqual(captures, 2)
        self.assertEqual(result["status"], "failure")
        self.assertTrue(manifest["possible_failure_example"])

    def test_capture_failure_after_tap_counts_the_accepted_input_attempt(self):
        game = _state([_tile("first", 20)])
        images = [Image.new("RGB", (1, 1), (1, 0, 0))]
        capture_count = 0
        tap_calls = []

        def capture_fn(_adb=None, _serial=None):
            nonlocal capture_count
            capture_count += 1
            if capture_count > 1:
                raise RuntimeError("capture failed")
            return CapturedFrame(images[0], 1.0)

        def tap_fn(x, y, _adb=None, _serial=None):
            tap_calls.append((x, y))
            return 0.5

        args = argparse.Namespace(
            adb=None,
            serial=None,
            debug_dir=Path(self.temp.name) / "capture-failure",
            execute=True,
            max_moves=None,
            loop=False,
            settle_seconds=0.0,
        )
        result = main_module.run_live(
            args,
            capture_fn=capture_fn,
            tap_fn=tap_fn,
            analyze_fn=lambda _image: game,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(tap_calls, [(50, 50)])
        self.assertEqual(result["status"], "device_error")
        self.assertEqual(result["action_count"], 1)
        self.assertEqual(result["input_count"], 1)

    def test_unrecognized_ui_stops_and_records_possible_failure(self):
        game = _state([_tile("first", 20)])
        unknown = _state(screen="unknown")

        result, taps, _ = self._run([game, unknown], max_moves=None)
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

        self.assertEqual(len(taps), 1)
        self.assertEqual(result["status"], "unrecognized_ui")
        self.assertTrue(manifest["possible_failure_example"])
        self.assertTrue(any(step["frames"]["after"] for step in manifest["recent_steps"]))

    def test_completion_without_safe_control_stops_without_tapping(self):
        complete = _state(screen="complete")

        result, taps, captures = self._run([complete], max_moves=None)

        self.assertEqual(taps, [])
        self.assertEqual(captures, 1)
        self.assertEqual(result["status"], "success")

    def test_paid_and_ad_controls_are_not_tapped_automatically(self):
        complete = _state(screen="complete", controls=(
            ScreenControl("continue", (80, 80), 0.99, cost=25),
            ScreenControl("continue", (140, 80), 0.99, requires_ad=True),
        ))

        result, taps, _ = self._run([complete], max_moves=None)

        self.assertEqual(taps, [])
        self.assertEqual(result["status"], "success")

    def test_free_continue_control_advances_and_resumes_game(self):
        complete = _state(screen="complete", controls=(ScreenControl("continue", (160, 300), 0.95),))
        game = _state([_tile("next-level", 20)])
        after_action = _state([])

        result, taps, captures = self._run([complete, game, after_action], max_moves=1)

        self.assertEqual(taps[0], (160, 300))
        self.assertEqual(taps[1], (50, 50))
        self.assertEqual(captures, 3)
        self.assertEqual(result["action_count"], 1)


if __name__ == "__main__":
    unittest.main()
