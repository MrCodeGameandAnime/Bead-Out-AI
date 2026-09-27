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


def _state(tiles=(), *, screen="game", controls=(), modal_substate=None):
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
        modal_substate=modal_substate,
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

    def test_out_of_space_page_transition_is_progress_but_repeated_page_is_no_change(self):
        close = (ScreenControl("close", (90, 90), 0.99),)
        offer = _state(screen="out_of_space", controls=close, modal_substate="space_offer")
        warning = _state(screen="out_of_space", controls=close, modal_substate="life_warning")

        self.assertNotEqual(main_module._state_signature(offer), main_module._state_signature(warning))
        self.assertEqual(
            main_module._control_transition_outcomes(offer, warning)[0].value,
            "ACTION_ACCEPTED",
        )
        self.assertEqual(
            main_module._control_transition_outcomes(warning, warning)[0].value,
            "NO_CHANGE",
        )

        result, taps, _ = self._run([offer, warning, warning])
        events = [
            json.loads(line)
            for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
        ]

        self.assertEqual(taps, [(90, 90), (90, 90)])
        self.assertEqual(result["status"], "no_progress")
        self.assertEqual(
            [event["outcomes"]["interaction"] for event in events],
            ["ACTION_ACCEPTED", "NO_CHANGE"],
        )

    def test_out_of_space_recovery_records_failure_then_starts_a_separate_attempt(self):
        game = _state([_tile("before-failure", 20)])
        offer = _state(
            screen="out_of_space",
            controls=(
                ScreenControl("free_space", (10, 10), 1.0, cost=900),
                ScreenControl("close", (90, 90), 0.99),
            ),
        )
        warning = _state(
            screen="out_of_space",
            controls=(
                ScreenControl("continue_for_free", (11, 11), 1.0, requires_ad=True),
                ScreenControl("close", (91, 91), 0.99),
            ),
        )
        failure = _state(
            screen="failure",
            controls=(
                ScreenControl("try_again", (12, 12), 1.0),
                ScreenControl("keep_going", (13, 13), 1.0),
                ScreenControl("dismiss_failure", (92, 92), 0.99),
            ),
        )
        home = _state(screen="home", controls=(ScreenControl("start_level", (93, 93), 0.99),))
        restarted_game = _state([_tile("after-restart", 20)])
        after_restart_action = _state([_tile("after-restart", 120)])
        transition_frame = _state(screen="unknown")
        home_transition_frame = _state(screen="unknown")

        states = [
            game, transition_frame, offer, warning, failure, home_transition_frame, home,
            restarted_game, after_restart_action,
        ]
        images = []
        state_by_pixel = {}
        for index, state in enumerate(states, start=1):
            pixel = (index, 0, 0)
            state_by_pixel[pixel] = state
            images.append(Image.new("RGB", (1, 1), pixel))
        cursor = 0
        taps = []
        debug_dir = Path(self.temp.name) / "recovery"

        def capture_fn(_adb=None, _serial=None):
            nonlocal cursor
            index = min(cursor, len(images) - 1)
            cursor += 1
            return CapturedFrame(images[index], 1.0)

        def tap_fn(x, y, _adb=None, _serial=None):
            taps.append((x, y))
            if (x, y) == (92, 92):
                manifests = list((debug_dir / "runs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                failure_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(failure_manifest["status"], "failure")
                records = [
                    json.loads(line)
                    for line in (debug_dir / "mechanic_evidence.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertIn("LEVEL_FAILURE", [record.get("strategic_outcome") for record in records])
            return 0.5

        args = argparse.Namespace(
            adb=None,
            serial=None,
            debug_dir=debug_dir,
            execute=True,
            max_moves=1,
            loop=False,
            settle_seconds=0.0,
        )
        result = main_module.run_live(
            args,
            capture_fn=capture_fn,
            tap_fn=tap_fn,
            analyze_fn=lambda image: state_by_pixel[image.getpixel((0, 0))],
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(taps, [(50, 50), (90, 90), (91, 91), (92, 92), (93, 93), (50, 50)])
        self.assertEqual(result["status"], "move_limit")
        manifests = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (debug_dir / "runs").glob("*/manifest.json")
        ]
        self.assertEqual(sorted(manifest["status"] for manifest in manifests), ["failure", "move_limit"])
        failed = next(manifest for manifest in manifests if manifest["status"] == "failure")
        self.assertTrue(failed["possible_failure_example"])
        self.assertTrue(any(step["outcomes"]["strategic"] == "LEVEL_FAILURE" for step in failed["recent_steps"]))
        self.assertTrue(any(step["before_state"]["screen"] == "out_of_space" for step in failed["recent_steps"]))
        failed_manifest_path = next(
            path for path in (debug_dir / "runs").glob("*/manifest.json")
            if json.loads(path.read_text(encoding="utf-8"))["run_id"] == failed["run_id"]
        )
        failed_events_path = failed_manifest_path.parent / failed["events_path"]
        failed_events = [
            json.loads(line)
            for line in failed_events_path.read_text(encoding="utf-8").splitlines()
        ]
        out_of_space_events = [
            event for event in failed_events
            if event["after_state"] is not None
            and event["after_state"]["screen"] == "out_of_space"
        ]
        self.assertTrue(out_of_space_events)
        self.assertTrue(all(event["outcomes"]["strategic"] == "IN_PROGRESS" for event in out_of_space_events))

        records = [
            json.loads(line)
            for line in (debug_dir / "mechanic_evidence.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        failure_episodes = [
            record for record in records
            if record.get("record_type") == "episode" and record.get("strategic_outcome") == "LEVEL_FAILURE"
        ]
        self.assertEqual(len(failure_episodes), 1)
        self.assertEqual(len(failure_episodes[0]["sequence"]), 1)

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
