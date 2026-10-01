import argparse
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from beads_bot.board import FeedObservation, GameState, ProtectedState, ScreenControl, Tile
from beads_bot.capture import CapturedFrame
from beads_bot import main as main_module


def _tile(tile_id, y, *, color="green", legality="RH", confidence=0.9):
    return Tile(tile_id, (20, y, 80, y + 60), color, legality, confidence, 0.9)


def _state(tiles=(), *, screen="game", controls=(), modal_substate=None, feed=None, protected_state=None):
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
        feed=feed or FeedObservation(None, (), 0.0, "unresolved"),
        controls=tuple(controls),
        modal_substate=modal_substate,
        protected_state=protected_state or ProtectedState(),
    )


class ContinuousRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, states, *, max_moves=1, execute=True, fresh_states=None, input_size=(240, 500)):
        states = list(states)
        fresh_states = None if fresh_states is None else list(fresh_states)
        images = []
        state_by_pixel = {}
        fresh_images = []
        for index, state in enumerate(states, start=1):
            pixel = (index, 0, 0)
            state_by_pixel[pixel] = state
            images.append(Image.new("RGB", (state.width, state.height), pixel))
        if fresh_states is not None:
            for index, state in enumerate(fresh_states, start=101):
                pixel = (index % 255, index // 255, 1)
                state_by_pixel[pixel] = state
                fresh_images.append(Image.new("RGB", (state.width, state.height), pixel))
        cursor = 0
        fresh_cursor = 0
        taps = []

        def capture_fn(_adb=None, _serial=None):
            nonlocal cursor
            index = min(cursor, len(images) - 1)
            cursor += 1
            return CapturedFrame(images[index], 1.0)

        def revalidation_capture_fn(_adb=None, _serial=None):
            nonlocal fresh_cursor
            if fresh_images:
                index = min(fresh_cursor, len(fresh_images) - 1)
                fresh_cursor += 1
                return CapturedFrame(fresh_images[index], 0.5)
            index = min(max(0, cursor - 1), len(images) - 1)
            return CapturedFrame(images[index], 0.5)

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
            input_size_fn=lambda _adb=None, _serial=None: input_size,
            revalidation_capture_fn=revalidation_capture_fn,
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
        self.assertEqual(captures, 4)
        self.assertEqual(result["status"], "move_limit")

    def test_move_limit_records_two_final_game_observations_without_another_tile_tap(self):
        state = _state([_tile("first", 20), _tile("second", 120)])

        result, taps, captures = self._run([state, state], max_moves=1)
        events = [
            json.loads(line)
            for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
        ]
        observations = [event for event in events if event.get("observation_reason") == "move_limit_boundary"]

        self.assertEqual(result["status"], "move_limit")
        self.assertEqual(result["action_count"], 1)
        self.assertEqual(taps, [(50, 50)])
        self.assertEqual(captures, 4)
        self.assertEqual(len(observations), 2)
        self.assertTrue(all(event.get("event_type") == "observation" for event in observations))
        for event in observations:
            self.assertIsNone(event["chosen"])
            self.assertIsNone(event["tap"])
            self.assertEqual(event["after_state"]["screen"], "game")
            self.assertTrue((Path(result["run_dir"]) / event["frames"]["after"]).is_file())

    def test_move_limit_delayed_out_of_space_enters_safe_recovery_before_finalizing(self):
        game = _state([_tile("first", 20), _tile("second", 120)])
        verified_game = _state([_tile("second", 120)])
        offer = _state(
            screen="out_of_space",
            controls=(ScreenControl("close", (90, 90), 0.99),),
            modal_substate="space_offer",
        )
        warning = _state(
            screen="out_of_space",
            controls=(ScreenControl("close", (91, 91), 0.99),),
            modal_substate="life_warning",
        )
        resumed = _state([_tile("resumed", 120)])

        result, taps, _ = self._run([game, verified_game, offer, warning, resumed], max_moves=1)
        events = [
            json.loads(line)
            for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
        ]
        boundary = next(event for event in events if event.get("observation_reason") == "move_limit_boundary")
        recovery_events = [event for event in events if event["before_state"]["screen"] == "out_of_space"]

        self.assertEqual(result["status"], "move_limit")
        self.assertEqual(taps, [(50, 50), (90, 90), (91, 91)])
        self.assertEqual(boundary["after_state"]["screen"], "out_of_space")
        self.assertEqual(boundary["after_state"]["modal_substate"], "space_offer")
        self.assertEqual([event["tap"] for event in recovery_events], [[90, 90], [91, 91]])
        self.assertEqual(recovery_events[0]["after_state"]["modal_substate"], "life_warning")
        self.assertEqual(recovery_events[1]["before_state"]["modal_substate"], "life_warning")
        self.assertEqual(recovery_events[1]["after_state"]["screen"], "game")

    def test_move_limit_delayed_failure_is_recorded_as_level_failure(self):
        game = _state([_tile("last-action", 20), _tile("remaining", 120)])
        verified_game = _state([_tile("remaining", 120)])
        failure = _state(screen="failure")

        result, taps, captures = self._run([game, verified_game, failure], max_moves=1)
        events = [
            json.loads(line)
            for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
        ]
        boundary = next(event for event in events if event.get("observation_reason") == "move_limit_boundary")
        evidence = [
            json.loads(line)
            for line in (Path(self.temp.name) / "debug" / "mechanic_evidence.jsonl").read_text(encoding="utf-8").splitlines()
        ]

        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["action_count"], 1)
        self.assertEqual(taps, [(50, 50)])
        self.assertEqual(captures, 3)
        self.assertEqual(boundary["after_state"]["screen"], "failure")
        self.assertEqual(boundary["outcomes"]["strategic"], "LEVEL_FAILURE")
        self.assertIn("LEVEL_FAILURE", [record.get("strategic_outcome") for record in evidence])

    def test_second_move_limit_reobservation_catches_late_failure_animation(self):
        game = _state([_tile("last-action", 20), _tile("remaining", 120)])
        verified_game = _state([_tile("remaining", 120)])
        failure = _state(screen="failure")

        result, taps, captures = self._run([game, verified_game, verified_game, failure], max_moves=1)
        events = [
            json.loads(line)
            for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
        ]
        observations = [event for event in events if event.get("observation_reason") == "move_limit_boundary"]

        self.assertEqual(result["status"], "failure")
        self.assertEqual(taps, [(50, 50)])
        self.assertEqual(captures, 4)
        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[-1]["after_state"]["screen"], "failure")

    def test_move_limit_delayed_completion_is_recorded_as_level_success(self):
        game = _state([_tile("last-action", 20), _tile("remaining", 120)])
        verified_game = _state([_tile("remaining", 120)])
        complete = _state(screen="complete")

        result, taps, captures = self._run([game, verified_game, complete], max_moves=1)
        events = [
            json.loads(line)
            for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
        ]
        boundary = next(event for event in events if event.get("observation_reason") == "move_limit_boundary")
        evidence = [
            json.loads(line)
            for line in (Path(self.temp.name) / "debug" / "mechanic_evidence.jsonl").read_text(encoding="utf-8").splitlines()
        ]

        self.assertEqual(result["status"], "success")
        self.assertEqual(taps, [(50, 50)])
        self.assertEqual(captures, 3)
        self.assertEqual(boundary["after_state"]["screen"], "complete")
        self.assertEqual(boundary["outcomes"]["strategic"], "LEVEL_SUCCESS")
        self.assertIn("LEVEL_SUCCESS", [record.get("strategic_outcome") for record in evidence])

    def test_move_limit_allows_failure_navigation_after_cap(self):
        game = _state([_tile("last-action", 20), _tile("remaining", 120)])
        verified_game = _state([_tile("remaining", 120)])
        failure = _state(
            screen="failure",
            controls=(ScreenControl("dismiss_failure", (92, 92), 0.99),),
        )
        unknown = _state(screen="unknown")

        result, taps, _ = self._run([game, verified_game, failure, unknown], max_moves=1)

        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["session_status"], "recovery_incomplete")
        self.assertEqual(taps, [(50, 50), (92, 92)])

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
        self.assertEqual(captures, 5)
        self.assertEqual(result["action_count"], 2)

    def test_unchanged_action_tries_next_candidate_without_repeating(self):
        state = _state([_tile("first", 20), _tile("second", 120)])

        result, taps, captures = self._run([state, state, state], max_moves=2)
        events = [
            json.loads(line) for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
        ]
        actions = [event for event in events if event["chosen"] is not None]

        self.assertEqual(len(taps), 2)
        self.assertEqual(captures, 5)
        self.assertEqual([event["chosen"]["tile"]["id"] for event in actions], ["first", "second"])
        self.assertEqual([event["outcomes"]["interaction"] for event in actions], ["NO_CHANGE", "NO_CHANGE"])

    def test_bbox_jitter_feed_metadata_noise_and_unrelated_tile_change_do_not_retry_same_candidate(self):
        chosen = _tile("chosen", 20)
        other = _tile("other", 120, color="blue")
        before = _state([chosen, other], feed=FeedObservation(
            "red", (), 0.55, "initial direct observation", current_status="direct",
        ))
        moved = Tile(
            id="regenerated-id",
            bbox=(23, 23, 83, 83),
            color=chosen.color,
            legality=chosen.legality,
            confidence=chosen.confidence,
            color_confidence=chosen.color_confidence,
        )
        after = GameState(
            screen="game",
            width=240,
            height=500,
            tiles=(moved, Tile(
                id="other-new-id",
                bbox=other.bbox,
                color="red",
                legality="DNH",
                confidence=other.confidence,
                color_confidence=other.color_confidence,
            )),
            board_region=(20, 23, 83, 180),
            feed=FeedObservation("red", (), 0.9, "tracked after outlet reacquisition", current_status="tracked"),
        )

        result, taps, _ = self._run([before, after], max_moves=None)

        self.assertEqual(len(taps), 1)
        self.assertEqual(result["status"], "no_progress")
        events = [
            json.loads(line)
            for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
        ]
        actions = [event for event in events if event["chosen"] is not None]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["outcomes"]["interaction"], "NO_CHANGE")

    def test_progress_resets_attempts_for_a_changed_state(self):
        before = _state([_tile("same", 20, color="green")])
        changed = _state([_tile("same", 20, color="green", legality="UNKNOWN")])
        after_second = _state([])

        result, taps, _ = self._run([before, changed, after_second], max_moves=2)
        events = [json.loads(line) for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(taps), 2)
        self.assertEqual(events[0]["chosen"]["key"], events[1]["chosen"]["key"])
        self.assertEqual(events[0]["chosen"]["tile"]["color"], "green")
        self.assertEqual(events[1]["chosen"]["tile"]["legality"], "UNKNOWN")

    def test_exhausted_unchanged_candidates_stop_without_retry(self):
        state = _state([_tile("first", 20), _tile("second", 120)])

        result, taps, captures = self._run([state, state, state], max_moves=None)

        self.assertEqual(len(taps), 2)
        self.assertEqual(captures, 3)
        self.assertEqual(result["status"], "no_progress")

    def test_level80_feed_change_reconsiders_previously_unchanged_physical_tile(self):
        feed_a = FeedObservation(
            "indigo", ("pink", "teal", "white", None), 0.9, "visible outlet",
            current_status="direct", upcoming_status="direct",
        )
        feed_b = FeedObservation(
            "white", ("teal", "cyan", "indigo", None), 0.9, "visible outlet",
            current_status="direct", upcoming_status="direct",
        )
        tiles = [_tile("target", 20, color="pink"), _tile("other", 120, color="orange")]
        initial = _state(tiles, feed=feed_a)
        unchanged_a = _state(tiles, feed=feed_a)
        advanced_b = _state(tiles, feed=feed_b)

        result, taps, _ = self._run(
            [initial, unchanged_a, advanced_b, advanced_b], max_moves=None
        )
        events = [
            json.loads(line)
            for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
        ]
        actions = [event for event in events if event.get("chosen") is not None]

        self.assertEqual(result["status"], "no_progress")
        self.assertEqual(len(taps), 4)
        self.assertEqual(
            [event["chosen"]["tile"]["id"] for event in actions],
            ["target", "other", "target", "other"],
        )
        self.assertTrue(all(event["outcomes"]["interaction"] == "NO_CHANGE" for event in actions))
        self.assertEqual(actions[0]["before_state"]["feed"]["current"], "indigo")
        self.assertEqual(actions[2]["before_state"]["feed"]["current"], "white")

    def test_key_overlay_change_is_in_runner_state_signature(self):
        plain = _tile("stable-id", 30)
        marked = Tile(
            id=plain.id,
            bbox=plain.bbox,
            color=plain.color,
            legality=plain.legality,
            confidence=plain.confidence,
            color_confidence=plain.color_confidence,
            mechanic_overlays=("key",),
        )

        self.assertNotEqual(
            main_module._state_signature(_state([plain])),
            main_module._state_signature(_state([marked])),
        )

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
            images.append(Image.new("RGB", (state.width, state.height), pixel))
        cursor = 0
        taps = []
        debug_dir = Path(self.temp.name) / "recovery"

        def capture_fn(_adb=None, _serial=None):
            nonlocal cursor
            index = min(cursor, len(images) - 1)
            cursor += 1
            return CapturedFrame(images[index], 1.0)

        def revalidation_capture_fn(_adb=None, _serial=None):
            index = min(max(0, cursor - 1), len(images) - 1)
            return CapturedFrame(images[index], 0.5)

        def tap_fn(x, y, _adb=None, _serial=None):
            taps.append((x, y))
            if (x, y) == (92, 92):
                event_files = list((debug_dir / "runs").glob("*/events.jsonl"))
                self.assertEqual(len(event_files), 1)
                events_before_navigation = [
                    json.loads(line) for line in event_files[0].read_text(encoding="utf-8").splitlines()
                ]
                self.assertIn("LEVEL_FAILURE", [event["outcomes"]["strategic"] for event in events_before_navigation])
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
            input_size_fn=lambda _adb=None, _serial=None: (240, 500),
            revalidation_capture_fn=revalidation_capture_fn,
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
        images = [Image.new("RGB", (game.width, game.height), (1, 0, 0))]
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
            input_size_fn=lambda _adb=None, _serial=None: (240, 500),
            revalidation_capture_fn=lambda _adb=None, _serial=None: CapturedFrame(images[0], 0.5),
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
        self.assertEqual(captures, 5)
        self.assertEqual(result["action_count"], 1)

    def test_jit_revalidation_uses_fresh_tile_center_and_journals_sent_coordinate(self):
        planned = _state([_tile("old-id", 20)])
        fresh_tile = Tile("new-id", (24, 23, 84, 83), "green", "RH", 0.9, 0.9)
        fresh = _state([fresh_tile])
        after = _state([])

        result, taps, _ = self._run([planned, after], fresh_states=[fresh], max_moves=1)
        event = json.loads(Path(result["events_path"]).read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(taps, [(54, 53)])
        self.assertEqual(event["execution"]["planned_coordinate"], [50, 50])
        self.assertEqual(event["execution"]["revalidated_capture_coordinate"], [54, 53])
        self.assertEqual(event["execution"]["outgoing_coordinate"], [54, 53])
        self.assertEqual(event["execution"]["actual_coordinate_sent_to_adb"], [54, 53])
        self.assertEqual((event["execution"]["capture_width"], event["execution"]["capture_height"]), (240, 500))
        self.assertEqual((event["execution"]["device_input_width"], event["execution"]["device_input_height"]), (240, 500))
        self.assertEqual(event["execution"]["coordinate_transform"], "identity")
        self.assertEqual(event["execution"]["executor_decision"], "ALLOWED")

    def test_jit_disappearance_or_modal_aborts_tile_action_without_tapping_old_point(self):
        planned = _state([_tile("stale", 20)])
        disappeared = _state([])
        result, taps, _ = self._run([planned], fresh_states=[disappeared], max_moves=None)

        self.assertEqual(taps, [])
        self.assertEqual(result["status"], "no_progress")
        events = [json.loads(line) for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events[0]["execution"]["executor_decision"], "ABORTED")
        self.assertEqual(events[0]["execution"]["actual_coordinate_sent_to_adb"], None)

        modal = _state(screen="unknown")
        result, taps, _ = self._run([planned], fresh_states=[modal], max_moves=None)
        self.assertEqual(taps, [])
        self.assertEqual(result["status"], "no_progress")

    def test_coordinate_space_mismatch_blocks_runner_input(self):
        game = _state([_tile("first", 20)])
        result, taps, _ = self._run([game], input_size=(2400, 500))

        self.assertEqual(taps, [])
        self.assertEqual(result["status"], "safety_violation")
        self.assertIn("coordinate_space_mismatch", result["reason"])

    def test_bottom_booster_control_strip_cannot_be_a_board_tap_target(self):
        game = _state([_tile("misdetected-booster", 435)])
        result, taps, _ = self._run([game], max_moves=None)

        self.assertEqual(taps, [])
        self.assertEqual(result["status"], "safety_violation")
        self.assertIn("protected_control_target", result["reason"])

    def test_coin_spend_overrides_local_acceptance_and_hard_stops_runner(self):
        before_protected = ProtectedState(2645, "KNOWN", "AVAILABLE", 4, "KNOWN")
        after_protected = ProtectedState(1745, "KNOWN", "CONSUMED", 5, "KNOWN")
        before_tile = Tile("key-tile", (20, 20, 80, 80), "cyan", "RH", 0.9, 0.9, mechanic_overlays=("key",))
        after_tile = Tile("key-tile-new", (20, 20, 80, 80), "cyan", "RH", 0.9, 0.9)
        before = _state([before_tile], protected_state=before_protected)
        after = _state([after_tile, _tile("unrelated", 120)], protected_state=after_protected)
        result, taps, _ = self._run([before, after, after], max_moves=None)
        event = json.loads(Path(result["events_path"]).read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(len(taps), 1)
        self.assertEqual(result["status"], "safety_violation")
        self.assertIn("unexpected_currency_decrease", result["reason"])
        self.assertEqual(event["outcomes"]["interaction"], "SAFETY_VIOLATION")
        self.assertEqual(event["outcomes"]["interaction_reason"], "unexpected_currency_decrease")
        self.assertIn("unexpected_booster_consumption", event["outcomes"]["safety_violations"])
        self.assertIn("unexpected_holder_capacity_change", event["outcomes"]["safety_violations"])
        self.assertEqual(event["execution"]["actual_coordinate_sent_to_adb"], list(taps[0]))
        self.assertEqual(event["execution"]["pre_action_protected_state"]["coin_balance"], 2645)
        self.assertEqual(event["execution"]["post_action_protected_state"]["coin_balance"], 1745)

    def test_coin_gain_and_normal_board_change_do_not_stop_runner(self):
        protected = ProtectedState(845, "KNOWN", "AVAILABLE", 4, "KNOWN")
        gained = ProtectedState(1745, "KNOWN", "AVAILABLE", 4, "KNOWN")
        before = _state([_tile("first", 20)], protected_state=protected)
        after = _state([], protected_state=gained)

        result, taps, _ = self._run([before, after], max_moves=1)

        self.assertEqual(taps, [(50, 50)])
        self.assertEqual(result["status"], "move_limit")


if __name__ == "__main__":
    unittest.main()
