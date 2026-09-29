import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from PIL import Image

from .board import GameState, InteractionOutcome, ScreenControl, StrategicOutcome
from .capture import CapturedFrame, capture_frame
from .debug import save_debug_image
from .input import tap
from .policy import ActionCandidate, Decision, choose_move, prune_attempted, rank_candidates
from .recording import RunRecorder, classify_transition
from .vision import FeedTracker, analyze_frame


_MAX_UNKNOWN_REOBSERVATIONS = 4
_MAX_MOVE_LIMIT_OBSERVATIONS = 2


def _boxes_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3]


def _state_dict(state: GameState, decision: Decision, times: dict[str, float]) -> dict:
    return {
        "screen": state.screen,
        "modal_substate": state.modal_substate,
        "difficulty": state.difficulty,
        "size": [state.width, state.height],
        "feed": asdict(state.feed),
        "board_region": state.board_region,
        "locks": [
            {
                **asdict(lock),
                "overlapping_tile_ids": [
                    tile.id for tile in state.tiles if _boxes_overlap(tile.bbox, lock.bbox)
                ],
            }
            for lock in state.locks
        ],
        "warnings": list(state.warnings),
        "tiles": [
            {
                "id": tile.id,
                "bbox": tile.bbox,
                "center": tile.center,
                "color": tile.color,
                "kind": tile.kind,
                "legality": tile.legality,
                "confidence": tile.confidence,
                "color_confidence": tile.color_confidence,
                "locked": tile.locked,
                "mechanic_overlays": tile.mechanic_overlays,
                "lock_marker_ids": [
                    lock.id for lock in state.locks if _boxes_overlap(tile.bbox, lock.bbox)
                ],
                "number": tile.number,
                "value_status": "UNKNOWN" if tile.kind == "special" and tile.number is None else None,
            }
            for tile in state.tiles
        ],
        "decision": {
            "tile_id": decision.tile.id if decision.tile else None,
            "center": decision.tile.center if decision.tile else None,
            "reason": decision.reason,
            "confidence": decision.confidence,
        },
        "candidates": [
            {
                "rank": rank,
                "key": candidate.key,
                "tile_id": candidate.tile.id,
                "center": candidate.tile.center,
                "color": candidate.tile.color,
                "kind": candidate.tile.kind,
                "legality": candidate.tile.legality,
                "score": candidate.score,
                "confidence": candidate.confidence,
                "context": {
                    "mechanic_id": candidate.context.mechanic_id,
                    "features": candidate.context.features,
                },
                "requirements": [asdict(requirement) for requirement in candidate.requirements],
                "assumptions": list(candidate.assumptions),
            }
            for rank, candidate in enumerate(decision.candidates, start=1)
        ],
        "outcomes": {
            "interaction": InteractionOutcome.NOT_ATTEMPTED.value,
            "strategic": StrategicOutcome.UNKNOWN.value,
        },
        "timings_ms": times,
    }


def _state_signature(state: GameState) -> tuple:
    return (
        state.screen,
        state.modal_substate,
        state.feed.current,
        state.feed.upcoming,
        state.progress,
        tuple(
            (
                tile.id,
                tile.bbox,
                tile.color,
                tile.legality,
                tile.kind,
                tile.locked,
                tile.number,
                tuple(tile.mechanic_overlays),
            )
            for tile in state.tiles
        ),
        tuple((lock.id, lock.center, lock.bbox) for lock in state.locks),
        tuple((control.kind, control.center, control.cost, control.requires_ad) for control in state.controls),
    )


def run_offline(image_path: Path, debug_dir: Path | None) -> dict:
    started = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    vision_started = time.perf_counter()
    state = analyze_frame(image)
    vision_ms = (time.perf_counter() - vision_started) * 1000
    policy_started = time.perf_counter()
    decision = choose_move(state)
    policy_ms = (time.perf_counter() - policy_started) * 1000
    if debug_dir:
        save_debug_image(image, state, debug_dir / f"{image_path.stem}_annotated.png", decision)
    times = {
        "capture": 0.0,
        "vision": round(vision_ms, 2),
        "policy": round(policy_ms, 2),
        "click": 0.0,
        "total": round((time.perf_counter() - started) * 1000, 2),
    }
    report = _state_dict(state, decision, times)
    report["execution"] = "offline: no input sent"
    return report


def _terminal_summary(recorder: RunRecorder, status: str, reason: str, action_count: int, input_count: int) -> dict:
    manifest_path = recorder.finish(status, reason)
    return {
        "run_id": recorder.run_id,
        "status": status,
        "reason": reason,
        "action_count": action_count,
        "input_count": input_count,
        "run_dir": str(recorder.run_dir.resolve()),
        "events_path": str(recorder.events_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "execution": "live: input sent" if input_count else "live: no input sent",
    }


def _record_observation(
    recorder: RunRecorder,
    *,
    frame: Image.Image,
    state: GameState,
    candidates: tuple[ActionCandidate, ...] = (),
    chosen: ActionCandidate | None = None,
    tap_point: tuple[int, int] | None = None,
    after_state: GameState | None = None,
    after_frame: Image.Image | None = None,
    interaction: InteractionOutcome = InteractionOutcome.NOT_ATTEMPTED,
    strategic: StrategicOutcome = StrategicOutcome.IN_PROGRESS,
    interaction_reason: str | None = None,
    local_effect_observed: bool | None = None,
    local_effect_attribution: str | None = None,
    timings: dict[str, float] | None = None,
    observation_reason: str | None = None,
    observation_index: int | None = None,
) -> str:
    return recorder.record_step(
        before_state=state,
        before_frame=frame,
        candidates=candidates,
        chosen=chosen,
        tap=tap_point,
        after_state=after_state,
        after_frame=after_frame,
        interaction_outcome=interaction,
        strategic_outcome=strategic,
        interaction_reason=interaction_reason,
        local_effect_observed=local_effect_observed,
        local_effect_attribution=local_effect_attribution,
        timings_ms=timings or {},
        uncertain_assumptions=chosen.assumptions if chosen else (),
        observation_reason=observation_reason,
        observation_index=observation_index,
    )


def _safe_continue(state: GameState) -> ScreenControl | None:
    safe = [
        control for control in state.controls
        if control.kind == "continue"
        and control.cost is None
        and not control.requires_ad
        and control.confidence >= 0.7
    ]
    return max(safe, key=lambda control: control.confidence, default=None)


def _safe_navigation_control(state: GameState) -> ScreenControl | None:
    """Return only the safe navigation control declared for this screen."""
    expected = {
        "home": "start_level",
        "out_of_space": "close",
        "failure": "dismiss_failure",
    }.get(state.screen)
    if expected is None:
        return None
    return max(
        (
            control for control in state.controls
            if control.kind == expected
            and control.cost is None
            and not control.requires_ad
            and control.confidence >= 0.86
        ),
        key=lambda control: control.confidence,
        default=None,
    )


def _control_transition_outcomes(
    before_state: GameState,
    after_state: GameState,
) -> tuple[InteractionOutcome, StrategicOutcome]:
    interaction = (
        InteractionOutcome.ACTION_ACCEPTED
        if _state_signature(before_state) != _state_signature(after_state)
        else InteractionOutcome.NO_CHANGE
    )
    if after_state.screen == "failure":
        strategic = StrategicOutcome.LEVEL_FAILURE
    elif after_state.screen == "complete":
        strategic = StrategicOutcome.LEVEL_SUCCESS
    elif after_state.screen in ("game", "home", "out_of_space"):
        strategic = StrategicOutcome.IN_PROGRESS
    else:
        strategic = StrategicOutcome.UNKNOWN
    return interaction, strategic


def run_live(
    args: argparse.Namespace,
    *,
    capture_fn: Callable = capture_frame,
    tap_fn: Callable = tap,
    analyze_fn: Callable = analyze_frame,
    recorder_factory: Callable = RunRecorder.create,
    sleep_fn: Callable = time.sleep,
) -> dict:
    """Run continuously when enabled; injected I/O keeps the loop testable offline."""
    debug_root = args.debug_dir or Path("root/debug")
    recorder = recorder_factory(debug_root / "runs", debug_root / "mechanic_evidence.jsonl")
    action_count = 0
    input_count = 0
    max_moves = getattr(args, "max_moves", None)
    attempt_history: list[dict] = []
    last_finalized: dict | None = None
    feed_tracker = FeedTracker()

    def observe_frame(frame: CapturedFrame) -> GameState:
        return feed_tracker.observe(frame.image, analyze_fn(frame.image))

    def finish_attempt(status: str, reason: str) -> dict:
        nonlocal last_finalized
        summary = _terminal_summary(recorder, status, reason, action_count, input_count)
        attempt_history.append({
            "run_id": summary["run_id"],
            "status": summary["status"],
            "reason": summary["reason"],
            "action_count": summary["action_count"],
            "manifest_path": summary["manifest_path"],
        })
        summary["attempt_history"] = list(attempt_history)
        last_finalized = summary
        return summary

    def recovery_incomplete(reason: str) -> dict:
        summary = dict(last_finalized or {})
        summary["session_status"] = "recovery_incomplete"
        summary["session_reason"] = reason
        summary["attempt_history"] = list(attempt_history)
        return summary

    try:
        current_frame = capture_fn(args.adb, args.serial)
        current_state = observe_frame(current_frame)
    except Exception as error:
        return finish_attempt("device_error", f"initial capture failed: {error}")

    attempted: list[ActionCandidate] = []
    unknown_reobservations = 0
    while True:
        state = current_state
        frame = current_frame.image
        if state.screen != "unknown":
            unknown_reobservations = 0
        if recorder is None:
            if state.screen == "unknown" and unknown_reobservations < _MAX_UNKNOWN_REOBSERVATIONS:
                try:
                    sleep_fn(max(0.0, args.settle_seconds))
                    next_frame = capture_fn(args.adb, args.serial)
                    next_state = observe_frame(next_frame)
                except Exception as error:
                    return recovery_incomplete(f"home screen re-observation failed: {error}")
                unknown_reobservations = (
                    unknown_reobservations + 1 if next_state.screen == "unknown" else 0
                )
                current_frame, current_state = next_frame, next_state
                continue
            if state.screen != "home":
                return recovery_incomplete(
                    f"expected the annotated home screen after failure dismissal; observed {state.screen}"
                )
            recorder = recorder_factory(debug_root / "runs", debug_root / "mechanic_evidence.jsonl")
            action_count = 0
            input_count = 0
            attempted.clear()

        if state.screen == "failure":
            _record_observation(
                recorder, frame=frame, state=state,
                interaction=InteractionOutcome.NOT_ATTEMPTED,
                strategic=StrategicOutcome.LEVEL_FAILURE,
            )
            failed_attempt = finish_attempt("failure", "confirmed Level Failed screen")
            if not args.execute:
                return failed_attempt
            control = _safe_navigation_control(state)
            if control is None:
                failed_attempt["session_status"] = "recovery_incomplete"
                failed_attempt["session_reason"] = "no annotated failure close target was recognized"
                return failed_attempt
            try:
                tap_fn(control.center[0], control.center[1], args.adb, args.serial)
                sleep_fn(max(0.0, args.settle_seconds))
                next_frame = capture_fn(args.adb, args.serial)
                next_state = observe_frame(next_frame)
            except Exception as error:
                failed_attempt["session_status"] = "recovery_incomplete"
                failed_attempt["session_reason"] = f"failure dismissal failed: {error}"
                return failed_attempt
            recorder = None
            action_count = 0
            input_count = 0
            attempted.clear()
            current_frame, current_state = next_frame, next_state
            continue

        if state.screen in ("out_of_space", "home"):
            control = _safe_navigation_control(state)
            if not args.execute:
                _record_observation(
                    recorder, frame=frame, state=state,
                    interaction=InteractionOutcome.NOT_ATTEMPTED,
                    strategic=StrategicOutcome.IN_PROGRESS,
                )
                return finish_attempt("dry_run", f"{state.screen} recognized; execution is disabled")
            if control is None:
                _record_observation(
                    recorder, frame=frame, state=state,
                    interaction=InteractionOutcome.NOT_ATTEMPTED,
                    strategic=StrategicOutcome.IN_PROGRESS,
                )
                return finish_attempt("unrecognized_ui", f"{state.screen} has no safe navigation control")
            try:
                click_ms = tap_fn(control.center[0], control.center[1], args.adb, args.serial)
                input_count += 1
                sleep_fn(max(0.0, args.settle_seconds))
                next_frame = capture_fn(args.adb, args.serial)
                next_state = observe_frame(next_frame)
            except Exception as error:
                _record_observation(
                    recorder,
                    frame=frame,
                    state=state,
                    tap_point=control.center,
                    interaction=InteractionOutcome.UNKNOWN,
                    strategic=StrategicOutcome.UNKNOWN,
                    timings={"tap": 0.0},
                )
                return finish_attempt("device_error", f"navigation action failed: {error}")

            interaction, strategic = _control_transition_outcomes(state, next_state)
            _record_observation(
                recorder,
                frame=frame,
                state=state,
                tap_point=control.center,
                after_state=next_state,
                after_frame=next_frame.image,
                interaction=interaction,
                strategic=strategic,
                timings={"tap": float(click_ms), "verify_capture": next_frame.elapsed_ms},
            )
            if interaction == InteractionOutcome.NO_CHANGE:
                return finish_attempt("no_progress", f"{state.screen} navigation target produced no state change")
            if state.screen == "home" and next_state.screen == "game":
                attempted.clear()
            current_frame, current_state = next_frame, next_state
            continue

        if state.screen == "unknown":
            if not args.execute or unknown_reobservations >= _MAX_UNKNOWN_REOBSERVATIONS:
                _record_observation(
                    recorder, frame=frame, state=state,
                    interaction=InteractionOutcome.NOT_ATTEMPTED,
                    strategic=StrategicOutcome.UNKNOWN,
                )
                reason = (
                    "screen remained unrecognized after passive re-observation"
                    if args.execute else "screen is unrecognized; execution is disabled"
                )
                return finish_attempt("unrecognized_ui", reason)
            try:
                sleep_fn(max(0.0, args.settle_seconds))
                next_frame = capture_fn(args.adb, args.serial)
                next_state = observe_frame(next_frame)
            except Exception as error:
                _record_observation(
                    recorder, frame=frame, state=state,
                    interaction=InteractionOutcome.NOT_ATTEMPTED,
                    strategic=StrategicOutcome.UNKNOWN,
                    timings={"verify_capture": 0.0},
                )
                return finish_attempt("device_error", f"passive screen re-observation failed: {error}")
            transition = classify_transition(state, next_state, None)
            _record_observation(
                recorder,
                frame=frame,
                state=state,
                after_state=next_state,
                after_frame=next_frame.image,
                interaction=transition.interaction,
                strategic=transition.strategic,
                interaction_reason=transition.interaction_reason,
                local_effect_observed=transition.local_effect_observed,
                local_effect_attribution=transition.local_effect_attribution,
                timings={"verify_capture": next_frame.elapsed_ms},
            )
            unknown_reobservations = (
                unknown_reobservations + 1 if next_state.screen == "unknown" else 0
            )
            current_frame, current_state = next_frame, next_state
            continue

        if state.screen not in ("game", "complete"):
            _record_observation(
                recorder, frame=frame, state=state,
                interaction=InteractionOutcome.NOT_ATTEMPTED,
                strategic=StrategicOutcome.UNKNOWN,
            )
            return finish_attempt("unrecognized_ui", "screen could not be recognized")

        if state.screen == "complete":
            control = _safe_continue(state)
            if not args.execute or control is None:
                reason = "level completed; no recognized free non-ad continuation" if control is None else "level completed; execution is disabled"
                _record_observation(
                    recorder, frame=frame, state=state,
                    interaction=InteractionOutcome.NOT_ATTEMPTED,
                    strategic=StrategicOutcome.LEVEL_SUCCESS,
                )
                return finish_attempt("success", reason)
            try:
                click_ms = tap_fn(control.center[0], control.center[1], args.adb, args.serial)
                input_count += 1
                sleep_fn(max(0.0, args.settle_seconds))
                next_frame = capture_fn(args.adb, args.serial)
                next_state = observe_frame(next_frame)
            except Exception as error:
                _record_observation(
                    recorder, frame=frame, state=state, tap_point=control.center,
                    interaction=InteractionOutcome.UNKNOWN,
                    strategic=StrategicOutcome.LEVEL_SUCCESS,
                    timings={"tap": 0.0},
                )
                return finish_attempt("device_error", f"continuation action failed: {error}")
            changed = _state_signature(state) != _state_signature(next_state)
            _record_observation(
                recorder,
                frame=frame,
                state=state,
                tap_point=control.center,
                after_state=next_state,
                after_frame=next_frame.image,
                interaction=InteractionOutcome.ACTION_ACCEPTED if changed else InteractionOutcome.NO_CHANGE,
                strategic=StrategicOutcome.LEVEL_SUCCESS,
                timings={"tap": float(click_ms), "verify_capture": next_frame.elapsed_ms},
            )
            if next_state.screen == "game":
                current_frame, current_state = next_frame, next_state
                attempted.clear()
                continue
            if next_state.screen == "failure":
                current_frame, current_state = next_frame, next_state
                continue
            if next_state.screen == "unknown":
                current_frame, current_state = next_frame, next_state
                continue
            return finish_attempt("success", "level completed; continuation did not leave the completion screen")

        if max_moves is not None and action_count >= max_moves:
            observation_state = state
            observation_frame = frame
            for observation_index in range(1, _MAX_MOVE_LIMIT_OBSERVATIONS + 1):
                try:
                    sleep_fn(max(0.0, args.settle_seconds))
                    next_frame = capture_fn(args.adb, args.serial)
                    next_state = observe_frame(next_frame)
                except Exception as error:
                    _record_observation(
                        recorder,
                        frame=observation_frame,
                        state=observation_state,
                        interaction=InteractionOutcome.UNKNOWN,
                        strategic=StrategicOutcome.UNKNOWN,
                        observation_reason="move_limit_boundary",
                        observation_index=observation_index,
                    )
                    return finish_attempt("device_error", f"move-limit observation failed: {error}")

                transition = classify_transition(observation_state, next_state, None)
                _record_observation(
                    recorder,
                    frame=observation_frame,
                    state=observation_state,
                    after_state=next_state,
                    after_frame=next_frame.image,
                    interaction=InteractionOutcome.NOT_ATTEMPTED,
                    strategic=transition.strategic,
                    interaction_reason=transition.interaction_reason,
                    local_effect_observed=transition.local_effect_observed,
                    local_effect_attribution=transition.local_effect_attribution,
                    timings={"verify_capture": next_frame.elapsed_ms},
                    observation_reason="move_limit_boundary",
                    observation_index=observation_index,
                )
                observation_state, observation_frame = next_state, next_frame.image
                current_frame, current_state = next_frame, next_state
                if next_state.screen != "game":
                    break
            else:
                return finish_attempt("move_limit", "user move limit reached; final observations remained in game")
            continue

        policy_started = time.perf_counter()
        prune_attempted(state, attempted)
        candidates = rank_candidates(
            state,
            attempted=tuple(attempted),
            evidence_for=recorder.evidence_for,
        )
        policy_ms = (time.perf_counter() - policy_started) * 1000
        if not candidates:
            _record_observation(
                recorder, frame=frame, state=state,
                interaction=InteractionOutcome.NOT_ATTEMPTED,
                strategic=StrategicOutcome.IN_PROGRESS,
                timings={"policy": round(policy_ms, 2)},
            )
            reason = "no plausible action" if not attempted else "all distinct candidates had no change for this state"
            return finish_attempt("no_progress", reason)

        chosen = candidates[0]
        if not args.execute:
            _record_observation(
                recorder, frame=frame, state=state, candidates=candidates, chosen=chosen,
                interaction=InteractionOutcome.NOT_ATTEMPTED,
                strategic=StrategicOutcome.IN_PROGRESS,
                timings={"policy": round(policy_ms, 2)},
            )
            return finish_attempt("dry_run", "execution is disabled")
        click_started = time.perf_counter()
        try:
            click_ms = tap_fn(chosen.tile.center[0], chosen.tile.center[1], args.adb, args.serial)
            input_count += 1
            action_count += 1
            sleep_fn(max(0.0, args.settle_seconds))
            next_frame = capture_fn(args.adb, args.serial)
            next_state = observe_frame(next_frame)
        except Exception as error:
            click_elapsed = (time.perf_counter() - click_started) * 1000
            _record_observation(
                recorder,
                frame=frame,
                state=state,
                candidates=candidates,
                chosen=chosen,
                tap_point=chosen.tile.center,
                interaction=InteractionOutcome.UNKNOWN,
                strategic=StrategicOutcome.UNKNOWN,
                timings={"policy": round(policy_ms, 2), "tap_and_capture": round(click_elapsed, 2)},
            )
            return finish_attempt("device_error", f"action execution failed: {error}")

        transition = classify_transition(state, next_state, chosen)
        _record_observation(
            recorder,
            frame=frame,
            state=state,
            candidates=candidates,
            chosen=chosen,
            tap_point=chosen.tile.center,
            after_state=next_state,
            after_frame=next_frame.image,
            interaction=transition.interaction,
            strategic=transition.strategic,
            interaction_reason=transition.interaction_reason,
            local_effect_observed=transition.local_effect_observed,
            local_effect_attribution=transition.local_effect_attribution,
            timings={
                "capture": current_frame.elapsed_ms,
                "policy": round(policy_ms, 2),
                "tap": float(click_ms),
                "verify_capture": next_frame.elapsed_ms,
            },
        )
        current_frame, current_state = next_frame, next_state
        if transition.interaction == InteractionOutcome.NO_CHANGE:
            attempted.append(chosen)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Beads Out CV agent")
    parser.add_argument("image", nargs="?", type=Path, help="analyze an existing screenshot")
    parser.add_argument("--live", action="store_true", help="capture the connected Android screen through ADB")
    parser.add_argument("--execute", action="store_true", help="continuously play, observing after every action")
    parser.add_argument("--loop", action="store_true", help="deprecated compatibility flag; --execute is continuous")
    parser.add_argument("--max-moves", type=int, default=None, help="optional cap on gameplay actions; default is unbounded")
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--adb", type=Path, help="path to adb.exe")
    parser.add_argument("--serial", help="ADB device serial when more than one device is connected")
    parser.add_argument("--debug-dir", type=Path, help="directory for run journals, frames, and evidence")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.execute and not args.live:
        raise SystemExit("--execute requires --live")
    if args.live:
        output = run_live(args)
    elif args.image:
        output = run_offline(args.image, args.debug_dir)
    else:
        raise SystemExit("Provide an image path or use --live")
    print(json.dumps(output, indent=2))
    return 0
