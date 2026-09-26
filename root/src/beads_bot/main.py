import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .capture import capture_frame
from .debug import save_debug_image
from .input import tap
from .policy import Decision, choose_move
from .vision import analyze_frame


def _state_dict(state, decision: Decision, times: dict[str, float]) -> dict:
    return {
        "screen": state.screen,
        "difficulty": state.difficulty,
        "size": [state.width, state.height],
        "feed": asdict(state.feed),
        "board_region": state.board_region,
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
                "locked": tile.locked,
                "number": tile.number,
            }
            for tile in state.tiles
        ],
        "decision": {
            "tile_id": decision.tile.id if decision.tile else None,
            "center": decision.tile.center if decision.tile else None,
            "reason": decision.reason,
            "confidence": decision.confidence,
        },
        "timings_ms": times,
    }


def _signature(state) -> tuple:
    return (
        state.screen,
        state.feed.current,
        tuple((tile.center, tile.color, tile.legality, tile.locked, tile.number) for tile in state.tiles),
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
    return _state_dict(state, decision, times)


def run_live(args: argparse.Namespace) -> list[dict]:
    results: list[dict] = []
    max_moves = args.max_moves if args.loop else 1
    for move_number in range(1, max_moves + 1):
        total_started = time.perf_counter()
        frame = capture_frame(args.adb, args.serial)
        vision_started = time.perf_counter()
        state = analyze_frame(frame.image)
        vision_ms = (time.perf_counter() - vision_started) * 1000
        policy_started = time.perf_counter()
        decision = choose_move(state)
        policy_ms = (time.perf_counter() - policy_started) * 1000
        record = _state_dict(
            state,
            decision,
            {"capture": frame.elapsed_ms, "vision": round(vision_ms, 2), "policy": round(policy_ms, 2), "click": 0.0},
        )
        if args.debug_dir:
            save_debug_image(frame.image, state, args.debug_dir / f"before_{move_number:03d}.png", decision)

        if decision.tile is None or decision.confidence < args.min_confidence:
            record["execution"] = "paused: no sufficiently confident legal move"
            pause_dir = args.debug_dir or Path("debug")
            save_debug_image(frame.image, state, pause_dir / f"paused_{move_number:03d}.png", decision)
            record["timings_ms"]["total"] = round((time.perf_counter() - total_started) * 1000, 2)
            results.append(record)
            break
        if not args.execute:
            record["execution"] = "dry-run: no input sent"
            record["timings_ms"]["total"] = round((time.perf_counter() - total_started) * 1000, 2)
            results.append(record)
            break

        x, y = decision.tile.center
        record["timings_ms"]["click"] = tap(x, y, args.adb, args.serial)
        time.sleep(max(0.0, args.settle_seconds))
        after_frame = capture_frame(args.adb, args.serial)
        verify_vision_started = time.perf_counter()
        after_state = analyze_frame(after_frame.image)
        record["timings_ms"]["verify_capture"] = after_frame.elapsed_ms
        record["timings_ms"]["verify_vision"] = round((time.perf_counter() - verify_vision_started) * 1000, 2)
        record["verification"] = {
            "changed": _signature(state) != _signature(after_state),
            "after_screen": after_state.screen,
            "after_tile_count": len(after_state.tiles),
        }
        record["timings_ms"]["total"] = round((time.perf_counter() - total_started) * 1000, 2)
        if args.debug_dir:
            save_debug_image(after_frame.image, after_state, args.debug_dir / f"after_{move_number:03d}.png")
        results.append(record)
        if not record["verification"]["changed"] or after_state.screen != "game":
            break
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Beads Out CV agent")
    parser.add_argument("image", nargs="?", type=Path, help="analyze an existing screenshot")
    parser.add_argument("--live", action="store_true", help="capture the connected Android screen through ADB")
    parser.add_argument("--execute", action="store_true", help="send one tap per verified loop iteration")
    parser.add_argument("--loop", action="store_true", help="continue after each tap only when the previous board changed")
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--adb", type=Path, help="path to adb.exe")
    parser.add_argument("--serial", help="ADB device serial when more than one device is connected")
    parser.add_argument("--debug-dir", type=Path, help="save annotated frames and before/after images")
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
