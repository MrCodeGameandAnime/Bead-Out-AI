"""Execution-level action classification, coordinate integrity, and side-effect checks.

This module deliberately sits below policy.  Policy may rank an uncertain tile,
but it cannot authorize a resource action or bypass just-in-time validation.
"""

from dataclasses import dataclass
from enum import Enum
from math import hypot

from .board import GameState, ScreenControl, Tile
from .matching import same_physical_tile, tile_interaction_state


class ActionRisk(str, Enum):
    BOARD_ACTION = "BOARD_ACTION"
    SAFE_NAVIGATION = "SAFE_NAVIGATION"
    RESOURCE_SPEND = "RESOURCE_SPEND"
    EXTERNAL_SIDE_EFFECT = "EXTERNAL_SIDE_EFFECT"
    UNKNOWN_ACTION = "UNKNOWN_ACTION"


class ExecutorDecision(str, Enum):
    ALLOWED = "ALLOWED"
    ABORTED = "ABORTED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class CoordinateMapping:
    allowed: bool
    capture_coordinate: tuple[int, int]
    input_coordinate: tuple[int, int] | None
    transform: str
    capture_size: tuple[int, int]
    input_size: tuple[int, int] | None
    reason: str | None = None


@dataclass(frozen=True)
class ActionValidation:
    executor_decision: ExecutorDecision
    reason: str
    planned_capture_coordinate: tuple[int, int]
    revalidated_capture_coordinate: tuple[int, int] | None
    outgoing_coordinate: tuple[int, int] | None
    target: Tile | None = None
    coordinate_mapping: CoordinateMapping | None = None


_SAFE_CONTROLS = {
    "close",
    "dismiss_failure",
    "start_level",
    "continue",
}


def classify_action_risk(action: str | None = None, *, control: ScreenControl | None = None) -> ActionRisk:
    if control is None:
        if action in ("tile-selection", "board-action", "gameplay-tile"):
            return ActionRisk.BOARD_ACTION
        if action in ("resource-spend", "purchase", "paid-retry", "booster"):
            return ActionRisk.RESOURCE_SPEND
        if action in ("ad", "external", "share", "external-side-effect"):
            return ActionRisk.EXTERNAL_SIDE_EFFECT
        if action in ("close", "dismiss_failure", "start_level", "continue"):
            return ActionRisk.SAFE_NAVIGATION
        return ActionRisk.UNKNOWN_ACTION
    if control.cost is not None:
        return ActionRisk.RESOURCE_SPEND
    if control.requires_ad:
        return ActionRisk.EXTERNAL_SIDE_EFFECT
    if control.kind in _SAFE_CONTROLS:
        return ActionRisk.SAFE_NAVIGATION
    return ActionRisk.UNKNOWN_ACTION


def map_capture_coordinate(
    coordinate: tuple[int, int],
    capture_size: tuple[int, int],
    input_size: tuple[int, int] | None,
) -> CoordinateMapping:
    """Map a pixel from screenshot space into ADB input space, failing closed.

    Only identity or uniform scaling is accepted.  Rotation/cropping or an
    unknown device size needs an explicit calibration before input is allowed.
    """
    capture_width, capture_height = capture_size
    x, y = coordinate
    if capture_width <= 0 or capture_height <= 0 or not (0 <= x < capture_width and 0 <= y < capture_height):
        return CoordinateMapping(False, coordinate, None, "unresolved", capture_size, input_size, "coordinate_space_mismatch")
    if input_size is None:
        return CoordinateMapping(False, coordinate, None, "unresolved", capture_size, None, "coordinate_space_mismatch")
    input_width, input_height = input_size
    if input_width <= 0 or input_height <= 0:
        return CoordinateMapping(False, coordinate, None, "unresolved", capture_size, input_size, "coordinate_space_mismatch")
    capture_ratio = capture_width / capture_height
    input_ratio = input_width / input_height
    if abs(capture_ratio - input_ratio) / max(capture_ratio, input_ratio) > 0.005:
        return CoordinateMapping(False, coordinate, None, "incompatible", capture_size, input_size, "coordinate_space_mismatch")
    if capture_size == input_size:
        mapped = coordinate
        transform = "identity"
    else:
        scale = input_width / capture_width
        mapped = (round(x * scale), round(y * scale))
        transform = "uniform_scale"
    if not (0 <= mapped[0] < input_width and 0 <= mapped[1] < input_height):
        return CoordinateMapping(False, coordinate, None, transform, capture_size, input_size, "coordinate_space_mismatch")
    return CoordinateMapping(True, coordinate, mapped, transform, capture_size, input_size)


def _compatible_target(planned: Tile, fresh: Tile) -> bool:
    if tile_interaction_state(planned) != tile_interaction_state(fresh):
        return False
    if planned.color is not None and fresh.color is not None and planned.color != fresh.color:
        return False
    return True


def _protected_control_region(state: GameState, point: tuple[int, int]) -> bool:
    """Exclude the bottom booster strip using screenshot-relative geometry."""
    x, y = point
    if y >= round(state.height * 0.86):
        return True
    return any(
        control.kind in {"booster", "extra_holder", "add_slot", "coin", "purchase"}
        and hypot(x - control.center[0], y - control.center[1]) <= max(24, min(state.width, state.height) * 0.045)
        for control in state.controls
    )


def revalidate_board_action(
    planned_state: GameState,
    candidate,
    fresh_state: GameState,
    capture_size: tuple[int, int],
    input_size: tuple[int, int] | None,
) -> ActionValidation:
    """Re-find the same tile in the latest state and derive its fresh tap point."""
    planned_coordinate = candidate.tile.center
    if fresh_state.screen != "game":
        return ActionValidation(ExecutorDecision.ABORTED, "stale_action_aborted", planned_coordinate, None, None)
    if fresh_state.width != capture_size[0] or fresh_state.height != capture_size[1]:
        return ActionValidation(ExecutorDecision.BLOCKED, "coordinate_space_mismatch", planned_coordinate, None, None)
    matches = [
        tile for tile in fresh_state.tiles
        if same_physical_tile(candidate.tile, tile, (planned_state.width, planned_state.height),
                              (fresh_state.width, fresh_state.height))
    ]
    if not matches:
        return ActionValidation(ExecutorDecision.ABORTED, "stale_action_aborted", planned_coordinate, None, None)
    fresh_tile = min(
        matches,
        key=lambda tile: hypot(
            candidate.tile.center[0] / planned_state.width - tile.center[0] / fresh_state.width,
            candidate.tile.center[1] / planned_state.height - tile.center[1] / fresh_state.height,
        ),
    )
    if not _compatible_target(candidate.tile, fresh_tile):
        return ActionValidation(ExecutorDecision.ABORTED, "stale_action_aborted", planned_coordinate, fresh_tile.center, None, fresh_tile)
    point = fresh_tile.center
    if fresh_state.board_region is None:
        return ActionValidation(ExecutorDecision.ABORTED, "stale_action_aborted", planned_coordinate, point, None, fresh_tile)
    left, top, right, bottom = fresh_tile.bbox
    inset_x = max(1, round((right - left) * 0.15))
    inset_y = max(1, round((bottom - top) * 0.15))
    safe_box = (left + inset_x, top + inset_y, right - inset_x, bottom - inset_y)
    board_left, board_top, board_right, board_bottom = fresh_state.board_region
    if not (
        safe_box[0] <= point[0] <= safe_box[2]
        and safe_box[1] <= point[1] <= safe_box[3]
        and board_left <= point[0] <= board_right
        and board_top <= point[1] <= board_bottom
    ):
        return ActionValidation(ExecutorDecision.BLOCKED, "protected_control_target", planned_coordinate, point, None, fresh_tile)
    if _protected_control_region(fresh_state, point):
        return ActionValidation(ExecutorDecision.BLOCKED, "protected_control_target", planned_coordinate, point, None, fresh_tile)
    mapping = map_capture_coordinate(point, capture_size, input_size)
    if not mapping.allowed:
        return ActionValidation(ExecutorDecision.BLOCKED, "coordinate_space_mismatch", planned_coordinate, point, None, fresh_tile, mapping)
    return ActionValidation(ExecutorDecision.ALLOWED, "revalidation_passed", planned_coordinate, point, mapping.input_coordinate, fresh_tile, mapping)


def revalidate_safe_control(
    planned_state: GameState,
    control: ScreenControl,
    fresh_state: GameState,
    capture_size: tuple[int, int],
    input_size: tuple[int, int] | None,
) -> ActionValidation:
    planned_coordinate = control.center
    risk = classify_action_risk(control=control)
    if risk != ActionRisk.SAFE_NAVIGATION or fresh_state.screen != planned_state.screen:
        reason = "protected_control_target" if risk in (ActionRisk.RESOURCE_SPEND, ActionRisk.EXTERNAL_SIDE_EFFECT) else "stale_action_aborted"
        return ActionValidation(ExecutorDecision.BLOCKED if reason == "protected_control_target" else ExecutorDecision.ABORTED,
                                reason, planned_coordinate, None, None)
    matches = [item for item in fresh_state.controls if item.kind == control.kind]
    if not matches:
        return ActionValidation(ExecutorDecision.ABORTED, "stale_action_aborted", planned_coordinate, None, None)
    fresh_control = max(matches, key=lambda item: item.confidence)
    if fresh_control.cost is not None or fresh_control.requires_ad:
        return ActionValidation(ExecutorDecision.BLOCKED, "protected_control_target", planned_coordinate, fresh_control.center, None)
    if hypot(
        control.center[0] / planned_state.width - fresh_control.center[0] / fresh_state.width,
        control.center[1] / planned_state.height - fresh_control.center[1] / fresh_state.height,
    ) > 0.04:
        return ActionValidation(ExecutorDecision.ABORTED, "stale_action_aborted", planned_coordinate, fresh_control.center, None)
    mapping = map_capture_coordinate(fresh_control.center, capture_size, input_size)
    if not mapping.allowed:
        return ActionValidation(ExecutorDecision.BLOCKED, "coordinate_space_mismatch", planned_coordinate, fresh_control.center, None, coordinate_mapping=mapping)
    return ActionValidation(ExecutorDecision.ALLOWED, "revalidation_passed", planned_coordinate, fresh_control.center, mapping.input_coordinate, coordinate_mapping=mapping)


def protected_state_violations(before_state: GameState, after_state: GameState, risk: ActionRisk) -> tuple[str, ...]:
    """Report protected deltas; no automated spend class currently has authorization."""
    # Even RESOURCE_SPEND is rejected by the executor today, so no risk class
    # suppresses detection. A future authorized purchase needs a separate,
    # explicit authorization token rather than merely a risk label.
    _ = risk
    before = before_state.protected_state
    after = after_state.protected_state
    violations: list[str] = []
    if (
        before.coin_status == "KNOWN" and after.coin_status == "KNOWN"
        and before.coin_balance is not None and after.coin_balance is not None
        and after.coin_balance < before.coin_balance
    ):
        violations.append("unexpected_currency_decrease")
    if before.extra_holder_booster == "AVAILABLE" and after.extra_holder_booster == "CONSUMED":
        violations.append("unexpected_booster_consumption")
    if (
        before.holder_capacity_status == "KNOWN" and after.holder_capacity_status == "KNOWN"
        and before.holder_capacity is not None and after.holder_capacity is not None
        and after.holder_capacity > before.holder_capacity
    ):
        violations.append("unexpected_holder_capacity_change")
    return tuple(violations)
