from dataclasses import replace
from pathlib import Path
import unittest

from PIL import Image, ImageDraw

from beads_bot.board import (
    FeedObservation,
    GameState,
    InteractionOutcome,
    ProtectedState,
    ScreenControl,
    Tile,
)
from beads_bot.capture import parse_input_size
from beads_bot.policy import rank_candidates
from beads_bot.recording import classify_transition
from beads_bot.safety import (
    ActionRisk,
    ExecutorDecision,
    map_capture_coordinate,
    protected_state_violations,
    revalidate_board_action,
    classify_action_risk,
)
from beads_bot.vision import analyze_frame


ROOT = Path(__file__).resolve().parents[1]
SAFETY_FIXTURES = ROOT / "img" / "safety"


def _tile(tile_id="tile-1", bbox=(300, 1500, 420, 1640), *, color="cyan", legality="RH",
          mechanic_overlays=()):
    return Tile(
        tile_id,
        bbox,
        color,
        legality,
        0.95,
        0.9,
        mechanic_overlays=mechanic_overlays,
    )


def _state(tiles=(), *, screen="game", width=1080, height=2400, controls=(), protected=None):
    tiles = tuple(tiles)
    board = None if not tiles else (
        min(tile.bbox[0] for tile in tiles),
        min(tile.bbox[1] for tile in tiles),
        max(tile.bbox[2] for tile in tiles),
        max(tile.bbox[3] for tile in tiles),
    )
    return GameState(
        screen=screen,
        width=width,
        height=height,
        tiles=tiles,
        board_region=board,
        feed=FeedObservation(None, (), 0.0, "unresolved"),
        controls=tuple(controls),
        protected_state=protected or ProtectedState(),
    )


class ProtectedPerceptionTests(unittest.TestCase):
    def _read(self, name):
        return analyze_frame(SAFETY_FIXTURES / name).protected_state

    def test_incident_a_before_and_after_read_all_protected_signals(self):
        before = self._read("resource_spend_incident_a_before.jpg")
        after = self._read("resource_spend_incident_a_after.jpg")

        self.assertEqual((before.coin_balance, before.coin_status), (2645, "KNOWN"))
        self.assertEqual((after.coin_balance, after.coin_status), (1745, "KNOWN"))
        self.assertEqual(before.extra_holder_booster, "AVAILABLE")
        self.assertEqual(after.extra_holder_booster, "CONSUMED")
        self.assertEqual((before.holder_capacity, after.holder_capacity), (4, 5))

    def test_incident_b_before_and_after_read_all_protected_signals(self):
        before = self._read("resource_spend_incident_b_before.jpg")
        after = self._read("resource_spend_incident_b_after.jpg")

        self.assertEqual((before.coin_balance, after.coin_balance), (1745, 845))
        self.assertEqual(before.coin_status, "KNOWN")
        self.assertEqual(after.coin_status, "KNOWN")
        self.assertEqual(before.extra_holder_booster, "AVAILABLE")
        self.assertEqual(after.extra_holder_booster, "CONSUMED")
        self.assertEqual((before.holder_capacity, after.holder_capacity), (4, 5))

    def test_both_real_incident_pairs_trigger_protected_side_effect_violations(self):
        for prefix, before_coins, after_coins in (
            ("a", 2645, 1745),
            ("b", 1745, 845),
        ):
            before = self._read(f"resource_spend_incident_{prefix}_before.jpg")
            after = self._read(f"resource_spend_incident_{prefix}_after.jpg")
            before_state = _state([_tile()], protected=before)
            after_state = _state([_tile()], protected=after)

            self.assertEqual(before.coin_balance, before_coins)
            self.assertEqual(after.coin_balance, after_coins)
            self.assertEqual(
                protected_state_violations(before_state, after_state, ActionRisk.BOARD_ACTION),
                (
                    "unexpected_currency_decrease",
                    "unexpected_booster_consumption",
                    "unexpected_holder_capacity_change",
                ),
            )

    def test_unmatched_coin_counter_is_unknown_instead_of_guessed(self):
        image = Image.open(SAFETY_FIXTURES / "resource_spend_incident_a_before.jpg").convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((830, 170, 980, 220), fill=(244, 240, 214))

        protected = analyze_frame(image).protected_state

        self.assertIsNone(protected.coin_balance)
        self.assertEqual(protected.coin_status, "UNKNOWN")


class CoordinateIntegrityTests(unittest.TestCase):
    def test_identity_geometry_sends_the_same_capture_coordinate(self):
        mapped = map_capture_coordinate((408, 1751), (1080, 2400), (1080, 2400))

        self.assertTrue(mapped.allowed)
        self.assertEqual(mapped.capture_coordinate, (408, 1751))
        self.assertEqual(mapped.input_coordinate, (408, 1751))
        self.assertEqual(mapped.transform, "identity")

    def test_same_aspect_ratio_uses_explicit_testable_scale(self):
        mapped = map_capture_coordinate((321, 456), (1080, 2400), (720, 1600))

        self.assertTrue(mapped.allowed)
        self.assertEqual(mapped.input_coordinate, (214, 304))
        self.assertEqual(mapped.transform, "uniform_scale")

    def test_unresolved_or_incompatible_input_geometry_fails_closed(self):
        unresolved = map_capture_coordinate((100, 200), (1080, 2400), None)
        incompatible = map_capture_coordinate((100, 200), (1080, 2400), (2400, 1080))

        self.assertFalse(unresolved.allowed)
        self.assertFalse(incompatible.allowed)
        self.assertIsNone(unresolved.input_coordinate)
        self.assertEqual(unresolved.reason, "coordinate_space_mismatch")

    def test_android_wm_size_parser_uses_override_and_rejects_unresolved_output(self):
        self.assertEqual(
            parse_input_size("Physical size: 1080x2400\nOverride size: 720x1600"),
            (720, 1600),
        )
        self.assertEqual(parse_input_size("Physical size: 1080x2400"), (1080, 2400))
        self.assertIsNone(parse_input_size("Physical size: unknown"))


class ActionRiskAndRevalidationTests(unittest.TestCase):
    def test_risk_classification_allows_only_board_and_whitelisted_safe_navigation(self):
        close = ScreenControl("close", (900, 300), 0.99)
        paid = ScreenControl("free_space", (500, 1500), 1.0, cost=900)
        ad = ScreenControl("continue", (500, 1500), 1.0, requires_ad=True)

        self.assertEqual(classify_action_risk("tile-selection"), ActionRisk.BOARD_ACTION)
        self.assertEqual(classify_action_risk(control=close), ActionRisk.SAFE_NAVIGATION)
        self.assertEqual(classify_action_risk(control=paid), ActionRisk.RESOURCE_SPEND)
        self.assertEqual(classify_action_risk(control=ad), ActionRisk.EXTERNAL_SIDE_EFFECT)
        self.assertEqual(classify_action_risk(control=ScreenControl("mystery", (1, 1), 1.0)), ActionRisk.UNKNOWN_ACTION)

    def test_jit_revalidation_uses_fresh_physical_tile_center_after_jitter(self):
        planned = _state([_tile()])
        candidate = rank_candidates(planned)[0]
        fresh_tile = replace(candidate.tile, id="regenerated", bbox=(303, 1502, 423, 1642))
        fresh = _state([fresh_tile])

        gate = revalidate_board_action(planned, candidate, fresh, (1080, 2400), (1080, 2400))

        self.assertEqual(gate.executor_decision, ExecutorDecision.ALLOWED)
        self.assertEqual(gate.planned_capture_coordinate, candidate.tile.center)
        self.assertEqual(gate.revalidated_capture_coordinate, fresh_tile.center)
        self.assertEqual(gate.outgoing_coordinate, fresh_tile.center)

    def test_jit_revalidation_aborts_disappeared_or_changed_targets_and_modal_frames(self):
        planned = _state([_tile()])
        candidate = rank_candidates(planned)[0]
        disappeared = revalidate_board_action(planned, candidate, _state([]), (1080, 2400), (1080, 2400))
        changed = revalidate_board_action(
            planned,
            candidate,
            _state([replace(candidate.tile, legality="DNH")]),
            (1080, 2400),
            (1080, 2400),
        )
        modal = revalidate_board_action(
            planned,
            candidate,
            _state(screen="out_of_space"),
            (1080, 2400),
            (1080, 2400),
        )

        for result in (disappeared, changed, modal):
            self.assertEqual(result.executor_decision, ExecutorDecision.ABORTED)
            self.assertEqual(result.reason, "stale_action_aborted")
            self.assertIsNone(result.outgoing_coordinate)

    def test_board_target_inside_bottom_protected_control_band_is_blocked(self):
        tile = _tile("booster-misclassified", (300, 2200, 420, 2340))
        planned = _state([tile])
        candidate = rank_candidates(planned)[0]

        gate = revalidate_board_action(planned, candidate, planned, (1080, 2400), (1080, 2400))

        self.assertEqual(gate.executor_decision, ExecutorDecision.BLOCKED)
        self.assertEqual(gate.reason, "protected_control_target")
        self.assertIsNone(gate.outgoing_coordinate)


class ProtectedStateInvariantTests(unittest.TestCase):
    def setUp(self):
        self.known = ProtectedState(2645, "KNOWN", "AVAILABLE", 4, "KNOWN")

    def test_each_unexpected_protected_decrease_or_purchase_signal_is_a_violation(self):
        cases = (
            (replace(self.known, coin_balance=1745), "unexpected_currency_decrease"),
            (replace(self.known, extra_holder_booster="CONSUMED"), "unexpected_booster_consumption"),
            (replace(self.known, holder_capacity=5), "unexpected_holder_capacity_change"),
        )
        before = _state([_tile()], protected=self.known)
        for protected_after, reason in cases:
            with self.subTest(reason=reason):
                after = _state([_tile()], protected=protected_after)
                self.assertIn(reason, protected_state_violations(before, after, ActionRisk.BOARD_ACTION))

    def test_coin_gains_and_normal_board_evolution_are_not_violations(self):
        before = _state([_tile()], protected=self.known)
        gain = _state(
            [_tile("new-id", color="orange")],
            protected=replace(self.known, coin_balance=2745),
        )

        self.assertEqual(protected_state_violations(before, gain, ActionRisk.BOARD_ACTION), ())

    def test_unknown_protected_values_do_not_authorize_or_invent_a_delta(self):
        unknown = ProtectedState()
        before = _state([_tile()], protected=unknown)
        after = _state([_tile()], protected=ProtectedState(1745, "KNOWN", "UNKNOWN", 5, "KNOWN"))

        self.assertEqual(protected_state_violations(before, after, ActionRisk.BOARD_ACTION), ())

    def test_local_overlay_acceptance_is_overridden_by_incident_a_resource_loss(self):
        before_tile = _tile(mechanic_overlays=("key",))
        chosen = rank_candidates(_state([before_tile], protected=self.known))[0]
        after_tile = replace(before_tile, mechanic_overlays=())
        after = _state(
            [after_tile],
            protected=ProtectedState(1745, "KNOWN", "CONSUMED", 5, "KNOWN"),
        )

        transition = classify_transition(
            _state([before_tile], protected=self.known),
            after,
            chosen,
            action_risk=ActionRisk.BOARD_ACTION,
        )

        self.assertEqual(transition.interaction, InteractionOutcome.SAFETY_VIOLATION)
        self.assertEqual(transition.interaction_reason, "unexpected_currency_decrease")
        self.assertIn("unexpected_booster_consumption", transition.safety_violations)
        self.assertIn("unexpected_holder_capacity_change", transition.safety_violations)

    def test_chosen_tile_disappearance_cannot_mask_incident_b_booster_consumption(self):
        before = _state([_tile()], protected=self.known)
        chosen = rank_candidates(before)[0]
        after = _state([], protected=replace(self.known, extra_holder_booster="CONSUMED"))

        transition = classify_transition(before, after, chosen, action_risk=ActionRisk.BOARD_ACTION)

        self.assertEqual(transition.interaction, InteractionOutcome.SAFETY_VIOLATION)
        self.assertEqual(transition.interaction_reason, "unexpected_booster_consumption")


if __name__ == "__main__":
    unittest.main()
