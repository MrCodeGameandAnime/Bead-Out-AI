import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from beads_bot.board import (
    EvidenceContext,
    FeedObservation,
    GameState,
    InteractionOutcome,
    StrategicOutcome,
    Tile,
)
from beads_bot.policy import rank_candidates
from beads_bot.recording import RunRecorder, classify_transition


_DEFAULT_CHOSEN = object()


def _state(tiles, *, screen="game", progress=None):
    tile_list = tuple(tiles)
    board = None if not tile_list else (
        min(tile.bbox[0] for tile in tile_list),
        min(tile.bbox[1] for tile in tile_list),
        max(tile.bbox[2] for tile in tile_list),
        max(tile.bbox[3] for tile in tile_list),
    )
    return GameState(
        screen=screen,
        width=240,
        height=400,
        tiles=tile_list,
        board_region=board,
        feed=FeedObservation(None, (), 0.0, "unresolved"),
        progress=progress,
    )


def _tile(tile_id="tile-1", y=30):
    return Tile(tile_id, (30, y, 90, y + 60), "green", "RH", 0.9, 0.9)


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.recorder = RunRecorder.create(root / "runs", root / "evidence.jsonl")
        self.frame = Image.new("RGB", (240, 400), "beige")
        self.before = _state([_tile()])
        self.candidate = rank_candidates(self.before)[0]

    def tearDown(self):
        self.temp.cleanup()

    def _record(self, *, before=None, after=None, candidates=None, chosen=_DEFAULT_CHOSEN,
                tap=(60, 60), interaction=None, strategic=None, assumptions=None):
        return self.recorder.record_step(
            before_state=before or self.before,
            before_frame=self.frame,
            candidates=tuple(candidates if candidates is not None else (self.candidate,)),
            chosen=self.candidate if chosen is _DEFAULT_CHOSEN else chosen,
            tap=tap,
            after_state=after or self.before,
            after_frame=self.frame,
            interaction_outcome=interaction or InteractionOutcome.ACTION_ACCEPTED,
            strategic_outcome=strategic or StrategicOutcome.IN_PROGRESS,
            timings_ms={"capture": 2.5, "tap": 1.2},
            uncertain_assumptions=tuple(assumptions or ("feed color is unknown",)),
        )

    def test_action_event_keeps_states_candidates_tap_and_both_outcomes(self):
        after = _state([])
        interaction, strategic = classify_transition(self.before, after, self.candidate)

        step_id = self._record(after=after, interaction=interaction, strategic=strategic)
        event = json.loads(self.recorder.events_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(event["step_id"], step_id)
        self.assertEqual(event["before_state"]["tiles"][0]["id"], "tile-1")
        self.assertEqual(event["after_state"]["tiles"], [])
        self.assertEqual(event["candidates"][0]["tile"]["id"], "tile-1")
        self.assertEqual(event["chosen"]["key"], self.candidate.key)
        self.assertEqual(event["tap"], [60, 60])
        self.assertEqual(event["outcomes"]["interaction"], "ACTION_ACCEPTED")
        self.assertEqual(event["outcomes"]["strategic"], "IN_PROGRESS")
        self.assertTrue((self.recorder.run_dir / event["frames"]["before"]).is_file())
        self.assertTrue((self.recorder.run_dir / event["frames"]["after"]).is_file())

    def test_failure_manifest_links_recent_action_sequence(self):
        for _ in range(10):
            self._record(after=self.before, interaction=InteractionOutcome.NO_CHANGE)

        manifest_path = self.recorder.finish("failure", "confirmed failure")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        event_lines = self.recorder.events_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(event_lines), 10)
        self.assertEqual(len(manifest["recent_steps"]), 8)
        self.assertEqual(manifest["recent_steps"][0]["step_id"], "step-000003")
        self.assertEqual(manifest["recent_steps"][-1]["step_id"], "step-000010")
        self.assertIn("before_state", manifest["recent_steps"][0])
        self.assertIn("candidates", manifest["recent_steps"][0])
        self.assertIn("tap", manifest["recent_steps"][0])
        self.assertIn("uncertain_assumptions", manifest["recent_steps"][0])
        self.assertEqual(manifest["status"], "failure")

    def test_no_candidate_observation_is_journaled(self):
        step_id = self._record(candidates=(), chosen=None, tap=None, interaction=InteractionOutcome.NOT_ATTEMPTED)
        manifest_path = self.recorder.finish("no_progress", "no plausible action")
        event = json.loads(self.recorder.events_path.read_text(encoding="utf-8").splitlines()[0])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(event["step_id"], step_id)
        self.assertEqual(event["candidates"], [])
        self.assertIsNone(event["chosen"])
        self.assertIsNone(event["tap"])
        self.assertEqual(event["outcomes"]["interaction"], "NOT_ATTEMPTED")
        self.assertEqual(manifest["reason"], "no plausible action")

    def test_tile_disappearance_is_acceptance_not_local_progress(self):
        after = _state([])

        interaction, strategic = classify_transition(self.before, after, self.candidate)
        self._record(after=after, interaction=interaction, strategic=strategic)
        tally = self.recorder.evidence_for(self.candidate.context)

        self.assertEqual(interaction, InteractionOutcome.ACTION_ACCEPTED)
        self.assertEqual(strategic, StrategicOutcome.IN_PROGRESS)
        self.assertEqual(tally.action_accepted, 1)
        self.assertEqual(tally.local_progress, 0)

    def test_unrecognized_after_state_does_not_prove_tap_was_accepted(self):
        after = _state([], screen="unknown")

        interaction, strategic = classify_transition(self.before, after, self.candidate)

        self.assertEqual(interaction, InteractionOutcome.UNKNOWN)
        self.assertEqual(strategic, StrategicOutcome.UNKNOWN)

    def test_explicit_progress_signal_records_local_progress(self):
        before = _state([_tile()], progress=1)
        after = _state([], progress=2)

        interaction, strategic = classify_transition(before, after, self.candidate)
        self._record(before=before, after=after, interaction=interaction, strategic=strategic)
        tally = self.recorder.evidence_for(self.candidate.context)

        self.assertEqual(interaction, InteractionOutcome.ACTION_ACCEPTED)
        self.assertEqual(strategic, StrategicOutcome.LOCAL_PROGRESS)
        self.assertEqual(tally.action_accepted, 1)
        self.assertEqual(tally.local_progress, 1)

    def test_level_outcome_is_stored_as_episode_evidence(self):
        after = _state([], screen="complete")
        interaction, strategic = classify_transition(self.before, after, self.candidate)
        self._record(after=after, interaction=interaction, strategic=strategic)
        self.recorder.finish("success", "level complete")
        records = [json.loads(line) for line in self.recorder.evidence_path.read_text(encoding="utf-8").splitlines()]
        episode = next(item for item in records if item["record_type"] == "episode")

        self.assertEqual(strategic, StrategicOutcome.LEVEL_SUCCESS)
        self.assertEqual(episode["mechanic_id"], "action-sequence")
        self.assertEqual(episode["strategic_outcome"], "LEVEL_SUCCESS")
        self.assertEqual(len(episode["sequence_contexts"]), 1)
        self.assertEqual(episode["sequence_contexts"][0]["mechanic_id"], "tile-selection")

    def test_context_ledger_round_trips_future_features(self):
        context = EvidenceContext(
            mechanic_id="magnet-lock",
            features={
                "tile_color": "green",
                "local_geometry": {"offset": 0.12, "depth": "raised"},
                "neighbors": [{"color": "blue", "legality": "RH"}],
                "unmodeled_future_feature": {"capture_revision": 3, "marker": "gold"},
            },
        )
        candidate = replace(self.candidate, context=context)
        self._record(chosen=candidate, candidates=(candidate,), interaction=InteractionOutcome.ACTION_ACCEPTED)
        richer_context = EvidenceContext(
            mechanic_id="magnet-lock",
            features={**context.features, "feature_added_after_recording": True},
        )

        tally = self.recorder.evidence_for(richer_context)
        stored = json.loads(self.recorder.evidence_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(tally.action_accepted, 1)
        self.assertEqual(stored["schema_version"], 1)
        self.assertEqual(stored["mechanic_id"], "magnet-lock")
        self.assertEqual(stored["features"]["unmodeled_future_feature"]["capture_revision"], 3)
        self.assertEqual(stored["features"]["neighbors"][0]["color"], "blue")

    def test_no_change_stays_contextual_and_does_not_mark_rule_invalid(self):
        self._record(after=self.before, interaction=InteractionOutcome.NO_CHANGE)

        tally = self.recorder.evidence_for(self.candidate.context)

        self.assertEqual(tally.no_change, 1)
        self.assertEqual(tally.action_accepted, 0)
        self.assertEqual(tally.level_failure, 0)


if __name__ == "__main__":
    unittest.main()
