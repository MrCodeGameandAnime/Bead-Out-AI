"""Incremental run journaling and outcome-specific mechanic evidence."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from PIL import Image

from .board import (
    EvidenceContext,
    EvidenceTally,
    GameState,
    InteractionOutcome,
    StrategicOutcome,
)
from .policy import ActionCandidate


def _value(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _state_data(state: GameState | None) -> dict | None:
    if state is None:
        return None
    data = _value(asdict(state))
    for tile_data, tile in zip(data["tiles"], state.tiles):
        tile_data["center"] = list(tile.center)
    return data


def _context_data(context: EvidenceContext) -> dict:
    return {"mechanic_id": context.mechanic_id, "features": _value(context.features)}


def _candidate_data(candidate: ActionCandidate) -> dict:
    return {
        "key": candidate.key,
        "context": _context_data(candidate.context),
        "tile": {**_value(asdict(candidate.tile)), "center": list(candidate.tile.center)},
        "score": candidate.score,
        "confidence": candidate.confidence,
        "requirements": _value(candidate.requirements),
        "assumptions": list(candidate.assumptions),
    }


def _outcome(value) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _signature(state: GameState) -> tuple:
    return (
        state.screen,
        state.feed.current,
        tuple(
            (tile.bbox, tile.color, tile.legality, tile.kind, tile.locked, tile.number)
            for tile in state.tiles
        ),
        tuple((lock.bbox, lock.center) for lock in state.locks),
    )


def classify_transition(
    before_state: GameState,
    after_state: GameState | None,
    chosen: ActionCandidate | None,
) -> tuple[InteractionOutcome, StrategicOutcome]:
    """Separate action acceptance from level progress for one observed transition."""
    if after_state is None:
        return InteractionOutcome.UNKNOWN, StrategicOutcome.UNKNOWN

    if after_state.screen == "unknown":
        interaction = InteractionOutcome.UNKNOWN
    elif chosen is None:
        interaction = InteractionOutcome.NOT_ATTEMPTED
    elif _signature(before_state) != _signature(after_state):
        interaction = InteractionOutcome.ACTION_ACCEPTED
    else:
        interaction = InteractionOutcome.NO_CHANGE

    if after_state.screen == "complete":
        strategic = StrategicOutcome.LEVEL_SUCCESS
    elif after_state.screen == "failure":
        strategic = StrategicOutcome.LEVEL_FAILURE
    elif (
        before_state.progress is not None
        and after_state.progress is not None
        and after_state.progress > before_state.progress
    ):
        strategic = StrategicOutcome.LOCAL_PROGRESS
    elif after_state.screen == "game":
        strategic = StrategicOutcome.IN_PROGRESS
    else:
        strategic = StrategicOutcome.UNKNOWN
    return interaction, strategic


def _flatten_features(features: Mapping[str, object], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in features.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_features(value, path))
        else:
            flattened[path] = json.dumps(_value(value), sort_keys=True, separators=(",", ":"))
    return flattened


def _contexts_comparable(record: Mapping[str, object], context: EvidenceContext) -> bool:
    if record.get("mechanic_id") != context.mechanic_id:
        return False
    recorded_features = record.get("features")
    if not isinstance(recorded_features, Mapping):
        return False
    recorded = _flatten_features(recorded_features)
    current = _flatten_features(context.features)
    shared = recorded.keys() & current.keys()
    if not shared:
        return not recorded and not current
    return all(recorded[key] == current[key] for key in shared)


class RunRecorder:
    def __init__(self, run_dir: Path, evidence_path: Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.evidence_path = Path(evidence_path)
        self.run_id = run_id
        self.frames_dir = self.run_dir / "frames"
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []
        self._finished = False

    @classmethod
    def create(cls, base_dir: Path, evidence_path: Path) -> "RunRecorder":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        run_dir = Path(base_dir) / run_id
        return cls(run_dir, Path(evidence_path), run_id)

    def _append_jsonl(self, path: Path, item: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(_value(item), ensure_ascii=False, sort_keys=True) + "\n")

    def _save_frame(self, frame: Image.Image | None, step_id: str, side: str) -> str | None:
        if frame is None:
            return None
        relative = Path("frames") / f"{step_id}-{side}.jpg"
        frame.convert("RGB").save(self.run_dir / relative, format="JPEG", quality=92)
        return relative.as_posix()

    def record_step(
        self,
        *,
        before_state: GameState,
        before_frame: Image.Image | None,
        candidates: tuple[ActionCandidate, ...],
        chosen: ActionCandidate | None,
        tap: tuple[int, int] | None,
        after_state: GameState | None,
        after_frame: Image.Image | None,
        interaction_outcome: InteractionOutcome | str,
        strategic_outcome: StrategicOutcome | str,
        timings_ms: Mapping[str, float],
        uncertain_assumptions: tuple[str, ...] = (),
    ) -> str:
        if self._finished:
            raise RuntimeError("cannot append to a finalized run")
        step_id = f"step-{len(self._events) + 1:06d}"
        before_path = self._save_frame(before_frame, step_id, "before")
        after_path = self._save_frame(after_frame, step_id, "after")
        event = {
            "schema_version": 1,
            "run_id": self.run_id,
            "step_id": step_id,
            "before_state": _state_data(before_state),
            "candidates": [_candidate_data(candidate) for candidate in candidates],
            "chosen": None if chosen is None else _candidate_data(chosen),
            "tap": None if tap is None else list(tap),
            "after_state": _state_data(after_state),
            "outcomes": {
                "interaction": _outcome(interaction_outcome),
                "strategic": _outcome(strategic_outcome),
            },
            "timings_ms": _value(timings_ms),
            "uncertain_assumptions": list(uncertain_assumptions or (chosen.assumptions if chosen else ())),
            "frames": {"before": before_path, "after": after_path},
        }
        self._append_jsonl(self.events_path, event)
        self._events.append(event)

        if chosen is not None:
            interaction = _outcome(interaction_outcome)
            strategic = _outcome(strategic_outcome)
            if interaction in (InteractionOutcome.ACTION_ACCEPTED.value, InteractionOutcome.NO_CHANGE.value) or strategic == StrategicOutcome.LOCAL_PROGRESS.value:
                self._append_jsonl(self.evidence_path, {
                    "schema_version": 1,
                    "record_type": "action",
                    **_context_data(chosen.context),
                    "interaction_outcome": interaction,
                    "strategic_outcome": strategic if strategic == StrategicOutcome.LOCAL_PROGRESS.value else None,
                    "run_id": self.run_id,
                    "step_id": step_id,
                })
        return step_id

    def evidence_for(self, context: EvidenceContext) -> EvidenceTally:
        counts = {
            "action_accepted": 0,
            "no_change": 0,
            "local_progress": 0,
            "level_success": 0,
            "level_failure": 0,
        }
        if not self.evidence_path.exists():
            return EvidenceTally()
        with self.evidence_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("record_type") != "action" or not _contexts_comparable(record, context):
                    continue
                interaction = record.get("interaction_outcome")
                strategic = record.get("strategic_outcome")
                if interaction == InteractionOutcome.ACTION_ACCEPTED.value:
                    counts["action_accepted"] += 1
                elif interaction == InteractionOutcome.NO_CHANGE.value:
                    counts["no_change"] += 1
                if strategic == StrategicOutcome.LOCAL_PROGRESS.value:
                    counts["local_progress"] += 1
        return EvidenceTally(**counts)

    def finish(self, status: str, reason: str, *, recent_steps: int = 8) -> Path:
        if self._finished:
            return self.manifest_path
        strategic_outcome = {
            "success": StrategicOutcome.LEVEL_SUCCESS.value,
            "failure": StrategicOutcome.LEVEL_FAILURE.value,
        }.get(status)
        if strategic_outcome:
            sequence = [
                {
                    "step_id": event["step_id"],
                    "candidate_key": event["chosen"]["key"],
                    "context": event["chosen"]["context"],
                    "score": event["chosen"]["score"],
                    "confidence": event["chosen"]["confidence"],
                    "interaction_outcome": event["outcomes"]["interaction"],
                }
                for event in self._events
                if event["chosen"] is not None
            ]
            self._append_jsonl(self.evidence_path, {
                "schema_version": 1,
                "record_type": "episode",
                "mechanic_id": "action-sequence",
                "features": {"action_count": len(sequence)},
                "strategic_outcome": strategic_outcome,
                "sequence_contexts": [item["context"] for item in sequence],
                "sequence": sequence,
                "run_id": self.run_id,
            })

        recent = self._events[-max(0, recent_steps):] if recent_steps else []
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": status,
            "reason": reason,
            "possible_failure_example": status in ("failure", "unrecognized_ui"),
            "step_count": len(self._events),
            "events_path": self.events_path.name,
            "recent_steps": recent,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._finished = True
        return self.manifest_path
