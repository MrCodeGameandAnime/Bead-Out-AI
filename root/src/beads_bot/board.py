from dataclasses import dataclass
from enum import Enum
from typing import Mapping


@dataclass(frozen=True)
class FeedObservation:
    current: str | None
    upcoming: tuple[str | None, ...]
    confidence: float
    source: str
    direction: str | None = None
    direction_confidence: float = 0.0


class InteractionOutcome(str, Enum):
    ACTION_ACCEPTED = "ACTION_ACCEPTED"
    NO_CHANGE = "NO_CHANGE"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    UNKNOWN = "UNKNOWN"


class StrategicOutcome(str, Enum):
    LOCAL_PROGRESS = "LOCAL_PROGRESS"
    LEVEL_SUCCESS = "LEVEL_SUCCESS"
    LEVEL_FAILURE = "LEVEL_FAILURE"
    IN_PROGRESS = "IN_PROGRESS"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ScreenControl:
    kind: str
    center: tuple[int, int]
    confidence: float
    cost: int | None = None
    requires_ad: bool = False


@dataclass(frozen=True)
class EvidenceContext:
    mechanic_id: str
    features: Mapping[str, object]


@dataclass(frozen=True)
class EvidenceTally:
    action_accepted: int = 0
    no_change: int = 0
    local_progress: int = 0
    level_success: int = 0
    level_failure: int = 0


@dataclass(frozen=True)
class Tile:
    id: str
    bbox: tuple[int, int, int, int]
    color: str | None
    legality: str
    confidence: float
    color_confidence: float
    kind: str = "tile"
    occupancy: int | None = None
    capacity: int | None = None
    locked: bool = False
    number: int | None = None

    @property
    def center(self) -> tuple[int, int]:
        left, top, right, bottom = self.bbox
        return ((left + right) // 2, (top + bottom) // 2)


@dataclass(frozen=True)
class LockMarker:
    id: str
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float


@dataclass(frozen=True)
class GameState:
    screen: str
    width: int
    height: int
    tiles: tuple[Tile, ...]
    board_region: tuple[int, int, int, int] | None
    feed: FeedObservation
    difficulty: str | None = None
    locks: tuple[LockMarker, ...] = ()
    warnings: tuple[str, ...] = ()
    progress: float | None = None
    controls: tuple[ScreenControl, ...] = ()
    modal_substate: str | None = None
