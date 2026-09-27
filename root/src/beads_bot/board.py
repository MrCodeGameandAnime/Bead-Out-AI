from dataclasses import dataclass


@dataclass(frozen=True)
class FeedObservation:
    current: str | None
    upcoming: tuple[str | None, ...]
    confidence: float
    source: str


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
