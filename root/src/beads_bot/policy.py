from dataclasses import dataclass

from .board import GameState, Tile


@dataclass(frozen=True)
class Decision:
    tile: Tile | None
    reason: str
    confidence: float


def choose_move(state: GameState) -> Decision:
    if state.screen != "game":
        return Decision(None, f"screen is {state.screen}; no move", 0.0)
    if state.feed.current is None or state.feed.confidence < 0.65:
        return Decision(None, "current conveyor color is unknown; pause for a fresh observation", 0.0)

    matches = [
        tile
        for tile in state.tiles
        if tile.legality == "RH" and tile.color == state.feed.current and not tile.locked
    ]
    if not matches:
        return Decision(None, f"no visibly raised {state.feed.current} tile is available", 0.0)

    matches.sort(key=lambda tile: (tile.bbox[1], tile.bbox[0]))
    tile = matches[0]
    return Decision(
        tile,
        f"{state.feed.current} is first on {state.feed.source}; this raised tile is selectable",
        round(min(state.feed.confidence, tile.confidence), 3),
    )
