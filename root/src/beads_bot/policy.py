from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from .board import EvidenceContext, EvidenceTally, GameState, Tile
from .matching import materially_changed_geometry, same_physical_tile, tile_semantics


EvidenceStatus = Literal["KNOWN", "LIKELY", "UNCERTAIN", "UNKNOWN"]

_STATUS_SUPPORT = {"KNOWN": 1.0, "LIKELY": 0.72, "UNCERTAIN": 0.35, "UNKNOWN": 0.20}
_REQUIREMENT_WEIGHTS = {
    "tile_legality": 0.32,
    "tile_color": 0.18,
    "visible_current_state": 0.06,
    "feed_state": 0.18,
    "special_rule": 0.12,
    "lock_rule": 0.10,
    "other_mechanics": 0.04,
}
_POSITION_KEY_BUCKETS = 25


@dataclass(frozen=True)
class RequirementEvidence:
    name: str
    status: EvidenceStatus
    confidence: float
    source: str


@dataclass(frozen=True)
class ActionCandidate:
    key: str
    context: EvidenceContext
    tile: Tile
    score: float
    confidence: float
    requirements: tuple[RequirementEvidence, ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    tile: Tile | None
    reason: str
    confidence: float
    candidates: tuple[ActionCandidate, ...] = ()


def _overlaps(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3]


def _lock_relation(state: GameState, tile: Tile) -> tuple[bool, float, str]:
    overlapping = [lock for lock in state.locks if _overlaps(tile.bbox, lock.bbox)]
    if tile.locked:
        return True, tile.confidence, "tile lock flag"
    if overlapping:
        return True, max(lock.confidence for lock in overlapping), "overlapping lock marker"
    return False, 1.0, "no lock overlap detected"


def _status(name: str, status: EvidenceStatus, confidence: float, source: str) -> RequirementEvidence:
    return RequirementEvidence(name, status, round(max(0.0, min(1.0, confidence)), 3), source)


def _requirements(state: GameState, tile: Tile) -> tuple[RequirementEvidence, ...]:
    if tile.legality == "RH":
        legality_status: EvidenceStatus = "KNOWN" if tile.confidence >= 0.65 else "LIKELY"
    else:
        legality_status = "UNCERTAIN"

    if tile.color is None:
        color_status: EvidenceStatus = "UNKNOWN"
    elif tile.color_confidence >= 0.75:
        color_status = "KNOWN"
    elif tile.color_confidence >= 0.5:
        color_status = "LIKELY"
    else:
        color_status = "UNCERTAIN"

    upcoming_offset = next(
        (index for index, color in enumerate(state.feed.upcoming) if color == tile.color and color is not None),
        None,
    )
    if state.feed.current is None and tile.color is not None and upcoming_offset is not None:
        direction_known = state.feed.direction_confidence >= 0.65
        feed_status = "LIKELY" if direction_known else "UNCERTAIN"
        feed_confidence = max(
            0.25,
            state.feed.confidence
            * (state.feed.direction_confidence if direction_known else 0.45)
            * (0.90 ** (upcoming_offset + 1)),
        )
        direction_text = "temporal order confirmed" if direction_known else "approach order unconfirmed"
        feed_source = f"{state.feed.source}; {direction_text}; visible upcoming color at offset {upcoming_offset + 1}"
    elif state.feed.current is None:
        feed_status: EvidenceStatus = "UNKNOWN"
        feed_confidence = 0.2
        feed_source = state.feed.source or "current feed color is unknown"
    elif state.feed.confidence < 0.5:
        feed_status = "UNCERTAIN"
        feed_confidence = max(0.35, state.feed.confidence)
        feed_source = state.feed.source
    elif tile.color is None:
        feed_status = "UNCERTAIN"
        feed_confidence = 0.35
        feed_source = f"{state.feed.source}; tile color is unknown"
    elif tile.color == state.feed.current:
        feed_status = "KNOWN" if state.feed.confidence >= 0.75 else "LIKELY"
        feed_confidence = state.feed.confidence
        feed_source = f"{state.feed.source}; color matches"
    elif tile.color is not None and upcoming_offset is not None:
        direction_known = state.feed.direction_confidence >= 0.65
        feed_status = "LIKELY" if direction_known else "UNCERTAIN"
        feed_confidence = max(
            0.25,
            state.feed.confidence
            * (state.feed.direction_confidence if direction_known else 0.45)
            * (0.90 ** (upcoming_offset + 1)),
        )
        direction_text = "temporal order confirmed" if direction_known else "approach order unconfirmed"
        feed_source = f"{state.feed.source}; {direction_text}; visible upcoming color at offset {upcoming_offset + 1}"
    else:
        feed_status = "UNCERTAIN"
        feed_confidence = 0.35
        feed_source = f"{state.feed.source}; tile/feed colors differ"

    if tile.kind != "special":
        special_status: EvidenceStatus = "KNOWN"
        special_confidence = 1.0
        special_source = "ordinary tile"
    elif tile.number is not None:
        special_status = "LIKELY"
        special_confidence = 0.75
        special_source = "special value was recognized"
    else:
        special_status = "UNCERTAIN"
        special_confidence = 0.4
        special_source = "special value is unknown"

    locked, lock_confidence, lock_source = _lock_relation(state, tile)
    lock_status: EvidenceStatus = "UNCERTAIN" if locked else "KNOWN"
    lock_confidence = min(lock_confidence, 0.55) if locked else 1.0

    if state.screen == "game":
        screen_status: EvidenceStatus = "KNOWN"
        screen_confidence = 1.0
    else:
        screen_status = "UNKNOWN"
        screen_confidence = 0.0

    other_status: EvidenceStatus = "KNOWN" if tile.kind in ("tile", "special") else "UNCERTAIN"
    other_source = "no additional tile mechanic observed" if other_status == "KNOWN" else f"tile kind is {tile.kind}"
    return (
        _status("tile_legality", legality_status, tile.confidence, f"relief classifier: {tile.legality}"),
        _status("tile_color", color_status, tile.color_confidence, f"tile color: {tile.color or 'unknown'}"),
        _status("visible_current_state", screen_status, screen_confidence, f"screen: {state.screen}"),
        _status("feed_state", feed_status, feed_confidence, feed_source),
        _status("special_rule", special_status, special_confidence, special_source),
        _status("lock_rule", lock_status, lock_confidence, lock_source),
        _status("other_mechanics", other_status, 1.0 if other_status == "KNOWN" else 0.35, other_source),
    )


def _neighbor_features(state: GameState, tile: Tile) -> list[dict[str, object]]:
    cx, cy = tile.center
    radius = max(tile.bbox[2] - tile.bbox[0], tile.bbox[3] - tile.bbox[1]) * 1.8
    neighbors = []
    for neighbor in state.tiles:
        if neighbor.id == tile.id:
            continue
        nx, ny = neighbor.center
        distance = ((nx - cx) ** 2 + (ny - cy) ** 2) ** 0.5
        if distance <= radius:
            locked, _, _ = _lock_relation(state, neighbor)
            neighbors.append((distance, {
                "color": neighbor.color,
                "kind": neighbor.kind,
                "legality": neighbor.legality,
                "locked": locked,
            }))
    neighbors.sort(key=lambda item: item[0])
    return [item[1] for item in neighbors]


def _context(state: GameState, tile: Tile) -> EvidenceContext:
    left, top, right, bottom = tile.bbox
    board = state.board_region or (0, 0, state.width, state.height)
    bx0, by0, bx1, by1 = board
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
    cx, cy = tile.center
    locked, _, _ = _lock_relation(state, tile)
    if tile.color == state.feed.current and state.feed.current is not None:
        feed_match = "current"
        feed_match_offset = 0
    else:
        upcoming_offset = next(
            (index for index, color in enumerate(state.feed.upcoming) if color == tile.color and color is not None),
            None,
        )
        feed_match = "imminent" if upcoming_offset is not None else "unmatched"
        feed_match_offset = upcoming_offset + 1 if upcoming_offset is not None else None
    return EvidenceContext(
        mechanic_id="tile-selection",
        features={
            "tile_color": tile.color,
            "tile_kind": tile.kind,
            "tile_legality": tile.legality,
            "tile_locked": tile.locked,
            "lock_overlap": locked,
            "feed_current": state.feed.current,
            "feed_upcoming": state.feed.upcoming,
            "feed_direction": state.feed.direction,
            "feed_direction_confidence": state.feed.direction_confidence,
            "feed_class": "known" if state.feed.current is not None else "unknown",
            "feed_match": feed_match,
            "feed_match_offset": feed_match_offset,
            "local_geometry": {
                "x": round((cx - bx0) / bw, 3),
                "y": round((cy - by0) / bh, 3),
                "aspect_ratio": round((right - left) / max(1, bottom - top), 3),
            },
            "neighbors": _neighbor_features(state, tile),
        },
    )


def _evidence_adjustment(tally: EvidenceTally) -> float:
    # Accepted taps inform interaction confidence. Explicit objective and level
    # outcomes carry progressively stronger strategic evidence.
    accepted = min(0.06, max(0, tally.action_accepted) * 0.015)
    unchanged = min(0.04, max(0, tally.no_change) * 0.01)
    local_progress = min(0.18, max(0, tally.local_progress) * 0.06)
    level_success = min(0.30, max(0, tally.level_success) * 0.15)
    level_failure = min(0.30, max(0, tally.level_failure) * 0.15)
    return accepted - unchanged + local_progress + level_success - level_failure


def _candidate(state: GameState, tile: Tile, evidence_for: Callable[[EvidenceContext], EvidenceTally] | None) -> ActionCandidate:
    requirements = _requirements(state, tile)
    context = _context(state, tile)
    support = sum(_REQUIREMENT_WEIGHTS[item.name] * _STATUS_SUPPORT[item.status] for item in requirements)
    tally = evidence_for(context) if evidence_for is not None else EvidenceTally()
    feed_match_bonus = 0.0
    if tile.color is not None and tile.color == state.feed.current:
        feed_match_bonus = 0.055 * min(1.0, state.feed.confidence)
    elif tile.color is not None:
        upcoming_offset = next(
            (index for index, color in enumerate(state.feed.upcoming) if color == tile.color and color is not None),
            None,
        )
        if upcoming_offset is not None:
            direction_support = state.feed.direction_confidence if state.feed.direction_confidence >= 0.65 else 0.40
            feed_match_bonus = (
                0.045
                * min(1.0, state.feed.confidence)
                * direction_support
                / (upcoming_offset + 1)
            )
    score = max(0.0, min(1.0, support + _evidence_adjustment(tally) + feed_match_bonus))
    confidence = sum(item.confidence for item in requirements) / len(requirements)
    assumptions = tuple(
        f"{item.name}={item.status}: {item.source}"
        for item in requirements
        if item.status in ("UNCERTAIN", "UNKNOWN")
    )
    position_key = (
        round(tile.center[0] / state.width * _POSITION_KEY_BUCKETS),
        round(tile.center[1] / state.height * _POSITION_KEY_BUCKETS),
    )
    return ActionCandidate(
        key=f"tile@{position_key[0]:02d},{position_key[1]:02d}",
        context=context,
        tile=tile,
        score=round(score, 4),
        confidence=round(confidence, 3),
        requirements=requirements,
        assumptions=assumptions,
    )


def _matches_attempt(
    candidate: ActionCandidate,
    prior: ActionCandidate | str,
    size: tuple[int, int],
) -> bool:
    if isinstance(prior, str):
        return candidate.key == prior
    return (
        same_physical_tile(prior.tile, candidate.tile, size, size)
        and not materially_changed_geometry(prior.tile, candidate.tile, size, size)
        and tile_semantics(prior.tile) == tile_semantics(candidate.tile)
    )


def rank_candidates(
    state: GameState,
    *,
    attempted: Iterable[ActionCandidate | str] = (),
    evidence_for: Callable[[EvidenceContext], EvidenceTally] | None = None,
) -> tuple[ActionCandidate, ...]:
    """Return plausible tile actions, ordered by local evidence and context."""
    if state.screen != "game":
        return ()
    candidates = [
        _candidate(state, tile, evidence_for)
        for tile in state.tiles
        if tile.legality != "DNH"
    ]
    attempted_items = tuple(attempted)
    candidates = [
        candidate for candidate in candidates
        if not any(_matches_attempt(candidate, prior, (state.width, state.height)) for prior in attempted_items)
    ]
    candidates.sort(key=lambda candidate: (
        candidate.score,
        candidate.confidence,
        -candidate.tile.center[1],
        -candidate.tile.center[0],
    ), reverse=True)
    return tuple(candidates)


def prune_attempted(state: GameState, attempted: list[ActionCandidate]) -> None:
    """Forget no-change taps once their physical tile disappears or changes state."""
    size = (state.width, state.height)
    attempted[:] = [
        prior for prior in attempted
        if any(
            same_physical_tile(prior.tile, tile, size, size)
            and not materially_changed_geometry(prior.tile, tile, size, size)
            and tile_semantics(prior.tile) == tile_semantics(tile)
            for tile in state.tiles
        )
    ]


def choose_move(
    state: GameState,
    *,
    attempted: Iterable[ActionCandidate | str] = (),
    evidence_for: Callable[[EvidenceContext], EvidenceTally] | None = None,
) -> Decision:
    candidates = rank_candidates(state, attempted=attempted, evidence_for=evidence_for)
    if not candidates:
        reason = f"screen is {state.screen}; no plausible tile action" if state.screen != "game" else "no plausible untried tile action"
        return Decision(None, reason, 0.0, candidates)
    best = candidates[0]
    return Decision(
        best.tile,
        f"highest-ranked plausible action; score {best.score:.3f}; assumptions {len(best.assumptions)}",
        best.confidence,
        candidates,
    )
