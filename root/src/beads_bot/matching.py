"""Spatial matching helpers for tile observations across frames."""

from math import hypot

from .board import LockMarker, Tile


def normalized_bbox(tile: Tile, width: int, height: int) -> tuple[float, float, float, float]:
    left, top, right, bottom = tile.bbox
    return left / width, top / height, right / width, bottom / height


def bbox_iou(
    first: Tile,
    second: Tile,
    first_size: tuple[int, int],
    second_size: tuple[int, int],
) -> float:
    first_box = normalized_bbox(first, *first_size)
    second_box = normalized_bbox(second, *second_size)
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first_box[2] - first_box[0]) * max(0.0, first_box[3] - first_box[1])
    second_area = max(0.0, second_box[2] - second_box[0]) * max(0.0, second_box[3] - second_box[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def normalized_center_distance(
    first: Tile,
    second: Tile,
    first_size: tuple[int, int],
    second_size: tuple[int, int],
) -> float:
    first_x, first_y = first.center
    second_x, second_y = second.center
    dx = first_x / first_size[0] - second_x / second_size[0]
    dy = first_y / first_size[1] - second_y / second_size[1]
    return hypot(dx, dy)


def same_physical_tile(
    first: Tile,
    second: Tile,
    first_size: tuple[int, int],
    second_size: tuple[int, int],
) -> bool:
    """Match by normalized location and overlap; generated IDs/colors are ignored."""
    first_box = normalized_bbox(first, *first_size)
    second_box = normalized_bbox(second, *second_size)
    first_diagonal = hypot(first_box[2] - first_box[0], first_box[3] - first_box[1])
    second_diagonal = hypot(second_box[2] - second_box[0], second_box[3] - second_box[1])
    center_tolerance = max(0.003, 0.20 * (first_diagonal + second_diagonal) / 2)
    return (
        bbox_iou(first, second, first_size, second_size) >= 0.50
        or normalized_center_distance(first, second, first_size, second_size) <= center_tolerance
    )


def tile_semantics(tile: Tile) -> tuple[object, ...]:
    return (
        tile.color,
        tile.legality,
        tile.kind,
        tile.locked,
        tile.number,
        tile.occupancy,
        tile.capacity,
    )


def tile_interaction_state(tile: Tile) -> tuple[object, ...]:
    """Return chosen-tile fields that currently prove an interaction."""
    return (
        tile.legality,
        tile.kind,
        tile.locked,
        tile.number,
        tile.occupancy,
        tile.capacity,
    )


def lock_overlay_present(tile: Tile, locks: tuple[LockMarker, ...]) -> bool:
    """Report whether a lock marker overlaps the tile's physical cell."""
    left, top, right, bottom = tile.bbox
    margin_x = max(1, (right - left) * 0.12)
    margin_y = max(1, (bottom - top) * 0.12)
    return any(
        lock.bbox[0] < right + margin_x
        and left - margin_x < lock.bbox[2]
        and lock.bbox[1] < bottom + margin_y
        and top - margin_y < lock.bbox[3]
        for lock in locks
    )


def in_local_neighborhood(
    anchor: Tile,
    tile: Tile,
    anchor_size: tuple[int, int],
    tile_size: tuple[int, int],
) -> bool:
    """Limit secondary attribution to immediately adjacent grid cells."""
    anchor_box = normalized_bbox(anchor, *anchor_size)
    tile_box = normalized_bbox(tile, *tile_size)
    anchor_width = anchor_box[2] - anchor_box[0]
    anchor_height = anchor_box[3] - anchor_box[1]
    tile_width = tile_box[2] - tile_box[0]
    tile_height = tile_box[3] - tile_box[1]
    dx = abs(anchor.center[0] / anchor_size[0] - tile.center[0] / tile_size[0])
    dy = abs(anchor.center[1] / anchor_size[1] - tile.center[1] / tile_size[1])
    return (
        dx <= 1.6 * max(anchor_width, tile_width)
        and dy <= 1.6 * max(anchor_height, tile_height)
    )


def local_neighborhood_changed(
    chosen: Tile,
    before_tiles: tuple[Tile, ...],
    after_tiles: tuple[Tile, ...],
    before_size: tuple[int, int],
    after_size: tuple[int, int],
) -> bool:
    before_neighbors = [
        tile for tile in before_tiles
        if not same_physical_tile(chosen, tile, before_size, before_size)
        and in_local_neighborhood(chosen, tile, before_size, before_size)
    ]
    after_neighbors = [
        tile for tile in after_tiles
        if not same_physical_tile(chosen, tile, before_size, after_size)
        and in_local_neighborhood(chosen, tile, before_size, after_size)
    ]

    for before_neighbor in before_neighbors:
        matches = [
            tile for tile in after_neighbors
            if same_physical_tile(before_neighbor, tile, before_size, after_size)
        ]
        if not matches:
            return True
        after_neighbor = min(
            matches,
            key=lambda tile: normalized_center_distance(
                before_neighbor,
                tile,
                before_size,
                after_size,
            ),
        )
        if (
            tile_semantics(before_neighbor) != tile_semantics(after_neighbor)
            or materially_changed_geometry(before_neighbor, after_neighbor, before_size, after_size)
        ):
            return True

    return any(
        not any(
            same_physical_tile(before_tile, after_neighbor, before_size, after_size)
            for before_tile in before_neighbors
        )
        for after_neighbor in after_neighbors
    )


def materially_changed_geometry(
    first: Tile,
    second: Tile,
    first_size: tuple[int, int],
    second_size: tuple[int, int],
) -> bool:
    first_box = normalized_bbox(first, *first_size)
    second_box = normalized_bbox(second, *second_size)
    diagonal = (
        hypot(first_box[2] - first_box[0], first_box[3] - first_box[1])
        + hypot(second_box[2] - second_box[0], second_box[3] - second_box[1])
    ) / 2
    return (
        bbox_iou(first, second, first_size, second_size) < 0.70
        or normalized_center_distance(first, second, first_size, second_size) > max(0.003, 0.06 * diagonal)
    )
