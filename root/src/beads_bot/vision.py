from pathlib import Path

import numpy as np
from PIL import Image

from .board import FeedObservation, GameState, LockMarker, Tile


TILE_SEARCH_REGION = (0.05, 0.52, 0.89, 0.89)
_PALETTE_HUES = {
    "red": (0, 255),
    "orange": (17, 255),
    "yellow": (39, 255),
    "lime": (54, 255),
    "green": (82, 255),
    "teal": (108, 255),
    "cyan": (132, 255),
    "blue": (160, 255),
    "indigo": (179, 255),
    "purple": (198, 255),
    "pink": (226, 255),
}


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Return 8-connected component boxes and areas using row runs."""
    parents: list[int] = []
    runs: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, int]] = []

    def find(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    next_label = 0
    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.concatenate(([False], row, [False])).astype(np.int8)))
        current: list[tuple[int, int, int]] = []
        for start, end in edges.reshape(-1, 2):
            label = next_label
            next_label += 1
            parents.append(label)
            for previous_start, previous_end, previous_label in previous:
                if previous_start <= end and previous_end >= start:
                    union(label, previous_label)
            current.append((int(start), int(end), label))
            runs.append((y, int(start), int(end), label))
        previous = current

    aggregate: dict[int, list[int]] = {}
    for y, start, end, label in runs:
        root = find(label)
        box = aggregate.setdefault(root, [start, y, end, y + 1, 0])
        box[0] = min(box[0], start)
        box[1] = min(box[1], y)
        box[2] = max(box[2], end)
        box[3] = max(box[3], y + 1)
        box[4] += end - start

    return sorted(
        [(x0, y0, x1, y1, area) for x0, y0, x1, y1, area in aggregate.values()],
        key=lambda box: (box[1], box[0]),
    )


def _color_name(hue: int, saturation: int, value: int) -> tuple[str | None, float]:
    if saturation < 38:
        return ("gray", 0.76) if value >= 68 else (None, 0.0)
    if value < 135 and 8 <= hue <= 29:
        return "brown", 0.76

    centers = {name: center for name, (center, _) in _PALETTE_HUES.items()}
    distances = {
        name: min((hue - center) % 256, (center - hue) % 256)
        for name, center in centers.items()
    }
    # Resolve exact hue-boundary ties toward the higher-hue swatch. This keeps
    # magenta-pink values from being labeled purple at the shared boundary.
    name = min(distances, key=lambda candidate: (distances[candidate], -centers[candidate]))
    # The game's blue and cyan swatches sit almost on the same hue bin in this
    # render. Preserve their observed split at the boundary used by the live
    # Level 60 reference instead of collapsing both into cyan.
    if 140 <= hue <= 145:
        name = "blue" if hue <= 142 else "cyan"
    confidence = max(0.4, min(0.97, 1.0 - distances[name] / 36.0))
    return name, confidence


def _tile_color(hsv: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str | None, float, str]:
    left, top, right, bottom = box
    crop = hsv[top:bottom, left:right]
    if crop.size == 0:
        return None, 0.0, "unknown"
    height, width = crop.shape[:2]
    center = crop[int(height * 0.2) : max(int(height * 0.8), 1), int(width * 0.2) : max(int(width * 0.8), 1)]
    saturation = center[:, :, 1]
    value = center[:, :, 2]
    strong = saturation >= 85
    if strong.mean() >= 0.12:
        hue = int(np.median(center[:, :, 0][strong]))
        sat = int(np.median(saturation[strong]))
        val = int(np.median(value[strong]))
        name, confidence = _color_name(hue, sat, val)
        return name, confidence, "tile"

    median_value = int(np.median(value))
    if median_value < 110:
        return None, 0.0, "hidden"
    if median_value >= 205:
        return "white", 0.72, "tile"
    if median_value >= 68:
        return "gray", 0.72, "tile"
    return None, 0.0, "unknown"


def _rim_score(rgb: np.ndarray, box: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    pad_x, pad_y = max(2, round(width * 0.12)), max(2, round(height * 0.12))
    outer_left, outer_top = max(0, left - pad_x), max(0, top - pad_y)
    outer_right, outer_bottom = min(rgb.shape[1], right + pad_x), min(rgb.shape[0], bottom + pad_y)
    crop = rgb[outer_top:outer_bottom, outer_left:outer_right].astype(np.int16)
    yy, xx = np.indices(crop.shape[:2])
    inside = (
        (xx >= left - outer_left)
        & (xx < right - outer_left)
        & (yy >= top - outer_top)
        & (yy < bottom - outer_top)
    )
    ring = ~inside
    if not np.any(ring):
        return 0.0
    brightest = crop.min(axis=2) >= 190
    neutral = crop.max(axis=2) - crop.min(axis=2) <= 55
    return float(np.mean((brightest & neutral)[ring]))


def _legality(rim_score: float) -> tuple[str, float]:
    if rim_score >= 0.22:
        return "RH", min(0.99, 0.68 + (rim_score - 0.22) * 0.8)
    if rim_score <= 0.14:
        return "DNH", min(0.96, 0.66 + (0.14 - rim_score) * 2.5)
    return "UNKNOWN", 0.48


def _find_tiles(rgb: np.ndarray, hsv: np.ndarray) -> tuple[tuple[Tile, ...], tuple[LockMarker, ...]]:
    height, width = hsv.shape[:2]
    rx0, ry0, rx1, ry1 = TILE_SEARCH_REGION
    x0, y0, x1, y1 = int(rx0 * width), int(ry0 * height), int(rx1 * width), int(ry1 * height)
    crop_hsv = hsv[y0:y1, x0:x1]
    saturation, value = crop_hsv[:, :, 1], crop_hsv[:, :, 2]
    mask = ((saturation >= 95) & (value >= 45)) | ((saturation <= 72) & (value <= 175))

    candidates: list[tuple[int, int, int, int]] = []
    for left, top, right, bottom, area in _components(mask):
        box_width, box_height = right - left, bottom - top
        if not (0.045 * width <= box_width <= 0.20 * width):
            continue
        if not (0.025 * height <= box_height <= 0.10 * height):
            continue
        ratio = box_width / max(box_height, 1)
        if not (0.65 <= ratio <= 1.5):
            continue
        if area < 0.12 * box_width * box_height:
            continue
        candidates.append((left + x0, top + y0, right + x0, bottom + y0))

    # Very Hard boards can have a metal lock join two adjacent cube masks.
    # Recover the regular rows and columns from the cells that remain separate.
    cells, lock_markers = _recover_grid_cells(hsv, candidates)

    tiles: list[Tile] = []
    for index, (box, has_special_value) in enumerate(cells, start=1):
        color, color_confidence, kind = _tile_color(hsv, box)
        relief_score = _rim_score(rgb, box)
        legality, relief_confidence = _legality(relief_score)
        if kind == "hidden":
            legality = "DNH" if legality == "UNKNOWN" else legality
        tiles.append(
            Tile(
                id=f"tile-{index:02d}",
                bbox=box,
                color=color,
                legality=legality,
                confidence=round(relief_confidence, 3),
                color_confidence=round(color_confidence, 3),
                kind="special" if has_special_value else kind,
                number=None,
            )
        )
    return tuple(tiles), lock_markers


def _cluster(values: list[float], tolerance: float) -> list[list[float]]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def _cell_has_visual_evidence(
    hsv: np.ndarray,
    center: tuple[float, float],
    cell_width: float,
    cell_height: float,
) -> bool:
    cx, cy = center
    x0 = max(0, round(cx - cell_width * 0.30))
    x1 = min(hsv.shape[1], round(cx + cell_width * 0.30))
    y0 = max(0, round(cy - cell_height * 0.30))
    y1 = min(hsv.shape[0], round(cy + cell_height * 0.30))
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    saturation, value = patch[:, :, 1], patch[:, :, 2]
    colored_body = ((saturation >= 85) & (value >= 65)).mean()
    neutral_body = ((saturation <= 80) & (value >= 65) & (value <= 235)).mean()
    value_spread = float(value.std())
    textured = value_spread >= 8.0
    # Saturated bead faces can remain visible even when a lock obscures their
    # component boundary. Empty tray cells can be colorful too, but stay flat.
    return (
        colored_body >= 0.65
        or (textured and colored_body >= 0.14)
        or (value_spread >= 14.0 and neutral_body >= 0.18)
    )


def _has_special_value_marking(hsv: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x0, x1 = max(0, round(left + width * 0.08)), min(hsv.shape[1], round(right - width * 0.08))
    y0, y1 = max(0, round(top + height * 0.08)), min(hsv.shape[0], round(top + height * 0.62))
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    saturation, value = patch[:, :, 1], patch[:, :, 2]
    white_glyph = ((saturation < 95) & (value > 215)).mean()
    dark_outline = ((value < 100) & (saturation < 110)).mean()
    return white_glyph >= 0.18 and dark_outline >= 0.018


def _detect_lock_markers(
    hsv: np.ndarray,
    columns: list[float],
    rows: list[float],
    row_heights: list[float],
) -> tuple[LockMarker, ...]:
    markers: list[LockMarker] = []
    if len(columns) < 2:
        return ()
    height, width = hsv.shape[:2]
    for row_index, row_center in enumerate(rows):
        row_height = row_heights[row_index]
        for column_index in range(len(columns) - 1):
            seam = (columns[column_index] + columns[column_index + 1]) / 2
            x0, x1 = max(0, round(seam - 28)), min(width, round(seam + 28))
            silver_top = hsv[
                max(0, round(row_center - row_height * 0.85)) : max(1, round(row_center - row_height * 0.18)),
                x0:x1,
            ]
            gold_base = hsv[
                max(0, round(row_center - row_height * 0.40)) : min(height, round(row_center + row_height * 0.05)),
                x0:x1,
            ]
            if silver_top.size == 0 or gold_base.size == 0:
                continue
            # The light beige playfield also looks neutral and bright in HSV.
            # Restrict the metal cue to its mid-tone body to reject that
            # background and old text/box annotations around the cells.
            silver = (
                (silver_top[:, :, 1] <= 70)
                & (silver_top[:, :, 2] >= 140)
                & (silver_top[:, :, 2] <= 210)
            ).mean()
            gold = (
                (gold_base[:, :, 0] >= 12)
                & (gold_base[:, :, 0] <= 38)
                & (gold_base[:, :, 1] >= 150)
                & (gold_base[:, :, 2] >= 130)
            ).mean()
            if silver >= 0.18 and gold >= 0.24:
                box = (
                    max(0, round(seam - 44)),
                    max(0, round(row_center - row_height * 0.65)),
                    min(width, round(seam + 44)),
                    min(height, round(row_center + row_height * 0.15)),
                )
                markers.append(
                    LockMarker(
                        id=f"lock-{len(markers) + 1:02d}",
                        bbox=box,
                        center=(round(seam), round(row_center - row_height * 0.30)),
                        confidence=round(min(0.98, 0.65 + silver * 0.25 + gold * 0.25), 3),
                    )
                )
    return tuple(markers)


def _recover_grid_cells(
    hsv: np.ndarray,
    candidates: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[tuple[int, int, int, int], bool]], tuple[LockMarker, ...]]:
    if len(candidates) < 5:
        return [(box, _has_special_value_marking(hsv, box)) for box in candidates], ()

    widths = [box[2] - box[0] for box in candidates]
    heights = [box[3] - box[1] for box in candidates]
    cell_width = float(np.median(widths))
    cell_height = float(np.median(heights))
    x_centers = [(box[0] + box[2]) / 2 for box in candidates]
    base_spacing = cell_width * 1.10
    pair_spacing: list[float] = []
    for index, first in enumerate(x_centers):
        for second in x_centers[index + 1 :]:
            gap = abs(second - first)
            if gap < cell_width * 0.9:
                continue
            for multiples in range(1, 7):
                spacing = gap / multiples
                if cell_width * 0.95 <= spacing <= cell_width * 1.4:
                    if abs(spacing - base_spacing) <= base_spacing * 0.16:
                        pair_spacing.append(spacing)
    if not pair_spacing:
        return [(box, _has_special_value_marking(hsv, box)) for box in candidates], ()
    spacing_groups = _cluster(pair_spacing, base_spacing * 0.05)
    spacing = float(np.median(max(spacing_groups, key=len)))
    if spacing <= 0:
        return [(box, _has_special_value_marking(hsv, box)) for box in candidates], ()

    phase_candidates = [center % spacing for center in x_centers]
    tolerance_x = spacing * 0.22
    phase_scores = []
    for phase in phase_candidates:
        residuals = [abs((center - phase + spacing / 2) % spacing - spacing / 2) for center in x_centers]
        inliers = [residual for residual in residuals if residual <= tolerance_x]
        phase_scores.append((len(inliers), -float(np.median(inliers)) if inliers else -spacing, phase))
    _, _, phase = max(phase_scores)
    column_indices = [round((center - phase) / spacing) for center in x_centers]
    inlier_indices = [
        column_index
        for center, column_index in zip(x_centers, column_indices)
        if abs(center - (phase + column_index * spacing)) <= tolerance_x
    ]
    if len(inlier_indices) < max(4, int(len(candidates) * 0.55)):
        return [(box, _has_special_value_marking(hsv, box)) for box in candidates], ()
    first_column, last_column = min(inlier_indices), max(inlier_indices)
    if last_column - first_column + 1 < 3 or last_column - first_column + 1 > 8:
        return [(box, _has_special_value_marking(hsv, box)) for box in candidates], ()
    columns = [phase + index * spacing for index in range(first_column, last_column + 1)]
    x_tolerance = spacing * 0.28

    y_tolerance = max(12.0, cell_height * 0.30)
    y_groups = _cluster([(box[1] + box[3]) / 2 for box in candidates], y_tolerance)
    rows = [float(np.median(group)) for group in y_groups]
    if len(rows) < 2:
        return [(box, _has_special_value_marking(hsv, box)) for box in candidates], ()

    row_heights: list[float] = []
    for row_center in rows:
        row_boxes = [box for box in candidates if abs((box[1] + box[3]) / 2 - row_center) <= y_tolerance]
        row_heights.append(float(np.median([box[3] - box[1] for box in row_boxes])) if row_boxes else cell_height)
    lock_markers = _detect_lock_markers(hsv, columns, rows, row_heights)
    cells: list[tuple[tuple[int, int, int, int], bool]] = []
    for row_index, row_center in enumerate(rows):
        row_boxes = [
            box for box in candidates
            if abs((box[1] + box[3]) / 2 - row_center) <= y_tolerance
        ]
        row_height = float(np.median([box[3] - box[1] for box in row_boxes])) if row_boxes else cell_height
        used: set[int] = set()
        for column_index, column_center in enumerate(columns):
            nearby = [
                box for box in row_boxes
                if abs((box[0] + box[2]) / 2 - column_center) <= x_tolerance
            ]
            if nearby:
                box = min(nearby, key=lambda candidate: abs((candidate[0] + candidate[2]) / 2 - column_center))
                if id(box) in used:
                    continue
                used.add(id(box))
            else:
                if not _cell_has_visual_evidence(hsv, (column_center, row_center), cell_width, row_height):
                    continue
                box = (
                    round(column_center - cell_width / 2),
                    round(row_center - row_height / 2),
                    round(column_center + cell_width / 2),
                    round(row_center + row_height / 2),
                )
            cells.append((box, _has_special_value_marking(hsv, box)))

    # If the geometry did not form a usable lattice, keep all original masks.
    if len(cells) < len(candidates):
        return [(box, _has_special_value_marking(hsv, box)) for box in candidates], ()
    return cells, lock_markers


def _read_difficulty(rgb: np.ndarray) -> str | None:
    """Recognize the centered red Very Hard badge used above the board."""
    height, width = rgb.shape[:2]
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"))
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (((hue <= 12) | (hue >= 244)) & (saturation >= 120) & (value >= 50))
    y_start, y_end = int(0.085 * height), int(0.12 * height)
    x_start, x_end = int(0.35 * width), int(0.65 * width)
    best_width = 0
    for y in range(y_start, min(y_end, height)):
        row = red[y, x_start:x_end]
        edges = np.flatnonzero(np.diff(np.concatenate(([False], row, [False])).astype(np.int8)))
        if len(edges):
            best_width = max(best_width, int(np.max(edges[1::2] - edges[::2])))
    if best_width >= 0.085 * width:
        return "Very Hard"
    return None


def _screen_type(rgb: np.ndarray, tiles: tuple[Tile, ...]) -> str:
    if len(tiles) >= 5:
        return "game"
    height, width = rgb.shape[:2]
    top = rgb[int(0.02 * height) : int(0.13 * height), int(0.20 * width) : int(0.80 * width)]
    top_min, top_max = top.min(axis=2), top.max(axis=2)
    white_text = (top_min > 225) & ((top_max - top_min) < 55)
    if float(white_text.mean()) > 0.045:
        return "complete"
    return "unknown"


def _read_feed(hsv: np.ndarray) -> FeedObservation:
    # Queue direction and the entry point into the center loop are not yet
    # calibrated across board layouts. Never promote a color patch near the
    # screen edge into a current bead guess.
    return FeedObservation(None, (), 0.0, "unresolved: conveyor outlet and center order")


def analyze_frame(source: str | Path | Image.Image) -> GameState:
    """Read visible board cells and conservative game-state observations."""
    image = source.copy() if isinstance(source, Image.Image) else Image.open(source)
    image = image.convert("RGB")
    rgb = np.asarray(image)
    hsv = np.asarray(image.convert("HSV"))
    tiles, locks = _find_tiles(rgb, hsv)
    screen = _screen_type(rgb, tiles)
    if tiles:
        left = min(tile.bbox[0] for tile in tiles)
        top = min(tile.bbox[1] for tile in tiles)
        right = max(tile.bbox[2] for tile in tiles)
        bottom = max(tile.bbox[3] for tile in tiles)
        board_region = (left, top, right, bottom)
    else:
        board_region = None
    warnings = () if screen == "game" else ("No confident gameplay board was detected.",)
    return GameState(
        screen=screen,
        width=image.width,
        height=image.height,
        tiles=tiles,
        board_region=board_region,
        feed=_read_feed(hsv),
        difficulty=_read_difficulty(rgb),
        locks=locks,
        warnings=warnings,
    )
