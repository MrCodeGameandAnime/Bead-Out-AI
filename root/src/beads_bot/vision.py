from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .board import FeedObservation, GameState, LockMarker, ScreenControl, Tile


TILE_SEARCH_REGION = (0.05, 0.52, 0.89, 0.89)
_NAVIGATION_REFERENCES = {
    # The sample rectangles are chosen away from the hand-drawn target marks.
    # Tap points below are the centers of those marked regions, normalized to
    # the corresponding annotated reference image dimensions.
    "home": {
        "reference": "home_screen.png",
        "region": (0.25, 0.725, 0.75, 0.79),
        "sample_size": (96, 24),
        "screen": "home",
        "control": "start_level",
        "target": (219 / 440, 738 / 982),
    },
    "failure": {
        "reference": "failure_03.jpg",
        "region": (0.16, 0.34, 0.75, 0.57),
        "sample_size": (96, 64),
        "screen": "failure",
        "control": "dismiss_failure",
        "target": (599 / 691, 466 / 1536),
    },
}
_NAVIGATION_MATCH_THRESHOLD = 0.86
_OUT_OF_SPACE_CONTROLS = {
    # The annotated images remain documentation for the marked top-right X.
    # The live Level 80 modal uses a bottom-center X, so its target is stored
    # independently from screen recognition and from the raw regression image.
    "annotated_header": ("close", (639 / 691, 268 / 1536)),
    "live_bottom": ("close", (0.5, 2273 / 2400)),
}
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


def _attached_halo_score(rgb: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Measure bright neutral pixels in a narrow band attached to the tile edge.

    Raised cubes have a continuous pale rim just outside their body. Sampling a
    broad neighborhood also captures bright playfield pixels, so keep this band
    close to the detected tile and require near-white, low-chroma pixels.
    """
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    pad_x, pad_y = max(2, round(width * 0.04)), max(2, round(height * 0.04))
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
    brightest = crop.min(axis=2) >= 210
    neutral = crop.max(axis=2) - crop.min(axis=2) <= 25
    return float(np.mean((brightest & neutral)[ring]))


def _legality(attached_halo_score: float) -> tuple[str, float]:
    # The local halo separates raised and depressed cubes across the reference
    # boards; broad outer-ring brightness is not a reliable relief cue.
    if attached_halo_score >= 0.20:
        return "RH", min(0.99, 0.68 + (attached_halo_score - 0.20) * 0.8)
    if attached_halo_score <= 0.08:
        return "DNH", min(0.96, 0.66 + (0.08 - attached_halo_score) * 2.5)
    return "UNKNOWN", 0.48


def _detect_tile_mechanic_overlays(
    hsv_image: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[str, ...]:
    """Detect the diagonal key symbol inside one tile's local image region.

    Key colors vary, so this looks for compact saturated-color components and
    classifies their local silhouette. Restricting the crop to a padded tile
    box avoids attaching nearby tray/UI graphics to the tile. The diagonal
    ring-and-shaft silhouette distinguishes the observed keys from the nearby
    vertically/horizontally oriented padlocks.
    """
    left, top, right, bottom = box
    cell_width, cell_height = right - left, bottom - top
    pad_x, pad_y = max(3, round(cell_width * 0.04)), max(3, round(cell_height * 0.04))
    x0, y0 = max(0, left - pad_x), max(0, top - pad_y)
    x1, y1 = min(hsv_image.shape[1], right + pad_x), min(hsv_image.shape[0], bottom + pad_y)
    if x1 <= x0 or y1 <= y0:
        return ()

    hsv = hsv_image[y0:y1, x0:x1]
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    vivid = (saturation >= 110) & (value >= 90)
    if int(vivid.sum()) < 80:
        return ()

    # Select local hue modes. Eight-unit bins tolerate compression and
    # brightness shifts; circular separation keeps red and orange distinct.
    histogram = np.bincount((hue[vivid] // 8).astype(np.int16), minlength=32)
    hue_modes: list[int] = []
    for bin_index in np.argsort(histogram)[::-1]:
        if histogram[bin_index] < max(24, int(vivid.sum() * 0.012)):
            break
        separated = all(
            min((int(bin_index) - prior) % 32, (prior - int(bin_index)) % 32) > 1
            for prior in hue_modes
        )
        if separated:
            hue_modes.append(int(bin_index))

    tile_area = max(1, cell_width * cell_height)
    for hue_bin in hue_modes:
        mode_hue = hue_bin * 8 + 4
        hue_values = hue.astype(np.int16)
        hue_distance = np.minimum(
            (hue_values - mode_hue) % 256,
            (mode_hue - hue_values) % 256,
        )
        mask = (hue_distance <= 11) & vivid
        for comp_left, comp_top, comp_right, comp_bottom, area in _components(mask):
            comp_width = comp_right - comp_left
            comp_height = comp_bottom - comp_top
            area_ratio = area / tile_area
            width_ratio = comp_width / max(cell_width, 1)
            height_ratio = comp_height / max(cell_height, 1)
            if not (
                0.08 <= area_ratio <= 0.36
                and 0.55 <= width_ratio <= 0.98
                and 0.38 <= height_ratio <= 0.78
                and comp_width / max(comp_height, 1) >= 1.08
            ):
                continue

            component = mask[comp_top:comp_bottom, comp_left:comp_right]
            ys, xs = np.nonzero(component)
            if len(xs) < 40:
                continue
            points = np.column_stack((xs, ys)).astype(np.float64)
            covariance = np.cov(points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            major = int(np.argmax(eigenvalues))
            elongation = float(eigenvalues[major] / max(eigenvalues[1 - major], 1e-6))
            vector = eigenvectors[:, major]
            angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
            if elongation >= 3.0 and 120.0 <= angle <= 178.0:
                return ("key",)
    return ()


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
        relief_score = _attached_halo_score(rgb, box)
        legality, relief_confidence = _legality(relief_score)
        mechanic_overlays = _detect_tile_mechanic_overlays(hsv, box)
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
                mechanic_overlays=mechanic_overlays,
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


def _normalized_region(
    image: Image.Image,
    region: tuple[float, float, float, float],
    sample_size: tuple[int, int],
) -> np.ndarray:
    width, height = image.size
    left, top, right, bottom = region
    crop = image.crop((
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    ))
    return np.asarray(crop.resize(sample_size, Image.Resampling.BILINEAR), dtype=np.int16)


@lru_cache(maxsize=len(_NAVIGATION_REFERENCES))
def _navigation_template(name: str) -> np.ndarray:
    reference = _NAVIGATION_REFERENCES[name]
    root = Path(__file__).resolve().parents[2]
    image = Image.open(root / "img" / reference["reference"]).convert("RGB")
    return _normalized_region(image, reference["region"], reference["sample_size"])


def _out_of_space_variant(rgb: np.ndarray) -> str | None:
    """Recognize the shared Out-of-Space modal from layout-level cues."""
    height, width = rgb.shape[:2]
    header = rgb[
        int(0.10 * height) : int(0.22 * height),
        int(0.08 * width) : int(0.92 * width),
    ]
    channel_max = header.max(axis=2)
    channel_min = header.min(axis=2)
    bright_text = (channel_min > 205) & ((channel_max - channel_min) < 80)
    if float(bright_text.mean()) < 0.05:
        return None

    blue_banner = (
        (header[:, :, 2] > header[:, :, 0] * 1.2)
        & (header[:, :, 2] > header[:, :, 1] * 0.9)
        & (header[:, :, 2] > 110)
    )
    if float(blue_banner.mean()) > 0.35:
        return "annotated_header"

    # The raw live modal dims the underlying board and presents its close X at
    # the bottom center; the annotated target references use a bright blue
    # header and a top-right X. These features distinguish the target layout
    # without requiring their pixels to match the annotations.
    if float(rgb.mean()) / 255.0 < 0.35:
        red_title = (
            (header[:, :, 0] > 150)
            & (header[:, :, 1] < 130)
            & (header[:, :, 2] < 140)
        )
        if float(red_title.mean()) > 0.01:
            return "live_bottom"
    return None


def _out_of_space_substate(rgb: np.ndarray) -> str:
    """Distinguish the offer and life-loss page inside the shared modal state."""
    height, width = rgb.shape[:2]
    body = rgb[
        int(0.30 * height) : int(0.59 * height),
        int(0.30 * width) : int(0.70 * width),
    ].astype(np.int16)
    red_heart = (
        (body[:, :, 0] >= 145)
        & (body[:, :, 1] <= 115)
        & (body[:, :, 2] <= 120)
        & (body[:, :, 0] >= body[:, :, 1] * 1.45)
    )
    return "life_warning" if float(red_heart.mean()) >= 0.10 else "space_offer"


def _navigation_screen(
    image: Image.Image,
    rgb: np.ndarray,
) -> tuple[str, tuple[ScreenControl, ...], str | None] | None:
    """Recognize navigation states, keeping screen cues separate from targets."""
    out_of_space_variant = _out_of_space_variant(rgb)
    if out_of_space_variant is not None:
        control_kind, (target_x, target_y) = _OUT_OF_SPACE_CONTROLS[out_of_space_variant]
        control = ScreenControl(
            control_kind,
            (round(target_x * image.width), round(target_y * image.height)),
            0.92 if out_of_space_variant == "live_bottom" else 0.98,
        )
        return "out_of_space", (control,), _out_of_space_substate(rgb)

    scores: list[tuple[float, str]] = []
    for name, reference in _NAVIGATION_REFERENCES.items():
        observed = _normalized_region(image, reference["region"], reference["sample_size"])
        template = _navigation_template(name)
        similarity = 1.0 - float(np.abs(observed - template).mean()) / 255.0
        scores.append((similarity, name))
    similarity, name = max(scores)
    if similarity < _NAVIGATION_MATCH_THRESHOLD:
        return None

    reference = _NAVIGATION_REFERENCES[name]
    target_x, target_y = reference["target"]
    control = ScreenControl(
        reference["control"],
        (round(target_x * image.width), round(target_y * image.height)),
        round(similarity, 3),
    )
    return reference["screen"], (control,), None


def _find_feed_outlet(hsv: np.ndarray) -> tuple[float, float, float] | None:
    """Locate the central exit gate from its dark violet body and proportions."""
    height, width = hsv.shape[:2]
    x0, x1 = int(0.30 * width), int(0.70 * width)
    y0, y1 = int(0.30 * height), int(0.48 * height)
    crop = hsv[y0:y1, x0:x1]
    mask = (
        (crop[:, :, 0] >= 160)
        & (crop[:, :, 0] <= 235)
        & (crop[:, :, 1] >= 50)
        & (crop[:, :, 2] <= 130)
    )
    candidates = []
    for left, top, right, bottom, area in _components(mask):
        box_width, box_height = right - left, bottom - top
        center_x = x0 + (left + right) / 2
        center_y = y0 + (top + bottom) / 2
        if not (0.055 * width <= box_width <= 0.16 * width):
            continue
        if not (0.018 * height <= box_height <= 0.06 * height):
            continue
        if not (1.0 <= box_width / max(1, box_height) <= 3.2):
            continue
        if not (0.475 * width <= center_x <= 0.525 * width):
            continue
        if not (0.34 * height <= center_y <= 0.44 * height):
            continue
        fill = area / max(1, box_width * box_height)
        candidates.append((fill, -abs(center_x - width / 2), center_x, y0 + top, box_width))
    if not candidates:
        return None
    _, _, _, top, gate_width = max(candidates)
    # The exit is centered on the central loop. Component centroids drift when
    # the purple track joins the gate mask, so use the stable screen center for
    # its horizontal anchor after validating the component's own geometry.
    return width / 2, float(top), float(gate_width)


def _feed_patch_color(
    hsv: np.ndarray,
    center: tuple[float, float],
    radius: int,
) -> tuple[str | None, float]:
    x, y = map(round, center)
    height, width = hsv.shape[:2]
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return None, 0.0

    pixels = patch.reshape(-1, 3)
    hue, saturation, value = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    white = (saturation < 38) & (value >= 195)
    colors: dict[str, int] = {}
    eligible = (saturation >= 50) & (value >= 75)
    for pixel_hue, pixel_saturation, pixel_value in pixels[eligible]:
        color, _ = _color_name(int(pixel_hue), int(pixel_saturation), int(pixel_value))
        if color is not None:
            colors[color] = colors.get(color, 0) + 1

    total = len(pixels)
    white_fraction = float(white.mean())
    if white_fraction >= 0.42 and float(value.std()) >= 10.0:
        return "white", round(min(0.94, 0.62 + white_fraction * 0.35), 3)
    if not colors:
        return None, 0.0
    color, count = max(colors.items(), key=lambda item: item[1])
    colored_fraction = count / max(1, int(eligible.sum()))
    if colored_fraction < 0.62 or count / total < 0.28:
        return None, round(colored_fraction, 3)
    return color, round(min(0.96, 0.60 + colored_fraction * 0.34), 3)


def _has_concealed_feed(hsv: np.ndarray) -> bool:
    """Recognize question-mark conveyor columns without assigning their colors."""
    height, width = hsv.shape[:2]
    top = hsv[int(0.03 * height) : int(0.34 * height)]
    side_columns = np.concatenate((
        top[:, int(0.04 * width) : int(0.22 * width)],
        top[:, int(0.78 * width) : int(0.96 * width)],
    ), axis=1)
    dark_beads = (side_columns[:, :, 1] < 80) & (side_columns[:, :, 2] < 90)
    bright_question_marks = (side_columns[:, :, 1] < 70) & (side_columns[:, :, 2] > 210)
    return float(dark_beads.mean()) >= 0.25 and float(bright_question_marks.mean()) >= 0.02


_FEED_OUTLET_UNSET = object()


def _read_feed(
    hsv: np.ndarray,
    outlet: tuple[float, float, float] | None | object = _FEED_OUTLET_UNSET,
    *,
    outlet_status: str = "direct",
) -> FeedObservation:
    """Read the bead at the visible outlet and a short, visible left approach."""
    if outlet is _FEED_OUTLET_UNSET:
        outlet = _find_feed_outlet(hsv)
        outlet_status = "direct" if outlet is not None else "unknown"
    if outlet is None:
        if _has_concealed_feed(hsv):
            return FeedObservation(
                None,
                (None,),
                0.0,
                "concealed conveyor region; current and upcoming colors unknown",
                upcoming_status="concealed",
                outlet_status="unknown",
            )
        return FeedObservation(
            None, (), 0.0, "unresolved: outlet geometry not found",
            outlet_status="unknown",
        )

    outlet_x, gate_top, gate_width = outlet
    bead_y = gate_top - gate_width * 0.105
    bead_radius = max(4, round(gate_width * 0.045))
    current, confidence = _feed_patch_color(hsv, (outlet_x, bead_y), bead_radius)
    if current is None:
        return FeedObservation(
            None, (), 0.0, "outlet found; bead color unresolved",
            outlet_status=str(outlet_status),
        )

    # The board's visible approach curves up and left from the exit gate.
    # Temporal observations below verify that this is the advancing direction.
    upcoming: list[str | None] = []
    previous = current
    relative_path = (
        (0.0, 0.0),
        (-0.80, -0.35),
        (-1.43, -1.0),
        (-1.55, -2.2),
        (-1.40, -3.2),
        (-0.75, -4.0),
    )
    waypoints = [
        (outlet_x + x * gate_width, bead_y + y * gate_width)
        for x, y in relative_path
    ]
    path_samples: list[tuple[float, float]] = []
    for start, end in zip(waypoints, waypoints[1:]):
        segment_length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        sample_count = max(1, round(segment_length / (gate_width * 0.08)))
        for index in range(sample_count):
            fraction = index / sample_count
            path_samples.append((
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            ))
    path_samples.append(waypoints[-1])
    sample_stride = max(1, round(gate_width * 0.16 / max(gate_width * 0.08, 1)))
    misses = 0
    for point_index in range(0, len(path_samples), sample_stride):
        color, sample_confidence = _feed_patch_color(
            hsv,
            path_samples[point_index],
            max(3, round(gate_width * 0.035)),
        )
        if color is None or sample_confidence < 0.55:
            misses += 1
            if upcoming and misses >= 3:
                upcoming.append(None)
                break
            continue
        misses = 0
        if color != previous:
            upcoming.append(color)
            previous = color
            if len([item for item in upcoming if item is not None]) >= 3:
                upcoming.append(None)
                break
    if not upcoming or upcoming[-1] is not None:
        upcoming.append(None)

    return FeedObservation(
        current,
        tuple(upcoming),
        confidence,
        (
            "visible outlet bead; left-approach lookahead pending temporal confirmation"
            if outlet_status == "direct"
            else "visible outlet bead at tracked outlet geometry; left-approach lookahead pending temporal confirmation"
        ),
        current_status="direct",
        upcoming_status="direct",
        outlet_status=str(outlet_status),
    )


class FeedTracker:
    """Track stable outlet geometry and briefly retain unresolved feed evidence."""

    MAX_CARRIED_FRAMES = 2
    CONFIDENCE_DECAY_PER_FRAME = 0.5

    def __init__(self) -> None:
        self._last_direct: FeedObservation | None = None
        self._dropout_age = 0
        self._outlet_geometry: tuple[float, float, float] | None = None
        self._direction: str | None = None
        self._direction_confidence = 0.0

    def _reset(self) -> None:
        self._last_direct = None
        self._dropout_age = 0
        self._outlet_geometry = None
        self._direction = None
        self._direction_confidence = 0.0

    @staticmethod
    def _geometry_is_consistent(
        candidate: tuple[float, float, float],
        previous: tuple[float, float, float],
    ) -> bool:
        candidate_x, candidate_top, candidate_width = candidate
        previous_x, previous_top, previous_width = previous
        width_ratio = candidate_width / max(previous_width, 1e-6)
        return (
            0.65 <= width_ratio <= 1.45
            and abs(candidate_x - previous_x) <= 0.025
            and abs(candidate_top - previous_top) <= 0.04
        )

    @staticmethod
    def _normalized_geometry(
        outlet: tuple[float, float, float],
        width: int,
        height: int,
    ) -> tuple[float, float, float]:
        center_x, top, gate_width = outlet
        return center_x / width, top / height, gate_width / width

    def _pixel_geometry(self, width: int, height: int) -> tuple[float, float, float] | None:
        if self._outlet_geometry is None:
            return None
        center_x, top, gate_width = self._outlet_geometry
        return center_x * width, top * height, gate_width * width

    def observe(self, source: str | Path | Image.Image, state: GameState) -> GameState:
        if state.screen != "game":
            self._reset()
            return state

        image = source.copy() if isinstance(source, Image.Image) else Image.open(source)
        image = image.convert("RGB")
        if image.width < 400 or image.height < 800:
            return state
        hsv = np.asarray(image.convert("HSV"))
        if _has_concealed_feed(hsv):
            self._reset()
            return replace(
                state,
                feed=FeedObservation(
                    None,
                    (None,),
                    0.0,
                    "concealed conveyor region; current and upcoming colors unknown",
                    upcoming_status="concealed",
                ),
            )

        height, width = hsv.shape[:2]
        outlet = _find_feed_outlet(hsv)
        previous_geometry = self._pixel_geometry(width, height)
        if outlet is not None and previous_geometry is not None:
            normalized_candidate = self._normalized_geometry(outlet, width, height)
            normalized_previous = self._outlet_geometry
            if normalized_previous is not None and not self._geometry_is_consistent(
                normalized_candidate,
                normalized_previous,
            ):
                outlet = None
        if outlet is not None:
            self._outlet_geometry = self._normalized_geometry(outlet, width, height)
            outlet_status = "direct"
        else:
            outlet = previous_geometry
            outlet_status = "tracked" if outlet is not None else "unknown"

        observed = _read_feed(hsv, outlet=outlet, outlet_status=outlet_status)
        if observed.current is not None:
            previous = self._last_direct
            if previous is not None and previous.current != observed.current:
                visible_ahead = [color for color in previous.upcoming if color is not None]
                if visible_ahead and observed.current == visible_ahead[0]:
                    self._direction = "from_left_toward_outlet"
                    self._direction_confidence = 0.86
                elif observed.current in visible_ahead:
                    self._direction = "from_left_toward_outlet"
                    self._direction_confidence = 0.70

            if self._direction is not None:
                observed = replace(
                    observed,
                    confidence=round(max(observed.confidence, self._direction_confidence), 3),
                    source=f"{observed.source}; temporal advancement confirmed from left",
                    direction=self._direction,
                    direction_confidence=self._direction_confidence,
                )
            observed = replace(
                observed,
                current_status="direct",
                upcoming_status="direct",
                outlet_status=outlet_status,
                age_frames=0,
            )
            self._last_direct = observed
            self._dropout_age = 0
            return replace(state, feed=observed)

        if observed.upcoming_status == "concealed":
            self._reset()
            return replace(state, feed=observed)

        self._dropout_age += 1
        if self._last_direct is not None and self._dropout_age <= self.MAX_CARRIED_FRAMES:
            decay = self.CONFIDENCE_DECAY_PER_FRAME ** self._dropout_age
            carried = replace(
                self._last_direct,
                confidence=round(self._last_direct.confidence * decay, 3),
                direction_confidence=round(self._last_direct.direction_confidence * decay, 3),
                source=(
                    f"temporarily carried feed belief (age {self._dropout_age}/"
                    f"{self.MAX_CARRIED_FRAMES} frames): {observed.source}"
                ),
                current_status="tracked",
                upcoming_status="tracked" if self._last_direct.upcoming else "unknown",
                outlet_status="tracked" if outlet is not None else "unknown",
                age_frames=self._dropout_age,
            )
            return replace(state, feed=carried)

        expired = replace(
            observed,
            current=None,
            upcoming=(),
            confidence=0.0,
            source=(
                f"{observed.source}; prior feed belief expired after "
                f"{self.MAX_CARRIED_FRAMES} unresolved frames"
            ),
            direction=self._direction,
            direction_confidence=round(
                self._direction_confidence * self.CONFIDENCE_DECAY_PER_FRAME ** self._dropout_age,
                3,
            ),
            current_status="unknown",
            upcoming_status="unknown",
            outlet_status=outlet_status,
            age_frames=self._dropout_age,
        )
        self._last_direct = None
        return replace(state, feed=expired)


def analyze_frame(source: str | Path | Image.Image) -> GameState:
    """Read visible board cells and conservative game-state observations."""
    image = source.copy() if isinstance(source, Image.Image) else Image.open(source)
    image = image.convert("RGB")
    rgb = np.asarray(image)
    hsv = np.asarray(image.convert("HSV"))
    navigation = _navigation_screen(image, rgb)
    if navigation is None:
        tiles, locks = _find_tiles(rgb, hsv)
        screen = _screen_type(rgb, tiles)
        controls: tuple[ScreenControl, ...] = ()
        modal_substate = None
    else:
        screen, controls, modal_substate = navigation
        tiles, locks = (), ()
    if tiles:
        left = min(tile.bbox[0] for tile in tiles)
        top = min(tile.bbox[1] for tile in tiles)
        right = max(tile.bbox[2] for tile in tiles)
        bottom = max(tile.bbox[3] for tile in tiles)
        board_region = (left, top, right, bottom)
    else:
        board_region = None
    recognized_screens = {"game", "home", "out_of_space", "failure"}
    warnings = () if screen in recognized_screens else ("No confident gameplay board was detected.",)
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
        controls=controls,
        modal_substate=modal_substate,
    )
