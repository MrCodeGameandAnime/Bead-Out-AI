from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .board import GameState
from .policy import Decision


_STATUS_COLORS = {
    "RH": (38, 205, 95, 255),
    "DNH": (235, 76, 65, 255),
    "UNKNOWN": (255, 186, 45, 255),
}


def save_debug_image(
    source: str | Path | Image.Image,
    state: GameState,
    destination: str | Path,
    decision: Decision | None = None,
) -> Path:
    """Save a frame with detected cells, mechanics, feed, and move overlay."""
    image = source.copy() if isinstance(source, Image.Image) else Image.open(source)
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    selected_id = decision.tile.id if decision and decision.tile else None

    for marker in state.locks:
        center_x, center_y = marker.center
        left, top, right, bottom = center_x - 38, center_y - 45, center_x + 38, center_y + 55
        draw.rectangle((left, top, right, bottom), outline=(255, 190, 45, 255), width=5)

    for tile in state.tiles:
        color = (175, 80, 230, 255) if tile.locked else _STATUS_COLORS.get(tile.legality, _STATUS_COLORS["UNKNOWN"])
        left, top, right, bottom = tile.bbox
        border_width = 5 if tile.id == selected_id else 3
        short_id = tile.id.rsplit("-", 1)[-1]
        text = f"{short_id} VALUE UNKNOWN" if tile.kind == "special" and tile.number is None else (
            f"{short_id} {tile.color or tile.kind} {tile.legality}"
        )
        label_top = max(0, top - 15)
        # Repaint the old label strip and its few pixels of overlap into the
        # tile so older detector labels do not compete with this regression.
        draw.rectangle((left, label_top, right, min(bottom, top + 18)), fill=(18, 24, 37, 235))
        draw.text((left + 2, label_top + 2), text, fill=(255, 255, 255, 255), font=font)
        draw.rectangle((left, top, right, bottom), outline=color, width=border_width)

    feed = state.feed.current if state.feed.current is not None else "unknown"
    summary = f"{state.screen} | feed 1: {feed} ({state.feed.confidence:.2f}) | {state.feed.source}"
    if state.difficulty:
        summary += f" | {state.difficulty}"
    draw.rectangle((8, 8, min(image.width - 8, 760), 30), fill=(18, 24, 37, 220))
    draw.text((14, 14), summary, fill=(255, 255, 255, 255), font=font)
    if decision is not None:
        decision_text = decision.reason if decision.tile is None else (
            f"SELECT {decision.tile.color} at {decision.tile.center}; confidence {decision.confidence:.2f}"
        )
        draw.rectangle((8, 34, min(image.width - 8, 680), 56), fill=(18, 24, 37, 220))
        draw.text((14, 40), decision_text[:100], fill=(255, 255, 255, 255), font=font)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(destination)
    return destination
