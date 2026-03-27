from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Zone:
    name: str
    x_range: Tuple[float, float]  # normalized 0-1
    y_range: Tuple[float, float]


# Minimap coordinate zones per map (normalized, origin = top-left of minimap image).
# These are approximate and may need tweaking per resolution/aspect ratio.
MAP_ZONES: Dict[str, List[Zone]] = {
    "ascent": [
        Zone("A site",   (0.60, 1.00), (0.00, 0.40)),
        Zone("B site",   (0.00, 0.40), (0.00, 0.40)),
        Zone("mid",      (0.35, 0.65), (0.30, 0.70)),
        Zone("A main",   (0.70, 1.00), (0.40, 0.70)),
        Zone("B main",   (0.00, 0.30), (0.40, 0.70)),
        Zone("CT spawn", (0.30, 0.70), (0.00, 0.30)),
        Zone("T spawn",  (0.30, 0.70), (0.70, 1.00)),
    ],
    "bind": [
        Zone("A site",  (0.60, 1.00), (0.00, 0.50)),
        Zone("B site",  (0.00, 0.40), (0.50, 1.00)),
        Zone("A short", (0.40, 0.70), (0.00, 0.40)),
        Zone("B long",  (0.30, 0.60), (0.60, 1.00)),
        Zone("hookah",  (0.00, 0.30), (0.30, 0.60)),
        Zone("T spawn", (0.30, 0.70), (0.40, 0.70)),
    ],
    "haven": [
        Zone("A site",  (0.70, 1.00), (0.20, 0.50)),
        Zone("B site",  (0.40, 0.60), (0.10, 0.40)),
        Zone("C site",  (0.00, 0.30), (0.20, 0.50)),
        Zone("mid",     (0.30, 0.70), (0.40, 0.70)),
        Zone("T spawn", (0.30, 0.70), (0.70, 1.00)),
    ],
    "split": [
        Zone("A site",  (0.60, 1.00), (0.10, 0.50)),
        Zone("B site",  (0.00, 0.40), (0.10, 0.50)),
        Zone("mid",     (0.35, 0.65), (0.30, 0.70)),
        Zone("T spawn", (0.30, 0.70), (0.70, 1.00)),
    ],
    "icebox": [
        Zone("A site",  (0.60, 1.00), (0.10, 0.50)),
        Zone("B site",  (0.00, 0.40), (0.10, 0.50)),
        Zone("mid",     (0.35, 0.65), (0.30, 0.70)),
        Zone("T spawn", (0.30, 0.70), (0.70, 1.00)),
    ],
    "lotus": [
        Zone("A site",  (0.65, 1.00), (0.10, 0.50)),
        Zone("B site",  (0.35, 0.65), (0.10, 0.45)),
        Zone("C site",  (0.00, 0.35), (0.10, 0.50)),
        Zone("mid",     (0.30, 0.70), (0.40, 0.70)),
        Zone("T spawn", (0.30, 0.70), (0.70, 1.00)),
    ],
}

_GRID = [
    ["top-left",    "top-center",    "top-right"],
    ["mid-left",    "center",        "mid-right"],
    ["bottom-left", "bottom-center", "bottom-right"],
]


def pos_to_zone(x: float, y: float, map_name: str) -> str:
    for zone in MAP_ZONES.get(map_name, []):
        if zone.x_range[0] <= x <= zone.x_range[1] and zone.y_range[0] <= y <= zone.y_range[1]:
            return zone.name
    col = min(int(x * 3), 2)
    row = min(int(y * 3), 2)
    return _GRID[row][col]


def enemies_to_callout(positions: List[Tuple[float, float]], map_name: str) -> str:
    if not positions:
        return ""
    zones = list(dict.fromkeys(pos_to_zone(x, y, map_name) for x, y in positions))
    count = len(positions)
    if count == 1:
        return f"One enemy at {zones[0]}"
    if count == 2:
        return f"Two enemies at {zones[0]}" if len(zones) == 1 else f"Two enemies, {zones[0]} and {zones[1]}"
    return f"{count} enemies spotted, {', '.join(zones[:2])}"
