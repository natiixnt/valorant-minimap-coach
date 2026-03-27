"""
Per-map surface material zones for Valorant.

Surface type affects footstep sound character:
  metal    — bright, ring-y, spectral centroid > 1100 Hz
  concrete — solid thud, 650-1100 Hz
  wood     — warm, 350-650 Hz
  carpet   — muffled, < 350 Hz

Zone coordinates are normalized (0-1) matching the callouts.py coordinate system.
Format: list of (x_min, x_max, y_min, y_max, surface_type)
First match wins (specific sub-areas before broad zones).

These are approximate community-observed values. Tune via config if needed.
"""
from typing import List, Optional, Tuple

SurfaceZone = Tuple[float, float, float, float, str]

_SURFACES: dict[str, List[SurfaceZone]] = {

    "ascent": [
        # B site - metal catwalks / industrial
        (0.05, 0.30, 0.05, 0.35, "metal"),
        # Mid market - wood/stone
        (0.30, 0.70, 0.30, 0.60, "wood"),
        # A site - stone/concrete courtyard
        (0.65, 0.95, 0.05, 0.50, "concrete"),
        # Attacker spawn - stone
        (0.30, 0.70, 0.70, 0.95, "concrete"),
        # Default
        (0.0, 1.0, 0.0, 1.0, "concrete"),
    ],

    "bind": [
        # B site showers - wet tile
        (0.05, 0.35, 0.05, 0.45, "concrete"),
        # B garage / hookah - concrete
        (0.35, 0.65, 0.05, 0.40, "concrete"),
        # A short / cave - stone
        (0.05, 0.35, 0.55, 0.95, "stone"),
        # A site lamps - carpet/wood
        (0.60, 0.95, 0.55, 0.95, "wood"),
        # Teleporters - metal grating
        (0.45, 0.60, 0.40, 0.60, "metal"),
        (0.0, 1.0, 0.0, 1.0, "concrete"),
    ],

    "haven": [
        # A site - stone/carpet
        (0.65, 0.95, 0.05, 0.40, "concrete"),
        # B site - stone
        (0.35, 0.65, 0.05, 0.50, "concrete"),
        # C site - stone
        (0.05, 0.35, 0.05, 0.45, "concrete"),
        # Mid courtyard - stone/wood
        (0.25, 0.75, 0.45, 0.70, "wood"),
        (0.0, 1.0, 0.0, 1.0, "concrete"),
    ],

    "split": [
        # A site - concrete/wood
        (0.60, 0.95, 0.05, 0.55, "concrete"),
        # B site - concrete
        (0.05, 0.40, 0.05, 0.55, "concrete"),
        # Mid vent - metal
        (0.38, 0.62, 0.20, 0.45, "metal"),
        # Ramps - concrete
        (0.35, 0.65, 0.55, 0.95, "concrete"),
        (0.0, 1.0, 0.0, 1.0, "concrete"),
    ],

    "icebox": [
        # B tube - metal
        (0.05, 0.30, 0.20, 0.45, "metal"),
        # B orange/kitchen - metal/concrete
        (0.05, 0.30, 0.05, 0.20, "concrete"),
        # A belt - metal conveyor
        (0.65, 0.95, 0.30, 0.55, "metal"),
        # Mid boiler - metal
        (0.35, 0.65, 0.35, 0.60, "metal"),
        # Snow areas - muffled (treat as carpet)
        (0.05, 0.95, 0.60, 0.95, "carpet"),
        (0.0, 1.0, 0.0, 1.0, "concrete"),
    ],

    "breeze": [
        # B site - wood planks
        (0.05, 0.35, 0.10, 0.50, "wood"),
        # A site - stone/wood
        (0.65, 0.95, 0.10, 0.55, "wood"),
        # Mid - outdoor stone
        (0.30, 0.70, 0.30, 0.70, "concrete"),
        # Caves - stone
        (0.05, 0.40, 0.55, 0.90, "concrete"),
        (0.0, 1.0, 0.0, 1.0, "wood"),
    ],

    "fracture": [
        # A site - wood/concrete
        (0.60, 0.95, 0.05, 0.50, "wood"),
        # B site - metal/industrial
        (0.05, 0.40, 0.05, 0.50, "metal"),
        # Mid - concrete/stone
        (0.35, 0.65, 0.30, 0.70, "concrete"),
        # Generator area - metal
        (0.10, 0.35, 0.55, 0.85, "metal"),
        (0.0, 1.0, 0.0, 1.0, "concrete"),
    ],

    "pearl": [
        # A site - stone/tile
        (0.60, 0.95, 0.05, 0.50, "concrete"),
        # B site - stone
        (0.05, 0.40, 0.05, 0.50, "concrete"),
        # Mid top/shops - tile
        (0.30, 0.70, 0.20, 0.55, "concrete"),
        # Underground B - stone
        (0.05, 0.40, 0.55, 0.90, "concrete"),
        (0.0, 1.0, 0.0, 1.0, "concrete"),
    ],

    "lotus": [
        # A site - stone/ancient
        (0.60, 0.95, 0.10, 0.55, "concrete"),
        # B site - stone
        (0.35, 0.65, 0.05, 0.45, "concrete"),
        # C site - stone/water
        (0.05, 0.35, 0.10, 0.55, "concrete"),
        # Rotating doors - metal
        (0.30, 0.70, 0.40, 0.65, "metal"),
        (0.0, 1.0, 0.0, 1.0, "concrete"),
    ],

    "sunset": [
        # A site - tile/concrete
        (0.60, 0.95, 0.10, 0.55, "concrete"),
        # B site - indoor carpet
        (0.05, 0.40, 0.10, 0.55, "carpet"),
        # Mid courtyard - tile
        (0.30, 0.70, 0.35, 0.65, "concrete"),
        (0.0, 1.0, 0.0, 1.0, "concrete"),
    ],

    "abyss": [
        # A site - metal grating/stone
        (0.60, 0.95, 0.05, 0.55, "metal"),
        # B site - metal
        (0.05, 0.40, 0.05, 0.55, "metal"),
        # Mid - metal catwalk
        (0.30, 0.70, 0.20, 0.60, "metal"),
        # No railings - treat open voids as metal
        (0.0, 1.0, 0.0, 1.0, "metal"),
    ],
}


def get_surface(x: float, y: float, map_name: str) -> str:
    """
    Return surface material at normalized position (x, y) on the given map.
    Falls back to "concrete" if map is unknown.
    """
    zones: List[SurfaceZone] = _SURFACES.get(map_name.lower(), [])
    for x_min, x_max, y_min, y_max, surface in zones:
        if x_min <= x < x_max and y_min <= y < y_max:
            return surface
    return "concrete"


def surface_matches(detected_surface: str, map_surface: str) -> bool:
    """
    Check if a detected footstep surface is consistent with the map zone.
    Allows one level of tolerance (e.g. concrete <-> wood are close).
    """
    if detected_surface == map_surface:
        return True
    # Close enough pairs
    compatible = {
        ("concrete", "wood"), ("wood", "concrete"),
        ("metal", "concrete"), ("concrete", "metal"),
    }
    return (detected_surface, map_surface) in compatible


def surface_to_voice(surface: str) -> str:
    """Human-readable surface description for callouts."""
    return {
        "metal": "metal",
        "wood": "wooden",
        "concrete": "stone",
        "carpet": "carpet",
    }.get(surface, surface)
