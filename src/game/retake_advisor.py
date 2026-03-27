"""
Post-plant retake advisor.

When the spike is planted, analyzes teammate positions relative to the spike
and generates concrete rotation advice: who is closest, what route to take,
how much time they have.

Map travel times are pre-computed from callout zone centroids using approximate
graph distances (not pathfinding -- just straight-line / zone-adjacency estimates).
These are tuned per map and intentionally conservative (add 20% for realistic play).

Output example:
  "Spike at B Long. Rotate: you from CT, teammate from Mid. Hold angle on B Main."

The advisor speaks only once per plant (debounced on spike_pos).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.maps.callouts import pos_to_zone


# Estimated travel time in seconds from zone A to zone B (rough, per map).
# Format: {map_name: {(zone_a, zone_b): seconds}}
# Only zone pairs that matter for retakes are listed; unknown pairs use fallback.
_RETAKE_TIMES: Dict[str, Dict[Tuple[str, str], float]] = {
    "ascent": {
        ("CT Spawn", "A Site"): 7.0,
        ("CT Spawn", "B Site"): 8.0,
        ("Mid Market", "A Site"): 6.5,
        ("Mid Market", "B Site"): 6.5,
        ("B Link", "B Site"): 3.5,
        ("A Link", "A Site"): 3.5,
    },
    "bind": {
        ("CT Spawn", "A Site"): 8.0,
        ("CT Spawn", "B Site"): 7.0,
        ("A Short", "A Site"): 3.5,
        ("B Long", "B Site"): 4.5,
        ("Teleporter", "A Site"): 5.5,
        ("Teleporter", "B Site"): 5.5,
    },
    "haven": {
        ("CT Spawn", "A Site"): 8.0,
        ("CT Spawn", "B Site"): 6.0,
        ("CT Spawn", "C Site"): 8.0,
        ("Mid Courtyard", "B Site"): 4.5,
        ("Mid Courtyard", "A Site"): 6.5,
        ("Mid Courtyard", "C Site"): 6.5,
    },
    "split": {
        ("CT Spawn", "A Site"): 7.0,
        ("CT Spawn", "B Site"): 7.0,
        ("Mid Vent", "A Site"): 4.5,
        ("Mid Vent", "B Site"): 4.5,
    },
    "icebox": {
        ("CT Spawn", "A Site"): 9.0,
        ("CT Spawn", "B Site"): 7.0,
        ("Mid Boiler", "B Site"): 5.5,
        ("Mid Boiler", "A Site"): 6.5,
    },
    "lotus": {
        ("CT Spawn", "A Site"): 7.0,
        ("CT Spawn", "B Site"): 6.0,
        ("CT Spawn", "C Site"): 8.0,
        ("Mid", "A Site"): 5.0,
        ("Mid", "B Site"): 5.0,
        ("Mid", "C Site"): 6.0,
    },
    "sunset": {
        ("CT Spawn", "A Site"): 7.0,
        ("CT Spawn", "B Site"): 8.0,
        ("Mid", "A Site"): 6.0,
        ("Mid", "B Site"): 5.0,
    },
    "abyss": {
        ("CT Spawn", "A Site"): 8.0,
        ("CT Spawn", "B Site"): 7.0,
        ("Mid", "A Site"): 6.0,
        ("Mid", "B Site"): 6.0,
    },
    "pearl": {
        ("CT Spawn", "A Site"): 7.0,
        ("CT Spawn", "B Site"): 8.0,
        ("Mid", "A Site"): 5.0,
        ("Mid", "B Site"): 6.0,
    },
}
_DEFAULT_RETAKE_TIME = 8.0
_POST_PLANT_TIME = 45.0   # seconds from plant to explosion


class RetakeAdvisor:
    def __init__(self) -> None:
        self._last_spike_pos: Optional[Tuple[float, float]] = None

    def advise(
        self,
        spike_pos: Tuple[float, float],
        team_positions: List[Tuple[float, float]],
        map_name: str,
        time_since_plant: float,
    ) -> Optional[str]:
        """
        Generate retake callout. Returns None if no new advice needed (same plant pos).

        spike_pos: normalized (x, y)
        team_positions: list of teammate normalized (x, y)
        map_name: current map
        time_since_plant: seconds since spike was planted
        """
        # Only advise once per plant location
        if self._last_spike_pos == spike_pos:
            return None
        self._last_spike_pos = spike_pos

        spike_zone = pos_to_zone(spike_pos[0], spike_pos[1], map_name)
        time_left = _POST_PLANT_TIME - time_since_plant

        if not team_positions:
            return f"Spike at {spike_zone}! {int(time_left)}s to defuse!"

        # Rank teammates by estimated travel time to spike zone
        ranked = self._rank_teammates(team_positions, spike_zone, map_name)

        if not ranked:
            return f"Spike at {spike_zone}! Rotate now! {int(time_left)}s left!"

        closest_time, closest_zone = ranked[0]

        if closest_time > time_left - 5:
            return (
                f"Spike at {spike_zone}! {int(time_left)}s left -- "
                f"too far, consider abandoning retake."
            )

        parts = []
        for t, zone in ranked[:2]:
            parts.append(f"{zone} ({int(t)}s away)")

        teammates_str = ", ".join(parts)
        return (
            f"Spike at {spike_zone}! Rotate: {teammates_str}. "
            f"{int(time_left)}s to defuse!"
        )

    def reset(self) -> None:
        self._last_spike_pos = None

    # ------------------------------------------------------------------

    def _rank_teammates(
        self,
        positions: List[Tuple[float, float]],
        spike_zone: str,
        map_name: str,
    ) -> List[Tuple[float, str]]:
        """Return list of (estimated_travel_seconds, current_zone) sorted by time."""
        results = []
        times = _RETAKE_TIMES.get(map_name.lower(), {})
        for x, y in positions:
            zone = pos_to_zone(x, y, map_name)
            t = (
                times.get((zone, spike_zone))
                or times.get((spike_zone, zone))
                or _DEFAULT_RETAKE_TIME
            )
            results.append((t, zone))
        return sorted(results, key=lambda r: r[0])
