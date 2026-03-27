"""
Play pattern recognition from enemy minimap positions.

Detects:
  RUSH      -- 3+ enemies tightly clustered, moving fast toward one site.
               "Rush detected! 4 enemies pushing B!"
  EXECUTE   -- 4-5 enemies all converging on one site zone (slower, coordinated).
               "Execute onto A incoming! 4 enemies."
  SPLIT     -- enemies split across both sites simultaneously.
               "Split! 2 on A, 2 on B."
  LURK      -- 1 enemy isolated far from the rest (often flanking).
               "Lurk detected -- check your flank."
  MID_CTRL  -- 2+ enemies in mid zone fighting for map control.
               "Enemies contesting mid."
  SPREAD    -- enemies widely distributed (standard spread / no clear intent).

Each detected play returns a PlayEvent with a voice callout and confidence.

Algorithm:
  - Cluster enemies using simple centroid grouping (no ML needed at 5 players max).
  - Compare cluster positions to site zones from callouts.py.
  - Velocity from consecutive frames used to separate rush (fast) from execute (slow).
  - Confirmation: pattern must hold for _CONFIRM_FRAMES consecutive ticks.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.maps.callouts import pos_to_zone


class PlayType(Enum):
    RUSH    = auto()
    EXECUTE = auto()
    SPLIT   = auto()
    LURK    = auto()
    MID_CTRL = auto()
    SPREAD  = auto()


@dataclass
class PlayEvent:
    play: PlayType
    voice: str
    confidence: float
    zones: List[str]        # zones involved


# Clustering radius: enemies within this normalized distance are in the same cluster
_CLUSTER_RADIUS = 0.18

# How many consecutive frames a pattern must hold before being reported
_CONFIRM_FRAMES = 3

# Minimum speed (normalized units/frame) to classify as a rush
_RUSH_SPEED_THRESHOLD = 0.004

# Site zones: zones that are on site A or B (or C for Haven, Lotus)
# We detect which zones belong to sites by checking zone names
_SITE_PREFIXES = ("A ", "B ", "C ")


class PlayDetector:
    def __init__(self) -> None:
        # ring buffer of last N frames: each is list of (x,y)
        self._history: deque = deque(maxlen=8)
        self._pattern_counts: Dict[PlayType, int] = {}
        self._last_reported: Dict[PlayType, float] = {}
        self._report_cooldown = 12.0   # seconds before same play type re-reported

    def update(
        self,
        enemies: List[Tuple[float, float]],
        map_name: str,
    ) -> Optional[PlayEvent]:
        """
        Call once per tick. Returns a PlayEvent when a pattern is confirmed,
        else None.
        """
        self._history.append(enemies)
        if not enemies or len(self._history) < _CONFIRM_FRAMES:
            return None

        # Cluster current positions
        clusters = _cluster(enemies)
        velocities = self._estimate_velocities()
        zones = [pos_to_zone(x, y, map_name) for x, y in enemies]

        event = self._classify(enemies, clusters, velocities, zones, map_name)
        if event is None:
            return None

        # Confirmation counter
        pt = event.play
        count = self._pattern_counts.get(pt, 0) + 1
        self._pattern_counts[pt] = count

        # Reset other patterns
        for other in list(self._pattern_counts):
            if other != pt:
                self._pattern_counts[other] = 0

        if count < _CONFIRM_FRAMES:
            return None

        # Cooldown
        now = time.monotonic()
        last = self._last_reported.get(pt, 0.0)
        if now - last < self._report_cooldown:
            return None

        self._last_reported[pt] = now
        self._pattern_counts[pt] = 0
        return event

    # ------------------------------------------------------------------

    def _classify(
        self,
        enemies: List[Tuple[float, float]],
        clusters: List[List[Tuple[float, float]]],
        velocities: Dict[int, Tuple[float, float]],
        zones: List[str],
        map_name: str,
    ) -> Optional[PlayEvent]:
        n = len(enemies)
        if n < 2:
            return None

        # --- LURK: exactly 1 enemy far from all others
        if len(clusters) >= 2:
            sizes = sorted([len(c) for c in clusters], reverse=True)
            if sizes[0] >= 2 and sizes[1] == 1:
                lurk_pos = _find_isolated(enemies, clusters)
                lurk_zone = pos_to_zone(lurk_pos[0], lurk_pos[1], map_name) if lurk_pos else "?"
                return PlayEvent(
                    play=PlayType.LURK,
                    voice=f"Lurker at {lurk_zone}! Check your flank.",
                    confidence=0.75,
                    zones=[lurk_zone],
                )

        # --- SPLIT: enemies on 2+ different sites
        site_groups: Dict[str, List[int]] = {}
        for i, z in enumerate(zones):
            for prefix in _SITE_PREFIXES:
                if z.startswith(prefix):
                    site = prefix.strip()
                    site_groups.setdefault(site, []).append(i)
                    break
        if len(site_groups) >= 2:
            parts = [f"{len(v)} on {k}" for k, v in sorted(site_groups.items())]
            voice = "Split attack! " + ", ".join(parts) + "!"
            return PlayEvent(
                play=PlayType.SPLIT,
                voice=voice,
                confidence=0.8,
                zones=list(site_groups.keys()),
            )

        # --- MID CONTROL: 2+ enemies in mid zones
        mid_enemies = [z for z in zones if z.lower().startswith("mid")]
        if len(mid_enemies) >= 2:
            return PlayEvent(
                play=PlayType.MID_CTRL,
                voice=f"{len(mid_enemies)} enemies contesting mid.",
                confidence=0.7,
                zones=list(set(mid_enemies)),
            )

        # --- RUSH / EXECUTE: big cluster on one site
        if clusters and len(clusters[0]) >= 3:
            main_cluster = clusters[0]
            site = _dominant_site(main_cluster, map_name)
            # Check speed
            speeds = [np.sqrt(vx**2 + vy**2)
                      for vx, vy in velocities.values()]
            avg_speed = float(np.mean(speeds)) if speeds else 0.0

            if avg_speed >= _RUSH_SPEED_THRESHOLD:
                voice = f"Rush! {len(main_cluster)} enemies pushing {site}! Fall back!"
                return PlayEvent(
                    play=PlayType.RUSH,
                    voice=voice,
                    confidence=0.85,
                    zones=[site] if site else [],
                )
            elif len(main_cluster) >= 4:
                voice = f"Execute onto {site}! {len(main_cluster)} enemies moving in!"
                return PlayEvent(
                    play=PlayType.EXECUTE,
                    voice=voice,
                    confidence=0.8,
                    zones=[site] if site else [],
                )

        return None

    def _estimate_velocities(self) -> Dict[int, Tuple[float, float]]:
        """Average velocity per enemy index over history frames."""
        if len(self._history) < 2:
            return {}
        prev_frame = list(self._history)[-2]
        curr_frame = list(self._history)[-1]
        if not prev_frame or not curr_frame:
            return {}
        result = {}
        for i, (cx, cy) in enumerate(curr_frame):
            best_j, best_d = -1, 1e9
            for j, (px, py) in enumerate(prev_frame):
                d = (cx - px) ** 2 + (cy - py) ** 2
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d < 0.1 ** 2:
                px, py = prev_frame[best_j]
                result[i] = (cx - px, cy - py)
        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _cluster(
    points: List[Tuple[float, float]],
) -> List[List[Tuple[float, float]]]:
    """Simple greedy clustering by distance threshold."""
    if not points:
        return []
    remaining = list(points)
    clusters = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        new_remaining = []
        for p in remaining:
            d = np.sqrt((p[0] - seed[0]) ** 2 + (p[1] - seed[1]) ** 2)
            if d <= _CLUSTER_RADIUS:
                cluster.append(p)
            else:
                new_remaining.append(p)
        remaining = new_remaining
        clusters.append(cluster)
    return sorted(clusters, key=len, reverse=True)


def _dominant_site(
    cluster: List[Tuple[float, float]], map_name: str
) -> str:
    """Find the most common site prefix among cluster zones."""
    zone_counts: Dict[str, int] = {}
    for x, y in cluster:
        z = pos_to_zone(x, y, map_name)
        for prefix in _SITE_PREFIXES:
            if z.startswith(prefix):
                site = prefix.strip()
                zone_counts[site] = zone_counts.get(site, 0) + 1
                break
    if not zone_counts:
        return ""
    return max(zone_counts, key=zone_counts.get)  # type: ignore[arg-type]


def _find_isolated(
    enemies: List[Tuple[float, float]],
    clusters: List[List[Tuple[float, float]]],
) -> Optional[Tuple[float, float]]:
    """Return the single enemy not in the main cluster."""
    main = set(map(tuple, clusters[0]))
    for e in enemies:
        if tuple(e) not in main:
            return e
    return None
