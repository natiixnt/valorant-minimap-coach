"""
Enemy zone occupancy tracker with transition detection.

Tracks which map zone each visible enemy occupies frame-to-frame.
Announces when an enemy transitions FROM one zone INTO another -- giving
the defender information about enemy movement direction before they arrive.

Example callouts:
  "Enemy leaving B Long, entering B Site"
  "2 enemies moving from Mid to A Link"

Algorithm:
  - Assign each frame's enemy list to the previous frame's enemies by
    nearest-neighbour matching (max 0.15 normalized units).
  - Maintain a zone-per-slot dict; announce on zone change.
  - Require _CONFIRM_FRAMES consecutive frames in new zone before announcing
    (avoids flickering zone boundaries triggering spurious callouts).
  - Per-slot cooldown prevents repeating the same transition.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from src.maps.callouts import pos_to_zone

_MAX_MATCH_DIST = 0.15   # normalized units; beyond this = new enemy, not moved
_CONFIRM_FRAMES = 2      # frames enemy must be in new zone before announcing
_SLOT_COOLDOWN  = 6.0    # seconds before same slot can announce again


class ZoneTracker:
    def __init__(self) -> None:
        # slot_id -> {zone, pending_zone, pending_count, last_announce_t, pos}
        self._slots: Dict[int, dict] = {}
        self._next_slot = 0

    def update(
        self,
        enemies: List[Tuple[float, float]],
        map_name: str,
    ) -> List[str]:
        """
        Call once per tick.
        Returns list of transition callout strings (may be empty).
        """
        now = time.monotonic()
        zones = [pos_to_zone(x, y, map_name) for x, y in enemies]

        # Match enemies to existing slots by nearest neighbour
        matched_slots: Dict[int, int] = {}   # slot_id -> enemy_idx
        unmatched_enemies = list(range(len(enemies)))
        unmatched_slots   = list(self._slots.keys())

        for slot_id in list(unmatched_slots):
            slot = self._slots[slot_id]
            prev_pos = slot["pos"]
            best_idx, best_d = -1, 1e9
            for ei in unmatched_enemies:
                ex, ey = enemies[ei]
                d = (ex - prev_pos[0]) ** 2 + (ey - prev_pos[1]) ** 2
                if d < best_d:
                    best_d = d
                    best_idx = ei
            if best_idx >= 0 and best_d < _MAX_MATCH_DIST ** 2:
                matched_slots[slot_id] = best_idx
                unmatched_enemies.remove(best_idx)
                unmatched_slots.remove(slot_id)

        # Unmatched old slots: enemy left vision -- remove after 2s
        for slot_id in list(self._slots.keys()):
            if slot_id not in matched_slots:
                if now - self._slots[slot_id].get("last_seen", now) > 2.0:
                    del self._slots[slot_id]
                else:
                    self._slots[slot_id]["last_seen"] = self._slots[slot_id].get(
                        "last_seen", now
                    )

        # New enemies: create slots
        for ei in unmatched_enemies:
            self._slots[self._next_slot] = {
                "zone": zones[ei],
                "pending_zone": None,
                "pending_count": 0,
                "last_announce_t": 0.0,
                "pos": enemies[ei],
                "last_seen": now,
            }
            self._next_slot += 1

        # Update matched slots, detect transitions
        callouts: List[str] = []
        for slot_id, ei in matched_slots.items():
            slot = self._slots[slot_id]
            new_zone = zones[ei]
            slot["pos"] = enemies[ei]
            slot["last_seen"] = now

            if new_zone == slot["zone"]:
                slot["pending_zone"]  = None
                slot["pending_count"] = 0
                continue

            # New zone -- start or continue pending confirmation
            if new_zone == slot["pending_zone"]:
                slot["pending_count"] += 1
            else:
                slot["pending_zone"]  = new_zone
                slot["pending_count"] = 1

            if slot["pending_count"] >= _CONFIRM_FRAMES:
                cooldown_ok = (now - slot["last_announce_t"]) > _SLOT_COOLDOWN
                if cooldown_ok and slot["zone"] and new_zone != slot["zone"]:
                    old_zone = slot["zone"]
                    callouts.append(
                        f"Enemy moving from {old_zone} to {new_zone}"
                    )
                    slot["last_announce_t"] = now
                slot["zone"]          = new_zone
                slot["pending_zone"]  = None
                slot["pending_count"] = 0

        return callouts

    def reset(self) -> None:
        self._slots.clear()
        self._next_slot = 0
