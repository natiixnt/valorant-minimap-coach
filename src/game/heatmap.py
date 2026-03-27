"""
Multi-round enemy danger zone heatmap.

Accumulates enemy sightings across rounds with exponential time decay.
Each map zone has a danger score 0-1. Zones sighted recently in this round
score higher than zones sighted two rounds ago.

Usage:
    hm = Heatmap()
    hm.add_sighting(zone="B Long", round_num=5)
    score = hm.score("B Long")          # 0-1
    hottest = hm.hottest_zones(n=3)     # [("B Long", 0.9), ("Mid", 0.6), ...]
    hm.end_round(6)                     # decay old data

The heatmap feeds into:
  - play_detector.py  (pattern confirmation across rounds)
  - Overlay visualization (hot zones shown in orange/red)
  - TTS: "B has been hot -- enemy showed up there last 3 rounds"
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Tuple


# Decay factor per round boundary (score multiplied by this at end-of-round)
_DECAY_PER_ROUND = 0.55

# Max rounds to retain history
_MAX_ROUNDS = 10

# Within-round: a sighting adds this base score
_SIGHTING_BASE = 0.25
_SIGHTING_MAX  = 1.0     # capped per zone per round


class Heatmap:
    def __init__(self) -> None:
        # zone -> round_num -> accumulated score
        self._data: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._current_round = 1
        self._round_start_ts: Dict[int, float] = {1: time.monotonic()}

    # ------------------------------------------------------------------

    def add_sighting(self, zone: str, round_num: int) -> None:
        """Record one enemy sighting in this zone during round_num."""
        current = self._data[zone][round_num]
        self._data[zone][round_num] = min(_SIGHTING_MAX, current + _SIGHTING_BASE)

    def end_round(self, next_round: int) -> None:
        """
        Called when a round ends. Advances the current round counter
        and prunes history older than _MAX_ROUNDS.
        """
        self._current_round = next_round
        self._round_start_ts[next_round] = time.monotonic()

        # Prune old rounds
        cutoff = next_round - _MAX_ROUNDS
        for zone in list(self._data):
            self._data[zone] = {
                r: v for r, v in self._data[zone].items() if r > cutoff
            }

    def score(self, zone: str) -> float:
        """
        Return danger score 0-1 for the given zone.
        Recent rounds weighted higher via exponential decay.
        """
        rounds = self._data.get(zone)
        if not rounds:
            return 0.0
        total = 0.0
        for r, v in rounds.items():
            age = max(0, self._current_round - r)   # 0 = this round; guard future rounds
            total += v * (_DECAY_PER_ROUND ** age)
        return float(min(1.0, total))

    def hottest_zones(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return the n zones with highest danger score, descending."""
        scores = [(z, self.score(z)) for z in self._data]
        scores = [(z, s) for z, s in scores if s > 0.05]
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:n]

    def zone_is_hot(self, zone: str, threshold: float = 0.4) -> bool:
        return self.score(zone) >= threshold

    def summary(self) -> str:
        """Human-readable top-3 danger zones for TTS or overlay."""
        top = self.hottest_zones(3)
        if not top:
            return "No danger zones established yet."
        parts = [f"{z} ({int(s * 100)}%)" for z, s in top]
        return "Hot zones: " + ", ".join(parts)

    def reset(self) -> None:
        """Full reset at match start."""
        self._data.clear()
        self._current_round = 1
        self._round_start_ts = {1: time.monotonic()}
