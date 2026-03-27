"""
Enemy trajectory extrapolation.

Uses the last N position frames to extrapolate where each enemy will be
in 1-2 seconds. Announces if an enemy is predicted to cross into a new zone.

This gives defenders ~1-2 second advance warning of an enemy arrival.

Algorithm:
  - Linear velocity estimated via least-squares fit over last _HISTORY_FRAMES.
  - Extrapolate position by _PREDICT_SEC seconds.
  - If predicted zone differs from current zone, announce predicted arrival.
  - Confidence decays with velocity uncertainty (high scatter = low confidence).

Callout example:
  "Enemy predicted at A Site in 1 second -- velocity from B Long"
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.maps.callouts import pos_to_zone

_HISTORY_FRAMES = 6     # frames of position history per slot
_PREDICT_SEC    = 1.5   # how far ahead to predict (seconds)
_FRAME_SEC      = 0.1   # seconds per frame (coach loop period)
_PREDICT_FRAMES = int(_PREDICT_SEC / _FRAME_SEC)
_MIN_SPEED      = 0.004  # min normalized units/frame to bother predicting
_CONF_THRESHOLD = 0.35   # minimum confidence to announce prediction


class TrajectoryPredictor:
    def __init__(self) -> None:
        # slot_id -> deque of (x, y) positions
        self._histories: Dict[int, deque] = {}
        self._next_slot = 0
        self._announced: Dict[int, str] = {}   # slot_id -> last announced zone

    def update(
        self,
        enemies: List[Tuple[float, float]],
        map_name: str,
    ) -> List[str]:
        """
        Call once per tick with current enemy positions.
        Returns list of prediction callout strings.
        """
        # Match enemies to slots (nearest neighbour, same logic as ZoneTracker)
        matched: Dict[int, int] = {}
        unmatched_e = list(range(len(enemies)))

        for slot_id, hist in list(self._histories.items()):
            if not hist:
                continue
            prev = hist[-1]
            best_i, best_d = -1, 1e9
            for ei in unmatched_e:
                d = (enemies[ei][0] - prev[0]) ** 2 + (enemies[ei][1] - prev[1]) ** 2
                if d < best_d:
                    best_d = d
                    best_i = ei
            if best_i >= 0 and best_d < 0.15 ** 2:
                matched[slot_id] = best_i
                unmatched_e.remove(best_i)

        # Prune unmatched slots (enemy out of sight)
        for sid in list(self._histories.keys()):
            if sid not in matched:
                del self._histories[sid]
                self._announced.pop(sid, None)

        # Add new slots for unmatched enemies
        for ei in unmatched_e:
            sid = self._next_slot
            self._next_slot += 1
            self._histories[sid] = deque(maxlen=_HISTORY_FRAMES)
            self._histories[sid].append(enemies[ei])
            matched[sid] = ei

        # Update histories
        for sid, ei in matched.items():
            self._histories[sid].append(enemies[ei])

        # Predict and generate callouts
        callouts: List[str] = []
        for sid, ei in matched.items():
            hist = list(self._histories[sid])
            if len(hist) < 3:
                continue
            result = self._predict_one(hist, map_name)
            if result is None:
                continue
            pred_zone, conf = result
            current_zone = pos_to_zone(enemies[ei][0], enemies[ei][1], map_name)
            if (pred_zone != current_zone
                    and conf >= _CONF_THRESHOLD
                    and self._announced.get(sid) != pred_zone):
                callouts.append(
                    f"Enemy heading toward {pred_zone} in ~{int(_PREDICT_SEC)}s"
                )
                self._announced[sid] = pred_zone

        return callouts

    def _predict_one(
        self, hist: list, map_name: str
    ) -> Optional[Tuple[str, float]]:
        """
        Returns (predicted_zone, confidence) or None.
        confidence: 0-1 based on velocity consistency.
        """
        xs = np.array([p[0] for p in hist])
        ys = np.array([p[1] for p in hist])
        t  = np.arange(len(hist), dtype=float)

        # Least-squares linear fit
        cx = np.polyfit(t, xs, 1)   # [slope, intercept]
        cy = np.polyfit(t, ys, 1)
        vx = float(cx[0])           # units/frame
        vy = float(cy[0])
        speed = np.sqrt(vx ** 2 + vy ** 2)

        if speed < _MIN_SPEED:
            return None   # enemy not moving

        # Residuals -> confidence (use actual regression line, not forced-through-first-point)
        pred_xs = np.polyval(cx, t)
        residuals = np.std(xs - pred_xs) + np.std(ys - np.polyval(cy, t))
        conf = float(np.exp(-residuals / (speed + 1e-9) * 5.0))

        # Extrapolate
        last_x, last_y = hist[-1]
        pred_x = float(np.clip(last_x + vx * _PREDICT_FRAMES, 0.0, 1.0))
        pred_y = float(np.clip(last_y + vy * _PREDICT_FRAMES, 0.0, 1.0))
        pred_zone = pos_to_zone(pred_x, pred_y, map_name)

        return pred_zone, conf

    def reset(self) -> None:
        self._histories.clear()
        self._announced.clear()
        self._next_slot = 0
