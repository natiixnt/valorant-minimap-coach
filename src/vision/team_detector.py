"""
Track ally (teammate) positions on the Valorant minimap.

Allies appear as filled cyan/teal circles. HSV range matches the team color
configured under detection.team_color_lower/upper in config.yaml.

Returns a list of normalized (x, y) positions, same coordinate system as enemy detector.
Also provides velocity estimation (pixels/frame) by comparing consecutive detections.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.capture.screen import MinimapFrame

_KERNEL = np.ones((3, 3), np.uint8)
_MAX_TEAMMATES = 4   # solo player excluded; up to 4 others visible


class TeamDetector:
    def __init__(self, config: dict) -> None:
        det = config.get("detection", {})
        self._lower = np.array(det.get("team_color_lower", [80, 100, 100]))
        self._upper = np.array(det.get("team_color_upper", [100, 255, 255]))
        self._min_area: float = float(det.get("min_contour_area", 5))
        self._prev: List[Tuple[float, float]] = []

    def detect(self, frame: MinimapFrame) -> List[Tuple[float, float]]:
        """
        Returns list of (x, y) normalized 0-1 teammate positions.
        Player icon is near the center and excluded (within 0.12 of center).
        """
        hsv = cv2.cvtColor(frame.data, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.data.shape[:2]
        positions = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self._min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            nx = M["m10"] / M["m00"] / w
            ny = M["m01"] / M["m00"] / h
            # Exclude the local player icon (center ±0.12)
            if abs(nx - 0.5) < 0.12 and abs(ny - 0.5) < 0.12:
                continue
            positions.append((nx, ny))

        # Keep up to 4 largest
        positions = positions[:_MAX_TEAMMATES]
        self._prev = positions
        return positions

    def velocity(
        self,
        current: List[Tuple[float, float]],
        prev: List[Tuple[float, float]],
    ) -> Dict[int, Tuple[float, float]]:
        """
        Match current to prev by nearest neighbour and compute (dx, dy) per frame.
        Returns {current_idx: (vx, vy)} for matched pairs.
        """
        if not prev or not current:
            return {}
        velocities = {}
        used = set()
        for i, (cx, cy) in enumerate(current):
            best_j, best_d = -1, 1e9
            for j, (px, py) in enumerate(prev):
                if j in used:
                    continue
                d = (cx - px) ** 2 + (cy - py) ** 2
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d < 0.05 ** 2:
                velocities[i] = (
                    current[i][0] - prev[best_j][0],
                    current[i][1] - prev[best_j][1],
                )
                used.add(best_j)
        return velocities
