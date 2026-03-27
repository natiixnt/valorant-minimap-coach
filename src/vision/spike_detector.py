"""
Detects the planted spike (C4) icon on the Valorant minimap.

When the spike is planted a pulsing yellow/orange icon appears at the plant
location. We require the marker to appear in several consecutive frames before
announcing, to avoid false positives from ability effects or HUD elements.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from src.capture.screen import MinimapFrame

# HSV range for the spike planted marker (warm yellow-orange pulse)
_SPIKE_LOWER = np.array([18, 180, 180])
_SPIKE_UPPER = np.array([32, 255, 255])

_MIN_AREA = 6
_CONFIRM_FRAMES = 4   # consecutive frames required to confirm plant
_CLEAR_FRAMES = 6     # consecutive absent frames required to clear


class SpikeDetector:
    def __init__(self) -> None:
        self._kernel = np.ones((3, 3), np.uint8)
        self._consec_seen = 0
        self._consec_absent = 0
        self._planted = False
        self._pos: Optional[Tuple[float, float]] = None

    def update(self, frame: MinimapFrame) -> Optional[Tuple[float, float]]:
        """
        Call once per coach loop tick.

        Returns the normalized (x, y) position the first time the spike is
        confirmed planted. Returns None every other tick.
        """
        hsv = cv2.cvtColor(frame.data, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _SPIKE_LOWER, _SPIKE_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.data.shape[:2]
        best: Optional[Tuple[float, float]] = None
        for cnt in contours:
            if cv2.contourArea(cnt) >= _MIN_AREA:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    best = (M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h)
                    break

        if best is not None:
            self._consec_seen += 1
            self._consec_absent = 0
            self._pos = best
            if self._consec_seen == _CONFIRM_FRAMES and not self._planted:
                self._planted = True
                return self._pos
        else:
            self._consec_absent += 1
            if self._consec_absent >= _CLEAR_FRAMES:
                self._consec_seen = 0
                self._planted = False
                self._pos = None

        return None

    def reset(self) -> None:
        """Call at round start to clear state."""
        self._consec_seen = 0
        self._consec_absent = 0
        self._planted = False
        self._pos = None

    @property
    def is_planted(self) -> bool:
        return self._planted

    @property
    def planted_pos(self) -> Optional[Tuple[float, float]]:
        return self._pos if self._planted else None
