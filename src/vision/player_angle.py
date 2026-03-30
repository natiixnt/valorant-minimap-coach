"""
Detect the local player's facing direction from the Valorant minimap.

The player icon is a bright white/cyan triangle (cone) centered on the minimap.
Its orientation indicates the direction the player is looking.

Algorithm:
  1. Crop a small region around the center of the minimap where the player icon sits.
  2. Threshold to isolate bright white/light-blue pixels (the icon).
  3. Find the largest contour -> image moments -> orientation angle.
  4. Map from OpenCV image angle (CCW from right) to game compass (CW from up/north).

Returns angle in degrees: 0 = up/north on the minimap, 90 = right/east, etc.
Returns None if the icon cannot be reliably detected.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.capture.screen import MinimapFrame

# Fraction of minimap width/height to crop around center for player icon search
_CENTER_CROP = 0.15

# HSV range for the player icon (bright white-cyan)
_ICON_LOWER = np.array([85, 0, 200])    # allow slight cyan tint + pure white
_ICON_UPPER = np.array([105, 80, 255])

# Fallback pure-white range (very low saturation)
_WHITE_LOWER = np.array([0, 0, 220])
_WHITE_UPPER = np.array([180, 40, 255])

_MIN_AREA = 8    # pixels — ignore tiny noise blobs


class PlayerAngleDetector:
    def __init__(self) -> None:
        self._kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame: MinimapFrame) -> Optional[float]:
        """
        Returns player facing angle in degrees (0=up/north, 90=right, 180=down, 270=left).
        Returns None if detection fails.
        """
        h, w = frame.data.shape[:2]
        cx, cy = w // 2, h // 2
        half = int(min(w, h) * _CENTER_CROP)

        # Crop center region
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        crop = frame.data[y0:y1, x0:x1]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Try cyan-white range first, then pure white
        mask = cv2.inRange(hsv, _ICON_LOWER, _ICON_UPPER)
        if cv2.countNonZero(mask) < _MIN_AREA:
            mask = cv2.inRange(hsv, _WHITE_LOWER, _WHITE_UPPER)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= _MIN_AREA and area > best_area:
                best_area = area
                best = cnt

        if best is None:
            return None

        if len(best) >= 5:
            # Fit ellipse for orientation
            try:
                _, _, angle_cv = cv2.fitEllipse(best)
                # cv2.fitEllipse angle: 0-180, CCW from horizontal (+x axis)
                # Convert to game compass: 0=up, CW
                compass = _cv_angle_to_compass(angle_cv, best, crop.shape)
                return compass
            except cv2.error:
                pass

        # Fallback: use image moments to find orientation
        M = cv2.moments(best)
        if M["m00"] < 1:
            return None

        # Compute centroid
        mc_x = M["m10"] / M["m00"]
        mc_y = M["m01"] / M["m00"]

        # Use mu20, mu02, mu11 to get orientation of best-fit ellipse
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]

        angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        angle_cv2 = float(np.rad2deg(angle_rad))

        compass = _cv_angle_to_compass(angle_cv2, best, crop.shape)
        return compass


def _cv_angle_to_compass(cv_angle: float, contour: np.ndarray, shape: Tuple[int, int]) -> float:
    """
    Map OpenCV angle to compass heading.

    We detect which end of the ellipse axis points toward the 'tip' of the triangle
    by checking where the contour's centroid-to-tip vector points.
    """
    M = cv2.moments(contour)
    if M["m00"] < 1:
        return (90.0 - cv_angle) % 360

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Find the point in the contour farthest from the centroid (= tip of triangle)
    pts = contour[:, 0, :]
    dists_sq = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
    tip = pts[np.argmax(dists_sq)]

    # Vector from centroid to tip
    dx = float(tip[0] - cx)
    dy = float(tip[1] - cy)

    # Image coords: y increases downward. Convert to map coords: 0=up.
    # angle from +y axis (up), clockwise
    compass = float(np.degrees(np.arctan2(dx, -dy))) % 360
    return compass
