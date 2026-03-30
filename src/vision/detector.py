from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.capture.screen import MinimapFrame


@dataclass
class DetectionResult:
    enemies: List[Tuple[float, float]]   # normalized (0-1) x, y
    teammates: List[Tuple[float, float]]
    timestamp: float
    enemy_count: int = field(init=False)

    def __post_init__(self):
        self.enemy_count = len(self.enemies)


class MinimapDetector:
    def __init__(self, config: dict):
        det = config["detection"]
        self.enemy_lower = np.array(det["enemy_color_lower"])
        self.enemy_upper = np.array(det["enemy_color_upper"])
        # Red wraps around hue=180 in HSV, so we need a second range (optional)
        self.enemy_lower2 = np.array(det["enemy_color_lower2"]) if "enemy_color_lower2" in det else None
        self.enemy_upper2 = np.array(det["enemy_color_upper2"]) if "enemy_color_upper2" in det else None
        self.team_lower = np.array(det["team_color_lower"])
        self.team_upper = np.array(det["team_color_upper"])
        self.min_area = det["min_contour_area"]
        self._kernel = np.ones((3, 3), np.uint8)

    def _find_blobs(
        self,
        hsv: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        lower2: Optional[np.ndarray] = None,
        upper2: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, float]]:
        mask = cv2.inRange(hsv, lower, upper)
        if lower2 is not None:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = hsv.shape[:2]
        positions = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                positions.append((M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h))
        return positions

    def detect(self, frame: MinimapFrame) -> DetectionResult:
        hsv = cv2.cvtColor(frame.data, cv2.COLOR_BGR2HSV)
        enemies = self._find_blobs(
            hsv, self.enemy_lower, self.enemy_upper, self.enemy_lower2, self.enemy_upper2
        )
        teammates = self._find_blobs(hsv, self.team_lower, self.team_upper)
        return DetectionResult(enemies=enemies, teammates=teammates, timestamp=frame.timestamp)
