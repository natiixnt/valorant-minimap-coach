"""
CV-based detection of ability/utility items on the Valorant minimap.

Each ability has a distinct color on the minimap that differs from team (cyan)
and enemy (red) blobs. Colors are tunable in config.yaml under
detection.abilities.<kind>.lower/upper if defaults don't match your monitor.

Tracks state across frames: reports abilities on first appearance and on
disappearance, so the coach speaks only when something changes.
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.capture.screen import MinimapFrame


@dataclass
class DetectedAbility:
    kind: str
    voice: Optional[str]                # text to speak on first detection; None = overlay-only
    display: str                        # short label for overlay
    color: str                          # hex color for overlay dot
    position: Tuple[float, float]       # normalized 0-1


# Default HSV signatures per ability type.
# Override any range in config.yaml: detection.abilities.<kind>.lower / .upper
_DEFAULTS: Dict[str, dict] = {
    "reyna_eye": {
        "lower": [145, 80, 140],
        "upper": [165, 255, 255],
        "voice": None,          # 3s, at kill location, whole team already knows
        "display": "Reyna Eye",
        "color": "#e040fb",
        "min_area": 6,
    },
    "viper_wall": {
        "lower": [55, 140, 110],
        "upper": [85, 255, 255],
        "voice": "Viper wall at {zone}",
        "display": "Viper Wall",
        "color": "#3ec17c",
        "min_area": 10,
    },
    "sage_wall": {
        "lower": [0, 0, 215],
        "upper": [180, 35, 255],
        "voice": "Sage wall at {zone}",
        "display": "Sage Wall",
        "color": "#ecf0f1",
        "min_area": 12,
    },
    "killjoy": {
        "lower": [20, 140, 140],
        "upper": [35, 255, 255],
        "voice": "Killjoy setup at {zone}",
        "display": "Killjoy",
        "color": "#f9c74f",
        "min_area": 5,
    },
    "sova_bolt": {
        "lower": [12, 170, 170],
        "upper": [28, 255, 255],
        "voice": None,          # Shock Dart: very brief, visible on screen
        "display": "Sova Bolt",
        "color": "#f77f00",
        "min_area": 5,
    },
    "phoenix_fire": {
        "lower": [5, 190, 170],
        "upper": [15, 255, 255],
        "voice": None,          # Hot Hands: ~8s, locally visible, team sees Phoenix use it
        "display": "Phoenix Fire",
        "color": "#ff6b35",
        "min_area": 8,
    },
    # Omen/Brimstone smokes show as dark teal/gray circles on minimap
    "omen_smoke": {
        "lower": [95, 60, 55],
        "upper": [125, 160, 130],
        "voice": "Smoke at {zone}, careful",
        "display": "Smoke",
        "color": "#78909c",
        "min_area": 14,
    },
    # Skye guide (Trailblazer) — bright teal flash
    "skye_guide": {
        "lower": [158, 130, 150],
        "upper": [172, 255, 255],
        "voice": None,          # ~4s, controlled by Skye, team knows it's active
        "display": "Skye Guide",
        "color": "#26c6da",
        "min_area": 5,
    },
    # Cypher camera — small bright cyan dot (slightly different hue from team)
    "cypher_camera": {
        "lower": [85, 180, 180],
        "upper": [100, 255, 255],
        "voice": "Cypher camera at {zone}, watch out",
        "display": "Cypher Cam",
        "color": "#00e5ff",
        "min_area": 3,
    },
    # KAY/O knife suppression circle — lime green
    "kayo_knife": {
        "lower": [40, 160, 160],
        "upper": [58, 255, 255],
        "voice": "KAY/O knife at {zone}, suppression",
        "display": "KAYO Knife",
        "color": "#aeea00",
        "min_area": 10,
    },
}

# How long without a detection before we consider the ability gone
_GONE_AFTER = 1.5  # seconds


def _build_signatures(config: dict) -> Dict[str, dict]:
    """Merge defaults with any per-ability overrides from config.yaml."""
    overrides = config.get("detection", {}).get("abilities", {})
    sigs = {}
    for kind, default in _DEFAULTS.items():
        sig = dict(default)
        if kind in overrides:
            ov = overrides[kind]
            if "lower" in ov:
                sig["lower"] = ov["lower"]
            if "upper" in ov:
                sig["upper"] = ov["upper"]
        sig["lower"] = np.array(sig["lower"])
        sig["upper"] = np.array(sig["upper"])
        sigs[kind] = sig
    return sigs


class AbilityDetector:
    def __init__(self, config: dict) -> None:
        self._sigs = _build_signatures(config)
        self._kernel = np.ones((3, 3), np.uint8)
        self._base_min = config.get("detection", {}).get("ability_min_area", 4)
        # kind -> (timestamp, position) of last detection
        self._active: Dict[str, Tuple[float, Tuple[float, float]]] = {}

    def _blobs(
        self,
        hsv: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        min_area: int,
    ) -> List[Tuple[float, float]]:
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = hsv.shape[:2]
        out = []
        for cnt in contours:
            if cv2.contourArea(cnt) < max(min_area, self._base_min):
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                out.append((M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h))
        return out

    def update(
        self, frame: MinimapFrame
    ) -> Tuple[List[DetectedAbility], List[str]]:
        """
        Run one detection pass.

        Returns:
            appeared: abilities detected for the first time this tick
            gone:     ability kinds that have not been seen for _GONE_AFTER seconds
        """
        now = time.time()
        hsv = cv2.cvtColor(frame.data, cv2.COLOR_BGR2HSV)
        seen_now: Dict[str, List[Tuple[float, float]]] = {}

        for kind, sig in self._sigs.items():
            positions = self._blobs(hsv, sig["lower"], sig["upper"], sig["min_area"])
            if positions:
                seen_now[kind] = positions

        appeared: List[DetectedAbility] = []
        for kind, positions in seen_now.items():
            if kind not in self._active:
                sig = self._sigs[kind]
                appeared.append(DetectedAbility(
                    kind=kind,
                    voice=sig["voice"],
                    display=sig["display"],
                    color=sig["color"],
                    position=positions[0],
                ))
            self._active[kind] = (now, positions[0])

        gone: List[str] = []
        stale = [k for k, (t, _) in self._active.items() if now - t > _GONE_AFTER]
        for k in stale:
            gone.append(k)
            del self._active[k]

        return appeared, gone

    @property
    def active(self) -> Dict[str, dict]:
        """Currently tracked abilities: kind -> sig dict (with display, color, position)."""
        return {
            k: {**self._sigs[k], "position": self._active[k][1]}
            for k in self._active if k in self._sigs
        }
