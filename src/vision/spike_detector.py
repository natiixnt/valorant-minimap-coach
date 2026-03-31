"""
Detects the planted spike (C4) on the Valorant minimap.

Two-phase approach to avoid false positives:

  Phase 1 – CANDIDATE (minimap only):
    Yellow-orange blob seen for _CONFIRM_FRAMES consecutive frames.
    Coach arms the audio beep detector but does NOT announce yet.
    Problem with minimap-only: the spike icon also appears when the spike is
    lying on the ground (dropped) or being partially planted (plant cancelled).

  Phase 2 – CONFIRMED (audio + minimap):
    First spike beep arrives in audio → true full-plant confirmed → announce.
    Spike beeps only start after the plant animation completes (Riot design).

  Fallback:
    If _AUDIO_CONFIRM_TIMEOUT seconds pass without an audio beep (audio
    disabled or missed), the candidate is auto-confirmed from minimap alone.

The caller (coach.py) is responsible for the two-phase handshake:
  1. Check spike_detector.is_candidate → arm audio beep detector
  2. Check audio_coach.spike_timer.has_started → call spike_detector.confirm_planted()
"""
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.capture.screen import MinimapFrame

# HSV range for the spike planted marker (warm yellow-orange pulse)
_SPIKE_LOWER = np.array([18, 180, 180])
_SPIKE_UPPER = np.array([32, 255, 255])

_MIN_AREA = 6
_CONFIRM_FRAMES = 4        # consecutive frames to enter CANDIDATE state
_CLEAR_FRAMES = 6          # consecutive absent frames to clear CANDIDATE (not PLANTED)
_AUDIO_CONFIRM_TIMEOUT = 5.0   # seconds: auto-confirm from minimap if no beep arrives


class SpikeDetector:
    def __init__(self) -> None:
        self._kernel = np.ones((3, 3), np.uint8)
        self._consec_seen = 0
        self._consec_absent = 0
        self._candidate = False      # minimap confirmed, audio not yet
        self._planted = False        # audio-confirmed (or timeout fallback)
        self._pos: Optional[Tuple[float, float]] = None
        self._candidate_since: float = 0.0

    def update(self, frame: MinimapFrame) -> None:
        """
        Call once per coach loop tick to update internal state.
        Does NOT return position directly - use is_candidate / is_planted / planted_pos.
        """
        hsv = cv2.cvtColor(frame.data, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _SPIKE_LOWER, _SPIKE_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        h, w = frame.data.shape[:2]
        best: Optional[Tuple[float, float]] = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= _MIN_AREA and area > best_area:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    best = (M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h)
                    best_area = area

        if best is not None:
            self._consec_seen += 1
            self._consec_absent = 0
            self._pos = best
            if self._consec_seen >= _CONFIRM_FRAMES and not self._candidate and not self._planted:
                self._candidate = True
                self._candidate_since = time.monotonic()
        else:
            self._consec_absent += 1
            if self._consec_absent >= _CLEAR_FRAMES and not self._planted:
                # Lost sight of spike and not confirmed planted - reset candidate
                self._candidate = False
                self._consec_seen = 0
                self._consec_absent = 0
                # Keep _pos so candidate_pos is still valid briefly

    def confirm_planted(self, pos: Optional[Tuple[float, float]] = None) -> None:
        """
        Called by coach.py when audio (or timeout) confirms the plant.
        pos: override position if minimap gave a more precise location.
        """
        self._planted = True
        self._candidate = False
        if pos is not None:
            self._pos = pos

    @property
    def candidate_timeout_reached(self) -> bool:
        """True if candidate has been waiting longer than _AUDIO_CONFIRM_TIMEOUT."""
        return (
            self._candidate
            and not self._planted
            and (time.monotonic() - self._candidate_since) > _AUDIO_CONFIRM_TIMEOUT
        )

    def reset(self) -> None:
        """Call at round start to clear all state."""
        self._consec_seen = 0
        self._consec_absent = 0
        self._candidate = False
        self._planted = False
        self._pos = None
        self._candidate_since = 0.0

    @property
    def is_candidate(self) -> bool:
        """Minimap has confirmed spike position but audio not yet received."""
        return self._candidate and not self._planted

    @property
    def candidate_pos(self) -> Optional[Tuple[float, float]]:
        """Current blob position when in candidate or planted state."""
        return self._pos

    @property
    def is_planted(self) -> bool:
        return self._planted

    @property
    def planted_pos(self) -> Optional[Tuple[float, float]]:
        return self._pos if self._planted else None
