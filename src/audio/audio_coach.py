"""
Pro audio analysis orchestrator.

Fuses:
  - Real-time stereo audio capture (WASAPI / BlackHole loopback)
  - Footstep onset detection (bandpass spectral flux)
  - ITD + ILD stereo direction estimation (azimuth, distance)
  - Player facing angle from minimap (PlayerAngleDetector)
  - Agent footstep classification (RandomForest on MFCCs)
  - Surface material classification (spectral centroid -> map zone cross-check)

Output: AudioFinding dataclass consumed by coach.py to produce voice callouts.

Threading model:
  AudioCoach.start() launches a background thread that reads from AudioCapture,
  processes audio, and pushes AudioFinding objects into a queue.
  The main coach loop calls AudioCoach.poll() each tick to drain the queue.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.audio.capture import AudioCapture, SAMPLE_RATE
from src.audio.footstep_detector import FootstepDetector, FootstepEvent
from src.audio.direction_estimator import DirectionEstimator, audio_az_to_map_direction, direction_to_map_pos
from src.audio.agent_classifier import AgentClassifier
from src.maps.callouts import pos_to_zone
from src.maps.surfaces import get_surface, surface_matches, surface_to_voice

# Model path (relative to cwd = project root)
_MODEL_PATH = "data/footstep_model.pkl"

# Minimum classifier confidence to report agent name (otherwise "someone")
_AGENT_CONF_THRESHOLD = 0.45

# Window around onset to feed into direction / classifier (samples)
_ANALYSIS_WINDOW = int(0.35 * SAMPLE_RATE)   # 350 ms

# Scale: rough meters per normalized map unit (calibrate per map if needed)
_M_PER_UNIT: dict[str, float] = {
    "ascent": 110.0, "bind": 100.0, "haven": 120.0, "split": 90.0,
    "icebox": 105.0, "breeze": 130.0, "fracture": 115.0, "pearl": 108.0,
    "lotus": 112.0, "sunset": 100.0, "abyss": 105.0,
}
_DEFAULT_M_PER_UNIT = 105.0


@dataclass
class AudioFinding:
    """One footstep event processed into game-useful information."""
    agent: str                          # shoe type: "heavy" | "medium" | "light" | "unknown"
    agent_role: str                     # same as agent (shoe type)
    surface: str                        # "metal" | "wood" | "concrete" | "carpet"
    azimuth_deg: float                  # relative to player: 0=ahead, +90=right
    map_direction_deg: float            # compass on map: 0=up, 90=right
    estimated_pos: Tuple[float, float]  # normalized (x,y) on minimap
    zone: str                           # map zone name e.g. "B Long", or "" if unknown
    distance_m: float                   # rough distance estimate
    confidence: float                   # overall confidence 0-1
    audio_clip: Optional[np.ndarray]    # mono float32 audio clip for data collection
    voice_text: str                     # ready-to-speak callout string


class AudioCoach:
    def __init__(self, config: dict) -> None:
        self._config = config
        audio_cfg = config.get("audio_coach", {})
        self._enabled: bool = audio_cfg.get("enabled", True)
        device_name: Optional[str] = audio_cfg.get("device", None)

        self._capture = AudioCapture(device_name=device_name)
        self._footstep_det = FootstepDetector()
        self._direction_est = DirectionEstimator()
        self._classifier = AgentClassifier()

        # Try to load pre-trained model
        if os.path.exists(_MODEL_PATH):
            self._classifier.load(_MODEL_PATH)
        else:
            print("[AudioCoach] No footstep model found. Agent identification disabled.")
            print(f"             Train with: python tools/collect_footsteps.py")

        self._queue: queue.Queue[AudioFinding] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Shared state updated by coach.py main loop
        self.player_facing: float = 0.0          # degrees, updated each frame
        self.player_pos: Tuple[float, float] = (0.5, 0.5)
        self.map_name: str = "unknown"

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._enabled:
            return
        self._running = True
        self._capture.start()
        self._thread = threading.Thread(
            target=self._analysis_loop, daemon=True, name="AudioCoach"
        )
        self._thread.start()
        print("[AudioCoach] Started.")

    def stop(self) -> None:
        self._running = False
        self._capture.stop()
        if self._thread:
            self._thread.join(timeout=3)

    def poll(self) -> List[AudioFinding]:
        """Drain and return all pending AudioFinding objects (called from coach loop)."""
        findings = []
        try:
            while True:
                findings.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return findings

    # ------------------------------------------------------------------
    def _analysis_loop(self) -> None:
        """Background thread: detect footsteps and push AudioFinding to queue."""
        prev_sample_count = 0
        while self._running:
            stereo = self._capture.read(n_samples=_ANALYSIS_WINDOW * 2)
            if stereo is None:
                time.sleep(0.05)
                continue

            # FootstepDetector works on mono but needs stereo for balance
            mono = (stereo[0] + stereo[1]) * 0.5
            # Feed only the new half to avoid re-processing
            events: List[FootstepEvent] = self._footstep_det.process(stereo)

            for event in events:
                finding = self._process_event(event, stereo)
                if finding is not None:
                    self._queue.put(finding)

            time.sleep(0.02)   # 20 ms tick

    def _process_event(
        self, event: FootstepEvent, stereo: np.ndarray
    ) -> Optional[AudioFinding]:
        # -- Direction
        az, dist_m = self._direction_est.estimate(stereo, event.amplitude_db)
        map_dir = audio_az_to_map_direction(az, self.player_facing)

        scale = _M_PER_UNIT.get(self.map_name.lower(), _DEFAULT_M_PER_UNIT)
        est_pos = (0.5, 0.5)
        if self.player_pos != (0.5, 0.5):
            est_pos = direction_to_map_pos(
                self.player_pos, map_dir, dist_m, scale_m_per_unit=scale
            )

        # -- Shoe type classification
        # Note: Valorant agents do NOT have unique footstep sounds per Riot.
        # We classify shoe type (heavy/medium/light) which maps to broad agent categories.
        mono_window = (stereo[0] + stereo[1]) * 0.5
        if len(mono_window) >= int(0.35 * SAMPLE_RATE):
            shoe_type, conf, _ = self._classifier.predict(mono_window)
        else:
            shoe_type, conf = "unknown", 0.0

        if conf < _AGENT_CONF_THRESHOLD or shoe_type == "unknown":
            shoe_display = "enemy"
        else:
            shoe_display = f"{shoe_type}-step enemy"  # e.g. "heavy-step enemy"

        # -- Zone name from estimated map position
        zone = ""
        if self.map_name != "unknown":
            zone = pos_to_zone(est_pos[0], est_pos[1], self.map_name)

        # -- Surface cross-check against map
        map_surface = get_surface(est_pos[0], est_pos[1], self.map_name)
        surface = event.surface
        surface_hint = ""
        if not surface_matches(surface, map_surface) and self.map_name != "unknown":
            conf *= 0.7
        if surface != "concrete":
            surface_hint = f" on {surface_to_voice(surface)}"

        # -- Build callout: prefer zone name over directional word
        dist_word = _dist_to_word(dist_m)
        if zone and zone != "Unknown":
            location = zone                        # "B Long", "Mid Courtyard", etc.
        else:
            location = _az_to_word(az)             # fallback: "right", "behind", etc.

        voice = f"{shoe_display} footstep at {location}, {dist_word}{surface_hint}"

        overall_conf = float(np.clip(
            (1.0 - abs(event.amplitude_db + 30) / 30.0) * 0.5 + conf * 0.5,
            0.0, 1.0
        ))

        return AudioFinding(
            agent=shoe_type,
            agent_role=shoe_type,
            surface=surface,
            azimuth_deg=az,
            map_direction_deg=map_dir,
            estimated_pos=est_pos,
            zone=zone,
            distance_m=dist_m,
            confidence=overall_conf,
            audio_clip=mono_window,
            voice_text=voice,
        )


# ------------------------------------------------------------------
def _az_to_word(az: float) -> str:
    """Convert azimuth angle to rough directional word."""
    if -22.5 < az <= 22.5:
        return "ahead"
    elif 22.5 < az <= 67.5:
        return "front right"
    elif 67.5 < az <= 112.5:
        return "right"
    elif 112.5 < az <= 157.5:
        return "rear right"
    elif az > 157.5 or az <= -157.5:
        return "behind"
    elif -157.5 < az <= -112.5:
        return "rear left"
    elif -112.5 < az <= -67.5:
        return "left"
    else:
        return "front left"


def _dist_to_word(dist_m: float) -> str:
    if dist_m < 8:
        return "very close"
    elif dist_m < 15:
        return "nearby"
    elif dist_m < 22:
        return "medium range"
    else:
        return "far"
