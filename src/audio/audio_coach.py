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
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.audio.capture import AudioCapture, SAMPLE_RATE
from src.audio.footstep_detector import FootstepDetector, FootstepEvent
from src.audio.direction_estimator import DirectionEstimator, audio_az_to_map_direction, direction_to_map_pos
from src.audio.agent_classifier import AgentClassifier
from src.audio.gunshot_detector import GunDetector, GunEvent
from src.audio.noise_gate import NoiseGate
from src.audio.round_audio import RoundAudioDetector
from src.audio.spike_audio import SpikeBeepDetector, SpikeTimer, DefuseSoundDetector
from src.maps.callouts import pos_to_zone
from src.maps.surfaces import get_surface, surface_matches, surface_to_voice
from src.game.enemy_agents import EnemyAgentTracker

# Model path (relative to cwd = project root)
_MODEL_PATH = "data/footstep_model.pkl"

# Minimum classifier confidence to report agent name (otherwise "someone")
_AGENT_CONF_THRESHOLD = 0.45

# Window around onset to feed into direction / classifier (samples)
_ANALYSIS_WINDOW = int(0.35 * SAMPLE_RATE)   # 350 ms

# _analysis_loop reads _ANALYSIS_WINDOW * 2 = 700 ms from the ring buffer each tick.
# Using 2x the single-event window ensures that even if an onset falls at the very
# start of the chunk, there is still 350 ms of context before it for the direction
# estimator and classifier. The loop sleeps only 20 ms between reads, so consecutive
# 700 ms windows overlap heavily - this is intentional: overlapping reads act as a
# sliding window so no onset can slip between two non-overlapping reads.

# Scale: rough meters per normalized map unit (calibrate per map if needed)
_M_PER_UNIT: dict[str, float] = {
    "ascent": 110.0, "bind": 100.0, "haven": 120.0, "split": 90.0,
    "icebox": 105.0, "breeze": 130.0, "fracture": 115.0, "pearl": 108.0,
    "lotus": 112.0, "sunset": 100.0, "abyss": 105.0,
}
_DEFAULT_M_PER_UNIT = 105.0

# Gunshot burst clustering
_GUN_BURST_WINDOW_S   = 0.5    # shots within this wall-clock window = burst
# 1.5 s suppression after a burst callout is announced: most Valorant full-auto
# magazines empty in ~1-1.2 s, so 1.5 s prevents the coach from calling out
# every individual round while still catching a second separate engagement.
_GUN_BURST_SUPPRESS_S = 1.5    # suppress further callouts after burst announced


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
        self._noise_gate    = NoiseGate()
        self._footstep_det  = FootstepDetector()
        self._gun_det       = GunDetector()
        self._direction_est = DirectionEstimator()
        self._classifier    = AgentClassifier()
        self._spike_beep    = SpikeBeepDetector()
        self._spike_timer   = SpikeTimer()
        self._defuse_sound  = DefuseSoundDetector()

        # Try to load pre-trained model
        if os.path.exists(_MODEL_PATH):
            self._classifier.load(_MODEL_PATH)
        else:
            print("[AudioCoach] No footstep model found. Shoe-type ID disabled.")
            print(f"             Train with: python tools/collect_footsteps.py")

        self._queue: queue.Queue = queue.Queue()   # AudioFinding | GunEvent
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Round audio event detector -- callbacks wired by coach.py
        self.round_audio = RoundAudioDetector()

        # Spike audio: beep detector + timer + defuse tracker exposed to coach.py
        self.spike_timer   = self._spike_timer
        self.defuse_sound  = self._defuse_sound

        # Shared state updated by coach.py main loop, read by audio thread.
        # Lock protects writes/reads across thread boundary.
        self._state_lock = threading.Lock()
        self._player_facing: float = 0.0
        self._player_pos: Tuple[float, float] = (0.5, 0.5)
        self._map_name: str = "unknown"
        self._enemy_agents: Optional[EnemyAgentTracker] = None

        # Footstep deduplication: suppress re-detections from overlapping 700ms windows
        self._last_step_wall_t: float = 0.0
        self._last_step_az: float = 999.0
        # Inter-step cadence tracking for walking/running detection (audio thread only)
        self._step_history: deque = deque(maxlen=6)
        # Gunshot deduplication and burst clustering state (audio thread only)
        self._last_gun_wall_t: float = 0.0
        self._last_gun_az: float = 999.0
        self._gun_burst: dict = {}   # sector -> (last_wall_t, count, suppress_until)

    @property
    def player_facing(self) -> float:
        with self._state_lock:
            return self._player_facing

    @player_facing.setter
    def player_facing(self, value: float) -> None:
        with self._state_lock:
            self._player_facing = value

    @property
    def player_pos(self) -> Tuple[float, float]:
        with self._state_lock:
            return self._player_pos

    @player_pos.setter
    def player_pos(self, value: Tuple[float, float]) -> None:
        with self._state_lock:
            self._player_pos = value

    @property
    def map_name(self) -> str:
        with self._state_lock:
            return self._map_name

    @map_name.setter
    def map_name(self, value: str) -> None:
        with self._state_lock:
            self._map_name = value

    @property
    def enemy_agents(self) -> Optional[EnemyAgentTracker]:
        with self._state_lock:
            return self._enemy_agents

    @enemy_agents.setter
    def enemy_agents(self, value: Optional[EnemyAgentTracker]) -> None:
        with self._state_lock:
            self._enemy_agents = value

    # ------------------------------------------------------------------
    def arm_spike_audio(self) -> None:
        """
        Arm the beep detector when minimap shows a spike candidate (not yet confirmed).
        Does NOT start the timer -- call on_spike_planted() when plant is audio-confirmed.
        """
        self._spike_beep.arm()

    def on_spike_planted(self, plant_time: float) -> None:
        """Confirm spike planted -- start timer and arm defuse detection."""
        self._spike_timer.on_spike_planted(plant_time)
        self._defuse_sound.arm()

    def on_round_start(self) -> None:
        """Clear per-round audio state so previous-round history doesn't bleed in."""
        self._footstep_det.reset()
        self._gun_det.reset()
        self._noise_gate.reset()
        self._last_step_wall_t = 0.0
        self._last_step_az = 999.0
        self._step_history.clear()
        self._last_gun_wall_t = 0.0
        self._last_gun_az = 999.0
        self._gun_burst.clear()

    def on_spike_resolved(self) -> None:
        """Disarm spike audio detection (defused or detonated)."""
        self._spike_beep.reset()
        self._spike_timer.reset()
        self._defuse_sound.reset()

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

    def poll(self) -> list:
        """Drain and return all pending AudioFinding and GunEvent objects."""
        items = []
        try:
            while True:
                items.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return items

    # ------------------------------------------------------------------
    def _analysis_loop(self) -> None:
        """Background thread: detect footsteps and round audio events."""
        while self._running:
            stereo = self._capture.read(n_samples=_ANALYSIS_WINDOW * 2)
            if stereo is None:
                time.sleep(0.05)
                continue

            mono = (stereo[0] + stereo[1]) * 0.5

            # Round audio event detection (round start horn, win/loss jingle)
            self.round_audio.process(mono)

            # Gunshot detection (before noise gate -- gunshots ARE the transients)
            gun_events: List[GunEvent] = self._gun_det.process(stereo)
            for ge in gun_events:
                now_m = time.monotonic()
                # Dedup: suppress re-detection of same shot in overlapping 700ms window
                if (now_m - self._last_gun_wall_t < 0.3
                        and abs(ge.azimuth_deg - self._last_gun_az) < 30.0):
                    continue
                self._last_gun_wall_t = now_m
                self._last_gun_az = ge.azimuth_deg
                ge = self._cluster_gun_event(ge, now_m)
                if ge is not None:
                    self._queue.put(ge)

            # Noise gate: suppress gunshot/explosion transients before footstep detection
            gated_l = self._noise_gate.process(stereo[0])
            gated_r = self._noise_gate.process(stereo[1])
            gated_stereo = np.stack([gated_l, gated_r])

            # Spike beep detection (on raw mono -- beeps are tonal, not gated)
            for beep_t in self._spike_beep.process(mono):
                self._spike_timer.add_beep(beep_t)

            # Defuse hum detection (sustained tone, not a transient)
            self._defuse_sound.process(mono)

            # Footstep detection on gated audio (false positives suppressed)
            events: List[FootstepEvent] = self._footstep_det.process(gated_stereo)

            for event in events:
                finding = self._process_event(event, stereo)
                if finding is None:
                    continue
                now_m = time.monotonic()
                # Dedup: suppress re-detection of same onset in overlapping 700ms window
                # Deduplication gate: the same footstep onset can re-appear in the next
                # 700 ms read because windows overlap by ~680 ms. The 0.08 s / 25 deg
                # thresholds are tight enough to catch the same event re-detected but
                # loose enough to allow two distinct rapid steps from similar directions.
                # 25 deg matches half the width of one directional bucket (_az_to_word),
                # so a step on the sector boundary can shift slightly without being dropped.
                if (now_m - self._last_step_wall_t < 0.08
                        and abs(finding.azimuth_deg - self._last_step_az) < 25.0):
                    continue
                self._last_step_wall_t = now_m
                self._last_step_az = finding.azimuth_deg
                # Cadence: classify walking vs running from inter-step interval
                self._step_history.append((now_m, finding.azimuth_deg))
                cadence = self._detect_cadence(finding.azimuth_deg, now_m)
                if cadence:
                    finding.voice_text = finding.voice_text.replace(
                        " footstep at ", f" {cadence} footstep at ", 1
                    )
                self._queue.put(finding)

            time.sleep(0.02)   # 20 ms tick

    def _process_event(
        self, event: FootstepEvent, stereo: np.ndarray
    ) -> Optional[AudioFinding]:
        # Snapshot shared state once to avoid repeated lock acquisitions
        with self._state_lock:
            player_facing = self._player_facing
            player_pos    = self._player_pos
            map_name      = self._map_name

        # -- Direction
        az, dist_m = self._direction_est.estimate(stereo, event.amplitude_db)
        # Blend onset-frame stereo balance as a third directional cue.
        # stereo_balance (-1..+1) maps directly to ±90° azimuth - same sign convention.
        # Using the onset frame is more precise than the full 700ms window ILD.
        # 18% weight for stereo_balance (vs 82% for ITD+ILD): balance is a simple
        # broadband L/R ratio without the frequency selectivity of true ILD, so it is
        # less accurate for off-axis angles. It adds useful independent signal at
        # low amplitude where ITD correlation peaks are weaker, but should not dominate.
        balance_az = event.stereo_balance * 90.0
        az = float(np.clip(az * 0.82 + balance_az * 0.18, -180.0, 180.0))
        map_dir = audio_az_to_map_direction(az, player_facing)

        scale = _M_PER_UNIT.get(map_name.lower(), _DEFAULT_M_PER_UNIT)
        est_pos = (0.5, 0.5)
        if player_pos != (0.5, 0.5):
            est_pos = direction_to_map_pos(
                player_pos, map_dir, dist_m, scale_m_per_unit=scale
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
            # Narrow down by enemy team composition if configured
            tracker = self.enemy_agents   # thread-safe property
            narrowed = tracker.callout_for_shoe_type(shoe_type) if tracker else None
            if narrowed:
                shoe_display = narrowed          # e.g. "Breach" or "Breach or Brimstone"
            else:
                shoe_display = f"{shoe_type}-step enemy"

        # -- Zone name from estimated map position
        zone = ""
        if map_name != "unknown":
            zone = pos_to_zone(est_pos[0], est_pos[1], map_name)

        # -- Surface cross-check against map
        map_surface = get_surface(est_pos[0], est_pos[1], map_name)
        surface = event.surface
        surface_hint = ""
        if not surface_matches(surface, map_surface) and map_name != "unknown":
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


    def _cluster_gun_event(
        self, ge: GunEvent, now_m: float
    ) -> Optional[GunEvent]:
        """
        Group rapid shots from the same direction into a single burst callout.

        First shot in a sector: emitted normally.
        Second shot within _GUN_BURST_WINDOW_S: voice changed to "burst of fire".
        Further shots within _GUN_BURST_SUPPRESS_S after burst: suppressed (return None).
        """
        sector = _az_to_word(ge.azimuth_deg)
        entry = self._gun_burst.get(sector)

        if entry is None:
            self._gun_burst[sector] = (now_m, 1, 0.0)
            return ge

        last_t, count, suppress_until = entry

        if now_m < suppress_until:
            return None  # still within post-burst suppression window

        if now_m - last_t <= _GUN_BURST_WINDOW_S:
            count += 1
            if count >= 2:
                gun_word = "suppressed burst" if ge.suppressed else "burst of fire"
                ge = GunEvent(
                    time_sec=ge.time_sec,
                    azimuth_deg=ge.azimuth_deg,
                    suppressed=ge.suppressed,
                    amplitude_db=ge.amplitude_db,
                    distance_hint=ge.distance_hint,
                    voice=f"{gun_word} {sector}",
                )
                self._gun_burst[sector] = (now_m, count, now_m + _GUN_BURST_SUPPRESS_S)
            else:
                self._gun_burst[sector] = (now_m, count, 0.0)
        else:
            # Gap too large: start fresh burst window
            self._gun_burst[sector] = (now_m, 1, 0.0)

        return ge

    def _detect_cadence(self, az: float, now_m: float) -> Optional[str]:
        """
        Returns 'running' | 'walking' | None based on inter-step interval.

        Requires at least one prior step from the same direction (within 45°).
        Running: steps 250-550 ms apart (Valorant running cadence).
        Walking: steps 550-800 ms apart.
        Outside these ranges: insufficient data or too irregular.
        """
        recent = [
            (t, a) for t, a in self._step_history
            if abs(a - az) < 45.0 and t < now_m - 0.01
        ]
        if not recent:
            return None
        prev_t = recent[-1][0]
        interval = now_m - prev_t
        # Valorant running cadence measured from community recordings: steps land
        # roughly every 300-450 ms at full sprint, so 0.25-0.55 s covers the range
        # with margin. Walking (slow-walk, not shift-walk) produces steps ~600-750 ms
        # apart; 0.55-0.80 s captures this band. Intervals outside both ranges are
        # too irregular to classify confidently (e.g. start of movement, directional change).
        if 0.25 <= interval <= 0.55:
            return "running"
        if 0.55 < interval <= 0.80:
            return "walking"
        return None


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
