import time
from typing import TYPE_CHECKING, List, Optional

import yaml

from src.audio.tts import TTSEngine
from src.audio.audio_coach import AudioCoach
from src.capture.screen import ScreenCapture
from src.game.economy import EconomyTracker
from src.game.heatmap import Heatmap
from src.game.play_detector import PlayDetector
from src.game.retake_advisor import RetakeAdvisor
from src.game.round_state import RoundState, State
from src.maps.callouts import enemies_to_callout, pos_to_zone, stack_callout
from src.telemetry.collector import DataCollector
from src.vision.ability_detector import AbilityDetector
from src.vision.ai_analyzer import AIAnalyzer
from src.vision.detector import MinimapDetector
from src.vision.map_detector import MapDetector
from src.vision.player_angle import PlayerAngleDetector
from src.vision.spike_detector import SpikeDetector
from src.vision.team_detector import TeamDetector

if TYPE_CHECKING:
    from src.ui.overlay import OverlayWindow

try:
    from src.vision.local_analyzer import LocalAnalyzer
    _HAS_LOCAL = True
except ImportError:
    _HAS_LOCAL = False


def load_config(path: str = "config.yaml") -> dict:
    import os, sys
    if getattr(sys, "frozen", False) and not os.path.exists(path):
        import shutil
        bundled = os.path.join(sys._MEIPASS, path)  # type: ignore[attr-defined]
        if os.path.exists(bundled):
            shutil.copy(bundled, path)
            print(f"[Config] Extracted {path} next to executable")
    with open(path) as f:
        return yaml.safe_load(f)


class Coach:
    def __init__(self, config: dict) -> None:
        self.config = config

        # Core CV + audio
        self.capture              = ScreenCapture(config)
        self.detector             = MinimapDetector(config)
        self.team_detector        = TeamDetector(config)
        self.ability_detector     = AbilityDetector(config)
        self.spike_detector       = SpikeDetector()
        self.player_angle_detector = PlayerAngleDetector()
        self.tts                  = TTSEngine(config)
        self.audio_coach          = AudioCoach(config)

        # Game intelligence
        self.round_state    = RoundState()
        self.heatmap        = Heatmap()
        self.play_detector  = PlayDetector()
        self.retake_advisor = RetakeAdvisor()
        self.economy        = EconomyTracker()

        # AI + telemetry
        self.collector = DataCollector(config)
        if config["ai"]["enabled"]:
            use_local = config["ai"].get("use_local_model") and _HAS_LOCAL
            self.ai = (LocalAnalyzer(config, collector=self.collector)  # type: ignore[name-defined]
                       if use_local else AIAnalyzer(config, collector=self.collector))
        else:
            self.ai = None

        # Map
        self._map_override: Optional[str] = config.get("map_override")
        self.map_detector = MapDetector(config, collector=self.collector) if not self._map_override else None
        self.map_name: str = self._map_override or "unknown"

        # State
        self._prev_enemy_count  = 0
        self._prev_team         : List = []
        self._callout_lang      = "EN"
        self._stack_frames: dict = {}
        self._stack_warned: set  = set()
        self._spike_active       = False
        self._spike_plant_time   = 0.0
        self._recent_ai_callouts: List[str] = []
        self._running            = True
        self._overlay: Optional["OverlayWindow"] = None

        # Wire round audio callbacks
        self.audio_coach.round_audio.on_round_start = self._on_round_start_audio
        self.audio_coach.round_audio.on_round_end   = self._on_round_end_audio

    # ------------------------------------------------------------------
    def set_overlay(self, overlay: "OverlayWindow") -> None:
        self._overlay = overlay
        overlay.on_mute_change = self.tts.set_muted
        overlay.on_feedback    = self.collector.submit_feedback

    def set_callout_lang(self, lang: str) -> None:
        self._callout_lang = lang

    def _ui(self, fn, *args) -> None:
        if self._overlay:
            self._overlay.after(0, lambda: fn(*args))

    # ------------------------------------------------------------------
    # Round audio callbacks (fire from AudioCoach background thread)
    # ------------------------------------------------------------------

    def _on_round_start_audio(self) -> None:
        self.round_state.on_round_start_sound()
        self.retake_advisor.reset()
        print(f"[Coach] Round {self.round_state.round_num} start (audio)")

    def _on_round_end_audio(self) -> None:
        prev_round = self.round_state.round_num
        self.round_state.on_round_end_sound()
        self.heatmap.end_round(self.round_state.round_num)
        # Report hot zones summary at round boundary
        hot = self.heatmap.hottest_zones(2)
        if hot:
            summary = self.heatmap.summary()
            self.tts.speak(summary, priority=False)

    # ------------------------------------------------------------------
    # Map detection
    # ------------------------------------------------------------------

    def _startup_map_detection(self) -> None:
        if self._map_override:
            print(f"[Coach] Map override: {self._map_override}")
            self._ui(self._overlay.update_map, self._map_override)  # type: ignore[union-attr]
            return
        print("[Coach] Detecting map from screen...")
        detected = self.map_detector.wait_for_map()  # type: ignore[union-attr]
        self.map_name = detected
        self._ui(self._overlay.update_map, detected)  # type: ignore[union-attr]
        self.tts.speak(f"Map detected: {detected}")

    def _refresh_map(self) -> None:
        if self._map_override or not self.map_detector:
            return
        new_map = self.map_detector.get_map()
        if new_map and new_map != self.map_name:
            self.map_name = new_map
            print(f"[Coach] Map changed: {new_map}")
            self._ui(self._overlay.update_map, new_map)  # type: ignore[union-attr]
            self.tts.speak(f"New map: {new_map}")
            self.audio_coach.map_name = new_map

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._startup_map_detection()
        self.audio_coach.map_name = self.map_name
        self.audio_coach.start()
        print(f"[Coach] Running. Map: {self.map_name}. Ctrl+C to stop.")
        self.tts.speak("Coach active")

        while self._running:
            frame  = self.capture.capture()
            result = self.detector.detect(frame)
            team   = self.team_detector.detect(frame)
            appeared, gone = self.ability_detector.update(frame)

            # -- Round state machine (tick)
            spike_planted = self.spike_detector.is_planted
            rs_event = self.round_state.update(result.enemy_count, spike_planted)
            if rs_event == "round_end":
                self.heatmap.end_round(self.round_state.round_num)
                econ_status = self.economy.status()
                self.tts.speak(econ_status.voice, priority=False)

            # -- Heatmap: record all visible enemy zones this tick
            for x, y in result.enemies:
                zone = pos_to_zone(x, y, self.map_name)
                self.heatmap.add_sighting(zone, self.round_state.round_num)

            # -- Play pattern detection
            play_event = self.play_detector.update(result.enemies, self.map_name)
            if play_event:
                self.tts.speak(play_event.voice, priority=True)
                self._ui(self._overlay.update_callout, play_event.voice)  # type: ignore[union-attr]

            # -- UI: enemies + team
            self._ui(
                self._overlay.update_enemies,  # type: ignore[union-attr]
                result.enemy_count,
                result.enemies,
            )

            # -- UI: active utility
            active_abs = [
                {"display": sig["display"], "color": sig["color"], "position": (0.5, 0.5)}
                for sig in self.ability_detector.active.values()
            ]
            self._ui(self._overlay.update_utility, active_abs)  # type: ignore[union-attr]

            # -- Player facing angle (feeds audio direction estimator)
            player_angle = self.player_angle_detector.detect(frame)
            if player_angle is not None:
                self.audio_coach.player_facing = player_angle

            # -- Player position from team dots (more accurate than mirroring enemies)
            if team:
                # Our position: centroid of team dots shifted slightly toward center
                avg_tx = sum(x for x, y in team) / len(team)
                avg_ty = sum(y for x, y in team) / len(team)
                self.audio_coach.player_pos = (
                    max(0.05, min(0.95, avg_tx)),
                    max(0.05, min(0.95, avg_ty)),
                )
            elif result.enemies:
                avg_ex = sum(x for x, y in result.enemies) / len(result.enemies)
                avg_ey = sum(y for x, y in result.enemies) / len(result.enemies)
                self.audio_coach.player_pos = (
                    max(0.1, min(0.9, 1.0 - avg_ex)),
                    max(0.1, min(0.9, 1.0 - avg_ey)),
                )

            # -- Spike planted
            spike_pos = self.spike_detector.update(frame)
            if spike_pos is not None:
                zone = pos_to_zone(spike_pos[0], spike_pos[1], self.map_name)
                spike_text = f"Spike planted at {zone}! Rotate!"
                self.tts.speak(spike_text, priority=True)
                self._ui(self._overlay.update_callout, spike_text)  # type: ignore[union-attr]
                self._spike_active    = True
                self._spike_plant_time = time.monotonic()
                self.round_state.on_spike_planted()
                self.economy.on_spike_planted()

            # -- Retake advisor (only in post-plant)
            if self._spike_active and self.spike_detector.planted_pos and team:
                time_since_plant = time.monotonic() - self._spike_plant_time
                advice = self.retake_advisor.advise(
                    self.spike_detector.planted_pos,
                    team,
                    self.map_name,
                    time_since_plant,
                )
                if advice:
                    self.tts.speak(advice, priority=True)
                    self._ui(self._overlay.update_callout, advice)  # type: ignore[union-attr]

            if not self.spike_detector.is_planted:
                self._spike_active = False

            # -- New enemies spotted
            if result.enemy_count > self._prev_enemy_count:
                callout = enemies_to_callout(result.enemies, self.map_name, self._callout_lang)
                if callout:
                    self.tts.speak(callout, priority=True)
                    self._ui(self._overlay.update_callout, callout)  # type: ignore[union-attr]

            # -- Site clear
            if self._prev_enemy_count >= 2 and result.enemy_count == 0:
                self.tts.speak("Site clear", priority=False)
                self._ui(self._overlay.update_callout, "Site clear")  # type: ignore[union-attr]

            self._prev_enemy_count = result.enemy_count
            self._prev_team        = team

            # -- Stack detection
            zone_counts: dict = {}
            for x, y in result.enemies:
                z = pos_to_zone(x, y, self.map_name)
                zone_counts[z] = zone_counts.get(z, 0) + 1
            for z, cnt in zone_counts.items():
                if cnt >= 3:
                    self._stack_frames[z] = self._stack_frames.get(z, 0) + 1
                    if self._stack_frames[z] == 3 and z not in self._stack_warned:
                        self._stack_warned.add(z)
                        stack_text = stack_callout(z, cnt, self.map_name, self._callout_lang)
                        self.tts.speak(stack_text, priority=True)
                        self._ui(self._overlay.update_callout, stack_text)  # type: ignore[union-attr]
                else:
                    self._stack_frames.pop(z, None)
                    self._stack_warned.discard(z)
            for z in list(self._stack_frames):
                if z not in zone_counts:
                    self._stack_frames.pop(z, None)
                    self._stack_warned.discard(z)

            # -- Ability callouts
            for ab in appeared:
                zone = pos_to_zone(ab.position[0], ab.position[1], self.map_name)
                text = ab.voice.format(zone=zone)
                self.tts.speak(text, priority=False)
                self._ui(self._overlay.update_callout, text)  # type: ignore[union-attr]

            # -- AI deep analysis
            if self.ai and result.enemy_count > 0 and self.ai.should_analyze():
                active_kinds = list(self.ability_detector.active.keys())
                ai_callout = self.ai.analyze(
                    frame, result.enemy_count, self.map_name, active_kinds,
                    spike_active=self._spike_active,
                    recent_callouts=self._recent_ai_callouts,
                )
                if ai_callout:
                    self.tts.speak(ai_callout)
                    sample_ts = self.collector.submit_minimap_callout(
                        frame.data, ai_callout, self.map_name, result.enemies,
                        spike_active=self._spike_active,
                        recent_callouts=self._recent_ai_callouts,
                    )
                    self._ui(self._overlay.update_ai, ai_callout, sample_ts)  # type: ignore[union-attr]
                    self._recent_ai_callouts = (self._recent_ai_callouts + [ai_callout])[-5:]

            # -- Pro audio: footstep zone callouts
            for finding in self.audio_coach.poll():
                self.tts.speak(finding.voice_text, priority=False)
                self._ui(self._overlay.update_callout, finding.voice_text)  # type: ignore[union-attr]
                if finding.zone and finding.audio_clip is not None:
                    self.collector.submit_footstep_audio(
                        finding.audio_clip,
                        finding.zone,
                        finding.agent,
                        self.map_name,
                        finding.surface,
                    )

            self._refresh_map()
            time.sleep(0.1)

        self._shutdown()

    def _shutdown(self) -> None:
        self.audio_coach.stop()
        self.tts.stop()
        self.capture.close()
        print("[Coach] Stopped.")
