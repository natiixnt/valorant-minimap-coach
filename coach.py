import time
from typing import TYPE_CHECKING, Optional

import yaml

from src.audio.tts import TTSEngine
from src.audio.audio_coach import AudioCoach
from src.capture.screen import ScreenCapture
from src.maps.callouts import enemies_to_callout, pos_to_zone, stack_callout
from src.vision.ability_detector import AbilityDetector
from src.vision.ai_analyzer import AIAnalyzer
from src.vision.detector import MinimapDetector
from src.vision.map_detector import MapDetector
from src.vision.player_angle import PlayerAngleDetector
from src.vision.spike_detector import SpikeDetector

if TYPE_CHECKING:
    from src.ui.overlay import OverlayWindow


def load_config(path: str = "config.yaml") -> dict:
    import os, sys
    # When running from a PyInstaller onefile bundle, resolve config relative to exe
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
        self.capture = ScreenCapture(config)
        self.detector = MinimapDetector(config)
        self.ability_detector = AbilityDetector(config)
        self.tts = TTSEngine(config)
        self.ai = AIAnalyzer(config) if config["ai"]["enabled"] else None
        self._map_override: Optional[str] = config.get("map_override")
        self.map_detector = MapDetector(config) if not self._map_override else None
        self.map_name: str = self._map_override or "unknown"
        self.spike_detector = SpikeDetector()
        self.player_angle_detector = PlayerAngleDetector()
        self.audio_coach = AudioCoach(config)
        self._prev_enemy_count = 0
        self._callout_lang: str = "EN"
        # zone -> consecutive frames with 3+ enemies in that zone
        self._stack_frames: dict = {}
        self._stack_warned: set = set()
        self._running = True
        # Set after construction, before run()
        self._overlay: Optional["OverlayWindow"] = None

    def set_overlay(self, overlay: "OverlayWindow") -> None:
        self._overlay = overlay
        overlay.on_mute_change = self.tts.set_muted

    def set_callout_lang(self, lang: str) -> None:
        self._callout_lang = lang

    def _ui(self, fn, *args) -> None:
        """Dispatch a UI update safely from any thread."""
        if self._overlay:
            self._overlay.after(0, lambda: fn(*args))

    # -----------------------------------------------------------------------
    # Map detection
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        self._startup_map_detection()
        self.audio_coach.map_name = self.map_name
        self.audio_coach.start()
        print(f"[Coach] Running. Map: {self.map_name}. Ctrl+C to stop.")
        self.tts.speak("Coach active")

        while self._running:
            frame = self.capture.capture()
            result = self.detector.detect(frame)
            appeared, gone = self.ability_detector.update(frame)

            # Update overlay: enemies
            self._ui(
                self._overlay.update_enemies,  # type: ignore[union-attr]
                result.enemy_count,
                result.enemies,
            )

            # Update overlay: active utility items
            active_abs = [
                {"display": sig["display"], "color": sig["color"],
                 "position": (0.5, 0.5)}  # position unknown at this level
                for sig in self.ability_detector.active.values()
            ]
            self._ui(self._overlay.update_utility, active_abs)  # type: ignore[union-attr]

            # Player facing angle for audio direction fusion
            player_angle = self.player_angle_detector.detect(frame)
            if player_angle is not None:
                self.audio_coach.player_facing = player_angle

            # Player position: use minimap center as best guess when no teammates tracked
            if result.enemies:
                # Rough player pos: opposite side of minimap from most enemies
                avg_ex = sum(x for x, y in result.enemies) / len(result.enemies)
                avg_ey = sum(y for x, y in result.enemies) / len(result.enemies)
                px = 1.0 - avg_ex
                py = 1.0 - avg_ey
                self.audio_coach.player_pos = (
                    max(0.1, min(0.9, px)),
                    max(0.1, min(0.9, py)),
                )

            # Fast CV path: spike planted
            spike_pos = self.spike_detector.update(frame)
            if spike_pos is not None:
                zone = pos_to_zone(spike_pos[0], spike_pos[1], self.map_name)
                spike_text = f"Spike planted at {zone}! Rotate!"
                self.tts.speak(spike_text, priority=True)
                self._ui(self._overlay.update_callout, spike_text)  # type: ignore[union-attr]

            # Fast CV path: new enemies spotted
            callout: Optional[str] = None
            if result.enemy_count > self._prev_enemy_count:
                callout = enemies_to_callout(result.enemies, self.map_name, self._callout_lang)
                if callout:
                    self.tts.speak(callout, priority=True)
                    self._ui(self._overlay.update_callout, callout)  # type: ignore[union-attr]

            # Enemies cleared callout
            if self._prev_enemy_count >= 2 and result.enemy_count == 0:
                cleared = "Site clear"
                self.tts.speak(cleared, priority=False)
                self._ui(self._overlay.update_callout, cleared)  # type: ignore[union-attr]

            self._prev_enemy_count = result.enemy_count

            # Stack detection: 3+ enemies in the same zone for 3 consecutive frames
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
            # Clear zones that are no longer visible
            for z in list(self._stack_frames):
                if z not in zone_counts:
                    self._stack_frames.pop(z, None)
                    self._stack_warned.discard(z)

            # Fast CV path: newly appeared abilities
            for ab in appeared:
                zone = pos_to_zone(ab.position[0], ab.position[1], self.map_name)
                text = ab.voice.format(zone=zone)
                self.tts.speak(text, priority=False)
                self._ui(self._overlay.update_callout, text)  # type: ignore[union-attr]

            # AI path: periodic deep analysis with ability context
            if self.ai and result.enemy_count > 0 and self.ai.should_analyze():
                active_kinds = list(self.ability_detector.active.keys())
                ai_callout = self.ai.analyze(
                    frame, result.enemy_count, self.map_name, active_kinds
                )
                if ai_callout:
                    self.tts.speak(ai_callout)
                    self._ui(self._overlay.update_ai, ai_callout)  # type: ignore[union-attr]

            # Pro audio path: footstep direction + agent identification
            for finding in self.audio_coach.poll():
                self.tts.speak(finding.voice_text, priority=False)
                self._ui(self._overlay.update_callout, finding.voice_text)  # type: ignore[union-attr]

            self._refresh_map()
            time.sleep(0.1)

        self._shutdown()

    def _shutdown(self) -> None:
        self.audio_coach.stop()
        self.tts.stop()
        self.capture.close()
        print("[Coach] Stopped.")
