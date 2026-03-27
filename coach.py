import time
from typing import TYPE_CHECKING, Optional

import yaml

from src.audio.tts import TTSEngine
from src.capture.screen import ScreenCapture
from src.maps.callouts import enemies_to_callout, pos_to_zone
from src.vision.ability_detector import AbilityDetector
from src.vision.ai_analyzer import AIAnalyzer
from src.vision.detector import MinimapDetector
from src.vision.map_detector import MapDetector

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
        self._prev_enemy_count = 0
        self._running = True
        # Set after construction, before run()
        self._overlay: Optional["OverlayWindow"] = None

    def set_overlay(self, overlay: "OverlayWindow") -> None:
        self._overlay = overlay
        overlay.on_mute_change = self.tts.set_muted

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

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        self._startup_map_detection()
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

            # Fast CV path: new enemies
            callout: Optional[str] = None
            if result.enemy_count > self._prev_enemy_count:
                callout = enemies_to_callout(result.enemies, self.map_name)
                if callout:
                    self.tts.speak(callout, priority=True)
                    self._ui(self._overlay.update_callout, callout)  # type: ignore[union-attr]

            self._prev_enemy_count = result.enemy_count

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

            self._refresh_map()
            time.sleep(0.1)

        self._shutdown()

    def _shutdown(self) -> None:
        self.tts.stop()
        self.capture.close()
        print("[Coach] Stopped.")
