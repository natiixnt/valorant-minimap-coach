#!/usr/bin/env python3
import signal
import time

import yaml

from src.audio.tts import TTSEngine
from src.capture.screen import ScreenCapture
from src.maps.callouts import enemies_to_callout
from src.vision.ai_analyzer import AIAnalyzer
from src.vision.detector import MinimapDetector
from src.vision.map_detector import MapDetector


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Coach:
    def __init__(self, config: dict):
        self.config = config
        self.capture = ScreenCapture(config)
        self.detector = MinimapDetector(config)
        self.tts = TTSEngine(config)
        self.ai = AIAnalyzer(config) if config["ai"]["enabled"] else None
        # map_override lets the user force a map in config.yaml; null = auto-detect
        self._map_override: str | None = config.get("map_override")
        self.map_detector = MapDetector(config) if not self._map_override else None
        self.map_name: str = self._map_override or "unknown"
        self._prev_enemy_count = 0
        self._running = True

    def _startup_map_detection(self) -> None:
        if self._map_override:
            print(f"[Coach] Map override: {self._map_override}")
            return
        print("[Coach] Detecting map from screen...")
        self.tts.speak("Detecting map", priority=False)
        detected = self.map_detector.wait_for_map()
        self.map_name = detected
        self.tts.speak(f"Map detected: {detected}", priority=False)

    def _refresh_map(self) -> None:
        if self._map_override or not self.map_detector:
            return
        new_map = self.map_detector.get_map()
        if new_map and new_map != self.map_name:
            self.map_name = new_map
            print(f"[Coach] Map changed to: {new_map}")
            self.tts.speak(f"New map: {new_map}", priority=False)

    def run(self) -> None:
        self._startup_map_detection()
        print(f"[Coach] Running. Map: {self.map_name}. Ctrl+C to stop.")
        self.tts.speak("Coach active", priority=False)

        while self._running:
            frame = self.capture.capture()
            result = self.detector.detect(frame)

            # Fast CV path: fire immediately when new enemies appear on minimap
            if result.enemy_count > self._prev_enemy_count:
                callout = enemies_to_callout(result.enemies, self.map_name)
                if callout:
                    self.tts.speak(callout, priority=True)

            self._prev_enemy_count = result.enemy_count

            # AI path: periodic deeper analysis while enemies are visible
            if self.ai and result.enemy_count > 0 and self.ai.should_analyze():
                ai_callout = self.ai.analyze(frame, result.enemy_count)
                if ai_callout:
                    self.tts.speak(ai_callout, priority=False)

            # Periodic map re-check (catches new games without restarting)
            self._refresh_map()

            time.sleep(0.1)

        self._shutdown()

    def _shutdown(self) -> None:
        self.tts.stop()
        self.capture.close()
        print("[Coach] Stopped.")


def main() -> None:
    config = load_config()
    coach = Coach(config)

    def handle_signal(sig, frame):
        coach._running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    coach.run()


if __name__ == "__main__":
    main()
