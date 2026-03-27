#!/usr/bin/env python3
import signal
import sys
import time

import yaml

from src.audio.tts import TTSEngine
from src.capture.screen import ScreenCapture
from src.maps.callouts import enemies_to_callout
from src.vision.ai_analyzer import AIAnalyzer
from src.vision.detector import MinimapDetector


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Coach:
    def __init__(self, config: dict):
        self.map_name: str = config.get("map", "unknown")
        self.capture = ScreenCapture(config)
        self.detector = MinimapDetector(config)
        self.tts = TTSEngine(config)
        self.ai = AIAnalyzer(config) if config["ai"]["enabled"] else None
        self._prev_enemy_count = 0
        self._running = True

    def run(self) -> None:
        print(f"[Coach] Running on map: {self.map_name}. Ctrl+C to stop.")
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
