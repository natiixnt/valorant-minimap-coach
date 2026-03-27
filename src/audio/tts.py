import json
import os
import queue
import threading
import time
from typing import Dict, List

import pyttsx3

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "user_settings.json")


class TTSEngine:
    def __init__(self, config: dict):
        cfg = config["audio"]
        self.cooldown: float = cfg["cooldown"]
        self._queue: queue.Queue = queue.Queue()
        self._last_spoken: Dict[str, float] = {}
        self._muted: bool = False
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", cfg["rate"])
        # Load saved voice
        try:
            with open(_SETTINGS_PATH) as f:
                saved = json.load(f)
            if "voice_id" in saved:
                self._engine.setProperty("voice", saved["voice_id"])
        except Exception:
            pass
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def speak(self, text: str, priority: bool = False) -> None:
        if self._muted:
            return
        now = time.time()
        if now - self._last_spoken.get(text, 0) < self.cooldown:
            return
        self._last_spoken[text] = now
        if priority:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        self._queue.put(text)

    def list_voices(self) -> List[dict]:
        voices = self._engine.getProperty("voices")
        return [
            {
                "name": v.name,
                "id":   v.id,
                "lang": v.languages[0] if v.languages else "",
            }
            for v in voices
        ]

    @property
    def current_voice_id(self) -> str:
        return self._engine.getProperty("voice") or ""

    def set_voice(self, voice_id: str) -> None:
        self._engine.setProperty("voice", voice_id)

    def preview(self, voice_id: str) -> None:
        self.set_voice(voice_id)
        self.speak("Enemy spotted, B site", priority=True)

    def _worker(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                self._engine.say(text)
                self._engine.runAndWait()
            except queue.Empty:
                continue

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
