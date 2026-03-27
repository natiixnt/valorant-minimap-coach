import json
import os
import queue
import threading
import time
from typing import Dict, List, Optional

import pyttsx3

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "user_settings.json")

# Sentinel to unblock the worker on stop()
_STOP = object()


class TTSEngine:
    def __init__(self, config: dict):
        cfg = config["audio"]
        self.cooldown: float = cfg["cooldown"]
        self._queue: queue.Queue = queue.Queue()
        self._last_spoken: Dict[str, float] = {}
        self._muted: bool = False

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", cfg["rate"])

        # Load saved voice -- done here in main thread before worker starts
        try:
            with open(_SETTINGS_PATH) as f:
                saved = json.load(f)
            if "voice_id" in saved:
                self._engine.setProperty("voice", saved["voice_id"])
        except Exception:
            pass

        # Pending voice change: set from main thread, applied by worker before next say()
        self._pending_voice: Optional[str] = None
        self._voice_lock = threading.Lock()

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
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        self._queue.put(text)

    def list_voices(self) -> List[dict]:
        # Safe to call from main thread -- only reads, not modifying engine state
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
        with self._voice_lock:
            pending = self._pending_voice
        if pending is not None:
            return pending
        return self._engine.getProperty("voice") or ""

    def set_voice(self, voice_id: str) -> None:
        # Queue the change for the worker thread to apply before next say()
        with self._voice_lock:
            self._pending_voice = voice_id

    def preview(self, voice_id: str) -> None:
        self.set_voice(voice_id)
        self.speak("Enemy spotted, B site", priority=True)

    def _worker(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is _STOP:
                break
            # Apply pending voice change in worker thread before speaking
            with self._voice_lock:
                voice = self._pending_voice
                self._pending_voice = None
            if voice:
                self._engine.setProperty("voice", voice)
            try:
                self._engine.say(item)
                self._engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Engine error: {e}")

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=3)
