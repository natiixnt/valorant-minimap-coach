import queue
import threading
import time
from typing import Dict

import pyttsx3


class TTSEngine:
    def __init__(self, config: dict):
        cfg = config["audio"]
        self.cooldown: float = cfg["cooldown"]
        self._queue: queue.Queue = queue.Queue()
        self._last_spoken: Dict[str, float] = {}
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", cfg["rate"])
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text: str, priority: bool = False) -> None:
        now = time.time()
        if now - self._last_spoken.get(text, 0) < self.cooldown:
            return
        self._last_spoken[text] = now
        if priority:
            # Drain queue so urgent callouts aren't delayed by queued ones
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        self._queue.put(text)

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
