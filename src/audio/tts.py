import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pyttsx3

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "user_settings.json")

_STOP = object()

# Max non-priority items allowed in queue at once.
# If full, oldest is dropped before adding the new one.
_MAX_QUEUE_DEPTH = 3


@dataclass(order=True)
class _Item:
    # Lower priority_key = spoken first (0 = urgent, 1 = normal)
    priority_key: int
    queued_at: float = field(compare=False)
    ttl: float       = field(compare=False)   # seconds before item is considered stale
    text: str        = field(compare=False)

    def is_stale(self) -> bool:
        return time.monotonic() - self.queued_at > self.ttl


class TTSEngine:
    """
    TTS with staleness-aware queue.

    speak(text, priority=False, ttl=2.5)
      - priority=True  : flush queue, speak immediately, TTL=10s
      - priority=False : enqueue with TTL; if queue depth >= _MAX_QUEUE_DEPTH,
                         drop the oldest non-priority item first

    Worker checks item.is_stale() before each say(). Stale items are silently
    dropped -- prevents speaking "enemy A" 5 seconds after the situation changed.

    Default TTL per callout category (pass explicitly from coach.py):
      - spike / defuse / gunshot : priority=True  (TTL 10s, flushes queue)
      - enemy spotted / zones    : ttl=2.5s
      - footsteps                : ttl=1.5s
      - economy / ult / heatmap  : ttl=12.0s  (informational, survives a few speaks)
    """

    def __init__(self, config: dict):
        cfg = config["audio"]
        self.cooldown: float = cfg["cooldown"]
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._queue_depth = 0          # count of non-priority items currently enqueued
        self._depth_lock  = threading.Lock()
        self._last_spoken: Dict[str, float] = {}
        self._muted: bool = False

        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", cfg["rate"])
        except Exception as e:
            print(f"[TTS] Engine init failed: {e}. Voice callouts disabled.")
            self._engine = None

        if self._engine:
            try:
                with open(_SETTINGS_PATH) as f:
                    saved = json.load(f)
                if "voice_id" in saved:
                    self._engine.setProperty("voice", saved["voice_id"])
            except Exception:
                pass

        self._pending_voice: Optional[str] = None
        self._voice_lock = threading.Lock()

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def speak(self, text: str, priority: bool = False, ttl: float = 2.5) -> None:
        if self._muted:
            return
        now_wall = time.time()
        if now_wall - self._last_spoken.get(text, 0) < self.cooldown:
            return
        self._last_spoken[text] = now_wall

        if priority:
            # Drain queue entirely, then enqueue with long TTL
            self._drain_queue()
            item = _Item(
                priority_key=0,
                queued_at=time.monotonic(),
                ttl=10.0,
                text=text,
            )
            self._queue.put(item)
        else:
            with self._depth_lock:
                if self._queue_depth >= _MAX_QUEUE_DEPTH:
                    # Drop oldest item by draining and re-enqueueing all but first
                    self._drop_oldest_nonpriority()
                self._queue_depth += 1

            item = _Item(
                priority_key=1,
                queued_at=time.monotonic(),
                ttl=ttl,
                text=text,
            )
            self._queue.put(item)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._depth_lock:
            self._queue_depth = 0

    def _drop_oldest_nonpriority(self) -> None:
        """Pull all items, discard the one that has been waiting longest, re-enqueue rest."""
        items: List[_Item] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break

        # Find oldest non-priority item (priority_key == 1)
        oldest_idx = None
        oldest_t = float("inf")
        for i, it in enumerate(items):
            if it.priority_key == 1 and it.queued_at < oldest_t:
                oldest_t = it.queued_at
                oldest_idx = i

        if oldest_idx is not None:
            items.pop(oldest_idx)
            self._queue_depth = max(0, self._queue_depth - 1)

        for it in items:
            self._queue.put(it)

    def list_voices(self) -> List[dict]:
        if not self._engine:
            return []
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
        if not self._engine:
            return ""
        return self._engine.getProperty("voice") or ""

    def set_voice(self, voice_id: str) -> None:
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

            with self._depth_lock:
                if item.priority_key == 1:
                    self._queue_depth = max(0, self._queue_depth - 1)

            # Drop stale items silently
            if isinstance(item, _Item) and item.is_stale():
                continue

            if not self._engine:
                continue
            with self._voice_lock:
                voice = self._pending_voice
                self._pending_voice = None
            if voice:
                self._engine.setProperty("voice", voice)
            try:
                self._engine.say(item.text)
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
