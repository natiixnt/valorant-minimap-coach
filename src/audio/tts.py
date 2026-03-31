import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
    dropped - prevents speaking "enemy A" 5 seconds after the situation changed.

    _last_spoken is updated only when an item is ACTUALLY SPOKEN (in the worker),
    not when it is queued. This prevents stale-dropped items from locking out
    the same callout via cooldown even though it was never heard.

    _pending_texts tracks what is currently in the queue to prevent double-queuing
    the same text before the worker has a chance to process it.

    Default TTL per callout category (pass explicitly from coach.py):
      - spike / rush / execute / split : priority=True  (TTL 10s, flushes queue)
      - lurk / mid_ctrl               : ttl=4.0s
      - enemy spotted                 : priority=True
      - zone transitions              : ttl=2.0s
      - trajectory prediction         : ttl=1.5s  (matches prediction window)
      - ability                       : ttl=4.0s
      - footsteps                     : ttl=2.0s
      - gunshot                       : ttl=1.5s
      - site clear                    : ttl=4.0s
      - economy / ult / heatmap       : ttl=12.0s  (informational, survives a few speaks)
    """

    def __init__(self, config: dict):
        cfg = config["audio"]
        self.cooldown: float = cfg["cooldown"]
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._queue_depth = 0          # count of non-priority items currently enqueued
        self._depth_lock  = threading.Lock()

        # _last_spoken: text -> wall time when SPOKEN (updated in worker, not in speak())
        # _pending_texts: set of texts currently sitting in queue (dedup guard)
        # _spoken_lock: guards both dicts from concurrent access between speak() and worker
        self._last_spoken: Dict[str, float] = {}
        self._pending_texts: Set[str] = set()
        self._spoken_lock = threading.Lock()

        # Token bucket: limits non-priority callout burst rate.
        # Capacity=2, refill=1 token per 4s -> max ~15 non-priority speaks/min sustained.
        # Priority calls bypass the bucket entirely.
        self._bucket_tokens: float = 2.0
        self._bucket_last_refill: float = time.monotonic()
        self._bucket_lock = threading.Lock()
        _BUCKET_CAPACITY    = 2.0
        _BUCKET_REFILL_RATE = 1.0 / 4.0  # tokens/second
        self._BUCKET_CAPACITY    = _BUCKET_CAPACITY
        self._BUCKET_REFILL_RATE = _BUCKET_REFILL_RATE

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
                if "tts_volume" in saved:
                    self._engine.setProperty("volume", float(saved["tts_volume"]))
            except Exception:
                pass

        self._pending_voice: Optional[str] = None
        self._voice_lock = threading.Lock()

        self._pending_volume: Optional[float] = None
        self._volume_lock = threading.Lock()

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def set_volume(self, vol: float) -> None:
        """Set TTS volume (0.0 = silent, 1.0 = full). Applied before next utterance."""
        with self._volume_lock:
            self._pending_volume = max(0.0, min(1.0, vol))

    def speak(self, text: str, priority: bool = False, ttl: float = 2.5) -> None:
        if self._muted:
            return

        if priority:
            # Priority callouts bypass cooldown - they flush the queue and speak now.
            # Drain queue first (also clears _pending_texts)
            self._drain_queue()
            # Add to pending, then enqueue
            with self._spoken_lock:
                self._pending_texts.add(text)
            item = _Item(
                priority_key=0,
                queued_at=time.monotonic(),
                ttl=10.0,
                text=text,
            )
            self._queue.put(item)
        else:
            # Token bucket rate limit: caps non-priority burst without silencing priority
            with self._bucket_lock:
                now_m = time.monotonic()
                elapsed = now_m - self._bucket_last_refill
                self._bucket_tokens = min(
                    self._BUCKET_CAPACITY,
                    self._bucket_tokens + elapsed * self._BUCKET_REFILL_RATE,
                )
                self._bucket_last_refill = now_m
                if self._bucket_tokens < 1.0:
                    return  # rate-limited: too many callouts in short window
                self._bucket_tokens -= 1.0

            with self._spoken_lock:
                # Already in queue: skip (dedup)
                if text in self._pending_texts:
                    return
                # Cooldown based on last actual speak time
                now_wall = time.time()
                if now_wall - self._last_spoken.get(text, 0) < self.cooldown:
                    return
                self._pending_texts.add(text)

            with self._depth_lock:
                if self._queue_depth >= _MAX_QUEUE_DEPTH:
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
        # All pending items are gone - clear the dedup set
        with self._spoken_lock:
            self._pending_texts.clear()

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
            dropped = items.pop(oldest_idx)
            with self._spoken_lock:
                self._pending_texts.discard(dropped.text)
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
        # Clear cooldown so repeated preview clicks always work
        with self._spoken_lock:
            self._last_spoken.pop("Enemy spotted, B site", None)
        self.speak("Enemy spotted, B site", priority=True)

    def _worker(self) -> None:
        while self._running:
            try:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is _STOP:
                    break

                with self._depth_lock:
                    if item.priority_key == 1:
                        self._queue_depth = max(0, self._queue_depth - 1)

                # Always remove from pending set when dequeued (spoken or stale)
                with self._spoken_lock:
                    self._pending_texts.discard(item.text)

                # Drop stale items silently - do NOT update _last_spoken.
                # This allows the same callout to be re-queued immediately next tick
                # if the situation is still relevant.
                if isinstance(item, _Item) and item.is_stale():
                    continue

                if not self._engine:
                    continue

                with self._volume_lock:
                    vol = self._pending_volume
                    self._pending_volume = None
                if vol is not None:
                    self._engine.setProperty("volume", vol)

                with self._voice_lock:
                    voice = self._pending_voice
                    self._pending_voice = None
                if voice:
                    self._engine.setProperty("voice", voice)
                try:
                    self._engine.say(item.text)
                    self._engine.runAndWait()
                    # Cooldown runs from when item was actually SPOKEN, not queued
                    with self._spoken_lock:
                        self._last_spoken[item.text] = time.time()
                except Exception as e:
                    print(f"[TTS] Engine error: {e}")
            except Exception as e:
                print(f"[TTS] Worker error: {e}")

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            print("[TTS] Worker thread did not stop within timeout")
