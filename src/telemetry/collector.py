"""
Background data collector for training dataset generation.

Opt-in only: data_collection.enabled must be set to true in config.yaml.
Off by default -- user consciously enables it.

Collects two types of samples:

  1. minimap_callout
     Sent after every successful Claude AI analysis callout.
     Payload: minimap JPEG + callout text + map name + enemy positions.
     Used to improve: AI callout quality, zone label accuracy.

  2. footstep_audio
     Sent when a footstep event is detected and a zone can be estimated.
     Payload: mono audio .npy clip (48 kHz float32) + zone name + shoe type + map name.
     Used to improve: shoe-type classifier, surface detector.

Features:
  - Deduplication: identical or near-identical frames are skipped (20 s window)
  - Version tagging: every sample carries app_version for dataset slicing
  - Feedback: users can mark callouts as good/bad from the overlay
  - Non-blocking queue: game loop is never stalled; samples dropped silently if full

Threading model:
  A single daemon thread drains a bounded queue (max 64 items).
  submit_*() methods are non-blocking and never raise.
"""
import hashlib
import io
import queue
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _frame_hash(img: np.ndarray) -> str:
    """Fast perceptual hash via 16x16 grayscale thumbnail."""
    tiny = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
    if tiny.ndim == 3:
        tiny = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(tiny.tobytes()).hexdigest()


class DataCollector:
    _QUEUE_SIZE     = 64
    _UPLOAD_TIMEOUT = 8     # seconds per HTTP request
    _JPEG_QUALITY   = 80
    _DEDUP_WINDOW   = 20.0  # skip identical frames seen within this many seconds

    def __init__(self, config: dict) -> None:
        cfg              = config.get("data_collection", {})
        self.enabled     = cfg.get("enabled", False)
        self.endpoint    = cfg.get("endpoint", "").rstrip("/")
        self._api_key    = cfg.get("api_key", "")
        self._app_version = config.get("app_version", "unknown")
        self._queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_SIZE)
        self._seen: dict[str, float] = {}   # hash -> last_seen timestamp

        if self.enabled and self.endpoint:
            t = threading.Thread(target=self._worker, daemon=True, name="DataCollector")
            t.start()
            print(f"[Collector] Enabled. Endpoint: {self.endpoint}  version: {self._app_version}")
        elif self.enabled:
            print("[Collector] data_collection.enabled=true but endpoint is empty -- disabled.")
            self.enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_minimap_callout(
        self,
        minimap_img:      np.ndarray,
        callout:          str,
        map_name:         str,
        enemy_positions:  List[Tuple[float, float]],
        spike_active:     bool = False,
        recent_callouts:  Optional[List[str]] = None,
        confidence:       float = 1.0,
    ) -> int:
        """
        Submit a minimap frame with the AI-generated callout as label.
        Returns the sample timestamp (used for feedback correlation).
        Non-blocking. Drops silently if queue is full.
        """
        if not self.enabled:
            return 0
        if self._is_duplicate(minimap_img):
            return 0
        ts = int(time.time())
        enemies_str = ";".join(f"{x:.4f},{y:.4f}" for x, y in enemy_positions)
        self._enqueue({
            "type":            "minimap_callout",
            "image":           self._encode_jpg(minimap_img),
            "ext":             "jpg",
            "label":           callout,
            "map":             map_name,
            "enemies":         enemies_str,
            "spike_active":    "1" if spike_active else "0",
            "recent_callouts": "|".join((recent_callouts or [])[-3:]),
            "conf":            round(confidence, 4),
            "app_version":     self._app_version,
            "ts":              ts,
        })
        return ts

    def submit_footstep_audio(
        self,
        audio_clip: np.ndarray,
        zone:       str,
        shoe_type:  str,
        map_name:   str,
        surface:    str = "",
    ) -> None:
        """
        Submit a raw footstep audio clip with its estimated zone label.
        audio_clip: float32 mono array at 48000 Hz, ~0.35 s.
        Non-blocking. Drops silently if queue is full.
        """
        if not self.enabled:
            return
        if audio_clip is None or len(audio_clip) == 0:
            return
        self._enqueue({
            "type":        "footstep_audio",
            "image":       self._encode_npy(audio_clip),
            "ext":         "npy",
            "label":       shoe_type,
            "map":         map_name,
            "zone":        zone,
            "surface":     surface,
            "conf":        0.0,
            "app_version": self._app_version,
            "ts":          int(time.time()),
        })

    def submit_feedback(self, ts: int, positive: bool) -> None:
        """
        Mark a previously submitted sample as good (positive=True) or bad.
        Called from the overlay feedback buttons.
        Non-blocking. Drops silently if queue is full.
        """
        if not self.enabled or not ts:
            return
        self._enqueue({
            "type":     "feedback",
            "ts":       ts,
            "positive": "1" if positive else "0",
        })

    # Legacy generic submit kept for map_detector.py compatibility
    def submit(
        self,
        img:        np.ndarray,
        label:      str,
        function:   str,
        confidence: float = 1.0,
    ) -> None:
        if not self.enabled:
            return
        if self._is_duplicate(img):
            return
        self._enqueue({
            "type":        function,
            "image":       self._encode_jpg(img),
            "ext":         "jpg",
            "label":       label,
            "map":         "",
            "conf":        round(confidence, 4),
            "app_version": self._app_version,
            "ts":          int(time.time()),
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_duplicate(self, img: np.ndarray) -> bool:
        """Return True if an identical frame was submitted within _DEDUP_WINDOW."""
        now = time.time()
        h = _frame_hash(img)
        # Lazy eviction: only rebuild when dict grows large (avoids O(n) every call)
        if len(self._seen) > 200:
            self._seen = {k: v for k, v in self._seen.items() if now - v < self._DEDUP_WINDOW}
        elif h in self._seen and now - self._seen[h] >= self._DEDUP_WINDOW:
            del self._seen[h]
        if h in self._seen:
            return True
        self._seen[h] = now
        return False

    def _enqueue(self, item: dict) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            print(f"[Collector] Queue full, dropping {item.get('type', 'unknown')}")

    _upload_errors = 0   # class-level counter for rate-limited logging

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                self._upload(item)
                DataCollector._upload_errors = 0
            except Exception as e:
                DataCollector._upload_errors += 1
                # Log first failure and every 10th afterwards to avoid log spam
                if DataCollector._upload_errors == 1 or DataCollector._upload_errors % 10 == 0:
                    print(f"[Collector] Upload failed ({DataCollector._upload_errors}x): {e}")
            finally:
                self._queue.task_done()

    def _upload(self, item: dict) -> None:
        try:
            import requests
        except ImportError:
            return

        headers = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        # Feedback items use a separate lightweight endpoint
        if item.get("type") == "feedback":
            resp = requests.post(
                f"{self.endpoint}/feedback",
                json={"ts": item["ts"], "positive": item["positive"] == "1"},
                headers=headers,
                timeout=self._UPLOAD_TIMEOUT,
            )
            resp.raise_for_status()
            return

        ext      = item.get("ext", "jpg")
        mime     = "audio/x-numpy" if ext == "npy" else "image/jpeg"
        data     = {k: str(v) for k, v in item.items() if k not in ("image", "ext")}
        resp = requests.post(
            f"{self.endpoint}/collect",
            files={"file": (f"sample.{ext}", item["image"], mime)},
            data=data,
            headers=headers,
            timeout=self._UPLOAD_TIMEOUT,
        )
        resp.raise_for_status()

    @staticmethod
    def _encode_jpg(img: np.ndarray) -> bytes:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, DataCollector._JPEG_QUALITY])
        return buf.tobytes()

    @staticmethod
    def _encode_npy(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, arr.astype(np.float32))
        return buf.getvalue()
