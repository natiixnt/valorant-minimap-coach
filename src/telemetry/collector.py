"""
Background data collector for training dataset generation.

Always-on by default in released builds.  Users can opt-out by setting
data_collection.enabled: false in config.yaml.

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
  Items are batched (_BATCH_SIZE or _BATCH_TIMEOUT) and pushed as a single
  commit to a Hugging Face dataset repo -- no intermediate server needed.

Dataset layout (mirrors server/collect_server.py):
  minimap_callout/<sha12>/{ts}_v{ver}_conf{c}.jpg   + .json sidecar
  footstep_audio/<label>/{ts}_v{ver}_conf{c}.npy    + .json sidecar
  map_detection/<label>/{ts}_v{ver}_conf{c}.jpg     + .json sidecar
  feedback/{ts}.json
"""
import hashlib
import io
import json
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


def _safe(s: str, max_len: int = 60) -> str:
    """Strip unsafe characters for use in repo paths."""
    return "".join(c for c in str(s) if c.isalnum() or c in "_-.")[:max_len]


# Default repo -- token is loaded from config.yaml (bundled with the exe, never in source).
_DEFAULT_HF_REPO = "naithai/valorant-minimap-coach"


class DataCollector:
    _QUEUE_SIZE    = 64
    _BATCH_SIZE    = 10     # items per HF commit (fewer = more real-time; more = fewer API calls)
    _BATCH_TIMEOUT = 120.0  # flush at least every N seconds even if batch isn't full
    _JPEG_QUALITY  = 80
    _DEDUP_WINDOW  = 20.0   # skip identical frames seen within this many seconds

    def __init__(self, config: dict) -> None:
        cfg               = config.get("data_collection", {})
        # Default enabled=True; user can opt-out via data_collection.enabled: false
        self.enabled      = cfg.get("enabled", True)
        self._hf_repo     = cfg.get("hf_repo", "").strip() or _DEFAULT_HF_REPO
        self._hf_token    = cfg.get("hf_token", "").strip()
        self._app_version = config.get("app_version", "unknown")
        self._queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_SIZE)
        self._seen: dict[str, float] = {}

        if self.enabled and self._hf_token:
            t = threading.Thread(target=self._worker, daemon=True, name="DataCollector")
            t.start()
            print(f"[Collector] Active. HF repo: {self._hf_repo}  version: {self._app_version}")
        elif self.enabled:
            print("[Collector] No hf_token in config -- data collection disabled.")

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
        batch: list = []
        last_flush = time.time()

        while True:
            # Block up to remaining time before next forced flush
            timeout = max(0.5, self._BATCH_TIMEOUT - (time.time() - last_flush))
            try:
                item = self._queue.get(timeout=timeout)
                batch.append(item)
                self._queue.task_done()
            except queue.Empty:
                pass

            should_flush = (
                len(batch) >= self._BATCH_SIZE
                or (batch and time.time() - last_flush >= self._BATCH_TIMEOUT)
            )
            if should_flush:
                self._flush_batch(batch)
                batch = []
                last_flush = time.time()

    def _flush_batch(self, batch: list) -> None:
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
        except ImportError:
            print("[Collector] huggingface_hub not installed. Run: pip install huggingface_hub")
            return

        api = HfApi(token=self._hf_token or None)
        operations = []

        for item in batch:
            try:
                operations.extend(self._item_to_ops(item))
            except Exception as e:
                print(f"[Collector] Failed to prepare {item.get('type', '?')}: {e}")

        if not operations:
            return

        try:
            api.create_commit(
                repo_id=self._hf_repo,
                repo_type="dataset",
                operations=operations,
                commit_message=f"coach data: {len(batch)} samples",
            )
            DataCollector._upload_errors = 0
        except Exception as e:
            DataCollector._upload_errors += 1
            if DataCollector._upload_errors == 1 or DataCollector._upload_errors % 10 == 0:
                print(f"[Collector] HF upload failed ({DataCollector._upload_errors}x): {e}")

    def _item_to_ops(self, item: dict) -> list:
        from huggingface_hub import CommitOperationAdd

        # Feedback: one small JSON file, no binary payload
        if item.get("type") == "feedback":
            content = json.dumps({
                "ts":       item["ts"],
                "positive": item["positive"] == "1",
            }, ensure_ascii=False).encode()
            return [CommitOperationAdd(
                path_in_repo=f"feedback/{item['ts']}.json",
                path_or_fileobj=io.BytesIO(content),
            )]

        function = _safe(item.get("type", "unknown"))
        label    = item.get("label", "unknown")
        ext      = item.get("ext", "jpg")
        ts       = item.get("ts", int(time.time()))
        ver      = _safe(str(item.get("app_version", "unknown")))
        conf     = float(item.get("conf", 0.0))

        # Directory key: hash for free-text labels, safe string otherwise
        if item.get("type") == "minimap_callout":
            key = hashlib.sha1(label.encode()).hexdigest()[:12]
        else:
            key = _safe(label)

        base = f"{function}/{key}"
        stem = f"{ts}_v{ver}_conf{conf:.2f}"

        meta = {k: v for k, v in item.items() if k not in ("image", "ext")}

        return [
            CommitOperationAdd(
                path_in_repo=f"{base}/{stem}.{ext}",
                path_or_fileobj=io.BytesIO(item["image"]),
            ),
            CommitOperationAdd(
                path_in_repo=f"{base}/{stem}.json",
                path_or_fileobj=io.BytesIO(json.dumps(meta, ensure_ascii=False).encode()),
            ),
        ]

    @staticmethod
    def _encode_jpg(img: np.ndarray) -> bytes:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, DataCollector._JPEG_QUALITY])
        return buf.tobytes()

    @staticmethod
    def _encode_npy(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, arr.astype(np.float32))
        return buf.getvalue()
