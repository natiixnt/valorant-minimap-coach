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

Threading model:
  A single daemon thread drains a bounded queue (max 64 items).
  submit_*() methods are non-blocking and never raise -- if the queue is
  full the sample is silently dropped. If the server is unreachable the
  item is also dropped (no retry, no blocking).

Wire-up in coach.py:
    self.collector = DataCollector(config)
    # after AI callout:
    self.collector.submit_minimap_callout(frame.data, callout, map_name, enemies)
    # after footstep finding:
    self.collector.submit_footstep_audio(finding.audio_clip, finding.zone,
                                         finding.agent, map_name)
"""
import io
import queue
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


class DataCollector:
    _QUEUE_SIZE    = 64
    _UPLOAD_TIMEOUT = 8    # seconds per HTTP request
    _JPEG_QUALITY  = 80

    def __init__(self, config: dict) -> None:
        cfg           = config.get("data_collection", {})
        self.enabled  = cfg.get("enabled", False)
        self.endpoint = cfg.get("endpoint", "").rstrip("/")
        self._api_key = cfg.get("api_key", "")
        self._queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_SIZE)

        if self.enabled and self.endpoint:
            t = threading.Thread(target=self._worker, daemon=True, name="DataCollector")
            t.start()
            print(f"[Collector] Enabled. Endpoint: {self.endpoint}")
        elif self.enabled:
            print("[Collector] data_collection.enabled=true but endpoint is empty -- disabled.")
            self.enabled = False

    # ------------------------------------------------------------------
    # Typed public API
    # ------------------------------------------------------------------

    def submit_minimap_callout(
        self,
        minimap_img: np.ndarray,
        callout: str,
        map_name: str,
        enemy_positions: List[Tuple[float, float]],
        confidence: float = 1.0,
    ) -> None:
        """
        Submit a minimap frame with the AI-generated callout as its label.
        Called after every successful Claude analysis so the server can build
        a labelled dataset of (minimap image -> callout text) pairs.

        enemy_positions: list of (x, y) normalized 0-1 from MinimapDetector.
        Non-blocking. Drops silently if queue is full.
        """
        if not self.enabled:
            return
        enemies_str = ";".join(f"{x:.4f},{y:.4f}" for x, y in enemy_positions)
        self._enqueue({
            "type":    "minimap_callout",
            "image":   self._encode_jpg(minimap_img),
            "ext":     "jpg",
            "label":   callout,
            "map":     map_name,
            "enemies": enemies_str,
            "conf":    round(confidence, 4),
            "ts":      int(time.time()),
        })

    def submit_footstep_audio(
        self,
        audio_clip: np.ndarray,
        zone: str,
        shoe_type: str,
        map_name: str,
        surface: str = "",
    ) -> None:
        """
        Submit a raw footstep audio clip with its estimated zone label.
        Called from coach.py when an AudioFinding with a known zone is produced.

        audio_clip: float32 mono array at 48000 Hz, ~0.35 s.
        zone:       e.g. "B Long" -- estimated map zone of the sound source.
        shoe_type:  "heavy" | "medium" | "light" | "unknown".
        surface:    "metal" | "wood" | "concrete" | "carpet" | "".

        The server stores these clips for later shoe-type classifier retraining.
        Labels are soft (estimated, not ground-truth) -- server-side review handles validation.
        Non-blocking. Drops silently if queue is full.
        """
        if not self.enabled:
            return
        if audio_clip is None or len(audio_clip) == 0:
            return
        self._enqueue({
            "type":     "footstep_audio",
            "image":    self._encode_npy(audio_clip),
            "ext":      "npy",
            "label":    shoe_type,
            "map":      map_name,
            "zone":     zone,
            "surface":  surface,
            "conf":     0.0,   # estimated label, server must validate
            "ts":       int(time.time()),
        })

    # Legacy generic submit (used by ai_analyzer.py and map_detector.py added by linter)
    def submit(
        self,
        img: np.ndarray,
        label: str,
        function: str,
        confidence: float = 1.0,
    ) -> None:
        if not self.enabled:
            return
        self._enqueue({
            "type":   function,
            "image":  self._encode_jpg(img),
            "ext":    "jpg",
            "label":  label,
            "map":    "",
            "conf":   round(confidence, 4),
            "ts":     int(time.time()),
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enqueue(self, item: dict) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass  # silent drop -- game loop must never block

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                self._upload(item)
            except Exception:
                pass  # silent drop on any error
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

        ext = item.get("ext", "jpg")
        mime = "audio/x-numpy" if ext == "npy" else "image/jpeg"
        filename = f"sample.{ext}"

        data = {k: str(v) for k, v in item.items() if k not in ("image", "ext")}

        requests.post(
            f"{self.endpoint}/collect",
            files={"file": (filename, item["image"], mime)},
            data=data,
            headers=headers,
            timeout=self._UPLOAD_TIMEOUT,
        )
        # raise_for_status intentionally omitted -- treat non-2xx as silent drop

    @staticmethod
    def _encode_jpg(img: np.ndarray) -> bytes:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, DataCollector._JPEG_QUALITY])
        return buf.tobytes()

    @staticmethod
    def _encode_npy(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, arr.astype(np.float32))
        return buf.getvalue()
