import base64
import time
from typing import Optional

import anthropic
import cv2
import mss
import numpy as np

KNOWN_MAPS = {
    "ascent", "bind", "haven", "split", "icebox",
    "lotus", "sunset", "abyss", "breeze", "fracture", "pearl",
}


class MapDetector:
    def __init__(self, config: dict):
        self.client = anthropic.Anthropic()
        self.model: str = config["ai"]["model"]
        cfg = config.get("map_detection", {})
        self.recheck_interval: float = cfg.get("recheck_interval", 300.0)
        self.startup_retry: float = cfg.get("startup_retry_interval", 10.0)
        self._detected: Optional[str] = None
        self._last_check: float = 0.0

    def _grab_screen(self) -> np.ndarray:
        with mss.mss() as sct:
            raw = sct.grab(sct.monitors[1])
        img = np.array(raw)[:, :, :3]
        # Downscale 3x - enough for Claude Vision, cheap on bandwidth
        h, w = img.shape[:2]
        return cv2.resize(img, (w // 3, h // 3))

    def _to_b64(self, img: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode()

    def detect_now(self) -> Optional[str]:
        """One-shot detection from current screen. Returns map name or None."""
        img = self._grab_screen()
        b64 = self._to_b64(img)
        maps = ", ".join(sorted(KNOWN_MAPS))
        prompt = (
            "This is a screenshot from the game Valorant. "
            "Identify which map is being played or previewed. "
            f"Answer with exactly one lowercase word from: {maps}. "
            "If you cannot determine the map with confidence, answer: unknown"
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=20,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            result = resp.content[0].text.strip().lower()
            return result if result in KNOWN_MAPS else None
        except Exception as e:
            print(f"[MapDetector] API error: {e}")
            return None

    def get_map(self) -> Optional[str]:
        """
        Returns the current map name.
        Re-detects if never detected or recheck_interval has elapsed.
        """
        now = time.time()
        needs_check = self._detected is None or (now - self._last_check) >= self.recheck_interval
        if not needs_check:
            return self._detected

        detected = self.detect_now()
        self._last_check = now
        if detected and detected != self._detected:
            print(f"[MapDetector] Map: {detected}")
            self._detected = detected
        return self._detected

    def wait_for_map(self) -> str:
        """
        Blocks until a map is detected (retries every startup_retry_interval).
        Returns the detected map name.
        """
        while True:
            result = self.detect_now()
            self._last_check = time.time()
            if result:
                self._detected = result
                print(f"[MapDetector] Detected: {result}")
                return result
            print(f"[MapDetector] Not in game yet, retrying in {self.startup_retry:.0f}s...")
            time.sleep(self.startup_retry)
