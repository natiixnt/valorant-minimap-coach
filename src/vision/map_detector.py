"""
Map detection with local template matching and Claude API fallback.

Lifecycle:
- First time a map is seen: Claude identifies it, screenshot saved as template.
- All subsequent detections: local histogram matching at ~1ms, zero API cost.
- After all 11 maps are learned: Claude client is never instantiated again.
"""
import base64
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import anthropic
import cv2
import mss
import numpy as np

if TYPE_CHECKING:
    from src.telemetry.collector import DataCollector

KNOWN_MAPS = {
    "ascent", "bind", "haven", "split", "icebox",
    "lotus", "sunset", "abyss", "breeze", "fracture", "pearl",
}

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "map_templates"
_THUMB_W, _THUMB_H = 320, 180
_MATCH_THRESHOLD = 0.82   # histogram correlation; 1.0 = perfect match
_MAX_TEMPLATES    = 5     # stored screenshots per map


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def _fingerprint(img: np.ndarray) -> np.ndarray:
    """
    Compact HSV histogram over the full image.
    32 hue x 16 sat x 16 val = 8 192 floats, normalized.
    Fast to compute (~0.3 ms) and robust to minor UI differences.
    """
    small = cv2.resize(img, (_THUMB_W, _THUMB_H))
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([hsv], [0, 1, 2], None,
                         [32, 16, 16], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def _score(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


# ---------------------------------------------------------------------------
# Template store
# ---------------------------------------------------------------------------

class _TemplateStore:
    """Loads, matches, and saves map fingerprints to disk."""

    def __init__(self) -> None:
        _TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        self._fps: dict[str, list[np.ndarray]] = {m: [] for m in KNOWN_MAPS}
        self._load()

    def _load(self) -> None:
        for path in _TEMPLATES_DIR.glob("*.jpg"):
            map_name = path.stem.rsplit("_", 1)[0]
            if map_name not in KNOWN_MAPS:
                continue
            img = cv2.imread(str(path))
            if img is not None:
                self._fps[map_name].append(_fingerprint(img))
        learned = self.learned_maps()
        if learned:
            print(f"[MapDetector] Loaded local templates for: {', '.join(sorted(learned))}")

    def learned_maps(self) -> set:
        return {m for m, fps in self._fps.items() if fps}

    def match(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """Return (best_map_or_None, confidence). confidence in [-1, 1]."""
        fp = _fingerprint(img)
        best_map: Optional[str] = None
        best_score = -1.0
        for map_name, stored in self._fps.items():
            for s in stored:
                sc = _score(fp, s)
                if sc > best_score:
                    best_score, best_map = sc, map_name
        return best_map, best_score

    def save(self, map_name: str, img: np.ndarray) -> None:
        if map_name not in KNOWN_MAPS:
            return
        # Evict oldest file if at capacity
        existing = sorted(_TEMPLATES_DIR.glob(f"{map_name}_*.jpg"))
        while len(existing) >= _MAX_TEMPLATES:
            existing[0].unlink()
            existing = existing[1:]

        path = _TEMPLATES_DIR / f"{map_name}_{int(time.time())}.jpg"
        small = cv2.resize(img, (_THUMB_W, _THUMB_H))
        cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._fps[map_name].append(_fingerprint(img))
        print(f"[MapDetector] Learned template: {path.name}")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class MapDetector:
    def __init__(self, config: dict, collector: "Optional[DataCollector]" = None) -> None:
        self._store     = _TemplateStore()
        map_cfg         = config.get("map_detection", {})
        self._model     = map_cfg.get("model", "claude-haiku-4-5-20251001")
        self.recheck_interval = map_cfg.get("recheck_interval", 300.0)
        self.startup_retry    = map_cfg.get("startup_retry_interval", 10.0)
        self._client: Optional[anthropic.Anthropic] = None  # lazy-init, only if needed
        self._collector = collector
        self._detected: Optional[str] = None
        self._last_check: float = 0.0
        self._sct = mss.mss()   # reuse single instance; avoids 5-10ms per-call overhead

        missing = KNOWN_MAPS - self._store.learned_maps()
        if not missing:
            print("[MapDetector] All maps learned. Claude will not be used for map detection.")
        else:
            print(f"[MapDetector] Missing templates for: {', '.join(sorted(missing))} "
                  f"(will use Claude on first encounter, then learn locally)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._sct.close()

    def _grab(self) -> np.ndarray:
        raw = self._sct.grab(self._sct.monitors[1])
        img = np.array(raw)[:, :, :3]
        h, w = img.shape[:2]
        # 4x downscale -> 480x270 on 1920x1080, fits in 1 image tile
        return cv2.resize(img, (w // 4, h // 4))

    def _claude_identify(self, img: np.ndarray) -> Optional[str]:
        if self._client is None:
            self._client = anthropic.Anthropic()
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64    = base64.b64encode(buf).decode()
        maps   = ", ".join(sorted(KNOWN_MAPS))
        for attempt in range(2):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=10,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image",
                             "source": {"type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": b64}},
                            {"type": "text",
                             "text": f"Valorant screenshot. Which map? One word from: {maps}. If unsure: unknown"},
                        ],
                    }],
                )
                result = resp.content[0].text.strip().lower()
                return result if result in KNOWN_MAPS else None
            except Exception as e:
                if attempt == 0:
                    print(f"[MapDetector] Claude error (retrying): {e}")
                    time.sleep(1.0)
                else:
                    print(f"[MapDetector] Claude error (giving up): {e}")
        return None

    # ------------------------------------------------------------------
    # Public API (same as before)
    # ------------------------------------------------------------------

    def detect_now(self) -> Optional[str]:
        img = self._grab()

        map_name, confidence = self._store.match(img)
        if map_name and confidence >= _MATCH_THRESHOLD:
            print(f"[MapDetector] Local match: {map_name} (conf={confidence:.2f})")
            return map_name

        # Confidence too low or no templates yet - ask Claude
        print(f"[MapDetector] Low confidence ({confidence:.2f}), querying Claude...")
        result = self._claude_identify(img)
        if result:
            self._store.save(result, img)
            # Hard case: local matching failed, Claude had to step in - valuable training sample
            if self._collector:
                self._collector.submit(img, result, "map_detection", confidence)
        return result

    def get_map(self) -> Optional[str]:
        now = time.time()
        if self._detected and (now - self._last_check) < self.recheck_interval:
            return self._detected
        detected = self.detect_now()
        self._last_check = now
        if detected and detected != self._detected:
            print(f"[MapDetector] Map: {detected}")
            self._detected = detected
        return self._detected

    def wait_for_map(self, timeout: float = 300.0) -> Optional[str]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.detect_now()
            self._last_check = time.time()
            if result:
                self._detected = result
                print(f"[MapDetector] Detected: {result}")
                return result
            print(f"[MapDetector] Not in game yet, "
                  f"retrying in {self.startup_retry:.0f}s...")
            time.sleep(self.startup_retry)
        print(f"[MapDetector] Timed out after {timeout:.0f}s waiting for map.")
        return None
