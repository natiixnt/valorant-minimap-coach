"""
Map detection with local template matching and Claude API fallback.

Lifecycle:
- First time a map is seen: Claude identifies it, screenshot saved as template.
- All subsequent detections: local histogram matching at ~1ms, zero API cost.
- After all 11 maps are learned: Claude client is never instantiated again.
"""
import base64
import sys
import threading
import time
import urllib.request
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

# Frozen exe: save templates next to the exe (persists across runs).
# Dev: save in repo root / data / map_templates.
if getattr(sys, "frozen", False):
    _BASE_DIR      = Path(sys.executable).parent
else:
    _BASE_DIR      = Path(__file__).resolve().parent.parent.parent

_TEMPLATES_DIR     = _BASE_DIR / "data" / "map_templates"
_RADAR_DIR         = _BASE_DIR / "data" / "map_radar"
_RADAR_THRESHOLD   = 0.60   # lower than full-screen; comparing minimap vs reference icon

# Official flat overhead displayIcon from valorant-api.com (public, no auth needed)
_RADAR_URLS: dict[str, str] = {
    "ascent":   "https://media.valorant-api.com/maps/7eaecc1b-4337-bbf6-6ab9-04b8f06b3319/displayicon.png",
    "split":    "https://media.valorant-api.com/maps/d960549e-485c-e861-8d71-aa9d1aed12a2/displayicon.png",
    "fracture": "https://media.valorant-api.com/maps/b529448b-4d60-346e-e89e-00a4c527a405/displayicon.png",
    "bind":     "https://media.valorant-api.com/maps/2c9d57ec-4431-9c5e-2939-8f9ef6dd5cba/displayicon.png",
    "breeze":   "https://media.valorant-api.com/maps/2fb9a4fd-47b8-4e7d-a969-74b4046ebd53/displayicon.png",
    "abyss":    "https://media.valorant-api.com/maps/224b0a95-48b9-f703-1bd8-67aca101a61f/displayicon.png",
    "lotus":    "https://media.valorant-api.com/maps/2fe4ed3a-450a-948b-6d6b-e89a78e680a9/displayicon.png",
    "sunset":   "https://media.valorant-api.com/maps/92584fbe-486a-b1b2-9faa-39b0f486b498/displayicon.png",
    "pearl":    "https://media.valorant-api.com/maps/fd267378-4d1d-484f-ff52-77821ed10dc2/displayicon.png",
    "icebox":   "https://media.valorant-api.com/maps/e2ad5c54-4114-a870-9641-8ea21279579a/displayicon.png",
    "haven":    "https://media.valorant-api.com/maps/2bee0dc9-4ffe-519b-1cbd-7fbe763a6047/displayicon.png",
}


# ---------------------------------------------------------------------------
# Radar reference store (downloaded from valorant-api.com, cached locally)
# ---------------------------------------------------------------------------

class _RadarStore:
    """Downloads official map overhead icons and fingerprints them for matching."""

    def __init__(self) -> None:
        _RADAR_DIR.mkdir(parents=True, exist_ok=True)
        self._fps: dict[str, np.ndarray] = {}
        threading.Thread(target=self._load_all, daemon=True, name="RadarStore").start()

    def _load_all(self) -> None:
        import socket
        loaded = 0
        for map_name, url in _RADAR_URLS.items():
            path = _RADAR_DIR / f"{map_name}.png"
            if not path.exists():
                tmp = path.with_suffix(".tmp")
                try:
                    old_timeout = socket.getdefaulttimeout()
                    socket.setdefaulttimeout(10)
                    try:
                        urllib.request.urlretrieve(url, tmp)
                    finally:
                        socket.setdefaulttimeout(old_timeout)
                    # Validate before committing: corrupt/truncated PNG returns None
                    if cv2.imread(str(tmp)) is None:
                        raise ValueError("downloaded file is not a readable image")
                    tmp.rename(path)
                except Exception as e:
                    print(f"[RadarStore] Download failed for {map_name}: {e}")
                    tmp.unlink(missing_ok=True)
                    continue
            img = cv2.imread(str(path))
            if img is not None:
                self._fps[map_name] = _fingerprint(img)
                loaded += 1
        if loaded:
            print(f"[RadarStore] Ready: {loaded}/{len(_RADAR_URLS)} map references loaded")

    def match(self, minimap_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._fps:
            return None, 0.0
        fp = _fingerprint(minimap_bgr)
        best_map, best_score, second_score = None, 0.0, 0.0
        for name, ref in self._fps.items():
            s = _score(fp, ref)
            if s > best_score:
                second_score = best_score
                best_score = s
                best_map = name
            elif s > second_score:
                second_score = s
        # Require clear margin over second-best to avoid false positives
        if best_score >= _RADAR_THRESHOLD and (best_score - second_score) >= 0.04:
            return best_map, best_score
        return None, best_score
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
            existing[0].unlink(missing_ok=True)
            existing = existing[1:]

        path = _TEMPLATES_DIR / f"{map_name}_{int(time.time())}.jpg"
        small = cv2.resize(img, (_THUMB_W, _THUMB_H))
        if not cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            print(f"[MapDetector] Failed to write template: {path}")
            return
        self._fps[map_name].append(_fingerprint(small))
        print(f"[MapDetector] Learned template: {path.name}")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class MapDetector:
    def __init__(self, config: dict, collector: "Optional[DataCollector]" = None) -> None:
        self._store       = _TemplateStore()
        self._radar_store = _RadarStore()   # async download starts immediately
        map_cfg           = config.get("map_detection", {})
        self._model       = map_cfg.get("model", "claude-haiku-4-5-20251001")
        self.recheck_interval = map_cfg.get("recheck_interval", 300.0)
        self.startup_retry    = map_cfg.get("startup_retry_interval", 10.0)
        self._client: Optional[anthropic.Anthropic] = None
        self._collector   = collector
        self._detected: Optional[str] = None
        self._last_check: float = 0.0
        self._sct         = mss.mss()

        # Minimap region for radar matching -- loaded from config; updated by auto-calibration
        self._minimap_region: Optional[dict] = config.get("minimap", {}).get("region")
        try:
            from src.ui.overlay import load_settings as _ls
            saved = _ls().get("minimap_region")
            if saved:
                self._minimap_region = saved
        except Exception:
            pass

        missing = KNOWN_MAPS - self._store.learned_maps()
        if not missing:
            print("[MapDetector] All maps learned locally.")
        else:
            print(f"[MapDetector] Missing local templates for: {', '.join(sorted(missing))} "
                  f"-- will use radar matching (no API key needed)")

    def set_minimap_region(self, region: dict) -> None:
        self._minimap_region = region

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._sct.close()

    def _grab_minimap(self) -> Optional[np.ndarray]:
        if not self._minimap_region:
            return None
        try:
            raw = self._sct.grab(self._minimap_region)
            return np.array(raw)[:, :, :3]
        except Exception:
            return None

    def _grab(self) -> np.ndarray:
        # monitors[0] is the virtual desktop spanning all displays; works on any
        # multi-monitor setup without needing to know which screen Valorant is on.
        raw = self._sct.grab(self._sct.monitors[0])
        img = np.array(raw)[:, :, :3]
        h, w = img.shape[:2]
        # 4x downscale -> 480x270 on 1920x1080, fits in 1 image tile
        return cv2.resize(img, (w // 4, h // 4))

    def _claude_identify(self, img: np.ndarray) -> Optional[str]:
        if self._client is None:
            self._client = anthropic.Anthropic()
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
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
                if not resp.content:
                    return None
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

        # 1. Fast local template matching (trained from previous sessions; ~1ms)
        map_name, confidence = self._store.match(img)
        if map_name and confidence >= _MATCH_THRESHOLD:
            print(f"[MapDetector] Local match: {map_name} (conf={confidence:.2f})")
            return map_name

        # 2. Radar matching against official map overhead icons -- no API key needed
        minimap_img = self._grab_minimap()
        if minimap_img is not None:
            result, conf = self._radar_store.match(minimap_img)
            if result:
                print(f"[MapDetector] Radar match: {result} (conf={conf:.2f})")
                self._store.save(result, img)   # persist so next session uses local match
                return result

        # 3. Claude API fallback (first-time, API key required)
        print(f"[MapDetector] Low confidence ({confidence:.2f}), querying Claude...")
        result = self._claude_identify(img)
        if result:
            self._store.save(result, img)
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
