import base64
import time
from typing import TYPE_CHECKING, List, Optional

import anthropic
import cv2

from src.capture.screen import MinimapFrame

if TYPE_CHECKING:
    from src.telemetry.collector import DataCollector


_UNCHANGED_FALLBACK = 10.0  # fire at most every 10s when scene hasn't changed


class AIAnalyzer:
    def __init__(self, config: dict, collector: "Optional[DataCollector]" = None):
        self.client   = anthropic.Anthropic()
        ai_cfg        = config.get("ai", {})
        self.model    = ai_cfg.get("model", "claude-opus-4-5")
        self.interval = ai_cfg.get("analyze_interval", 5.0)
        self._last_call:     float = 0.0
        self._last_api_call: float = 0.0
        self._last_state:    tuple = ()
        self._last_sample_ts: int  = 0   # ts of last submitted sample, for feedback
        self._collector = collector

    def should_analyze(self) -> bool:
        return time.time() - self._last_call >= self.interval

    def analyze(
        self,
        frame:            MinimapFrame,
        enemy_count:      int,
        map_name:         str = "unknown",
        active_abilities: Optional[List[str]] = None,
        spike_active:     bool = False,
        recent_callouts:  Optional[List[str]] = None,
    ) -> Optional[str]:
        if not self.should_analyze():
            return None

        now = time.time()
        self._last_call = now

        state = (enemy_count, frozenset(active_abilities or []), spike_active)
        if state == self._last_state and now - self._last_api_call < _UNCHANGED_FALLBACK:
            return None

        self._last_api_call = now
        self._last_state    = state

        ok, buf = cv2.imencode(".jpg", frame.data, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        img_b64 = base64.b64encode(buf).decode()

        parts: List[str] = [f"Valorant minimap, map {map_name}, {enemy_count} enemies."]
        if spike_active:
            parts.append("Spike is planted.")
        if active_abilities:
            parts.append(f"Active: {', '.join(active_abilities)}.")
        if recent_callouts:
            parts.append(f"Recent: {'; '.join(recent_callouts[-2:])}.")
        parts.append("One new tactical callout max 10 words (threat + location). Callout only.")
        prompt = " ".join(parts)

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=30,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg", "data": img_b64,
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            if not resp.content:
                return None
            callout = resp.content[0].text.strip()
            self._last_sample_ts = int(now)
            return callout
        except Exception as e:
            print(f"[AI] Error: {e}")
            return None
