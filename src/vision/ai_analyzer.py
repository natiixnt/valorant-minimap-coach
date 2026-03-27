import base64
import time
from typing import Optional

import anthropic
import cv2

from src.capture.screen import MinimapFrame


class AIAnalyzer:
    def __init__(self, config: dict):
        self.client = anthropic.Anthropic()
        self.model: str = config["ai"]["model"]
        self.interval: float = config["ai"]["analyze_interval"]
        self._last_call: float = 0.0

    def should_analyze(self) -> bool:
        return time.time() - self._last_call >= self.interval

    def analyze(self, frame: MinimapFrame, enemy_count: int, map_name: str = "unknown") -> Optional[str]:
        if not self.should_analyze():
            return None
        self._last_call = time.time()

        _, buf = cv2.imencode(".jpg", frame.data, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf).decode()

        prompt = (
            f"Valorant minimap, map: {map_name}. "
            f"CV detected ~{enemy_count} visible enemies. "
            "Give one short voice callout (max 10 words) like a smart teammate. "
            "Focus on: danger, rotation advice, or flanking warnings. "
            "Reply with callout text only."
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=60,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            print(f"[AI] Error: {e}")
            return None
