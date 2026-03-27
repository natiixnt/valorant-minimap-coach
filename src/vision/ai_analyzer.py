import base64
import time
from typing import List, Optional

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

    def analyze(
        self,
        frame: MinimapFrame,
        enemy_count: int,
        map_name: str = "unknown",
        active_abilities: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not self.should_analyze():
            return None
        self._last_call = time.time()

        _, buf = cv2.imencode(".jpg", frame.data, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf).decode()

        ability_ctx = ""
        if active_abilities:
            ability_ctx = f" CV also sees these active: {', '.join(active_abilities)}."

        prompt = (
            f"Valorant minimap, map: {map_name}. "
            f"CV detected ~{enemy_count} enemy blobs.{ability_ctx} "
            "From the minimap image identify: "
            "(1) which agent icons you see by their silhouette shape (both team and enemy), "
            "(2) any utility visible: Reyna eyes, Sova recon bolts/drone, Viper walls/orbs, "
            "Cypher cameras/tripwires, Killjoy turrets/nanoswarms/alarmbots, "
            "Sage walls, Omen smokes, Brimstone smokes, Phoenix walls. "
            "Give ONE tactical callout (max 12 words) naming the most dangerous agent or utility and their location. "
            "Be specific. Examples: 'Jett on B, Reyna eye blocking A main', "
            "'Sova drone mid, two enemies rotating B'. "
            "Reply with the callout text only."
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=80,
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
