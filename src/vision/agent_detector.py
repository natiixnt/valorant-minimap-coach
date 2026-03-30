"""
Enemy agent auto-detection via Claude vision.

Takes a screenshot during round 1 buy phase (when all 10 agent portraits
are visible in the HUD health bar strip at the top of the screen) and
asks Claude to identify the 5 enemy team agents.

Lifecycle:
- Called once at game start (round 1 buy phase, ~3s after round start sound)
- Runs in a background thread -- never blocks the coach loop
- On success: updates EnemyAgentTracker and prints detected agents
- On failure: silent -- coach continues with no composition info

Result is cached for the entire game; re-detection is triggered if the
map changes (i.e. a new game starts).
"""
from __future__ import annotations

import base64
import threading
import time
from typing import TYPE_CHECKING, Callable, List, Optional

import anthropic
import cv2
import mss
import numpy as np

from src.audio.agent_classifier import ALL_AGENTS

if TYPE_CHECKING:
    from src.game.enemy_agents import EnemyAgentTracker

# Delay after round-start trigger before taking the screenshot.
# Gives the game HUD time to fully appear after the round start sound.
_SCREENSHOT_DELAY = 3.0

_AGENT_LIST = ", ".join(sorted(ALL_AGENTS))


class AgentDetector:
    """
    Detects enemy team composition once per game using Claude vision.

    Usage:
        detector = AgentDetector(config)
        # On round 1 start:
        detector.detect_async(tracker, on_done=lambda agents: print(agents))
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("map_detection", {})
        self._model = cfg.get("model", "claude-haiku-4-5-20251001")
        self._client: Optional[anthropic.Anthropic] = None
        self._lock = threading.Lock()
        self._detected_this_game = False
        self._sct = mss.mss()

    def reset(self) -> None:
        """Call when a new game starts (map change) to allow re-detection."""
        with self._lock:
            self._detected_this_game = False

    def detect_async(
        self,
        tracker: "EnemyAgentTracker",
        on_done: Optional[Callable[[List[str]], None]] = None,
    ) -> None:
        """
        Trigger background detection. Returns immediately.
        Calls on_done(agent_list) on the same background thread when finished.
        Safe to call multiple times -- only runs once per game.
        """
        with self._lock:
            if self._detected_this_game:
                return
            self._detected_this_game = True  # reserve slot

        t = threading.Thread(
            target=self._run,
            args=(tracker, on_done),
            daemon=True,
            name="AgentDetector",
        )
        t.start()

    # ------------------------------------------------------------------

    def _run(
        self,
        tracker: "EnemyAgentTracker",
        on_done: Optional[Callable[[List[str]], None]],
    ) -> None:
        time.sleep(_SCREENSHOT_DELAY)
        img = self._grab()
        agents = self._claude_identify(img)
        if agents:
            tracker.set_agents(agents)
            print(f"[AgentDetector] Enemy team: {', '.join(agents)}")
        else:
            with self._lock:
                self._detected_this_game = False  # allow retry next round
            print("[AgentDetector] Could not identify enemy agents from screen.")
        if on_done:
            on_done(agents or [])

    def _grab(self) -> np.ndarray:
        raw = self._sct.grab(self._sct.monitors[0])
        img = np.array(raw)[:, :, :3]
        h, w = img.shape[:2]
        return cv2.resize(img, (w // 4, h // 4))

    def _claude_identify(self, img: np.ndarray) -> List[str]:
        if self._client is None:
            self._client = anthropic.Anthropic()
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return []
        b64 = base64.b64encode(buf).decode()

        prompt = (
            "This is a Valorant game screenshot taken during the buy phase. "
            "At the TOP of the screen there is a row of agent portrait icons -- "
            "left side is your team, right side is the ENEMY team (5 icons). "
            "Identify the 5 ENEMY team agents. "
            f"Known agents: {_AGENT_LIST}. "
            "Reply with ONLY the 5 agent names separated by commas, lowercase. "
            "Example: jett, reyna, brimstone, sage, viper. "
            "If you cannot see them clearly, reply: unknown"
        )

        for attempt in range(2):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=40,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image",
                             "source": {"type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": b64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                )
                if not resp.content:
                    return []
                raw = resp.content[0].text.strip().lower()
                if raw == "unknown":
                    return []
                agents = [a.strip() for a in raw.split(",")]
                valid = [a for a in agents if a in ALL_AGENTS]
                if len(valid) >= 3:   # accept partial if at least 3 recognised
                    return valid[:5]
            except Exception as e:
                if attempt == 0:
                    print(f"[AgentDetector] Claude error (retrying): {e}")
                    time.sleep(1.0)
                else:
                    print(f"[AgentDetector] Claude error: {e}")
        return []

    def close(self) -> None:
        self._sct.close()
