import time

import mss
import numpy as np
from dataclasses import dataclass


@dataclass
class MinimapFrame:
    data: np.ndarray
    timestamp: float
    region: dict


class ScreenCapture:
    def __init__(self, config: dict):
        self.region = config["minimap"]["region"]
        self._sct = mss.mss()

    def capture(self) -> MinimapFrame:
        raw = self._sct.grab(self.region)
        # mss returns BGRA; drop alpha for OpenCV
        frame = np.array(raw)[:, :, :3]
        return MinimapFrame(data=frame, timestamp=time.time(), region=self.region)

    def close(self):
        self._sct.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
