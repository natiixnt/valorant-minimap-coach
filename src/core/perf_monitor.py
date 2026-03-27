"""
Loop timing monitor.

Tracks per-tick processing time and warns when the coach loop falls behind
its target interval (100 ms). Automatically adjusts sleep time to compensate.

Usage:
    mon = PerfMonitor(target_interval=0.1)
    while running:
        t = mon.tick_start()
        # ... do work ...
        mon.tick_end()           # logs if over budget
        time.sleep(mon.sleep_time())
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque


class PerfMonitor:
    def __init__(self, target_interval: float = 0.1, history: int = 60) -> None:
        self._target   = target_interval
        self._history: Deque[float] = deque(maxlen=history)
        self._t_start  = 0.0
        self._overruns = 0

    def tick_start(self) -> float:
        self._t_start = time.monotonic()
        return self._t_start

    def tick_end(self) -> float:
        elapsed = time.monotonic() - self._t_start
        self._history.append(elapsed)
        if elapsed > self._target * 1.5:
            self._overruns += 1
            if self._overruns % 10 == 1:
                print(f"[Perf] Tick took {elapsed*1000:.0f} ms "
                      f"(target {self._target*1000:.0f} ms) -- "
                      f"{self._overruns} overruns total")
        return elapsed

    def sleep_time(self) -> float:
        """Return how long to sleep so the full cycle matches target_interval."""
        if not self._history:
            return self._target
        last = self._history[-1]
        return max(0.0, self._target - last)

    def avg_ms(self) -> float:
        if not self._history:
            return 0.0
        return float(sum(self._history) / len(self._history)) * 1000

    def p95_ms(self) -> float:
        if not self._history:
            return 0.0
        arr = sorted(self._history)
        idx = max(0, int(len(arr) * 0.95) - 1)
        return arr[idx] * 1000
