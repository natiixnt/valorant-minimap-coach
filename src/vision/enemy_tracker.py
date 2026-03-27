import copy
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Two blobs within this normalized distance are considered the same enemy
_MATCH_RADIUS = 0.12


@dataclass
class TrackedEnemy:
    position: Tuple[float, float]
    first_seen: float
    last_seen: float = field(init=False)
    alpha: float = field(init=False)

    def __post_init__(self) -> None:
        self.last_seen = self.first_seen
        self.alpha = 1.0


class EnemyTracker:
    """
    Maintains identity of enemy blobs across frames.
    Positions fade and are pruned after fade_after seconds without re-detection.

    Thread safety: update() is called from the coach thread; tick() is called
    from the UI/tkinter thread. A lock protects _tracked across both.
    """

    def __init__(self, fade_after: float = 5.0) -> None:
        self.fade_after = fade_after
        self._tracked: List[TrackedEnemy] = []
        self._lock = threading.Lock()

    def update(self, positions: List[Tuple[float, float]]) -> None:
        """
        Called from the coach thread with freshly detected positions.
        Matches against existing tracked enemies by proximity, adds new ones.
        """
        now = time.time()
        with self._lock:
            for pos in positions:
                best, best_dist = None, _MATCH_RADIUS
                for e in self._tracked:
                    d = ((pos[0] - e.position[0]) ** 2 + (pos[1] - e.position[1]) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist, best = d, e
                if best is not None:
                    best.position = pos
                    best.last_seen = now
                    best.alpha = 1.0
                else:
                    self._tracked.append(TrackedEnemy(position=pos, first_seen=now))

    def tick(self, now: Optional[float] = None) -> List[TrackedEnemy]:
        """
        Called from the UI thread (via after()).
        Decays alpha, prunes dead enemies, returns surviving list.
        """
        t = now if now is not None else time.time()
        with self._lock:
            live = []
            for e in self._tracked:
                e.alpha = max(0.0, 1.0 - (t - e.last_seen) / self.fade_after)
                if e.alpha > 0.0:
                    live.append(e)
            self._tracked = live
            return [copy.copy(e) for e in live]
