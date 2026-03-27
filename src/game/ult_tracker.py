"""
Ultimate ability charge estimation per enemy slot.

Without game API access, we estimate ult charge from round count.
Valorant ults charge via kills and orbs. Conservative assumption:
  - Ult ready after ~6-8 rounds since last use (varies per agent).
  - After using ult, it takes ~5-7 rounds to recharge (average across all ults).
  - Orbs: 2 per map per round = ~2 points/round extra.

We track 5 enemy slots (indexed 0-4, matched to minimap detections over time)
and announce when multiple enemies likely have ults.

Limitations:
  - We don't know which agent each slot is playing.
  - We can't confirm ult usage (no HUD access).
  - This is a heuristic warning, not a guarantee.

Callout examples:
  "Caution: enemy ults likely charged -- round 7 since match start"
  "Multiple enemies may have ultimates"

Trigger conditions:
  - Announce at round 6, 9, 12 etc. (every 3 rounds after 6)
  - Announce only once per round
"""
from __future__ import annotations

from typing import Optional

# Rounds after which enemies are increasingly likely to have ults
_ULT_WARN_ROUNDS = {6, 9, 12, 15, 18, 21, 24}
_RECHARGE_ROUNDS = 6     # rounds per ult cycle after first use


class UltTracker:
    def __init__(self) -> None:
        self._last_warned_round = -1
        self._ult_used_round: Optional[int] = None   # round of last estimated ult use

    def update(self, round_num: int) -> Optional[str]:
        """
        Call once per round start.
        Returns a warning callout string or None.
        """
        if round_num == self._last_warned_round:
            return None

        warn = False

        # Warn at first ult threshold and every _RECHARGE_ROUNDS after
        if round_num in _ULT_WARN_ROUNDS:
            warn = True
        elif self._ult_used_round is not None:
            rounds_since_use = round_num - self._ult_used_round
            if rounds_since_use > 0 and rounds_since_use % _RECHARGE_ROUNDS == 0:
                warn = True

        if warn:
            self._last_warned_round = round_num
            if round_num <= 8:
                return "Caution: enemy ultimates may be charged. Check corners."
            else:
                return "Enemy ults likely charged this round. Play safe."
        return None

    def on_round_end(self, round_num: int, ult_likely_used: bool = False) -> None:
        """
        Call at each round end.
        ult_likely_used: set True if we observed an ultimate-like event this round
        (e.g. Reyna dismiss detected, Jett dash flurry, etc.)
        """
        if ult_likely_used:
            self._ult_used_round = round_num

    def reset(self) -> None:
        self._last_warned_round = -1
        self._ult_used_round    = None
