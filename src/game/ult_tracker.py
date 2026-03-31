"""
Ultimate ability charge estimation per enemy slot.

Without game API access, we estimate ult charge from round count.
Valorant ults charge via kills (1pt), deaths (1pt), orb pickups (1pt),
spike plant (1pt), spike defuse completion (1pt).
  - Ult point costs range from 6 (Reyna, Cypher) to 9 (Breach) - varies per agent.
  - Typical recharge: 3-5 rounds with average kill count + 1-2 orbs per half.
  - Orbs: 2 per map per HALF (not per round) - collected once, not respawning.
    Fracture has 4 orbs per half.

We warn conservatively (better to warn too early than miss).

Limitations:
  - We don't know which agent each slot is playing.
  - We can't confirm ult usage (no HUD access).
  - This is a heuristic warning, not a guarantee.

Callout examples:
  "Caution: enemy ults likely charged - round 4"
  "Enemy ults likely charged this round. Play safe."

Trigger conditions:
  - First warn at round 4 (earliest realistic charge at 6pt cost + 1 orb)
  - Re-warn every 4 rounds after detected use
  - Announce only once per round
"""
from __future__ import annotations

from typing import Optional

# Rounds after which enemies are increasingly likely to have ults.
# Earliest: 6pt ult + 1 orb pickup = 5 kills needed, achievable by round 3-4.
# Using round 4 as conservative first warning.
_ULT_WARN_ROUNDS = {4, 7, 10, 13, 16, 19, 22}
_RECHARGE_ROUNDS = 4     # rounds per ult cycle - verified typical 3-5, using 4


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
            if round_num <= 5:
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
