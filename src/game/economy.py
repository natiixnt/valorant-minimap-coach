"""
Round economy tracker and buy recommendation.

Without game API access we estimate enemy economy from round results.
Assumptions follow standard Valorant economy rules:
  - Loss bonus: 1900 / 2400 / 2900 / 3400 / 3900 (streak-based, resets on win)
  - Win bonus: 3000
  - Starting credits: 800 (pistol round), carry from previous
  - Eco pistol loadout: ~800-1200 creds
  - Force buy: ~2000-2600 creds
  - Full buy: ~3800-4500 creds

Round result is detected via:
  - Enemy count going to 0 (our kills) = WIN
  - Spike defuse (implicit from round state = WIN on defense)
  - Time expiry with spike planted = WIN on attack / LOSS on defense
  Imperfect, but directionally correct for economy estimation.

Output:
  - enemy_buy_status: "eco" | "force" | "full" | "unknown"
  - enemy_credits: estimated credit count
  - our_recommendation: "save" | "force" | "full"
  - voice: ready-to-speak callout
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EconomyStatus:
    round_num: int
    enemy_credits: int
    enemy_buy: str          # "eco" | "force" | "full" | "unknown"
    our_recommendation: str # "save" | "force" | "full"
    voice: str


# Credit thresholds
_ECO_MAX   = 1800
_FORCE_MAX = 3200

# Loss bonus by streak (1 loss, 2 losses, 3+)
_LOSS_BONUS = [1900, 2400, 2900, 3400, 3900]
_WIN_BONUS  = 3000
_ROUND_BONUS_KILL = 200
_STARTING = 800    # round 1 pistol

# Spike plant bonus
_PLANT_BONUS = 300


class EconomyTracker:
    def __init__(self) -> None:
        self._round = 1
        self._enemy_credits = _STARTING
        self._our_credits = _STARTING
        self._enemy_loss_streak = 0
        self._our_loss_streak = 0
        self._history: List[EconomyStatus] = []

    # ------------------------------------------------------------------
    # External triggers
    # ------------------------------------------------------------------

    def on_round_end(self, our_win: bool, kills_by_us: int = 0) -> None:
        """
        Update economy after a round ends.
        our_win: True if local team won the round.
        kills_by_us: estimated kills (each worth 200 creds to enemies if lost).
        """
        if our_win:
            # We won: enemies get loss bonus, we get 3000 win
            self._enemy_loss_streak += 1
            streak_idx = min(self._enemy_loss_streak - 1, len(_LOSS_BONUS) - 1)
            loss = _LOSS_BONUS[streak_idx]
            self._enemy_credits = max(0, self._enemy_credits - 3500) + loss
            # Deduct kills from enemy economy (rough: assume ~3500 per full buy)
            self._our_credits = min(9000, self._our_credits + _WIN_BONUS)
            self._our_loss_streak = 0
        else:
            # We lost: enemies get win bonus, we get loss bonus
            self._enemy_loss_streak = 0
            self._enemy_credits = min(9000, self._enemy_credits + _WIN_BONUS)
            self._our_loss_streak += 1
            streak_idx = min(self._our_loss_streak - 1, len(_LOSS_BONUS) - 1)
            loss = _LOSS_BONUS[streak_idx]
            self._our_credits = max(0, self._our_credits - 3500) + loss

        self._round += 1

        # Cap at 9000
        self._enemy_credits = min(9000, self._enemy_credits)
        self._our_credits = min(9000, self._our_credits)

    def on_spike_planted(self) -> None:
        """Attackers receive plant bonus."""
        self._enemy_credits = min(9000, self._enemy_credits + _PLANT_BONUS)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def status(self) -> EconomyStatus:
        ec = self._enemy_credits
        if ec <= _ECO_MAX:
            enemy_buy = "eco"
        elif ec <= _FORCE_MAX:
            enemy_buy = "force"
        else:
            enemy_buy = "full"

        oc = self._our_credits
        if oc <= _ECO_MAX:
            our_rec = "save"
        elif oc <= _FORCE_MAX:
            our_rec = "force"
        else:
            our_rec = "full"

        voice = self._build_voice(enemy_buy, our_rec, ec)
        return EconomyStatus(
            round_num=self._round,
            enemy_credits=ec,
            enemy_buy=enemy_buy,
            our_recommendation=our_rec,
            voice=voice,
        )

    def _build_voice(self, enemy_buy: str, our_rec: str, enemy_credits: int) -> str:
        eco_map = {"eco": "on eco", "force": "force buying", "full": "full buying"}
        rec_map = {"save": "save this round", "force": "force buy", "full": "full buy"}
        return (
            f"Round {self._round}: enemies probably {eco_map[enemy_buy]} "
            f"(~{enemy_credits} creds). We should {rec_map[our_rec]}."
        )

    def reset(self) -> None:
        self.__init__()
