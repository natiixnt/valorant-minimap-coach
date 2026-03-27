"""
Enemy agent composition tracker.

Tracks which 5 agents the enemy team is playing and which are still alive
this round. When a footstep shoe type is detected, provides the list of
alive enemy agents that match -- narrowing "heavy footstep" to
"Breach or Brimstone" when those are the only heavy-boot enemies alive.

Usage:
    tracker = EnemyAgentTracker()
    tracker.set_agents(["jett", "reyna", "brimstone", "sage", "viper"])

    # each round start:
    tracker.new_round()

    # when an enemy is confirmed dead (optional, improves precision):
    tracker.mark_dead("jett")

    # in audio callout:
    candidates = tracker.candidates_for_shoe_type("heavy")
    # -> ["brimstone", "viper", "sage"]  (all heavy-boot agents still alive)
"""
from __future__ import annotations

from typing import List, Optional

from src.audio.agent_classifier import AGENT_SHOE_TYPE

# Display name overrides (some agents are nicer capitalised differently)
_DISPLAY: dict[str, str] = {
    "kayo": "KAY/O",
}


def agent_display(name: str) -> str:
    """Human-readable agent name for TTS/display."""
    return _DISPLAY.get(name.lower(), name.capitalize())


class EnemyAgentTracker:
    """Tracks enemy team composition and per-round survival state."""

    def __init__(self) -> None:
        self._agents: List[str] = []          # lowercase agent names, up to 5
        self._dead: set[str] = set()          # agents confirmed dead this round

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_agents(self, agents: List[str]) -> None:
        """Set the full enemy team (call once at game start or composition change)."""
        self._agents = [a.strip().lower() for a in agents if a.strip()][:5]
        self._dead.clear()

    def clear(self) -> None:
        self._agents.clear()
        self._dead.clear()

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def new_round(self) -> None:
        """Reset alive state at the start of each round."""
        self._dead.clear()

    def mark_dead(self, agent: str) -> None:
        """Mark an enemy agent as eliminated this round."""
        self._dead.add(agent.strip().lower())

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        return len(self._agents) > 0

    def alive_agents(self) -> List[str]:
        return [a for a in self._agents if a not in self._dead]

    def candidates_for_shoe_type(self, shoe_type: str) -> List[str]:
        """
        Return alive enemy agents whose shoe type matches.
        Returns [] if no agents configured or none match.
        """
        return [
            a for a in self.alive_agents()
            if AGENT_SHOE_TYPE.get(a) == shoe_type
        ]

    def callout_for_shoe_type(self, shoe_type: str) -> Optional[str]:
        """
        Build a display string narrowing down the agent(s) for a shoe type.

        Returns:
            "Breach"               -- exactly one candidate
            "Breach or Brimstone"  -- two candidates
            None                   -- zero or 3+ candidates (fall back to generic)
        """
        candidates = self.candidates_for_shoe_type(shoe_type)
        if not candidates:
            return None
        if len(candidates) == 1:
            return agent_display(candidates[0])
        if len(candidates) == 2:
            return f"{agent_display(candidates[0])} or {agent_display(candidates[1])}"
        # 3+ candidates -- not narrow enough to be useful, let caller use generic
        return None
