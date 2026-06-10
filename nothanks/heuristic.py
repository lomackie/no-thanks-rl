"""A simple, run-aware heuristic policy.

This is the baseline opponent and the default rollout policy. It is intentionally
cheap and explainable, not optimal.

Reasoning: taking the card right now changes the mover's score by
``score_delta(card) - pot`` (cards go up by the run-aware delta; chips go up by
the pot, and chips are worth -1 each). If that immediate change is favourable —
i.e. ``<= threshold`` — we take. A chipless player is forced to take.
"""

from __future__ import annotations

from .engine import State, score_delta


def take_cost(s: State) -> int:
    """Immediate score change for the mover if they take now (lower = better)."""
    return score_delta(s.cards[s.to_move], s.active) - s.pot


def heuristic_action(s: State, threshold: int = 0) -> str:
    if s.chips[s.to_move] == 0:
        return "take"
    return "take" if take_cost(s) <= threshold else "pass"
