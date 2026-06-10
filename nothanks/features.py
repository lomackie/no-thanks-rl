"""Turn a ``State`` into a fixed-length, mover-relative feature vector.

Everything is encoded **from the perspective of the player to move**: the mover
is "seat 0", and opponents follow in turn order (``to_move+1``, ``to_move+2`` …).
That makes one network serve every seat, and it lines up with the value head,
which predicts a vector of expected final scores in the *same* seat order (seat 0
= the mover). See :mod:`nothanks.valuefn`.

A feature vector is specific to a player count: there is one card block per seat,
so a net trained for 3 players takes 3-player features. Card-value features are
multi-hot over the 33 deck slots (cards 3..35 → index ``card-3``): a *separate*
block for each seat's holdings (in mover-relative order), one for the pile, and a
one-hot for the face-up card. Each opponent gets its own block on purpose —
aggregating opponents' cards would alias states the per-seat value head must tell
apart. Those multi-hots are what let the net reason about runs.
"""

from __future__ import annotations

import numpy as np

from .engine import DECK_HIGH, DECK_LOW, State, score_cards, score_delta

N_CARDS = DECK_HIGH - DECK_LOW + 1  # 33 deck slots

# Rough normalisers so scalar inputs land near unit scale.
_CHIP_NORM = 20.0
_SCORE_NORM = 80.0
_CARD_NORM = float(DECK_HIGH)


def _card_multihot(cards) -> np.ndarray:
    v = np.zeros(N_CARDS, dtype=np.float32)
    for c in cards:
        v[c - DECK_LOW] = 1.0
    return v


def seat_order(to_move: int, n: int) -> list[int]:
    """Absolute seat indices in mover-relative order (mover first)."""
    return [(to_move + k) % n for k in range(n)]


def feature_dim(n_players: int) -> int:
    """Length of the vector returned by :func:`features` for ``n_players``."""
    # Per-seat held block (n) + pile + active one-hot, then per-seat (chips, score)
    # plus the mover's delta and the pot.
    return (n_players + 2) * N_CARDS + 2 * n_players + 2


def features(s: State) -> np.ndarray:
    """Mover-relative feature vector for non-terminal state ``s`` (float32)."""
    p = s.to_move
    n = s.n_players
    order = seat_order(p, n)

    # One card-holding block per seat, mover first, then opponents in turn order.
    held = [_card_multihot(s.cards[q]) for q in order]
    remaining = _card_multihot(s.remaining)
    active_onehot = np.zeros(N_CARDS, dtype=np.float32)
    active_onehot[s.active - DECK_LOW] = 1.0

    # Per-seat scalars in mover-relative order: chips and current card-score.
    chips = np.array([s.chips[q] / _CHIP_NORM for q in order], dtype=np.float32)
    scores = np.array(
        [score_cards(s.cards[q]) / _SCORE_NORM for q in order], dtype=np.float32
    )

    delta = np.float32(score_delta(s.cards[p], s.active) / _CARD_NORM)
    pot = np.float32(s.pot / _CHIP_NORM)

    return np.concatenate(
        [
            *held,
            remaining,
            active_onehot,
            chips,
            scores,
            np.array([delta, pot], dtype=np.float32),
        ]
    )
