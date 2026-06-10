"""Core No Thanks game engine.

Design notes
------------
The whole game is perfect information *except* for the identity of the removed
cards and the draw order. We model that single source of randomness explicitly:
the unflipped pile is a ``frozenset`` (``remaining``) and the act of flipping the
next card is a *chance event* that draws uniformly from it. The same ``State``
type is therefore used both for:

  * concrete play (``step`` samples a single outcome), and
  * exact solving (``take_outcomes`` enumerates every outcome with its
    probability).

Chip counts are always fully known state — they are derivable from public
actions, so there is no reason to hide them.

Scoring: each player's final score is the sum of the *lowest card in each
consecutive run* of their captured cards, minus their chip count. Lower is
better. A card ``c`` is the bottom of a run iff ``c - 1`` is not also held, which
gives the one-line ``score_cards`` below.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

# Standard starting chips by player count (rulebook).
STARTING_CHIPS = {3: 11, 4: 11, 5: 11, 6: 9, 7: 7}

# Standard deck: cards 3..35 inclusive.
DECK_LOW = 3
DECK_HIGH = 35


def full_deck() -> list[int]:
    return list(range(DECK_LOW, DECK_HIGH + 1))


@dataclass(frozen=True)
class State:
    """An immutable game state.

    A state always presents the mover with a decision on ``active`` unless the
    game is over, in which case ``active is None``.
    """

    chips: tuple[int, ...]            # chips held by each player
    cards: tuple[frozenset[int], ...]  # cards captured by each player
    active: int | None               # the face-up card under decision (None => terminal)
    pot: int                         # chips sitting on the active card
    to_move: int                     # index of the player to act
    remaining: frozenset[int]        # unflipped pile (order is irrelevant)

    @property
    def n_players(self) -> int:
        return len(self.chips)


def is_terminal(s: State) -> bool:
    return s.active is None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_cards(cards) -> int:
    """Sum of the lowest card in each maximal consecutive run."""
    cs = set(cards)
    return sum(c for c in cs if (c - 1) not in cs)


def score_delta(cards, c: int) -> int:
    """Change in card-score from adding card ``c`` to ``cards``.

    Equivalent to ``score_cards(cards | {c}) - score_cards(cards)`` but O(1).
    Negative values mean capturing ``c`` *lowers* (improves) the card score,
    which happens when it bridges or extends an existing run downward.
    """
    if c in cards:
        return 0
    gained = 0 if (c - 1) in cards else c        # does c become a new run-bottom?
    displaced = (c + 1) if (c + 1) in cards else 0  # c+1 stops being a run-bottom
    return gained - displaced


def final_scores(s: State) -> tuple[int, ...]:
    """Final scores for every player (lower is better)."""
    return tuple(score_cards(s.cards[i]) - s.chips[i] for i in range(s.n_players))


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def initial_state(
    n_players: int,
    deck: list[int],
    start_chips: int | None = None,
) -> State:
    """Build the opening state from an ordered ``deck`` (top card flipped first).

    ``deck`` is the draw pile *after* any cards have been removed; its first
    element becomes the opening face-up card. For solver/oracle use you can pass
    a small custom deck (e.g. ``[3,4,5,6,7]``) and a small ``start_chips``.
    """
    if start_chips is None:
        start_chips = STARTING_CHIPS[n_players]
    active = deck[0]
    remaining = frozenset(deck[1:])
    chips = tuple(start_chips for _ in range(n_players))
    cards: tuple[frozenset[int], ...] = tuple(frozenset() for _ in range(n_players))
    return State(chips, cards, active, 0, 0, remaining)


def new_game(
    n_players: int,
    n_removed: int = 9,
    start_chips: int | None = None,
    rng: random.Random | None = None,
) -> State:
    """A freshly shuffled standard game with ``n_removed`` cards set aside."""
    rng = rng or random.Random()
    deck = full_deck()
    rng.shuffle(deck)
    if n_removed:
        deck = deck[n_removed:]
    return initial_state(n_players, deck, start_chips)


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #

def legal_actions(s: State) -> tuple[str, ...]:
    if s.active is None:
        return ()
    if s.chips[s.to_move] > 0:
        return ("take", "pass")
    return ("take",)  # a chipless player is forced to take


def apply_pass(s: State) -> State:
    """Pay one chip onto the active card and pass to the next player."""
    p = s.to_move
    chips = list(s.chips)
    chips[p] -= 1
    return replace(
        s,
        chips=tuple(chips),
        pot=s.pot + 1,
        to_move=(p + 1) % s.n_players,
    )


def take_outcomes(s: State) -> list[tuple[float, State]]:
    """All ``(probability, next_state)`` results of the mover taking the card.

    The taker collects the card and the pot, then flips the next card — a chance
    event over ``remaining``. If the pile is empty the game ends.
    """
    p = s.to_move
    chips = list(s.chips)
    chips[p] += s.pot
    cards = list(s.cards)
    cards[p] = cards[p] | {s.active}
    chips_t = tuple(chips)
    cards_t = tuple(cards)

    if not s.remaining:
        terminal = replace(s, chips=chips_t, cards=cards_t, active=None, pot=0)
        return [(1.0, terminal)]

    prob = 1.0 / len(s.remaining)
    outcomes = []
    for c in s.remaining:
        nxt = replace(
            s,
            chips=chips_t,
            cards=cards_t,
            active=c,
            pot=0,
            remaining=s.remaining - {c},
            to_move=p,  # the taker keeps acting on the new card
        )
        outcomes.append((prob, nxt))
    return outcomes


def step(s: State, action: str, rng: random.Random) -> State:
    """Apply ``action`` for concrete play, sampling chance outcomes."""
    if action == "pass":
        return apply_pass(s)
    if action == "take":
        outcomes = take_outcomes(s)
        if len(outcomes) == 1:
            return outcomes[0][1]
        r = rng.random()
        cum = 0.0
        for prob, nxt in outcomes:
            cum += prob
            if r <= cum:
                return nxt
        return outcomes[-1][1]
    raise ValueError(f"unknown action {action!r}")
