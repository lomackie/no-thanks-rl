"""Hidden removed cards: information sets, determinization, and PIMC evaluation.

Everything before this module treats ``State.remaining`` (the unflipped pile) as
known. That is the *god view*. In the real game **9 cards are removed face-down**
and never revealed, so a player cannot see which cards are still in the pile and
which were set aside. This module models that single piece of hidden information.

What a player actually knows
----------------------------
All of the dynamic state except the *composition* of the pile is public: chips,
captured cards, the face-up card, the pot, and whose turn it is. From the public
record a player can also derive:

* ``seen``  — every card that has been flipped (captured by anyone, plus the
  active card);
* ``unseen`` = ``deck − seen`` — the cards that are *either* still in the pile or
  among the removed nine; the player cannot tell which;
* ``pile_remaining`` = ``(len(deck) − n_removed) − len(seen)`` — how many of the
  unseen cards are still to be drawn (the rest of ``unseen`` are the removed
  ones). This count is public because the total number of cards that get played
  is fixed by the rules.

An :class:`InfoSet` is exactly this public knowledge. The hidden state is *which*
``pile_remaining`` of the ``unseen`` cards form the real pile.

Determinization (PIMC)
----------------------
:func:`determinize` samples one consistent world — a random ``pile_remaining``
subset of ``unseen`` becomes ``State.remaining``; the rest are treated as removed
(they simply never appear). The result is an ordinary perfect-information
:class:`~nothanks.engine.State` that *any* existing evaluator can analyse. Averaging
a per-world evaluation over many sampled worlds is **Perfect-Information Monte
Carlo** (PIMC): :func:`evaluate_determinized`.

PIMC's blind spot. Each world is solved as if its pile were common knowledge, so
PIMC cannot value information itself (it suffers the usual *strategy fusion* /
*non-locality* artefacts). It is nonetheless the standard, strong baseline for
hidden-information move analysis and the natural next step before full IS-MCTS.

Consistency check. With ``n_removed == 0`` there is nothing hidden:
``pile_remaining == len(unseen)``, so the only determinization is the true pile
and :func:`evaluate_determinized` reduces exactly to the underlying evaluator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from .engine import State, full_deck

# A per-world evaluator: a concrete State -> the usual engine-eval dict (with at
# least "actions" {action: score-vector} and "mover_ev" {action: float}). Any of
# solver.evaluate, montecarlo.evaluate_mc, valuefn.evaluate_v fits this shape.
Evaluator = Callable[[State], dict]


@dataclass(frozen=True)
class InfoSet:
    """A player's public knowledge of a position — everything but the pile's identity.

    ``deck`` is the full card universe *before* removal and ``n_removed`` the
    number set aside, so the standard game is ``deck=frozenset(full_deck())`` with
    ``n_removed=9``. The remaining fields mirror the public parts of
    :class:`~nothanks.engine.State`.
    """

    chips: tuple[int, ...]
    cards: tuple[frozenset[int], ...]
    active: int | None
    pot: int
    to_move: int
    deck: frozenset[int]
    n_removed: int

    @property
    def n_players(self) -> int:
        return len(self.chips)


def info_from_state(s: State, n_removed: int, deck: frozenset[int] | None = None) -> InfoSet:
    """Project a god-view :class:`State` onto the public knowledge of an observer.

    Drops the pile's composition (``s.remaining``) but keeps everything public.
    ``deck`` defaults to the standard 3..35 universe; pass a custom set for the
    small testbed games. The result is consistent with ``s`` — the true pile is
    always one of the worlds :func:`determinize` can draw.
    """
    if deck is None:
        deck = frozenset(full_deck())
    return InfoSet(
        chips=s.chips,
        cards=s.cards,
        active=s.active,
        pot=s.pot,
        to_move=s.to_move,
        deck=deck,
        n_removed=n_removed,
    )


def seen(info: InfoSet) -> frozenset[int]:
    """Every card that has been flipped: captured by anyone, plus the active card."""
    s: set[int] = set()
    for held in info.cards:
        s |= held
    if info.active is not None:
        s.add(info.active)
    return frozenset(s)


def unseen(info: InfoSet) -> frozenset[int]:
    """Cards still hidden: either in the pile or among the removed nine."""
    return info.deck - seen(info)


def pile_remaining(info: InfoSet) -> int:
    """How many unseen cards are still in the draw pile (the rest are removed)."""
    return (len(info.deck) - info.n_removed) - len(seen(info))


def legal_actions(info: InfoSet) -> tuple[str, ...]:
    """The mover's legal actions, derivable from *public* knowledge alone.

    Mirrors :func:`nothanks.engine.legal_actions` but reads the :class:`InfoSet`:
    the action set depends only on the face-up card and the mover's chip count,
    both of which are public, so it is identical in every determinized world.
    """
    if info.active is None:
        return ()
    if info.chips[info.to_move] > 0:
        return ("take", "pass")
    return ("take",)  # a chipless player is forced to take


def determinize(info: InfoSet, rng: random.Random) -> State:
    """Sample one consistent world: a concrete :class:`State` with a guessed pile.

    A random ``pile_remaining`` subset of the unseen cards becomes the pile; the
    rest are implicitly the removed cards (they never appear). All public fields
    are carried over unchanged.
    """
    if info.active is None:
        raise ValueError("cannot determinize a terminal info set")
    k = pile_remaining(info)
    candidates = sorted(unseen(info))
    if not 0 <= k <= len(candidates):
        raise ValueError(f"inconsistent info set: pile_remaining={k}, unseen={len(candidates)}")
    pile = rng.sample(candidates, k)
    return State(
        chips=info.chips,
        cards=info.cards,
        active=info.active,
        pot=info.pot,
        to_move=info.to_move,
        remaining=frozenset(pile),
    )


def evaluate_determinized(
    info: InfoSet,
    evaluator: Evaluator,
    n_worlds: int = 200,
    rng: random.Random | None = None,
) -> dict:
    """PIMC move analysis: average a per-world evaluation over sampled worlds.

    For each of ``n_worlds`` sampled worlds we determinize the pile and run
    ``evaluator`` on the resulting perfect-information state, then average the
    per-action score vectors and the mover's own EV across worlds.

    The legal action set is public (it depends only on the active card and the
    mover's chips), so it is identical in every world. ``stderr`` is the standard
    error of the across-world mean of the mover's EV — it captures the **belief
    uncertainty** over which cards are hidden (and, if ``evaluator`` itself
    samples, its sampling noise too); a gap between actions is only meaningful
    when it comfortably exceeds the combined ``stderr``.

    The returned ``actions`` vectors are in whatever seat frame ``evaluator``
    uses (absolute for the solver / Monte-Carlo evaluators, mover-relative for
    :func:`nothanks.valuefn.evaluate_v`); ``mover_ev`` is unambiguous either way.
    """
    if info.active is None:
        raise ValueError("terminal info set has no moves to evaluate")
    rng = rng or random.Random()
    n = info.n_players

    vec_sum: dict[str, list[float]] = {}
    own_sum: dict[str, float] = {}
    own_sumsq: dict[str, float] = {}
    for _ in range(n_worlds):
        s = determinize(info, rng)
        ev = evaluator(s)
        for a, vec in ev["actions"].items():
            if a not in vec_sum:
                vec_sum[a] = [0.0] * n
                own_sum[a] = 0.0
                own_sumsq[a] = 0.0
            for i in range(n):
                vec_sum[a][i] += vec[i]
            own = ev["mover_ev"][a]
            own_sum[a] += own
            own_sumsq[a] += own * own

    actions = {a: tuple(x / n_worlds for x in v) for a, v in vec_sum.items()}
    mover_ev = {a: own_sum[a] / n_worlds for a in own_sum}
    stderr = {}
    for a in own_sum:
        var = max(own_sumsq[a] / n_worlds - mover_ev[a] ** 2, 0.0)
        stderr[a] = math.sqrt(var / n_worlds)

    best_action = min(mover_ev, key=lambda a: mover_ev[a])
    return {
        "to_move": info.to_move,
        "actions": actions,
        "mover_ev": mover_ev,
        "stderr": stderr,
        "best_action": best_action,
        "n_worlds": n_worlds,
        "pile_remaining": pile_remaining(info),
        "n_hidden": len(unseen(info)),
    }


def determinized_action(
    info: InfoSet,
    evaluator: Evaluator,
    n_worlds: int = 200,
    rng: random.Random | None = None,
) -> str:
    """The **honest** playing policy: pick a move from public information only.

    This is the non-cheating counterpart to :func:`nothanks.valuefn.greedy_action`,
    which reads the god-view ``State`` (and so its ``remaining`` pile, revealing
    which cards were removed). Here the input is an :class:`InfoSet`, so the
    removed cards are *never* inspected: the move is chosen by PIMC — averaging
    ``evaluator`` over worlds consistent with the public record
    (:func:`evaluate_determinized`) and taking the action that minimises the
    mover's own expected final score.

    Because the decision is a pure function of ``info`` (plus ``rng``), two
    distinct god-view states that share an info set — i.e. differ only in which
    nine cards are hidden — yield the *same* action. Forced moves are returned
    directly, with no sampling.
    """
    acts = legal_actions(info)
    if not acts:
        raise ValueError("terminal info set has no action to choose")
    if len(acts) == 1:
        return acts[0]
    return evaluate_determinized(info, evaluator, n_worlds=n_worlds, rng=rng)["best_action"]
