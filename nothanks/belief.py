"""Belief-exact dynamics and exploitability under genuinely hidden removed cards.

:mod:`nothanks.exploit` grades a policy on the *perfect-information* testbed: its
best-responder sees the true ``State`` (and so the removed cards), which is not a
legal strategy in the real game. This module computes the **belief-correct** best
response — one constrained to act on :class:`~nothanks.imperfect.InfoSet`\\ s, never
the hidden nine — and the resulting exploitability.

The key reduction
-----------------
For policies that read only public knowledge (the heuristic, the determinized
bot, any info-set policy), the hidden game is *exactly* a Markov game on info
sets. Two facts make this work:

* **Chance marginalises to uniform-over-unseen.** Under a uniform random removal,
  the next flipped card is uniform over ``unseen(info)`` — each unseen card has
  probability ``1/|unseen|`` of being drawn next (a removed card simply never
  exhausts the pile). The game ends when ``pile_remaining`` reaches 0. So we never
  enumerate which nine cards are hidden; the belief is summarised by the public
  counts. (With ``n_removed == 0`` this is the ordinary draw and everything below
  reduces to :mod:`nothanks.exploit`.)
* **The belief is strategy-independent.** Nature draws the removed set
  independently of play, so the posterior over it given a node depends only on
  which cards have been *seen* — a deterministic function of the info set. The
  belief is therefore pinned by the info set alone, which is exactly why a plain
  per-info-set backward induction yields a correct best response: there is no need
  for the reach-probability weighting that imperfect-information best responses
  normally require.

Scoring uses only captured cards and chips (both public), so terminal values are
public too. Everything here is exact and, like the solver, tractable only on
small games.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .engine import score_cards, score_delta
from .imperfect import InfoSet, legal_actions, pile_remaining, unseen

# A belief-game policy maps a non-terminal info set to a legal action. Unlike a
# ``montecarlo.Policy`` (which takes a god-view ``State``) this *cannot* peek at
# the removed cards — that is the whole point.
InfoPolicy = Callable[[InfoSet], str]


def is_terminal(info: InfoSet) -> bool:
    return info.active is None


def final_scores(info: InfoSet) -> tuple[float, ...]:
    """Final scores from public state alone (captured cards − chips, lower better)."""
    return tuple(
        float(score_cards(info.cards[i]) - info.chips[i]) for i in range(info.n_players)
    )


# --------------------------------------------------------------------------- #
# Info-set transitions (the belief-marginalised dynamics)
# --------------------------------------------------------------------------- #

def apply_pass(info: InfoSet) -> InfoSet:
    """Pay one chip onto the active card and pass — a public, deterministic move."""
    p = info.to_move
    chips = list(info.chips)
    chips[p] -= 1
    return replace(
        info, chips=tuple(chips), pot=info.pot + 1, to_move=(p + 1) % info.n_players
    )


def take_outcomes(info: InfoSet) -> list[tuple[float, InfoSet]]:
    """All ``(probability, next_info)`` results of the mover taking the active card.

    The taker collects the card and pot, then the next card is flipped — a chance
    event that, marginalised over the unknown removal, is **uniform over the
    unseen cards** (``1/|unseen|`` each). If no pile cards remain
    (``pile_remaining == 0``) the game ends. The taker keeps the turn, mirroring
    :func:`nothanks.engine.take_outcomes`.
    """
    p = info.to_move
    chips = list(info.chips)
    chips[p] += info.pot
    cards = list(info.cards)
    cards[p] = cards[p] | {info.active}
    chips_t = tuple(chips)
    cards_t = tuple(cards)

    if pile_remaining(info) == 0:
        terminal = replace(info, chips=chips_t, cards=cards_t, active=None, pot=0)
        return [(1.0, terminal)]

    candidates = unseen(info)  # excludes the just-captured card (it is now "seen")
    prob = 1.0 / len(candidates)
    outcomes = []
    for c in candidates:
        nxt = replace(info, chips=chips_t, cards=cards_t, active=c, pot=0, to_move=p)
        outcomes.append((prob, nxt))
    return outcomes


# --------------------------------------------------------------------------- #
# Public-information policies on info sets
# --------------------------------------------------------------------------- #

def heuristic_info_action(info: InfoSet, threshold: int = 0) -> str:
    """The run-aware heuristic (see :mod:`nothanks.heuristic`) on an info set.

    Identical rule — take iff ``score_delta(card) - pot <= threshold``, forced at
    0 chips — but expressed over public knowledge, so it is a legal hidden-game
    strategy.
    """
    p = info.to_move
    if info.chips[p] == 0:
        return "take"
    cost = score_delta(info.cards[p], info.active) - info.pot
    return "take" if cost <= threshold else "pass"


class _HeuristicPolicy:
    """Callable (hence picklable) form of :func:`make_heuristic_policy`."""

    def __init__(self, threshold: int):
        self.threshold = threshold

    def __call__(self, info: InfoSet) -> str:
        return heuristic_info_action(info, self.threshold)


def make_heuristic_policy(threshold: int = 0) -> InfoPolicy:
    return _HeuristicPolicy(threshold)


# --------------------------------------------------------------------------- #
# Belief-exact policy evaluation and self-interested optimum
# --------------------------------------------------------------------------- #

def policy_value(
    info: InfoSet, policy: InfoPolicy, memo: dict | None = None
) -> tuple[float, ...]:
    """Exact expected final-score vector when every seat follows ``policy``.

    The belief-game analogue of :func:`nothanks.montecarlo.policy_value`: chance is
    the uniform-over-unseen draw, enumerated rather than sampled.
    """
    if memo is None:
        memo = {}
    if is_terminal(info):
        return final_scores(info)
    cached = memo.get(info)
    if cached is not None:
        return cached

    action = policy(info)
    if action == "pass":
        v = policy_value(apply_pass(info), policy, memo)
    else:
        n = info.n_players
        acc = [0.0] * n
        for prob, nxt in take_outcomes(info):
            sub = policy_value(nxt, policy, memo)
            for i in range(n):
                acc[i] += prob * sub[i]
        v = tuple(acc)
    memo[info] = v
    return v


def solve(info: InfoSet, memo: dict | None = None) -> tuple[tuple[float, ...], str | None]:
    """Self-interested optimal value vector and best action under hidden cards.

    Each mover minimises its *own* expected score assuming every future mover does
    the same — the belief-game counterpart of :func:`nothanks.solver.solve`, with
    the same equilibrium-selection caveat (the tie-break picks *a* subgame-perfect
    equilibrium; under indifference the value vector is not unique). Exact only on
    small games. Returns ``(value_vector, best_action)`` (``best_action`` is
    ``None`` at a terminal info set).
    """
    if memo is None:
        memo = {}
    if is_terminal(info):
        return final_scores(info), None
    cached = memo.get(info)
    if cached is not None:
        return cached

    p = info.to_move
    n = info.n_players
    best_v: tuple[float, ...] | None = None
    best_a: str | None = None
    for action in legal_actions(info):
        if action == "pass":
            v, _ = solve(apply_pass(info), memo)
        else:
            acc = [0.0] * n
            for prob, nxt in take_outcomes(info):
                sub, _ = solve(nxt, memo)
                for i in range(n):
                    acc[i] += prob * sub[i]
            v = tuple(acc)
        if best_v is None or v[p] < best_v[p]:
            best_v, best_a = v, action

    memo[info] = (best_v, best_a)
    return best_v, best_a


def optimal_policy(memo: dict | None = None) -> InfoPolicy:
    """A deterministic self-interested-optimal info-set policy (exploitability ~0).

    The hidden-game analogue of :func:`nothanks.exploit.optimal_policy`: a fixed
    point of best response, so :func:`exploitability` reports ~0 for it. Tractable
    only on small games.
    """
    if memo is None:
        memo = {}
    return lambda info: solve(info, memo)[1]


# --------------------------------------------------------------------------- #
# Belief-correct best response and exploitability
# --------------------------------------------------------------------------- #

def best_response_value(
    info: InfoSet, hero: int, policy: InfoPolicy, memo: dict | None = None
) -> tuple[float, ...]:
    """Expected scores when ``hero`` best-responds on info sets and others play ``policy``.

    The hero minimises its own expected score at every info set it is to move on —
    choosing a *single* action per info set, since it cannot see the removed cards
    — while the other seats follow ``policy``; chance is the belief draw. Because
    the belief is pinned by the info set (strategy-independent), this plain
    backward induction is the exact best response.

    The ``memo`` is valid for a single ``(hero, policy)`` pair — pass a fresh dict
    per hero (as :func:`exploitability` does).
    """
    if memo is None:
        memo = {}
    if is_terminal(info):
        return final_scores(info)
    cached = memo.get(info)
    if cached is not None:
        return cached

    n = info.n_players

    def value_of(action: str) -> tuple[float, ...]:
        if action == "pass":
            return best_response_value(apply_pass(info), hero, policy, memo)
        acc = [0.0] * n
        for prob, nxt in take_outcomes(info):
            sub = best_response_value(nxt, hero, policy, memo)
            for i in range(n):
                acc[i] += prob * sub[i]
        return tuple(acc)

    if info.to_move == hero:
        best: tuple[float, ...] | None = None
        for action in legal_actions(info):
            v = value_of(action)
            if best is None or v[hero] < best[hero]:
                best = v
        v = best
    else:
        v = value_of(policy(info))

    memo[info] = v
    return v


def exploitability(info: InfoSet, policy: InfoPolicy) -> dict:
    """Per-seat and aggregate belief-correct best-response gain against ``policy``.

    ``base`` is the value when every seat follows ``policy``; ``br[i]`` is seat
    ``i``'s value when it best-responds *on info sets* (others on ``policy``). The
    gain ``base[i] - br[i]`` is the seat's exploitability under genuinely hidden
    cards — how much a belief-aware deviation can lower its own score. ``total`` /
    ``max`` summarise across seats (lower is closer to an unexploitable fixed
    point). With ``n_removed == 0`` this equals :func:`nothanks.exploit.exploitability`.
    """
    n = info.n_players
    base = policy_value(info, policy)
    br = []
    gain = []
    for hero in range(n):
        v = best_response_value(info, hero, policy, {})
        br.append(v[hero])
        gain.append(base[hero] - v[hero])
    return {
        "base": tuple(base),
        "br": tuple(br),
        "gain": tuple(gain),
        "total": sum(gain),
        "max": max(gain),
    }
