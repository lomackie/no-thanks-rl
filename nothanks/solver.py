"""Exact backward-induction solver — the ground-truth oracle for tiny games.

This solves the perfect-information stochastic game exactly via memoised
recursion. The value of a state is the *vector* of expected final scores, one
per player, under the assumption that every player minimises their **own**
expected score (a subgame-perfect solution). Because only one player moves at a
time and chance events are explicit, this is well defined for any number of
players — there is no 2-player-zero-sum requirement.

Equilibrium selection caveat: with 3+ self-interested players, backward induction
yields *a* subgame-perfect equilibrium under this implementation's tie-break
(legal-action order — ``take`` before ``pass``). When a mover is indifferent,
different tie-breaks can change the *other* players' values, so the equilibrium
value vector is not unique; "optimal play" here always means this selection.

Tractability: the state space explodes combinatorially, so this is only for
*small* configurations (a reduced deck and few chips). Use it as an oracle to
validate Monte-Carlo and learned evaluators, not to solve the full game.
"""

from __future__ import annotations

from .engine import (
    State,
    apply_pass,
    final_scores,
    is_terminal,
    legal_actions,
    take_outcomes,
)


def _expected(outcomes, memo, n) -> tuple[float, ...]:
    acc = [0.0] * n
    for prob, nxt in outcomes:
        v = solve(nxt, memo)
        for i in range(n):
            acc[i] += prob * v[i]
    return tuple(acc)


def solve(s: State, memo: dict | None = None) -> tuple[float, ...]:
    """Expected final-score vector under self-interested optimal play."""
    if memo is None:
        memo = {}
    if is_terminal(s):
        return tuple(float(x) for x in final_scores(s))
    cached = memo.get(s)
    if cached is not None:
        return cached

    n = s.n_players
    p = s.to_move
    best: tuple[float, ...] | None = None
    for action in legal_actions(s):
        if action == "pass":
            v = solve(apply_pass(s), memo)
        else:
            v = _expected(take_outcomes(s), memo, n)
        if best is None or v[p] < best[p]:
            best = v
    memo[s] = best
    return best


def evaluate(s: State, memo: dict | None = None) -> dict:
    """Per-move analysis for the mover, like an engine eval.

    Returns a dict with each legal action's expected score *vector*, the mover's
    own expected score for that action, and the recommended action (the one
    minimising the mover's own expected score).
    """
    if memo is None:
        memo = {}
    p = s.to_move
    actions: dict[str, tuple[float, ...]] = {}
    for action in legal_actions(s):
        if action == "pass":
            actions[action] = solve(apply_pass(s), memo)
        else:
            actions[action] = _expected(take_outcomes(s), memo, s.n_players)
    best_action = min(actions, key=lambda a: actions[a][p])
    return {
        "to_move": p,
        "actions": {a: v for a, v in actions.items()},
        "mover_ev": {a: v[p] for a, v in actions.items()},
        "best_action": best_action,
    }
