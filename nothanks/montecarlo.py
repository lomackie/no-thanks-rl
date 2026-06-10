"""Monte-Carlo EV evaluator — engine-style move analysis by rollout.

The exact :mod:`~nothanks.solver` is only tractable on toy decks. For the real
24-card removal deck we instead estimate move values by *rollout*: fix the
mover's first action, then let a cheap fixed policy (the run-aware heuristic)
play every seat to the end, and average the final scores over many sampled draw
orders. This is the classic TD-Gammon-style one-ply lookahead, but with a
rollout standing in for a learned value function.

What value is being estimated?
------------------------------
Unlike the solver — which reports EV under *self-interested optimal* play — a
rollout reports EV under the **rollout policy**. The two are different numbers,
so the solver is *not* the ground truth for this estimator. The correct ground
truth is the *exact expectation of the same policy* (:func:`policy_value`,
:func:`exact_action_values`), obtained by enumerating chance instead of sampling
it. On tiny games the sampler must converge to that exact value; that is the
validation in the tests.

The only randomness here is the draw order (``remaining`` is known state). The
heuristic is deterministic, so a seeded ``rng`` makes every rollout reproducible.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from .engine import (
    State,
    apply_pass,
    final_scores,
    is_terminal,
    legal_actions,
    step,
    take_outcomes,
)
from .heuristic import heuristic_action

# A policy maps a non-terminal state to a legal action.
Policy = Callable[[State], str]


def make_policy(threshold: int = 0) -> Policy:
    """A deterministic rollout policy from the run-aware heuristic."""
    return lambda s: heuristic_action(s, threshold)


default_policy: Policy = make_policy(0)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def rollout(s: State, policy: Policy, rng: random.Random) -> tuple[int, ...]:
    """Play ``s`` to a terminal state under ``policy`` and return final scores."""
    while not is_terminal(s):
        s = step(s, policy(s), rng)
    return final_scores(s)


def evaluate_mc(
    s: State,
    n_rollouts: int = 2000,
    policy: Policy | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Estimate per-move EV for the mover by rollout, like an engine eval.

    For each legal action we apply it once (sampling the draw for ``take``), roll
    the rest of the game out under ``policy`` for every seat, and average the
    final-score vectors over ``n_rollouts`` samples. ``best_action`` minimises the
    mover's own expected score. ``stderr`` is the standard error of that own-score
    estimate, so a gap between two actions is only meaningful when it comfortably
    exceeds their combined ``stderr``.
    """
    policy = policy or default_policy
    rng = rng or random.Random()
    n = s.n_players
    p = s.to_move

    actions: dict[str, tuple[float, ...]] = {}
    stderr: dict[str, float] = {}
    for action in legal_actions(s):
        acc = [0.0] * n
        own_sum = 0.0
        own_sumsq = 0.0
        for _ in range(n_rollouts):
            nxt = step(s, action, rng)
            scores = rollout(nxt, policy, rng)
            for i in range(n):
                acc[i] += scores[i]
            own_sum += scores[p]
            own_sumsq += scores[p] * scores[p]
        actions[action] = tuple(x / n_rollouts for x in acc)
        mean = own_sum / n_rollouts
        var = max(own_sumsq / n_rollouts - mean * mean, 0.0)
        stderr[action] = math.sqrt(var / n_rollouts)

    best_action = min(actions, key=lambda a: actions[a][p])
    return {
        "to_move": p,
        "actions": actions,
        "mover_ev": {a: v[p] for a, v in actions.items()},
        "stderr": stderr,
        "best_action": best_action,
        "n_rollouts": n_rollouts,
    }


# --------------------------------------------------------------------------- #
# Exact policy evaluation — the ground truth the sampler converges to
# --------------------------------------------------------------------------- #

def policy_value(
    s: State,
    policy: Policy | None = None,
    memo: dict | None = None,
) -> tuple[float, ...]:
    """Exact expected final-score vector when every seat follows ``policy``.

    Same shape as :func:`nothanks.solver.solve`, but it *follows* the fixed
    policy at every decision instead of minimising. Chance (the draw) is
    enumerated, not sampled, so the result is exact — making this the convergence
    target for :func:`evaluate_mc` on small games.
    """
    policy = policy or default_policy
    if memo is None:
        memo = {}
    if is_terminal(s):
        return tuple(float(x) for x in final_scores(s))
    cached = memo.get(s)
    if cached is not None:
        return cached

    action = policy(s)
    if action == "pass":
        v = policy_value(apply_pass(s), policy, memo)
    else:
        n = s.n_players
        acc = [0.0] * n
        for prob, nxt in take_outcomes(s):
            sub = policy_value(nxt, policy, memo)
            for i in range(n):
                acc[i] += prob * sub[i]
        v = tuple(acc)
    memo[s] = v
    return v


def exact_action_values(
    s: State,
    policy: Policy | None = None,
    memo: dict | None = None,
) -> dict:
    """Exact per-move EV vectors: take the action, then everyone follows ``policy``.

    The exact analogue of :func:`evaluate_mc` — its ``actions`` vectors are what
    that estimator's sampled vectors converge to as ``n_rollouts`` grows.
    """
    policy = policy or default_policy
    if memo is None:
        memo = {}
    n = s.n_players
    p = s.to_move

    actions: dict[str, tuple[float, ...]] = {}
    for action in legal_actions(s):
        if action == "pass":
            actions[action] = policy_value(apply_pass(s), policy, memo)
        else:
            acc = [0.0] * n
            for prob, nxt in take_outcomes(s):
                sub = policy_value(nxt, policy, memo)
                for i in range(n):
                    acc[i] += prob * sub[i]
            actions[action] = tuple(acc)

    best_action = min(actions, key=lambda a: actions[a][p])
    return {
        "to_move": p,
        "actions": actions,
        "mover_ev": {a: v[p] for a, v in actions.items()},
        "best_action": best_action,
    }
