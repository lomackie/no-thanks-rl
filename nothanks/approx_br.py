"""Approximate best response — exploitability you can measure on the full game.

:func:`nothanks.belief.exploitability` is the honest robustness metric, but its
exact backward induction dies beyond toy decks, so on the real 9-removed game a
policy's quality was previously unfalsifiable ("beats the heuristic" is relative
to an opponent we chose). This module closes that gap by *training* the best
responder: freeze the candidate :data:`~nothanks.belief.InfoPolicy` in every seat
but one, and learn the hero seat's reply.

Why this is plain RL: the belief reduction (:mod:`nothanks.belief`) makes the
hidden game an ordinary Markov game on info sets, and with all other seats frozen
the hero faces a plain MDP — no reach weighting, no belief machinery. So the
trainer is the same TD(λ) + one-ply-greedy-improvement loop as
:mod:`nothanks.beliefnet`, with the opponents' moves supplied by the candidate.

Reading the number: the learned responder's unilateral gain is a **lower bound**
on the true exploitability — any gain a learner finds, an exact best response
also finds; what the learner misses is simply unmeasured. A large gain falsifies
a policy's claim to optimality; a small one is necessary-but-not-sufficient
evidence. On tiny decks the bound is validated against the exact
:func:`nothanks.belief.best_response_value` (and is also *upper*-bounded by it,
which the tests pin from both sides).
"""

from __future__ import annotations

import math
import random

import numpy as np

from .belief import InfoPolicy, final_scores, is_terminal
from .beliefnet import (
    belief_step,
    greedy_info_action,
    make_info_net,
    new_belief_game,
)
from .features import info_features
from .imperfect import InfoSet, legal_actions
from .train import _Step, _lambda_returns
from .valuefn import ValueNet


def train_best_response(
    policy: InfoPolicy,
    hero: int,
    n_players: int = 3,
    iterations: int = 40,
    games_per_iter: int = 80,
    epochs_per_iter: int = 4,
    batch_size: int = 256,
    lr: float = 0.01,
    lam: float = 0.9,
    eps_start: float = 0.3,
    eps_end: float = 0.05,
    target_refresh: int = 5,
    hidden: int = 64,
    deck=None,
    n_removed: int = 9,
    start_chips: int | None = None,
    seed: int = 0,
    log: bool = False,
) -> ValueNet:
    """Learn seat ``hero``'s reply to ``policy`` (all other seats frozen on it).

    Episodes are belief games where the hero plays ε-greedy on its own info net
    and everyone else follows ``policy``; every visited info set gets a TD(λ)
    target, so the net learns the value of the *joint* play (hero's reply +
    frozen opponents) — exactly the function whose greedy improvement is the
    hero's approximate best response (:func:`br_policy`).
    """
    net = make_info_net(n_players, hidden=hidden, seed=seed)
    target = net.copy()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    for it in range(iterations):
        frac = it / max(iterations - 1, 1)
        eps = eps_start + (eps_end - eps_start) * frac
        if it % target_refresh == 0:
            target = net.copy()

        games = []
        for _ in range(games_per_iter):
            info = new_belief_game(n_players, deck=deck, n_removed=n_removed,
                                   start_chips=start_chips, rng=rng)
            steps: list[_Step] = []
            while not is_terminal(info):
                feat = info_features(info)
                mover = info.to_move
                if mover == hero:
                    acts = legal_actions(info)
                    if len(acts) == 1:
                        a = acts[0]
                    elif rng.random() < eps:
                        a = rng.choice(acts)
                    else:
                        a = greedy_info_action(info, net)
                else:
                    a = policy(info)
                nxt = belief_step(info, a, rng)
                final_abs = (np.array(final_scores(nxt), dtype=float)
                             if is_terminal(nxt) else None)
                steps.append(_Step(feat, mover, final_abs))
                info = nxt
            games.append(steps)

        XT = [_lambda_returns(net, g, lam, target=target) for g in games]
        X = np.concatenate([x for x, _ in XT])
        T = np.concatenate([t for _, t in XT])

        last_loss = 0.0
        for _ in range(epochs_per_iter):
            perm = np_rng.permutation(len(X))
            for i in range(0, len(X), batch_size):
                idx = perm[i : i + batch_size]
                last_loss = net.train_step(X[idx], T[idx], lr)
        if log:
            print(f"hero {hero}  iter {it:3d}  eps {eps:.2f}  "
                  f"steps {len(X):5d}  loss {last_loss:.3f}")

    return net


def br_policy(policy: InfoPolicy, net: ValueNet, hero: int) -> InfoPolicy:
    """The deviating joint policy: hero greedy on its BR net, others on ``policy``.

    Deterministic, so on tiny games :func:`nothanks.belief.policy_value` evaluates
    it *exactly* — that is how the learned gain is graded against the exact best
    response in the tests.
    """
    def combined(info: InfoSet) -> str:
        if info.to_move == hero:
            return greedy_info_action(info, net)
        return policy(info)

    return combined


def estimate_gain_mc(
    policy: InfoPolicy,
    net: ValueNet,
    hero: int,
    n_players: int = 3,
    n_games: int = 1000,
    deck=None,
    n_removed: int = 9,
    start_chips: int | None = None,
    seed: int = 50_000,
) -> dict:
    """Monte-Carlo estimate of the hero's unilateral gain on the full game.

    Plays ``n_games`` belief games twice — everyone on ``policy`` (``base``)
    versus hero deviating to its BR net (``br``) — with shared per-game seeds (the
    deals pair up; play diverges where the hero's move differs). ``gain`` is
    ``base − br`` for the hero seat: positive means the candidate is measurably
    exploitable; read it against ``stderr`` (of the paired per-game differences).
    """
    deviate = br_policy(policy, net, hero)
    diffs: list[float] = []
    base_total = br_total = 0.0
    for i in range(n_games):
        scores = []
        for pol in (policy, deviate):
            rng = random.Random(seed + i)
            info = new_belief_game(n_players, deck=deck, n_removed=n_removed,
                                   start_chips=start_chips, rng=rng)
            while not is_terminal(info):
                info = belief_step(info, pol(info), rng)
            scores.append(final_scores(info)[hero])
        base_total += scores[0]
        br_total += scores[1]
        diffs.append(scores[0] - scores[1])
    gain = sum(diffs) / n_games
    var = sum((d - gain) ** 2 for d in diffs) / n_games
    return {
        "hero": hero,
        "base": base_total / n_games,
        "br": br_total / n_games,
        "gain": gain,
        "stderr": math.sqrt(var / n_games),
        "n_games": n_games,
    }


def approx_exploitability(
    policy: InfoPolicy,
    n_players: int = 3,
    deck=None,
    n_removed: int = 9,
    start_chips: int | None = None,
    n_games: int = 1000,
    seed: int = 0,
    log: bool = False,
    **train_kwargs,
) -> dict:
    """Per-seat learned-best-response gains — the scaled `belief.exploitability`.

    Trains one BR net per hero seat (``train_kwargs`` forward to
    :func:`train_best_response`) and estimates each gain by paired Monte Carlo.
    Same summary shape as the exact metric (``gain`` per seat, ``total``/``max``)
    plus per-seat ``stderr``; remember every entry is a *lower bound* on the true
    exploitability, not an estimate of it.
    """
    gains: list[float] = []
    stderrs: list[float] = []
    for hero in range(n_players):
        net = train_best_response(policy, hero, n_players=n_players, deck=deck,
                                  n_removed=n_removed, start_chips=start_chips,
                                  seed=seed + hero, log=log, **train_kwargs)
        res = estimate_gain_mc(policy, net, hero, n_players=n_players, deck=deck,
                               n_removed=n_removed, start_chips=start_chips,
                               n_games=n_games, seed=seed + 50_000)
        gains.append(res["gain"])
        stderrs.append(res["stderr"])
        if log:
            print(f"hero {hero}: gain {res['gain']:+.3f} ± {res['stderr']:.3f}")
    return {
        "gain": tuple(gains),
        "stderr": tuple(stderrs),
        "total": sum(gains),
        "max": max(gains),
    }
