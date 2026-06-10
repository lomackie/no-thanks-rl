"""Info-set-native value net: honest evals and self-play with no PIMC wrapper.

The god-view net (:mod:`nothanks.valuefn` / :mod:`nothanks.train`) encodes
``State.remaining``, so honesty has to be bolted on at play time by averaging it
over determinized worlds (``train.pimc_policy``). That costs ``n_worlds`` forward
passes per move and keeps a subtle bias: the net's values are policy-evaluations
of trajectories played by a policy that *saw* the hidden cards.

This module trains the net directly on the **belief game** instead — the Markov
game on info sets that :mod:`nothanks.belief` showed is exactly the hidden game
for public-information players. Features are public knowledge only
(:func:`nothanks.features.info_features`: the ``unseen`` multi-hot replaces the
pile multi-hot, plus the ``pile_remaining`` count), the simulator is the belief
dynamics (``pass`` deterministic, ``take`` draws uniform over unseen), and the
TD(λ) machinery is shared with :mod:`nothanks.train` unchanged. The result:

* ``greedy_info_action`` is already honest — one one-ply lookahead per move, two
  worlds behind the same info set get the same encoding and hence the same move;
* :func:`make_greedy_info_policy` is a deterministic ``InfoPolicy``, so
  :func:`nothanks.belief.exploitability` can grade it exactly on small games;
* the net is the natural fast leaf for IS-MCTS
  (:func:`nothanks.ismcts.make_value_leaf`).

Sanity anchor: at ``n_removed=0`` the belief game *is* the real game (unseen =
pile), so this training setup is the god-view setup with an extra constant
feature — strength should match the god-view net there.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np

from .belief import (
    InfoPolicy,
    apply_pass,
    final_scores,
    heuristic_info_action,
    is_terminal,
    take_outcomes,
)
from .engine import STARTING_CHIPS, full_deck, new_game
from .engine import final_scores as state_final_scores
from .engine import is_terminal as state_is_terminal
from .engine import step as state_step
from .features import info_feature_dim, info_features, seat_order
from .heuristic import heuristic_action
from .imperfect import InfoSet, info_from_state, legal_actions
from .train import _Step, _lambda_returns
from .valuefn import ValueNet


def make_info_net(n_players: int, hidden: int = 64, seed: int = 0) -> ValueNet:
    """A value net over public info-set features (same MLP, honest inputs)."""
    return ValueNet(n_players, hidden=hidden, seed=seed,
                    in_dim=info_feature_dim(n_players))


def predict_info(net: ValueNet, info: InfoSet) -> np.ndarray:
    """Mover-frame value vector of an info set (the info-net ``predict``)."""
    return net.forward(info_features(info)[None, :])[0]


# --------------------------------------------------------------------------- #
# Simulating the belief game
# --------------------------------------------------------------------------- #

def new_belief_game(
    n_players: int,
    deck=None,
    n_removed: int = 9,
    start_chips: int | None = None,
    rng: random.Random | None = None,
) -> InfoSet:
    """A fresh belief game: the opening card is uniform over the deck.

    That is the exact marginal of the real setup (remove ``n_removed`` uniformly,
    flip the top of the shuffled rest — by symmetry every card is equally likely
    to open). ``deck=None`` is the standard 3..35 universe.
    """
    rng = rng or random.Random()
    deck = frozenset(full_deck() if deck is None else deck)
    if start_chips is None:
        start_chips = STARTING_CHIPS[n_players]
    active = rng.choice(sorted(deck))
    return InfoSet(
        chips=tuple(start_chips for _ in range(n_players)),
        cards=tuple(frozenset() for _ in range(n_players)),
        active=active,
        pot=0,
        to_move=0,
        deck=deck,
        n_removed=n_removed,
    )


def belief_step(info: InfoSet, action: str, rng: random.Random) -> InfoSet:
    """Apply ``action`` on the belief game, sampling the uniform-over-unseen draw."""
    if action == "pass":
        return apply_pass(info)
    if action == "take":
        outs = take_outcomes(info)
        if len(outs) == 1:
            return outs[0][1]
        return rng.choice(outs)[1]  # non-terminal take outcomes are equiprobable
    raise ValueError(f"unknown action {action!r}")


# --------------------------------------------------------------------------- #
# One-ply lookahead on the info net (the honest engine eval)
# --------------------------------------------------------------------------- #

def info_action_values(info: InfoSet, net: ValueNet) -> dict[str, np.ndarray]:
    """Expected-score vector per legal action, in ``info``'s mover-relative frame.

    The info-set analogue of :func:`nothanks.valuefn.action_values`: a ``take``
    keeps the mover's perspective (chance enumerated exactly, successors batched
    into one forward pass); a ``pass`` rotates the successor's vector one seat.
    """
    p = info.to_move
    n = info.n_players
    out: dict[str, np.ndarray] = {}
    for action in legal_actions(info):
        if action == "pass":
            out[action] = np.roll(predict_info(net, apply_pass(info)), 1)
        else:
            acc = np.zeros(n)
            feats: list[np.ndarray] = []
            probs: list[float] = []
            for prob, nxt in take_outcomes(info):
                if is_terminal(nxt):
                    fs = final_scores(nxt)
                    acc += prob * np.array([fs[q] for q in seat_order(p, n)])
                else:
                    feats.append(info_features(nxt))
                    probs.append(prob)
            if feats:
                preds = net.forward(np.stack(feats))
                acc += (np.array(probs)[:, None] * preds).sum(axis=0)
            out[action] = acc
    return out


def evaluate_info(info: InfoSet, net: ValueNet) -> dict:
    """Engine-style move eval from public knowledge alone (mover is seat 0)."""
    av = info_action_values(info, net)
    best_action = min(av, key=lambda a: av[a][0])
    return {
        "to_move": info.to_move,
        "actions": {a: tuple(float(x) for x in v) for a, v in av.items()},
        "mover_ev": {a: float(v[0]) for a, v in av.items()},
        "best_action": best_action,
    }


def greedy_info_action(info: InfoSet, net: ValueNet) -> str:
    """Action minimising the mover's own predicted score — honest by construction."""
    av = info_action_values(info, net)
    return min(av, key=lambda a: av[a][0])


def make_greedy_info_policy(net: ValueNet) -> InfoPolicy:
    """A deterministic ``InfoPolicy`` for :func:`nothanks.belief.exploitability`."""
    return lambda info: greedy_info_action(info, net)


# --------------------------------------------------------------------------- #
# Self-play TD(λ) on the belief game
# --------------------------------------------------------------------------- #

# Same shape as train.Behavior, but over info sets.
InfoBehavior = Callable[[InfoSet, ValueNet, random.Random, float], str]


def greedy_info_behavior(info: InfoSet, net: ValueNet, rng: random.Random, eps: float) -> str:
    """ε-greedy on the info net's one-ply lookahead (pure self-play)."""
    actions = legal_actions(info)
    if len(actions) == 1:
        return actions[0]
    if rng.random() < eps:
        return rng.choice(actions)
    return greedy_info_action(info, net)


def heuristic_info_behavior(info: InfoSet, net: ValueNet, rng: random.Random, eps: float) -> str:
    """ε-greedy on the run-aware heuristic — the calibrating data policy."""
    actions = legal_actions(info)
    if len(actions) == 1:
        return actions[0]
    if rng.random() < eps:
        return rng.choice(actions)
    return heuristic_info_action(info, 0)


def selfplay_belief_game(
    net: ValueNet,
    rng: random.Random,
    eps: float,
    behavior: InfoBehavior,
    deck=None,
    n_removed: int = 9,
    start_chips: int | None = None,
) -> list[_Step]:
    """Play one belief game under ``behavior``; return its decision steps.

    Reuses :class:`nothanks.train._Step` so the λ-return machinery is shared —
    the frame logic is identical, only the simulator and features differ.
    """
    info = new_belief_game(net.n_players, deck=deck, n_removed=n_removed,
                           start_chips=start_chips, rng=rng)
    steps: list[_Step] = []
    while not is_terminal(info):
        feat = info_features(info)
        mover = info.to_move
        a = behavior(info, net, rng, eps)
        nxt = belief_step(info, a, rng)
        final_abs = np.array(final_scores(nxt), dtype=float) if is_terminal(nxt) else None
        steps.append(_Step(feat, mover, final_abs))
        info = nxt
    return steps


def train_info(
    n_players: int = 3,
    iterations: int = 60,
    games_per_iter: int = 80,
    epochs_per_iter: int = 6,
    batch_size: int = 256,
    lr: float = 0.01,
    lam: float = 0.9,
    eps_start: float = 0.3,
    eps_end: float = 0.05,
    heur_frac_start: float = 1.0,
    heur_frac_end: float = 1.0,
    target_refresh: int = 1,
    hidden: int = 64,
    deck=None,
    n_removed: int = 9,
    start_chips: int | None = None,
    seed: int = 0,
    log: bool = False,
) -> ValueNet:
    """Train an info-set value net by self-play TD(λ) on the belief game.

    Mirrors :func:`nothanks.train.train` — same curriculum
    (``heur_frac_start → heur_frac_end``), target network, and λ-return targets —
    but data generation, features, and bootstraps all live on info sets, so the
    net never sees a removed card during training either. ``deck`` (an iterable of
    card values, default the full 3..35) allows tiny-game configurations for exact
    grading by :func:`nothanks.belief.exploitability`.
    """
    net = make_info_net(n_players, hidden=hidden, seed=seed)
    target = net.copy()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    for it in range(iterations):
        frac = it / max(iterations - 1, 1)
        eps = eps_start + (eps_end - eps_start) * frac
        heur_frac = heur_frac_start + (heur_frac_end - heur_frac_start) * frac

        if it % target_refresh == 0:
            target = net.copy()

        games = []
        for _ in range(games_per_iter):
            behavior = heuristic_info_behavior if rng.random() < heur_frac else greedy_info_behavior
            games.append(
                selfplay_belief_game(net, rng, eps, behavior, deck=deck,
                                     n_removed=n_removed, start_chips=start_chips)
            )
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
            print(f"iter {it:3d}  eps {eps:.2f}  heur {heur_frac:.2f}  "
                  f"steps {len(X):5d}  loss {last_loss:.3f}")

    return net


# --------------------------------------------------------------------------- #
# Grading on real games
# --------------------------------------------------------------------------- #

def head_to_head_info(
    net: ValueNet,
    n_games: int = 2000,
    seed: int = 20_000,
    n_removed: int = 9,
) -> dict:
    """Real games: greedy(info net) in seat 0 vs the heuristic elsewhere.

    The no-peek counterpart of :func:`nothanks.train.head_to_head` — and, unlike
    :func:`nothanks.train.head_to_head_hidden`, it needs no PIMC averaging (one
    one-ply lookahead per move), so it runs at god-view speed while staying
    honest. Lower mean is better; read the gap against ``vnet_stderr``.
    """
    n = net.n_players
    heur = lambda s: heuristic_action(s, 0)  # noqa: E731
    v_scores: list[float] = []
    h_total = 0.0
    v_wins = 0
    for i in range(n_games):
        rng = random.Random(seed + i)
        s = new_game(n, n_removed=n_removed, rng=rng)
        while not state_is_terminal(s):
            if s.to_move == 0:
                a = greedy_info_action(info_from_state(s, n_removed=n_removed), net)
            else:
                a = heur(s)
            s = state_step(s, a, rng)
        sc = state_final_scores(s)
        v_scores.append(sc[0])
        h_total += sum(sc[1:]) / (n - 1)
        if sc[0] <= min(sc[1:]):
            v_wins += 1
    v_mean = sum(v_scores) / n_games
    v_var = sum((x - v_mean) ** 2 for x in v_scores) / n_games
    return {
        "vnet_mean": v_mean,
        "vnet_stderr": math.sqrt(v_var / n_games),
        "heuristic_mean": h_total / n_games,
        "win_rate": v_wins / n_games,
        "parity": 1.0 / n,
    }
