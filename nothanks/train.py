"""Self-play TD(λ) training for the value net (TD-Gammon style).

Every seat picks moves by one-ply lookahead on the *current* net (ε-greedy for
exploration); the net is then regressed toward the **λ-return** of each visited
state. The λ-return blends every n-step bootstrap from λ=0 (one-ply TD, low
variance but slow to propagate the terminal signal across a ~24-ply game) up to
λ=1 (the realised final score, unbiased but very high variance). The middle
ground is what makes the opening actually reflect the end-game outcome without
the wild variance of a single game's score.

Frames. Final scores are naturally in **absolute** seat order, so the λ-return
recursion is done there — no per-step rotation. Only when storing a training
target is the absolute return rotated into that state's mover-relative frame (the
frame the net predicts in; see :mod:`nothanks.valuefn`). A state's net value is
converted to absolute by the inverse rotation ``np.roll(predict, mover)``.

Targets are recomputed once per iteration from a snapshot of the net and then
fitted for a few epochs (fitted-value-iteration style) for stability.

Run a quick training + sanity demo with ``just train``.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from .engine import State, final_scores, is_terminal, legal_actions, new_game, step
from .features import features, seat_order
from .heuristic import heuristic_action
from .valuefn import ValueNet, greedy_action

# A behaviour policy decides the move actually played during self-play. It sees
# the net so it can be greedy-on-net, heuristic, or any mix.
Behavior = Callable[[State, ValueNet, random.Random, float], str]


def greedy_behavior(s: State, net: ValueNet, rng: random.Random, eps: float) -> str:
    """ε-greedy on the net's own one-ply lookahead (pure self-play)."""
    actions = legal_actions(s)
    if len(actions) == 1:
        return actions[0]
    if rng.random() < eps:
        return rng.choice(actions)
    return greedy_action(s, net)


def heuristic_behavior(s: State, net: ValueNet, rng: random.Random, eps: float) -> str:
    """ε-greedy on the heuristic — competent data for policy evaluation.

    Self-play from a blank net collapses (greedy moves are dominated by value
    noise, so it only ever sees bad trajectories). Generating data with the
    heuristic instead yields a net that is *calibrated* to real play, and whose
    one-ply-greedy policy is a policy-improvement step over the heuristic.
    """
    actions = legal_actions(s)
    if len(actions) == 1:
        return actions[0]
    if rng.random() < eps:
        return rng.choice(actions)
    return heuristic_action(s, 0)


BEHAVIORS: dict[str, Behavior] = {
    "greedy": greedy_behavior,
    "heuristic": heuristic_behavior,
}


class _Step:
    """One decision in an episode: features, the mover, and (if last) the result."""

    __slots__ = ("feat", "mover", "final_abs")

    def __init__(self, feat, mover, final_abs):
        self.feat = feat
        self.mover = mover
        self.final_abs = final_abs  # absolute final scores if this step ends the game


def selfplay_game(
    net: ValueNet,
    rng: random.Random,
    eps: float,
    behavior: Behavior,
    n_removed: int = 9,
    start_chips: int | None = None,
) -> list[_Step]:
    """Play one self-play game under ``behavior``; return its decision steps."""
    n = net.n_players
    s = new_game(n, n_removed=n_removed, start_chips=start_chips, rng=rng)
    steps: list[_Step] = []
    while not is_terminal(s):
        feat = features(s)
        mover = s.to_move
        a = behavior(s, net, rng, eps)
        nxt = step(s, a, rng)
        final_abs = np.array(final_scores(nxt), dtype=float) if is_terminal(nxt) else None
        steps.append(_Step(feat, mover, final_abs))
        s = nxt
    return steps


def _lambda_returns(net: ValueNet, steps: list[_Step], lam: float) -> tuple:
    """λ-return targets for one episode: returns (X, T) in mover-relative frames."""
    n = net.n_players
    feats = np.stack([st.feat for st in steps])
    preds = net.forward(feats)  # mover-frame value of each visited state
    # Absolute-frame value U(s) = inverse rotation of the mover-frame prediction.
    U_abs = np.stack([np.roll(preds[i], steps[i].mover) for i in range(len(steps))])

    T = np.zeros((len(steps), n))
    G = None  # λ-return (absolute) of the *next* state, built back-to-front
    for t in range(len(steps) - 1, -1, -1):
        st = steps[t]
        if st.final_abs is not None:        # next state is terminal
            G = st.final_abs
        else:                               # blend one-ply bootstrap with the tail
            G = (1.0 - lam) * U_abs[t + 1] + lam * G
        T[t] = [G[(st.mover + k) % n] for k in range(n)]  # back to mover frame
    return feats, T


def train(
    n_players: int = 3,
    iterations: int = 60,
    games_per_iter: int = 80,
    epochs_per_iter: int = 6,
    batch_size: int = 256,
    lr: float = 0.01,
    lam: float = 0.9,
    eps_start: float = 0.3,
    eps_end: float = 0.05,
    behavior: str = "heuristic",
    hidden: int = 64,
    n_removed: int = 9,
    start_chips: int | None = None,
    seed: int = 0,
    log: bool = False,
) -> ValueNet:
    """Train a value net by self-play TD(λ).

    ``behavior`` selects the data-generating policy: ``"heuristic"`` (default —
    competent play, yields a calibrated evaluator and an improvement step) or
    ``"greedy"`` (pure self-play on the net, which can collapse from scratch).
    """
    net = ValueNet(n_players, hidden=hidden, seed=seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    behavior_fn = BEHAVIORS[behavior]

    for it in range(iterations):
        frac = it / max(iterations - 1, 1)
        eps = eps_start + (eps_end - eps_start) * frac

        games = [
            selfplay_game(net, rng, eps, behavior_fn,
                          n_removed=n_removed, start_chips=start_chips)
            for _ in range(games_per_iter)
        ]
        # Build λ-return targets from a single net snapshot, then fit them.
        XT = [_lambda_returns(net, g, lam) for g in games]
        X = np.concatenate([x for x, _ in XT])
        T = np.concatenate([t for _, t in XT])

        last_loss = 0.0
        for _ in range(epochs_per_iter):
            perm = np_rng.permutation(len(X))
            for i in range(0, len(X), batch_size):
                idx = perm[i : i + batch_size]
                last_loss = net.train_step(X[idx], T[idx], lr)

        if log:
            print(f"iter {it:3d}  eps {eps:.2f}  steps {len(X):5d}  loss {last_loss:.3f}")

    return net


def head_to_head(
    net: ValueNet,
    n_games: int = 2000,
    seed: int = 20_000,
    n_removed: int = 9,
) -> dict:
    """Pit greedy(net) in seat 0 against the heuristic in the other seats.

    Lower mean score is better. ``win_rate`` is how often seat 0 ties-or-beats
    every opponent; for ``n`` equally-skilled players that is about ``1/n``.
    """
    n = net.n_players
    heur = lambda s: heuristic_action(s, 0)  # noqa: E731
    vnet = lambda s: greedy_action(s, net)   # noqa: E731
    v_total = h_total = 0.0
    v_wins = 0
    for i in range(n_games):
        rng = random.Random(seed + i)
        s = new_game(n, n_removed=n_removed, rng=rng)
        while not is_terminal(s):
            a = vnet(s) if s.to_move == 0 else heur(s)
            s = step(s, a, rng)
        sc = final_scores(s)
        v_total += sc[0]
        h_total += sum(sc[1:]) / (n - 1)
        if sc[0] <= min(sc[1:]):
            v_wins += 1
    return {
        "vnet_mean": v_total / n_games,
        "heuristic_mean": h_total / n_games,
        "win_rate": v_wins / n_games,
        "parity": 1.0 / n,
    }


def _demo() -> None:
    """Train a 3-player net, show an engine eval, then validate policy strength."""
    from .valuefn import evaluate_v

    print("training a 3-player value net by self-play TD(λ) on heuristic data...")
    net = train(n_players=3, log=True)

    # Engine-style eval of a full opening: instant, one forward pass per successor.
    s = new_game(3, n_removed=9, rng=random.Random(1))
    ev = evaluate_v(s, net)
    print(f"\nengine eval — card {s.active} face-up (pot {s.pot}), P{ev['to_move']} to move:")
    for a, v in ev["mover_ev"].items():
        print(f"  {a:5s} -> EV {v:+7.2f}   vec {tuple(round(x, 1) for x in ev['actions'][a])}")
    print(f"  best: {ev['best_action']}")

    # Validation that actually means something: does greedy(net) beat the baseline?
    print("\nvalidation — greedy(net) seat 0 vs heuristic seats (lower is better):")
    h2h = head_to_head(net)
    print(f"  value-net mean score {h2h['vnet_mean']:.2f}  vs  heuristic {h2h['heuristic_mean']:.2f}")
    print(f"  win/tie rate {h2h['win_rate']:.1%}  (parity for 3 players ≈ {h2h['parity']:.0%})")


if __name__ == "__main__":
    _demo()
