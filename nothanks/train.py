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
from .imperfect import determinized_action, info_from_state
from .valuefn import ValueNet, evaluate_v, greedy_action

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


def _lambda_returns(
    net: ValueNet, steps: list[_Step], lam: float, target: ValueNet | None = None
) -> tuple:
    """λ-return targets for one episode: returns (X, T) in mover-relative frames.

    Bootstraps use ``target`` (a frozen *target network*) when given, otherwise the
    live ``net`` — holding the bootstrap value fixed across several iterations is
    what stabilises fitted-value iteration.
    """
    target = target if target is not None else net
    n = net.n_players
    feats = np.stack([st.feat for st in steps])
    preds = target.forward(feats)  # mover-frame value of each visited state
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
    heur_frac_start: float = 1.0,
    heur_frac_end: float = 1.0,
    target_refresh: int = 1,
    hidden: int = 64,
    n_removed: int = 9,
    start_chips: int | None = None,
    seed: int = 0,
    log: bool = False,
) -> ValueNet:
    """Train a value net by self-play TD(λ).

    Data-generating policy — the curriculum. Each *game* is generated either by the
    heuristic (with probability ``heur_frac``) or by ε-greedy self-play on the
    current net, where ``heur_frac`` is annealed linearly from ``heur_frac_start``
    to ``heur_frac_end`` over training. The defaults (``1.0 → 1.0``) reproduce the
    pure-heuristic baseline: competent data that calibrates the net and makes
    greedy(net) a one-step policy improvement. Lowering ``heur_frac_end`` (e.g. to
    ``0.25``) shifts toward **pure self-play** in the second half — approximate
    policy iteration that can push *past* the heuristic ceiling — while the
    remaining heuristic-anchored games keep it from collapsing into value-noise
    trajectories (the failure mode of greedy-from-blank).

    Stability. Targets are bootstrapped from a frozen **target network** refreshed
    every ``target_refresh`` iterations (``1`` = the old snapshot-per-iteration
    behaviour), then fitted for ``epochs_per_iter`` epochs.
    """
    net = ValueNet(n_players, hidden=hidden, seed=seed)
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
            behavior_fn = heuristic_behavior if rng.random() < heur_frac else greedy_behavior
            games.append(
                selfplay_game(net, rng, eps, behavior_fn,
                              n_removed=n_removed, start_chips=start_chips)
            )
        # Build λ-return targets against the frozen target net, then fit them.
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


def net_vs_net(
    net_a: ValueNet,
    net_b: ValueNet,
    n_games: int = 1500,
    seed: int = 30_000,
    n_removed: int = 9,
) -> dict:
    """Seat-balanced head-to-head: greedy(``net_a``) vs greedy(``net_b``).

    Seat 0 has a large structural advantage in this game (it acts first and can
    sweep), so a fair comparison rotates ``net_a`` through *every* seat — replaying
    the same ``n_games`` games for each seat assignment (a paired design, lower
    variance) — and averages. Lower ``a_mean`` than ``b_mean`` means ``net_a`` is
    the stronger policy; ``a_win_rate`` is its tie-or-beat rate against
    ``parity = 1/n``. This is how we tell whether self-play overtook the
    heuristic-only baseline net.
    """
    n = net_a.n_players
    a = lambda s: greedy_action(s, net_a)  # noqa: E731
    b = lambda s: greedy_action(s, net_b)  # noqa: E731
    a_total = b_total = 0.0
    a_wins = 0
    games = 0
    for seat in range(n):
        for i in range(n_games):
            rng = random.Random(seed + i)  # same games across seats (paired)
            s = new_game(n, n_removed=n_removed, rng=rng)
            while not is_terminal(s):
                s = step(s, a(s) if s.to_move == seat else b(s), rng)
            sc = final_scores(s)
            a_total += sc[seat]
            b_total += (sum(sc) - sc[seat]) / (n - 1)
            if sc[seat] <= min(sc):
                a_wins += 1
            games += 1
    return {
        "a_mean": a_total / games,
        "b_mean": b_total / games,
        "a_win_rate": a_wins / games,
        "parity": 1.0 / n,
    }


def pimc_policy(net: ValueNet, n_removed: int = 9, n_worlds: int = 100):
    """The deployable, **non-cheating** bot: greedy(net) wrapped in PIMC.

    The plain :func:`nothanks.valuefn.greedy_action` reads the god-view ``State``
    (and so the true pile, which reveals the removed cards). This wrapper instead
    projects the referee's state onto the mover's :class:`~nothanks.imperfect.InfoSet`
    — dropping the pile's identity — and chooses by determinized lookahead over
    worlds consistent with public knowledge only. The returned ``policy(s, rng)``
    therefore never depends on which cards were removed.

    With ``n_removed == 0`` nothing is hidden and it reduces to plain greedy(net).
    """
    evaluator = lambda st: evaluate_v(st, net)  # noqa: E731

    def policy(s: State, rng: random.Random) -> str:
        info = info_from_state(s, n_removed=n_removed)
        return determinized_action(info, evaluator, n_worlds=n_worlds, rng=rng)

    return policy


def head_to_head_hidden(
    net: ValueNet,
    n_games: int = 300,
    seed: int = 20_000,
    n_removed: int = 9,
    n_worlds: int = 80,
) -> dict:
    """Like :func:`head_to_head`, but the net plays from public info only (PIMC).

    Seat 0 is the honest PIMC bot; the other seats follow the heuristic (which is
    already a function of public state). Because the bot never sees the removed
    cards, this is the fair measure of its *real* playing strength. Slower than
    :func:`head_to_head`: every bot move averages over ``n_worlds`` determinized
    worlds, so keep ``n_games``/``n_worlds`` modest.
    """
    n = net.n_players
    bot = pimc_policy(net, n_removed=n_removed, n_worlds=n_worlds)
    heur = lambda s: heuristic_action(s, 0)  # noqa: E731
    v_total = h_total = 0.0
    v_wins = 0
    for i in range(n_games):
        rng = random.Random(seed + i)
        s = new_game(n, n_removed=n_removed, rng=rng)
        while not is_terminal(s):
            a = bot(s, rng) if s.to_move == 0 else heur(s)
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
    """Train a heuristic-only baseline and a self-play net; show self-play wins."""
    from .valuefn import evaluate_v

    print("baseline — value net on pure heuristic data (TD-λ):")
    base = train(n_players=3, log=True)

    print("\nself-play — warmup on heuristic, then anneal to greedy self-play"
          " with a target net (slower: greedy data costs a forward pass per move):")
    sp = train(n_players=3, heur_frac_start=1.0, heur_frac_end=0.25,
               target_refresh=5, log=True)

    # Engine-style eval of a full opening: instant, one forward pass per successor.
    s = new_game(3, n_removed=9, rng=random.Random(1))
    ev = evaluate_v(s, sp)
    print(f"\nengine eval (self-play net) — card {s.active} face-up (pot {s.pot}),"
          f" P{ev['to_move']} to move:")
    for a, v in ev["mover_ev"].items():
        print(f"  {a:5s} -> EV {v:+7.2f}   vec {tuple(round(x, 1) for x in ev['actions'][a])}")
    print(f"  best: {ev['best_action']}")

    # Two metrics, because they say different things. Against the heuristic, the
    # heuristic-trained baseline specialises in exploiting it; head-to-head between
    # the nets is the fairer test of general strength.
    print("\nvs heuristic (greedy seat 0 vs heuristic seats; baseline overfits this):")
    for name, net in (("baseline ", base), ("self-play", sp)):
        h2h = head_to_head(net, n_games=1500)
        print(f"  {name}: net {h2h['vnet_mean']:.2f}  vs  heuristic"
              f" {h2h['heuristic_mean']:.2f}   win/tie {h2h['win_rate']:.1%}"
              f"  (parity ≈ {h2h['parity']:.0%})")

    print("\nself-play vs baseline (seat-balanced; lower mean / >parity win = stronger):")
    nvn = net_vs_net(sp, base, n_games=800)
    print(f"  self-play {nvn['a_mean']:.2f}  vs  baseline {nvn['b_mean']:.2f}"
          f"   self-play win/tie {nvn['a_win_rate']:.1%}  (parity ≈ {nvn['parity']:.0%})")

    # The graders above hand the net the god-view state (it sees the removed
    # cards). The deployable bot must not: head_to_head_hidden plays the net via
    # PIMC on its info set only, so its move never depends on the hidden nine.
    print("\nhonest bot — net via PIMC (public info only) vs heuristic (no cheating):")
    hid = head_to_head_hidden(sp, n_games=200, n_worlds=80)
    print(f"  self-play (PIMC) {hid['vnet_mean']:.2f}  vs  heuristic"
          f" {hid['heuristic_mean']:.2f}   win/tie {hid['win_rate']:.1%}"
          f"  (parity ≈ {hid['parity']:.0%})")


if __name__ == "__main__":
    _demo()
