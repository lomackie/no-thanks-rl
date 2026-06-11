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
import pathlib
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
from .engine import STARTING_CHIPS, full_deck, new_game, score_delta
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


_MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"


def default_net_path(n_players: int) -> pathlib.Path | None:
    """The conventionally-named saved info net for a player count, if any.

    Newest generation first: ``_v3`` (the cheap-anchor-take repair of roadmap
    step 19) over ``_v2`` (the gapped-high-card repair of step 16) over the
    base name (the 3p original, kept as the historical reference the
    step-11..14 scripts measured). ``None`` when no saved net exists for this
    player count, in which case callers fall back to the heuristic-playout
    leaf.
    """
    for name in (f"info_net_{n_players}p_v3.npz",
                 f"info_net_{n_players}p_v2.npz",
                 f"info_net_{n_players}p.npz"):
        path = _MODELS_DIR / name
        if path.exists():
            return path
    return None


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


class _GreedyInfoPolicy:
    """Callable (hence picklable) form of :func:`make_greedy_info_policy`."""

    def __init__(self, net: ValueNet):
        self.net = net

    def __call__(self, info: InfoSet) -> str:
        return greedy_info_action(info, self.net)


def make_greedy_info_policy(net: ValueNet) -> InfoPolicy:
    """A deterministic ``InfoPolicy`` for :func:`nothanks.belief.exploitability`."""
    return _GreedyInfoPolicy(net)


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


class _SearchInfoBehavior:
    """Callable (hence picklable) form of :func:`make_search_info_behavior`."""

    def __init__(self, n_iter: int, c: float, leaf: str = "net"):
        self.n_iter = n_iter
        self.c = c
        self.leaf = leaf

    def __call__(self, info: InfoSet, net: ValueNet, rng: random.Random, eps: float) -> str:
        from .ismcts import ismcts_action, make_value_leaf

        actions = legal_actions(info)
        if len(actions) == 1:
            return actions[0]
        if rng.random() < eps:
            return rng.choice(actions)
        evaluator = make_value_leaf(net) if self.leaf == "net" else None
        return ismcts_action(info, n_iter=self.n_iter,
                             evaluator=evaluator, c=self.c, rng=rng)


def make_search_info_behavior(n_iter: int = 200, c: float = 30.0,
                              leaf: str = "net") -> InfoBehavior:
    """ε-greedy on an IS-MCTS move with the *current* net as leaf.

    The expert-iteration data policy (roadmap step 16): games played by the
    search are stronger than one-ply greedy play, so their λ-returns pull the
    net toward values the search would realise — in particular on the moves
    where one-ply lookahead and the search disagree (the gapped-high-card take
    bias is exactly such a class). Expensive (~one search per decision), so it
    is annealed in via ``search_frac`` rather than used for every game.

    ``leaf="playout"`` searches over the honest heuristic playout instead of
    the net (roadmap step 19). A net-leaf searcher inherits the net's own
    state-value biases, so its games cannot correct them; the playout leaf is
    independent of the net and plays the cheap-anchor takes correctly, making
    its games the corrective trajectories. Slower (a full playout per search
    iteration), so it gets its own smaller band (``psearch_frac``).
    """
    return _SearchInfoBehavior(n_iter, c, leaf)


def selfplay_belief_game(
    net: ValueNet,
    rng: random.Random,
    eps: float,
    behavior: InfoBehavior,
    deck=None,
    n_removed: int = 9,
    start_chips: int | None = None,
    deviate_at: int | None = None,
    deviate_take_at: int | None = None,
    take_margin: int = 2,
) -> list[_Step]:
    """Play one belief game under ``behavior``; return its decision steps.

    Reuses :class:`nothanks.train._Step` so the λ-return machinery is shared —
    the frame logic is identical, only the simulator and features differ.

    ``deviate_at`` is the **exploring-deviation** trick (step 16's calibration
    lesson): at the ``deviate_at``-th *free* decision (one with a real choice),
    one uniform-random action is forced, play continues on-policy, and only the
    strictly-post-deviation suffix is returned. Rare states — e.g. "just took a
    gapped high card" — get visited, and their values reflect *competent*
    continuation; unlike a high sustained ε, the on-policy value scale is not
    inflated (the deviation move itself and everything before it are dropped,
    so no target ever averages over the forced random move). If the game ends
    before the ``deviate_at``-th free decision the whole game is returned (it
    was an ordinary on-policy game). May return ``[]`` when the deviation was
    the final decision.

    ``deviate_take_at`` is the **take-biased** variant (roadmap step 19): the
    forced action is ``take``, at the ``deviate_take_at``-th *cheap-take
    opportunity* — a free decision where ``score_delta(active) − pot ≤
    take_margin``. Uniform deviations rarely land on these decisions *and*
    pick take, so the net never sees competent play after owning a cheap run
    anchor (e.g. the opening 3 for 2 chips) and learns to undervalue those
    states — a bias the search then inherits through every leaf of the take
    subtree, where no iteration budget can fix it. Same suffix-only training,
    so the calibration property is unchanged. At most one of ``deviate_at`` /
    ``deviate_take_at`` should be set.
    """
    info = new_belief_game(net.n_players, deck=deck, n_removed=n_removed,
                           start_chips=start_chips, rng=rng)
    steps: list[_Step] = []
    n_free = 0
    n_cheap = 0
    cut = -1
    while not is_terminal(info):
        feat = info_features(info)
        mover = info.to_move
        acts = legal_actions(info)
        if deviate_at is not None and len(acts) > 1 and n_free == deviate_at:
            a = rng.choice(acts)
            cut = len(steps)  # drop this step and everything before it
        elif (deviate_take_at is not None and len(acts) > 1
              and score_delta(info.cards[mover], info.active) - info.pot <= take_margin):
            if n_cheap == deviate_take_at:
                a = "take"
                cut = len(steps)
            else:
                a = behavior(info, net, rng, eps)
            n_cheap += 1
        else:
            a = behavior(info, net, rng, eps)
        if len(acts) > 1:
            n_free += 1
        nxt = belief_step(info, a, rng)
        final_abs = np.array(final_scores(nxt), dtype=float) if is_terminal(nxt) else None
        steps.append(_Step(feat, mover, final_abs))
        info = nxt
    return steps[cut + 1:] if cut >= 0 else steps


def _pick_behavior(u: float, heur_frac: float, search_frac: float,
                   search_behavior: InfoBehavior, psearch_frac: float = 0.0,
                   psearch_behavior: InfoBehavior | None = None) -> InfoBehavior:
    """The curriculum draw: heuristic / search / playout-search / greedy by one
    uniform sample (band order keeps the historical map when the new band is 0)."""
    if u < heur_frac:
        return heuristic_info_behavior
    if u < heur_frac + search_frac:
        return search_behavior
    if u < heur_frac + search_frac + psearch_frac:
        return psearch_behavior
    return greedy_info_behavior


def _selfplay_batch(args) -> list:
    """Worker for the ``n_jobs`` pool: play a batch of pre-seeded belief games.

    Everything arrives pickled (the net snapshot, the behavior objects, the rule
    parameters); the per-game seeds were drawn by the parent, so the generated
    data is a pure function of the spec — deterministic, and (because the parent
    chunks the spec list contiguously and concatenates results in order)
    independent of the worker count.
    """
    net, specs, eps, deck, n_removed, start_chips, take_margin = args
    return [
        selfplay_belief_game(net, random.Random(game_seed), eps, behavior,
                             deck=deck, n_removed=n_removed,
                             start_chips=start_chips, deviate_at=deviate_at,
                             deviate_take_at=deviate_take_at,
                             take_margin=take_margin)
        for game_seed, behavior, deviate_at, deviate_take_at in specs
    ]


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
    search_frac_start: float = 0.0,
    search_frac_end: float = 0.0,
    search_iters: int = 200,
    search_c: float = 30.0,
    psearch_frac_start: float = 0.0,
    psearch_frac_end: float = 0.0,
    deviation_frac: float = 0.0,
    deviation_horizon: int = 30,
    take_dev_frac: float = 0.0,
    take_dev_horizon: int = 8,
    take_dev_margin: int = 2,
    target_refresh: int = 1,
    hidden: int = 64,
    deck=None,
    n_removed: int = 9,
    start_chips: int | None = None,
    n_jobs: int = 1,
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

    ``search_frac`` (annealed ``search_frac_start → search_frac_end``, alongside
    ``heur_frac``) is the expert-iteration leg: that fraction of games is played
    by IS-MCTS with the current net as leaf (``search_iters`` iterations,
    exploration constant ``search_c`` — final-score scale, so ~30 on the full
    deck). Anneal it *up* from 0 so the searches only start once the net carries
    signal (an untrained leaf actively misleads the search); the remaining games
    split between the heuristic and greedy self-play as before. Requires
    ``heur_frac + search_frac ≤ 1`` throughout.

    ``psearch_frac`` (annealed like the others; roadmap step 19) is a second
    search band whose games are played by IS-MCTS over the **heuristic-playout
    leaf** instead of the net. The net-leaf searcher inherits the net's own
    state-value biases (step 16's lesson), so its games can't correct them;
    the playout leaf is independent of the net and supplies corrective
    trajectories for the classes the net mis-values (cheap-anchor takes).
    Requires ``heur_frac + search_frac + psearch_frac ≤ 1`` throughout.

    ``deviation_frac`` of games carry one **exploring deviation** (see
    :func:`selfplay_belief_game`): a single uniform-random action at a uniform
    free-decision index in ``[0, deviation_horizon)``, with only the
    post-deviation suffix trained on. This is the coverage mechanism that does
    *not* inflate the value scale — sustained ε does, and a value net used as a
    search leaf must stay calibrated against the true terminal scores it is
    mixed with inside the tree.

    ``take_dev_frac`` of the *remaining* games carry a **take-biased
    deviation** instead (step 19, see :func:`selfplay_belief_game`): a forced
    ``take`` at a uniform cheap-take-opportunity index in
    ``[0, take_dev_horizon)`` (cheap = ``score_delta − pot ≤ take_dev_margin``).
    The two kinds are mutually exclusive per game, so the effective take-share
    is ``(1 − deviation_frac) · take_dev_frac``.

    ``n_jobs > 1`` farms game generation (the wall-clock bottleneck, especially
    the search games) out to a process pool; the SGD fit stays in the parent.
    Per-game seeds are then drawn by the parent up front, so a run is
    deterministic given ``seed`` and independent of the worker count — but it
    does not reproduce the ``n_jobs=1`` stream, which keeps the original
    sequential RNG semantics for comparability with earlier runs.
    """
    net = make_info_net(n_players, hidden=hidden, seed=seed)
    target = net.copy()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    search_behavior = make_search_info_behavior(n_iter=search_iters, c=search_c)
    psearch_behavior = make_search_info_behavior(n_iter=search_iters, c=search_c,
                                                 leaf="playout")
    pool = None
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=n_jobs)

    try:
        for it in range(iterations):
            frac = it / max(iterations - 1, 1)
            eps = eps_start + (eps_end - eps_start) * frac
            heur_frac = heur_frac_start + (heur_frac_end - heur_frac_start) * frac
            search_frac = search_frac_start + (search_frac_end - search_frac_start) * frac
            psearch_frac = psearch_frac_start + (psearch_frac_end - psearch_frac_start) * frac

            if it % target_refresh == 0:
                target = net.copy()

            def draw_deviation() -> tuple[int | None, int | None]:
                # No rng draw at all when off: keeps the historical stream.
                if deviation_frac > 0.0 and rng.random() < deviation_frac:
                    return rng.randrange(deviation_horizon), None
                if take_dev_frac > 0.0 and rng.random() < take_dev_frac:
                    return None, rng.randrange(take_dev_horizon)
                return None, None

            if pool is None:
                games = []
                for _ in range(games_per_iter):
                    behavior = _pick_behavior(rng.random(), heur_frac,
                                              search_frac, search_behavior,
                                              psearch_frac, psearch_behavior)
                    deviate_at, deviate_take_at = draw_deviation()
                    games.append(
                        selfplay_belief_game(net, rng, eps, behavior, deck=deck,
                                             n_removed=n_removed,
                                             start_chips=start_chips,
                                             deviate_at=deviate_at,
                                             deviate_take_at=deviate_take_at,
                                             take_margin=take_dev_margin)
                    )
            else:
                specs = [
                    (rng.randrange(1 << 62),
                     _pick_behavior(rng.random(), heur_frac, search_frac,
                                    search_behavior, psearch_frac,
                                    psearch_behavior),
                     *draw_deviation())
                    for _ in range(games_per_iter)
                ]
                chunk = -(-len(specs) // n_jobs)  # contiguous => order-stable
                futures = [
                    pool.submit(_selfplay_batch,
                                (net, specs[i:i + chunk], eps, deck,
                                 n_removed, start_chips, take_dev_margin))
                    for i in range(0, len(specs), chunk)
                ]
                games = [g for f in futures for g in f.result()]
            games = [g for g in games if g]  # a deviation can consume a whole game
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
                      f"search {search_frac:.2f}  psearch {psearch_frac:.2f}  "
                      f"steps {len(X):5d}  loss {last_loss:.3f}",
                      flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

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
