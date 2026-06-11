"""Information-Set MCTS: a single search tree over info sets.

Why this exists
---------------
:func:`nothanks.imperfect.evaluate_determinized` (PIMC) is the strong baseline for
hidden-card play, but it has a structural blind spot. PIMC samples a *complete*
world — it fixes the entire future draw order — and then evaluates that world as
if the pile were common knowledge, finally averaging over worlds. Two errors hide
in there:

* **Strategy fusion.** Because each world is solved independently, PIMC implicitly
  lets the player choose a *different* action in worlds it cannot actually tell
  apart. A real strategy must commit to one move per info set; PIMC's per-world
  optimisation does not, so it overvalues positions whose worth depends on knowing
  the unknowable.
* **Non-locality.** Fixing the future draw order leaks information the mover should
  not have — in No Thanks the *next flipped card* is public, but the cards *after*
  it are not, yet PIMC's determinization pins them all down up front.

:mod:`nothanks.belief` showed that, for public-information policies, the hidden
game is *exactly* a Markov game on info sets: after a ``take`` the next card is
uniform over ``unseen`` and is only revealed when drawn. IS-MCTS searches **that**
game directly — one tree keyed by :class:`~nothanks.imperfect.InfoSet`, with the
chance card sampled *as the tree is descended* (and re-sampled every iteration)
rather than fixed in advance. Statistics are therefore shared across all worlds
behind an info set and the search commits to a single action per info set, which
is precisely what removes the strategy-fusion / non-locality artefacts. (This is
the single-observer IS-MCTS of Cowling, Powley & Whitehouse 2012, specialised to
No Thanks, where the only hidden state is the removal so info-set transitions are
the belief dynamics of :mod:`nothanks.belief`.)

Selfish multi-agent backup
--------------------------
Every decision node — whoever is to move — minimises *its own* expected final
score, the same self-interested convention as :func:`nothanks.belief.solve` (the
belief optimum, exploitability ~0). So each node reads the value vector's entry
for its own mover and uses a lower-confidence bound (we are minimising). On the
small games we can check exactly, the root action converges to ``belief.solve``'s
— *empirically*: selfish multi-agent UCT has no general convergence theorem
(unlike two-player zero-sum UCT), so treat that as a tested observation, not a
guarantee. Measured by :func:`nothanks.belief.exploitability`, IS-MCTS is *less
exploitable than PIMC*, quantifying the value PIMC leaves on the table.

The leaf evaluator is pluggable (:data:`LeafEvaluator`); the default is an honest
heuristic playout on the belief game, so the whole search only ever touches public
information — it never inspects the removed cards.

Scaling note: the exploration constant ``c`` is on the scale of *final scores*
(the LCB subtracts ``c·sqrt(log N / n)`` from a mean score). The default 1.5 suits
the tiny testbed configurations the tests solve exactly; on the full deck, where
score gaps between actions run to tens of points, it is far too timid — a single
noisy playout can pin an action at one visit forever. Full-game entry points
(:func:`nothanks.train.head_to_head_ismcts`, the CLI) default to ``c≈30``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .belief import (
    apply_pass,
    final_scores,
    heuristic_info_action,
    is_terminal,
    take_outcomes,
)
from .features import info_features
from .imperfect import InfoSet, legal_actions

# Estimates the absolute-frame expected final-score vector of a *non-terminal*
# info set under selfish play. The default is a heuristic playout, but any honest
# estimator fits (e.g. a determinized value-net average) as long as it returns one
# score per seat in absolute seat order and never peeks at the removed cards.
LeafEvaluator = Callable[[InfoSet, random.Random], tuple[float, ...]]


def _sample_child(info: InfoSet, rng: random.Random) -> InfoSet:
    """Sample one ``take`` outcome — the belief chance draw (uniform over unseen).

    :func:`nothanks.belief.take_outcomes` enumerates every equiprobable next card
    (or the single terminal outcome when the pile is exhausted); picking one
    uniformly *is* the chance event, so the search never fixes the future pile.
    """
    return rng.choice(take_outcomes(info))[1]


def heuristic_rollout(info: InfoSet, rng: random.Random, threshold: int = 0) -> tuple[float, ...]:
    """Play the belief game to the end with the run-aware heuristic; return scores.

    The default :data:`LeafEvaluator`. It stays entirely on info sets — passes are
    deterministic, takes sample the belief draw — so the returned absolute-frame
    final-score vector is an honest, no-peek leaf estimate.
    """
    while not is_terminal(info):
        if heuristic_info_action(info, threshold) == "pass":
            info = apply_pass(info)
        else:
            info = _sample_child(info, rng)
    return final_scores(info)


class _ValueLeaf:
    """Callable (hence picklable) form of :func:`make_value_leaf`."""

    def __init__(self, net):
        self.net = net

    def __call__(self, info: InfoSet, _rng: random.Random) -> tuple[float, ...]:
        pred = self.net.forward(info_features(info)[None, :])[0]
        return tuple(float(x) for x in np.roll(pred, info.to_move))


def make_value_leaf(net) -> LeafEvaluator:
    """A :data:`LeafEvaluator` from an **info-set** value net (one forward pass).

    ``net`` must be an info-set net (:func:`nothanks.beliefnet.make_info_net`):
    it consumes :func:`nothanks.features.info_features` — public knowledge only —
    so the leaf is honest by construction. The net predicts in the mover-relative
    frame; rolling by ``to_move`` converts to the absolute frame the backup sums.
    This is what scales the search to the full deck: a forward pass replaces a
    whole heuristic playout, and the net's value is a far stronger leaf estimate.
    The returned object is picklable, so it can cross process boundaries (the
    ``n_jobs`` training pools).
    """
    return _ValueLeaf(net)


@dataclass
class _Node:
    """Per-info-set search statistics. ``Wa[a]`` sums the absolute value vector."""

    info: InfoSet
    n: int = 0
    na: dict[str, int] = field(default_factory=dict)
    wa: dict[str, list[float]] = field(default_factory=dict)

    def mean_own(self, a: str) -> float:
        """Mean own-score of action ``a`` for *this* node's mover (lower is better)."""
        return self.wa[a][self.info.to_move] / self.na[a]


def _select(node: _Node, c: float, rng: random.Random) -> str:
    """Pick an action to descend: any unvisited first, else min lower-confidence bound.

    The mover minimises its own score, so the exploration term is *subtracted* —
    an action is attractive when its mean own-score is low or it is under-sampled.
    """
    acts = legal_actions(node.info)
    unvisited = [a for a in acts if node.na.get(a, 0) == 0]
    if unvisited:
        return rng.choice(unvisited)
    log_n = math.log(node.n + 1)
    return min(acts, key=lambda a: node.mean_own(a) - c * math.sqrt(log_n / node.na[a]))


def _simulate(
    tree: dict[InfoSet, _Node],
    info: InfoSet,
    evaluator: LeafEvaluator,
    c: float,
    rng: random.Random,
) -> tuple[float, ...]:
    """One MCTS iteration from ``info``; returns the absolute-frame score vector.

    Reaching an info set not yet in the tree is the expansion step: add it and hand
    back a leaf estimate (no stats recorded there this visit). Otherwise select an
    action, transition (deterministic pass / sampled take), recurse, then back up
    the returned vector into this node's chosen-action stats.
    """
    if is_terminal(info):
        return final_scores(info)
    node = tree.get(info)
    if node is None:
        tree[info] = _Node(info)
        return evaluator(info, rng)

    a = _select(node, c, rng)
    child = apply_pass(info) if a == "pass" else _sample_child(info, rng)
    value = _simulate(tree, child, evaluator, c, rng)

    n = info.n_players
    node.n += 1
    node.na[a] = node.na.get(a, 0) + 1
    acc = node.wa.get(a)
    if acc is None:
        node.wa[a] = list(value)
    else:
        for i in range(n):
            acc[i] += value[i]
    return value


def ismcts_evaluate(
    info: InfoSet,
    n_iter: int = 800,
    evaluator: LeafEvaluator | None = None,
    c: float = 1.5,
    rng: random.Random | None = None,
) -> dict:
    """Run IS-MCTS from ``info`` and report per-action statistics.

    Same dict shape as :func:`nothanks.imperfect.evaluate_determinized`:
    ``actions`` (mean absolute score vector per action), ``mover_ev`` (the mover's
    own mean), ``best_action`` (the *most-visited* root child — the robust choice,
    tie-broken by best mean), plus ``visits`` and ``n_iter``. Forced positions are
    reported without search.
    """
    if info.active is None:
        raise ValueError("terminal info set has no moves to evaluate")
    evaluator = evaluator or heuristic_rollout
    rng = rng or random.Random()

    acts = legal_actions(info)
    if len(acts) == 1:  # forced move: nothing to search
        return {
            "to_move": info.to_move,
            "actions": {},
            "mover_ev": {},
            "visits": {acts[0]: 0},
            "best_action": acts[0],
            "n_iter": 0,
        }

    tree: dict[InfoSet, _Node] = {}
    for _ in range(n_iter):
        _simulate(tree, info, evaluator, c, rng)
    root = tree[info]

    p = info.to_move
    actions = {a: tuple(x / root.na[a] for x in root.wa[a]) for a in root.na}
    mover_ev = {a: actions[a][p] for a in actions}
    # Most-visited child is the robust pick; ties broken toward the lower mean.
    best_action = max(root.na, key=lambda a: (root.na[a], -mover_ev[a]))
    return {
        "to_move": p,
        "actions": actions,
        "mover_ev": mover_ev,
        "visits": dict(root.na),
        "best_action": best_action,
        "n_iter": n_iter,
    }


def ismcts_action(
    info: InfoSet,
    n_iter: int = 800,
    evaluator: LeafEvaluator | None = None,
    c: float = 1.5,
    rng: random.Random | None = None,
) -> str:
    """The IS-MCTS move for ``info`` — honest (a pure function of public knowledge).

    Like :func:`nothanks.imperfect.determinized_action` it takes an
    :class:`InfoSet`, never a god-view state, so the removed cards are never
    inspected. Forced moves short-circuit.
    """
    acts = legal_actions(info)
    if not acts:
        raise ValueError("terminal info set has no action to choose")
    if len(acts) == 1:
        return acts[0]
    return ismcts_evaluate(info, n_iter, evaluator, c, rng)["best_action"]


def _info_key(info: InfoSet) -> str:
    """A canonical, process-stable string key for an info set.

    ``hash(info)`` would do within one process, but frozenset iteration order and
    (for any future str field) hash salting make it fragile across processes; a
    sorted textual key seeds :class:`random.Random` stably everywhere (str seeding
    is hashed with sha512, not the salted ``hash``).
    """
    cards = ";".join(",".join(map(str, sorted(c))) for c in info.cards)
    deck = ",".join(map(str, sorted(info.deck)))
    return (f"{info.chips}|{cards}|{info.active}|{info.pot}|{info.to_move}"
            f"|{deck}|{info.n_removed}")


class _ISMCTSPolicy:
    """Callable (hence picklable) form of :func:`make_ismcts_policy`."""

    def __init__(self, n_iter: int, evaluator: LeafEvaluator, c: float, seed: int):
        self.n_iter = n_iter
        self.evaluator = evaluator
        self.c = c
        self.seed = seed

    def __call__(self, info: InfoSet) -> str:
        acts = legal_actions(info)
        if len(acts) == 1:
            return acts[0]
        rng = random.Random(f"{self.seed}|{_info_key(info)}")
        return ismcts_evaluate(info, self.n_iter, self.evaluator, self.c,
                               rng)["best_action"]


def make_ismcts_policy(
    n_iter: int = 800,
    evaluator: LeafEvaluator | None = None,
    c: float = 1.5,
    seed: int = 0,
):
    """A deterministic :data:`nothanks.belief.InfoPolicy` wrapping IS-MCTS.

    :func:`nothanks.belief.exploitability` evaluates a policy by exact backward
    induction, so it needs ``policy(info)`` to be a *deterministic* function of the
    info set. We get that by seeding the search rng from ``seed`` and a canonical
    key of the info set (:func:`_info_key`), so the same info set always yields the
    same move — making the IS-MCTS player gradeable against the belief-correct best
    response, the headline measurement this module targets. Picklable (given a
    picklable ``evaluator``), so it can be a frozen opponent inside the ``n_jobs``
    training pools.
    """
    return _ISMCTSPolicy(n_iter, evaluator or heuristic_rollout, c, seed)


class ISMCTSBot:
    """The deployable IS-MCTS player: a persistent tree reused across moves.

    :func:`ismcts_action` rebuilds its tree from scratch every call. Within one
    game that wastes work: after a move and the opponents' replies, the new root
    is a descendant info set whose statistics the previous search already
    visited. Keeping one ``tree`` for the bot's lifetime lets every ``act`` start
    warm (stale nodes from unreachable lines are never visited again and are
    merely memory). Honest like everything here: ``act`` consumes an
    :class:`~nothanks.imperfect.InfoSet`, never a god-view state.

    Use one bot per game (or per seat); the tree is only meaningful within a
    single rule configuration (``deck``/``n_removed``/player count).
    """

    def __init__(
        self,
        n_iter: int = 800,
        evaluator: LeafEvaluator | None = None,
        c: float = 1.5,
        seed: int = 0,
    ):
        self.n_iter = n_iter
        self.evaluator = evaluator or heuristic_rollout
        self.c = c
        self.rng = random.Random(seed)
        self.tree: dict[InfoSet, _Node] = {}

    def act(self, info: InfoSet) -> str:
        acts = legal_actions(info)
        if not acts:
            raise ValueError("terminal info set has no action to choose")
        if len(acts) == 1:
            return acts[0]
        for _ in range(self.n_iter):
            _simulate(self.tree, info, self.evaluator, self.c, self.rng)
        root = self.tree[info]
        return max(root.na, key=lambda a: (root.na[a], -root.mean_own(a)))
