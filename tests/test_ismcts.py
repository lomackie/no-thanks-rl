"""Information-Set MCTS on the belief game (nothanks.ismcts).

The guarantees worth pinning: (1) the search is honest — it acts on info sets and
never inspects the removed cards, and is reproducible given a seed; (2) forced
moves short-circuit without searching; (3) it converges to the belief-correct
self-interested optimum (``belief.solve``); (4) measured by the belief-correct
exploitability of :mod:`nothanks.belief`, it leaves *less* on the table than PIMC,
which is the whole point — a single info-set tree removes PIMC's strategy fusion.
"""

import random

import pytest

from nothanks.engine import initial_state
from nothanks import belief
from nothanks.imperfect import (
    InfoSet,
    determinize,
    determinized_action,
    info_from_state,
    legal_actions,
)
from nothanks.ismcts import (
    ISMCTSBot,
    heuristic_rollout,
    ismcts_action,
    ismcts_evaluate,
    make_ismcts_policy,
    make_value_leaf,
)
from nothanks.solver import evaluate as solver_evaluate

UNIVERSE = frozenset({3, 4, 5, 6, 7, 8})


def _small_info(deck, n_removed, start_chips=2, n_players=2):
    s = initial_state(n_players, deck, start_chips=start_chips)
    return info_from_state(s, n_removed=n_removed, deck=frozenset(deck))


def test_heuristic_rollout_returns_absolute_score_vector():
    info = _small_info([3, 4, 5, 6, 7], n_removed=1)
    v = heuristic_rollout(info, random.Random(0))
    assert len(v) == info.n_players
    assert all(isinstance(x, float) for x in v)


def test_forced_move_short_circuits_without_searching():
    # A chipless mover has one legal action; the search must return it directly and
    # never call the (here, exploding) leaf evaluator.
    info = InfoSet(
        chips=(0, 2), cards=(frozenset(), frozenset()), active=5, pot=0,
        to_move=0, deck=UNIVERSE, n_removed=2,
    )
    assert legal_actions(info) == ("take",)

    def boom(_info, _rng):
        raise AssertionError("evaluator must not run for a forced move")

    assert ismcts_action(info, n_iter=99, evaluator=boom) == "take"


def test_action_is_a_pure_function_of_public_info():
    # Two god-view worlds that differ only in which cards are hidden share an info
    # set; since the search consumes the info set, the move cannot depend on the
    # removed cards, and a fixed seed makes it reproducible.
    info = _small_info([3, 4, 5, 6, 7, 8], n_removed=2)
    wa = determinize(info, random.Random(1))
    wb = determinize(info, random.Random(2))
    assert wa.remaining != wb.remaining
    assert info_from_state(wa, 2, deck=UNIVERSE) == info_from_state(wb, 2, deck=UNIVERSE) == info

    a1 = ismcts_action(info, n_iter=400, rng=random.Random(0))
    a2 = ismcts_action(info, n_iter=400, rng=random.Random(0))
    assert a1 == a2
    assert a1 in legal_actions(info)


def test_evaluate_reports_stats_and_picks_most_visited():
    info = _small_info([3, 4, 5, 6, 7, 8], n_removed=2)
    res = ismcts_evaluate(info, n_iter=600, rng=random.Random(0))
    assert set(res) == {"to_move", "actions", "mover_ev", "visits", "best_action", "n_iter"}
    assert set(res["visits"]) == set(legal_actions(info))
    assert sum(res["visits"].values()) <= res["n_iter"]
    # best_action is the most-visited root child (the robust choice).
    assert res["best_action"] == max(res["visits"], key=res["visits"].get)
    assert set(res["actions"]) <= set(legal_actions(info))


def test_converges_to_belief_optimum_action():
    # A position where the self-interested belief optimum is to PASS. IS-MCTS,
    # searching the info-set game, recovers it robustly across seeds.
    info = InfoSet(
        chips=(1, 1), cards=(frozenset(), frozenset()), active=7, pot=2,
        to_move=0, deck=UNIVERSE, n_removed=2,
    )
    assert belief.solve(info, {})[1] == "pass"
    for seed in range(4):
        assert ismcts_action(info, n_iter=1200, rng=random.Random(seed)) == "pass"


def test_no_removal_converges_to_perfect_info_optimum():
    # With nothing hidden the belief game is the ordinary game; the search must
    # still land on the optimal opening move.
    info = _small_info([3, 4, 5, 6, 7], n_removed=0)
    opt = belief.solve(info, {})[1]
    for seed in range(3):
        assert ismcts_action(info, n_iter=1200, rng=random.Random(seed)) == opt


def test_less_exploitable_than_pimc_under_hidden_cards():
    # The headline: belief-correct exploitability of IS-MCTS < that of PIMC. PIMC is
    # given the *strongest* per-world leaf (the exact solver), so any residual gain
    # is pure strategy fusion — fixing a whole world it cannot actually see. The
    # info-set tree commits to one move per info set and undercuts that floor.
    # Deterministic: every policy is seeded from the info set (hash of int/frozenset
    # fields is stable), so these totals are reproducible.
    info = _small_info([3, 4, 5, 6, 7], n_removed=1)

    def pimc_policy(i):
        return determinized_action(
            i, solver_evaluate, n_worlds=8, rng=random.Random(hash((0, i)))
        )

    heur = belief.exploitability(info, belief.make_heuristic_policy(0))["total"]
    pimc = belief.exploitability(info, pimc_policy)["total"]
    ismcts = belief.exploitability(info, make_ismcts_policy(n_iter=1500, c=1.5, seed=0))["total"]

    assert ismcts < pimc < heur
    assert ismcts == pytest.approx(belief.exploitability(
        info, make_ismcts_policy(n_iter=1500, c=1.5, seed=0))["total"])  # reproducible


def test_value_leaf_is_honest_and_absolute_frame():
    # The net leaf must return one score per seat in ABSOLUTE order (mover-frame
    # prediction rolled by to_move) and be identical across worlds behind the
    # same info set (it consumes public features only).
    from nothanks.beliefnet import make_info_net, predict_info

    net = make_info_net(2, hidden=8, seed=1)
    info = _small_info([3, 4, 5, 6, 7, 8], n_removed=2)
    leaf = make_value_leaf(net)
    v = leaf(info, random.Random(0))
    assert len(v) == 2

    import numpy as np
    want = np.roll(predict_info(net, info), info.to_move)
    assert np.allclose(v, want)
    # Pure function of the info set: a second call agrees (the rng is unused).
    assert leaf(info, random.Random(99)) == v


def test_ismcts_with_trained_value_leaf_finds_belief_optimum():
    # The deployment path: train an info net on this configuration and use it as
    # the search leaf. An *untrained* net leaf actively misleads the search here —
    # biased leaf values do not wash out at these iteration counts — so the leaf
    # must carry signal, which is exactly what training provides.
    from nothanks.beliefnet import train_info

    info = InfoSet(
        chips=(1, 1), cards=(frozenset(), frozenset()), active=7, pot=2,
        to_move=0, deck=UNIVERSE, n_removed=2,
    )
    assert belief.solve(info, {})[1] == "pass"
    net = train_info(n_players=2, deck=sorted(UNIVERSE), n_removed=2,
                     start_chips=2, iterations=25, games_per_iter=40,
                     epochs_per_iter=2, hidden=32, heur_frac_start=1.0,
                     heur_frac_end=0.5, target_refresh=3, seed=0)
    leaf = make_value_leaf(net)
    for seed in range(4):
        assert ismcts_action(info, n_iter=800, evaluator=leaf,
                             rng=random.Random(seed)) == "pass"


def test_bot_reuses_its_tree_across_moves():
    info = _small_info([3, 4, 5, 6, 7, 8], n_removed=2)
    bot = ISMCTSBot(n_iter=200, seed=0)
    a = bot.act(info)
    assert a in legal_actions(info)
    size_after_first = len(bot.tree)
    assert size_after_first > 0
    assert info in bot.tree

    # Acting at a successor info set keeps the old statistics (warm start) and
    # only grows the tree.
    nxt = belief.apply_pass(info)
    b = bot.act(nxt)
    assert b in legal_actions(nxt)
    assert info in bot.tree  # previous root still there
    assert len(bot.tree) >= size_after_first


def test_bot_forced_move_short_circuits():
    info = InfoSet(
        chips=(0, 2), cards=(frozenset(), frozenset()), active=5, pot=0,
        to_move=0, deck=UNIVERSE, n_removed=2,
    )

    def boom(_info, _rng):
        raise AssertionError("evaluator must not run for a forced move")

    bot = ISMCTSBot(n_iter=50, evaluator=boom, seed=0)
    assert bot.act(info) == "take"
    assert bot.tree == {}
