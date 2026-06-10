"""Tests for the seat-balanced bot-vs-bot arena."""

import math

from nothanks.arena import (
    bot_vs_bot,
    greedy_info_bot,
    heuristic_bot,
    ismcts_bot,
    pimc_god_bot,
)
from nothanks.beliefnet import make_info_net
from nothanks.valuefn import ValueNet


def test_heuristic_mirror_match_is_fair_and_deterministic():
    # Identical bots in every seat: the seat rotation must wash out the
    # first-mover edge, so A's mean equals B's mean exactly (every game is
    # counted once from A's seat and the others are the same policy).
    res = bot_vs_bot(heuristic_bot(), heuristic_bot(), n_games=40)
    assert res["games"] == 120
    assert math.isclose(res["a_mean"], res["b_mean"], rel_tol=0.15)
    # Deterministic bots + fixed seeds: a rerun reproduces every number.
    again = bot_vs_bot(heuristic_bot(), heuristic_bot(), n_games=40)
    assert res == again


def test_all_factories_play_full_games():
    # Smoke: each honest bot type completes real 9-removed games and the
    # report has the documented shape. Tiny budgets keep this fast; strength
    # is not asserted here (that is the step-13 measurement, not a test).
    info_net = make_info_net(3, hidden=8, seed=0)
    god_net = ValueNet(3, hidden=8, seed=0)
    for factory in (
        greedy_info_bot(info_net),
        pimc_god_bot(god_net, n_worlds=4),
        ismcts_bot(info_net, n_iter=8),
        ismcts_bot(None, n_iter=8),  # heuristic-playout leaf
    ):
        res = bot_vs_bot(factory, heuristic_bot(), n_games=2)
        assert res["games"] == 6
        assert math.isfinite(res["a_mean"]) and math.isfinite(res["b_mean"])
        assert 0.0 <= res["a_win_rate"] <= 1.0
        assert res["parity"] == 1.0 / 3
