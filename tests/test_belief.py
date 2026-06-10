"""Belief-exact exploitability under hidden removed cards (nothanks.belief).

The headline guarantees: (1) with nothing removed it must agree exactly with the
perfect-information exploit module; (2) the belief-game self-interested optimum is
a fixed point of best response (~0 exploitability); (3) the heuristic is genuinely
exploitable once cards are hidden; (4) the belief dynamics conserve probability
and stay public.
"""

import random

from nothanks.belief import (
    apply_pass,
    best_response_value,
    exploitability,
    final_scores,
    make_heuristic_policy,
    optimal_policy,
    policy_value,
    take_outcomes,
)
from nothanks.engine import initial_state, new_game, step
from nothanks.exploit import exploitability as state_exploitability
from nothanks.exploit import optimal_policy as state_optimal_policy
from nothanks.heuristic import heuristic_action
from nothanks.imperfect import info_from_state, pile_remaining, unseen
from nothanks.montecarlo import make_policy, policy_value as state_policy_value


def _small_info(deck, start_chips, n_removed, n_players=3, plies=0, seed=0):
    rng = random.Random(seed)
    s = initial_state(n_players, deck, start_chips=start_chips)
    for _ in range(plies):
        s = step(s, heuristic_action(s, 0), rng)
    return info_from_state(s, n_removed=n_removed, deck=frozenset(deck))


def test_take_outcomes_probability_and_publicness():
    deck = [3, 4, 5, 6, 7, 8]
    info = _small_info(deck, start_chips=2, n_removed=2)
    outs = take_outcomes(info)
    assert abs(sum(p for p, _ in outs) - 1.0) < 1e-12
    # One child per unseen card; each is uniform; the draw never reveals removed cards.
    assert len(outs) == len(unseen(info))
    nexts = {nxt.active for _, nxt in outs}
    assert nexts == set(unseen(info))
    for _, nxt in outs:
        assert pile_remaining(nxt) == pile_remaining(info) - 1


def test_no_removal_matches_perfect_info_exploit():
    # With nothing removed, info set <-> state and the belief draw is the real
    # draw, so belief policy-value and exploitability must equal the exact
    # perfect-information versions to numerical precision.
    deck = [3, 4, 5, 6, 7]
    s = initial_state(3, deck, start_chips=2)
    info = info_from_state(s, n_removed=0, deck=frozenset(deck))

    bv = policy_value(info, make_heuristic_policy(0))
    sv = state_policy_value(s, make_policy(0))
    assert all(abs(a - b) < 1e-9 for a, b in zip(bv, sv))

    be = exploitability(info, make_heuristic_policy(0))
    se = state_exploitability(s, make_policy(0))
    for key in ("base", "br", "gain"):
        assert all(abs(a - b) < 1e-9 for a, b in zip(be[key], se[key]))
    assert abs(be["total"] - se["total"]) < 1e-9

    # The perfect-info optimum and the belief optimum coincide here too.
    bo = exploitability(info, optimal_policy())
    so = state_exploitability(s, state_optimal_policy())
    assert abs(bo["total"]) < 1e-9 and abs(so["total"]) < 1e-9


def test_belief_optimum_is_a_fixed_point_of_best_response():
    # Self-interested optimal play under hidden cards cannot be exploited by a
    # belief-aware deviation: exploitability ~ 0.
    deck = [3, 4, 5, 6, 7, 8]
    info = _small_info(deck, start_chips=2, n_removed=2, n_players=2)
    res = exploitability(info, optimal_policy())
    assert res["total"] < 1e-9
    assert all(g >= -1e-12 for g in res["gain"])  # deviating never helps


def test_heuristic_is_exploitable_under_hidden_cards():
    deck = [3, 4, 5, 6, 7, 8]
    info = _small_info(deck, start_chips=2, n_removed=2, n_players=2)
    res = exploitability(info, make_heuristic_policy(0))
    # A best-responder can never do worse; the heuristic is not a fixed point, so
    # at least one seat has a strictly positive gain.
    assert all(g >= -1e-9 for g in res["gain"])
    assert res["max"] > 1e-6


def test_best_response_beats_or_matches_following_the_policy():
    # br[hero] (best-responding) <= base[hero] (just following policy), per seat.
    deck = [3, 4, 5, 6, 7, 8]
    info = _small_info(deck, start_chips=3, n_removed=2, n_players=3, plies=2)
    policy = make_heuristic_policy(0)
    base = policy_value(info, policy)
    for hero in range(info.n_players):
        v = best_response_value(info, hero, policy, {})
        assert v[hero] <= base[hero] + 1e-9


def test_final_scores_and_pass_are_public():
    info = _small_info([3, 4, 5, 6, 7, 8], start_chips=2, n_removed=2)
    p = info.to_move
    nxt = apply_pass(info)
    assert nxt.chips[p] == info.chips[p] - 1
    assert nxt.pot == info.pot + 1
    assert nxt.to_move == (p + 1) % info.n_players
    # Scores ignore the hidden cards entirely.
    fs = final_scores(info)
    assert len(fs) == info.n_players
