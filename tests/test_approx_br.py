"""Approximate best response (nothanks.approx_br).

The learned responder's gain is a lower bound on true exploitability, and on
tiny games the exact machinery grades it from both sides: at any single opening
the learned deviation can never beat the exact best response (a theorem), and a
competent learner must recover a decent fraction of the exact gain (the
empirical bar). The Monte-Carlo gain estimator is checked against the exact
value of the *same* learned deviation averaged over the opening distribution.
Everything is seeded, so these are deterministic.
"""

import random

from nothanks import belief
from nothanks.approx_br import (
    br_policy,
    estimate_gain_mc,
    train_best_response,
)
from nothanks.beliefnet import new_belief_game
from nothanks.imperfect import InfoSet, legal_actions

DECK = [3, 4, 5, 6, 7]
SETUP = dict(n_players=2, deck=DECK, n_removed=1, start_chips=2)

_NET_CACHE: dict = {}


def _opening(card: int) -> InfoSet:
    return InfoSet(chips=(2, 2), cards=(frozenset(), frozenset()), active=card,
                   pot=0, to_move=0, deck=frozenset(DECK), n_removed=1)


def _train(policy, hero, seed=0):
    key = (hero, seed)
    if key not in _NET_CACHE:  # training is ~1s; share across tests
        _NET_CACHE[key] = train_best_response(
            policy, hero, eps_end=0.02, seed=seed, **SETUP,
        )
    return _NET_CACHE[key]


def test_learned_gain_is_sandwiched_by_exact_best_response():
    info = new_belief_game(rng=random.Random(3), **SETUP)
    policy = belief.make_heuristic_policy(0)
    exact = belief.exploitability(info, policy)
    hero = max(range(2), key=lambda h: exact["gain"][h])
    assert exact["gain"][hero] > 1e-6  # the heuristic leaks here (cf. test_belief)

    net = _train(policy, hero)
    # Exact value of the learned deviation: hero on greedy(net), others on policy.
    combined = belief.policy_value(info, br_policy(policy, net, hero))
    gain_hat = exact["base"][hero] - combined[hero]

    # Upper bound is a theorem (exact BR is optimal); the lower bar is empirical —
    # the learner must recover a substantial fraction of the available gain.
    assert gain_hat <= exact["gain"][hero] + 1e-9
    assert gain_hat >= 0.5 * exact["gain"][hero], (gain_hat, exact["gain"][hero])


def test_estimate_gain_mc_matches_exact_average_over_openings():
    policy = belief.make_heuristic_policy(0)
    hero = 1  # the exploitable seat on this configuration (see test above)
    net = _train(policy, hero)
    deviate = br_policy(policy, net, hero)

    # Ground truth for the estimator: the exact gain of the SAME learned
    # deviation, averaged over the uniform opening card (the estimator's
    # initial-state distribution). Note this is the value of the learned
    # responder, not of the exact best response.
    exact_avg = sum(
        belief.policy_value(_opening(c), policy)[hero]
        - belief.policy_value(_opening(c), deviate)[hero]
        for c in DECK
    ) / len(DECK)

    res = estimate_gain_mc(policy, net, hero, n_games=3000, seed=123, **SETUP)
    assert set(res) == {"hero", "base", "br", "gain", "stderr", "n_games"}
    assert res["stderr"] > 0.0
    assert abs(res["gain"] - exact_avg) < 5 * res["stderr"], (res, exact_avg)
    # At the default training budget the bound is informative (positive), i.e.
    # the heuristic's leak is detected without any exact solving.
    assert res["gain"] > 0.0


def test_br_policy_is_deterministic_and_legal():
    policy = belief.make_heuristic_policy(0)
    net = _train(policy, hero=0, seed=1)
    deviate = br_policy(policy, net, hero=0)
    info = new_belief_game(rng=random.Random(8), **SETUP)

    a = deviate(info)
    assert a in legal_actions(info)
    assert deviate(info) == a  # pure function of the info set
