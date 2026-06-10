"""Info-set-native value net (nothanks.beliefnet).

What must hold: (1) the public features really are public — two worlds behind one
info set encode identically (the exact property the god-view features violate);
(2) the belief-game simulator follows the belief dynamics; (3) the one-ply
lookahead respects the mover-frame conventions (take keeps perspective, pass
rotates); (4) training runs end-to-end on the belief game and the greedy policy
is a legal deterministic InfoPolicy, gradeable by belief.exploitability; (5) the
trained net actually learns — on a tiny game it is less exploitable than an
untrained one.
"""

import random

import numpy as np

from nothanks import belief
from nothanks.beliefnet import (
    belief_step,
    evaluate_info,
    greedy_info_action,
    head_to_head_info,
    info_action_values,
    make_greedy_info_policy,
    make_info_net,
    new_belief_game,
    predict_info,
    selfplay_belief_game,
    train_info,
)
from nothanks.engine import is_terminal as state_is_terminal
from nothanks.engine import new_game, step
from nothanks.features import info_feature_dim, info_features
from nothanks.heuristic import heuristic_action
from nothanks.imperfect import determinize, info_from_state, legal_actions


def _midgame_info(seed=7, plies=3, n_removed=9):
    rng = random.Random(seed)
    s = new_game(3, n_removed=n_removed, rng=rng)
    for _ in range(plies):
        if state_is_terminal(s):
            break
        s = step(s, heuristic_action(s, 0), rng)
    return info_from_state(s, n_removed=n_removed)


def test_info_features_shape_and_publicness():
    info = _midgame_info()
    f = info_features(info)
    assert f.shape == (info_feature_dim(3),)
    assert f.dtype == np.float32
    # The decisive property: worlds that differ only in the hidden nine share an
    # info set and therefore an encoding — the god-view features fail this.
    wa = determinize(info, random.Random(1))
    wb = determinize(info, random.Random(2))
    assert wa.remaining != wb.remaining
    fa = info_features(info_from_state(wa, 9))
    fb = info_features(info_from_state(wb, 9))
    assert np.array_equal(fa, fb)


def test_info_net_predicts_and_never_peeks():
    net = make_info_net(3, hidden=8, seed=1)
    info = _midgame_info()
    v = predict_info(net, info)
    assert v.shape == (3,)
    assert np.all(np.isfinite(v))
    # Same info set => same value, regardless of the true pile behind it.
    wa = determinize(info, random.Random(1))
    wb = determinize(info, random.Random(2))
    va = predict_info(net, info_from_state(wa, 9))
    vb = predict_info(net, info_from_state(wb, 9))
    assert np.allclose(va, vb)


def test_belief_game_simulation_follows_belief_dynamics():
    rng = random.Random(0)
    info = new_belief_game(3, deck=[3, 4, 5, 6, 7, 8], n_removed=2, start_chips=3, rng=rng)
    assert info.active in info.deck
    seen_terminal = False
    for _ in range(200):
        if belief.is_terminal(info):
            seen_terminal = True
            break
        a = rng.choice(legal_actions(info))
        info = belief_step(info, a, rng)
    assert seen_terminal
    assert len(belief.final_scores(info)) == 3


def test_one_ply_lookahead_frames():
    # Pass must be the successor's prediction rotated one seat; take must stay in
    # the mover's frame (and equal the probability-weighted successor average).
    net = make_info_net(3, hidden=8, seed=2)
    info = _midgame_info(seed=11)
    av = info_action_values(info, net)
    assert set(av) == set(legal_actions(info))

    by_hand = np.roll(predict_info(net, belief.apply_pass(info)), 1)
    assert np.allclose(av["pass"], by_hand)

    acc = np.zeros(3)
    from nothanks.features import seat_order
    for prob, nxt in belief.take_outcomes(info):
        if belief.is_terminal(nxt):
            fs = belief.final_scores(nxt)
            acc += prob * np.array([fs[q] for q in seat_order(info.to_move, 3)])
        else:
            acc += prob * predict_info(net, nxt)
    assert np.allclose(av["take"], acc)


def test_evaluate_info_shape_and_greedy_consistency():
    net = make_info_net(3, hidden=8, seed=3)
    info = _midgame_info(seed=5)
    ev = evaluate_info(info, net)
    assert ev["to_move"] == info.to_move
    assert set(ev["actions"]) == set(legal_actions(info))
    for a in ev["actions"]:
        assert ev["mover_ev"][a] == ev["actions"][a][0]
    assert ev["best_action"] == greedy_info_action(info, net)


def test_selfplay_belief_game_steps_have_targetable_shape():
    net = make_info_net(3, hidden=8, seed=4)
    from nothanks.beliefnet import heuristic_info_behavior
    steps = selfplay_belief_game(net, random.Random(0), eps=0.2,
                                 behavior=heuristic_info_behavior)
    assert steps and steps[-1].final_abs is not None
    assert all(st.final_abs is None for st in steps[:-1])
    assert steps[0].feat.shape == (info_feature_dim(3),)


def test_train_info_smoke_and_grading_runs():
    net = train_info(n_players=3, iterations=2, games_per_iter=4,
                     epochs_per_iter=1, seed=0)
    info = _midgame_info()
    assert np.all(np.isfinite(predict_info(net, info)))
    res = head_to_head_info(net, n_games=10)
    assert set(res) == {"vnet_mean", "vnet_stderr", "heuristic_mean", "win_rate", "parity"}
    assert 0.0 <= res["win_rate"] <= 1.0


def test_trained_info_net_is_gradeable_and_learns_on_tiny_game():
    # The whole point of the info net: belief.exploitability can grade it exactly.
    deck = [3, 4, 5, 6, 7]
    kwargs = dict(n_players=3, deck=deck, n_removed=1, start_chips=2,
                  epochs_per_iter=2, games_per_iter=40, lam=0.9,
                  heur_frac_start=1.0, heur_frac_end=0.5, target_refresh=3)
    blank = make_info_net(3, hidden=32, seed=9)
    trained = train_info(iterations=25, hidden=32, seed=0, **kwargs)

    info = new_belief_game(3, deck=deck, n_removed=1, start_chips=2,
                           rng=random.Random(3))
    ex_blank = belief.exploitability(info, make_greedy_info_policy(blank))
    ex_trained = belief.exploitability(info, make_greedy_info_policy(trained))
    # Both are valid info policies (the metric runs); training must reduce the
    # best-response gain — the untrained net is essentially a random policy.
    assert ex_trained["total"] < ex_blank["total"]
