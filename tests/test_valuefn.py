import random

import numpy as np

from nothanks.engine import initial_state, is_terminal, legal_actions, new_game, step
from nothanks.features import feature_dim, features, seat_order
from nothanks.montecarlo import make_policy, policy_value
from nothanks.valuefn import ValueNet, action_values, evaluate_v


def test_feature_dim_and_determinism():
    s = new_game(3, n_removed=9, rng=random.Random(0))
    f = features(s)
    assert f.shape == (feature_dim(3),)
    assert np.array_equal(f, features(s))  # pure function of state
    assert f.dtype == np.float32


def test_backprop_matches_numerical_gradient():
    net = ValueNet(n_players=3, hidden=8, seed=1)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, feature_dim(3)))
    T = rng.normal(size=(5, 3))

    def loss(net):
        Y = net.forward(X)
        r = Y - T
        return 0.5 * float((r * r).sum()) / X.shape[0]

    Y = net.forward(X)
    grads = net.backward((Y - T) / X.shape[0])

    eps = 1e-6
    for name in ("W1", "b1", "W2", "b2"):
        P = getattr(net, name)
        flat = P.ravel()
        g = grads[name].ravel()
        for j in range(0, flat.size, max(flat.size // 7, 1)):  # spot-check entries
            orig = flat[j]
            flat[j] = orig + eps
            lp = loss(net)
            flat[j] = orig - eps
            lm = loss(net)
            flat[j] = orig
            num = (lp - lm) / (2 * eps)
            assert abs(num - g[j]) < 1e-4, (name, j, num, g[j])


def test_overfit_small_batch():
    net = ValueNet(n_players=3, hidden=32, seed=2)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(16, feature_dim(3)))
    T = rng.normal(size=(16, 3))
    first = net.train_step(X, T, lr=0.02)
    for _ in range(3000):
        loss = net.train_step(X, T, lr=0.02)
    assert loss < first
    assert loss < 1e-3  # an MLP should comfortably fit 16 points


def test_take_keeps_perspective_pass_rotates():
    # action_values must return vectors in the mover's frame for both actions.
    s = initial_state(3, [3, 4, 5, 6, 7], start_chips=2)
    net = ValueNet(n_players=3, hidden=8, seed=4)
    av = action_values(s, net)
    assert set(av) == set(legal_actions(s))
    for v in av.values():
        assert v.shape == (3,)

    # The rotation is exactly np.roll by one seat: rebuild it by hand.
    from nothanks.engine import apply_pass

    sp = apply_pass(s)
    by_hand = np.roll(net.predict(sp), 1)
    assert np.allclose(av["pass"], by_hand)


def test_evaluate_v_shape_and_best_action():
    s = new_game(4, n_removed=9, rng=random.Random(5))
    net = ValueNet(n_players=4, hidden=16, seed=6)
    ev = evaluate_v(s, net)
    assert ev["to_move"] == s.to_move
    assert set(ev["actions"]) == set(legal_actions(s))
    assert ev["best_action"] in legal_actions(s)
    # mover_ev is seat-0 of the action vector.
    for a in ev["actions"]:
        assert ev["mover_ev"][a] == ev["actions"][a][0]


def test_save_load_roundtrip(tmp_path):
    net = ValueNet(n_players=3, hidden=16, seed=7)
    s = new_game(3, n_removed=9, rng=random.Random(8))
    before = net.predict(s)
    path = tmp_path / "net.npz"
    net.save(path)
    loaded = ValueNet.load(path)
    assert np.allclose(before, loaded.predict(s))


def test_net_can_regress_exact_policy_value():
    # Supervised check that features+net capture real signal: label states on a
    # tiny deck with the EXACT heuristic-policy value (the step-2 oracle) and fit.
    policy = make_policy(0)
    deck = [3, 4, 5, 6, 7]
    rng = random.Random(0)

    # Collect distinct non-terminal states by random self-play.
    states = {}
    for _ in range(400):
        s = initial_state(3, deck, start_chips=2)
        while not is_terminal(s):
            states[s] = True
            a = rng.choice(legal_actions(s))
            s = step(s, a, rng)

    memo: dict = {}
    X, T = [], []
    for s in states:
        pv = policy_value(s, policy, memo)
        X.append(features(s))
        T.append([pv[q] for q in seat_order(s.to_move, 3)])
    X = np.stack(X)
    T = np.array(T, dtype=float)

    net = ValueNet(n_players=3, hidden=64, seed=1)
    first = net.train_step(X, T, lr=0.01)  # full-batch (dataset is small)
    for _ in range(6000):
        loss = net.train_step(X, T, lr=0.01)
    assert loss < 0.1 * first  # learns the value surface well

    # Predictions should correlate strongly with the exact targets.
    pred = net.forward(X)
    corr = np.corrcoef(pred.ravel(), T.ravel())[0, 1]
    assert corr > 0.95, corr
