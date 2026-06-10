import random

import numpy as np

from nothanks.features import seat_order
from nothanks.train import (
    BEHAVIORS,
    _lambda_returns,
    head_to_head,
    selfplay_game,
    train,
)
from nothanks.valuefn import ValueNet


def _episode(net, seed=0):
    rng = random.Random(seed)
    return selfplay_game(net, rng, eps=0.3, behavior=BEHAVIORS["heuristic"])


def test_lambda_one_is_monte_carlo_return():
    # With λ=1 every state's target is just the realised final scores, reordered
    # into that state's mover frame — the Monte-Carlo return.
    net = ValueNet(3, hidden=8, seed=1)
    steps = _episode(net, seed=2)
    final_abs = steps[-1].final_abs
    assert final_abs is not None
    X, T = _lambda_returns(net, steps, lam=1.0)
    n = net.n_players
    for st, target in zip(steps, T):
        want = [final_abs[(st.mover + k) % n] for k in range(n)]
        assert np.allclose(target, want)


def test_lambda_zero_is_one_ply_bootstrap():
    # With λ=0 a non-terminal step's target is the successor's value, rotated to
    # the mover frame; the terminal step's target is the final scores.
    net = ValueNet(3, hidden=8, seed=3)
    steps = _episode(net, seed=4)
    X, T = _lambda_returns(net, steps, lam=0.0)
    n = net.n_players

    # Recompute absolute successor values independently.
    preds = net.forward(np.stack([st.feat for st in steps]))
    U_abs = [np.roll(preds[i], steps[i].mover) for i in range(len(steps))]

    for t, st in enumerate(steps):
        if st.final_abs is not None:
            want = [st.final_abs[(st.mover + k) % n] for k in range(n)]
        else:
            g = U_abs[t + 1]
            want = [g[(st.mover + k) % n] for k in range(n)]
        assert np.allclose(T[t], want), t


def test_lambda_returns_shapes_and_frame():
    net = ValueNet(4, hidden=8, seed=5)
    rng = random.Random(6)
    steps = selfplay_game(net, rng, eps=0.2, behavior=BEHAVIORS["heuristic"])
    X, T = _lambda_returns(net, steps, lam=0.7)
    assert X.shape == (len(steps), X.shape[1])
    assert T.shape == (len(steps), 4)
    # Target frame: component 0 is the mover's own score (seat_order index 0).
    assert seat_order(steps[0].mover, 4)[0] == steps[0].mover


def test_train_smoke_runs_and_predicts():
    # A couple of tiny iterations should run end-to-end and yield finite values.
    net = train(n_players=3, iterations=2, games_per_iter=4, epochs_per_iter=1, seed=0)
    from nothanks.engine import new_game

    s = new_game(3, n_removed=9, rng=random.Random(0))
    v = net.predict(s)
    assert v.shape == (3,)
    assert np.all(np.isfinite(v))


def test_head_to_head_structure():
    net = ValueNet(3, hidden=8, seed=7)
    res = head_to_head(net, n_games=20)
    assert set(res) == {"vnet_mean", "heuristic_mean", "win_rate", "parity"}
    assert 0.0 <= res["win_rate"] <= 1.0
    assert abs(res["parity"] - 1 / 3) < 1e-9
