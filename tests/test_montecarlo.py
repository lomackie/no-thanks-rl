import random

from nothanks.engine import initial_state, is_terminal, legal_actions
from nothanks.montecarlo import (
    evaluate_mc,
    exact_action_values,
    make_policy,
    policy_value,
    rollout,
)


def test_rollout_reaches_terminal_and_is_reproducible():
    s = initial_state(3, [3, 4, 5, 6, 7, 8, 9], start_chips=3)
    policy = make_policy(0)
    # A rollout always ends in a fully-scored terminal vector.
    scores = rollout(s, policy, random.Random(0))
    assert len(scores) == 3
    assert all(isinstance(x, int) for x in scores)
    # Deterministic policy + same seed => identical rollout.
    a = rollout(s, policy, random.Random(7))
    b = rollout(s, policy, random.Random(7))
    assert a == b


def test_policy_value_is_mean_of_rollouts():
    # Exact expectation of the policy must equal the average of sampled rollouts.
    s = initial_state(3, [3, 4, 5, 6, 7], start_chips=2)
    policy = make_policy(0)
    exact = policy_value(s, policy)

    rng = random.Random(123)
    n = 40000
    acc = [0.0] * 3
    for _ in range(n):
        scores = rollout(s, policy, rng)
        for i in range(3):
            acc[i] += scores[i]
    mean = [x / n for x in acc]
    for got, want in zip(mean, exact):
        assert abs(got - want) < 0.1, (mean, exact)


def test_evaluate_mc_converges_to_exact_action_values():
    s = initial_state(3, [3, 4, 5, 6, 7], start_chips=2)
    policy = make_policy(0)
    exact = exact_action_values(s, policy)

    mc = evaluate_mc(s, n_rollouts=40000, policy=policy, rng=random.Random(99))

    assert set(mc["actions"]) == set(exact["actions"])
    for action, want_vec in exact["actions"].items():
        got_vec = mc["actions"][action]
        for got, want in zip(got_vec, want_vec):
            assert abs(got - want) < 0.1, (action, got_vec, want_vec)
    # The mover's own EV estimate should sit within a few standard errors.
    p = s.to_move
    for action in exact["actions"]:
        gap = abs(mc["mover_ev"][action] - exact["actions"][action][p])
        assert gap < 5 * mc["stderr"][action] + 1e-9


def test_evaluate_mc_best_action_matches_exact_when_clear():
    # On the degenerate no-removal sweep, taking the opener is clearly correct;
    # the estimator should agree with the exact policy analysis.
    s = initial_state(3, [3, 4, 5, 6, 7], start_chips=2)
    policy = make_policy(0)
    exact = exact_action_values(s, policy)
    mc = evaluate_mc(s, n_rollouts=20000, policy=policy, rng=random.Random(1))
    assert mc["best_action"] == exact["best_action"]


def test_evaluate_mc_handles_forced_take():
    # A chipless mover has only one legal action; evaluate_mc must still work.
    s = initial_state(2, [3, 4, 5], start_chips=0)
    assert legal_actions(s) == ("take",)
    mc = evaluate_mc(s, n_rollouts=200, rng=random.Random(0))
    assert set(mc["actions"]) == {"take"}
    assert mc["best_action"] == "take"
    assert not is_terminal(s)
