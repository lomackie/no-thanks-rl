import random

from nothanks.engine import full_deck, initial_state, is_terminal, new_game, step
from nothanks.heuristic import heuristic_action
from nothanks.imperfect import (
    determinize,
    evaluate_determinized,
    info_from_state,
    pile_remaining,
    seen,
    unseen,
)
from nothanks.montecarlo import evaluate_mc
from nothanks.solver import evaluate as solver_evaluate


def test_info_set_derivations_match_real_game():
    # Public counts must reconcile: unseen = pile-still-down + removed.
    rng = random.Random(3)
    s = new_game(3, n_removed=9, rng=rng)
    for _ in range(8):
        if is_terminal(s):
            break
        s = step(s, heuristic_action(s, 0), rng)

    info = info_from_state(s, n_removed=9, deck=frozenset(full_deck()))
    # The true pile is consistent with the observer's belief.
    assert pile_remaining(info) == len(s.remaining)
    assert s.remaining <= unseen(info)
    assert len(unseen(info)) - pile_remaining(info) == 9  # the hidden removed cards
    # Seen = everything flipped (captured by anyone + the active card).
    flipped = set().union(*s.cards) | {s.active}
    assert seen(info) == frozenset(flipped)


def test_determinize_is_valid_and_preserves_public_state():
    rng = random.Random(1)
    s = new_game(4, n_removed=9, rng=rng)
    for _ in range(5):
        s = step(s, heuristic_action(s, 0), rng)
    info = info_from_state(s, n_removed=9)

    for seed in range(20):
        world = determinize(info, random.Random(seed))
        assert world.remaining <= unseen(info)
        assert len(world.remaining) == pile_remaining(info)
        # Public fields are carried through untouched.
        assert world.chips == info.chips
        assert world.cards == info.cards
        assert world.active == info.active
        assert world.pot == info.pot
        assert world.to_move == info.to_move


def test_no_removal_reduces_to_direct_eval():
    # With nothing removed there is a single consistent world (the true pile),
    # so determinized eval must equal the direct exact eval regardless of n_worlds.
    s = initial_state(3, [3, 4, 5, 6, 7], start_chips=2)
    deck = frozenset(s.remaining | {s.active})
    info = info_from_state(s, n_removed=0, deck=deck)

    assert pile_remaining(info) == len(unseen(info))  # nothing hidden
    direct = solver_evaluate(s)
    det = evaluate_determinized(info, solver_evaluate, n_worlds=7, rng=random.Random(0))

    assert det["best_action"] == direct["best_action"]
    for a, vec in direct["actions"].items():
        for got, want in zip(det["actions"][a], vec):
            assert abs(got - want) < 1e-9
    for a in direct["mover_ev"]:
        assert abs(det["mover_ev"][a] - direct["mover_ev"][a]) < 1e-9
        assert det["stderr"][a] < 1e-6  # the only world is fixed => ~no variance


def test_evaluate_determinized_structure_on_hidden_game():
    rng = random.Random(5)
    s = new_game(3, n_removed=9, rng=rng)
    for _ in range(4):
        s = step(s, heuristic_action(s, 0), rng)
    info = info_from_state(s, n_removed=9)

    evaluator = lambda st: evaluate_mc(st, n_rollouts=30, rng=rng)  # noqa: E731
    det = evaluate_determinized(info, evaluator, n_worlds=15, rng=rng)

    assert det["n_hidden"] - det["pile_remaining"] == 9
    assert set(det["actions"]) == set(det["mover_ev"]) == set(det["stderr"])
    assert det["best_action"] in det["mover_ev"]
    for a, vec in det["actions"].items():
        assert len(vec) == 3
        assert det["stderr"][a] >= 0.0
