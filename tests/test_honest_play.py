"""The deployed bot must not use knowledge of the removed cards.

The value net is, by design, a *god-view* per-world evaluator: its features
include the pile composition, so two worlds that differ only in which nine cards
are hidden get different values. The honest playing policy
(``determinized_action`` / ``pimc_policy``) wraps that evaluator in PIMC so the
*decision* is a pure function of the public :class:`InfoSet` — it can no longer
swing on the removed cards. These tests pin down both halves of that claim.
"""

import random

import numpy as np

from nothanks.engine import is_terminal, new_game, step
from nothanks.heuristic import heuristic_action
from nothanks.imperfect import (
    determinize,
    determinized_action,
    info_from_state,
    legal_actions,
)
from nothanks.train import pimc_policy
from nothanks.valuefn import ValueNet, evaluate_v


def _midgame_info(seed=7, plies=3, n_removed=9):
    """A non-terminal standard game advanced a few honest plies, with its info set."""
    rng = random.Random(seed)
    s = new_game(3, n_removed=n_removed, rng=rng)
    for _ in range(plies):
        if is_terminal(s):
            break
        s = step(s, heuristic_action(s, 0), rng)
    return s, info_from_state(s, n_removed=n_removed)


def test_godview_net_value_leaks_removed_cards():
    """Sanity: the raw net *does* read the hidden cards (that's the cheat we wrap)."""
    net = ValueNet(3, hidden=8, seed=1)
    _, info = _midgame_info()
    # Two worlds with the SAME public info but different removed (hidden) cards.
    wa = determinize(info, random.Random(1))
    wb = determinize(info, random.Random(2))
    assert wa.remaining != wb.remaining
    assert info_from_state(wa, 9) == info_from_state(wb, 9) == info
    # The god-view value depends on which cards are hidden — that is the leak.
    assert not np.allclose(net.predict(wa), net.predict(wb))


def test_honest_action_is_a_function_of_public_info_only():
    net = ValueNet(3, hidden=8, seed=1)
    _, info = _midgame_info()
    evaluator = lambda st: evaluate_v(st, net)  # noqa: E731
    # Same info + same rng => same decision, no matter the true pile.
    a1 = determinized_action(info, evaluator, n_worlds=40, rng=random.Random(0))
    a2 = determinized_action(info, evaluator, n_worlds=40, rng=random.Random(0))
    assert a1 == a2
    assert a1 in legal_actions(info)


def test_pimc_policy_ignores_the_true_removed_cards():
    """The deployable bot returns the same move for any world behind one info set."""
    net = ValueNet(3, hidden=8, seed=2)
    _, info = _midgame_info(seed=11)
    bot = pimc_policy(net, n_removed=9, n_worlds=60)
    # Build several god-view states sharing this info set (different hidden cards).
    worlds = [determinize(info, random.Random(k)) for k in range(5)]
    assert len({w.remaining for w in worlds}) > 1  # genuinely different piles
    # Seed the bot's sampler identically per call: its move can't depend on `w`.
    moves = {bot(w, random.Random(99)) for w in worlds}
    assert len(moves) == 1


def test_forced_take_needs_no_sampling():
    # A chipless mover has a single legal action; the honest policy returns it
    # directly (and would work even with an evaluator that must not be called).
    _, info = _midgame_info()
    info = info.__class__(
        chips=(0,) + info.chips[1:],  # mover has no chips => forced take
        cards=info.cards,
        active=info.active,
        pot=info.pot,
        to_move=0,
        deck=info.deck,
        n_removed=info.n_removed,
    )

    def boom(_st):
        raise AssertionError("evaluator must not run for a forced move")

    assert determinized_action(info, boom, n_worlds=99) == "take"


def test_head_to_head_hidden_structure():
    net = ValueNet(3, hidden=8, seed=3)
    from nothanks.train import head_to_head_hidden

    res = head_to_head_hidden(net, n_games=8, n_worlds=10)
    assert set(res) == {"vnet_mean", "heuristic_mean", "win_rate", "parity"}
    assert 0.0 <= res["win_rate"] <= 1.0
    assert abs(res["parity"] - 1 / 3) < 1e-9


def test_no_removal_reduces_to_godview_greedy():
    # With nothing removed there is exactly one consistent world (the true pile),
    # so the honest policy must reproduce the god-view greedy move at every state.
    from nothanks.valuefn import greedy_action

    net = ValueNet(3, hidden=16, seed=4)
    rng = random.Random(0)
    s = new_game(3, n_removed=0, rng=rng)  # full 33-card deck: nothing hidden
    evaluator = lambda st: evaluate_v(st, net)  # noqa: E731
    checked = 0
    while not is_terminal(s):
        info = info_from_state(s, n_removed=0)
        # n_worlds=1 => the single true world, no averaging that could flip a tie.
        honest = determinized_action(info, evaluator, n_worlds=1, rng=random.Random(0))
        assert honest == greedy_action(s, net)
        checked += 1
        s = step(s, honest, rng)
    assert checked > 5
