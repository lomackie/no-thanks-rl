from nothanks.engine import (
    initial_state,
    legal_actions,
    apply_pass,
    take_outcomes,
    final_scores,
    score_cards,
    score_delta,
)
from nothanks.solver import solve, evaluate


def test_score_runs():
    assert score_cards([]) == 0
    assert score_cards([5]) == 5
    assert score_cards([5, 6, 7]) == 5          # one run, only the 5 counts
    assert score_cards([5, 7]) == 12            # two singletons
    assert score_cards([5, 6, 7, 10, 11]) == 15  # runs 5.. and 10..


def test_score_delta_matches_bruteforce():
    base = {5, 6, 10}
    for c in range(3, 13):
        expected = score_cards(base | {c}) - score_cards(base)
        assert score_delta(base, c) == expected, c


def test_pot_collected_on_take():
    s = initial_state(3, [3, 4, 5], start_chips=2)
    s = apply_pass(s)            # p0 passes, pot=1
    s = apply_pass(s)            # p1 passes, pot=2
    # p2 takes the 3 with 2 chips on it; any flipped next card is fine here.
    nxt = take_outcomes(s)[0][1]
    assert 3 in nxt.cards[2]
    assert nxt.chips[2] == 2 + 2  # kept 2, collected 2 from pot
    assert nxt.to_move == 2       # taker acts again


def test_solver_runs_on_tiny_game():
    # 3 players, deck 3..7, 2 chips each — small enough to solve exactly.
    s = initial_state(3, [3, 4, 5, 6, 7], start_chips=2)
    memo: dict = {}
    values = solve(s, memo)
    assert len(values) == 3
    # Final scores are zero-sum-ish only in totals of cards/chips; just sanity:
    info = evaluate(s, memo)
    assert info["best_action"] in ("take", "pass")
    # Total captured card points across players must equal a fixed amount in any
    # terminal line, so expected totals should be finite, well-defined floats.
    assert all(isinstance(v, float) for v in values)
