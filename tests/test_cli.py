"""The position-input CLI (nothanks.cli).

Parsing must round-trip the documented format and reject inconsistent positions
with readable errors; eval must run end-to-end on a real position for each
method and print an EV per legal action; train must save a loadable info net.
"""

import random

import pytest

from nothanks.beliefnet import make_info_net
from nothanks.cli import build_info, build_parser, main, parse_card_list, parse_cards
from nothanks.imperfect import pile_remaining


def _eval_args(**over):
    base = dict(chips="9,11,10", cards="3-5,22;17;", active=26, pot=3,
                to_move=0, n_removed=9)
    base.update(over)
    args = [
        "eval",
        "--chips", base["chips"],
        "--cards", base["cards"],
        "--active", str(base["active"]),
        "--pot", str(base["pot"]),
        "--to-move", str(base["to_move"]),
        "--n-removed", str(base["n_removed"]),
    ]
    return args


def test_parse_card_list_ranges_and_empty():
    assert parse_card_list("3-5,22") == frozenset({3, 4, 5, 22})
    assert parse_card_list("") == frozenset()
    assert parse_card_list(" 7 , 9-10 ") == frozenset({7, 9, 10})
    with pytest.raises(ValueError):
        parse_card_list("9-7")


def test_parse_cards_per_seat_groups():
    assert parse_cards("3-5,22;17;") == (
        frozenset({3, 4, 5, 22}), frozenset({17}), frozenset())


def test_build_info_round_trips_the_position():
    args = build_parser().parse_args(_eval_args())
    info = build_info(args)
    assert info.chips == (9, 11, 10)
    assert info.cards[0] == frozenset({3, 4, 5, 22})
    assert info.active == 26
    assert info.pot == 3
    assert pile_remaining(info) == (33 - 9) - 6  # 6 cards seen


@pytest.mark.parametrize("bad, msg", [
    (dict(cards="3-5;3;"), "more than one seat"),
    (dict(active=3), "already captured"),
    (dict(active=40), "not in the deck"),
    (dict(to_move=5), "--to-move"),
])
def test_build_info_rejects_inconsistent_positions(bad, msg):
    args = build_parser().parse_args(_eval_args(**bad))
    with pytest.raises(ValueError, match=msg):
        build_info(args)


def test_eval_ismcts_runs_and_prints_ev(capsys):
    main(_eval_args() + ["--method", "ismcts", "--n-iter", "50", "--seed", "0"])
    out = capsys.readouterr().out
    assert "take" in out and "pass" in out and "best:" in out


def test_eval_pimc_runs(capsys):
    main(_eval_args() + ["--method", "pimc", "--n-worlds", "4",
                         "--rollouts", "8", "--seed", "0"])
    out = capsys.readouterr().out
    assert "best:" in out and "worlds" in out


def test_eval_net_method_with_saved_info_net(tmp_path, capsys):
    net = make_info_net(3, hidden=8, seed=0)
    path = tmp_path / "info_net.npz"
    net.save(path)
    main(_eval_args() + ["--method", "net", "--net", str(path)])
    out = capsys.readouterr().out
    assert "best:" in out


def test_eval_net_method_rejects_godview_net(tmp_path):
    from nothanks.valuefn import ValueNet

    net = ValueNet(3, hidden=8, seed=0)  # god-view dims
    path = tmp_path / "god.npz"
    net.save(path)
    with pytest.raises(SystemExit, match="not an info-set net"):
        main(_eval_args() + ["--method", "net", "--net", str(path)])


def test_forced_position_short_circuits(capsys):
    main(_eval_args(chips="0,11,10") + ["--method", "ismcts"])
    out = capsys.readouterr().out
    assert "forced: take" in out


def test_train_saves_loadable_net(tmp_path, capsys):
    out_path = tmp_path / "net.npz"
    main(["train", "--out", str(out_path), "--iterations", "1",
          "--games-per-iter", "2", "--hidden", "8", "--grade", "0"])
    assert out_path.exists()
    from nothanks.features import info_feature_dim
    from nothanks.valuefn import ValueNet

    loaded = ValueNet.load(out_path)
    assert loaded.in_dim == info_feature_dim(loaded.n_players)


def test_format_cards_groups_runs():
    from nothanks.cli import format_cards

    assert format_cards(frozenset()) == "-"
    assert format_cards({3, 4, 5, 22}) == "3-5,22"
    assert format_cards({7}) == "7"
    assert format_cards({3, 5, 6, 9}) == "3,5-6,9"


def test_play_scripted_game_reaches_the_end(monkeypatch, capsys):
    # The human takes every card: taking retains the turn, so the game runs to
    # completion in 24 prompts regardless of what the AI would do. Tiny search
    # budget (heuristic-playout leaf) keeps this fast.
    monkeypatch.setattr("builtins.input", lambda _prompt="": "t")
    main(["play", "--net", "", "--n-iter", "8", "--seed", "0"])
    out = capsys.readouterr().out
    assert "you are P0" in out
    assert "game over" in out
    assert "winner" in out


def test_play_quits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "q")
    main(["play", "--net", "", "--n-iter", "8", "--seed", "0"])
    assert "quit." in capsys.readouterr().out
