"""Roadmap step 19, part 2: gate the retrained 4p/5p nets.

Per step 18, the gate that matters for the deployed bot is the leaf arena
(200-iter IS-MCTS, new leaf vs old leaf); the one-ply arena is reported for
context (step 19's 3p result showed the better leaf can be the worse one-ply
player). The 5p run-connecting-8 smoke is a regression test in
tests/test_beliefnet.py.

Run from the repo root: uv run python scripts/step19_grade_45p.py
"""

import time

from nothanks.arena import bot_vs_bot, greedy_info_bot, ismcts_bot
from nothanks.valuefn import ValueNet


def grade(n_players: int) -> None:
    old = ValueNet.load(f"models/info_net_{n_players}p.npz")
    new = ValueNet.load(f"models/info_net_{n_players}p_v2.npz")

    print(f"\n=== {n_players}p ===")
    print("one-ply arena: greedy(new) vs greedy(old), 300 games/seat-config")
    t0 = time.time()
    res = bot_vs_bot(greedy_info_bot(new), greedy_info_bot(old),
                     n_games=300, n_players=n_players)
    print(f"   new {res['a_mean']:.2f} ± {res['a_stderr']:.2f}  old {res['b_mean']:.2f}  "
          f"new win/tie {res['a_win_rate']:.1%} (parity {res['parity']:.0%}, "
          f"{res['games']} games, {time.time() - t0:.0f}s)")

    print("leaf gate: IS-MCTS(new leaf) vs IS-MCTS(old leaf), 200 iters, 100 games/seat-config")
    t0 = time.time()
    res = bot_vs_bot(ismcts_bot(new, n_iter=200), ismcts_bot(old, n_iter=200),
                     n_games=100, n_players=n_players)
    print(f"   new {res['a_mean']:.2f} ± {res['a_stderr']:.2f}  old {res['b_mean']:.2f}  "
          f"new win/tie {res['a_win_rate']:.1%} (parity {res['parity']:.0%}, "
          f"{res['games']} games, {time.time() - t0:.0f}s)")


def main() -> None:
    for n in (4, 5):
        grade(n)


if __name__ == "__main__":
    main()
