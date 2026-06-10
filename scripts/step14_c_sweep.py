"""Roadmap step 14 (open caveat): firm up the IS-MCTS exploration constant.

The full-deck default c=30 was picked from one position's behaviour. Small
sweep over c x n_iter, graded by head_to_head_ismcts (vs heuristic, value-net
leaf) on paired deals (shared seed across configs), so the columns are
comparable game-for-game.

Run from the repo root: uv run python scripts/step14_c_sweep.py
"""

import time

from nothanks.ismcts import make_value_leaf
from nothanks.train import head_to_head_ismcts
from nothanks.valuefn import ValueNet

net = ValueNet.load("models/info_net_3p.npz")
leaf = make_value_leaf(net)

N_GAMES = 80
print(f"{'c':>5} {'n_iter':>6} | {'bot':>7} {'+/-':>5} {'heur':>7} {'win/tie':>8}")
for n_iter in (200, 400):
    for c in (10.0, 30.0, 60.0):
        t0 = time.time()
        res = head_to_head_ismcts(n_games=N_GAMES, n_iter=n_iter,
                                  evaluator=leaf, c=c, seed=20_000)
        print(f"{c:5.0f} {n_iter:6d} | {res['bot_mean']:7.2f} "
              f"{res['bot_stderr']:5.2f} {res['heuristic_mean']:7.2f} "
              f"{res['win_rate']:8.1%}  ({time.time() - t0:.0f}s)", flush=True)
