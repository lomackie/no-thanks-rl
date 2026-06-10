"""Roadmap step 13: fair bot-vs-bot grading of the three honest bots.

All previous full-game numbers (51.7 / 49.7 / 40.4) were vs-the-heuristic —
the exploitation-prone grader. Here the deployable bots play *each other*,
seat-balanced (nothanks.arena.bot_vs_bot), to establish the real ordering:

  info  — one-ply greedy on the info-set net (instant)
  pimc  — god-view net made honest by PIMC at play time (n_worlds=80)
  mcts  — IS-MCTS, info-net leaf, n_iter=200, c=30 (the deployable searcher)

Run from the repo root: uv run python scripts/step13_arena.py
"""

import time

from nothanks.arena import bot_vs_bot, greedy_info_bot, ismcts_bot, pimc_god_bot
from nothanks.valuefn import ValueNet

info_net = ValueNet.load("models/info_net_3p.npz")
god_net = ValueNet.load("models/god_net_3p.npz")

bots = {
    "info": greedy_info_bot(info_net),
    "pimc": pimc_god_bot(god_net, n_worlds=80),
    "mcts": ismcts_bot(info_net, n_iter=200, c=30.0),
}

N_GAMES = 200  # per seat assignment; x3 seats = 600 games per pairing

for a, b in (("mcts", "info"), ("mcts", "pimc"), ("info", "pimc")):
    t0 = time.time()
    res = bot_vs_bot(bots[a], bots[b], n_games=N_GAMES)
    print(f"{a} vs {b}: {a} {res['a_mean']:.2f} ± {res['a_stderr']:.2f}  "
          f"{b} {res['b_mean']:.2f}   {a} win/tie {res['a_win_rate']:.1%} "
          f"(parity {res['parity']:.0%}, {res['games']} games, "
          f"{time.time() - t0:.0f}s)", flush=True)
