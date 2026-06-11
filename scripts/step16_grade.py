"""Roadmap step 16: grade the search-curriculum repair of the info net.

Acceptance criteria (CLAUDE.md):
  1. the one-ply eval flips to *take* on the step-12 smoke position (also a
     regression test in tests/test_beliefnet.py);
  2. ``head_to_head_info`` does not regress vs the old net's ~51.7 ± 0.4;
  3. seat-balanced ``arena.bot_vs_bot`` (new vs old) does not regress — both
     as one-ply greedy bots and as IS-MCTS leaves;
  4. (separately, slow) rerunning scripts/step11_greedy_info.py with the new
     net shows the seat-0 learned-BR gain dropping from +4.45 ± 1.07.

Run from the repo root: uv run python scripts/step16_grade.py
"""

import time

from nothanks.arena import bot_vs_bot, greedy_info_bot, ismcts_bot
from nothanks.beliefnet import evaluate_info, head_to_head_info
from nothanks.engine import full_deck
from nothanks.imperfect import InfoSet
from nothanks.valuefn import ValueNet

SMOKE = InfoSet(
    chips=(9, 11, 10),
    cards=(frozenset({3, 4, 5, 22}), frozenset({17}), frozenset()),
    active=26, pot=3, to_move=0,
    deck=frozenset(full_deck()), n_removed=9,
)

old = ValueNet.load("models/info_net_3p.npz")
new = ValueNet.load("models/info_net_3p_v2.npz")

print("1) smoke position (card 26, pot 3, mover holds 3-5+22) — ground truth: take")
for name, net in (("old", old), ("new", new)):
    ev = evaluate_info(SMOKE, net)
    print(f"   {name}: best={ev['best_action']:4s}  "
          f"take {ev['mover_ev']['take']:+.2f}  pass {ev['mover_ev']['pass']:+.2f}")

print("\n2) head_to_head_info, 2000 games (old net's historical: ~51.7 ± 0.4)")
for name, net in (("old", old), ("new", new)):
    t0 = time.time()
    res = head_to_head_info(net, n_games=2000)
    print(f"   {name}: net {res['vnet_mean']:.2f} ± {res['vnet_stderr']:.2f}  "
          f"heuristic {res['heuristic_mean']:.2f}  win/tie {res['win_rate']:.1%} "
          f"(parity {res['parity']:.0%})  ({time.time() - t0:.0f}s)")

print("\n3a) arena: greedy(new) vs greedy(old), seat-balanced 600 games/seat-config")
t0 = time.time()
res = bot_vs_bot(greedy_info_bot(new), greedy_info_bot(old), n_games=600)
print(f"   new {res['a_mean']:.2f} ± {res['a_stderr']:.2f}  old {res['b_mean']:.2f}  "
      f"new win/tie {res['a_win_rate']:.1%} (parity {res['parity']:.0%}, "
      f"{res['games']} games, {time.time() - t0:.0f}s)")

print("\n3b) arena: IS-MCTS(new leaf) vs IS-MCTS(old leaf), 200 iters, 150 games/seat-config")
t0 = time.time()
res = bot_vs_bot(ismcts_bot(new, n_iter=200), ismcts_bot(old, n_iter=200), n_games=150)
print(f"   new {res['a_mean']:.2f} ± {res['a_stderr']:.2f}  old {res['b_mean']:.2f}  "
      f"new win/tie {res['a_win_rate']:.1%} (parity {res['parity']:.0%}, "
      f"{res['games']} games, {time.time() - t0:.0f}s)")
