"""Roadmap step 19: grade the cheap-anchor-take repair of the 3p info net.

Acceptance criteria:
  1. the opening-3 smoke position (3 face-up, pot 2, third seat to move —
     ground truth: take by 8.15 ± 1.76, scripts/adjudicate_opening3.py) flips
     to *take* under the net-leaf search at 2000 iterations (the v2 net leaf
     prefers pass even at 20k);
  2. the step-16 smoke position (gapped 26 for pot 3) stays flipped to take
     one-ply — no regression on the previous repair;
  3. seat-balanced ``arena.bot_vs_bot`` (v3 vs v2) does not regress — both as
     one-ply greedy bots and (the gate that matters for the deployed bot) as
     200-iter IS-MCTS leaves.

Run from the repo root: uv run python scripts/step19_grade.py [candidate.npz]
(default candidate: models/info_net_3p_v3.npz, baseline: the v2 net)
"""

import random
import sys
import time

from nothanks.arena import bot_vs_bot, greedy_info_bot, ismcts_bot
from nothanks.beliefnet import evaluate_info
from nothanks.engine import full_deck
from nothanks.imperfect import InfoSet
from nothanks.ismcts import ismcts_evaluate, make_value_leaf
from nothanks.valuefn import ValueNet

OPENING3 = InfoSet(
    chips=(11, 10, 10),
    cards=(frozenset(), frozenset(), frozenset()),
    active=3, pot=2, to_move=0,
    deck=frozenset(full_deck()), n_removed=9,
)

SMOKE16 = InfoSet(
    chips=(9, 11, 10),
    cards=(frozenset({3, 4, 5, 22}), frozenset({17}), frozenset()),
    active=26, pot=3, to_move=0,
    deck=frozenset(full_deck()), n_removed=9,
)


def main() -> None:
    candidate = sys.argv[1] if len(sys.argv) > 1 else "models/info_net_3p_v3.npz"
    old = ValueNet.load("models/info_net_3p_v2.npz")
    new = ValueNet.load(candidate)
    print(f"candidate: {candidate}\n")

    print("1) opening-3 smoke (3 face-up, pot 2) — ground truth: take (+8.15 ± 1.76)")
    print("   net-leaf search, 2000 iters:")
    for name, net in (("v2", old), ("v3", new)):
        ev = ismcts_evaluate(OPENING3, n_iter=2000, evaluator=make_value_leaf(net),
                             c=30.0, rng=random.Random(1))
        print(f"   {name}: best={ev['best_action']:4s}  "
              f"take {ev['mover_ev']['take']:+.2f}  pass {ev['mover_ev']['pass']:+.2f}  "
              f"visits {ev['visits']}")
    print("   one-ply:")
    for name, net in (("v2", old), ("v3", new)):
        ev = evaluate_info(OPENING3, net)
        print(f"   {name}: best={ev['best_action']:4s}  "
              f"take {ev['mover_ev']['take']:+.2f}  pass {ev['mover_ev']['pass']:+.2f}")

    print("\n2) step-16 smoke (gapped 26, pot 3) — must stay take one-ply")
    for name, net in (("v2", old), ("v3", new)):
        ev = evaluate_info(SMOKE16, net)
        print(f"   {name}: best={ev['best_action']:4s}  "
              f"take {ev['mover_ev']['take']:+.2f}  pass {ev['mover_ev']['pass']:+.2f}")

    print("\n3a) arena: greedy(v3) vs greedy(v2), seat-balanced 600 games/seat-config")
    t0 = time.time()
    res = bot_vs_bot(greedy_info_bot(new), greedy_info_bot(old), n_games=600)
    print(f"   v3 {res['a_mean']:.2f} ± {res['a_stderr']:.2f}  v2 {res['b_mean']:.2f}  "
          f"v3 win/tie {res['a_win_rate']:.1%} (parity {res['parity']:.0%}, "
          f"{res['games']} games, {time.time() - t0:.0f}s)")

    print("\n3b) arena: IS-MCTS(v3 leaf) vs IS-MCTS(v2 leaf), 200 iters, 150 games/seat-config")
    t0 = time.time()
    res = bot_vs_bot(ismcts_bot(new, n_iter=200), ismcts_bot(old, n_iter=200),
                     n_games=150)
    print(f"   v3 {res['a_mean']:.2f} ± {res['a_stderr']:.2f}  v2 {res['b_mean']:.2f}  "
          f"v3 win/tie {res['a_win_rate']:.1%} (parity {res['parity']:.0%}, "
          f"{res['games']} games, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
