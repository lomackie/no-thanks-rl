"""Re-adjudicate the step-16 smoke position under newer continuation bots.

Step 12's verdict (take by 5.9 ± 1.3, 500 paired playouts) used the *original*
net as every bot's leaf. If the v3 net genuinely evaluates gapped high cards
better, the verdict could shift under stronger continuation play — so replay
the paired-playout adjudication with the v2 and v3 nets as the bots' leaves
and see whether "take" survives a change of opponent model.
"""
from __future__ import annotations

import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from nothanks.belief import final_scores, is_terminal
from nothanks.beliefnet import belief_step
from nothanks.engine import full_deck
from nothanks.imperfect import InfoSet
from nothanks.ismcts import ISMCTSBot, make_value_leaf
from nothanks.valuefn import ValueNet

SMOKE16 = InfoSet(
    chips=(9, 11, 10),
    cards=(frozenset({3, 4, 5, 22}), frozenset({17}), frozenset()),
    active=26, pot=3, to_move=0,
    deck=frozenset(full_deck()), n_removed=9,
)


def playout_pair(args) -> tuple[float, float]:
    net_path, seed = args
    net = ValueNet.load(net_path)
    out = []
    for first_action in ("take", "pass"):
        rng = random.Random(seed)
        bots = [ISMCTSBot(n_iter=200, evaluator=make_value_leaf(net), c=30.0,
                          seed=seed + 7 * i)
                for i in range(SMOKE16.n_players)]
        info = belief_step(SMOKE16, first_action, rng)
        while not is_terminal(info):
            info = belief_step(info, bots[info.to_move].act(info), rng)
        out.append(final_scores(info)[0])
    return out[0], out[1]


def main(n_games: int = 200, workers: int = 3) -> None:
    for net_path in ("models/info_net_3p_v2.npz", "models/info_net_3p_v3.npz"):
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(playout_pair,
                                  [(net_path, 1000 + i) for i in range(n_games)]))
        diffs = [vt - vp for vt, vp in pairs]
        mt = sum(vt for vt, _ in pairs) / n_games
        mp = sum(vp for _, vp in pairs) / n_games
        d_mean = sum(diffs) / n_games
        d_var = sum((d - d_mean) ** 2 for d in diffs) / n_games
        d_se = math.sqrt(d_var / n_games)
        print(f"{net_path}: take {mt:+.2f}  pass {mp:+.2f}  "
              f"take−pass {d_mean:+.2f} ± {d_se:.2f}  "
              f"({n_games} paired games, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200,
         int(sys.argv[2]) if len(sys.argv) > 2 else 3)
