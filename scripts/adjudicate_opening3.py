"""Adjudicate the opening position: card 3 face-up, pot 2, mover has 11 chips.

Paired playouts: force take vs force pass at the root, then all seats play
IS-MCTS (200 iters, v2 net leaf); lower seat-0 mean is the right move.
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

POSITION = InfoSet(
    chips=(11, 10, 10),
    cards=(frozenset(), frozenset(), frozenset()),
    active=3, pot=2, to_move=0,
    deck=frozenset(full_deck()), n_removed=9,
)

NET_PATH = "models/info_net_3p_v2.npz"  # overridden by argv[2]


def playout_pair(args) -> tuple[float, float]:
    net_path, seed = args
    net = ValueNet.load(net_path)
    out = []
    for first_action in ("take", "pass"):
        rng = random.Random(seed)
        bots = [ISMCTSBot(n_iter=200, evaluator=make_value_leaf(net), c=30.0,
                          seed=seed + 7 * i)
                for i in range(POSITION.n_players)]
        info = belief_step(POSITION, first_action, rng)
        while not is_terminal(info):
            info = belief_step(info, bots[info.to_move].act(info), rng)
        out.append(final_scores(info)[0])
    return out[0], out[1]


def main(n_games: int = 300, net_path: str = NET_PATH) -> None:
    t0 = time.time()
    print(f"continuation bots' leaf: {net_path}")
    with ProcessPoolExecutor(max_workers=6) as pool:
        pairs = list(pool.map(playout_pair,
                              [(net_path, 1000 + i) for i in range(n_games)]))
    diffs = [vt - vp for vt, vp in pairs]
    mt = sum(vt for vt, _ in pairs) / n_games
    mp = sum(vp for _, vp in pairs) / n_games
    d_mean = sum(diffs) / n_games
    d_var = sum((d - d_mean) ** 2 for d in diffs) / n_games
    d_se = math.sqrt(d_var / n_games)
    print(f"seat-0 EV over {n_games} paired games: take {mt:+.2f}   pass {mp:+.2f}")
    print(f"take - pass = {d_mean:+.2f} ± {d_se:.2f}  (positive favours pass)")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300,
         sys.argv[2] if len(sys.argv) > 2 else NET_PATH)
