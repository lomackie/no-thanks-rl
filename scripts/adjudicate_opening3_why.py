"""Instrumented version of adjudicate_opening3: where do the 8 points come from?

For each paired game, track who eventually takes the 3 and at what pot,
plus seat 0's final chips and card points, in both arms.
"""
from __future__ import annotations

import random
import sys
import time
from collections import Counter
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

NET_PATH = "models/info_net_3p_v2.npz"


def play_arm(first_action: str, seed: int) -> dict:
    net = ValueNet.load(NET_PATH)
    rng = random.Random(seed)
    bots = [ISMCTSBot(n_iter=200, evaluator=make_value_leaf(net), c=30.0,
                      seed=seed + 7 * i)
            for i in range(POSITION.n_players)]
    info = POSITION
    taker3, pot3 = None, None
    action = first_action
    while True:
        if info.active == 3 and action == "take":
            taker3, pot3 = info.to_move, info.pot
        info = belief_step(info, action, rng)
        if is_terminal(info):
            break
        action = bots[info.to_move].act(info)
    return {
        "score0": final_scores(info)[0],
        "chips0": info.chips[0],
        "taker3": taker3,
        "pot3": pot3,
    }


def playout_pair(seed: int) -> tuple[dict, dict]:
    return play_arm("take", seed), play_arm("pass", seed)


def main(n_games: int = 300) -> None:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as pool:
        pairs = list(pool.map(playout_pair, range(1000, 1000 + n_games)))
    for name, arm in (("take", [t for t, _ in pairs]), ("pass", [p for _, p in pairs])):
        n = len(arm)
        score = sum(g["score0"] for g in arm) / n
        chips = sum(g["chips0"] for g in arm) / n
        cardpts = score + chips  # score = cards - chips
        takers = Counter(g["taker3"] for g in arm)
        pots = [g["pot3"] for g in arm if g["pot3"] is not None]
        mean_pot = sum(pots) / len(pots) if pots else float("nan")
        print(f"{name}-arm: seat0 score {score:+.2f} = cards {cardpts:.2f} - chips {chips:.2f}")
        print(f"  who took the 3: {dict(sorted(takers.items(), key=lambda kv: str(kv[0])))}"
              f"   mean pot when taken: {mean_pot:.2f}")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
