"""Roadmap step 12: adjudicate the PIMC-vs-net disagreement.

On the smoke-test position (card 26, pot 3, mover holds 3-5+22) PIMC-over-
rollouts says *take* while both net-based methods say *pass* by ~5 points.
Two experiments settle which evaluator to trust:

Part A — the position class on a tiny deck, where ``belief.solve`` is exact.
    Mover holds a low card, the active card is gapped away from it, and the gap
    cards are unseen (so possibly removed). For each position we compare the
    exact belief-optimal action/Q-values against an **exact PIMC**: the solver
    run in every consistent world, averaged with equal weight (worlds are
    enumerated, not sampled, so there is zero noise — any flip versus the
    belief optimum is pure strategy fusion). PIMC-over-rollouts (the CLI's
    ``pimc`` method) is shown alongside. ``ev_cost`` is the exact price, in
    expected points, of following PIMC's choice instead of the optimum.

Part B — the actual smoke position, settled by play.
    Paired playouts: force take vs force pass at the root, then *all* seats
    play the strongest honest bot (IS-MCTS, info-net leaf). The arm with the
    lower seat-0 mean is the right move; read the gap against the paired
    stderr.

Run from the repo root: uv run python scripts/step12_adjudicate.py
"""

import math
import random
import time
from itertools import combinations

from nothanks import solver
from nothanks.belief import (
    apply_pass,
    final_scores,
    is_terminal,
    solve,
    take_outcomes,
)
from nothanks.beliefnet import belief_step
from nothanks.engine import State, full_deck
from nothanks.imperfect import InfoSet, pile_remaining, unseen
from nothanks.ismcts import ISMCTSBot, make_value_leaf
from nothanks.montecarlo import evaluate_mc
from nothanks.valuefn import ValueNet


# --------------------------------------------------------------------------- #
# Part A: exact adjudication of the position class on a tiny deck
# --------------------------------------------------------------------------- #

def exact_q(info: InfoSet, memo: dict) -> dict[str, float]:
    """Mover's exact Q-value per action under belief-optimal continuation."""
    p = info.to_move
    q = {}
    q["pass"] = solve(apply_pass(info), memo)[0][p]
    q["take"] = sum(prob * solve(nxt, memo)[0][p] for prob, nxt in take_outcomes(info))
    return q


def exact_pimc_q(info: InfoSet, memo: dict) -> dict[str, float]:
    """Mover's PIMC Q-values with the exact solver leaf, worlds *enumerated*.

    Every ``pile_remaining``-subset of the unseen cards is an equiprobable
    world; the per-world evaluation is exact, so the only error left in the
    average is PIMC's own (strategy fusion / non-locality).
    """
    k = pile_remaining(info)
    cands = sorted(unseen(info))
    worlds = list(combinations(cands, k))
    p = info.to_move
    acc = {"take": 0.0, "pass": 0.0}
    for pile in worlds:
        s = State(chips=info.chips, cards=info.cards, active=info.active,
                  pot=info.pot, to_move=info.to_move, remaining=frozenset(pile))
        ev = solver.evaluate(s, memo)
        for a in acc:
            acc[a] += ev["mover_ev"][a]
    return {a: v / len(worlds) for a, v in acc.items()}


def rollout_pimc_action(info: InfoSet, seed: int = 0) -> str:
    """The CLI's ``pimc`` baseline: MC rollouts per sampled world."""
    from nothanks.imperfect import evaluate_determinized

    rng = random.Random(seed)
    per_world = lambda s: evaluate_mc(  # noqa: E731
        s, n_rollouts=200, rng=random.Random(rng.randrange(1 << 62)))
    return evaluate_determinized(info, per_world, n_worlds=200, rng=rng)["best_action"]


def part_a() -> None:
    deck = frozenset(range(3, 11))  # 8 cards, n_removed=2: belief.solve is exact
    positions = []
    for hold, active in (((3,), 7), ((3, 4), 8)):
        for pot in range(4):
            positions.append((hold, active, pot))

    print("Part A — tiny-deck position class (deck 3..10, 2 removed, 3 players)")
    print("mover holds a low card; active is gapped away; gap cards unseen\n")
    print(f"{'hold':>8} {'act':>3} {'pot':>3} | {'exact':>5} {'Q(take)':>8} {'Q(pass)':>8}"
          f" | {'PIMCx':>5} {'evcost':>6} | {'PIMCroll':>8}")

    n_disagree = 0
    total_cost = 0.0
    for hold, active, pot in positions:
        info = InfoSet(
            chips=(4, 4, 4),
            cards=(frozenset(hold), frozenset({10}), frozenset()),
            active=active, pot=pot, to_move=0, deck=deck, n_removed=2,
        )
        memo: dict = {}
        q = exact_q(info, memo)
        exact_best = min(q, key=q.get)
        pq = exact_pimc_q(info, {})
        pimc_best = min(pq, key=pq.get)
        ev_cost = q[pimc_best] - q[exact_best]
        roll_best = rollout_pimc_action(info)
        flag = "  <-- fusion flip" if pimc_best != exact_best and ev_cost > 1e-9 else ""
        if pimc_best != exact_best:
            n_disagree += 1
            total_cost += ev_cost
        print(f"{str(set(hold)):>8} {active:>3} {pot:>3} | {exact_best:>5}"
              f" {q['take']:+8.3f} {q['pass']:+8.3f} | {pimc_best:>5} {ev_cost:6.3f}"
              f" | {roll_best:>8}{flag}")
    print(f"\nexact-leaf PIMC flips {n_disagree}/{len(positions)} positions, "
          f"total exact EV cost {total_cost:.3f}")


# --------------------------------------------------------------------------- #
# Part B: the real smoke position, settled by strong-bot playouts
# --------------------------------------------------------------------------- #

SMOKE = InfoSet(
    chips=(9, 11, 10),
    cards=(frozenset({3, 4, 5, 22}), frozenset({17}), frozenset()),
    active=26, pot=3, to_move=0,
    deck=frozenset(full_deck()), n_removed=9,
)


def playout_from(info: InfoSet, first_action: str, net: ValueNet, seed: int) -> float:
    """Force ``first_action`` at the root, then all seats play IS-MCTS; seat-0 score."""
    rng = random.Random(seed)
    bots = [ISMCTSBot(n_iter=200, evaluator=make_value_leaf(net), c=30.0, seed=seed + 7 * i)
            for i in range(info.n_players)]
    info = belief_step(info, first_action, rng)
    while not is_terminal(info):
        info = belief_step(info, bots[info.to_move].act(info), rng)
    return final_scores(info)[0]


def part_b(n_games: int = 150) -> None:
    net = ValueNet.load("models/info_net_3p.npz")
    print("\nPart B — smoke position (card 26, pot 3, mover holds 3-5+22),")
    print(f"both arms played out by IS-MCTS all seats, {n_games} paired games/arm\n")
    t0 = time.time()
    diffs = []
    means = {"take": 0.0, "pass": 0.0}
    for i in range(n_games):
        vt = playout_from(SMOKE, "take", net, seed=1000 + i)
        vp = playout_from(SMOKE, "pass", net, seed=1000 + i)
        means["take"] += vt
        means["pass"] += vp
        diffs.append(vt - vp)
    for a in means:
        means[a] /= n_games
    d_mean = sum(diffs) / n_games
    d_var = sum((d - d_mean) ** 2 for d in diffs) / n_games
    d_se = math.sqrt(d_var / n_games)
    print(f"  seat-0 EV: take {means['take']:+.2f}   pass {means['pass']:+.2f}")
    print(f"  take - pass = {d_mean:+.2f} ± {d_se:.2f}  (positive favours pass)")
    print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    part_a()
    part_b()
