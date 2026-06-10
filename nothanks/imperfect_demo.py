"""Step 4 demo: hidden removed cards via determinization, plus exploitability.

Three parts:
  1. consistency — with nothing removed, determinized eval == direct exact eval;
  2. PIMC — a mid-game full standard position (9 cards genuinely hidden) analysed
     by determinized Monte-Carlo, with per-move EV ± across-world stderr;
  3. exploitability — on a tiny game, the optimal policy is unexploitable while
     the heuristic leaves a best-responder some gain.

Run with ``just imperfect`` (or ``python -m nothanks.imperfect_demo``).
"""

from __future__ import annotations

import random

from .engine import full_deck, initial_state, is_terminal, new_game, step
from .exploit import exploitability, optimal_policy
from .heuristic import heuristic_action
from .imperfect import (
    determinize,
    evaluate_determinized,
    info_from_state,
    pile_remaining,
    unseen,
)
from .montecarlo import evaluate_mc
from .solver import evaluate as solver_evaluate


def _demo_consistency() -> None:
    print("1. consistency check — nothing removed, so nothing is hidden")
    s = initial_state(3, [3, 4, 5, 6, 7], start_chips=2)
    deck = frozenset(s.remaining | {s.active})
    info = info_from_state(s, n_removed=0, deck=deck)

    direct = solver_evaluate(s)
    det = evaluate_determinized(info, solver_evaluate, n_worlds=5, rng=random.Random(0))
    direct_ev = {a: round(v, 3) for a, v in direct["mover_ev"].items()}
    det_ev = {a: round(v, 3) for a, v in det["mover_ev"].items()}
    print(f"   pile_remaining={pile_remaining(info)}  unseen={len(unseen(info))} (0 hidden)")
    print(f"   direct exact mover EV : {direct_ev}")
    print(f"   determinized mover EV : {det_ev}")
    print("   -> identical (the only world is the true pile)\n")


def _demo_pimc() -> None:
    print("2. PIMC — a full standard position with 9 cards genuinely hidden")
    # Play a few heuristic moves into a real 9-removed game to get a mid position.
    rng = random.Random(7)
    s = new_game(3, n_removed=9, rng=rng)
    for _ in range(6):
        if is_terminal(s):
            break
        s = step(s, heuristic_action(s, 0), rng)
    info = info_from_state(s, n_removed=9, deck=frozenset(full_deck()))

    print(f"   card {info.active} face-up (pot {info.pot}), P{info.to_move} to move,"
          f" chips {info.chips}")
    print(f"   {len(unseen(info))} unseen cards: {pile_remaining(info)} still in the pile,"
          f" {len(unseen(info)) - pile_remaining(info)} removed & hidden")

    evaluator = lambda st: evaluate_mc(st, n_rollouts=120, rng=rng)  # noqa: E731
    det = evaluate_determinized(info, evaluator, n_worlds=120, rng=rng)
    for a in det["mover_ev"]:
        print(f"   {a:5s} -> EV {det['mover_ev'][a]:+7.2f}  ± {det['stderr'][a]:.2f}"
              f"   vec {tuple(round(x,1) for x in det['actions'][a])}")
    print(f"   best (under belief): {det['best_action']}\n")


def _demo_exploitability() -> None:
    print("3. exploitability — best-response gain vs a fixed policy (tiny game)")
    s = initial_state(3, [3, 4, 5, 6, 7, 8], start_chips=3)

    heur = lambda st: heuristic_action(st, 0)  # noqa: E731
    exp_h = exploitability(s, heur)
    exp_o = exploitability(s, optimal_policy())

    print(f"   heuristic : base {tuple(round(x,2) for x in exp_h['base'])}"
          f"  gain/seat {tuple(round(x,2) for x in exp_h['gain'])}"
          f"  total {exp_h['total']:.2f}")
    print(f"   optimal   : base {tuple(round(x,2) for x in exp_o['base'])}"
          f"  gain/seat {tuple(round(x,2) for x in exp_o['gain'])}"
          f"  total {exp_o['total']:.2f}")
    print("   -> optimal is its own best response (≈0); the heuristic is exploitable")


def main() -> None:
    _demo_consistency()
    _demo_pimc()
    _demo_exploitability()


if __name__ == "__main__":
    main()
