"""Print a Monte-Carlo engine-style evaluation, with a tiny-game sanity check.

Run with ``just mc-demo`` (or ``uv run python -m nothanks.mc_demo``).
"""

from __future__ import annotations

import random

from .engine import initial_state, new_game
from .montecarlo import evaluate_mc, exact_action_values, make_policy


def _print_eval(info: dict, *, stderr: bool) -> None:
    p = info["to_move"]
    for action, ev in info["mover_ev"].items():
        vec = tuple(round(x, 2) for x in info["actions"][action])
        err = f"  +/- {info['stderr'][action]:.3f}" if stderr else ""
        print(f"  {action:5s} -> P{p} EV {ev:+.3f}{err}   vec {vec}")
    print(f"  best: {info['best_action']}")


def main() -> None:
    policy = make_policy(0)

    # 1) Validation: on a tiny game the sampler should match the exact policy EV.
    deck = [3, 4, 5, 6, 7]
    s = initial_state(3, deck, start_chips=2)
    print(f"tiny game {deck}, 3 players, 2 chips — heuristic policy")
    print("exact policy EV:")
    _print_eval(exact_action_values(s, policy), stderr=False)
    print(f"monte-carlo (10k rollouts):")
    _print_eval(evaluate_mc(s, 10_000, policy, random.Random(0)), stderr=True)

    # 2) The real target: a full 24-card removal deck, far beyond the solver.
    s = new_game(4, n_removed=9, rng=random.Random(42))
    print(f"\nfull game, 4 players, card {s.active} face-up (pot {s.pot}), "
          f"{len(s.remaining)} cards left")
    _print_eval(evaluate_mc(s, 5_000, policy, random.Random(0)), stderr=True)


if __name__ == "__main__":
    main()
