"""Print an engine-style exact evaluation of a tiny opening.

Run with ``just demo`` (or ``uv run python -m nothanks.demo``).
"""

from __future__ import annotations

from .engine import initial_state
from .solver import evaluate


def main() -> None:
    # 3 players, reduced deck 3..9, 3 chips each — small enough to solve exactly.
    deck = [3, 4, 5, 6, 7, 8, 9]
    s = initial_state(3, deck, start_chips=3)
    memo: dict = {}
    info = evaluate(s, memo)

    print(f"deck {deck}, 3 players, 3 chips each")
    print(f"card {s.active} face-up (pot {s.pot}), player {s.to_move} to move\n")
    for action, ev in info["mover_ev"].items():
        vec = tuple(round(x, 2) for x in info["actions"][action])
        print(f"  {action:5s} -> P{s.to_move} expected {ev:+.3f}   full vector {vec}")
    print(f"\n  best: {info['best_action']}")
    print(f"  (states solved: {len(memo):,})")


if __name__ == "__main__":
    main()
