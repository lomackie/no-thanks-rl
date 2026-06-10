"""Roadmap step 11b: approximate exploitability of the IS-MCTS bot (small budget).

Candidate: the deterministic IS-MCTS policy (value-net leaf, n_iter=200, c=30 —
the head_to_head_ismcts grading budget). Every candidate move during BR training
runs a full search, so the budget here is deliberately small: read the result as
an even looser lower bound than step 11a's.

Run from the repo root: uv run python scripts/step11_ismcts.py
"""

import time

from nothanks.approx_br import approx_exploitability
from nothanks.ismcts import make_ismcts_policy, make_value_leaf
from nothanks.valuefn import ValueNet

net = ValueNet.load("models/info_net_3p.npz")
policy = make_ismcts_policy(n_iter=200, evaluator=make_value_leaf(net),
                            c=30.0, seed=0)

t0 = time.time()
res = approx_exploitability(policy, n_players=3, n_removed=9,
                            n_games=300, seed=0, log=True,
                            iterations=15, games_per_iter=40)
elapsed = time.time() - t0

print("\n=== step 11b: IS-MCTS policy (net leaf, n_iter=200, c=30), 9-removed ===")
for hero in range(3):
    print(f"  hero {hero}: gain {res['gain'][hero]:+.3f} ± {res['stderr'][hero]:.3f}")
print(f"  total {res['total']:+.3f}   max {res['max']:+.3f}")
print(f"  ({elapsed:.0f}s)")
