"""Roadmap step 11a: approximate exploitability of the deployable greedy
info-net policy on the standard 9-removed game.

Candidate: greedy one-ply lookahead on models/info_net_3p.npz (the honest,
deployable eval). Per-seat learned best-response gains are *lower bounds* on
true exploitability — large gain falsifies optimality, small gain is only
necessary evidence.

Run from the repo root: uv run python scripts/step11_greedy_info.py
"""

import time

from nothanks.approx_br import approx_exploitability
from nothanks.beliefnet import make_greedy_info_policy
from nothanks.valuefn import ValueNet

net = ValueNet.load("models/info_net_3p.npz")
policy = make_greedy_info_policy(net)

t0 = time.time()
res = approx_exploitability(policy, n_players=3, n_removed=9,
                            n_games=1000, seed=0, log=True)
elapsed = time.time() - t0

print("\n=== step 11a: greedy info-net policy, 9-removed full game ===")
for hero in range(3):
    print(f"  hero {hero}: gain {res['gain'][hero]:+.3f} ± {res['stderr'][hero]:.3f}")
print(f"  total {res['total']:+.3f}   max {res['max']:+.3f}")
print(f"  ({elapsed:.0f}s)")
