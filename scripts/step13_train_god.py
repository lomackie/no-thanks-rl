"""Roadmap step 13 prerequisite: train and save the god-view self-play net.

Same recipe as the stronger net in train._demo (heuristic warmup annealed to
self-play, target net refreshed every 5 iters). Saved so the arena can wrap it
in PIMC (train.pimc_policy) without retraining.

Run from the repo root: uv run python scripts/step13_train_god.py
"""

import time

from nothanks.train import head_to_head, train

t0 = time.time()
net = train(n_players=3, heur_frac_start=1.0, heur_frac_end=0.25,
            target_refresh=5, seed=0, log=True)
net.save("models/god_net_3p.npz")
print(f"saved models/god_net_3p.npz  ({time.time() - t0:.0f}s)")

res = head_to_head(net, n_games=1000)
print(f"god-view sanity (peeking grader): net {res['vnet_mean']:.2f} vs "
      f"heuristic {res['heuristic_mean']:.2f}, win/tie {res['win_rate']:.1%}")
