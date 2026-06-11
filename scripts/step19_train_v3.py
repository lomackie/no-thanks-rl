"""Roadmap step 19: repair the cheap-anchor-take bias (models/info_net_3p_v3.npz).

The bug (found by a user in real play, adjudicated by paired all-IS-MCTS
playouts, scripts/adjudicate_opening3.py): the v2 net undervalues states where
the mover owns a cheap run anchor — e.g. taking the opening 3 for 2 chips —
and because every leaf of the take subtree is such a state, the net-leaf
search inherits the bias at *any* iteration budget (still preferring pass at
20k iters). A second instance (run-connecting takes, 5p) was shallow enough
for 2000 search iterations to fix; this one needed training data.

The shipped recipe is step 16's plus **take-biased deviations**
(``take_dev_frac 0.5`` of the non-uniform-deviation games): force one ``take``
at a random cheap-take opportunity (score_delta − pot ≤ 2), continue
on-policy, train on the post-deviation suffix only — competent continuations
from anchor-owning states, value scale untouched (step 16's calibration
lesson).

A first attempt also added **playout-leaf search games** (``psearch_frac
0 → 0.15``); it fixed the opening-3 too but reinjected the heuristic
rollout's biases (step 12's finding) and lost the leaf-gate arena, so it was
dropped. The shipped net loses the *one-ply* arena to v2 (64.2 vs 59.2) but
wins the **leaf gate** decisively (IS-MCTS 200 iters: 54.6 vs 60.0, win/tie
41.3% vs 33% parity) — and per step 18, the leaf gate is the one that matters
for the deployed bot, which always searches.

Run from the repo root (~20 min on 6 cores):
    uv run python scripts/step19_train_v3.py
"""

import time

from nothanks.beliefnet import train_info


def main() -> None:
    t0 = time.time()
    net = train_info(
        n_players=3,
        iterations=80,
        games_per_iter=80,
        eps_start=0.3,
        eps_end=0.05,
        heur_frac_start=1.0,     # exactly step 16's bands
        heur_frac_end=0.25,
        search_frac_start=0.0,
        search_frac_end=0.4,
        search_iters=200,
        search_c=30.0,
        deviation_frac=0.5,      # exactly step 16's uniform deviations
        deviation_horizon=30,
        take_dev_frac=0.5,       # new: forced cheap-anchor takes on the rest
        take_dev_horizon=8,
        take_dev_margin=2,
        target_refresh=5,
        hidden=64,
        n_removed=9,
        n_jobs=6,
        seed=0,
        log=True,
    )
    net.save("models/info_net_3p_v3.npz")
    print(f"saved models/info_net_3p_v3.npz  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":  # required: the n_jobs pool respawns __main__ on macOS
    main()
