"""Roadmap step 16: train the repaired 3p info net (models/info_net_3p_v2.npz).

Two mechanisms on top of the original curriculum, found over three attempts:

* the **expert-iteration leg** (``search_frac 0 → 0.4``): that fraction of
  games is played by IS-MCTS with the current net as leaf. At 0.2 with default
  exploration the smoke position did not flip (the searcher inherits the net's
  gapped-high-card bias, so search games still avoid those takes);
* **exploring deviations** (``deviation_frac 0.5``): one uniform-random action
  at a random decision, on-policy continuation, train on the post-deviation
  suffix only. This is the coverage fix that *keeps the value scale
  calibrated* — the in-between attempt used sustained ε (``eps_end 0.15``)
  instead, which flipped the smoke position but inflated all values by ~20
  points (TD(λ) is on-policy, so the net learns the ε-noisy policy's worse
  scores) and made the net a *worse* IS-MCTS leaf than the original (leaf
  values get compared against true terminal scores inside the tree).

This recipe flips the smoke position, beats the original net in both
seat-balanced arenas (one-ply 57.5 vs 64.7; as the 200-iter search leaf
54.9 vs 58.8), and keeps a sane value scale. ``head_to_head_info`` reads
~58.7 vs the original's ~51.7 — that grader rewards heuristic-overfitting,
and this net's data is only ~25-60% heuristic games.

Run from the repo root (~15 min on 6 cores):
    uv run python scripts/step16_train_v2.py
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
        eps_end=0.05,            # anneal low: calibration (see module docstring)
        heur_frac_start=1.0,
        heur_frac_end=0.25,
        search_frac_start=0.0,
        search_frac_end=0.4,     # expert-iteration leg, annealed in
        search_iters=200,
        search_c=30.0,
        deviation_frac=0.5,      # exploring deviations: coverage sans inflation
        deviation_horizon=30,
        target_refresh=5,
        hidden=64,
        n_removed=9,
        # The shipped net was trained with n_jobs=6; the parallel data stream
        # is worker-count invariant, so any n_jobs>1 reproduces it exactly
        # (n_jobs=1 uses the older sequential stream and will differ).
        n_jobs=6,
        seed=0,
        log=True,
    )
    net.save("models/info_net_3p_v2.npz")
    print(f"saved models/info_net_3p_v2.npz  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":  # required: the n_jobs pool respawns __main__ on macOS
    main()
