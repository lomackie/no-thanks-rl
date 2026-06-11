"""Roadmap step 19, part 2: retrain the 4p/5p nets with the cheap-anchor recipe.

**Outcome: rejected — no ``_v2`` files ship for 4p/5p** (see CLAUDE.md step
19). The retrains regressed the smoke positions the old nets already got
right (the old 4p net never had the bias; the old 5p net's one instance is
covered by the 2000-iter budget at deployment) and lost their arenas
(``scripts/step19_grade_45p.py``). Kept as the runnable record; if rerun,
the products are preferred by ``default_net_path`` automatically, so gate
before keeping the files.

Run from the repo root (~1-2 h on 6 cores for both):
    uv run python scripts/step19_train_45p.py [4|5]
"""

import sys
import time

from nothanks.beliefnet import train_info


def train_one(n_players: int) -> None:
    t0 = time.time()
    net = train_info(
        n_players=n_players,
        iterations=80,
        games_per_iter=80,
        eps_start=0.3,
        eps_end=0.05,
        heur_frac_start=1.0,
        heur_frac_end=0.25,
        search_frac_start=0.0,
        search_frac_end=0.4,
        search_iters=200,
        search_c=30.0,
        deviation_frac=0.5,
        deviation_horizon=30,
        take_dev_frac=0.5,
        take_dev_horizon=8,
        take_dev_margin=2,
        target_refresh=5,
        hidden=64,
        n_removed=9,
        n_jobs=6,
        seed=0,
        log=True,
    )
    path = f"models/info_net_{n_players}p_v2.npz"
    net.save(path)
    print(f"saved {path}  ({time.time() - t0:.0f}s)")


def main() -> None:
    counts = [int(sys.argv[1])] if len(sys.argv) > 1 else [4, 5]
    for n in counts:
        train_one(n)


if __name__ == "__main__":  # required: the n_jobs pool respawns __main__ on macOS
    main()
