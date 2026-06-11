"""Roadmap step 17: a search-capable best response to the IS-MCTS bot.

Step 11b's one-ply responder returned hugely negative gains against the
searcher — the hero class was too weak to probe it, so the lower bound was
vacuous. Here the hero net is trained exactly as before
(:func:`nothanks.approx_br.train_best_response` against the frozen IS-MCTS
candidate), but *deployed* as a searcher too: an IS-MCTS policy with the
BR-trained net as leaf, at the candidate's own budget. The joint deviation is
still a deterministic ``InfoPolicy`` (``make_ismcts_policy`` seeds from the
canonical info-set key), so the measured gain stays a valid lower bound on
true exploitability.

Reading the number: small gain = necessary evidence only; a large gain that
bigger budgets don't shrink is the trigger for equilibrium-aware training.

Slow (searches on both sides of both arms — expect ~1h per hero at the default
budget). One hero per invocation so the seats can run as parallel processes:

    uv run python scripts/step17_search_br.py --hero 0 &
    uv run python scripts/step17_search_br.py --hero 1 &
    uv run python scripts/step17_search_br.py --hero 2 &
"""

import argparse
import time

from nothanks.approx_br import (
    deviation_policy,
    estimate_deviation_gain_mc,
    train_best_response,
)
from nothanks.ismcts import make_ismcts_policy, make_value_leaf
from nothanks.valuefn import ValueNet


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hero", type=int, required=True, help="deviating seat (0..2)")
    p.add_argument("--net", default="models/info_net_3p_v2.npz",
                   help="info net for the candidate's (and a fresh hero's) leaf")
    p.add_argument("--n-iter", type=int, default=200,
                   help="IS-MCTS budget for candidate and hero alike")
    p.add_argument("--c", type=float, default=30.0)
    p.add_argument("--iterations", type=int, default=15, help="BR training iters")
    p.add_argument("--games-per-iter", type=int, default=40)
    p.add_argument("--n-games", type=int, default=300, help="paired MC games")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="worker processes for episode generation and MC games")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    net = ValueNet.load(args.net)
    candidate = make_ismcts_policy(n_iter=args.n_iter,
                                   evaluator=make_value_leaf(net),
                                   c=args.c, seed=args.seed)

    t0 = time.time()
    # Warm-start from the candidate's own net: a from-scratch hero net at this
    # budget is a misleading search leaf (first run: gains of -35 to -56, the
    # deviating searcher scoring 94-114 — worse than vacuous). Initialised at
    # the candidate, "no training signal" deviates to ~the candidate itself.
    hero_net = train_best_response(
        candidate, args.hero, n_players=3, n_removed=9,
        iterations=args.iterations, games_per_iter=args.games_per_iter,
        init_net=net, n_jobs=args.n_jobs, seed=args.seed + args.hero, log=True,
    )
    t_train = time.time() - t0
    print(f"hero {args.hero}: BR net trained ({t_train:.0f}s)", flush=True)

    # The step-17 move: the hero deviates with a *search* over its BR net, at
    # the candidate's own budget — a responder in the same class as the bot.
    searcher = make_ismcts_policy(n_iter=args.n_iter,
                                  evaluator=make_value_leaf(hero_net),
                                  c=args.c, seed=args.seed + 1000 + args.hero)
    deviate = deviation_policy(candidate, searcher, args.hero)

    res = estimate_deviation_gain_mc(
        candidate, deviate, args.hero, n_players=3, n_removed=9,
        n_games=args.n_games, n_jobs=args.n_jobs, seed=args.seed + 50_000,
    )
    print(f"\n=== step 17: search-capable BR vs IS-MCTS bot "
          f"(net {args.net}, {args.n_iter} iters, c={args.c}) ===")
    print(f"  hero {args.hero}: base {res['base']:+.3f}  br {res['br']:+.3f}  "
          f"gain {res['gain']:+.3f} ± {res['stderr']:.3f}  "
          f"({res['n_games']} paired games)")
    print(f"  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
