"""Distill the bot's policy into human-readable take/pass rules.

Arguably the truest reading of the project goal ("derive a strategy"): query
the strongest honest bot across many positions and compress its choices into
rules a human can play at the table. The compression template is the
heuristic's own: **take iff ``score_delta(card) − pot ≤ T``** — the net cost,
in points, of taking right now. The heuristic fixes ``T = 0``; here ``T`` is
*fitted* to the bot's decisions, globally and then per context (card size,
chip stock, game phase, run connectivity), and every fit reports its agreement
percentage, which says how rule-like the bot actually is in that context —
the residual disagreement is the part of its strategy a one-number threshold
cannot express.

Decisions are collected from self-play on the **belief game** (all seats on
the candidate policy), so everything here is public-information only; forced
takes (no chips) carry no information about the policy and are skipped.

Run as a module for the report: ``python -m nothanks.distill --help``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable

from .belief import InfoPolicy, is_terminal
from .beliefnet import belief_step, new_belief_game
from .engine import score_delta
from .imperfect import InfoSet, legal_actions, pile_remaining


@dataclass(frozen=True)
class Decision:
    """One non-forced choice by the policy, with the public context that drove it."""

    cost: int        # score_delta(mover's cards, active) − pot: net points to take now
    action: str      # what the policy chose
    card: int        # the face-up card
    pot: int
    chips: int       # mover's chips (≥1: forced takes are excluded)
    pile_left: int   # cards still to be flipped
    connects: bool   # taking extends/bridges one of the mover's runs


def collect_decisions(
    policy: InfoPolicy,
    n_games: int = 200,
    n_players: int = 3,
    deck=None,
    n_removed: int = 9,
    start_chips: int | None = None,
    seed: int = 0,
) -> list[Decision]:
    """Self-play ``n_games`` belief games (all seats on ``policy``); log choices."""
    out: list[Decision] = []
    for i in range(n_games):
        rng = random.Random(seed + i)
        info = new_belief_game(n_players, deck=deck, n_removed=n_removed,
                               start_chips=start_chips, rng=rng)
        while not is_terminal(info):
            acts = legal_actions(info)
            if len(acts) == 1:
                info = belief_step(info, acts[0], rng)
                continue
            a = policy(info)
            p = info.to_move
            delta = score_delta(info.cards[p], info.active)
            out.append(Decision(
                cost=delta - info.pot,
                action=a,
                card=info.active,
                pot=info.pot,
                chips=info.chips[p],
                pile_left=pile_remaining(info),
                connects=delta < info.active,
            ))
            info = belief_step(info, a, rng)
    return out


def agreement(decisions: Iterable[Decision], threshold: int) -> float:
    """Fraction of decisions matching ``take iff cost ≤ threshold``."""
    ds = list(decisions)
    hit = sum(1 for d in ds if (d.action == "take") == (d.cost <= threshold))
    return hit / len(ds) if ds else 1.0


def fit_threshold(decisions: Iterable[Decision]) -> tuple[int, float]:
    """The threshold maximising agreement with the policy's choices.

    Candidates only need to be the observed costs (and one below the minimum:
    the "never take" rule); between two observed costs the rule is constant.
    Ties break toward the smaller (more conservative) threshold.
    """
    ds = list(decisions)
    if not ds:
        return 0, 1.0
    candidates = sorted({d.cost for d in ds} | {min(d.cost for d in ds) - 1})
    best_t, best_acc = candidates[0], -1.0
    for t in candidates:
        acc = agreement(ds, t)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def threshold_table(
    decisions: Iterable[Decision],
    bucket: Callable[[Decision], str],
) -> list[tuple[str, int, float, int]]:
    """Fit a threshold per ``bucket``; rows of (label, T, agreement, n)."""
    groups: dict[str, list[Decision]] = {}
    for d in decisions:
        groups.setdefault(bucket(d), []).append(d)
    rows = []
    for label in sorted(groups):
        t, acc = fit_threshold(groups[label])
        rows.append((label, t, acc, len(groups[label])))
    return rows


# --------------------------------------------------------------------------- #
# The printed report
# --------------------------------------------------------------------------- #

def _print_table(title: str, rows: list[tuple[str, int, float, int]]) -> None:
    print(f"\n{title}")
    print(f"  {'context':<14} {'T*':>4} {'agree':>7} {'n':>7}")
    for label, t, acc, n in rows:
        print(f"  {label:<14} {t:>4d} {acc:>6.1%} {n:>7d}")


def report(decisions: list[Decision]) -> None:
    """Print the distilled strategy: global rule, then per-context refinements."""
    t, acc = fit_threshold(decisions)
    base = agreement(decisions, 0)
    takes = sum(1 for d in decisions if d.action == "take")
    print(f"{len(decisions)} non-forced decisions, {takes / len(decisions):.1%} takes")
    print(f"\nglobal rule — take iff score_delta(card) − pot ≤ T:")
    print(f"  fitted T* = {t}  (agreement {acc:.1%}; the heuristic's T=0 "
          f"agrees {base:.1%})")

    # Zero-padded labels so the lexicographic sort in threshold_table is numeric.
    def card_band(d: Decision) -> str:
        for lo, hi in ((3, 9), (10, 14), (15, 19), (20, 24), (25, 29)):
            if d.card <= hi:
                return f"card {lo:02d}-{hi:02d}"
        return "card 30-35"

    def chip_band(d: Decision) -> str:
        for lo, hi in ((1, 2), (3, 5), (6, 9)):
            if d.chips <= hi:
                return f"chips {lo:02d}-{hi:02d}"
        return "chips 10+"

    def phase_band(d: Decision) -> str:
        for lo, hi in ((1, 6), (7, 12), (13, 18)):
            if d.pile_left <= hi:
                return f"pile {lo:02d}-{hi:02d}"
        return "pile 19+"

    _print_table("by face-up card value:", threshold_table(decisions, card_band))
    _print_table("by mover's chip stock:", threshold_table(decisions, chip_band))
    _print_table("by game phase (cards left in pile):",
                 threshold_table(decisions, phase_band))
    _print_table("by run connectivity (does taking extend/bridge a run?):",
                 threshold_table(
                     decisions,
                     lambda d: "connects" if d.connects else "isolated"))


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="nothanks.distill",
        description="distill the bot's policy into threshold rules")
    p.add_argument("--policy", choices=("mcts", "net"), default="mcts",
                   help="mcts = IS-MCTS bot (strongest, slower); net = one-ply info net")
    p.add_argument("--net", default=None,
                   help="path to the saved info-set net (.npz); default: the "
                        "models/info_net_3p convention (preferring _v2)")
    p.add_argument("--games", type=int, default=0,
                   help="self-play games to sample (default: 120 mcts / 600 net)")
    p.add_argument("--n-iter", type=int, default=200, help="IS-MCTS iterations")
    p.add_argument("--c", type=float, default=30.0, help="IS-MCTS exploration constant")
    p.add_argument("--n-removed", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from .valuefn import ValueNet

    net_path = args.net
    if net_path is None:
        from .beliefnet import default_net_path

        found = default_net_path(3)
        if found is None:
            raise SystemExit("no saved 3p info net found; pass --net")
        net_path = str(found)
    net = ValueNet.load(net_path)
    if args.policy == "net":
        from .beliefnet import make_greedy_info_policy

        policy = make_greedy_info_policy(net)
        n_games = args.games or 600
        label = "one-ply greedy info net"
    else:
        from .ismcts import make_ismcts_policy, make_value_leaf

        policy = make_ismcts_policy(n_iter=args.n_iter,
                                    evaluator=make_value_leaf(net),
                                    c=args.c, seed=args.seed)
        n_games = args.games or 120
        label = f"IS-MCTS (net leaf, {args.n_iter} iters, c={args.c:g})"

    print(f"distilling: {label}, {n_games} self-play games, "
          f"{args.n_removed} removed\n")
    decisions = collect_decisions(policy, n_games=n_games,
                                  n_removed=args.n_removed, seed=args.seed)
    report(decisions)


if __name__ == "__main__":
    main()
