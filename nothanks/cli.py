"""The chess-engine-style front end: evaluate an arbitrary position from the shell.

Everything before this module could only analyse positions reached from
``new_game`` inside a demo. This CLI closes the loop on the project goal: type in
any position — chips, captured cards, the face-up card, the pot, whose turn — and
get the per-move EV table from an *honest* evaluator (the position is parsed into
an :class:`~nothanks.imperfect.InfoSet`, so no method here can peek at the
removed cards).

Position format
---------------
``--cards`` is one group per seat, ``;``-separated, each a comma list of cards
with ranges (``"3-5,22;17;"`` = seat 0 holds {3,4,5,22}, seat 1 holds {17},
seat 2 nothing). ``--chips`` is a comma list, one per seat, and implies the
player count. Example::

    python -m nothanks.cli eval --chips 9,11,10 --cards "3-5,22;17;" \
        --active 26 --pot 3 --to-move 0

Methods: ``ismcts`` (default — IS-MCTS with heuristic playouts, or a net leaf
with ``--net``), ``net`` (one-ply lookahead on a trained info net, instant), and
``pimc`` (determinized Monte-Carlo rollouts; the strategy-fusion-prone baseline,
kept for comparison).

When methods disagree: trust ``ismcts`` by default (it is by far the strongest
player in fair bot-vs-bot grading), but on close take/pass calls over an
isolated or gapped high card with a decent pot, treat net-based EVs with
caution — the trained net systematically overprices those takes, and the search
inherits the bias when the take arm is visit-starved (read the visit counts).
``pimc``'s EVs are the *rollout policy's*, biased toward heuristic play. See
CLAUDE.md roadmap step 12 for the adjudication behind this.

``train`` trains the honest info-set net (:func:`nothanks.beliefnet.train_info`)
and saves it: train once, analyse many times. ``play`` is an interactive
terminal game against :class:`~nothanks.ismcts.ISMCTSBot` (the browser version
is ``python -m nothanks.web``).
"""

from __future__ import annotations

import argparse
import random
import sys

from .engine import DECK_HIGH, DECK_LOW, full_deck
from .imperfect import InfoSet, evaluate_determinized, legal_actions, pile_remaining, unseen


def parse_card_list(spec: str) -> frozenset[int]:
    """Parse ``"3-5,22"`` → ``{3,4,5,22}``; the empty string is the empty set."""
    cards: set[int] = set()
    spec = spec.strip()
    if not spec:
        return frozenset()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if lo > hi:
                raise ValueError(f"bad range {part!r}")
            cards.update(range(lo, hi + 1))
        else:
            cards.add(int(part))
    return frozenset(cards)


def parse_cards(spec: str) -> tuple[frozenset[int], ...]:
    """Parse per-seat holdings: ``;``-separated card lists, one group per seat."""
    return tuple(parse_card_list(group) for group in spec.split(";"))


def build_info(args) -> InfoSet:
    """Build and validate the :class:`InfoSet` described by the CLI arguments."""
    chips = tuple(int(c) for c in args.chips.split(","))
    n = len(chips)
    cards = parse_cards(args.cards) if args.cards else tuple(frozenset() for _ in range(n))
    if len(cards) != n:
        raise ValueError(f"--cards has {len(cards)} seat groups but --chips has {n} seats")
    deck = frozenset(parse_card_list(args.deck)) if args.deck else frozenset(full_deck())

    info = InfoSet(
        chips=chips,
        cards=cards,
        active=args.active,
        pot=args.pot,
        to_move=args.to_move,
        deck=deck,
        n_removed=args.n_removed,
    )

    # Consistency checks with actionable messages (the engine would just misbehave).
    if any(c < 0 for c in chips):
        raise ValueError("chips must be non-negative")
    if not 0 <= args.to_move < n:
        raise ValueError(f"--to-move must be in 0..{n - 1}")
    held = [c for hand in cards for c in hand]
    if len(held) != len(set(held)):
        raise ValueError("a card appears in more than one seat's holdings")
    bad = [c for c in {*held, args.active} if c not in deck]
    if bad:
        raise ValueError(f"cards {sorted(bad)} are not in the deck "
                         f"({DECK_LOW}..{DECK_HIGH} unless --deck is given)")
    if args.active in set(held):
        raise ValueError(f"active card {args.active} is already captured")
    if pile_remaining(info) < 0:
        raise ValueError(
            f"inconsistent position: {len(unseen(info))} cards unseen but "
            f"pile_remaining={pile_remaining(info)} — too many cards have been seen "
            f"for a deck of {len(deck)} with {args.n_removed} removed")
    return info


def _load_info_net(path):
    from .features import info_feature_dim
    from .valuefn import ValueNet

    net = ValueNet.load(path)
    if net.in_dim != info_feature_dim(net.n_players):
        raise SystemExit(
            f"{path} is not an info-set net (in_dim={net.in_dim}); "
            "train one with `python -m nothanks.cli train`")
    return net


def _print_eval(info: InfoSet, ev: dict, extra: dict | None = None) -> None:
    print(f"P{info.to_move} to move — card {info.active} face-up, pot {info.pot}, "
          f"pile {pile_remaining(info)} of {len(unseen(info))} unseen")
    for a in legal_actions(info):
        if a not in ev["mover_ev"]:
            print(f"  {a:5s} -> (forced alternative not evaluated)")
            continue
        line = f"  {a:5s} -> EV {ev['mover_ev'][a]:+7.2f}"
        if "stderr" in ev and a in ev["stderr"]:
            line += f"  ± {ev['stderr'][a]:.2f}"
        if "visits" in ev and a in ev["visits"]:
            line += f"  ({ev['visits'][a]} visits)"
        print(line)
    print(f"  best: {ev['best_action']}")
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")


def cmd_eval(args) -> None:
    info = build_info(args)
    acts = legal_actions(info)
    if len(acts) == 1:
        print(f"P{info.to_move} to move — forced: {acts[0]} (no chips)")
        return
    rng = random.Random(args.seed)

    if args.method == "net":
        if not args.net:
            raise SystemExit("--method net needs --net (train one with the train command)")
        from .beliefnet import evaluate_info

        net = _load_info_net(args.net)
        if net.n_players != info.n_players:
            raise SystemExit(f"net is for {net.n_players} players, position has {info.n_players}")
        _print_eval(info, evaluate_info(info, net))
    elif args.method == "ismcts":
        from .ismcts import ismcts_evaluate, make_value_leaf

        leaf = None
        if args.net:
            net = _load_info_net(args.net)
            if net.n_players != info.n_players:
                raise SystemExit(f"net is for {net.n_players} players, position has {info.n_players}")
            leaf = make_value_leaf(net)
        ev = ismcts_evaluate(info, n_iter=args.n_iter, evaluator=leaf, c=args.c, rng=rng)
        _print_eval(info, ev, {"iterations": ev["n_iter"],
                               "leaf": "info net" if leaf else "heuristic playout"})
    elif args.method == "pimc":
        from .montecarlo import evaluate_mc

        per_world = lambda s: evaluate_mc(  # noqa: E731
            s, n_rollouts=args.rollouts, rng=random.Random(rng.randrange(1 << 62)))
        ev = evaluate_determinized(info, per_world, n_worlds=args.n_worlds, rng=rng)
        _print_eval(info, ev, {"worlds": ev["n_worlds"],
                               "note": "PIMC: strategy-fusion-prone baseline"})
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit(f"unknown method {args.method}")


def cmd_train(args) -> None:
    from .beliefnet import head_to_head_info, train_info

    net = train_info(
        n_players=args.n_players,
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        heur_frac_start=1.0,
        heur_frac_end=args.heur_frac_end,
        target_refresh=args.target_refresh,
        hidden=args.hidden,
        n_removed=args.n_removed,
        seed=args.seed,
        log=True,
    )
    net.save(args.out)
    print(f"saved info-set net to {args.out}")
    if args.grade:
        res = head_to_head_info(net, n_games=args.grade, n_removed=args.n_removed)
        print(f"vs heuristic over {args.grade} games: net {res['vnet_mean']:.2f} "
              f"± {res['vnet_stderr']:.2f}  heuristic {res['heuristic_mean']:.2f}  "
              f"win/tie {res['win_rate']:.1%} (parity {res['parity']:.0%})")


def format_cards(cards) -> str:
    """Compact run display: ``{3,4,5,22}`` → ``"3-5,22"``; empty → ``"-"``."""
    if not cards:
        return "-"
    out: list[str] = []
    run: list[int] = []
    for c in sorted(cards):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            out.append(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}")
        run = [c]
    out.append(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}")
    return ",".join(out)


def _print_board(s, human: int) -> None:
    print(f"\ncard {s.active} face-up, pot {s.pot} — {len(s.remaining)} cards left in pile")
    for q in range(s.n_players):
        who = f"you (P{q})" if q == human else f"P{q}"
        print(f"  {who:9s} chips {s.chips[q]:2d}   cards {format_cards(s.cards[q])}")


def cmd_play(args) -> None:
    from .engine import final_scores, is_terminal, new_game, step
    from .imperfect import info_from_state
    from .ismcts import ISMCTSBot, make_value_leaf

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    rng = random.Random(seed)
    n = args.n_players
    human = args.seat
    if not 0 <= human < n:
        raise ValueError(f"--seat must be in 0..{n - 1}")

    leaf = None
    if args.net:
        net = _load_info_net(args.net)
        if net.n_players != n:
            raise SystemExit(f"net is for {net.n_players} players, game has {n}")
        leaf = make_value_leaf(net)
    bots = {q: ISMCTSBot(n_iter=args.n_iter, evaluator=leaf, c=args.c, seed=seed + q)
            for q in range(n) if q != human}

    print(f"No Thanks — you are P{human} of {n}; {args.n_removed} cards removed "
          f"(AI: IS-MCTS, {args.n_iter} iters, "
          f"{'info-net leaf' if leaf else 'heuristic-playout leaf'})")
    print("lowest score wins: a card costs its face value, runs count their "
          "lowest card only, each chip is -1")

    s = new_game(n, n_removed=args.n_removed, rng=rng)
    while not is_terminal(s):
        if s.to_move == human:
            _print_board(s, human)
            if s.chips[human] == 0:
                print("  no chips — forced take")
                a = "take"
            else:
                while True:
                    try:
                        raw = input("  (t)ake or (p)ass? ").strip().lower()
                    except EOFError:
                        print("\nquit.")
                        return
                    if raw in ("q", "quit"):
                        print("quit.")
                        return
                    if raw in ("t", "take"):
                        a = "take"
                        break
                    if raw in ("p", "pass"):
                        a = "pass"
                        break
                    print("  t / p / q")
        else:
            a = bots[s.to_move].act(info_from_state(s, n_removed=args.n_removed))
            verb = f"takes card {s.active} (+{s.pot} chips)" if a == "take" else "passes"
            print(f"P{s.to_move} {verb}")
        s = step(s, a, rng)

    print("\ngame over — final scores (lower is better):")
    sc = final_scores(s)
    best = min(sc)
    for q in range(n):
        who = f"you (P{q})" if q == human else f"P{q}"
        mark = "  <- winner" if sc[q] == best else ""
        print(f"  {who:9s} {sc[q]:4d}   cards {format_cards(s.cards[q])}, "
              f"chips {s.chips[q]}{mark}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nothanks", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("eval", help="evaluate a position (engine-style EV table)")
    e.add_argument("--chips", required=True, help="comma list, one per seat: 9,11,10")
    e.add_argument("--cards", default="", help="per-seat holdings: '3-5,22;17;'")
    e.add_argument("--active", type=int, required=True, help="face-up card")
    e.add_argument("--pot", type=int, default=0, help="chips on the face-up card")
    e.add_argument("--to-move", type=int, default=0, help="seat to act (default 0)")
    e.add_argument("--n-removed", type=int, default=9)
    e.add_argument("--deck", default="", help="card universe, default 3-35")
    e.add_argument("--method", choices=("ismcts", "net", "pimc"), default="ismcts")
    e.add_argument("--net", default="", help="path to a saved info-set net (.npz)")
    e.add_argument("--n-iter", type=int, default=2000, help="IS-MCTS iterations")
    e.add_argument("--c", type=float, default=30.0,
                   help="IS-MCTS exploration constant, on the scale of final "
                        "scores (full game ~30; tiny decks ~1.5)")
    e.add_argument("--n-worlds", type=int, default=100, help="PIMC worlds")
    e.add_argument("--rollouts", type=int, default=200, help="PIMC rollouts per world")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(fn=cmd_eval)

    t = sub.add_parser("train", help="train + save the honest info-set net")
    t.add_argument("--out", required=True, help="output path (.npz)")
    t.add_argument("--n-players", type=int, default=3)
    t.add_argument("--iterations", type=int, default=60)
    t.add_argument("--games-per-iter", type=int, default=80)
    t.add_argument("--heur-frac-end", type=float, default=0.25,
                   help="curriculum endpoint (1.0 = heuristic-only data)")
    t.add_argument("--target-refresh", type=int, default=5)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--n-removed", type=int, default=9)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--grade", type=int, default=500,
                   help="games for the post-train grading run (0 = skip)")
    t.set_defaults(fn=cmd_train)

    g = sub.add_parser("play", help="interactive terminal game vs the IS-MCTS bot")
    g.add_argument("--n-players", type=int, default=3)
    g.add_argument("--seat", type=int, default=0, help="your seat (default 0)")
    g.add_argument("--net", default="models/info_net_3p.npz",
                   help="info-set net for the search leaf ('' = heuristic playouts)")
    g.add_argument("--n-iter", type=int, default=400, help="IS-MCTS iterations per move")
    g.add_argument("--c", type=float, default=30.0, help="IS-MCTS exploration constant")
    g.add_argument("--n-removed", type=int, default=9)
    g.add_argument("--seed", type=int, default=None, help="fix the deal (default: random)")
    g.set_defaults(fn=cmd_play)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main(sys.argv[1:])
