"""Fair bot-vs-bot grading: seat-balanced matches between the honest bots.

Every full-game number before this module was vs-the-heuristic — the grader
this project itself flagged as exploitation-prone (a bot can score well there
by exploiting the heuristic specifically, not by playing well in general).
:func:`nothanks.train.net_vs_net` is the fair design but is hard-wired to
god-view greedy nets; this module generalises it to any *honest* bot so the
three deployable players (greedy info-net, PIMC-wrapped god net, IS-MCTS) can
be ranked against each other.

A bot here is a per-game object: ``BotFactory(seed) -> Bot`` builds a fresh
instance for each game (IS-MCTS keeps a tree across the game's moves; the
stateless bots just close over their net), and ``Bot(state, rng) -> action``
makes one move. Bots receive the referee's god-view :class:`State` but every
factory in this module projects it onto the mover's
:class:`~nothanks.imperfect.InfoSet` before deciding, so none of them can peek
at the removed cards — same contract as :func:`nothanks.train.pimc_policy`.

:func:`bot_vs_bot` is the match harness: bot A rotates through every seat
(seat 0 has a large structural first-mover edge), re-dealing the same
``n_games`` shuffles for each seat assignment, with B filling the other seats.
The shared per-game seed pairs the deals only — play diverges at the first
differing action — so the variance reduction is partial.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from .engine import State, final_scores, is_terminal, new_game, step
from .heuristic import heuristic_action
from .imperfect import info_from_state
from .valuefn import ValueNet

# One move from the referee's state (projected to public info internally).
Bot = Callable[[State, random.Random], str]
# A fresh bot per game; ``seed`` derandomises any internal search.
BotFactory = Callable[[int], Bot]


def heuristic_bot(threshold: int = 0) -> BotFactory:
    """The run-aware baseline as a factory (already a public-info policy)."""
    return lambda seed: lambda s, rng: heuristic_action(s, threshold)


def greedy_info_bot(net: ValueNet, n_removed: int = 9) -> BotFactory:
    """One-ply greedy on the info-set net — honest by construction, instant."""
    from .beliefnet import greedy_info_action

    def make(seed: int) -> Bot:
        return lambda s, rng: greedy_info_action(
            info_from_state(s, n_removed=n_removed), net)

    return make


def pimc_god_bot(net: ValueNet, n_removed: int = 9, n_worlds: int = 80) -> BotFactory:
    """The god-view net made honest at play time by PIMC (train.pimc_policy)."""
    from .train import pimc_policy

    def make(seed: int) -> Bot:
        policy = pimc_policy(net, n_removed=n_removed, n_worlds=n_worlds)
        return lambda s, rng: policy(s, rng)

    return make


def ismcts_bot(
    info_net: ValueNet | None = None,
    n_iter: int = 200,
    c: float = 30.0,
    n_removed: int = 9,
) -> BotFactory:
    """The deployable searcher: persistent in-game tree, info-net (or playout) leaf."""
    from .ismcts import ISMCTSBot, make_value_leaf

    leaf = make_value_leaf(info_net) if info_net is not None else None

    def make(seed: int) -> Bot:
        bot = ISMCTSBot(n_iter=n_iter, evaluator=leaf, c=c, seed=seed)
        return lambda s, rng: bot.act(info_from_state(s, n_removed=n_removed))

    return make


def bot_vs_bot(
    make_a: BotFactory,
    make_b: BotFactory,
    n_players: int = 3,
    n_games: int = 200,
    seed: int = 40_000,
    n_removed: int = 9,
) -> dict:
    """Seat-balanced match: A in one seat (rotated through all), B elsewhere.

    Lower ``a_mean`` than ``b_mean`` means A is the stronger policy;
    ``a_win_rate`` is A's tie-or-beat rate against ``parity = 1/n``. Read the
    mean gap against ``a_stderr`` (the per-game spread of A's score; the paired
    deals shrink the *comparison* noise further, so this is conservative).
    """
    a_scores: list[float] = []
    b_total = 0.0
    a_wins = 0
    for seat in range(n_players):
        for i in range(n_games):
            rng = random.Random(seed + i)  # same deals across seat assignments
            bot_a = make_a(seed + 31 * seat + i)
            bots_b = {q: make_b(seed + 31 * seat + i + 1_000_003 * (q + 1))
                      for q in range(n_players) if q != seat}
            s = new_game(n_players, n_removed=n_removed, rng=rng)
            while not is_terminal(s):
                mover = s.to_move
                a = bot_a(s, rng) if mover == seat else bots_b[mover](s, rng)
                s = step(s, a, rng)
            sc = final_scores(s)
            a_scores.append(sc[seat])
            b_total += (sum(sc) - sc[seat]) / (n_players - 1)
            if sc[seat] <= min(sc):
                a_wins += 1
    games = len(a_scores)
    a_mean = sum(a_scores) / games
    a_var = sum((x - a_mean) ** 2 for x in a_scores) / games
    return {
        "a_mean": a_mean,
        "a_stderr": math.sqrt(a_var / games),
        "b_mean": b_total / games,
        "a_win_rate": a_wins / games,
        "parity": 1.0 / n_players,
        "games": games,
    }
