# no-thanks-rl

Tools for understanding the card game **No Thanks** — a game engine, an exact
solver that acts as a ground-truth EV oracle on small configurations, a
Monte-Carlo rollout evaluator that scales to the full 24-card removal deck, and a
self-play (TD-λ) value function that gives fast, rollout-free evals and plays
stronger than the heuristic baseline.

The goal is a *chess-engine-style EV evaluator*: given a position, show the
expected value of each move and the value of the state — not just a bot to play
against.

## The game

Deck of cards **3–35** (33 cards). In the standard game **9 are removed at
random and unseen**. Each player starts with chips (11 for 3–5 players). On your
turn you either **pay a chip** onto the face-up card to pass, or **take the card
and all chips on it**; with no chips you must take. Cards score their face value
but a **consecutive run only counts its lowest card**, and **chips are −1 each**.
Lowest total wins. Taking a card lets you keep acting (you flip and face the next
card); you only yield the turn by passing.

## Install & run

Requires [uv](https://docs.astral.sh/uv/) and (optionally) [just](https://github.com/casey/just).

```sh
just sync     # uv sync
just test     # uv run pytest
just demo     # exact-solve a tiny game and print an opening eval
just mc-demo  # Monte-Carlo eval: tiny-game sanity check + a full opening
just train    # train a self-play value net, eval an opening, grade it vs heuristic
just imperfect # hidden removed cards: determinized (PIMC) eval + exploitability
just repl     # python REPL with the project importable
```

Or without `just`: `uv sync`, `uv run pytest`, `uv run python -m nothanks.demo`.

## Modules

- `nothanks/engine.py` — game state, transitions, and run-aware scoring.
- `nothanks/heuristic.py` — a simple run-aware baseline / rollout policy.
- `nothanks/solver.py` — exact backward-induction EV oracle for small games.
- `nothanks/montecarlo.py` — Monte-Carlo rollout EV evaluator for the full deck,
  plus an exact policy-evaluation counterpart that the sampler is validated against.
- `nothanks/features.py` — mover-relative feature encoding of a state.
- `nothanks/valuefn.py` — tiny NumPy MLP value function (vector-valued head) and
  the fast one-ply `evaluate_v` move eval.
- `nothanks/train.py` — self-play TD(λ) training (heuristic→self-play curriculum
  with a target network) plus head-to-head and seat-balanced net-vs-net graders.
- `nothanks/imperfect.py` — hidden removed cards: information sets, determinization,
  and PIMC move analysis wrapping any per-world evaluator.
- `nothanks/exploit.py` — exact best-response / exploitability for the testbed.
- `nothanks/demo.py` — prints an engine-style exact evaluation of a tiny opening.
- `nothanks/mc_demo.py` — Monte-Carlo eval demo (sampler vs. exact on a tiny game).
- `nothanks/imperfect_demo.py` — determinized (PIMC) eval + exploitability demo.
