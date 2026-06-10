# no-thanks-rl

Tools for understanding the card game **No Thanks** — a game engine, an exact
solver that acts as a ground-truth EV oracle on small configurations, a
Monte-Carlo rollout evaluator that scales to the full 24-card removal deck, a
self-play (TD-λ) value function trained on public information only, and an
Information-Set MCTS bot that combines the two — the strongest honest player
here (win/tie ~85% vs the heuristic baseline), never peeking at the nine
removed cards.

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
just play     # play against the bot in a browser (http://localhost:8000)
just play-cli # play against the bot in the terminal
just distill  # print the bot's strategy as human-readable threshold rules
just repl     # python REPL with the project importable
```

Or without `just`: `uv sync`, `uv run pytest`, `uv run python -m nothanks.demo`.

## Evaluate a position

The point of the project: type in any position and get the per-move EV table.
Holdings are `;`-separated per seat, comma lists with ranges:

```sh
just eval --chips 9,11,10 --cards "3-5,22;17;" --active 26 --pot 3 --to-move 0
```

The default method is IS-MCTS with honest heuristic playouts. Train the info-set
value net once for stronger, faster evals, then pass it back in:

```sh
just train-info --out models/info_net_3p.npz
just eval --net models/info_net_3p.npz --method net  --chips 9,11,10 \
    --cards "3-5,22;17;" --active 26 --pot 3          # instant one-ply eval
just eval --net models/info_net_3p.npz --chips 9,11,10 \
    --cards "3-5,22;17;" --active 26 --pot 3          # IS-MCTS with the net leaf
```

Every method parses the position into an *info set* (public knowledge only), so
nothing here can peek at the nine removed cards.

## Modules

- `nothanks/engine.py` — game state, transitions, and run-aware scoring.
- `nothanks/heuristic.py` — a simple run-aware baseline / rollout policy.
- `nothanks/solver.py` — exact backward-induction EV oracle for small games.
- `nothanks/montecarlo.py` — Monte-Carlo rollout EV evaluator for the full deck,
  plus an exact policy-evaluation counterpart that the sampler is validated against.
- `nothanks/features.py` — mover-relative feature encodings: god-view (`features`)
  and public info-set (`info_features`).
- `nothanks/valuefn.py` — tiny NumPy MLP value function (vector-valued head) and
  the fast one-ply `evaluate_v` move eval.
- `nothanks/train.py` — self-play TD(λ) training (heuristic→self-play curriculum
  with a target network) plus head-to-head and seat-balanced net-vs-net graders.
- `nothanks/imperfect.py` — hidden removed cards: information sets, determinization,
  and PIMC move analysis wrapping any per-world evaluator.
- `nothanks/exploit.py` — exact best-response / exploitability for the testbed.
- `nothanks/belief.py` — the hidden game as a Markov game on info sets: exact
  belief-correct policy evaluation, optimum, and exploitability.
- `nothanks/ismcts.py` — Information-Set MCTS (fixes PIMC's strategy fusion), the
  value-net leaf, and the deployable persistent-tree bot.
- `nothanks/beliefnet.py` — info-set-native value net: honest self-play training on
  the belief game, honest one-ply evals, no PIMC wrapper needed.
- `nothanks/approx_br.py` — approximate best response: learned-exploitability
  lower bounds that scale to the full game.
- `nothanks/arena.py` — seat-balanced bot-vs-bot matches between the honest bots
  (the fair grader: no heuristic opponent to overfit to).
- `nothanks/distill.py` — distills the bot's policy into human-readable take/pass
  threshold rules with per-context agreement percentages (`just distill`).
- `nothanks/cli.py` — the position-input CLI (`just eval`, `just train-info`),
  plus an interactive terminal game vs the bot (`just play-cli`).
- `nothanks/web.py` — browser game against the IS-MCTS bot (`just play`).
- `nothanks/demo.py` — prints an engine-style exact evaluation of a tiny opening.
- `nothanks/mc_demo.py` — Monte-Carlo eval demo (sampler vs. exact on a tiny game).
- `nothanks/imperfect_demo.py` — determinized (PIMC) eval + exploitability demo.

## Play against it

```sh
just play      # browser game at http://localhost:8000
just play-cli  # the same bot in the terminal
```

The browser game can show live analysis (per-move EV and a projected final
score for every player), and has an **advisor mode**: relay the moves of a
real-life game — after a take, enter the card that was actually flipped — and
the bot recommends your moves as you go.

## The strategy, in one table

`just distill` queries the bot across thousands of self-play decisions and fits
the rule *take iff `score_delta(card) − pot ≤ T`* (the heuristic's own template;
`score_delta` is what the card adds to your score given your runs). The bot
plays roughly that rule with **T ≈ −3** — demand about a 3-chip premium, more
for big cards (T ≈ −5 for 30–35) — and earns its edge where no threshold rule
fits: positions where taking extends or bridges one of its runs.
