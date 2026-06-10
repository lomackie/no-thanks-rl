# No Thanks dev tasks. Run `just` to list.

default:
    @just --list

# Install/sync the uv environment (incl. dev deps)
sync:
    uv sync

# Run the test suite
test *args:
    uv run pytest {{args}}

# Exact-solve a tiny game and print an engine-style opening eval
demo:
    uv run python -m nothanks.demo

# Monte-Carlo eval: tiny-game sanity check + a full 24-card opening
mc-demo:
    uv run python -m nothanks.mc_demo

# Train a self-play value net (TD-λ), show an eval, and grade it vs the heuristic
train:
    uv run python -m nothanks.train

# Hidden removed cards: determinized (PIMC) eval + exploitability checks
imperfect:
    uv run python -m nothanks.imperfect_demo

# Engine-style EV eval of an arbitrary position (see `just eval --help`)
eval *args:
    uv run python -m nothanks.cli eval {{args}}

# Train + save the honest info-set net (then `just eval --net models/...npz ...`)
train-info *args:
    uv run python -m nothanks.cli train {{args}}

# Play No Thanks against the AI in a browser (opens at http://localhost:8000)
play *args:
    uv run --group web python -m nothanks.web {{args}}

# Open a Python REPL with the project importable
repl:
    uv run python
