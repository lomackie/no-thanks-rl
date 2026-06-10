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

# Open a Python REPL with the project importable
repl:
    uv run python
