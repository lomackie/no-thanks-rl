"""Web UI server for playing No Thanks against the AI.

Run with `just play` (or `uv run python -m nothanks.web`).
Visit http://localhost:8000 in a browser.

Sessions are in-memory only; each browser tab gets its own game. The AI is
ISMCTSBot backed by the trained info-set net (models/info_net_3p.npz) when
available, falling back to heuristic playouts.
"""

from __future__ import annotations

import os
import pathlib
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import (
    DECK_HIGH,
    DECK_LOW,
    State,
    final_scores,
    full_deck,
    is_terminal,
    new_game,
    score_cards,
    step,
)
from .imperfect import InfoSet, info_from_state, legal_actions, pile_remaining, unseen
from .ismcts import ISMCTSBot, make_value_leaf

_HERE = pathlib.Path(__file__).parent
_MODELS_DIR = _HERE.parent / "models"
_DEFAULT_NET = _MODELS_DIR / "info_net_3p.npz"

_executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="No Thanks!")

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

@dataclass
class GameSession:
    state: State
    n_removed: int
    deck: frozenset[int]
    human_seat: int
    bots: list[ISMCTSBot | None]
    rng: random.Random
    log: list[dict] = field(default_factory=list)


_sessions: dict[str, GameSession] = {}


def _make_bots(
    n_players: int,
    human_seat: int,
    n_iter: int,
    net_path: pathlib.Path | None,
    seed: int,
) -> list[ISMCTSBot | None]:
    evaluator = None
    if net_path and net_path.exists():
        try:
            from .features import info_feature_dim
            from .valuefn import ValueNet

            net = ValueNet.load(str(net_path))
            if net.n_players == n_players:
                evaluator = make_value_leaf(net)
        except Exception:
            pass

    bots: list[ISMCTSBot | None] = []
    for seat in range(n_players):
        if seat == human_seat:
            bots.append(None)
        else:
            bots.append(ISMCTSBot(n_iter=n_iter, evaluator=evaluator, c=30.0,
                                  seed=seed + seat))
    return bots


def _state_to_dict(session: GameSession) -> dict[str, Any]:
    s = session.state
    n = s.n_players
    info = info_from_state(s, session.n_removed, session.deck)

    players = []
    for seat in range(n):
        hand = sorted(s.cards[seat])
        card_score = score_cards(hand)
        players.append({
            "seat": seat,
            "is_human": seat == session.human_seat,
            "chips": s.chips[seat],
            "cards": hand,
            "card_score": card_score,
            "score": card_score - s.chips[seat],
        })

    result: dict[str, Any] = {
        "session_id": None,  # filled by caller
        "game_over": is_terminal(s),
        "human_seat": session.human_seat,
        "to_move": s.to_move,
        "is_human_turn": (not is_terminal(s)) and s.to_move == session.human_seat,
        "active_card": s.active,
        "pot": s.pot,
        "pile_remaining": pile_remaining(info) if not is_terminal(s) else 0,
        "unseen_count": len(unseen(info)) if not is_terminal(s) else 0,
        "players": players,
        "log": session.log[-30:],
        "final_scores": list(final_scores(s)) if is_terminal(s) else None,
        "legal_actions": list(legal_actions(info)) if not is_terminal(s) else [],
    }
    return result


def _run_ai_turns(session: GameSession) -> list[dict]:
    """Advance the game through AI turns until it's the human's turn or game over."""
    events: list[dict] = []
    while (not is_terminal(session.state) and
           session.state.to_move != session.human_seat):
        seat = session.state.to_move
        bot = session.bots[seat]
        assert bot is not None

        info = info_from_state(session.state, session.n_removed, session.deck)
        action = bot.act(info)

        card = session.state.active
        pot_before = session.state.pot
        session.state = step(session.state, action, session.rng)

        evt: dict[str, Any] = {"seat": seat, "action": action, "card": card}
        if action == "take":
            evt["chips_gained"] = pot_before
        session.log.append(evt)
        events.append(evt)

    return events


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

class NewGameRequest(BaseModel):
    n_players: int = 3
    human_seat: int | None = None  # None = random
    n_iter: int = 400
    n_removed: int = 9
    seed: int | None = None


class ActionRequest(BaseModel):
    session_id: str
    action: str  # "take" or "pass"


@app.post("/api/new_game")
def api_new_game(req: NewGameRequest) -> dict:
    if not 3 <= req.n_players <= 7:
        raise HTTPException(400, "n_players must be 3..7")

    seed = req.seed if req.seed is not None else random.randrange(1 << 31)
    rng = random.Random(seed)

    deck_list = full_deck()
    rng.shuffle(deck_list)
    if req.n_removed:
        removed = set(deck_list[:req.n_removed])
        play_deck = deck_list[req.n_removed:]
    else:
        removed = set()
        play_deck = deck_list

    from .engine import initial_state
    state = initial_state(req.n_players, play_deck)
    deck_fs = frozenset(full_deck())

    human_seat = req.human_seat
    if human_seat is None:
        human_seat = rng.randrange(req.n_players)
    if not 0 <= human_seat < req.n_players:
        raise HTTPException(400, "human_seat out of range")

    net_path = _DEFAULT_NET if req.n_players == 3 else None
    bots = _make_bots(req.n_players, human_seat, req.n_iter, net_path, seed)

    session_id = str(uuid.uuid4())
    session = GameSession(
        state=state,
        n_removed=req.n_removed,
        deck=deck_fs,
        human_seat=human_seat,
        bots=bots,
        rng=rng,
    )
    session.log.append({"event": "game_start", "n_players": req.n_players,
                         "human_seat": human_seat, "n_removed": req.n_removed})
    _sessions[session_id] = session

    # AI moves before the human's first turn (if human isn't seat 0)
    _run_ai_turns(session)

    data = _state_to_dict(session)
    data["session_id"] = session_id
    return data


@app.post("/api/action")
def api_action(req: ActionRequest) -> dict:
    session = _sessions.get(req.session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    if is_terminal(session.state):
        raise HTTPException(400, "game is already over")
    if session.state.to_move != session.human_seat:
        raise HTTPException(400, "not the human's turn")

    info = info_from_state(session.state, session.n_removed, session.deck)
    acts = legal_actions(info)
    if req.action not in acts:
        raise HTTPException(400, f"illegal action {req.action!r}; legal: {acts}")

    card = session.state.active
    pot_before = session.state.pot
    session.state = step(session.state, req.action, session.rng)

    evt: dict[str, Any] = {
        "seat": session.human_seat,
        "action": req.action,
        "card": card,
        "is_human": True,
    }
    if req.action == "take":
        evt["chips_gained"] = pot_before
    session.log.append(evt)

    # Advance AI turns
    _run_ai_turns(session)

    data = _state_to_dict(session)
    data["session_id"] = req.session_id
    return data


@app.get("/api/state/{session_id}")
def api_state(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    data = _state_to_dict(session)
    data["session_id"] = session_id
    return data


# --------------------------------------------------------------------------- #
# Static file serving
# --------------------------------------------------------------------------- #

_STATIC = _HERE / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(str(_STATIC / "index.html"))


def main():
    import uvicorn
    uvicorn.run("nothanks.web:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
