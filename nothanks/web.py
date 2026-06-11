"""Web UI server for playing No Thanks against the AI.

Run with `just play` (or `uv run python -m nothanks.web`).
Visit http://localhost:8000 in a browser.

Sessions are in-memory only; each browser tab gets its own game. The AI is
ISMCTSBot backed by the trained info-set net for the player count
(``models/info_net_{n}p.npz``, preferring a ``_v2`` repair when present) when
available, falling back to heuristic playouts.

Two modes:

* **Play** — a game against the bot, optionally with live analysis (per-move EV
  for the human and a projected final score per player, from one IS-MCTS search
  of the human's info set).
* **Advisor** — no referee at all: the user relays the moves of a *real* game
  (a take also needs the card that was actually flipped next) and the server
  tracks the public :class:`~nothanks.imperfect.InfoSet`, recommending a move
  with EVs at every decision. Undo supported, since real-game entry has typos.
"""

from __future__ import annotations

import os
import pathlib
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .belief import final_scores as info_final_scores
from .engine import (
    DECK_HIGH,
    DECK_LOW,
    STARTING_CHIPS,
    State,
    final_scores,
    full_deck,
    is_terminal,
    new_game,
    score_cards,
    step,
)
from .imperfect import InfoSet, info_from_state, legal_actions, pile_remaining, unseen
from .ismcts import ISMCTSBot, LeafEvaluator, ismcts_evaluate, make_value_leaf

from .beliefnet import default_net_path

_HERE = pathlib.Path(__file__).parent

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
    n_iter: int = 400
    leaf: LeafEvaluator | None = None
    hints: bool = True
    log: list[dict] = field(default_factory=list)


@dataclass
class AdvisorSession:
    """A relayed real game: just the public info-set history, no referee/RNG."""

    history: list[InfoSet]
    advised_seat: int
    n_iter: int = 400
    leaf: LeafEvaluator | None = None

    @property
    def info(self) -> InfoSet:
        return self.history[-1]


_sessions: dict[str, GameSession] = {}
_advisors: dict[str, AdvisorSession] = {}


def _load_leaf(n_players: int, net_path: pathlib.Path | None) -> LeafEvaluator | None:
    if net_path and net_path.exists():
        try:
            from .valuefn import ValueNet

            net = ValueNet.load(str(net_path))
            if net.n_players == n_players:
                return make_value_leaf(net)
        except Exception:
            pass
    return None


def _make_bots(
    n_players: int,
    human_seat: int,
    n_iter: int,
    evaluator: LeafEvaluator | None,
    seed: int,
) -> list[ISMCTSBot | None]:
    bots: list[ISMCTSBot | None] = []
    for seat in range(n_players):
        if seat == human_seat:
            bots.append(None)
        else:
            bots.append(ISMCTSBot(n_iter=n_iter, evaluator=evaluator, c=30.0,
                                  seed=seed + seat))
    return bots


def _analysis(info: InfoSet, leaf: LeafEvaluator | None, n_iter: int) -> dict | None:
    """One IS-MCTS search of the mover's info set, shaped for the UI.

    ``mover_ev`` is the mover's expected final score per action (lower is
    better); ``proj_scores`` is the per-seat expected final-score vector under
    the recommended action — the "strength of position" readout. Honest like
    everything else: the search sees only the info set.
    """
    if info.active is None:
        return None
    acts = legal_actions(info)
    if len(acts) == 1:
        return {"forced": acts[0], "best_action": acts[0],
                "mover_ev": {}, "visits": {}, "proj_scores": None}
    ev = ismcts_evaluate(info, n_iter=n_iter, evaluator=leaf, c=30.0,
                         rng=random.Random(0xA11CE))
    return {
        "forced": None,
        "best_action": ev["best_action"],
        "mover_ev": {a: round(v, 2) for a, v in ev["mover_ev"].items()},
        "visits": ev["visits"],
        "proj_scores": [round(x, 1) for x in ev["actions"][ev["best_action"]]],
    }


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
    # Live analysis on the human's decision only (their info set; no peeking).
    if (result["is_human_turn"] and session.hints):
        result["analysis"] = _analysis(info, session.leaf, session.n_iter)
    else:
        result["analysis"] = None
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
    hints: bool = True  # per-move EV + projected scores on human turns


class ActionRequest(BaseModel):
    session_id: str
    action: str  # "take" or "pass"


class AdvisorNewRequest(BaseModel):
    n_players: int = 3
    n_removed: int = 9
    advised_seat: int = 0
    first_card: int
    start_chips: int | None = None  # None = the rulebook count for n_players
    n_iter: int = 400


class AdvisorMoveRequest(BaseModel):
    session_id: str
    action: str  # what actually happened in the real game
    next_card: int | None = None  # the card flipped after a take (if any remain)


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

    leaf = _load_leaf(req.n_players, default_net_path(req.n_players))
    bots = _make_bots(req.n_players, human_seat, req.n_iter, leaf, seed)

    session_id = str(uuid.uuid4())
    session = GameSession(
        state=state,
        n_removed=req.n_removed,
        deck=deck_fs,
        human_seat=human_seat,
        bots=bots,
        rng=rng,
        n_iter=req.n_iter,
        leaf=leaf,
        hints=req.hints,
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
# Advisor mode: relay a real game, get recommendations
# --------------------------------------------------------------------------- #

def _advisor_take(info: InfoSet, next_card: int | None) -> InfoSet:
    """The mover takes the active card; the *known* next card is flipped.

    The belief-game counterpart samples the flip (uniform over unseen); here
    the user tells us which card actually appeared in their physical game.
    With no pile cards left the take ends the game.
    """
    p = info.to_move
    chips = list(info.chips)
    chips[p] += info.pot
    cards = list(info.cards)
    cards[p] = cards[p] | {info.active}
    nxt = dict(chips=tuple(chips), cards=tuple(cards), pot=0)
    if pile_remaining(info) == 0:
        return replace(info, active=None, **nxt)
    return replace(info, active=next_card, **nxt)  # taker keeps the turn


def _advisor_to_dict(session: AdvisorSession) -> dict[str, Any]:
    info = session.info
    n = info.n_players
    over = info.active is None

    players = []
    for seat in range(n):
        hand = sorted(info.cards[seat])
        card_score = score_cards(hand)
        players.append({
            "seat": seat,
            "is_human": seat == session.advised_seat,
            "chips": info.chips[seat],
            "cards": hand,
            "card_score": card_score,
            "score": card_score - info.chips[seat],
        })

    return {
        "session_id": None,  # filled by caller
        "mode": "advisor",
        "game_over": over,
        "human_seat": session.advised_seat,
        "to_move": info.to_move,
        "active_card": info.active,
        "pot": info.pot,
        "pile_remaining": pile_remaining(info) if not over else 0,
        "unseen": sorted(unseen(info)),  # the card picker's choices after a take
        "players": players,
        "legal_actions": list(legal_actions(info)),
        "final_scores": list(info_final_scores(info)) if over else None,
        "can_undo": len(session.history) > 1,
        "analysis": _analysis(info, session.leaf, session.n_iter),
    }


@app.post("/api/advisor/new")
def api_advisor_new(req: AdvisorNewRequest) -> dict:
    if not 3 <= req.n_players <= 7:
        raise HTTPException(400, "n_players must be 3..7")
    if not 0 <= req.advised_seat < req.n_players:
        raise HTTPException(400, "advised_seat out of range")
    deck_fs = frozenset(full_deck())
    if req.first_card not in deck_fs:
        raise HTTPException(400, f"first_card must be {DECK_LOW}..{DECK_HIGH}")
    if not 0 <= req.n_removed <= len(deck_fs) - 2:
        raise HTTPException(400, "n_removed out of range")
    chips = req.start_chips if req.start_chips is not None else STARTING_CHIPS[req.n_players]
    if chips < 1:
        raise HTTPException(400, "start_chips must be positive")

    info = InfoSet(
        chips=tuple(chips for _ in range(req.n_players)),
        cards=tuple(frozenset() for _ in range(req.n_players)),
        active=req.first_card,
        pot=0,
        to_move=0,
        deck=deck_fs,
        n_removed=req.n_removed,
    )
    session = AdvisorSession(history=[info], advised_seat=req.advised_seat,
                             n_iter=req.n_iter,
                             leaf=_load_leaf(req.n_players, default_net_path(req.n_players)))
    session_id = str(uuid.uuid4())
    _advisors[session_id] = session
    data = _advisor_to_dict(session)
    data["session_id"] = session_id
    return data


@app.post("/api/advisor/move")
def api_advisor_move(req: AdvisorMoveRequest) -> dict:
    session = _advisors.get(req.session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    info = session.info
    if info.active is None:
        raise HTTPException(400, "game is already over")
    acts = legal_actions(info)
    if req.action not in acts:
        raise HTTPException(400, f"illegal action {req.action!r}; legal: {acts}")

    if req.action == "pass":
        from .belief import apply_pass

        session.history.append(apply_pass(info))
    else:
        if pile_remaining(info) > 0:
            if req.next_card is None:
                raise HTTPException(400, "a take needs next_card (the card flipped after it)")
            if req.next_card not in unseen(info):
                raise HTTPException(
                    400, f"card {req.next_card} has already been seen (or is not in the deck)")
        session.history.append(_advisor_take(info, req.next_card))

    data = _advisor_to_dict(session)
    data["session_id"] = req.session_id
    return data


@app.post("/api/advisor/undo")
def api_advisor_undo(req: dict) -> dict:
    session = _advisors.get(req.get("session_id", ""))
    if session is None:
        raise HTTPException(404, "session not found")
    if len(session.history) > 1:
        session.history.pop()
    data = _advisor_to_dict(session)
    data["session_id"] = req["session_id"]
    return data


@app.get("/api/advisor/state/{session_id}")
def api_advisor_state(session_id: str) -> dict:
    session = _advisors.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    data = _advisor_to_dict(session)
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
