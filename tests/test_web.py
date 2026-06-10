"""Web API: live analysis in play mode and the advisor (relayed real game).

Needs the optional web deps; run with
``uv run --group web --with httpx pytest tests/test_web.py``
(skipped under the plain dev environment).
"""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from nothanks.web import app  # noqa: E402

client = TestClient(app)

FAST = {"n_iter": 16}  # tiny searches keep the suite quick


def _new_game(**over):
    body = {"n_players": 3, "n_removed": 9, "seed": 7, "human_seat": 0, **FAST, **over}
    resp = client.post("/api/new_game", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_play_mode_analysis_on_human_turn():
    data = _new_game(hints=True)
    assert data["is_human_turn"]
    a = data["analysis"]
    assert a is not None
    # Opening position: both actions legal, EVs + visits + a per-seat projection.
    assert set(a["mover_ev"]) == {"take", "pass"}
    assert a["best_action"] in ("take", "pass")
    assert len(a["proj_scores"]) == 3
    assert sum(a["visits"].values()) > 0


def test_play_mode_hints_off_sends_no_analysis():
    data = _new_game(hints=False)
    assert data["analysis"] is None


def _new_advisor(**over):
    body = {"n_players": 3, "n_removed": 9, "advised_seat": 0,
            "first_card": 26, **FAST, **over}
    resp = client.post("/api/advisor/new", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_advisor_relays_a_real_game():
    data = _new_advisor()
    sid = data["session_id"]
    assert data["active_card"] == 26
    assert data["to_move"] == 0
    assert data["analysis"]["best_action"] in ("take", "pass")
    assert 26 not in data["unseen"]

    # Seat 0 passes: deterministic public move, turn advances, pot grows.
    data = client.post("/api/advisor/move",
                       json={"session_id": sid, "action": "pass"}).json()
    assert data["to_move"] == 1
    assert data["pot"] == 1
    assert data["players"][0]["chips"] == 10

    # Seat 1 takes; the real game flipped card 17 next. Taker keeps the turn.
    data = client.post("/api/advisor/move",
                       json={"session_id": sid, "action": "take", "next_card": 17}).json()
    assert data["to_move"] == 1
    assert data["players"][1]["cards"] == [26]
    assert data["players"][1]["chips"] == 12  # 11 + the 1-chip pot
    assert data["active_card"] == 17
    assert data["can_undo"]


def test_advisor_take_requires_a_valid_next_card():
    sid = _new_advisor()["session_id"]
    resp = client.post("/api/advisor/move",
                       json={"session_id": sid, "action": "take"})
    assert resp.status_code == 400
    assert "next_card" in resp.json()["detail"]
    # The first card is face-up, so it cannot be flipped again.
    resp = client.post("/api/advisor/move",
                       json={"session_id": sid, "action": "take", "next_card": 26})
    assert resp.status_code == 400


def test_advisor_undo_restores_the_previous_position():
    sid = _new_advisor()["session_id"]
    client.post("/api/advisor/move", json={"session_id": sid, "action": "pass"})
    data = client.post("/api/advisor/undo", json={"session_id": sid}).json()
    assert data["to_move"] == 0
    assert data["pot"] == 0
    assert not data["can_undo"]  # back at the opening position


def test_advisor_rejects_bad_setup():
    assert client.post("/api/advisor/new",
                       json={"first_card": 2, **FAST}).status_code == 400
    assert client.post("/api/advisor/new",
                       json={"first_card": 26, "advised_seat": 5, **FAST}).status_code == 400
