"""A tiny NumPy MLP value function and the fast V-based move eval.

``ValueNet`` maps a mover-relative feature vector (see :mod:`nothanks.features`)
to a **vector** of expected final scores — one per seat, seat 0 being the mover,
opponents following in turn order. The vector output is what makes multiplayer
lookahead consistent: after a ``take`` the mover keeps the turn, so the successor
is already in the mover's perspective; after a ``pass`` the turn advances one
seat, so we rotate the successor's vector by one (``np.roll(v, 1)``) to read it
back in the mover's frame. A single scalar "mover value" could not be rotated
like that, which is why the head is vector-valued.

``evaluate_v`` is then the engine eval — same shape as
:func:`nothanks.solver.evaluate` and :func:`nothanks.montecarlo.evaluate_mc` —
but it reads move EV in *one forward pass per successor* instead of rolling out,
which is the whole point of step 3.

The net is a 2-layer MLP (tanh hidden, linear head) with hand-written backprop;
no autograd dependency. Training lives in :mod:`nothanks.train`.
"""

from __future__ import annotations

import numpy as np

from .engine import (
    State,
    apply_pass,
    final_scores,
    is_terminal,
    legal_actions,
    take_outcomes,
)
from .features import feature_dim, features, seat_order


class ValueNet:
    """2-layer MLP: features -> (n_players,) expected final-score vector."""

    _PARAMS = ("W1", "b1", "W2", "b2")

    def __init__(self, n_players: int, hidden: int = 64, seed: int = 0):
        self.n_players = n_players
        self.hidden = hidden
        d = feature_dim(n_players)
        rng = np.random.default_rng(seed)
        # He-ish init for the tanh layer; small head.
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), size=(hidden, d)).astype(np.float64)
        self.b1 = np.zeros(hidden)
        self.W2 = (rng.normal(0, 0.1, size=(n_players, hidden))).astype(np.float64)
        self.b2 = np.zeros(n_players)
        self._reset_adam()

    def _reset_adam(self) -> None:
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._PARAMS}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._PARAMS}
        self._t = 0

    # -- forward / backward ------------------------------------------------- #

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (B, D) -> Y: (B, n_players). Caches activations for ``backward``."""
        z1 = X @ self.W1.T + self.b1
        a1 = np.tanh(z1)
        Y = a1 @ self.W2.T + self.b2
        self._cache = (X, a1)
        return Y

    def backward(self, dY: np.ndarray) -> dict:
        """Gradients of a scalar loss given dL/dY (same shape as Y)."""
        X, a1 = self._cache
        gW2 = dY.T @ a1
        gb2 = dY.sum(axis=0)
        da1 = dY @ self.W2
        dz1 = da1 * (1.0 - a1 * a1)
        gW1 = dz1.T @ X
        gb1 = dz1.sum(axis=0)
        return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

    def train_step(
        self,
        X: np.ndarray,
        T: np.ndarray,
        lr: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> float:
        """One Adam MSE step on batch (X, T). Returns mean per-sample loss."""
        Y = self.forward(X)
        B = X.shape[0]
        resid = Y - T
        loss = 0.5 * float((resid * resid).sum()) / B
        grads = self.backward(resid / B)

        b1, b2 = betas
        self._t += 1
        for name in self._PARAMS:
            g = grads[name]
            m = self._m[name] = b1 * self._m[name] + (1 - b1) * g
            v = self._v[name] = b2 * self._v[name] + (1 - b2) * g * g
            mhat = m / (1 - b1 ** self._t)
            vhat = v / (1 - b2 ** self._t)
            getattr(self, name)[...] -= lr * mhat / (np.sqrt(vhat) + eps)
        return loss

    # -- convenience -------------------------------------------------------- #

    def predict(self, s: State) -> np.ndarray:
        """Value vector for ``s`` in its own mover-relative perspective."""
        return self.forward(features(s)[None, :])[0]

    def save(self, path) -> None:
        np.savez(
            path,
            n_players=self.n_players,
            hidden=self.hidden,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
        )

    @classmethod
    def load(cls, path) -> "ValueNet":
        d = np.load(path)
        net = cls(int(d["n_players"]), int(d["hidden"]))
        net.W1, net.b1, net.W2, net.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        net._reset_adam()
        return net


# --------------------------------------------------------------------------- #
# V-based move evaluation (one-ply lookahead, no rollouts)
# --------------------------------------------------------------------------- #

def action_values(s: State, net: ValueNet) -> dict[str, np.ndarray]:
    """Expected-score vector per legal action, in ``s``'s mover-relative frame."""
    p = s.to_move
    n = s.n_players
    out: dict[str, np.ndarray] = {}
    for action in legal_actions(s):
        if action == "pass":
            sp = apply_pass(s)            # turn advances to seat 1 (p+1)
            out[action] = np.roll(net.predict(sp), 1)  # rotate back to p's frame
        else:  # take: mover keeps the turn, so perspective is unchanged
            acc = np.zeros(n)
            for prob, nxt in take_outcomes(s):
                if is_terminal(nxt):
                    fs = final_scores(nxt)
                    vec = np.array([fs[q] for q in seat_order(p, n)], dtype=float)
                else:
                    vec = net.predict(nxt)
                acc += prob * vec
            out[action] = acc
    return out


def evaluate_v(s: State, net: ValueNet) -> dict:
    """Engine-style move eval from the value net (mover is seat 0)."""
    av = action_values(s, net)
    best_action = min(av, key=lambda a: av[a][0])
    return {
        "to_move": s.to_move,
        "actions": {a: tuple(float(x) for x in v) for a, v in av.items()},
        "mover_ev": {a: float(v[0]) for a, v in av.items()},
        "best_action": best_action,
    }


def greedy_action(s: State, net: ValueNet) -> str:
    """Action minimising the mover's own predicted final score."""
    av = action_values(s, net)
    return min(av, key=lambda a: av[a][0])
