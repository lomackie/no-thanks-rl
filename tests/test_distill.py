"""Policy distillation (nothanks.distill).

The fitter must recover a policy that *is* a threshold rule exactly — the
heuristic (take iff cost ≤ 0) distils to T=0 with 100% agreement — and the
report machinery must partition decisions without losing any.
"""

from nothanks.belief import make_heuristic_policy
from nothanks.distill import (
    agreement,
    collect_decisions,
    fit_threshold,
    threshold_table,
)

TINY = dict(deck=range(3, 12), n_removed=2, start_chips=4, n_games=60)


def test_heuristic_distils_to_its_own_threshold():
    ds = collect_decisions(make_heuristic_policy(0), seed=1, **TINY)
    assert ds  # the tiny games produce real (non-forced) decisions
    t, acc = fit_threshold(ds)
    assert t == 0
    assert acc == 1.0
    # And a shifted threshold rule is recovered too.
    ds3 = collect_decisions(make_heuristic_policy(3), seed=1, **TINY)
    t3, acc3 = fit_threshold(ds3)
    assert t3 == 3
    assert acc3 == 1.0


def test_decisions_exclude_forced_takes_and_carry_context():
    ds = collect_decisions(make_heuristic_policy(0), seed=2, **TINY)
    for d in ds:
        assert d.chips >= 1  # chipless moves are forced, not decisions
        assert d.action in ("take", "pass")
        # connects=True must coincide with a discounted take (delta < card).
        assert d.connects == (d.cost + d.pot < d.card)


def test_threshold_table_partitions_all_decisions():
    ds = collect_decisions(make_heuristic_policy(0), seed=3, **TINY)
    rows = threshold_table(ds, lambda d: "even" if d.card % 2 == 0 else "odd")
    assert sum(n for _, _, _, n in rows) == len(ds)
    for _, _, acc, _ in rows:
        # Every bucket of an exact threshold policy fits perfectly.
        assert acc == 1.0
    # agreement() is consistent with the fitted numbers on the whole set.
    assert agreement(ds, 0) == 1.0
