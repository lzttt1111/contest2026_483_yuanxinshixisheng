from __future__ import annotations

import run


def test_full_twelve_word_gate_is_order_independent_and_exact() -> None:
    reversed_full = tuple(reversed(run.ALL_TWELVE_ALGORITHMS))
    assert run.is_full_twelve_selection(reversed_full)
    assert not run.is_full_twelve_selection(reversed_full[:-1])
    assert not run.is_full_twelve_selection(
        (*reversed_full[:-1], reversed_full[0])
    )
