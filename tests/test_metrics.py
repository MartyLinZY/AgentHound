"""Tests for aggregate ranking metrics."""

from loghound_agent_debug.evaluate import summarize_rows


def test_mean_and_median_rank_are_reported():
    # A one-row sample makes both statistics unambiguous and easy to audit.
    summary = summarize_rows([{
        "steps": 5, "first_root_rank": 3, "hit_at_1": 0, "hit_at_3": 1, "hit_at_5": 1,
        "mrr": 1 / 3,
    }])
    assert summary["Mean Rank"] == 3
    assert summary["Median Rank"] == 3
    assert summary["n"] == 1
