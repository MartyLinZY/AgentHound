"""Tests for aggregate ranking metrics."""

from loghound_agent_debug.evaluate import rows_from_results, summarize_rows
from loghound_agent_debug.schema import Label


def test_step_summary_only_reports_top_k_and_mrr():
    summary = summarize_rows([{
        "steps": 5, "hit_at_1": 0, "hit_at_3": 1, "hit_at_5": 1, "hit_at_10": 1,
        "mrr": 1 / 3,
    }])
    assert summary["n"] == 1
    assert summary["step_level"] == {
        "Hit@1": 0,
        "Hit@3": 1,
        "Hit@5": 1,
        "Hit@10": 1,
        "MRR": 1 / 3,
    }
    assert "Mean Rank" not in summary
    assert "Median Rank" not in summary


def test_agent_level_hit_uses_distinct_agent_ranking():
    rankings = [{
        "trajectory_id": "t1",
        "ranking": [
            {"rank": 1, "source_step_id": "1", "agent": "distractor"},
            {"rank": 2, "source_step_id": "2", "agent": "distractor"},
            {"rank": 3, "source_step_id": "3", "agent": "root-agent"},
        ],
    }]
    labels = {"t1": Label("t1", ["3"], ["root-agent"])}

    rows = rows_from_results(rankings, labels)
    summary = summarize_rows(rows)

    assert rows[0]["predicted_agent"] == "distractor"
    assert rows[0]["agent_hit_at_1"] == 0
    assert rows[0]["agent_hit_at_3"] == 1
    assert rows[0]["agent_mrr"] == 1 / 2
    assert rows[0]["hit_at_1"] == 0
    assert rows[0]["hit_at_3"] == 1
    assert "first_root_rank" not in rows[0]
    assert "first_agent_rank" not in rows[0]
    assert summary["agent_level"]["Hit@1"] == 0
    assert summary["agent_level"]["Hit@3"] == 1
    assert summary["agent_level"]["MRR"] == 1 / 2
    assert summary["step_level"]["Hit@3"] == 1
