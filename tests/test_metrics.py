from loghound_agent_debug.evaluate import summarize_rows


def test_normalized_rank_uses_zero_based_numerator():
    summary = summarize_rows([{
        "steps": 5, "first_root_rank": 3, "hit_at_1": 0, "hit_at_3": 1, "hit_at_5": 1,
        "mrr": 1 / 3, "normalized_rank": (3 - 1) / (5 - 1), "map": 1 / 3,
    }])
    assert summary["Mean NR"] == 0.5
    assert summary["n"] == 1
