from backend.evals.run_quality_eval import _aggregate, _fact_equal, _pct


def test_percentage_and_fact_comparison_are_stable():
    assert _pct(3, 4) == 75.0
    assert _pct(0, 0) == 0.0
    assert _fact_equal("bpm", 120.0000001, 120.0)
    assert not _fact_equal("title", "Signal", "signal")


def test_aggregate_uses_micro_averages_and_minimum_vector_count():
    rounds = [
        {
            "vector_audit": {"valid_vectorized_tracks": 10, "integrity_rate_pct": 100.0},
            "intent": {"passed": 4, "total": 5, "accuracy_pct": 80.0},
            "tools": {"passed": 3, "total": 4},
            "rag": {"hits_at_3": 2, "queries": 3, "facts_verified": 20, "facts_checked": 24},
        },
        {
            "vector_audit": {"valid_vectorized_tracks": 9, "integrity_rate_pct": 90.0},
            "intent": {"passed": 5, "total": 5, "accuracy_pct": 100.0},
            "tools": {"passed": 4, "total": 4},
            "rag": {"hits_at_3": 3, "queries": 3, "facts_verified": 24, "facts_checked": 24},
        },
    ]
    result = _aggregate(rounds)
    assert result["vectorized_tracks"] == 9
    assert result["intent_accuracy_pct"] == 90.0
    assert result["tool_success_rate_pct"] == 87.5
    assert result["rag_accuracy_pct"] == 83.33
    assert result["rag_truthfulness_pct"] == 91.67
