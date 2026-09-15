from __future__ import annotations

from fpl_optimizer.backtest import _actual_points_for_gw, _naive_prediction


def test_actual_points_for_gw_sums_matching_rounds_only():
    history = [
        {"round": 3, "total_points": 5},
        {"round": 4, "total_points": 2},
        {"round": 4, "total_points": 9},  # a double gameweek: two fixtures, same round
    ]
    assert _actual_points_for_gw(history, gw=4) == 11
    assert _actual_points_for_gw(history, gw=3) == 5
    assert _actual_points_for_gw(history, gw=5) == 0


def test_naive_prediction_is_average_points_per_played_match():
    history = [
        {"minutes": 90, "total_points": 6},
        {"minutes": 90, "total_points": 2},
        {"minutes": 0, "total_points": 0},  # unused sub appearance, excluded
    ]
    assert _naive_prediction(history) == 4.0


def test_naive_prediction_with_no_history_is_zero():
    assert _naive_prediction([]) == 0.0
