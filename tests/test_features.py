from __future__ import annotations

from fpl_optimizer.features import (
    PlayerForm,
    compute_player_form,
    opponent_history_modifier,
    position_rate_priors,
    shrink_form,
)
from fpl_optimizer.predict import _poisson_sf


def _history_entry(minutes=90, goals=0, assists=0, xg=0.0, xa=0.0, opponent_team=1, bps=20):
    return {
        "minutes": minutes,
        "goals_scored": goals,
        "assists": assists,
        "expected_goals": xg,
        "expected_assists": xa,
        "bps": bps,
        "tackles": 1,
        "clearances_blocks_interceptions": 1,
        "recoveries": 1,
        "saves": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "opponent_team": opponent_team,
        "total_points": 2,
    }


def test_compute_player_form_no_minutes_returns_zeroed_form():
    form = compute_player_form([])
    assert form.games_seen == 0
    assert form.goals_per_90 == 0


def test_compute_player_form_extrapolates_from_a_single_fluke_match():
    # one goal in 15 minutes -> a naive per-90 rate would be wildly high;
    # compute_player_form itself doesn't shrink (that's shrink_form's job),
    # so this documents the raw (unshrunk) number is indeed extreme.
    history = [_history_entry(minutes=15, goals=1, xg=0.8)]
    form = compute_player_form(history)
    assert form.goals_per_90 > 5


def test_shrink_form_pulls_small_samples_toward_the_prior():
    small_sample = compute_player_form([_history_entry(minutes=15, goals=1, xg=0.8)])
    prior = {
        "goals_per_90": 0.3,
        "assists_per_90": 0.2,
        "xg_per_90": 0.25,
        "xa_per_90": 0.15,
        "bps_per_90": 20.0,
        "defensive_actions_per_90": 6.0,
        "saves_per_90": 0.0,
        "yellow_per_90": 0.1,
        "red_per_90": 0.01,
    }
    shrunk = shrink_form(small_sample, prior, k_minutes=270.0)
    assert shrunk.goals_per_90 < small_sample.goals_per_90
    assert shrunk.goals_per_90 > prior["goals_per_90"]

    # a long run of matches should be shrunk far less than a single fluke game
    big_sample_history = [_history_entry(minutes=90, goals=1, xg=0.5) for _ in range(30)]
    big_sample = compute_player_form(big_sample_history)
    shrunk_big = shrink_form(big_sample, prior, k_minutes=270.0)
    small_shrinkage = small_sample.goals_per_90 - shrunk.goals_per_90
    big_shrinkage = big_sample.goals_per_90 - shrunk_big.goals_per_90
    assert big_shrinkage < small_shrinkage


def test_position_rate_priors_is_minutes_weighted_average():
    heavy = compute_player_form([_history_entry(minutes=90, goals=1) for _ in range(10)])
    light = compute_player_form([_history_entry(minutes=5, goals=0)])
    priors = position_rate_priors({"FWD": [heavy, light]})
    # dominated by `heavy` since it carries far more sample_minutes
    assert priors["FWD"]["goals_per_90"] > 0.5


def test_opponent_history_modifier_shrinks_small_samples_toward_one():
    history = [_history_entry(opponent_team=99, minutes=90)] + [{**_history_entry(minutes=90), "total_points": 2}]
    history[0]["total_points"] = 20  # one big haul against this specific opponent
    modifier, n = opponent_history_modifier(history, opponent_team_id=99)
    assert n == 1
    assert modifier > 1.0
    assert modifier <= 1.15  # capped


def test_opponent_history_modifier_no_meetings_is_neutral():
    history = [_history_entry(opponent_team=5, minutes=90)]
    modifier, n = opponent_history_modifier(history, opponent_team_id=123)
    assert modifier == 1.0
    assert n == 0


def test_poisson_sf_monotonic_and_bounded():
    assert _poisson_sf(0, mean=2.0) == 1.0  # P(X >= 0) is always 1
    assert 0.0 <= _poisson_sf(10, mean=2.0) <= 1.0
    # higher mean -> higher chance of reaching the same threshold
    assert _poisson_sf(10, mean=5.0) > _poisson_sf(10, mean=2.0)
    assert _poisson_sf(10, mean=0.0) == 0.0
