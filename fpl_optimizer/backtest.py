"""Backtesting: validate the expected-points model against gameweeks that
have already been played.

For a past gameweek `gw`, predictions are rebuilt using only the data that
would genuinely have been available beforehand — match history with
`round < gw`, and team strength ratings computed only from fixtures with
`event < gw` — then compared against what actually happened in `gw`. Two
views:

  - point-level accuracy: correlation and error between predicted and actual
    points across every player, benchmarked against a naive baseline (each
    player's own season-to-date average) to check the model is actually
    adding information, not just restating recent form.
  - squad-level validation: what the optimizer's recommended squad (built
    from pre-gameweek predictions) would actually have scored, compared to
    the average and highest scores real FPL managers achieved that
    gameweek (both reported directly by the API).

Two approximations, both noted in the results: today's prices stand in for
the historical price at the time (the API doesn't expose price history),
and today's injury/availability status stands in for the status at the
time (retroactive availability isn't exposed either).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import ScoringRules, SquadRules
from .features import (
    compute_player_form,
    compute_team_strengths,
    expected_minutes,
    opponent_history_modifier,
    position_rate_priors,
    shrink_form,
)
from .optimize import Squad, build_squad
from .predict import _fixture_points


def _actual_points_for_gw(history: list[dict], gw: int) -> float:
    return float(sum(h["total_points"] for h in history if h.get("round") == gw))


def _naive_prediction(prior_history: list[dict]) -> float:
    """Baseline: this player's own season-to-date average points per match,
    with no fixture, form-recency, or opponent information at all."""
    played = [h for h in prior_history if h.get("minutes", 0) > 0]
    if not played:
        return 0.0
    return float(sum(h["total_points"] for h in played) / len(played))


def predict_as_of(
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    summaries: dict[int, dict],
    scoring_rules: ScoringRules,
    gw: int,
) -> pd.DataFrame:
    """The same model as predict.predict, rebuilt strictly from information
    available before `gw`. Adds predicted_points, naive_points and (when
    known) actual_points columns to a copy of players_df."""
    prior_fixtures = fixtures_df[fixtures_df["event"] < gw]
    team_strength = compute_team_strengths(prior_fixtures, teams_df)
    this_gw_fixtures = fixtures_df[fixtures_df["event"] == gw]

    prior_histories: dict[int, list[dict]] = {}
    raw_forms = {}
    for pid in players_df.index:
        history = summaries.get(pid, {}).get("history", [])
        prior_history = [h for h in history if h.get("round", 0) < gw]
        prior_histories[pid] = prior_history
        raw_forms[pid] = compute_player_form(prior_history)

    forms_by_position: dict[str, list] = {}
    for pid, form in raw_forms.items():
        if form.games_seen > 0:
            forms_by_position.setdefault(players_df.loc[pid, "position"], []).append(form)
    priors = position_rate_priors(forms_by_position)

    predicted_points, naive_points, actual_points = [], [], []
    for pid, row in players_df.iterrows():
        prior_history = prior_histories[pid]
        form = shrink_form(raw_forms[pid], priors[row["position"]])
        # retroactive injury/availability status isn't exposed by the API;
        # assume fully fit and let recent minutes drive the estimate instead
        minutes = expected_minutes(form, "a", None)

        team_fixtures = this_gw_fixtures[
            (this_gw_fixtures["team_h"] == row["team"]) | (this_gw_fixtures["team_a"] == row["team"])
        ]
        total_pts = 0.0
        for _, fx in team_fixtures.iterrows():
            is_home = fx["team_h"] == row["team"]
            opponent_id = fx["team_a"] if is_home else fx["team_h"]
            opp_modifier, _ = opponent_history_modifier(prior_history, opponent_id)
            pts, _ = _fixture_points(
                position=row["position"],
                form=form,
                scoring=scoring_rules,
                team_strength=team_strength,
                team_id=row["team"],
                opponent_id=opponent_id,
                is_home=is_home,
                minutes=minutes,
                opp_modifier=opp_modifier,
            )
            total_pts += pts

        full_history = summaries.get(pid, {}).get("history", [])
        predicted_points.append(round(total_pts, 3))
        naive_points.append(round(_naive_prediction(prior_history), 3))
        actual_points.append(_actual_points_for_gw(full_history, gw))

    out = players_df.copy()
    out["predicted_points"] = predicted_points
    out["naive_points"] = naive_points
    out["actual_points"] = actual_points
    out["expected_points"] = out["predicted_points"]  # so build_squad can consume it directly
    return out


@dataclass
class GameweekBacktest:
    gameweek: int
    predictions: pd.DataFrame
    pearson_r: float
    spearman_r: float
    mae: float
    naive_pearson_r: float
    naive_mae: float
    squad: Squad
    squad_actual_points: float
    average_entry_score: int | None
    highest_score: int | None


def backtest_gameweek(
    bootstrap: dict,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    summaries: dict[int, dict],
    scoring_rules: ScoringRules,
    rules: SquadRules,
    gw: int,
) -> GameweekBacktest:
    predicted = predict_as_of(players_df, teams_df, fixtures_df, summaries, scoring_rules, gw)

    pearson_r = float(predicted["predicted_points"].corr(predicted["actual_points"]))
    # Spearman rank correlation without a scipy dependency: Pearson's r on the ranks
    spearman_r = float(predicted["predicted_points"].rank().corr(predicted["actual_points"].rank()))
    mae = float((predicted["predicted_points"] - predicted["actual_points"]).abs().mean())
    naive_pearson_r = float(predicted["naive_points"].corr(predicted["actual_points"]))
    naive_mae = float((predicted["naive_points"] - predicted["actual_points"]).abs().mean())

    squad = build_squad(predicted, rules)
    squad_actual = sum(predicted.loc[i, "actual_points"] for i in squad.starting_ids)
    squad_actual += predicted.loc[squad.captain_id, "actual_points"]  # captain doubles in real scoring too

    event = next((e for e in bootstrap["events"] if e["id"] == gw), None)

    return GameweekBacktest(
        gameweek=gw,
        predictions=predicted[["name", "position", "team_name", "predicted_points", "naive_points", "actual_points"]],
        pearson_r=pearson_r,
        spearman_r=spearman_r,
        mae=mae,
        naive_pearson_r=naive_pearson_r,
        naive_mae=naive_mae,
        squad=squad,
        squad_actual_points=round(squad_actual, 1),
        average_entry_score=event["average_entry_score"] if event else None,
        highest_score=event["highest_score"] if event else None,
    )


def finished_gameweeks(bootstrap: dict) -> list[int]:
    return sorted(e["id"] for e in bootstrap["events"] if e["finished"])
