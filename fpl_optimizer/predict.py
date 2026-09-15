"""The expected-points (xP) model.

For every player and every fixture they have in the requested gameweek
window, estimate FPL points from first principles:

  appearance   - from expected minutes
  goals        - blend of recent actual goals and underlying xG, scaled by
                 how leaky the opponent's defence has been and home/away
  assists      - same idea, using xA
  clean sheet  - Poisson P(0 conceded), team-level, needs no individual data
  goals conceded (GK/DEF penalty) - expected value of the -1-per-2 rule
  saves        - GK only, recent saves rate
  defensive contribution - Poisson P(actions >= threshold), new 2025/26+ rule
  bonus        - rough proxy from recent BPS rate
  cards        - small expected deduction from recent booking rate
  opponent history - a small nudge from how this player has fared against
                 this specific opponent this season (heavily shrunk)

Everything is returned with its breakdown so results are explainable rather
than a black-box number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import ScoringRules
from .features import (
    PlayerForm,
    TeamStrength,
    compute_player_form,
    compute_team_strengths,
    expected_minutes,
    opponent_history_modifier,
    position_rate_priors,
    shrink_form,
)

GOALS_WEIGHT_ON_XG = 0.6  # vs. weight on actual recent goals/assists rate
BONUS_BPS_BASELINE = 20.0  # recent-form bps/90 above which bonus points start to appear
BONUS_SCALE = 0.045  # points of expected bonus per bps/90 above the baseline
BONUS_CAP = 3.0


def _poisson_sf(threshold: int, mean: float) -> float:
    """P(X >= threshold) for X ~ Poisson(mean), computed directly since
    thresholds are small (10-12) and this avoids a scipy dependency."""
    if threshold <= 0:
        return 1.0
    if mean <= 0:
        return 0.0
    term = math.exp(-mean)  # P(X=0)
    cdf = term  # cdf accumulates P(X < threshold) = sum_{k=0}^{threshold-1} P(X=k)
    for k in range(1, threshold):
        term *= mean / k
        cdf += term
    return min(1.0, max(0.0, 1.0 - cdf))


@dataclass
class FixturePrediction:
    player_id: int
    gameweek: int
    opponent_team: int
    is_home: bool
    expected_minutes: float
    expected_points: float
    breakdown: dict[str, float] = field(default_factory=dict)


def _fixture_points(
    position: str,
    form: PlayerForm,
    scoring: ScoringRules,
    team_strength: TeamStrength,
    team_id: int,
    opponent_id: int,
    is_home: bool,
    minutes: float,
    opp_modifier: float,
) -> tuple[float, dict[str, float]]:
    home_baseline = team_strength.league_avg_home_goals
    away_baseline = team_strength.league_avg_away_goals
    mid_baseline = (home_baseline + away_baseline) / 2.0
    home_factor = home_baseline / mid_baseline
    away_factor = away_baseline / mid_baseline
    venue_factor = home_factor if is_home else away_factor

    minutes_share = minutes / 90.0
    opponent_defence_leakiness = team_strength.defence[opponent_id]  # >1 = concedes more than average
    attack_multiplier = opponent_defence_leakiness * venue_factor * opp_modifier

    goals_rate = GOALS_WEIGHT_ON_XG * form.xg_per_90 + (1 - GOALS_WEIGHT_ON_XG) * form.goals_per_90
    assists_rate = GOALS_WEIGHT_ON_XG * form.xa_per_90 + (1 - GOALS_WEIGHT_ON_XG) * form.assists_per_90
    expected_goals = goals_rate * minutes_share * attack_multiplier
    expected_assists = assists_rate * minutes_share * attack_multiplier

    # team-level: how many goals does THIS team concede against THIS opponent
    opponent_attack = team_strength.attack[opponent_id]
    own_defence_leakiness = team_strength.defence[team_id]
    opponent_baseline = away_baseline if is_home else home_baseline
    expected_goals_conceded = opponent_baseline * opponent_attack * own_defence_leakiness
    p_clean_sheet = math.exp(-expected_goals_conceded)

    appearance_pts = scoring.long_play * min(minutes / 60.0, 1.0) if minutes > 0 else 0.0
    goal_pts = expected_goals * scoring.goals_scored.get(position, 0)
    assist_pts = expected_assists * scoring.assists
    cs_pts = p_clean_sheet * scoring.clean_sheets.get(position, 0) if minutes >= 60 else 0.0
    conceded_pts = (expected_goals_conceded / 2.0) * scoring.goals_conceded.get(position, 0) * minutes_share

    save_pts = 0.0
    if position == "GKP":
        expected_saves = form.saves_per_90 * minutes_share
        save_pts = (expected_saves / 3.0) * 1.0  # 1 pt per 3 saves

    threshold = scoring.defensive_contribution_threshold.get(position, 999999)
    expected_actions = form.defensive_actions_per_90 * minutes_share
    p_hit_threshold = _poisson_sf(threshold, expected_actions) if expected_actions > 0 else 0.0
    dc_pts = p_hit_threshold * scoring.defensive_contribution.get(position, 0)

    expected_bps = form.bps_per_90 * minutes_share
    bonus_pts = min(BONUS_CAP, max(0.0, expected_bps - BONUS_BPS_BASELINE) * BONUS_SCALE)

    card_pts = -(form.yellow_per_90 * minutes_share) * abs(scoring.yellow_cards)
    card_pts += -(form.red_per_90 * minutes_share) * abs(scoring.red_cards)

    total = appearance_pts + goal_pts + assist_pts + cs_pts + conceded_pts + save_pts + dc_pts + bonus_pts + card_pts

    breakdown = {
        "appearance": round(appearance_pts, 3),
        "goals": round(goal_pts, 3),
        "assists": round(assist_pts, 3),
        "clean_sheet": round(cs_pts, 3),
        "goals_conceded": round(conceded_pts, 3),
        "saves": round(save_pts, 3),
        "defensive_contribution": round(dc_pts, 3),
        "bonus": round(bonus_pts, 3),
        "cards": round(card_pts, 3),
        "expected_goals": round(expected_goals, 3),
        "expected_assists": round(expected_assists, 3),
        "p_clean_sheet": round(p_clean_sheet, 3),
        "opponent_history_modifier": round(opp_modifier, 3),
    }
    return total, breakdown


def predict(
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    summaries: dict[int, dict],
    scoring_rules: ScoringRules,
    start_gw: int,
    num_gws: int = 1,
) -> pd.DataFrame:
    """Expected points for every player, summed over `num_gws` gameweeks
    starting at `start_gw` (naturally handles blank/double gameweeks, since a
    player's `fixtures` list may contain 0, 1 or 2 entries per gameweek).

    Returns players_df with `expected_points` and a `breakdown` column (list
    of per-fixture dicts) added.
    """
    team_strength = compute_team_strengths(fixtures_df, teams_df)
    target_gws = set(range(start_gw, start_gw + num_gws))

    raw_forms: dict[int, PlayerForm] = {
        pid: compute_player_form(summaries.get(pid, {}).get("history", [])) for pid in players_df.index
    }
    forms_by_position: dict[str, list[PlayerForm]] = {}
    for pid, form in raw_forms.items():
        if form.games_seen > 0:
            forms_by_position.setdefault(players_df.loc[pid, "position"], []).append(form)
    priors = position_rate_priors(forms_by_position)

    exp_points = []
    breakdowns = []
    exp_minutes_col = []
    for pid, row in players_df.iterrows():
        summary = summaries.get(pid, {})
        history = summary.get("history", [])
        upcoming = summary.get("fixtures", [])
        form = shrink_form(raw_forms[pid], priors[row["position"]])
        minutes = expected_minutes(form, row["status"], row["chance_of_playing_next_round"])

        relevant = [f for f in upcoming if f.get("event") in target_gws]
        total_pts = 0.0
        fixture_breakdowns = []
        for fx in relevant:
            is_home = bool(fx["is_home"])
            opponent_id = fx["team_a"] if is_home else fx["team_h"]
            opp_modifier, n_h2h = opponent_history_modifier(history, opponent_id)
            pts, breakdown = _fixture_points(
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
            breakdown["gameweek"] = fx["event"]
            breakdown["opponent"] = teams_df.loc[opponent_id, "short_name"] if opponent_id in teams_df.index else "?"
            breakdown["is_home"] = is_home
            breakdown["h2h_sample_size"] = n_h2h
            total_pts += pts
            fixture_breakdowns.append(breakdown)

        exp_points.append(round(total_pts, 3))
        breakdowns.append(fixture_breakdowns)
        exp_minutes_col.append(round(minutes, 1))

    out = players_df.copy()
    out["expected_points"] = exp_points
    out["expected_minutes"] = exp_minutes_col
    out["breakdown"] = breakdowns
    out["num_fixtures"] = out["breakdown"].apply(len)
    return out
