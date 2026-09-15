"""Feature engineering: turn raw history into the signals the points model
needs — team attack/defence strength, a player's recent underlying output,
expected minutes, and how a player has fared against this particular
opponent before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

RECENCY_HALF_LIFE_GAMES = 5.0  # a match this many games ago carries half the weight
MINUTES_LOOKBACK = 6  # how many recent appearances inform expected-minutes
SAMPLE_SHRINKAGE_GAMES = 6.0  # pseudo-games of "league average" blended into team ratings
OPPONENT_HISTORY_SHRINKAGE = 2.0  # pseudo-games of "no effect" blended into head-to-head modifier
OPPONENT_HISTORY_CAP = 0.15  # head-to-head modifier is clipped to +/-15%
RATE_SHRINKAGE_MINUTES = 270.0  # ~3 full matches of position-average blended into small samples

_RATE_FIELDS = (
    "goals_per_90",
    "assists_per_90",
    "xg_per_90",
    "xa_per_90",
    "bps_per_90",
    "defensive_actions_per_90",
    "saves_per_90",
    "yellow_per_90",
    "red_per_90",
)


def _recency_weights(n: int, half_life: float = RECENCY_HALF_LIFE_GAMES) -> np.ndarray:
    """Weight 1.0 for the most recent of n items, decaying by half every `half_life` games."""
    if n == 0:
        return np.array([])
    decay = math.log(2) / half_life
    ages = np.arange(n - 1, -1, -1)  # last item age=0 (most recent), first item age=n-1
    return np.exp(-decay * ages)


@dataclass(frozen=True)
class TeamStrength:
    attack: dict[int, float]  # team_id -> multiplier, 1.0 = league average attack
    defence: dict[int, float]  # team_id -> multiplier, 1.0 = league average defence (lower is better, so this is inverted: >1 = leaks more)
    league_avg_home_goals: float
    league_avg_away_goals: float
    games_played: dict[int, int]


def compute_team_strengths(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame) -> TeamStrength:
    """A simple Poisson-style attack/defence rating from this season's finished
    matches, recency-weighted and shrunk toward league average early in the
    season (when a handful of results would otherwise be over-trusted) using
    the FPL's own preseason `strength_overall_*` rating as the prior.
    """
    finished = fixtures_df[
        fixtures_df["finished"].fillna(False) & fixtures_df["team_h_score"].notna()
    ].sort_values("event")

    team_ids = list(teams_df["id"])
    overall = {
        tid: (teams_df.loc[tid, "strength_overall_home"] + teams_df.loc[tid, "strength_overall_away"]) / 2.0
        for tid in team_ids
    }
    avg_overall = float(np.mean(list(overall.values()))) if overall else 3.0
    avg_overall = avg_overall or 3.0

    if finished.empty:
        # no results yet: attack/defence entirely driven by the preseason prior
        attack = {tid: overall[tid] / avg_overall for tid in team_ids}
        defence = {tid: avg_overall / overall[tid] for tid in team_ids}
        return TeamStrength(attack, defence, 1.4, 1.1, {tid: 0 for tid in team_ids})

    n = len(finished)
    w = _recency_weights(n)
    home_scored = finished["team_h_score"].to_numpy(dtype=float)
    away_scored = finished["team_a_score"].to_numpy(dtype=float)
    league_avg_home_goals = float(np.average(home_scored, weights=w))
    league_avg_away_goals = float(np.average(away_scored, weights=w))
    league_avg_goals = (league_avg_home_goals + league_avg_away_goals) / 2.0

    long_rows = []
    for (idx, row), weight in zip(finished.iterrows(), w):
        long_rows.append((row["team_h"], row["team_h_score"], row["team_a_score"], weight))
        long_rows.append((row["team_a"], row["team_a_score"], row["team_h_score"], weight))
    long_df = pd.DataFrame(long_rows, columns=["team", "scored", "conceded", "weight"])

    attack: dict[int, float] = {}
    defence: dict[int, float] = {}
    games_played: dict[int, int] = {}
    for tid in team_ids:
        sub = long_df[long_df["team"] == tid]
        games_played[tid] = len(sub)
        prior_attack = overall[tid] / avg_overall
        prior_defence = avg_overall / overall[tid]
        if sub.empty:
            attack[tid] = prior_attack
            defence[tid] = prior_defence
            continue
        wsum = sub["weight"].sum()
        data_scored_rate = float(np.average(sub["scored"], weights=sub["weight"])) / league_avg_goals
        data_conceded_rate = float(np.average(sub["conceded"], weights=sub["weight"])) / league_avg_goals
        # shrink the data-driven rate toward the preseason prior when few games have been played
        shrink = wsum / (wsum + SAMPLE_SHRINKAGE_GAMES)
        attack[tid] = shrink * data_scored_rate + (1 - shrink) * prior_attack
        defence[tid] = shrink * data_conceded_rate + (1 - shrink) * prior_defence

    return TeamStrength(attack, defence, league_avg_home_goals, league_avg_away_goals, games_played)


@dataclass(frozen=True)
class PlayerForm:
    minutes_per_game: float
    start_rate: float  # fraction of recent appearances with >=60 minutes
    goals_per_90: float
    assists_per_90: float
    xg_per_90: float
    xa_per_90: float
    bps_per_90: float
    defensive_actions_per_90: float
    saves_per_90: float
    yellow_per_90: float
    red_per_90: float
    games_seen: int
    sample_minutes: float  # recency-weighted total minutes behind the per-90 rates, for shrinkage


def _per90(total_weighted: float, weight_sum_minutes: float) -> float:
    return (total_weighted / weight_sum_minutes * 90.0) if weight_sum_minutes > 0 else 0.0


def compute_player_form(history: list[dict]) -> PlayerForm:
    """Recency-weighted per-90 rates from a player's match log this season."""
    played = [h for h in history if h.get("minutes", 0) > 0]
    if not played:
        return PlayerForm(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    n = len(played)
    w = _recency_weights(n)
    minutes = np.array([h["minutes"] for h in played], dtype=float)
    goals = np.array([h["goals_scored"] for h in played], dtype=float)
    assists = np.array([h["assists"] for h in played], dtype=float)
    xg = np.array([float(h.get("expected_goals", 0) or 0) for h in played], dtype=float)
    xa = np.array([float(h.get("expected_assists", 0) or 0) for h in played], dtype=float)
    bps = np.array([h.get("bps", 0) for h in played], dtype=float)
    tackles = np.array([h.get("tackles", 0) or 0 for h in played], dtype=float)
    cbi = np.array([h.get("clearances_blocks_interceptions", 0) or 0 for h in played], dtype=float)
    saves = np.array([h.get("saves", 0) or 0 for h in played], dtype=float)
    yellows = np.array([h.get("yellow_cards", 0) or 0 for h in played], dtype=float)
    reds = np.array([h.get("red_cards", 0) or 0 for h in played], dtype=float)

    weighted_minutes = float(np.sum(w * minutes))
    recent = played[-MINUTES_LOOKBACK:]
    recent_w = _recency_weights(len(recent))
    recent_minutes = np.array([h["minutes"] for h in recent], dtype=float)
    recent_starts = (recent_minutes >= 60).astype(float)

    return PlayerForm(
        minutes_per_game=float(np.average(minutes, weights=w)),
        start_rate=float(np.average(recent_starts, weights=recent_w)),
        goals_per_90=_per90(float(np.sum(w * goals)), weighted_minutes),
        assists_per_90=_per90(float(np.sum(w * assists)), weighted_minutes),
        xg_per_90=_per90(float(np.sum(w * xg)), weighted_minutes),
        xa_per_90=_per90(float(np.sum(w * xa)), weighted_minutes),
        bps_per_90=_per90(float(np.sum(w * bps)), weighted_minutes),
        defensive_actions_per_90=_per90(float(np.sum(w * (tackles + cbi))), weighted_minutes),
        saves_per_90=_per90(float(np.sum(w * saves)), weighted_minutes),
        yellow_per_90=_per90(float(np.sum(w * yellows)), weighted_minutes),
        red_per_90=_per90(float(np.sum(w * reds)), weighted_minutes),
        games_seen=n,
        sample_minutes=weighted_minutes,
    )


def position_rate_priors(forms_by_position: dict[str, list[PlayerForm]]) -> dict[str, dict[str, float]]:
    """Minutes-weighted average per-90 rate for each position — the shrinkage
    target for players who don't have much playing time behind their own
    numbers yet (a defender's one lucky goal in 20 minutes shouldn't
    extrapolate to a 9-goals-per-90 rate)."""
    priors: dict[str, dict[str, float]] = {}
    for pos, forms in forms_by_position.items():
        total_minutes = sum(f.sample_minutes for f in forms) or 1.0
        priors[pos] = {
            field: sum(getattr(f, field) * f.sample_minutes for f in forms) / total_minutes
            for field in _RATE_FIELDS
        }
    return priors


def shrink_form(form: PlayerForm, prior: dict[str, float], k_minutes: float = RATE_SHRINKAGE_MINUTES) -> PlayerForm:
    """Blend a player's own per-90 rates with their position's average,
    weighted by how much playing time backs their own numbers: a player with
    two starts is described mostly by the position prior, a player with a
    season of minutes almost entirely by their own numbers."""
    weight = form.sample_minutes / (form.sample_minutes + k_minutes)
    updates = {field: weight * getattr(form, field) + (1 - weight) * prior[field] for field in _RATE_FIELDS}
    return replace(form, **updates)


def expected_minutes(form: PlayerForm, status: str, chance_of_playing_next_round: float | None) -> float:
    """Expected minutes next fixture, blending recent playing time with FPL's
    own injury/rotation flag (`chance_of_playing_next_round`)."""
    if form.games_seen == 0:
        base = 45.0  # unknown quantity (new signing, no minutes yet): assume a coin-flip squad player
    else:
        base = 90.0 * form.start_rate + form.minutes_per_game * (1 - form.start_rate) * 0.3

    if chance_of_playing_next_round is not None:
        availability = chance_of_playing_next_round / 100.0
    elif status == "a":
        availability = 1.0
    elif status == "d":
        availability = 0.6
    else:  # injured / suspended / unavailable / left the club
        availability = 0.0

    return min(90.0, base * availability)


def opponent_history_modifier(history: list[dict], opponent_team_id: int) -> tuple[float, int]:
    """How this player has performed against this specific opponent this
    season, as a small multiplicative nudge on top of the general model.
    Heavily shrunk toward 1.0 (no effect) since within-season sample sizes
    against one opponent are tiny (usually 0-2 games).
    """
    matches = [h for h in history if h.get("opponent_team") == opponent_team_id and h.get("minutes", 0) > 0]
    if not matches:
        return 1.0, 0
    avg_pts = float(np.mean([h["total_points"] for h in matches]))
    baseline = float(np.mean([h["total_points"] for h in history if h.get("minutes", 0) > 0])) or 2.0
    n = len(matches)
    shrink = n / (n + OPPONENT_HISTORY_SHRINKAGE)
    raw_effect = (avg_pts - baseline) / max(baseline, 1.0)
    modifier = 1.0 + shrink * raw_effect
    return float(np.clip(modifier, 1 - OPPONENT_HISTORY_CAP, 1 + OPPONENT_HISTORY_CAP)), n
