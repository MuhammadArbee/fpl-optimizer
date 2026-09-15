"""Turn raw FPL API payloads into tidy DataFrames, and expose the game's own
rules (scoring, squad structure, budget) instead of hardcoding them — the API
returns them directly and they do change between seasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

NUMERIC_PLAYER_COLS = [
    "form",
    "points_per_game",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "ict_index",
    "influence",
    "creativity",
    "threat",
    "selected_by_percent",
]


@dataclass(frozen=True)
class SquadRules:
    """Squad-building constraints, read from the live API rather than assumed."""

    budget: float
    squad_size: int
    starting_size: int
    max_per_team: int
    # position -> (min in 15-man squad == max, since squad_select is exact)
    squad_count: dict[str, int]
    # position -> (min, max) allowed in the starting XI
    starting_min: dict[str, int]
    starting_max: dict[str, int]


@dataclass(frozen=True)
class ScoringRules:
    """Points-per-action, read live so rule changes (e.g. GKP goals = 10 in
    2026/27) are picked up automatically."""

    goals_scored: dict[str, int]
    assists: int
    clean_sheets: dict[str, int]
    goals_conceded: dict[str, int]
    saves_per_point: int
    bonus_per_point: int
    yellow_cards: int
    red_cards: int
    own_goals: int
    penalties_saved: int
    penalties_missed: int
    long_play: int
    short_play: int
    defensive_contribution: dict[str, int]
    # not exposed by the API; published FPL rule for 2025/26+.
    defensive_contribution_threshold: dict[str, int] = field(
        default_factory=lambda: {"DEF": 10, "MID": 12, "FWD": 12, "GKP": 999999}
    )


def teams_frame(bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["teams"])
    return df.set_index("id", drop=False)


def players_frame(bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["elements"])
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["price"] = df["now_cost"] / 10.0
    df["name"] = (df["first_name"] + " " + df["second_name"]).str.strip()
    for col in NUMERIC_PLAYER_COLS:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
    return df.set_index("id", drop=False)


def fixtures_frame(fixtures: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(fixtures)


def current_and_next_gameweek(bootstrap: dict) -> tuple[int | None, int]:
    """(current_gw_id_or_None, next_gw_id) — current is None before GW1's deadline."""
    events = bootstrap["events"]
    current = next((e["id"] for e in events if e.get("is_current")), None)
    nxt = next((e["id"] for e in events if e.get("is_next")), None)
    if nxt is None:
        unfinished = [e["id"] for e in events if not e["finished"]]
        nxt = unfinished[0] if unfinished else events[-1]["id"]
    return current, nxt


def squad_rules(bootstrap: dict) -> SquadRules:
    rules = bootstrap["game_config"]["rules"]
    pos_by_short = {et["singular_name_short"]: et for et in bootstrap["element_types"]}
    return SquadRules(
        budget=rules["squad_total_spend"] / 10.0,
        squad_size=rules["squad_squadsize"],
        starting_size=rules["squad_squadplay"],
        max_per_team=rules["squad_team_limit"],
        squad_count={pos: et["squad_select"] for pos, et in pos_by_short.items()},
        starting_min={pos: et["squad_min_play"] for pos, et in pos_by_short.items()},
        starting_max={pos: et["squad_max_play"] for pos, et in pos_by_short.items()},
    )


def scoring_rules(bootstrap: dict) -> ScoringRules:
    s = bootstrap["game_config"]["scoring"]
    return ScoringRules(
        goals_scored=s["goals_scored"],
        assists=s["assists"],
        clean_sheets=s["clean_sheets"],
        goals_conceded=s["goals_conceded"],
        saves_per_point=s["saves"],
        bonus_per_point=s["bonus"],
        yellow_cards=s["yellow_cards"],
        red_cards=s["red_cards"],
        own_goals=s["own_goals"],
        penalties_saved=s["penalties_saved"],
        penalties_missed=s["penalties_missed"],
        long_play=s["long_play"],
        short_play=s["short_play"],
        defensive_contribution=s["defensive_contribution"],
    )
