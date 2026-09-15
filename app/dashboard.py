"""Streamlit dashboard for the FPL optimizer.

Run with:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpl_optimizer import api
from fpl_optimizer.data import current_and_next_gameweek, fixtures_frame, players_frame, scoring_rules, squad_rules, teams_frame
from fpl_optimizer.optimize import build_squad
from fpl_optimizer.predict import predict

st.set_page_config(page_title="FPL Squad Optimizer", layout="wide")

FORMATION_ROWS = ["GKP", "DEF", "MID", "FWD"]


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def load_predictions(horizon: int, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, object, int]:
    bootstrap = api.get_bootstrap(refresh=refresh)
    fixtures = api.get_fixtures(refresh=refresh)
    teams_df = teams_frame(bootstrap)
    players_df = players_frame(bootstrap)
    players_df["team_name"] = players_df["team"].map(teams_df["short_name"])
    fixtures_df = fixtures_frame(fixtures)
    rules = squad_rules(bootstrap)
    scoring = scoring_rules(bootstrap)
    _, next_gw = current_and_next_gameweek(bootstrap)

    ids = list(players_df.index)
    progress = st.progress(0.0, text="Fetching player histories from the FPL API...")

    def on_progress(done, total):
        progress.progress(done / total, text=f"Fetching player histories... {done}/{total}")

    summaries = api.get_all_element_summaries(ids, refresh=refresh, on_progress=on_progress)
    progress.empty()

    predicted = predict(players_df, teams_df, fixtures_df, summaries, scoring, start_gw=next_gw, num_gws=horizon)
    return predicted, teams_df, rules, next_gw


def render_pitch(squad, predicted: pd.DataFrame):
    for pos in FORMATION_ROWS:
        ids = [i for i in squad.starting_ids if predicted.loc[i, "position"] == pos]
        cols = st.columns(len(ids) if ids else 1)
        for col, pid in zip(cols, ids):
            row = predicted.loc[pid]
            tag = " (C)" if pid == squad.captain_id else " (VC)" if pid == squad.vice_captain_id else ""
            col.markdown(
                f"**{row['name']}{tag}**\n\n"
                f"{row['team_name']} · £{row['price']}m\n\n"
                f"xP: {row['expected_points']:.1f}"
            )
    st.markdown("---")
    st.caption("Bench (substitution order)")
    bench_cols = st.columns(len(squad.bench_ids))
    for col, pid in zip(bench_cols, squad.bench_ids):
        row = predicted.loc[pid]
        col.markdown(f"{row['name']}\n\n{row['team_name']} · £{row['price']}m · xP {row['expected_points']:.1f}")


def render_player_breakdown(row: pd.Series):
    st.subheader(f"{row['name']} — {row['position']}, {row['team_name']}, £{row['price']}m")
    st.write(f"Expected minutes next fixture: **{row['expected_minutes']}**")
    for fx in row["breakdown"]:
        venue = "Home" if fx["is_home"] else "Away"
        st.markdown(f"**GW{fx['gameweek']} vs {fx['opponent']} ({venue})**")
        parts = {
            k: v
            for k, v in fx.items()
            if k in ("appearance", "goals", "assists", "clean_sheet", "goals_conceded", "saves", "defensive_contribution", "bonus", "cards")
            and abs(v) > 0.005
        }
        st.dataframe(pd.DataFrame([parts]), hide_index=True)
        if fx["h2h_sample_size"] > 0:
            st.caption(f"Opponent-history modifier: {fx['opponent_history_modifier']}x, from {fx['h2h_sample_size']} past meeting(s) this season.")


def main():
    st.title("Fantasy Premier League — Squad Optimizer")
    st.caption(
        "Expected points blend recent form, underlying xG/xA, fixture difficulty, clean-sheet probability, "
        "defensive-contribution points, and head-to-head history — then an ILP solver picks the 15-man squad "
        "that maximises points under the £100m budget and FPL's own squad rules."
    )

    with st.sidebar:
        st.header("Settings")
        horizon = st.slider("Gameweek horizon", min_value=1, max_value=6, value=1, help="Weigh expected points over this many upcoming gameweeks")
        budget = st.number_input("Budget (£m)", min_value=50.0, max_value=100.0, value=100.0, step=0.5)
        refresh = st.button("Refresh data from FPL API")

    with st.spinner("Loading..."):
        predicted, teams_df, rules, next_gw = load_predictions(horizon, refresh)

    tab_squad, tab_players, tab_player_detail = st.tabs(["Optimal Squad", "All Players", "Player Explorer"])

    with tab_squad:
        squad = build_squad(predicted, rules, budget=budget)
        c1, c2, c3 = st.columns(3)
        c1.metric("Squad cost", f"£{squad.total_cost}m")
        c2.metric("Predicted starting-XI points", f"{squad.expected_points}")
        c3.metric("Gameweek", f"GW{next_gw}")
        render_pitch(squad, predicted)

    with tab_players:
        st.dataframe(
            predicted[["name", "position", "team_name", "price", "expected_points", "expected_minutes", "form", "selected_by_percent"]]
            .sort_values("expected_points", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=600,
        )

    with tab_player_detail:
        name = st.text_input("Search for a player")
        if name:
            matches = predicted[predicted["name"].str.contains(name, case=False, na=False)]
            for _, row in matches.head(5).iterrows():
                render_player_breakdown(row)


if __name__ == "__main__":
    main()
