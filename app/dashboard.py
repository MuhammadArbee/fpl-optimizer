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
from fpl_optimizer.backtest import backtest_gameweek, finished_gameweeks
from fpl_optimizer.data import current_and_next_gameweek, fixtures_frame, players_frame, scoring_rules, squad_rules, teams_frame
from fpl_optimizer.optimize import build_squad
from fpl_optimizer.predict import predict, season_table

st.set_page_config(page_title="FPL Squad Optimizer", layout="wide")

FORMATION_ROWS = ["GKP", "DEF", "MID", "FWD"]


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def load_raw_data(refresh: bool):
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
    return bootstrap, players_df, teams_df, fixtures_df, rules, scoring, summaries, next_gw


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def load_predictions(horizon: int, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, object, int]:
    bootstrap, players_df, teams_df, fixtures_df, rules, scoring, summaries, next_gw = load_raw_data(refresh)
    predicted = predict(players_df, teams_df, fixtures_df, summaries, scoring, start_gw=next_gw, num_gws=horizon)
    return predicted, teams_df, rules, next_gw


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def load_season(refresh: bool):
    """Project every remaining gameweek this season (GW-by-GW), not just a
    lump total — the full-season answer to "who should I use each week"."""
    bootstrap, players_df, teams_df, fixtures_df, rules, scoring, summaries, next_gw = load_raw_data(refresh)
    last_gw = bootstrap["events"][-1]["id"]
    gws = list(range(next_gw, last_gw + 1))
    predicted = predict(players_df, teams_df, fixtures_df, summaries, scoring, start_gw=next_gw, num_gws=len(gws))
    table = season_table(predicted, gws)
    return predicted, table, rules, gws


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def run_backtest(refresh: bool):
    bootstrap, players_df, teams_df, fixtures_df, rules, scoring, summaries, _ = load_raw_data(refresh)
    gws = [g for g in finished_gameweeks(bootstrap) if g >= 2]
    return [backtest_gameweek(bootstrap, players_df, teams_df, fixtures_df, summaries, scoring, rules, gw) for gw in gws]


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

    tab_squad, tab_players, tab_player_detail, tab_season, tab_backtest = st.tabs(
        ["Optimal Squad", "All Players", "Player Explorer", "Season Planner", "Backtest"]
    )

    with tab_squad:
        squad = build_squad(predicted, rules, budget=budget)
        c1, c2, c3 = st.columns(3)
        c1.metric("Squad cost", f"£{squad.total_cost}m")
        points_label = "Predicted starting-XI points" if horizon == 1 else f"Predicted points ({horizon} GWs total)"
        c2.metric(points_label, f"{squad.expected_points}")
        c3.metric("Gameweek", f"GW{next_gw}" if horizon == 1 else f"GW{next_gw}-{next_gw + horizon - 1}")
        if horizon > 1:
            st.caption(f"That's a total across {horizon} gameweeks — roughly {squad.expected_points / horizon:.1f}/GW on average.")
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

    with tab_season:
        st.caption(
            "Projects every remaining gameweek this season, individually — not a single lumped total. "
            "Assumes the same squad all season (no transfers), so treat far-future gameweeks as a rough guide: "
            "form, injuries and prices will all move between now and then."
        )
        with st.spinner("Projecting the rest of the season (this fetches full player histories, can take a minute)..."):
            season_predicted, table, season_rules, gws = load_season(refresh)

        view = st.radio("Show", ["My optimal squad", "Top players overall"], horizontal=True)
        if view == "Top players overall":
            top_n = st.slider("How many players", 5, 50, 15)
            chosen = table.sort_values("season_total", ascending=False).head(top_n)
        else:
            season_squad = build_squad(season_predicted, season_rules, budget=budget)
            chosen = table.loc[season_squad.squad_ids].sort_values(["position", "season_total"], ascending=[True, False])

        gw_cols = [f"GW{g}" for g in gws]
        st.dataframe(
            chosen[["name", "position", "price", "season_total", "avg_per_gw"] + gw_cols],
            use_container_width=True,
            height=500,
        )
        st.caption("Download the full player x gameweek matrix (all ~660 players) as CSV:")
        st.download_button(
            "Download full season projection (CSV)",
            table.to_csv().encode("utf-8"),
            file_name=f"fpl_season_projection_gw{gws[0]}-{gws[-1]}.csv",
            mime="text/csv",
        )

    with tab_backtest:
        st.caption(
            "Rebuilds predictions using only data that would have been available before each already-played "
            "gameweek, then checks them against what actually happened — the honest test of whether this model "
            "is worth anything, not just a demo of it running."
        )
        with st.spinner("Backtesting finished gameweeks..."):
            results = run_backtest(refresh)
        if not results:
            st.info("No finished gameweeks (beyond GW1) yet — nothing to backtest against.")
        else:
            summary = pd.DataFrame(
                [
                    {
                        "GW": r.gameweek,
                        "Model r": round(r.pearson_r, 3),
                        "Naive r": round(r.naive_pearson_r, 3),
                        "Model MAE": round(r.mae, 3),
                        "Naive MAE": round(r.naive_mae, 3),
                        "Squad pts": r.squad_actual_points,
                        "Avg manager": r.average_entry_score,
                        "Top manager": r.highest_score,
                    }
                    for r in results
                ]
            )
            st.dataframe(summary, hide_index=True, use_container_width=True)

            avg_model_r = summary["Model r"].mean()
            avg_naive_r = summary["Naive r"].mean()
            avg_squad_pts = summary["Squad pts"].mean()
            avg_mgr = summary["Avg manager"].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Model vs. naive baseline (r)", f"{avg_model_r:.3f}", f"{avg_model_r - avg_naive_r:+.3f}")
            c2.metric("Backtested squad pts/GW", f"{avg_squad_pts:.1f}")
            c3.metric("Avg. real manager pts/GW", f"{avg_mgr:.1f}")

            if avg_model_r <= avg_naive_r:
                st.warning(
                    "Over these gameweeks, the full model doesn't yet beat the naive baseline (each player's own "
                    "season-to-date average). This early in a season, team-strength and fixture-difficulty signals "
                    "are themselves built on very few matches, so they add noise rather than signal — and 2-4 "
                    "gameweeks is too small a sample to draw a firm conclusion either way. Worth re-checking as "
                    "more gameweeks accumulate."
                )
            else:
                st.success("Over these gameweeks, the full model beats the naive season-average baseline.")

            gw_choice = st.selectbox("Inspect one gameweek's predictions vs. actual", [r.gameweek for r in results])
            chosen = next(r for r in results if r.gameweek == gw_choice)
            st.dataframe(
                chosen.predictions.sort_values("predicted_points", ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=400,
            )


if __name__ == "__main__":
    main()
