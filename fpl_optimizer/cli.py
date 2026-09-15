"""Command-line entry point.

    fpl fetch                          # refresh all cached FPL data
    fpl squad --horizon 3               # best 15 under budget, weighing the next 3 GWs
    fpl season                          # project every remaining gameweek this season, GW-by-GW
    fpl transfers --team-id 1234567     # transfer suggestions for a real FPL team
    fpl player "Mohamed Salah"          # explain one player's expected points
    fpl backtest                        # validate predictions against already-finished gameweeks
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import api, ml
from .backtest import backtest_gameweek, finished_gameweeks
from .data import current_and_next_gameweek, fixtures_frame, players_frame, scoring_rules, squad_rules, teams_frame
from .optimize import build_squad, optimize_transfers
from .predict import predict, season_table

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 30)


def _load_raw_data(refresh: bool):
    print("Loading FPL data..." + (" (refreshing from API)" if refresh else " (cached)"), file=sys.stderr)
    bootstrap = api.get_bootstrap(refresh=refresh)
    fixtures = api.get_fixtures(refresh=refresh)
    teams_df = teams_frame(bootstrap)
    players_df = players_frame(bootstrap)
    players_df["team_name"] = players_df["team"].map(teams_df["short_name"])
    fixtures_df = fixtures_frame(fixtures)
    rules = squad_rules(bootstrap)
    scoring = scoring_rules(bootstrap)
    _, next_gw = current_and_next_gameweek(bootstrap)

    print(f"Fetching player histories ({len(players_df)} players)...", file=sys.stderr)

    def progress(done, total):
        if done % 100 == 0 or done == total:
            print(f"  {done}/{total}", file=sys.stderr)

    summaries = api.get_all_element_summaries(list(players_df.index), refresh=refresh, on_progress=progress)
    return bootstrap, players_df, teams_df, fixtures_df, rules, scoring, summaries, next_gw


def _predict(players_df, teams_df, fixtures_df, summaries, scoring, start_gw, num_gws, engine: str):
    if engine == "linear":
        try:
            model, _ = ml.load_model()
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        return ml.predict_with_model(players_df, teams_df, fixtures_df, summaries, model, start_gw=start_gw, num_gws=num_gws)
    return predict(players_df, teams_df, fixtures_df, summaries, scoring, start_gw=start_gw, num_gws=num_gws)


def _load_data(refresh: bool, horizon: int, engine: str = "heuristic"):
    bootstrap, players_df, teams_df, fixtures_df, rules, scoring, summaries, next_gw = _load_raw_data(refresh)
    predicted = _predict(players_df, teams_df, fixtures_df, summaries, scoring, next_gw, horizon, engine)
    return predicted, teams_df, rules, next_gw


def cmd_fetch(args):
    _load_data(refresh=True, horizon=1)
    print("Cache refreshed.", file=sys.stderr)


def cmd_squad(args):
    predicted, teams_df, rules, next_gw = _load_data(refresh=args.refresh, horizon=args.horizon, engine=args.engine)
    budget = args.budget if args.budget is not None else rules.budget
    squad = build_squad(predicted, rules, budget=budget)

    span = f"GW{next_gw}" if args.horizon == 1 else f"GW{next_gw}-{next_gw + args.horizon - 1}"
    print(f"\nRecommended squad for {span} ({args.horizon} gameweek{'s' if args.horizon > 1 else ''})")
    print(f"Budget used: £{squad.total_cost}m / £{budget}m")
    if args.horizon == 1:
        print(f"Predicted starting-XI points: {squad.expected_points}\n")
    else:
        print(
            f"Predicted starting-XI points: {squad.expected_points} total over {args.horizon} gameweeks "
            f"(~{squad.expected_points / args.horizon:.1f}/GW average)\n"
        )
    print(squad.summary(predicted).to_string(index=False))
    print(f"\nCaptain: {predicted.loc[squad.captain_id, 'name']}")
    print(f"Vice-captain: {predicted.loc[squad.vice_captain_id, 'name']}")


def cmd_transfers(args):
    predicted, teams_df, rules, next_gw = _load_data(refresh=args.refresh, horizon=args.horizon, engine=args.engine)

    entry = api.get_entry(args.team_id, refresh=args.refresh)
    picks_gw = entry.get("current_event") or (next_gw - 1)
    picks = api.get_entry_picks(args.team_id, picks_gw, refresh=args.refresh)
    current_ids = [p["element"] for p in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10.0
    free_transfers = args.free_transfers if args.free_transfers is not None else 1

    plan = optimize_transfers(current_ids, predicted, rules, bank=bank, free_transfers=free_transfers)

    print(f"\nTransfer plan for team {args.team_id} ('{entry.get('name', '?')}'), GW{next_gw}")
    print(f"Bank: £{bank}m   Free transfers assumed: {free_transfers}")
    if not plan.transfers_out:
        print("No changes recommended — your squad is already close to optimal for this horizon.")
    else:
        print(f"Points hit: -{plan.points_hit}   Net expected gain: {plan.net_expected_points_gain}\n")
        for out_id, in_id in zip(plan.transfers_out, plan.transfers_in):
            out_p, in_p = predicted.loc[out_id], predicted.loc[in_id]
            print(f"  OUT: {out_p['name']:<25} (£{out_p['price']}m, xP {out_p['expected_points']})")
            print(f"  IN:  {in_p['name']:<25} (£{in_p['price']}m, xP {in_p['expected_points']})")
    print(f"\nNew squad predicted starting-XI points: {plan.new_squad.expected_points}")
    print(plan.new_squad.summary(predicted).to_string(index=False))


def cmd_player(args):
    predicted, teams_df, rules, next_gw = _load_data(refresh=args.refresh, horizon=args.horizon, engine=args.engine)
    matches = predicted[predicted["name"].str.contains(args.name, case=False, na=False)]
    if matches.empty:
        print(f"No player matching '{args.name}'", file=sys.stderr)
        sys.exit(1)
    for pid, row in matches.iterrows():
        print(f"\n{row['name']} ({row['position']}, {row['team_name']}, £{row['price']}m)")
        print(
            f"Expected minutes: {row['expected_minutes']}   "
            f"Expected points: {row['expected_points']} total over {args.horizon} gameweek{'s' if args.horizon > 1 else ''}"
        )
        for fx in row["breakdown"]:
            venue = "H" if fx["is_home"] else "A"
            print(f"  GW{fx['gameweek']} vs {fx['opponent']} ({venue}): {fx['total']:.2f} pts")
            for k in ("goals", "assists", "clean_sheet", "goals_conceded", "saves", "defensive_contribution", "bonus", "cards", "appearance"):
                if abs(fx.get(k, 0.0)) > 0.005:
                    print(f"      {k}: {fx[k]:+.2f}")
            if fx["h2h_sample_size"] > 0:
                print(f"      opponent history modifier: {fx['opponent_history_modifier']}x (from {fx['h2h_sample_size']} past meeting(s))")
        if args.engine == "linear":
            print("  (linear engine: a fitted regression doesn't decompose into goals/assists/etc. — only the total is meaningful)")


def cmd_season(args):
    bootstrap, players_df, teams_df, fixtures_df, rules, scoring, summaries, next_gw = _load_raw_data(args.refresh)
    last_gw = bootstrap["events"][-1]["id"]
    gws = list(range(next_gw, last_gw + 1))

    print(f"\nProjecting GW{next_gw} through GW{last_gw} ({len(gws)} gameweeks)...", file=sys.stderr)
    predicted = _predict(players_df, teams_df, fixtures_df, summaries, scoring, next_gw, len(gws), args.engine)
    table = season_table(predicted, gws)

    if args.top:
        chosen = table.sort_values("season_total", ascending=False).head(args.top)
        print(f"\nTop {args.top} players by projected points, GW{next_gw}-{last_gw}:\n")
    else:
        squad = build_squad(predicted, rules, budget=args.budget if args.budget is not None else rules.budget)
        chosen = table.loc[squad.squad_ids].sort_values(["position", "season_total"], ascending=[True, False])
        print(f"\nFull-season squad projection, GW{next_gw}-{last_gw} (assumes no transfers along the way):\n")

    display_cols = ["name", "position", "price", "season_total", "avg_per_gw"]
    if args.full:
        display_cols = ["name", "position", "price"] + [f"GW{g}" for g in gws] + ["season_total", "avg_per_gw"]
    print(chosen[display_cols].to_string(index=False))

    if args.csv:
        table.to_csv(args.csv)
        print(f"\nFull player x gameweek matrix ({len(table)} players x {len(gws)} gameweeks) written to {args.csv}", file=sys.stderr)


def cmd_backtest(args):
    bootstrap, players_df, teams_df, fixtures_df, rules, scoring, summaries, next_gw = _load_raw_data(args.refresh)
    gws = args.gws if args.gws else finished_gameweeks(bootstrap)
    gws = [g for g in gws if g >= 2]  # GW1 has no prior history to build a model from
    if not gws:
        print("No finished gameweeks (after GW1) available to backtest yet.", file=sys.stderr)
        sys.exit(1)

    print(f"\nBacktesting gameweeks {gws}")
    print(
        "(approximations: today's prices and availability stand in for the values at the time,\n"
        " since the API doesn't expose price/injury history)\n"
    )

    # forward-chaining linear-model comparison: train only on gameweeks strictly
    # before the one being scored, same discipline as the heuristic backtest
    training_df = ml.build_training_set(players_df, teams_df, fixtures_df, summaries, scoring, gws)
    linear_by_gw = {f.test_gw: f for f in ml.evaluate_forward_chaining(training_df)}
    have_linear = len(linear_by_gw) > 0

    header = f"{'GW':>3}  {'naive r':>8}  {'heur. r':>8}"
    if have_linear:
        header += f"  {'linear r':>8}"
    header += f"  {'heur. MAE':>10}"
    if have_linear:
        header += f"  {'linear MAE':>10}"
    header += f"  {'squad pts':>10}  {'avg mgr':>8}  {'top mgr':>8}"
    print(header)
    print("-" * len(header))

    results = []
    for gw in gws:
        result = backtest_gameweek(bootstrap, players_df, teams_df, fixtures_df, summaries, scoring, rules, gw)
        results.append(result)
        line = f"{result.gameweek:>3}  {result.naive_pearson_r:>8.3f}  {result.pearson_r:>8.3f}"
        if have_linear:
            fold = linear_by_gw.get(gw)
            line += f"  {fold.linear_r:>8.3f}" if fold else f"  {'--':>8}"
        line += f"  {result.mae:>10.3f}"
        if have_linear:
            fold = linear_by_gw.get(gw)
            line += f"  {fold.linear_mae:>10.3f}" if fold else f"  {'--':>10}"
        line += f"  {result.squad_actual_points:>10.1f}  {result.average_entry_score or float('nan'):>8}  {result.highest_score or float('nan'):>8}"
        print(line)

    n = len(results)
    avg_r = sum(r.pearson_r for r in results) / n
    avg_naive_r = sum(r.naive_pearson_r for r in results) / n
    avg_squad_pts = sum(r.squad_actual_points for r in results) / n
    avg_mgr = sum(r.average_entry_score for r in results if r.average_entry_score is not None) / n
    print("-" * len(header))
    print(
        f"\nHeuristic model correlation {'beats' if avg_r > avg_naive_r else 'does not beat'} the naive "
        f"season-average baseline ({avg_r:.3f} vs {avg_naive_r:.3f} Pearson r)."
    )
    if have_linear:
        linear_gws = sorted(linear_by_gw.keys())
        comparable = [r for r in results if r.gameweek in linear_by_gw]
        avg_linear_r = sum(f.linear_r for f in linear_by_gw.values()) / len(linear_by_gw)
        avg_naive_r_comparable = sum(r.naive_pearson_r for r in comparable) / len(comparable)
        avg_heur_r_comparable = sum(r.pearson_r for r in comparable) / len(comparable)
        print(
            f"Over the {len(linear_gws)} gameweek(s) the linear model could be evaluated on (GW{linear_gws}, "
            f"forward-chained — never trained on the gameweek being scored):"
        )
        print(
            f"  linear {'beats' if avg_linear_r > avg_naive_r_comparable else 'does not beat'} naive "
            f"({avg_linear_r:.3f} vs {avg_naive_r_comparable:.3f}), and "
            f"{'beats' if avg_linear_r > avg_heur_r_comparable else 'does not beat'} the heuristic "
            f"({avg_linear_r:.3f} vs {avg_heur_r_comparable:.3f})."
        )
    else:
        print("(not enough finished gameweeks yet for a forward-chaining linear-model comparison — needs at least 2 to train+test)")
    print(f"The optimizer's backtested squads averaged {avg_squad_pts:.1f} pts/GW vs an average manager's {avg_mgr:.1f}.")


def cmd_train(args):
    bootstrap, players_df, teams_df, fixtures_df, rules, scoring, summaries, next_gw = _load_raw_data(args.refresh)
    gws = [g for g in finished_gameweeks(bootstrap) if g >= 2]
    if len(gws) < 2:
        print("Need at least 2 finished gameweeks (beyond GW1) to train and validate a model.", file=sys.stderr)
        sys.exit(1)

    print(f"\nBuilding training set from gameweeks {gws}...", file=sys.stderr)
    training_df = ml.build_training_set(players_df, teams_df, fixtures_df, summaries, scoring, gws)
    print(f"{len(training_df)} player-fixture rows.\n")

    print("Forward-chaining validation (train on earlier gameweeks only, never the one being scored):\n")
    folds = ml.evaluate_forward_chaining(training_df)
    if not folds:
        print("Not enough data for even one validation fold yet.", file=sys.stderr)
        sys.exit(1)

    header = f"{'test GW':>8}  {'n_train':>8}  {'naive r':>8}  {'heur. r':>8}  {'linear r':>8}  {'naive MAE':>10}  {'heur. MAE':>10}  {'linear MAE':>10}"
    print(header)
    print("-" * len(header))
    for f in folds:
        print(
            f"{f.test_gw:>8}  {f.n_train:>8}  {f.naive_r:>8.3f}  {f.heuristic_r:>8.3f}  {f.linear_r:>8.3f}  "
            f"{f.naive_mae:>10.3f}  {f.heuristic_mae:>10.3f}  {f.linear_mae:>10.3f}"
        )
    avg_linear_r = sum(f.linear_r for f in folds) / len(folds)
    avg_heuristic_r = sum(f.heuristic_r for f in folds) / len(folds)
    avg_naive_r = sum(f.naive_r for f in folds) / len(folds)
    print("-" * len(header))
    print(f"\nAverage r — naive: {avg_naive_r:.3f}, heuristic: {avg_heuristic_r:.3f}, linear: {avg_linear_r:.3f}")

    print("\nTraining final model on all available gameweeks...", file=sys.stderr)
    model = ml.train_linear_model(training_df)
    ml.save_model(model, training_df, folds)
    print(f"Saved to {ml.MODEL_PATH}")

    ridge = model.named_steps["ridge"]
    coefs = sorted(zip(ml.FEATURE_NAMES, ridge.coef_), key=lambda kv: -abs(kv[1]))
    print("\nLearned feature weights (standardized — larger magnitude = more influence), strongest first:")
    for name, coef in coefs[:12]:
        print(f"  {name:<28} {coef:+.3f}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--horizon", type=int, default=1, help="number of gameweeks to weigh (default: 1)")
    common.add_argument("--refresh", action="store_true", help="force a re-fetch from the FPL API instead of using the cache")
    common.add_argument(
        "--engine",
        choices=["heuristic", "linear"],
        default="heuristic",
        help="'heuristic' (default): the hand-built scoring-rule model. 'linear': a Ridge regression trained on "
        "finished gameweeks via `fpl train` — requires that to have been run first.",
    )

    parser = argparse.ArgumentParser(prog="fpl", description="Fantasy Premier League squad optimizer")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="refresh all cached FPL data").set_defaults(func=cmd_fetch)

    p_squad = sub.add_parser("squad", help="build the best 15-man squad under budget", parents=[common])
    p_squad.add_argument("--budget", type=float, default=None, help="override budget in £m (default: 100.0, read from the API)")
    p_squad.set_defaults(func=cmd_squad)

    p_transfers = sub.add_parser("transfers", help="suggest transfers for an existing FPL team", parents=[common])
    p_transfers.add_argument("--team-id", type=int, required=True, help="your FPL team/entry ID (from the URL on the FPL site)")
    p_transfers.add_argument("--free-transfers", type=int, default=None, help="free transfers available (default: 1)")
    p_transfers.set_defaults(func=cmd_transfers)

    p_player = sub.add_parser("player", help="explain one player's expected-points breakdown", parents=[common])
    p_player.add_argument("name", help="player name (or partial match)")
    p_player.set_defaults(func=cmd_player)

    p_season = sub.add_parser(
        "season",
        help="project points for every remaining gameweek this season (GW-by-GW, not just a lump total)",
    )
    p_season.add_argument("--refresh", action="store_true", help="force a re-fetch from the FPL API instead of using the cache")
    p_season.add_argument(
        "--engine",
        choices=["heuristic", "linear"],
        default="heuristic",
        help="'heuristic' (default) or 'linear' (a trained model — run `fpl train` first)",
    )
    p_season.add_argument("--budget", type=float, default=None, help="override budget in £m (default: 100.0)")
    p_season.add_argument("--top", type=int, default=None, help="show the top N players by season total instead of building a squad")
    p_season.add_argument("--full", action="store_true", help="show every gameweek's column instead of just the season total/average")
    p_season.add_argument("--csv", type=str, default=None, help="write the full player x gameweek matrix (all players) to this CSV path")
    p_season.set_defaults(func=cmd_season)

    p_backtest = sub.add_parser(
        "backtest",
        help="validate the model against already-finished gameweeks",
        parents=[common],
    )
    p_backtest.add_argument(
        "--gws", type=int, nargs="+", default=None, help="specific gameweeks to test (default: all finished so far)"
    )
    p_backtest.set_defaults(func=cmd_backtest)

    p_train = sub.add_parser(
        "train",
        help="fit a Ridge regression on finished gameweeks and validate it against the heuristic model",
    )
    p_train.add_argument("--refresh", action="store_true", help="force a re-fetch from the FPL API instead of using the cache")
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
