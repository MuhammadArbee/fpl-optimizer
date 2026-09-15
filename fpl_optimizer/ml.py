"""A learned alternative to the hand-tuned heuristic model.

`predict.py`'s heuristic bakes in hand-picked constants — a 60/40 blend of
xG vs. actual goals, a BPS-to-bonus-points scaling factor, and so on — that
were reasoned about rather than fit to data. This module instead trains a
**Ridge regression** (L2-regularized linear regression) on finished
gameweeks: same underlying features (form, fixture difficulty, clean-sheet
probability — see `predict.fixture_context`), but the weight on each one is
learned from what actually happened, not guessed.

Why Ridge rather than plain linear regression: with only a handful of
finished gameweeks so far this season, training rows number in the low
thousands and several features are correlated (e.g. `xg_per_90` and
`goals_per_90`). Plain least-squares regression would happily overfit that
noise; Ridge's L2 penalty keeps coefficients modest and stable, and
`RidgeCV` picks the penalty strength itself via cross-validation so there's
no manual tuning knob to get wrong.

Train with `fpl train`, which also reports forward-chaining validation
(train on earlier gameweeks, test on a later one — never the reverse) so
the reported accuracy can't leak future information, matching the
backtesting philosophy in `backtest.py`. The fitted model is saved
(coefficients + a StandardScaler, via joblib) alongside a JSON sidecar
recording feature names, training gameweeks, and the learned coefficients in
plain numbers — so what got learned is inspectable, not a black box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import _actual_points_for_gw, _naive_prediction, prepare_as_of, team_fixtures_for
from .data import ScoringRules
from .features import opponent_history_modifier
from .predict import _fixture_points, fixture_context

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "model"
MODEL_PATH = MODEL_DIR / "linear_model.joblib"
META_PATH = MODEL_DIR / "linear_model_meta.json"

FEATURE_NAMES = [
    "price",
    "is_gkp",
    "is_def",
    "is_mid",
    "is_fwd",
    "minutes_share",
    "is_home",
    "goals_per_90",
    "assists_per_90",
    "xg_per_90",
    "xa_per_90",
    "bps_per_90",
    "defensive_actions_per_90",
    "saves_per_90",
    "yellow_per_90",
    "red_per_90",
    "opponent_defence_leakiness",
    "opponent_attack",
    "attack_multiplier",
    "expected_goals",
    "expected_assists",
    "p_clean_sheet",
    "expected_actions",
    "expected_bps",
    "opponent_history_modifier",
    "naive_points",  # this player's own season-to-date average — a strong prior the model can lean on or override
]


def featurize(position: str, price: float, form, ctx: dict, naive_points: float) -> dict[str, float]:
    return {
        "price": price,
        "is_gkp": float(position == "GKP"),
        "is_def": float(position == "DEF"),
        "is_mid": float(position == "MID"),
        "is_fwd": float(position == "FWD"),
        "minutes_share": ctx["minutes_share"],
        "is_home": ctx["is_home"],
        "goals_per_90": form.goals_per_90,
        "assists_per_90": form.assists_per_90,
        "xg_per_90": form.xg_per_90,
        "xa_per_90": form.xa_per_90,
        "bps_per_90": form.bps_per_90,
        "defensive_actions_per_90": form.defensive_actions_per_90,
        "saves_per_90": form.saves_per_90,
        "yellow_per_90": form.yellow_per_90,
        "red_per_90": form.red_per_90,
        "opponent_defence_leakiness": ctx["opponent_defence_leakiness"],
        "opponent_attack": ctx["opponent_attack"],
        "attack_multiplier": ctx["attack_multiplier"],
        "expected_goals": ctx["expected_goals"],
        "expected_assists": ctx["expected_assists"],
        "p_clean_sheet": ctx["p_clean_sheet"],
        "expected_actions": ctx["expected_actions"],
        "expected_bps": ctx["expected_bps"],
        "opponent_history_modifier": ctx["opponent_history_modifier"],
        "naive_points": naive_points,
    }


def build_training_set(
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    summaries: dict[int, dict],
    scoring_rules: ScoringRules,
    gws: list[int],
) -> pd.DataFrame:
    """One row per (player, fixture) across `gws`, with features computed
    strictly from data available before that gameweek, the heuristic
    model's own prediction (for comparison), and the actual points scored —
    the label to learn."""
    rows = []
    for gw in gws:
        ctx = prepare_as_of(players_df, teams_df, fixtures_df, summaries, gw)
        for pid, row in players_df.iterrows():
            prior_history = ctx.prior_histories[pid]
            form = ctx.forms[pid]
            minutes = ctx.minutes[pid]
            naive = _naive_prediction(prior_history)
            full_history = summaries.get(pid, {}).get("history", [])

            for _, fx in team_fixtures_for(ctx, row["team"]).iterrows():
                is_home = fx["team_h"] == row["team"]
                opponent_id = fx["team_a"] if is_home else fx["team_h"]
                opp_modifier, _ = opponent_history_modifier(prior_history, opponent_id)
                fixture_ctx = fixture_context(form, ctx.team_strength, row["team"], opponent_id, is_home, minutes, opp_modifier)
                heuristic_pts, _ = _fixture_points(
                    position=row["position"],
                    form=form,
                    scoring=scoring_rules,
                    team_strength=ctx.team_strength,
                    team_id=row["team"],
                    opponent_id=opponent_id,
                    is_home=is_home,
                    minutes=minutes,
                    opp_modifier=opp_modifier,
                )
                features = featurize(row["position"], row["price"], form, fixture_ctx, naive)
                rows.append(
                    {
                        "gw": gw,
                        "player_id": pid,
                        "position": row["position"],
                        "naive_points": naive,
                        "heuristic_points": round(heuristic_pts, 3),
                        "actual_points": _actual_points_for_gw(full_history, gw),
                        **features,
                    }
                )
    return pd.DataFrame(rows)


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 30))),
        ]
    )


def train_linear_model(training_df: pd.DataFrame, feature_names: list[str] = FEATURE_NAMES) -> Pipeline:
    X = training_df[feature_names].to_numpy()
    y = training_df["actual_points"].to_numpy()
    pipeline = _make_pipeline()
    pipeline.fit(X, y)
    return pipeline


@dataclass
class FoldResult:
    test_gw: int
    train_gws: list[int]
    n_train: int
    n_test: int
    naive_r: float
    heuristic_r: float
    linear_r: float
    naive_mae: float
    heuristic_mae: float
    linear_mae: float


def evaluate_forward_chaining(training_df: pd.DataFrame, feature_names: list[str] = FEATURE_NAMES) -> list[FoldResult]:
    """Expanding-window validation: to score gameweek N, train only on
    gameweeks strictly before N. Never trains on the gameweek being scored,
    and never on a *later* one — the only way this evaluation is honest
    about how the model would have performed at the time."""
    gws = sorted(int(g) for g in training_df["gw"].unique())
    results = []
    for test_gw in gws[1:]:  # first gw has no earlier data to train on
        train_gws = [g for g in gws if g < test_gw]
        train_rows = training_df[training_df["gw"].isin(train_gws)]
        test_rows = training_df[training_df["gw"] == test_gw]
        if len(train_rows) < 20:
            continue

        model = train_linear_model(train_rows, feature_names)
        linear_pred = model.predict(test_rows[feature_names].to_numpy())

        actual = test_rows["actual_points"]
        results.append(
            FoldResult(
                test_gw=test_gw,
                train_gws=train_gws,
                n_train=len(train_rows),
                n_test=len(test_rows),
                naive_r=float(test_rows["naive_points"].corr(actual)),
                heuristic_r=float(test_rows["heuristic_points"].corr(actual)),
                linear_r=float(pd.Series(linear_pred).corr(actual.reset_index(drop=True))),
                naive_mae=float((test_rows["naive_points"] - actual).abs().mean()),
                heuristic_mae=float((test_rows["heuristic_points"] - actual).abs().mean()),
                linear_mae=float(np.abs(linear_pred - actual.to_numpy()).mean()),
            )
        )
    return results


def save_model(model: Pipeline, training_df: pd.DataFrame, fold_results: list[FoldResult], feature_names: list[str] = FEATURE_NAMES) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    scaler: StandardScaler = model.named_steps["scaler"]
    ridge: RidgeCV = model.named_steps["ridge"]
    coefficients = dict(zip(feature_names, ridge.coef_.tolist()))

    meta = {
        "feature_names": feature_names,
        "trained_on_gws": sorted(training_df["gw"].unique().tolist()),
        "n_training_rows": len(training_df),
        "alpha": float(ridge.alpha_),
        "intercept": float(ridge.intercept_),
        "coefficients": coefficients,
        "cross_validation": [
            {
                "test_gw": f.test_gw,
                "train_gws": f.train_gws,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "naive_r": round(f.naive_r, 4),
                "heuristic_r": round(f.heuristic_r, 4),
                "linear_r": round(f.linear_r, 4),
                "naive_mae": round(f.naive_mae, 4),
                "heuristic_mae": round(f.heuristic_mae, 4),
                "linear_mae": round(f.linear_mae, 4),
            }
            for f in fold_results
        ],
    }
    META_PATH.write_text(json.dumps(meta, indent=2))


def load_model() -> tuple[Pipeline, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run `fpl train` first.")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def predict_with_model(
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    summaries: dict[int, dict],
    model: Pipeline,
    start_gw: int,
    num_gws: int = 1,
    feature_names: list[str] = FEATURE_NAMES,
) -> pd.DataFrame:
    """Same shape of output as predict.predict(), but each fixture's points
    come from the trained linear model instead of the scoring-rule formula.
    The per-fixture breakdown only carries a `total` (plus fixture/opponent
    metadata) — a fitted regression doesn't decompose into "X points for
    goals, Y for assists" the way the rule-based heuristic does, so we don't
    fabricate a breakdown that isn't there."""
    from .features import compute_player_form, compute_team_strengths, expected_minutes, position_rate_priors, shrink_form

    team_strength = compute_team_strengths(fixtures_df, teams_df)
    target_gws = set(range(start_gw, start_gw + num_gws))

    raw_forms = {pid: compute_player_form(summaries.get(pid, {}).get("history", [])) for pid in players_df.index}
    forms_by_position: dict[str, list] = {}
    for pid, form in raw_forms.items():
        if form.games_seen > 0:
            forms_by_position.setdefault(players_df.loc[pid, "position"], []).append(form)
    priors = position_rate_priors(forms_by_position)

    exp_points, breakdowns, exp_minutes_col = [], [], []
    for pid, row in players_df.iterrows():
        summary = summaries.get(pid, {})
        history = summary.get("history", [])
        upcoming = summary.get("fixtures", [])
        form = shrink_form(raw_forms[pid], priors[row["position"]])
        minutes = expected_minutes(form, row["status"], row["chance_of_playing_next_round"])
        naive = _naive_prediction(history)

        relevant = [f for f in upcoming if f.get("event") in target_gws]
        total_pts = 0.0
        fixture_breakdowns = []
        for fx in relevant:
            is_home = bool(fx["is_home"])
            opponent_id = fx["team_a"] if is_home else fx["team_h"]
            opp_modifier, n_h2h = opponent_history_modifier(history, opponent_id)
            ctx = fixture_context(form, team_strength, row["team"], opponent_id, is_home, minutes, opp_modifier)
            features = featurize(row["position"], row["price"], form, ctx, naive)
            X = np.array([[features[f] for f in feature_names]])
            pts = float(model.predict(X)[0])
            total_pts += pts
            fixture_breakdowns.append(
                {
                    "total": round(pts, 3),
                    "gameweek": fx["event"],
                    "opponent": teams_df.loc[opponent_id, "short_name"] if opponent_id in teams_df.index else "?",
                    "is_home": is_home,
                    "h2h_sample_size": n_h2h,
                    "opponent_history_modifier": round(opp_modifier, 3),
                }
            )
        exp_points.append(round(total_pts, 3))
        breakdowns.append(fixture_breakdowns)
        exp_minutes_col.append(round(minutes, 1))

    out = players_df.copy()
    out["expected_points"] = exp_points
    out["expected_minutes"] = exp_minutes_col
    out["breakdown"] = breakdowns
    out["num_fixtures"] = out["breakdown"].apply(len)
    return out
