"""Unit tests for the learned-model pipeline.

These test the ML mechanics directly (featurization shape, fit/predict,
forward-chaining fold logic) on small synthetic data rather than exercising
the full build_training_set integration, which needs a realistic FPL data
shape (players/teams/fixtures/summaries) to mean anything — that path is
covered by running `fpl train` against live data instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_optimizer.features import PlayerForm
from fpl_optimizer.ml import FEATURE_NAMES, evaluate_forward_chaining, featurize, train_linear_model

DUMMY_FORM = PlayerForm(
    minutes_per_game=80,
    start_rate=0.9,
    goals_per_90=0.3,
    assists_per_90=0.2,
    xg_per_90=0.35,
    xa_per_90=0.18,
    bps_per_90=22,
    defensive_actions_per_90=8,
    saves_per_90=0,
    yellow_per_90=0.1,
    red_per_90=0.0,
    games_seen=4,
    sample_minutes=320,
)

DUMMY_CTX = {
    "minutes_share": 0.9,
    "is_home": 1.0,
    "opponent_defence_leakiness": 1.1,
    "opponent_attack": 0.95,
    "attack_multiplier": 1.05,
    "expected_goals": 0.3,
    "expected_assists": 0.15,
    "p_clean_sheet": 0.35,
    "expected_actions": 7.0,
    "expected_bps": 20.0,
    "opponent_history_modifier": 1.0,
}


def test_featurize_returns_every_declared_feature():
    features = featurize("MID", price=7.5, form=DUMMY_FORM, ctx=DUMMY_CTX, naive_points=4.2)
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert features["price"] == 7.5
    assert features["is_mid"] == 1.0
    assert features["is_def"] == 0.0
    assert features["naive_points"] == 4.2


def _synthetic_training_df(n_per_gw: int = 30, n_gws: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(2, 2 + n_gws):
        for i in range(n_per_gw):
            row = {name: float(rng.normal()) for name in FEATURE_NAMES}
            row["is_gkp"] = 0.0
            # actual points genuinely driven by a couple of features, plus noise —
            # so a fitted model has real signal to recover, not just overfit noise
            row["actual_points"] = 2.0 + 3.0 * row["minutes_share"] + 1.5 * row["expected_goals"] + rng.normal(scale=0.5)
            row["naive_points"] = row["actual_points"] + rng.normal(scale=1.0)
            row["heuristic_points"] = row["actual_points"] + rng.normal(scale=1.5)
            row["gw"] = gw
            row["player_id"] = i
            row["position"] = "MID"
            rows.append(row)
    return pd.DataFrame(rows)


def test_train_linear_model_fits_and_predicts():
    training_df = _synthetic_training_df()
    model = train_linear_model(training_df)
    preds = model.predict(training_df[FEATURE_NAMES].to_numpy())
    assert len(preds) == len(training_df)
    # should recover a meaningfully positive relationship with the two true drivers
    residual_corr = np.corrcoef(preds, training_df["actual_points"])[0, 1]
    assert residual_corr > 0.5


def test_evaluate_forward_chaining_never_trains_on_future_or_current_gw():
    training_df = _synthetic_training_df(n_gws=3)
    folds = evaluate_forward_chaining(training_df)

    gws_present = sorted(training_df["gw"].unique())
    assert [f.test_gw for f in folds] == gws_present[1:]  # first gw can never be tested (no prior data)
    for f in folds:
        assert all(g < f.test_gw for g in f.train_gws)
        assert f.n_train == len(training_df[training_df["gw"] < f.test_gw])


def test_evaluate_forward_chaining_skips_folds_with_too_little_training_data():
    training_df = _synthetic_training_df(n_per_gw=5, n_gws=2)  # 5 rows in the only training fold, below the floor
    folds = evaluate_forward_chaining(training_df)
    assert folds == []
