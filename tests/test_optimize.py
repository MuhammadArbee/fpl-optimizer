"""Constraint-correctness tests for the squad-selection ILP.

These use small synthetic player pools rather than live FPL data, so they
run fast and deterministically and don't depend on network access.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_optimizer.data import SquadRules
from fpl_optimizer.optimize import build_squad, optimize_transfers

RULES = SquadRules(
    budget=100.0,
    squad_size=15,
    starting_size=11,
    max_per_team=3,
    squad_count={"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
    starting_min={"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1},
    starting_max={"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3},
)


def _make_pool() -> pd.DataFrame:
    """6 teams x enough players per position that the budget and per-team
    cap actually bind, with a wide spread of price/expected_points so the
    optimizer has real trade-offs to make. Needs at least 5 teams so that
    15 players are reachable at all under a 3-per-team cap."""
    rows = []
    pid = 0
    positions = {"GKP": 3, "DEF": 8, "MID": 8, "FWD": 5}
    for team in ["A", "B", "C", "D", "E", "F"]:
        for pos, count in positions.items():
            for k in range(count):
                pid += 1
                price = 4.0 + (pid % 10) * 0.9
                xp = 2.0 + (pid % 13) * 0.7
                rows.append(
                    {
                        "id": pid,
                        "name": f"{pos}_{team}_{k}",
                        "position": pos,
                        "team": team,
                        "team_name": team,
                        "price": round(price, 1),
                        "expected_points": round(xp, 2),
                    }
                )
    return pd.DataFrame(rows).set_index("id", drop=False)


def test_build_squad_respects_all_constraints():
    pool = _make_pool()
    squad = build_squad(pool, RULES)

    assert len(squad.squad_ids) == 15
    assert len(squad.starting_ids) == 11
    assert len(squad.bench_ids) == 4
    assert squad.total_cost <= RULES.budget + 1e-6

    picked = pool.loc[squad.squad_ids]
    counts = picked["position"].value_counts().to_dict()
    assert counts == RULES.squad_count

    team_counts = picked["team"].value_counts()
    assert (team_counts <= RULES.max_per_team).all()

    starting = pool.loc[squad.starting_ids]
    start_counts = starting["position"].value_counts().to_dict()
    for pos, count in start_counts.items():
        assert RULES.starting_min[pos] <= count <= RULES.starting_max[pos]

    assert squad.captain_id in squad.starting_ids
    assert squad.vice_captain_id in squad.starting_ids
    assert squad.captain_id != squad.vice_captain_id
    # captain should be the highest-xP starter (gets doubled, so must be the best pick)
    best_starter = starting["expected_points"].idxmax()
    assert squad.captain_id == best_starter


def test_build_squad_prefers_strictly_better_player():
    """If player B is cheaper AND scores higher than player A in the same
    slot, an optimal solver must never prefer A when both are affordable."""
    pool = _make_pool()
    dominated_id = pool[pool["position"] == "MID"].index[0]
    pool.loc[dominated_id, "price"] = 20.0
    pool.loc[dominated_id, "expected_points"] = 0.1

    squad = build_squad(pool, RULES)
    assert dominated_id not in squad.squad_ids


def test_build_squad_infeasible_budget_raises():
    pool = _make_pool()
    with pytest.raises(RuntimeError):
        build_squad(pool, RULES, budget=1.0)


def test_optimize_transfers_takes_free_upgrade():
    pool = _make_pool()
    starting_squad = build_squad(pool, RULES).squad_ids

    # a clear upgrade: cheaper AND much higher expected points than every
    # player currently in the squad, in a position/team that has room.
    upgrade_id = pool.index.max() + 1
    a_mid_out = pool.loc[starting_squad][pool.loc[starting_squad, "position"] == "MID"].index[0]
    pool.loc[upgrade_id] = {
        "id": upgrade_id,
        "name": "Super Sub",
        "position": "MID",
        "team": pool.loc[a_mid_out, "team"],
        "team_name": pool.loc[a_mid_out, "team"],
        "price": pool.loc[a_mid_out, "price"],
        "expected_points": pool["expected_points"].max() + 5,
    }

    plan = optimize_transfers(starting_squad, pool, RULES, bank=0.0, free_transfers=1, max_transfers=1)
    assert upgrade_id in plan.transfers_in
    assert plan.points_hit == 0
    assert plan.net_expected_points_gain > 0
