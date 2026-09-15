"""Squad selection as an integer program.

Picking 15 players under a budget, position quotas, and a 3-per-club cap to
maximise expected points is a knapsack-family problem — small enough (~700
players) to solve to proven optimality with a general MILP solver rather than
reach for a heuristic. Google OR-Tools' bundled CBC backend needs no separate
install and ships native binaries for both Intel and Apple Silicon.

Two problems are modelled, both built on the same underlying ILP:
  - build_squad:        pick a full 15-man squad from scratch under budget.
  - optimize_transfers:  starting from an existing squad, pick the best set
                         of transfers within a free-transfer count (extra
                         transfers cost 4 points each, so the search directly
                         maximises expected-points-minus-hits).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from ortools.linear_solver import pywraplp

from .data import SquadRules

POINTS_HIT_PER_TRANSFER = 4
BENCH_WEIGHT = 0.1  # bench players count a little toward the objective (autosub value)


@dataclass
class Squad:
    squad_ids: list[int]  # all 15
    starting_ids: list[int]  # 11
    bench_ids: list[int]  # 4, in substitution order
    captain_id: int
    vice_captain_id: int
    total_cost: float
    expected_points: float  # starting XI + captain double, bench excluded

    def summary(self, players_df: pd.DataFrame) -> pd.DataFrame:
        cols = ["name", "position", "team_name", "price", "expected_points"]
        df = players_df.loc[self.squad_ids, cols].copy()
        df["role"] = df.index.map(
            lambda i: "C" if i == self.captain_id else "VC" if i == self.vice_captain_id else ""
        )
        df["starting"] = df.index.isin(self.starting_ids)
        return df.sort_values(["starting", "position"], ascending=[False, True])


def _pick_captain_vice(players_df: pd.DataFrame, starting_ids: list[int]) -> tuple[int, int]:
    ranked = players_df.loc[starting_ids].sort_values("expected_points", ascending=False)
    return ranked.index[0], ranked.index[1]


def _solve_squad_ilp(
    players_df: pd.DataFrame,
    rules: SquadRules,
    budget: float,
    locked_ids: list[int],
    max_new_players: int | None,
) -> Squad:
    """The shared ILP core: pick 15 + a starting XI maximising expected
    points, under budget/position/club constraints. `max_new_players` (when
    given) caps how many picks may fall outside `locked_ids` — used by the
    transfer optimizer to limit how many players change; `build_squad` leaves
    it uncapped.
    """
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("could not create an OR-Tools MILP solver (CBC backend unavailable)")

    ids = list(players_df.index)
    in_squad = {i: solver.BoolVar(f"squad_{i}") for i in ids}
    in_starting = {i: solver.BoolVar(f"start_{i}") for i in ids}

    for i in ids:
        solver.Add(in_starting[i] <= in_squad[i])  # starting XI is a subset of the squad

    solver.Maximize(
        solver.Sum(
            in_starting[i] * players_df.loc[i, "expected_points"]
            + (in_squad[i] - in_starting[i]) * players_df.loc[i, "expected_points"] * BENCH_WEIGHT
            for i in ids
        )
    )

    solver.Add(solver.Sum(in_squad[i] for i in ids) == rules.squad_size)
    solver.Add(solver.Sum(in_starting[i] for i in ids) == rules.starting_size)
    solver.Add(solver.Sum(in_squad[i] * players_df.loc[i, "price"] for i in ids) <= budget)

    for pos, count in rules.squad_count.items():
        pos_ids = players_df.index[players_df["position"] == pos]
        solver.Add(solver.Sum(in_squad[i] for i in pos_ids) == count)
        solver.Add(solver.Sum(in_starting[i] for i in pos_ids) >= rules.starting_min[pos])
        solver.Add(solver.Sum(in_starting[i] for i in pos_ids) <= rules.starting_max[pos])

    for team_id in players_df["team"].unique():
        team_ids = players_df.index[players_df["team"] == team_id]
        solver.Add(solver.Sum(in_squad[i] for i in team_ids) <= rules.max_per_team)

    if max_new_players is not None:
        locked_set = set(locked_ids)
        solver.Add(solver.Sum(in_squad[i] for i in ids if i not in locked_set) <= max_new_players)
    else:
        for i in locked_ids:
            solver.Add(in_squad[i] == 1)

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"solver did not find an optimal squad (status={status})")

    squad_ids = [i for i in ids if in_squad[i].solution_value() > 0.5]
    starting_ids = [i for i in ids if in_starting[i].solution_value() > 0.5]
    bench_ids = sorted(
        (i for i in squad_ids if i not in starting_ids),
        key=lambda i: -players_df.loc[i, "expected_points"],
    )
    captain_id, vice_id = _pick_captain_vice(players_df, starting_ids)
    starting_points = sum(players_df.loc[i, "expected_points"] for i in starting_ids)
    starting_points += players_df.loc[captain_id, "expected_points"]  # captain doubles

    return Squad(
        squad_ids=squad_ids,
        starting_ids=starting_ids,
        bench_ids=bench_ids,
        captain_id=captain_id,
        vice_captain_id=vice_id,
        total_cost=round(sum(players_df.loc[i, "price"] for i in squad_ids), 1),
        expected_points=round(starting_points, 2),
    )


def build_squad(
    players_df: pd.DataFrame,
    rules: SquadRules,
    budget: float | None = None,
    locked_ids: list[int] | None = None,
    excluded_ids: list[int] | None = None,
) -> Squad:
    """Choose the 15-man squad (and best starting XI + captain within it)
    that maximises expected points under budget, quota, and club-limit
    constraints, all read from `rules` rather than hardcoded.
    """
    budget = rules.budget if budget is None else budget
    candidates = players_df[~players_df.index.isin(excluded_ids or [])]
    return _solve_squad_ilp(candidates, rules, budget, locked_ids or [], max_new_players=None)


@dataclass
class TransferPlan:
    transfers_out: list[int]
    transfers_in: list[int]
    new_squad: Squad
    points_hit: int
    net_expected_points_gain: float


def optimize_transfers(
    current_squad_ids: list[int],
    players_df: pd.DataFrame,
    rules: SquadRules,
    bank: float,
    free_transfers: int,
    max_transfers: int = 5,
) -> TransferPlan:
    """Search over 0..max_transfers changes to find the transfer count and
    combination that maximises (expected points gained) - (4 * paid hits),
    respecting the budget (squad value + bank) and all squad rules.

    This re-solves the squad ILP once per candidate transfer count
    (0..max_transfers), each time capping how many players may come from
    outside the current squad, then keeps whichever count nets the highest
    score. That's a handful of small MILP solves rather than one huge
    combinatorial search.
    """
    current_value = players_df.loc[current_squad_ids, "price"].sum()
    total_budget = round(current_value + bank, 1)

    best: TransferPlan | None = None
    for n_transfers in range(0, max_transfers + 1):
        squad = _solve_squad_ilp(players_df, rules, total_budget, current_squad_ids, max_new_players=n_transfers)
        transfers_out = [i for i in current_squad_ids if i not in squad.squad_ids]
        transfers_in = [i for i in squad.squad_ids if i not in current_squad_ids]
        hits = max(0, len(transfers_out) - free_transfers) * POINTS_HIT_PER_TRANSFER
        net = squad.expected_points - hits
        if best is None or net > best.net_expected_points_gain:
            best = TransferPlan(
                transfers_out=transfers_out,
                transfers_in=transfers_in,
                new_squad=squad,
                points_hit=hits,
                net_expected_points_gain=round(net, 2),
            )
    assert best is not None
    return best
