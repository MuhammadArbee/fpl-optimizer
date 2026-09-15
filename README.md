# FPL Squad Optimizer

Picks the best possible Fantasy Premier League squad under the £100m budget
by combining a from-scratch expected-points model with an integer
programming solver. Given live data from the official FPL API, it answers
two questions every manager actually has:

- **"What's the best 15 I can build this budget?"**
- **"What transfers should I make to my existing team?"**

Not affiliated with the Premier League or Fantasy Premier League — it's a
personal project built against their public, unauthenticated API.

## Why this is a harder problem than it looks

Two separate problems are bundled into "who should I pick":

1. **Prediction** — how many points will each of ~700 players score next
   gameweek? This needs form, underlying quality (xG/xA), who they're
   playing, whether that opponent is easy to score against, and whether
   they'll actually get minutes.
2. **Selection** — given those predictions, which 15 players, at which
   prices, actually fit inside a budget, a 2/5/5/3 position quota, and a
   3-per-club limit, while also picking the best possible starting XI and
   captain from within that squad? This is a knapsack-family combinatorial
   optimization problem — checking every combination is intractable, but it
   solves in under a second as an integer program.

This project treats them as two separate, testable stages: `predict.py`
produces an expected-points number (with a full breakdown, not a black box),
and `optimize.py` finds the provably-optimal squad given those numbers.

## The expected-points model

For every player and every fixture in the requested gameweek window
(`predict.py`):

| Factor | Source | How it's used |
|---|---|---|
| Recent output | Match-by-match history this season | Recency-weighted (half-life ~5 games) goals, assists, xG, xA, BPS, cards, saves, defensive actions per 90 |
| Underlying vs. actual | FPL's own `expected_goals`/`expected_assists` | Blended 60/40 with actual goals/assists — xG is less noisy short-term, but real finishing ability matters too |
| Fixture difficulty | Team-level attack/defence ratings, computed from this season's actual results (goals scored/conceded, home/away split), Poisson-style, shrunk toward FPL's preseason strength rating early in the season when results are still a small sample | Scales expected goals/assists up against weak defences, down against strong ones |
| Clean sheets | Same team ratings | Poisson P(0 goals conceded) — a team-level property, not tied to any one player |
| Minutes risk | Recent start rate + FPL's `chance_of_playing_next_round`/injury status | Expected minutes, which scale every per-fixture stat and gate clean-sheet eligibility (<60 min = no CS points) |
| Small-sample noise | — | Per-90 rates are shrunk toward the position's minutes-weighted average, so e.g. one lucky goal in 15 minutes doesn't extrapolate into an absurd rate |
| Defensive contribution | Tackles + clearances/blocks/interceptions per 90 | P(actions ≥ threshold) via Poisson, using the 2025/26+ scoring rule (10 for DEF, 12 for MID/FWD) |
| Bonus points | Recent BPS rate | Rough proxy: points scale above a BPS baseline |
| Opponent history | This player's record against this specific opponent, this season | A small, heavily-shrunk nudge (capped ±15%) — sample sizes of 0-2 games are treated as mostly noise |
| Scoring rules | Read live from the API's own `game_config.scoring` | Not hardcoded — this season introduced 10-point GK goals and defensive-contribution points, and the model picks that up automatically rather than assuming last season's rules |

Every number is explainable: `fpl player "name"` and the dashboard's Player
Explorer tab print the full points breakdown per fixture, not just a final
score.

## The optimizer

`optimize.py` models squad selection as a mixed-integer program (via Google
OR-Tools' CBC backend) rather than a greedy heuristic, so the result is
*provably* the best 15-man squad + starting XI + captain achievable under:

- Budget (read from the API — currently £100m)
- Exact position quotas (2 GKP / 5 DEF / 5 MID / 3 FWD)
- Valid starting-XI formations (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD)
- Max 3 players per real-world club

`optimize_transfers` reuses the same ILP to answer "given my current 15,
what should I change?" — it solves the squad problem once per candidate
transfer count (0 through 5), capping how many players may come from outside
the current squad each time, then picks whichever count maximises expected
points minus the `-4`-per-transfer hit for going over your free transfers.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CLI

```bash
fpl squad                              # best 15 under a £100m budget, next GW
fpl squad --horizon 3                  # weigh the next 3 gameweeks instead of 1
fpl squad --budget 95                  # a tighter budget
fpl transfers --team-id 1234567        # transfer suggestions for a real FPL team
fpl player "Salah"                     # explain one player's expected points
fpl fetch                              # force-refresh the local data cache
```

(`fpl` isn't installed as a standalone command — run these as
`python -m fpl_optimizer.cli <command>`, or `pip install -e .` first.)

Your FPL team ID is the number in the URL when you view your team on
fantasy.premierleague.com (e.g. `.../entry/1234567/event/5`).

### Dashboard

```bash
streamlit run app/dashboard.py
```

Gives you an interactive pitch view of the optimal squad, a sortable table
of every player's expected points, and a per-player breakdown explorer.

### Tests

```bash
pytest
```

Tests use small synthetic player pools (no network access needed) to check
the optimizer actually respects every constraint — budget, position quotas,
club limits, valid formations — and that it never leaves a strictly better,
affordable player on the bench.

## Project layout

```
fpl_optimizer/
  api.py        FPL API client with disk caching
  data.py       raw JSON -> DataFrames; reads squad/scoring rules live from the API
  features.py   team strength ratings, player form, minutes model, shrinkage
  predict.py    the expected-points model
  optimize.py   the ILP squad builder and transfer optimizer
  cli.py        command-line interface
app/
  dashboard.py  Streamlit UI
tests/
  test_optimize.py, test_features.py
```

## Known limitations / natural next steps

- **Opponent history** only looks at this season's meetings (the API's
  match-by-match log doesn't go back further); a multi-season head-to-head
  signal would need an external historical dataset.
- **Bonus points** are a rough BPS-based proxy, not a simulation of the
  actual top-3-in-match BPS ranking.
- **No price-change or long-term squad planning** (e.g. holding a transfer
  for a better week, chip strategy) — each run is a fresh optimization, not
  a multi-week plan.
- The model has not been backtested against actual gameweek outcomes; it's
  built on sound statistical principles (Poisson goal models, recency
  weighting, Bayesian shrinkage) but hasn't been validated for calibration.
