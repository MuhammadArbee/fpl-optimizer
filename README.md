# FPL Squad Optimizer

Picks the best possible Fantasy Premier League squad under the £100m budget
by combining a from-scratch expected-points model with an integer
programming solver. Given live data from the official FPL API, it answers
two questions every manager actually has:

- **"What's the best 15 I can build this budget?"**
- **"What transfers should I make to my existing team?"**
- **"How is each player projected to score, gameweek by gameweek, for the rest of the season?"**

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
fpl squad --horizon 3                  # weigh the next 3 gameweeks (reports a TOTAL across all 3, not a per-GW figure)
fpl squad --budget 95                  # a tighter budget
fpl season                             # project every remaining gameweek this season, GW-by-GW, for your optimal squad
fpl season --top 20                    # skip squad-building; just rank the top 20 players by season-long total
fpl season --full --csv out.csv        # print every gameweek's column, and export the full player x gameweek matrix
fpl transfers --team-id 1234567        # transfer suggestions for a real FPL team
fpl player "Salah"                     # explain one player's expected points, gameweek by gameweek
fpl fetch                              # force-refresh the local data cache
```

`--horizon` sums predicted points across N gameweeks into a single figure —
useful for "who's best over this run of fixtures," but easy to misread as a
single-week score once N gets large. `fpl season` is the one that actually
answers "predict every gameweek" — it never lumps weeks together, always
reporting one column per gameweek plus a season total and a per-GW average
for comparison. It assumes the same 15 players are kept all season (no
transfers along the way), so treat far-future gameweeks as a rough guide —
prices, injuries, and form will all move between now and then.

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

### Backtesting

```bash
fpl backtest                    # test against every finished gameweek so far
fpl backtest --gws 3 4          # test specific gameweeks
```

This rebuilds predictions using *only* data that would genuinely have been
available before each gameweek (match history up to that point, team
strength from earlier results only) and checks them against what actually
happened — the honest test of whether the model is worth anything, not just
a demo of it running. It reports:

- **Point-level accuracy**: Pearson/Spearman correlation and mean absolute
  error between predicted and actual points, benchmarked against a naive
  baseline (each player's own season-to-date average, with no fixture or
  form-recency information at all).
- **Squad-level validation**: what the optimizer's recommended squad would
  actually have scored that gameweek, against the average and highest
  scores real FPL managers achieved (both reported by the API itself).

**Results as measured against GW2-4 of the 2026/27 season** (the only
finished gameweeks so far):

| GW | Model r | Naive r | Model MAE | Naive MAE | Squad pts | Avg. manager |
|----|---------|---------|-----------|-----------|-----------|--------------|
| 2  | 0.310   | 0.411   | 1.93      | 1.35      | 49        | 81           |
| 3  | 0.397   | 0.420   | 1.85      | 1.33      | 53        | 51           |
| 4  | 0.392   | 0.483   | 1.97      | 1.35      | 46        | 69           |

Told straight: **over this tiny sample, the full model does not yet beat
the naive "just use their recent average" baseline**, and the backtested
squads underperformed the average real manager in 2 of 3 gameweeks. I
looked into why rather than just reporting the number:

- I tried strengthening the small-sample shrinkage (the mechanism that
  prevents one lucky early game from dominating a player's rate) across a
  wide range of settings. It didn't help — correlation was flat to
  slightly worse, and squad points got worse as shrinkage increased. So
  the gap isn't an undershrunk-outlier bug with an easy fix.
- The more likely cause: this is GW2-4 of a new season. The team-level
  attack/defence ratings that drive the fixture-difficulty and clean-sheet
  logic are themselves built from only 1-3 finished matches per team at
  that point — too little to add real signal, so they mostly add noise on
  top of a simpler "recent form" signal the naive baseline already
  captures well. A model that leans on fixture context should earn its
  keep *more*, not less, as the season's sample size grows.
- 3 gameweeks is also just a very small, high-variance sample for a sport
  with this much single-match randomness (one red card or deflection swings
  a result by several points) — not enough to distinguish a genuinely worse
  model from bad luck.

The honest conclusion: **the model is not yet validated as better than a
trivial baseline**, and this needs re-checking once more gameweeks have
been played. I'm leaving the backtest command in specifically so that claim
is checkable, not asserted. Re-run `fpl backtest` periodically as the season
progresses — if the gap doesn't close, that's a real signal to revisit
`predict.py`'s fixture-difficulty weighting rather than trust it by default.

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
  backtest.py   validates predictions against already-finished gameweeks
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
- The model **has** been backtested (see above) — and, honestly, hasn't yet
  demonstrated it beats a naive baseline over the 3 gameweeks available so
  far this season. It's built on sound statistical principles (Poisson goal
  models, recency weighting, Bayesian shrinkage) but early-season team
  strength ratings are themselves low-sample, which likely blunts the
  fixture-difficulty signal the model leans on. Worth re-running `fpl
  backtest` as more gameweeks accumulate before trusting its picks blindly.
