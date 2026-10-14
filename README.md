# LoL Worlds Value Finder

A Python toolkit and Streamlit dashboard for betting on the League of Legends World Championship. It:

1. **Downloads real results** (every top-league and international series since 2025) from the official LoL
   Esports API and builds **Elo ratings**.
2. **Prices** each match from the model and compares it with **Sportsbet** and **PointsBet** decimal odds to find **+EV** bets.
3. **Sizes** stakes with an **optimised fractional Kelly** bankroll manager tuned for a high risk appetite (default bankroll **$500 AUD**).
4. **Reports** picks for upcoming Worlds matches: model win %, the minimum price worth taking and the exact
   stake at each price. There is also an optional Streamlit dashboard.

## Quick start

```bash
pip install -r requirements.txt        # Python 3.11+
python -m lolworlds.fetch              # download real results + upcoming Worlds fixtures (~1 min)
python -m lolworlds.picks              # picks for the next 3 days
```

Optional:

```bash
streamlit run app/dashboard.py         # dashboard (downloads results itself on first run)
python -m lolworlds.ingest             # print the full Elo table, write data/processed/*.csv
python -m pytest                       # run the test suite
```

## Project layout

```
config.toml                    bankroll, bookmakers, risk profiles, Elo parameters, team aliases
app/dashboard.py               Streamlit UI
src/lolworlds/
  config.py                    loads config.toml into typed settings
  fetch.py                     downloads real results + Worlds fixtures from the LoL Esports API
  ingest.py                    data ingestion: native CSV or Oracle's Elixir export -> games, metrics, Elo
  elo.py                       map-level Elo model, Bo1/Bo3/Bo5 series conversion, calibration
  value.py                     de-vig, best price per side, model vs market, +EV selection
  bankroll.py                  simultaneous + fractional Kelly, caps, drawdown brake, Monte Carlo
  ledger.py                    bet ledger (CSV) and bankroll tracking
  picks.py                     text report of picks for upcoming fixtures
data/
  raw/                         downloaded matches.csv + fixtures.csv (git-ignored)
  odds/                        your odds.csv (git-ignored)
  processed/                   generated ratings/history (git-ignored)
  bankroll/                    your bet ledger (git-ignored)
tests/                         pytest suite
```

## Data

### Match history: `data/raw/matches.csv`

`python -m lolworlds.fetch` downloads it from the official LoL Esports API (the one lolesports.com uses). It
covers LCK, LPL, LEC, LCS (and 2025 LTA North), CBLOL (and LTA South), LCP, First Stand, MSI, EWC, the Americas
Cup and Worlds, from 2025 on. Second-tier leagues are left out because their teams rarely play anyone else, which
makes their ratings unreliable. The API gives **series scores only**, so each series becomes individual games with
sides unknown: no blue-side bonus, and no gold/kill stats for the margin-of-victory multiplier.

You can use richer data instead. Any of these formats is detected automatically:

- **Oracle's Elixir** yearly export (`2026_LoL_esports_match_data_from_OraclesElixir.csv` from
  [oracleselixir.com](https://oracleselixir.com/tools/downloads)). Drop it in and point `paths.matches` in
  `config.toml` at it, or run `python -m lolworlds.ingest --input <file>`. To use several years, concatenate the
  files (same columns).
- **Native**: one row per game.

| column | required | notes |
|---|---|---|
| `date` | yes | ISO date/time |
| `league` | yes | `LCK`, `LPL`, `LEC`, `LCS`, `LCP`, `CBLOL`, ... International events: `WLDs`, `MSI`, `FST`, `EWC` |
| `blue_team`, `red_team` | yes | |
| `winner` | yes | team name, or `blue` / `red` |
| `sides_known` | no | `False` if blue/red are just labels (no side bonus applied); default `True` |
| `gd15` | no | blue's gold lead at 15:00 (used for the margin-of-victory multiplier and form stats) |
| `blue_kills`, `red_kills` | no | also feeds the margin-of-victory multiplier |
| `game_id`, `event`, `series_id`, `game_no`, `best_of`, `duration_min`, `*_towers`, `*_dragons`, `*_barons` | no | |

### Odds: `data/odds/odds.csv` (optional)

Sportsbet and PointsBet don't publish a data feed, so prices are never downloaded. Without them, `picks` gives
the **minimum price** worth betting at and the stake at each price, which you compare with the apps yourself.
To get exact recommendations, enter prices here: one row per match per bookmaker. Add rows in the dashboard (**Add a match**, below the slate) or edit the CSV:

```csv
match_id,start_time,event,stage,best_of,team_a,team_b,bookmaker,odds_a,odds_b
W26-QF1,2026-10-24T19:00:00+11:00,Worlds 2026,Quarterfinal,5,Gen.G,Bilibili Gaming,Sportsbet,1.45,2.70
W26-QF1,2026-10-24T19:00:00+11:00,Worlds 2026,Quarterfinal,5,Gen.G,Bilibili Gaming,PointsBet,1.48,2.60
```

- `start_time` needs a UTC offset (AEDT is `+11:00`). The dashboard groups matches by Sydney date.
- Odds are **head-to-head (match winner)** decimal prices. `best_of` must match the market: a Bo1 price is a single-map price.
- If a bookmaker spells a team differently from your match data, add it to `[team_aliases]` in `config.toml`.
  Matching is already case-insensitive. A team the model has never seen is shown but never bet.

## How it works

### 1. Elo model (`elo.py`, `ingest.py`)

- Ratings update **per map**. Series odds come from the map probability *p*, with maps assumed independent:
  Bo3 = p²(3 − 2p), Bo5 = p³(10 − 15p + 6p²).
- **Region priors** seed new teams (LCK/LPL 1650, LEC 1500, LCS 1425, LCP 1395, CBLOL 1350), so the model
  doesn't need dozens of international games to learn the regional gap. A team's home league is inferred from its
  domestic games. These gaps were tuned on 2025–26 results for the best log loss on international games.
- **K-factor** 24, ×1.5 at international events (the only cross-region signal), ×1.5 for a team's first 10 games.
- **Margin of victory** (when per-game stats are available): wins with a big gold lead at 15 minutes or a large
  kill gap move ratings more. A FiveThirtyEight-style correction damps the update when the favourite wins and
  boosts it for upsets.
- **Blue side** gets +15 Elo in historical games where sides are known. Future series are priced side-neutral.
- **Season carry-over**: each new year, ratings regress 25% toward the region prior to account for roster churn.
- On the real 2025–26 data, pre-game predictions pick the winner **63.7%** of the time (log loss 0.637, Brier
  0.223, versus 0.693 / 0.25 for a coin flip), and do slightly better on international games (log loss 0.618).
  `picks`, `ingest` and the dashboard's **Elo ratings** tab all report these. Every parameter is in `config.toml`.

### 2. Value engine (`value.py`)

For each match:

1. Take the **best price for each team** across Sportsbet and PointsBet.
2. Strip each book's margin (the **power** method by default, which corrects favourite-longshot bias) and average
   the results to get a **fair market probability**.
3. Blend: `bet_prob = model_weight × model + (1 − model_weight) × market`. Shrinking toward the market protects
   against the "winner's curse": the bets where your model disagrees most with the market are disproportionately
   the ones where the model is wrong.
4. **EV** = `bet_prob × odds − 1`. A side qualifies if EV ≥ `min_ev`, the odds are inside `[min_odds, max_odds]`,
   and the model is within `max_prob_gap` of the market. A huge gap usually means the model is missing news (a
   roster sub, a patch, a visa issue), not a real edge.
5. At most one side per match: the higher-EV one.

### 3. Bankroll management (`bankroll.py`)

The Kelly criterion maximises long-run bankroll growth. This implementation adds the following for a
same-day slate of bets:

| step | what it does |
|---|---|
| **Simultaneous Kelly** | Bets on the same day share one bankroll, so adding up stand-alone Kelly stakes over-bets. The optimiser maximises expected log-wealth over every win/loss combination of the slate (exact for up to 14 bets). |
| **Fractional multiplier** | Multiply by the profile's `kelly_fraction`, e.g. 0.5 = half Kelly. |
| **Drawdown brake** | If the bankroll falls `drawdown_brake_pct` below its peak, the multiplier is halved until it recovers. |
| **Caps** | `max_stake_pct` per bet, and `max_exposure_pct` across all unsettled bets (including ones already in your ledger). |
| **Rounding** | Round down to $0.50. Drop stakes under $1 (`stake_rounding` / `min_stake`). |

Risk profiles (`config.toml`; switch or fine-tune live in the sidebar):

| profile | Kelly × | model weight | min EV | max stake | max exposure | odds range | max model-vs-market gap |
|---|---|---|---|---|---|---|---|
| conservative | 0.20 | 50% | 5% | 5% | 20% | 1.30–3.50 | 10% |
| balanced | 0.33 | 60% | 4% | 8% | 30% | 1.25–4.00 | 15% |
| **aggressive** (default) | **0.50** | **75%** | **3%** | **15%** | **50%** | **1.20–5.00** | **20%** |
| maximum | 0.75 | 85% | 2% | 25% | 75% | 1.15–8.00 | 30% |

**Why the cap is 1.0× Kelly, even for a high risk appetite:** full Kelly already gives the fastest long-run growth.
Betting more than full Kelly gives *slower* growth *and* a higher risk of ruin, so it is worse on both counts.
At around 2× Kelly, expected growth drops to roughly zero. Model error also means your real Kelly fraction is smaller than the
one computed, so even 1.0× on paper usually over-bets in practice. Half Kelly keeps about 75% of full-Kelly growth
with far smaller drawdowns, which is why it is the aggressive default.

The **Risk simulator** tab replays the day's bets thousands of times at every Kelly multiplier. It shows the
median outcome, the 10th–90th percentile range and the chance of a 50% drawdown. It then suggests the multiplier
with the best median outcome that stays inside your drawdown tolerance. The *edge realised* slider asks what
happens if your edge is only partly real.

## Dashboard tour

- **Sidebar**: ledger bankroll (starting $500 + settled P/L, or override with your live account balances), risk
  profile and fine-tune sliders, data sources, reload.
- **Today's slate**: match day picker (defaults to today in Sydney time), KPIs, a matches table with both books'
  prices (best price highlighted), **recommended bets with exact stakes**, a box to tick the bets you placed and log
  them to the ledger, an "All sides" breakdown explaining why each side was or wasn't bet, and **Add a match**.
- **Elo ratings**: ratings table with form and early-game stats, matchup calculator (fair odds for any Bo1/3/5),
  top-20 chart, calibration table.
- **Bankroll & bets**: P/L, ROI, record, drawdown, bankroll curve, settle open bets (Won / Lost / Void), full
  ledger with CSV download.
- **Risk simulator**: described above.

### Daily workflow

1. Run `python -m lolworlds.fetch` (or press **Download latest results** in the dashboard) to pick up new results.
2. Run `python -m lolworlds.picks` for minimum prices, or enter the day's Sportsbet and PointsBet prices with
   **Add a match** (or in `data/odds/odds.csv`) for exact stakes.
3. Check the recommendations. Read the *All sides* table for anything flagged "check news".
4. Place the bets, tick them in **Log bets you placed** (adjust odds or stake to what you actually got), and save.
5. After the matches, settle them in **Bankroll & bets**. The next day's stakes size off the new bankroll.

## Caveats

- There is no historical odds data, so the strategy's profitability can't be backtested. Calibration shows the
  model is better than a coin flip, not that it beats bookmakers, who also price in information the model lacks.
- An Elo model knows nothing about roster changes, substitutes, patches or meta shifts. Bookmakers do. Treat big
  model-vs-market gaps with suspicion; the `max_prob_gap` filter exists for that reason.
- Series prices assume maps are independent. Real maps within a series are correlated (draft adaptation,
  momentum, tilt), so Bo3/Bo5 prices can be off in either direction.
- Australian bookmakers can limit accounts that win consistently. Stake limits may cap what you can actually get on.
- Past model accuracy (see calibration) doesn't guarantee future returns.

**Gamble responsibly.** Only bet what you can afford to lose, and set deposit limits in your Sportsbet and
PointsBet accounts. Free, confidential support is available 24/7 from Gambling Help Online on **1800 858 858**
([gamblinghelponline.org.au](https://www.gamblinghelponline.org.au)). You can self-exclude from all licensed
Australian wagering providers through [BetStop](https://www.betstop.gov.au).
