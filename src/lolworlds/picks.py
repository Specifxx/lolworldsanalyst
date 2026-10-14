"""Picks report for upcoming Worlds fixtures, no dashboard needed.

For each fixture: model win probabilities, fair odds, the minimum price worth
betting each side at, and the exact stake at a ladder of prices. If you have
real Sportsbet/PointsBet prices in ``data/odds/odds.csv``, those matches get
exact recommendations from the full value engine instead.

    python -m lolworlds.fetch          # refresh results + fixtures first
    python -m lolworlds.picks          # next 3 days
    python -m lolworlds.picks --days 7 --bankroll 540
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .bankroll import SizingContext, recommend_stakes
from .config import RiskProfile, Settings, load_settings
from .elo import EloModel, fair_odds, series_win_prob
from .ingest import build_ratings
from .ledger import Ledger, bankroll_state
from .value import clean_odds, find_value, resolve_team

# Typical two-way margin on esports head-to-head markets, used to estimate the
# market's fair probability when only one side's price is known.
ASSUMED_MARGIN = 0.05


def single_price_check(model_p: float, odds: float, profile: RiskProfile) -> tuple[bool, float, float]:
    """(qualifies, bet_prob, ev) for one side at one price, estimating the market from that price."""
    market_p = min(1.0, 1.0 / (odds * (1 + ASSUMED_MARGIN)))
    bet_p = profile.model_weight * model_p + (1 - profile.model_weight) * market_p
    ev = bet_p * odds - 1
    ok = (ev >= profile.min_ev and profile.min_odds <= odds <= profile.max_odds
          and abs(model_p - market_p) <= profile.max_prob_gap)
    return ok, bet_p, ev


def price_ladder(model_p: float, profile: RiskProfile, ctx: SizingContext, steps: int = 5) -> list[dict]:
    """Stake at the minimum qualifying price and a few prices above it."""
    grid = np.round(np.arange(max(1.01, profile.min_odds), profile.max_odds + 0.001, 0.01), 2)
    ok = [o for o in grid if single_price_check(model_p, o, profile)[0]]
    if not ok:
        return []
    lo, hi = ok[0], ok[-1]
    points = sorted({round(float(x), 2) for x in np.linspace(lo, min(hi, lo * 1.35), steps)})
    rows = []
    for o in points:
        _, bet_p, ev = single_price_check(model_p, o, profile)
        bet = pd.DataFrame([{"bet_prob": bet_p, "odds": o, "ev": ev}])
        stake = recommend_stakes(bet, profile, ctx).stake.iloc[0]
        rows.append({"odds": o, "ev": ev, "stake": float(stake)})
    return rows


def _match_odds(fixture: pd.Series, odds: pd.DataFrame, model: EloModel, settings: Settings) -> pd.DataFrame:
    """Rows of the odds file that price this fixture (by match id, or by teams within a day)."""
    if odds.empty:
        return odds
    by_id = odds[odds.match_id.astype(str) == str(fixture.match_id)]
    if not by_id.empty:
        return by_id
    known = set(model.teams)
    want = {resolve_team(fixture.team_a, known, settings.team_aliases),
            resolve_team(fixture.team_b, known, settings.team_aliases)}
    pair = [
        {resolve_team(a, known, settings.team_aliases), resolve_team(b, known, settings.team_aliases)} == want
        for a, b in zip(odds.team_a, odds.team_b)
    ]
    close = (odds.start_time - fixture.start_time).abs() <= pd.Timedelta(days=1)
    return odds[pd.Series(pair, index=odds.index) & close]


def build_report(settings: Settings, profile: RiskProfile, bankroll: float, peak: float, open_exposure: float,
                 days: int = 3, now: pd.Timestamp | None = None) -> str:
    ratings = build_ratings(settings)
    model = ratings.model
    tz = settings.timezone
    now = now or pd.Timestamp.now(tz=tz)

    fixtures = pd.read_csv(settings.paths.fixtures) if settings.paths.fixtures.exists() else pd.DataFrame()
    if not fixtures.empty:
        fixtures["start_time"] = pd.to_datetime(fixtures.start_time, utc=True).dt.tz_convert(tz)
        fixtures = fixtures[(fixtures.start_time >= now - pd.Timedelta(hours=3))
                            & (fixtures.start_time <= now + pd.Timedelta(days=days))]
    odds = clean_odds(pd.read_csv(settings.paths.odds), settings) if settings.paths.odds.exists() else pd.DataFrame()
    ctx = SizingContext(bankroll=bankroll, peak_bankroll=peak, open_exposure=open_exposure,
                        stake_rounding=settings.stake_rounding, min_stake=settings.min_stake)

    cal = ratings.calibration
    lines = [
        f"# Worlds picks - {now:%a %d %b %Y, %H:%M} {tz}",
        "",
        f"Bankroll **${bankroll:,.2f}** · profile **{profile.name}** ({profile.kelly_fraction:.2f}x Kelly, "
        f"max {profile.max_stake_pct:.0%} per bet) · model trained on {len(ratings.games):,} real games to "
        f"{ratings.games.date.max():%d %b %Y} (accuracy {cal['accuracy']:.1%}, log loss {cal['log_loss']:.3f})",
        "",
    ]
    if fixtures.empty:
        lines.append(f"No Worlds fixtures with both teams confirmed in the next {days} days.")
        return "\n".join(lines)

    known = set(model.teams)
    for fx in fixtures.sort_values("start_time").itertuples(index=False):
        a = resolve_team(fx.team_a, known, settings.team_aliases)
        b = resolve_team(fx.team_b, known, settings.team_aliases)
        lines += [f"## {fx.team_a} vs {fx.team_b}", "",
                  f"{fx.start_time:%a %d %b, %I:%M %p} · {fx.event} {fx.stage} · Bo{int(fx.best_of)}", ""]
        if a is None or b is None:
            lines += ["Not enough data on one of these teams to price this match - no bet.", ""]
            continue
        p_map = model.predict_map(a, b)
        p = series_win_prob(p_map, int(fx.best_of))
        lines += [
            "| | Elo | Model win % | Fair odds |", "|---|---|---|---|",
            f"| {fx.team_a} | {model.rating(a):.0f} | {p:.1%} | {fair_odds(p):.2f} |",
            f"| {fx.team_b} | {model.rating(b):.0f} | {1 - p:.1%} | {fair_odds(1 - p):.2f} |",
            "",
        ]

        quoted = _match_odds(pd.Series(fx._asdict()), odds, model, settings)
        if not quoted.empty:
            _, sides, bets = find_value(quoted, model, settings, profile)
            stakes = recommend_stakes(bets, profile, ctx)
            if stakes.empty or (stakes.stake <= 0).all():
                lines += ["**With your entered prices: no bet.** "
                          + "; ".join(f"{r.selection} @ {r.odds:.2f}: {r.reason or 'no edge'}" for r in sides.itertuples()), ""]
            for r in stakes[stakes.stake > 0].itertuples():
                lines += [f"**BET: {r.selection} @ {r.odds:.2f} on {r.bookmaker} - stake ${r.stake:.2f}** "
                          f"(EV {r.ev:+.1%}, returns ${r.to_return:.2f})", ""]
            continue

        for team, side_p in ((fx.team_a, p), (fx.team_b, 1 - p)):
            ladder = price_ladder(side_p, profile, ctx)
            if not ladder:
                lines.append(f"- **{team}**: no bet at any realistic price.")
                continue
            steps = ", ".join(f"{r['odds']:.2f} → ${r['stake']:.2f}" for r in ladder)
            lines.append(f"- **{team}**: bet only at **{ladder[0]['odds']:.2f} or better**. Stake by price: {steps}")
        lines.append("")

    lines += [
        "---",
        "How to use: take the better of Sportsbet/PointsBet. If neither reaches the minimum price, skip the match. "
        "Stakes assume one bet per match. If you take several bets on the same day, send me the prices and I'll size "
        "them together. The model knows results only - not roster subs, patches or news.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print picks for upcoming Worlds fixtures.")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--bankroll", type=float, help="override the ledger bankroll")
    parser.add_argument("--profile", help="risk profile name (default from config.toml)")
    parser.add_argument("--config")
    parser.add_argument("--out", help="also write the report to this markdown file")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    profile = settings.risk_profiles[args.profile or settings.risk_profile]
    state = bankroll_state(Ledger(settings.paths.ledger).load(), settings.starting_bankroll)
    bankroll = args.bankroll if args.bankroll is not None else state.bankroll
    report = build_report(settings, profile, bankroll, max(state.peak, bankroll), state.open_exposure, args.days)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")


if __name__ == "__main__":
    main()
