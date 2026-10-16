"""Picks report for upcoming Worlds matches, no dashboard needed.

Pulls every PointsBet LoL Worlds market automatically, prices the series markets
(match result, maps handicap, total maps, correct score, map winner) from the
Elo model, and sizes the bets jointly. Matches PointsBet hasn't listed yet get
the minimum price worth betting at and the stake at a ladder of prices.

    python -m lolworlds.fetch                     # refresh results + fixtures first
    python -m lolworlds.picks                     # PointsBet markets + next 3 days of fixtures
    python -m lolworlds.picks --link <pointsbet link> [--link ...]
    python -m lolworlds.picks --bankroll 540 --no-pointsbet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import pointsbet
from .bankroll import SizingContext, recommend_stakes
from .config import RiskProfile, Settings, load_settings
from .elo import EloModel, fair_odds
from .ingest import build_ratings
from .ledger import Ledger, bankroll_state
from .markets import KIND_LABELS, MatchMarkets, plan_bets, price_match
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


def analyse_markets(matches: list[MatchMarkets], model: EloModel, settings: Settings, profile: RiskProfile,
                    ctx: SizingContext, skip_match_ids: set[str] = frozenset()) -> tuple[dict, pd.DataFrame]:
    """Price every selection of every match, then size the qualifying bets jointly.

    Returns ({match_id: (selection table, info)}, stakes table).
    """
    known = set(model.teams)
    priced = {}
    for m in matches:
        a = resolve_team(m.team_a, known, settings.team_aliases)
        b = resolve_team(m.team_b, known, settings.team_aliases)
        p_map = model.predict_map(a, b) if a and b else None
        table, info = price_match(m, p_map, profile, settings.devig_method, settings.elo.series_concentration)
        if m.match_id in skip_match_ids:
            table["qualifies"] = False
            table.loc[table.reason == "", "reason"] = "you already have an open bet on this match"
        info.update(model_team_a=a, model_team_b=b)
        priced[m.match_id] = (table, info)
    bets, full = plan_bets(matches, priced)
    stakes = recommend_stakes(bets, profile, ctx, full_kelly=full) if not bets.empty else bets
    return priced, stakes


def _match_section(m: MatchMarkets, table: pd.DataFrame, info: dict, tz: str,
                   stakes: pd.DataFrame | None = None) -> list[str]:
    start = m.start_time.tz_convert(tz)
    lines = [f"### {m.team_a} vs {m.team_b}", "",
             f"{start:%a %d %b, %I:%M %p} · Bo{m.best_of} · {m.bookmaker}", ""]
    if info.get("model_series") is not None:
        mkt = info["market_series"]
        mkt_txt = f"market {mkt:.0%}" if mkt is not None else "no match-result market"
        lines += [f"{m.team_a} to win: model **{info['model_series']:.0%}** "
                  f"(map {info['model_map']:.0%}) vs {mkt_txt}.", ""]
    if info.get("blocked"):
        lines += [f"No bets on this match: {info['blocked']}.", ""]
    priced = table[table.kind.isin(KIND_LABELS)]
    if not priced.empty and info.get("model_series") is not None:
        staked = {}
        if stakes is not None and not stakes.empty:
            mine = stakes[stakes.match_id == m.match_id]
            staked = dict(zip(mine.selection + "|" + mine.market, mine.stake))
        lines += ["| Market | Selection | Odds | Model | Market | EV | Bet |", "|---|---|---|---|---|---|---|"]
        for r in priced.itertuples():
            stake = staked.get(f"{r.selection}|{r.market}", 0.0)
            mark = f"**${stake:.2f}**" if stake > 0 else ("covered" if r.qualifies else "")
            lines.append(f"| {KIND_LABELS[r.kind]} | {r.selection} | {r.odds:.2f} | {r.model_prob:.0%} | "
                         f"{r.market_prob:.0%} | {r.ev:+.1%} | {mark} |")
        lines.append("")
        if (priced.qualifies & ~(priced.selection + "|" + priced.market).isin(
                [k for k, v in staked.items() if v > 0])).any():
            lines += ["_covered_ = also +EV, but a linked bet on this match already captures that edge.", ""]
    unpriced = table[~table.kind.isin(KIND_LABELS)]
    if not unpriced.empty:
        n_markets = unpriced.market.nunique()
        lines += [f"{n_markets} more markets (kills, first blood, odd/even) not priced: the model has no "
                  "per-game kill data yet.", ""]
    return lines


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
                 days: int = 3, now: pd.Timestamp | None = None, links: list[str] | None = None,
                 use_pointsbet: bool = True, skip_match_ids: set[str] = frozenset()) -> str:
    ratings = build_ratings(settings)
    model = ratings.model
    tz = settings.timezone
    now = now or pd.Timestamp.now(tz=tz)
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

    # 1. Live PointsBet markets.
    matches: list[MatchMarkets] = []
    if use_pointsbet or links:
        try:
            if links:
                for link in links:
                    matches += pointsbet.fetch_link(link)
            else:
                matches = pointsbet.fetch_worlds()
        except pointsbet.PointsBetError as err:
            lines += [f"_Couldn't load PointsBet markets: {err}_", ""]
    matches = [m for m in matches if m.start_time.tz_convert(tz) >= now]
    if matches:
        priced, stakes = analyse_markets(matches, model, settings, profile, ctx, skip_match_ids)
        lines += ["## Bets to place", ""]
        placed = stakes[stakes.stake > 0] if not stakes.empty else stakes
        if placed.empty:
            lines += ["None right now: no PointsBet price clears your profile's filters.", ""]
        else:
            lines += ["| Match | Market | Selection | Odds | Stake | Returns | EV |", "|---|---|---|---|---|---|---|"]
            for r in placed.itertuples():
                lines.append(f"| {r.match} | {KIND_LABELS[r.kind]} | **{r.selection}** | {r.odds:.2f} | "
                             f"**${r.stake:.2f}** | ${r.to_return:.2f} | {r.ev:+.1%} |")
            lines += ["", f"Total staked **${placed.stake.sum():.2f}** ({placed.stake.sum() / bankroll:.0%} of bankroll), "
                      f"expected profit ${placed.expected_profit.sum():.2f}. Bets on the same match were sized together "
                      "because they win and lose together.", ""]
        lines += ["## Every priced market", ""]
        for m in sorted(matches, key=lambda m: m.start_time):
            lines += _match_section(m, *priced[m.match_id], tz, stakes)

    # 2. Fixtures PointsBet hasn't listed yet: minimum prices.
    fixtures = pd.read_csv(settings.paths.fixtures) if settings.paths.fixtures.exists() else pd.DataFrame()
    if not fixtures.empty:
        fixtures["start_time"] = pd.to_datetime(fixtures.start_time, utc=True).dt.tz_convert(tz)
        fixtures = fixtures[(fixtures.start_time >= now) & (fixtures.start_time <= now + pd.Timedelta(days=days))]
        known = set(model.teams)
        covered = {frozenset({resolve_team(m.team_a, known, settings.team_aliases),
                              resolve_team(m.team_b, known, settings.team_aliases)}) for m in matches}
        fixtures = fixtures[[frozenset({resolve_team(r.team_a, known, settings.team_aliases),
                                        resolve_team(r.team_b, known, settings.team_aliases)}) not in covered
                             for r in fixtures.itertuples()]]
    odds = clean_odds(pd.read_csv(settings.paths.odds), settings) if settings.paths.odds.exists() else pd.DataFrame()
    if not fixtures.empty:
        lines += ["## Not on PointsBet yet: minimum prices", ""]
    for fx in fixtures.sort_values("start_time").itertuples(index=False) if not fixtures.empty else []:
        lines += _fixture_section(fx, model, settings, profile, ctx, odds)
    if not matches and fixtures.empty:
        lines.append(f"No Worlds matches with both teams confirmed in the next {days} days.")

    lines += [
        "---",
        "The model knows results only - not roster subs, patches or news. Prices move: re-run just before betting.",
    ]
    return "\n".join(lines)


def _fixture_section(fx, model: EloModel, settings: Settings, profile: RiskProfile, ctx: SizingContext,
                     odds: pd.DataFrame) -> list[str]:
    known = set(model.teams)
    a = resolve_team(fx.team_a, known, settings.team_aliases)
    b = resolve_team(fx.team_b, known, settings.team_aliases)
    lines = [f"### {fx.team_a} vs {fx.team_b}", "",
             f"{fx.start_time:%a %d %b, %I:%M %p} · {fx.event} {fx.stage} · Bo{int(fx.best_of)}", ""]
    if a is None or b is None:
        return lines + ["Not enough data on one of these teams to price this match - no bet.", ""]
    p = model.predict_series(a, b, int(fx.best_of))
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
        return lines
    for team, side_p in ((fx.team_a, p), (fx.team_b, 1 - p)):
        ladder = price_ladder(side_p, profile, ctx)
        if not ladder:
            lines.append(f"- **{team}**: no bet at any realistic price.")
            continue
        steps = ", ".join(f"{r['odds']:.2f} → ${r['stake']:.2f}" for r in ladder)
        lines.append(f"- **{team}**: bet only at **{ladder[0]['odds']:.2f} or better**. Stake by price: {steps}")
    return lines + [""]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print picks for upcoming Worlds fixtures.")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--bankroll", type=float, help="override the ledger bankroll")
    parser.add_argument("--profile", help="risk profile name (default from config.toml)")
    parser.add_argument("--config")
    parser.add_argument("--out", help="also write the report to this markdown file")
    parser.add_argument("--link", action="append", help="PointsBet match or competition link (repeatable)")
    parser.add_argument("--no-pointsbet", action="store_true", help="don't download PointsBet markets")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    profile = settings.risk_profiles[args.profile or settings.risk_profile]
    ledger = Ledger(settings.paths.ledger).load()
    state = bankroll_state(ledger, settings.starting_bankroll)
    bankroll = args.bankroll if args.bankroll is not None else state.bankroll
    open_ids = set(ledger[ledger.status == "open"].match_id.astype(str))
    report = build_report(settings, profile, bankroll, max(state.peak, bankroll), state.open_exposure, args.days,
                          links=args.link, use_pointsbet=not args.no_pointsbet, skip_match_ids=open_ids)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")


if __name__ == "__main__":
    main()
