"""Value engine: compare model probabilities with bookmaker odds to find +EV bets.

Odds CSV layout (long format, one row per match per bookmaker)::

    match_id,start_time,event,stage,best_of,team_a,team_b,bookmaker,odds_a,odds_b
    W26-S3-01,2026-10-15T19:00:00+11:00,Worlds 2026,Swiss R3,3,Gen.G,G2 Esports,Sportsbet,1.30,3.45

``start_time`` should carry a UTC offset (AEDT is +11:00). ``odds_*`` are decimal
head-to-head (match winner) prices.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import RiskProfile, Settings
from .elo import EloModel, series_win_prob

ODDS_REQUIRED = ["match_id", "start_time", "team_a", "team_b", "bookmaker", "odds_a", "odds_b"]


# -- pricing primitives -----------------------------------------------------
def implied_prob(odds: float) -> float:
    return 1.0 / odds


def overround(odds: list[float]) -> float:
    """Bookmaker margin, e.g. 0.06 means the book's implied probabilities sum to 106%."""
    return sum(1.0 / o for o in odds) - 1.0


def remove_vig(odds: list[float], method: str = "power") -> list[float]:
    """Fair probabilities from a complete market's decimal odds."""
    raw = [1.0 / o for o in odds]
    if method == "multiplicative":
        total = sum(raw)
        return [r / total for r in raw]
    if method == "power":
        # Find k with sum(raw_i ** k) == 1. k > 1 shades longshots more than favourites,
        # matching the favourite-longshot bias in how books distribute their margin.
        lo, hi = 0.5, 5.0
        for _ in range(100):
            k = (lo + hi) / 2
            if sum(r**k for r in raw) > 1.0:
                lo = k
            else:
                hi = k
        k = (lo + hi) / 2
        probs = [r**k for r in raw]
        total = sum(probs)
        return [p / total for p in probs]
    raise ValueError(f"unknown devig method: {method}")


def expected_value(prob: float, odds: float) -> float:
    """Expected profit per $1 staked."""
    return prob * odds - 1.0


# -- odds loading -----------------------------------------------------------
def canonical_bookmaker(name: str, enabled: tuple[str, ...]) -> str | None:
    key = name.strip().lower().replace(" ", "")
    for book in enabled:
        if book.lower().replace(" ", "") == key:
            return book
    return None


def load_odds(path, settings: Settings) -> pd.DataFrame:
    odds = pd.read_csv(path)
    missing = [c for c in ODDS_REQUIRED if c not in odds.columns]
    if missing:
        raise ValueError(f"odds file is missing required columns: {missing}")
    return clean_odds(odds, settings)


def clean_odds(odds: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    o = odds.copy()
    for name, default in (("event", ""), ("stage", ""), ("best_of", 3)):
        o[name] = o[name].fillna(default) if name in o else default
    o["best_of"] = pd.to_numeric(o["best_of"], errors="coerce").fillna(3).astype(int)
    o["start_time"] = pd.to_datetime(o["start_time"], errors="coerce", utc=True).dt.tz_convert(settings.timezone)
    o["bookmaker"] = [canonical_bookmaker(str(b), settings.bookmakers) for b in o["bookmaker"]]
    for name in ("odds_a", "odds_b"):
        o[name] = pd.to_numeric(o[name], errors="coerce")
    o = o.dropna(subset=["start_time", "bookmaker", "odds_a", "odds_b"])
    o = o[(o.odds_a > 1.0) & (o.odds_b > 1.0)]
    # Keep the latest price if a book is listed twice for the same match.
    return o.drop_duplicates(["match_id", "bookmaker"], keep="last").reset_index(drop=True)


# -- team name matching -----------------------------------------------------
def resolve_team(name: str, known: set[str], aliases: dict[str, str]) -> str | None:
    """Map a bookmaker's team name onto a name the Elo model knows."""
    name = name.strip()
    if name in known:
        return name
    lowered = {k.lower(): k for k in known}
    for candidate in (aliases.get(name), *(v for k, v in aliases.items() if k.lower() == name.lower())):
        if candidate and candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return lowered.get(name.lower())


# -- match-level view -------------------------------------------------------
def market_board(odds: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """One row per match: each book's prices, best price per side and the fair market line."""
    rows = []
    for match_id, group in odds.groupby("match_id", sort=False):
        first = group.iloc[0]
        row = {
            "match_id": match_id,
            "start_time": first.start_time,
            "event": first.event,
            "stage": first.stage,
            "best_of": int(first.best_of),
            "team_a": first.team_a,
            "team_b": first.team_b,
        }
        fair_a = []
        for book in settings.bookmakers:
            quote = group[group.bookmaker == book]
            if quote.empty:
                row[f"{book}_a"] = row[f"{book}_b"] = np.nan
                continue
            q = quote.iloc[-1]
            row[f"{book}_a"], row[f"{book}_b"] = q.odds_a, q.odds_b
            row[f"{book}_margin"] = overround([q.odds_a, q.odds_b])
            fair_a.append(remove_vig([q.odds_a, q.odds_b], settings.devig_method)[0])
        best_a = group.loc[group.odds_a.idxmax()]
        best_b = group.loc[group.odds_b.idxmax()]
        row.update(
            best_odds_a=best_a.odds_a,
            best_book_a=best_a.bookmaker,
            best_odds_b=best_b.odds_b,
            best_book_b=best_b.bookmaker,
            market_prob_a=float(np.mean(fair_a)),
            # Margin left if you take the best price on each side (negative = arbitrage).
            best_line_margin=overround([best_a.odds_a, best_b.odds_b]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_model_probs(board: pd.DataFrame, model: EloModel, aliases: dict[str, str]) -> pd.DataFrame:
    known = set(model.teams)
    b = board.copy()
    b["model_team_a"] = [resolve_team(t, known, aliases) for t in b.team_a]
    b["model_team_b"] = [resolve_team(t, known, aliases) for t in b.team_b]
    probs, maps = [], []
    for r in b.itertuples(index=False):
        if r.model_team_a is None or r.model_team_b is None:
            probs.append(np.nan)
            maps.append(np.nan)
            continue
        p_map = model.predict_map(r.model_team_a, r.model_team_b)
        maps.append(p_map)
        probs.append(series_win_prob(p_map, int(r.best_of)))
    b["model_map_prob_a"] = maps
    b["model_prob_a"] = probs
    return b


# -- bet-level view ---------------------------------------------------------
def evaluate_sides(board: pd.DataFrame, profile: RiskProfile) -> pd.DataFrame:
    """Two rows per match (one per team) with probabilities, best price and EV."""
    rows = []
    for r in board.itertuples(index=False):
        for side, other in (("a", "b"), ("b", "a")):
            model_p = r.model_prob_a if side == "a" else 1 - r.model_prob_a
            market_p = r.market_prob_a if side == "a" else 1 - r.market_prob_a
            odds = getattr(r, f"best_odds_{side}")
            rated = not math.isnan(model_p)
            bet_p = profile.model_weight * model_p + (1 - profile.model_weight) * market_p if rated else np.nan
            ev = expected_value(bet_p, odds) if rated else np.nan

            reasons = []
            if not rated:
                reasons.append("team not in rating data")
            else:
                if ev < profile.min_ev:
                    reasons.append(f"EV {ev:+.1%} below {profile.min_ev:.0%}")
                if abs(model_p - market_p) > profile.max_prob_gap:
                    reasons.append(
                        f"model vs market gap {abs(model_p - market_p):.0%} > {profile.max_prob_gap:.0%} - check news"
                    )
            if not profile.min_odds <= odds <= profile.max_odds:
                reasons.append(f"odds outside {profile.min_odds:.2f}-{profile.max_odds:.2f}")

            rows.append(
                {
                    "match_id": r.match_id,
                    "start_time": r.start_time,
                    "event": r.event,
                    "stage": r.stage,
                    "best_of": r.best_of,
                    "selection": getattr(r, f"team_{side}"),
                    "opponent": getattr(r, f"team_{other}"),
                    "bookmaker": getattr(r, f"best_book_{side}"),
                    "odds": odds,
                    "implied_prob": implied_prob(odds),
                    "market_prob": market_p,
                    "model_prob": model_p,
                    "bet_prob": bet_p,
                    "edge": bet_p - implied_prob(odds) if rated else np.nan,
                    "model_ev": expected_value(model_p, odds) if rated else np.nan,
                    "ev": ev,
                    "qualifies": not reasons,
                    "reason": "; ".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def select_value_bets(sides: pd.DataFrame) -> pd.DataFrame:
    """Qualifying bets, at most one side per match (the higher EV)."""
    q = sides[sides.qualifies]
    if q.empty:
        return q.copy()
    best = q.sort_values("ev", ascending=False).drop_duplicates("match_id")
    return best.sort_values(["start_time", "ev"], ascending=[True, False], ignore_index=True)


def find_value(odds: pd.DataFrame, model: EloModel, settings: Settings, profile: RiskProfile):
    """Full pipeline for a set of odds. Returns (board, sides, value_bets)."""
    if odds.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    board = add_model_probs(market_board(odds, settings), model, settings.team_aliases)
    sides = evaluate_sides(board, profile)
    return board, sides, select_value_bets(sides)
