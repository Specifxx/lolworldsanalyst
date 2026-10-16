"""Series markets priced from one map-win probability.

Every way a best-of-N can play out is a sequence of map results. Enumerating
those sequences prices anything the series settles - match winner, map
handicap, total maps, correct score, map N winner - exactly. Maps are linked
within a series (beta-binomial, ``series_concentration`` in config.toml) because
real series produce more sweeps than independent maps would. The same sequences
are the scenarios for sizing correlated bets on one match jointly (KC -1.5 maps
and KC 3-0 mostly win or lose together).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bankroll import MAX_TOTAL_FRACTION, optimize_kelly
from .config import RiskProfile
from .elo import score_sequence_prob, series_win_prob
from .value import remove_vig

PRICED_KINDS = ("match_winner", "map_handicap", "total_maps", "correct_score", "map_winner")
KIND_LABELS = {
    "match_winner": "Match result",
    "map_handicap": "Maps handicap",
    "total_maps": "Total maps",
    "correct_score": "Correct score",
    "map_winner": "Map winner",
}


@dataclass(frozen=True)
class Selection:
    market: str                       # bookmaker's market name, e.g. "Maps Handicap"
    kind: str                         # one of PRICED_KINDS, or "unpriced"
    label: str                        # outcome as the bookmaker shows it
    odds: float
    side: str | None = None           # "a" / "b" for team outcomes
    line: float | None = None         # maps handicap, or total-maps line
    over: bool | None = None          # total maps: True = over, False = under
    score: tuple[int, int] | None = None  # correct score from the named team's view
    map_no: int | None = None
    market_id: str = ""               # groups one market's outcomes for de-vigging
    note: str = ""                    # why an unpriced market isn't priced


@dataclass
class MatchMarkets:
    """One match and every selection a bookmaker offers on it."""
    match_id: str
    start_time: pd.Timestamp
    team_a: str
    team_b: str
    best_of: int | None
    bookmaker: str
    selections: list[Selection] = field(default_factory=list)
    name: str = ""


# -- series outcomes --------------------------------------------------------
def sequences(best_of: int) -> list[tuple[bool, ...]]:
    """Every complete series as map results (True = team A won that map)."""
    need = best_of // 2 + 1
    out = []
    for n in range(need, best_of + 1):
        for seq in itertools.product((True, False), repeat=n):
            a = sum(seq)
            winner_maps, winner_took_last = (a, seq[-1]) if a >= n - a else (n - a, not seq[-1])
            if winner_maps == need and winner_took_last:
                out.append(seq)
    return out


def sequence_probs(p_map: float, seqs: list[tuple[bool, ...]], concentration: float = 0.0) -> np.ndarray:
    return np.array([score_sequence_prob(p_map, sum(s), len(s) - sum(s), concentration) for s in seqs])


def settle(sel: Selection, seq: tuple[bool, ...]) -> bool | None:
    """Does the selection win in this series? None = void (e.g. map 5 never played)."""
    a_maps = sum(seq)
    b_maps = len(seq) - a_maps
    mine, theirs = (a_maps, b_maps) if sel.side == "a" else (b_maps, a_maps)
    if sel.kind == "match_winner":
        return mine > theirs
    if sel.kind == "map_handicap":
        return mine + sel.line > theirs
    if sel.kind == "total_maps":
        return len(seq) > sel.line if sel.over else len(seq) < sel.line
    if sel.kind == "correct_score":
        return (mine, theirs) == sel.score
    if sel.kind == "map_winner":
        if sel.map_no > len(seq):
            return None
        return seq[sel.map_no - 1] == (sel.side == "a")
    raise ValueError(f"cannot settle a {sel.kind} selection")


def returns_matrix(selections: list[Selection], seqs: list[tuple[bool, ...]]) -> np.ndarray:
    """Profit per $1 on each selection (columns) in each series outcome (rows)."""
    out = np.zeros((len(seqs), len(selections)))
    for j, sel in enumerate(selections):
        for i, seq in enumerate(seqs):
            result = settle(sel, seq)
            out[i, j] = 0.0 if result is None else (sel.odds - 1.0 if result else -1.0)
    return out


def win_void_prob(sel: Selection, p_map: float, best_of: int, concentration: float = 0.0) -> tuple[float, float]:
    seqs = sequences(best_of)
    probs = sequence_probs(p_map, seqs, concentration)
    results = [settle(sel, s) for s in seqs]
    win = sum(p for p, r in zip(probs, results) if r is True)
    void = sum(p for p, r in zip(probs, results) if r is None)
    return float(win), float(void)


def implied_map_prob(series_prob: float, best_of: int, concentration: float = 0.0) -> float:
    """Map win probability that gives this series win probability."""
    lo, hi = 1e-6, 1 - 1e-6
    for _ in range(80):
        mid = (lo + hi) / 2
        if series_win_prob(mid, best_of, concentration) < series_prob:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# -- pricing ------------------------------------------------------------------
def market_series_prob(match: MatchMarkets, method: str = "power") -> float | None:
    """Fair probability that team A wins the series, from the match-result market."""
    by_side = {s.side: s.odds for s in match.selections if s.kind == "match_winner"}
    if {"a", "b"} <= set(by_side):
        return remove_vig([by_side["a"], by_side["b"]], method)[0]
    return None


def price_match(match: MatchMarkets, p_model_map: float | None, profile: RiskProfile,
                devig_method: str = "power", concentration: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Every selection with model/market/bet probabilities, EV and whether to bet it."""
    bo = match.best_of
    kappa = concentration
    p_series_mkt = market_series_prob(match, devig_method)
    info = {"model_map": p_model_map, "market_series": p_series_mkt, "model_series": None,
            "bet_map": None, "blocked": "", "concentration": kappa}
    if p_model_map is None:
        info["blocked"] = "team not in rating data"
    elif bo is None:
        info["blocked"] = "series length unknown"
    else:
        info["model_series"] = series_win_prob(p_model_map, bo, kappa)
        if p_series_mkt is None:
            info["bet_map"] = p_model_map
        else:
            p_mkt_map = implied_map_prob(p_series_mkt, bo, kappa)
            info["bet_map"] = profile.model_weight * p_model_map + (1 - profile.model_weight) * p_mkt_map
            gap = abs(info["model_series"] - p_series_mkt)
            if gap > profile.max_prob_gap:
                info["blocked"] = f"model vs market gap {gap:.0%} > {profile.max_prob_gap:.0%} - check news"

    # Fair (de-vigged) probability of each outcome within its own market.
    fair = {}
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(match.selections):
        groups.setdefault(s.market_id, []).append(i)
    for idx in groups.values():
        if len(idx) >= 2:
            for i, f in zip(idx, remove_vig([match.selections[i].odds for i in idx], devig_method)):
                fair[i] = f

    rows = []
    for i, s in enumerate(match.selections):
        row = {"match_id": match.match_id, "market": s.market, "kind": s.kind, "selection": s.label,
               "odds": s.odds, "market_prob": fair.get(i, np.nan), "model_prob": np.nan, "bet_prob": np.nan,
               "void_prob": 0.0, "ev": np.nan, "qualifies": False, "reason": ""}
        if s.kind not in PRICED_KINDS:
            row["reason"] = s.note or "not modelled"
        elif info["bet_map"] is None:
            row["reason"] = info["blocked"]
        else:
            row["model_prob"] = win_void_prob(s, p_model_map, bo, kappa)[0]
            win, void = win_void_prob(s, info["bet_map"], bo, kappa)
            ev = win * s.odds + void - 1
            row.update(bet_prob=win, void_prob=void, ev=ev)
            reasons = []
            if info["blocked"]:
                reasons.append(info["blocked"])
            if ev < profile.min_ev:
                reasons.append(f"EV {ev:+.1%} below {profile.min_ev:.0%}")
            if not profile.min_odds <= s.odds <= profile.max_odds:
                reasons.append(f"odds outside {profile.min_odds:.2f}-{profile.max_odds:.2f}")
            row["qualifies"] = not reasons
            row["reason"] = "; ".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows), info


# -- joint sizing -------------------------------------------------------------
MAX_BETS_PER_MATCH = 6
MAX_JOINT_STATES = 200_000


def joint_kelly(per_match: list[tuple[np.ndarray, np.ndarray]],
                max_total: float = MAX_TOTAL_FRACTION) -> list[np.ndarray]:
    """Full-Kelly fractions for bets across several matches.

    ``per_match`` holds (outcome probabilities, returns matrix) for each match. Matches
    are independent, so the joint outcome space is their product. If that gets too
    large, each match is solved on its own and the total is scaled to ``max_total``.
    """
    per_match = [(p, r) for p, r in per_match]
    if not per_match:
        return []
    states = math.prod(len(p) for p, _ in per_match)
    if states > MAX_JOINT_STATES:
        parts = [optimize_kelly(p, r, max_total) for p, r in per_match]
        total = sum(x.sum() for x in parts)
        scale = min(1.0, max_total / total) if total > 0 else 1.0
        return [x * scale for x in parts]

    probs = np.ones(1)
    returns = np.zeros((1, 0))
    for p, r in per_match:
        s_prev, s_new = len(probs), len(p)
        probs = np.outer(probs, p).ravel()
        returns = np.hstack([np.repeat(returns, s_new, axis=0), np.tile(r, (s_prev, 1))])
    full = optimize_kelly(probs, returns, max_total)
    out, start = [], 0
    for _, r in per_match:
        out.append(full[start:start + r.shape[1]])
        start += r.shape[1]
    return out


def plan_bets(matches: list[MatchMarkets], priced: dict[str, tuple[pd.DataFrame, dict]]) -> tuple[pd.DataFrame, np.ndarray]:
    """Qualifying selections across matches and their joint full-Kelly fractions."""
    chosen, blocks = [], []
    for m in matches:
        table, info = priced[m.match_id]
        q = table[table.qualifies].sort_values("ev", ascending=False).head(MAX_BETS_PER_MATCH)
        if q.empty:
            continue
        sels = [m.selections[i] for i in q.index]
        seqs = sequences(m.best_of)
        blocks.append((sequence_probs(info["bet_map"], seqs, info["concentration"]), returns_matrix(sels, seqs)))
        chosen.append(q.assign(start_time=m.start_time, team_a=m.team_a, team_b=m.team_b,
                               bookmaker=m.bookmaker, match=f"{m.team_a} vs {m.team_b}"))
    if not chosen:
        return pd.DataFrame(), np.zeros(0)
    fractions = joint_kelly(blocks)
    return pd.concat(chosen, ignore_index=True), np.concatenate(fractions)
