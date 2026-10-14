"""Aggressive bankroll management with an optimised fractional Kelly criterion.

Stake sizing for a day's slate of value bets:

1. **Simultaneous Kelly.** Bets on the same day share one bankroll, so summing
   each bet's stand-alone Kelly stake over-bets. Instead we maximise expected log
   wealth over every win/loss combination of the slate (exact for up to 14 bets).
2. **Fractional multiplier** from the risk profile (e.g. 0.5 = half Kelly). The
   probabilities fed in are already shrunk toward the market (see ``value.py``),
   which guards against the model's own over-confidence.
3. **Drawdown brake.** If the bankroll is down ``drawdown_brake_pct`` from its peak,
   the multiplier is cut (default halved) until it recovers.
4. **Caps.** Per-bet cap (``max_stake_pct``) and a total open-exposure cap
   (``max_exposure_pct``, which counts bets already placed but not settled).
5. **Rounding.** Stakes are rounded down to the configured increment and dropped
   if below the minimum stake.

``simulate_bankroll`` replays a slate many times to compare Kelly multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RiskProfile

MAX_EXACT_BETS = 14
MAX_TOTAL_FRACTION = 0.95


def kelly_fraction(prob: float, odds: float) -> float:
    """Stand-alone full-Kelly fraction of bankroll for one bet (0 if no edge)."""
    return max(0.0, (prob * odds - 1.0) / (odds - 1.0))


def _project(v: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum(x) <= cap}."""
    x = np.clip(v, 0.0, None)
    if x.sum() <= cap:
        return x
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - cap
    rho = np.nonzero(u - css / np.arange(1, len(u) + 1) > 0)[0][-1]
    return np.clip(v - css[rho] / (rho + 1), 0.0, None)


def _scenarios(probs: np.ndarray, odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probability and per-bet return of each of the 2^n win/loss combinations."""
    n = len(probs)
    wins = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
    scenario_prob = np.prod(np.where(wins, probs, 1.0 - probs), axis=1)
    returns = np.where(wins, odds - 1.0, -1.0)
    return scenario_prob, returns


def expected_log_growth(fractions, probs, odds) -> float:
    f, p, o = (np.asarray(x, dtype=float) for x in (fractions, probs, odds))
    scenario_prob, returns = _scenarios(p, o)
    wealth = 1.0 + returns @ f
    return -np.inf if np.any(wealth <= 0) else float(scenario_prob @ np.log(wealth))


def simultaneous_kelly(probs, odds, max_total: float = MAX_TOTAL_FRACTION,
                       max_iter: int = 2000, tol: float = 1e-10) -> np.ndarray:
    """Full-Kelly fractions for independent bets placed at the same time."""
    p = np.asarray(probs, dtype=float)
    o = np.asarray(odds, dtype=float)
    single = np.array([kelly_fraction(pi, oi) for pi, oi in zip(p, o)])
    if len(p) <= 1:
        return np.minimum(single, max_total)
    if len(p) > MAX_EXACT_BETS:
        total = single.sum()
        return single * min(1.0, max_total / total) if total > 0 else single

    scenario_prob, returns = _scenarios(p, o)

    def growth(f: np.ndarray) -> float:
        wealth = 1.0 + returns @ f
        return -np.inf if np.any(wealth <= 0) else float(scenario_prob @ np.log(wealth))

    # Projected gradient ascent with Armijo backtracking; the objective is concave.
    f = _project(single * 0.5, max_total)
    g_f, step = growth(f), 1.0
    for _ in range(max_iter):
        grad = returns.T @ (scenario_prob / (1.0 + returns @ f))
        while True:
            cand = _project(f + step * grad, max_total)
            g_c = growth(cand)
            if g_c >= g_f + 1e-4 * grad @ (cand - f) or step < 1e-12:
                break
            step *= 0.5
        if step < 1e-12:
            break
        moved = np.max(np.abs(cand - f))
        f, g_f = cand, g_c
        if moved < tol:
            break
        step *= 2.0
    return f


def kelly_multiplier(profile: RiskProfile, bankroll: float, peak_bankroll: float) -> tuple[float, bool]:
    """Profile's Kelly fraction, cut by the drawdown brake when it is engaged."""
    drawdown = 1.0 - bankroll / peak_bankroll if peak_bankroll > 0 else 0.0
    braked = drawdown >= profile.drawdown_brake_pct
    return profile.kelly_fraction * (profile.drawdown_brake_multiplier if braked else 1.0), braked


def apply_caps(fractions: np.ndarray, profile: RiskProfile, bankroll: float,
               open_exposure: float = 0.0) -> tuple[np.ndarray, np.ndarray, bool]:
    """Per-bet then total-exposure caps. Returns (fractions, per_bet_capped mask, exposure_scaled)."""
    per_bet = np.minimum(fractions, profile.max_stake_pct)
    room = max(0.0, profile.max_exposure_pct * bankroll - open_exposure) / bankroll if bankroll > 0 else 0.0
    total = per_bet.sum()
    scaled = total > room + 1e-12
    if scaled:
        per_bet = per_bet * (room / total)
    return per_bet, fractions > profile.max_stake_pct + 1e-12, scaled


@dataclass(frozen=True)
class SizingContext:
    bankroll: float
    peak_bankroll: float
    open_exposure: float = 0.0
    stake_rounding: float = 0.50
    min_stake: float = 1.00


STAKE_COLUMNS = ["single_kelly", "full_kelly", "kelly_multiplier", "stake_pct", "stake",
                 "to_return", "expected_profit", "notes"]


def recommend_stakes(bets: pd.DataFrame, profile: RiskProfile, ctx: SizingContext) -> pd.DataFrame:
    """Add stake recommendations to value bets (needs ``bet_prob``, ``odds`` and ``ev`` columns)."""
    out = bets.copy().reset_index(drop=True)
    if out.empty or ctx.bankroll <= 0:
        for name in STAKE_COLUMNS:
            out[name] = pd.Series(dtype="object" if name == "notes" else float)
        return out

    p = out["bet_prob"].to_numpy(dtype=float)
    o = out["odds"].to_numpy(dtype=float)
    full = simultaneous_kelly(p, o)
    mult, braked = kelly_multiplier(profile, ctx.bankroll, ctx.peak_bankroll)
    capped, per_bet_capped, exposure_scaled = apply_caps(full * mult, profile, ctx.bankroll, ctx.open_exposure)

    stake = np.floor(capped * ctx.bankroll / ctx.stake_rounding + 1e-9) * ctx.stake_rounding
    notes = []
    for i in range(len(out)):
        n = []
        if braked:
            n.append("drawdown brake on")
        if per_bet_capped[i]:
            n.append(f"capped at {profile.max_stake_pct:.0%} per bet")
        if exposure_scaled:
            n.append(f"scaled to {profile.max_exposure_pct:.0%} exposure limit")
        if stake[i] < ctx.min_stake and capped[i] * ctx.bankroll >= 0.01:
            stake[i] = 0.0
            n.append(f"below ${ctx.min_stake:.2f} minimum")
        notes.append("; ".join(n))

    out["single_kelly"] = [kelly_fraction(pi, oi) for pi, oi in zip(p, o)]
    out["full_kelly"] = full
    out["kelly_multiplier"] = mult
    out["stake_pct"] = stake / ctx.bankroll
    out["stake"] = stake
    out["to_return"] = stake * o
    out["expected_profit"] = stake * out["ev"].to_numpy(dtype=float)
    out["notes"] = notes
    return out


def simulate_bankroll(
    probs,
    odds,
    *,
    start: float,
    profile: RiskProfile,
    multipliers=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0),
    market_probs=None,
    edge_realisation: float = 1.0,
    n_slates: int = 30,
    n_paths: int = 4000,
    seed: int = 7,
) -> pd.DataFrame:
    """Replay a slate ``n_slates`` times per path to compare Kelly multipliers.

    Stakes use the believed ``probs``; outcomes are drawn from
    ``market + edge_realisation * (probs - market)``, so ``edge_realisation=0.5``
    asks "what if only half my edge is real?". Caps are applied, but rounding,
    minimum stakes and the drawdown brake are not.
    """
    p = np.asarray(probs, dtype=float)
    o = np.asarray(odds, dtype=float)
    true_p = p if market_probs is None else np.asarray(market_probs) + edge_realisation * (p - np.asarray(market_probs))
    full = simultaneous_kelly(p, o)

    rng = np.random.default_rng(seed)
    # Common random numbers: every multiplier sees the same match results.
    wins = rng.random((n_paths, n_slates, len(p))) < np.clip(true_p, 0, 1)
    unit_returns = np.where(wins, o - 1.0, -1.0)

    rows = []
    for m in multipliers:
        f, _, _ = apply_caps(full * m, profile, bankroll=1.0)
        wealth = start * np.cumprod(1.0 + unit_returns @ f, axis=1)
        peak = np.maximum.accumulate(np.maximum(wealth, start), axis=1)
        max_drawdown = np.max(1.0 - wealth / peak, axis=1)
        final = wealth[:, -1]
        rows.append(
            {
                "multiplier": m,
                "median_final": float(np.median(final)),
                "p10_final": float(np.percentile(final, 10)),
                "p90_final": float(np.percentile(final, 90)),
                "prob_profit": float(np.mean(final > start)),
                "prob_halved": float(np.mean(final < start / 2)),
                "prob_drawdown_50": float(np.mean(max_drawdown >= 0.5)),
                "median_max_drawdown": float(np.median(max_drawdown)),
            }
        )
    return pd.DataFrame(rows)


def suggest_multiplier(sim: pd.DataFrame, max_prob_drawdown_50: float = 0.25) -> float | None:
    """Multiplier with the best median outcome whose chance of a 50% drawdown stays within tolerance."""
    ok = sim[sim.prob_drawdown_50 <= max_prob_drawdown_50]
    if ok.empty:
        return None
    return float(ok.loc[ok.median_final.idxmax(), "multiplier"])
