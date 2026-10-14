"""Bet ledger (CSV) and bankroll tracking."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER_COLUMNS = [
    "bet_id", "placed_at", "match_id", "start_time", "event", "selection", "opponent",
    "bookmaker", "odds", "stake", "bet_prob", "ev", "status", "payout", "settled_at", "notes",
]
NUMERIC_COLUMNS = ["odds", "stake", "bet_prob", "ev", "payout"]
RESULTS = ("won", "lost", "void")


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            df = pd.DataFrame(columns=LEDGER_COLUMNS)
        else:
            # Pin text columns: an all-empty column (e.g. settled_at) would otherwise load as float.
            text = {c: object for c in LEDGER_COLUMNS if c not in NUMERIC_COLUMNS}
            df = pd.read_csv(self.path, dtype=text)
        for name in LEDGER_COLUMNS:
            if name not in df:
                df[name] = np.nan
        df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
        return df[LEDGER_COLUMNS].astype({c: object for c in LEDGER_COLUMNS if c not in NUMERIC_COLUMNS})

    def _save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df[LEDGER_COLUMNS].to_csv(self.path, index=False)

    def add_bets(self, bets: pd.DataFrame) -> list[str]:
        """Record placed bets. Needs selection, opponent, bookmaker, odds and stake columns."""
        if bets.empty:
            return []
        new = bets.copy()
        new["bet_id"] = [uuid.uuid4().hex[:8] for _ in range(len(new))]
        new["placed_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        new["status"] = "open"
        new["payout"] = np.nan
        new["settled_at"] = np.nan
        for name in LEDGER_COLUMNS:
            if name not in new:
                new[name] = np.nan
        current = self.load()
        frames = [f for f in (current, new[LEDGER_COLUMNS]) if not f.empty]
        self._save(pd.concat(frames, ignore_index=True))
        return list(new["bet_id"])

    def settle(self, bet_id: str, result: str) -> None:
        if result not in RESULTS:
            raise ValueError(f"result must be one of {RESULTS}")
        df = self.load()
        mask = df.bet_id == bet_id
        if not mask.any():
            raise KeyError(f"no bet with id {bet_id}")
        stake = df.loc[mask, "stake"].astype(float)
        odds = df.loc[mask, "odds"].astype(float)
        df.loc[mask, "payout"] = {"won": stake * odds, "lost": stake * 0.0, "void": stake}[result]
        df.loc[mask, "status"] = result
        df.loc[mask, "settled_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        self._save(df)

    def remove(self, bet_id: str) -> None:
        df = self.load()
        self._save(df[df.bet_id != bet_id])


@dataclass(frozen=True)
class BankrollState:
    starting: float
    bankroll: float          # starting + settled profit/loss
    open_exposure: float     # stakes on unsettled bets
    peak: float
    profit: float
    turnover: float          # total staked on settled bets
    settled_bets: int
    wins: int
    open_bets: int

    @property
    def available(self) -> float:
        return self.bankroll - self.open_exposure

    @property
    def roi(self) -> float:
        return self.profit / self.turnover if self.turnover else 0.0

    @property
    def win_rate(self) -> float:
        decided = self.settled_bets
        return self.wins / decided if decided else 0.0

    @property
    def drawdown(self) -> float:
        return 1.0 - self.bankroll / self.peak if self.peak > 0 else 0.0


def bankroll_history(ledger: pd.DataFrame, starting: float) -> pd.DataFrame:
    """Bankroll after each settled bet, in settlement order (first row is the start)."""
    settled = ledger[ledger.status.isin(["won", "lost"])].copy()
    settled["profit"] = settled.payout.astype(float) - settled.stake.astype(float)
    settled = settled.sort_values("settled_at", kind="stable")
    start_row = pd.DataFrame({"bet_no": [0], "label": ["Start"], "profit": [0.0], "bankroll": [starting]})
    if settled.empty:
        return start_row
    rows = pd.DataFrame(
        {
            "bet_no": np.arange(1, len(settled) + 1),
            "label": (settled.selection + " @ " + settled.odds.astype(float).map("{:.2f}".format)).to_numpy(),
            "profit": settled.profit.to_numpy(),
            "bankroll": starting + settled.profit.cumsum().to_numpy(),
        }
    )
    return pd.concat([start_row, rows], ignore_index=True)


def bankroll_state(ledger: pd.DataFrame, starting: float) -> BankrollState:
    history = bankroll_history(ledger, starting)
    decided = ledger[ledger.status.isin(["won", "lost"])]
    open_bets = ledger[ledger.status == "open"]
    return BankrollState(
        starting=starting,
        bankroll=float(history.bankroll.iloc[-1]),
        open_exposure=float(open_bets.stake.astype(float).sum()),
        peak=float(history.bankroll.max()),
        profit=float(history.bankroll.iloc[-1] - starting),
        turnover=float(decided.stake.astype(float).sum()),
        settled_bets=len(decided),
        wins=int((decided.status == "won").sum()),
        open_bets=len(open_bets),
    )
