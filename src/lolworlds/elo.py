"""Map-level Elo ratings for pro League of Legends, converted to series odds.

Ratings are updated once per game (map). A series price (Bo1/Bo3/Bo5) is derived
from the single-map win probability, treating maps as independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EloSettings


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B in one game."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def series_win_prob(p_map: float, best_of: int) -> float:
    """Probability of winning a best-of-N series given a per-map win probability."""
    if best_of < 1 or best_of % 2 == 0:
        raise ValueError(f"best_of must be a positive odd number, got {best_of}")
    need = best_of // 2 + 1
    q = 1.0 - p_map
    # Win the final map after k losses spread over the earlier maps.
    return sum(math.comb(need - 1 + k, k) * p_map**need * q**k for k in range(need))


def fair_odds(prob: float) -> float:
    return math.inf if prob <= 0 else 1.0 / prob


@dataclass
class TeamState:
    rating: float
    league: str | None
    games: int = 0
    wins: int = 0
    last_year: int | None = None
    last_date: pd.Timestamp | None = None


class EloModel:
    def __init__(self, settings: EloSettings, home_leagues: dict[str, str] | None = None):
        self.settings = settings
        self.home_leagues = home_leagues or {}
        self.teams: dict[str, TeamState] = {}

    # -- lookups ------------------------------------------------------------
    def knows(self, team: str) -> bool:
        return team in self.teams

    def rating(self, team: str) -> float:
        if team in self.teams:
            return self.teams[team].rating
        return self.settings.prior_for(self.home_leagues.get(team))

    def predict_map(self, team_a: str, team_b: str, a_side: str | None = None) -> float:
        """Map win probability for A. ``a_side`` is 'blue', 'red' or None (unknown)."""
        bonus = {"blue": self.settings.blue_side_bonus, "red": -self.settings.blue_side_bonus}
        return expected_score(self.rating(team_a) + bonus.get(a_side, 0.0), self.rating(team_b))

    def predict_series(self, team_a: str, team_b: str, best_of: int = 3) -> float:
        return series_win_prob(self.predict_map(team_a, team_b), best_of)

    # -- updates ------------------------------------------------------------
    def _state(self, team: str) -> TeamState:
        if team not in self.teams:
            league = self.home_leagues.get(team)
            self.teams[team] = TeamState(rating=self.settings.prior_for(league), league=league)
        return self.teams[team]

    def _regress_to_prior(self, state: TeamState, year: int) -> None:
        """Pull ratings toward the region prior at the start of each new year (roster churn)."""
        if state.last_year is not None and year > state.last_year:
            prior = self.settings.prior_for(state.league)
            keep = self.settings.season_carryover ** (year - state.last_year)
            state.rating = prior + keep * (state.rating - prior)

    def _k_factor(self, state: TeamState, international: bool) -> float:
        k = self.settings.base_k
        if international:
            k *= self.settings.international_k_multiplier
        if state.games < self.settings.new_team_games:
            k *= self.settings.new_team_k_multiplier
        return k

    def _mov_multiplier(self, winner_gd15: float, winner_kill_diff: float, winner_edge: float) -> float:
        dominance = 1.0
        if not math.isnan(winner_gd15):
            dominance += self.settings.mov_gold_weight * math.tanh(max(winner_gd15, 0.0) / 4000.0)
        if not math.isnan(winner_kill_diff):
            dominance += self.settings.mov_kill_weight * math.tanh(max(winner_kill_diff, 0.0) / 12.0)
        # Damp the update when the favourite wins big, boost it for upsets
        # (autocorrelation correction from FiveThirtyEight's Elo models).
        edge = min(max(winner_edge, -1000.0), 1000.0)
        return dominance * 2.2 / (2.2 + 0.001 * edge)

    def update(
        self,
        *,
        blue: str,
        red: str,
        blue_win: bool,
        date: pd.Timestamp,
        league: str,
        gd15: float = math.nan,
        kill_diff: float = math.nan,
        neutral: bool = False,
    ) -> dict:
        """Apply one game. ``gd15``/``kill_diff`` are from blue's perspective.

        ``neutral=True`` means the sides aren't known, so no blue-side bonus is applied.
        """
        sb, sr = self._state(blue), self._state(red)
        for state in (sb, sr):
            self._regress_to_prior(state, date.year)

        bonus = 0.0 if neutral else self.settings.blue_side_bonus
        p_blue = expected_score(sb.rating + bonus, sr.rating)
        sign = 1.0 if blue_win else -1.0
        mult = self._mov_multiplier(sign * gd15, sign * kill_diff, sign * (sb.rating + bonus - sr.rating))
        international = league in self.settings.international_leagues
        delta_blue = self._k_factor(sb, international) * mult * (float(blue_win) - p_blue)
        delta_red = self._k_factor(sr, international) * mult * (p_blue - float(blue_win))

        record = {
            "blue_rating_pre": sb.rating,
            "red_rating_pre": sr.rating,
            "blue_games_pre": sb.games,
            "red_games_pre": sr.games,
            "p_blue": p_blue,
            "delta_blue": delta_blue,
            "delta_red": delta_red,
        }
        for state, delta, won in ((sb, delta_blue, blue_win), (sr, delta_red, not blue_win)):
            state.rating += delta
            state.games += 1
            state.wins += int(won)
            state.last_year = date.year
            state.last_date = date
        return record

    def fit(self, games: pd.DataFrame) -> pd.DataFrame:
        """Run every game in date order. Returns games with pre-game probabilities attached."""
        records = [
            self.update(
                blue=g.blue_team,
                red=g.red_team,
                blue_win=bool(g.blue_win),
                date=g.date,
                league=g.league,
                gd15=float(g.gd15),
                kill_diff=float(g.blue_kills - g.red_kills),
                neutral=not getattr(g, "sides_known", True),
            )
            for g in games.itertuples(index=False)
        ]
        return pd.concat([games.reset_index(drop=True), pd.DataFrame(records)], axis=1)

    def table(self) -> pd.DataFrame:
        rows = [
            {
                "team": team,
                "league": s.league or "?",
                "rating": round(s.rating, 1),
                "games": s.games,
                "wins": s.wins,
                "losses": s.games - s.wins,
                "win_rate": s.wins / s.games if s.games else np.nan,
                "last_played": s.last_date,
            }
            for team, s in self.teams.items()
        ]
        table = pd.DataFrame(rows).sort_values("rating", ascending=False, ignore_index=True)
        table.insert(0, "rank", np.arange(1, len(table) + 1))
        return table


def calibration_summary(history: pd.DataFrame, min_prior_games: int = 5) -> dict:
    """Log loss / Brier / accuracy of the pre-game predictions, skipping cold-start games."""
    h = history[(history.blue_games_pre >= min_prior_games) & (history.red_games_pre >= min_prior_games)]
    if h.empty:
        return {"games": 0, "log_loss": np.nan, "brier": np.nan, "accuracy": np.nan}
    p = h.p_blue.clip(1e-6, 1 - 1e-6).to_numpy()
    y = h.blue_win.to_numpy(dtype=float)
    return {
        "games": int(len(h)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
    }


def calibration_table(history: pd.DataFrame, bins: int = 10, min_prior_games: int = 5) -> pd.DataFrame:
    """Predicted vs actual win rate per probability bucket (for a reliability check)."""
    h = history[(history.blue_games_pre >= min_prior_games) & (history.red_games_pre >= min_prior_games)]
    edges = np.linspace(0, 1, bins + 1)
    bucket = pd.cut(h.p_blue, edges, include_lowest=True)
    out = (
        h.groupby(bucket, observed=True)
        .agg(games=("blue_win", "size"), predicted=("p_blue", "mean"), actual=("blue_win", "mean"))
        .reset_index(names="bucket")
    )
    out["bucket"] = out["bucket"].astype(str)
    return out
