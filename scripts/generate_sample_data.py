"""Generate SYNTHETIC demo data so the toolkit runs before you add real data.

Writes ``data/sample/matches.csv`` (two seasons of simulated games) and
``data/sample/odds.csv`` (simulated Sportsbet/PointsBet prices for an upcoming
Worlds slate). Team names are real organisations for familiarity, but every
result, stat and price is randomly generated - none of it is real.

    python scripts/generate_sample_data.py [--slate-start 2026-10-15]
"""

from __future__ import annotations

import argparse
import itertools
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample"
AEDT = ZoneInfo("Australia/Sydney")

LEAGUES = {
    "LCK": (1650, ["Gen.G", "Hanwha Life Esports", "T1", "KT Rolster", "Dplus KIA", "BNK FearX"]),
    "LPL": (1640, ["Bilibili Gaming", "Anyone's Legend", "Top Esports", "JD Gaming", "Invictus Gaming", "Weibo Gaming"]),
    "LEC": (1510, ["G2 Esports", "Fnatic", "Movistar KOI", "Karmine Corp", "Team Vitality", "Team Heretics"]),
    "LCS": (1450, ["FlyQuest", "Cloud9", "Team Liquid", "Shopify Rebellion", "Dignitas"]),
    "LCP": (1440, ["CTBC Flying Oyster", "GAM Esports", "PSG Talon", "Team Secret Whales", "DetonatioN FocusMe"]),
    "CBLOL": (1380, ["paiN Gaming", "LOUD", "Vivo Keyd Stars", "FURIA"]),
}
WORLDS_SLOTS = {"LCK": 4, "LPL": 4, "LEC": 3, "LCS": 2, "LCP": 2, "CBLOL": 1}
MSI_SLOTS = {"LCK": 2, "LPL": 2, "LEC": 2, "LCS": 1, "LCP": 1, "CBLOL": 1}
BLUE_EDGE = 15.0


def p_win(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


class World:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.strength = {
            team: mean + self.rng.normal(0, 70)
            for mean, teams in LEAGUES.values()
            for team in teams
        }
        self.home = {team: lg for lg, (_, teams) in LEAGUES.items() for team in teams}
        self.rows: list[dict] = []

    def new_season(self) -> None:
        """Roster churn: strengths regress toward the region mean plus fresh noise."""
        for team, s in self.strength.items():
            mean = LEAGUES[self.home[team]][0]
            self.strength[team] = mean + 0.8 * (s - mean) + self.rng.normal(0, 30)

    def form(self, team: str) -> float:
        return self.strength[team] + self.rng.normal(0, 15)  # day-to-day variance

    def play_game(self, when: datetime, league: str, event: str, series_id: str, game_no: int,
                  best_of: int, a: str, b: str, ra: float, rb: float) -> bool:
        rng = self.rng
        blue, red, rblue, rred = (a, b, ra, rb) if rng.random() < 0.5 else (b, a, rb, ra)
        blue_win = rng.random() < p_win(rblue + BLUE_EDGE, rred)
        rw, rl = (rblue, rred) if blue_win else (rred, rblue)
        gd15_w = rng.normal(700 + 0.8 * (rw - rl), 2200)
        kills_w = rng.poisson(16 + max(gd15_w, 0) / 1500)
        kills_l = rng.poisson(8 + max(-gd15_w, 0) / 1500)
        sign = 1 if blue_win else -1
        stats = {
            "towers": (int(rng.integers(8, 12)), int(rng.integers(1, 7))),
            "dragons": (int(rng.integers(2, 5)), int(rng.integers(0, 3))),
            "barons": (int(rng.integers(1, 3)), int(rng.integers(0, 2))),
            "kills": (kills_w, kills_l),
        }
        row = {
            "game_id": f"{series_id}-G{game_no}",
            "date": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "league": league,
            "event": event,
            "series_id": series_id,
            "game_no": game_no,
            "best_of": best_of,
            "blue_team": blue,
            "red_team": red,
            "winner": blue if blue_win else red,
            "duration_min": round(float(np.clip(rng.normal(32 - gd15_w / 2500, 4), 20, 50)), 1),
            "gd15": int(round(sign * gd15_w)),
        }
        for stat, (w, l) in stats.items():
            row[f"blue_{stat}"], row[f"red_{stat}"] = (w, l) if blue_win else (l, w)
        self.rows.append(row)
        return (blue_win and blue == a) or (not blue_win and red == a)

    def play_series(self, when: datetime, league: str, event: str, best_of: int, a: str, b: str) -> None:
        series_id = f"{event}-{when:%m%d%H%M}-{a}-{b}".replace(" ", "")
        ra, rb = self.form(a), self.form(b)
        need, wins_a, wins_b, game_no = best_of // 2 + 1, 0, 0, 0
        while wins_a < need and wins_b < need:
            game_no += 1
            t = when + timedelta(minutes=50 * (game_no - 1))
            if self.play_game(t, league, event, series_id, game_no, best_of, a, b, ra, rb):
                wins_a += 1
            else:
                wins_b += 1

    def domestic_split(self, year: int, split: str, start: date, end: date) -> None:
        for league, (_, teams) in LEAGUES.items():
            pairs = list(itertools.permutations(teams, 2))  # double round robin
            self.rng.shuffle(pairs)
            span = (end - start).days
            for i, (a, b) in enumerate(pairs):
                day = start + timedelta(days=int(i * span / len(pairs)))
                when = datetime(day.year, day.month, day.day, 8 + 3 * (i % 3))
                self.play_series(when, league, f"{league} {year} {split}", 3, a, b)

    def qualified(self, slots: dict[str, int]) -> list[str]:
        teams = []
        for league, n in slots.items():
            ranked = sorted(LEAGUES[league][1], key=lambda t: -self.strength[t])
            teams += ranked[:n]
        return teams

    def international(self, league: str, event: str, slots: dict[str, int], start: date,
                      days: int, series_per_day: int, best_of_cycle: list[int]) -> list[str]:
        teams = self.qualified(slots)
        for d in range(days):
            day = start + timedelta(days=d)
            order = self.rng.permutation(teams)
            for k in range(series_per_day):
                a, b = order[2 * k], order[2 * k + 1]
                when = datetime(day.year, day.month, day.day, 9 + 3 * k)
                self.play_series(when, league, event, best_of_cycle[d % len(best_of_cycle)], a, b)
        return teams


def build_matches(world: World) -> list[str]:
    for year in (2025, 2026):
        if year == 2026:
            world.new_season()
        world.domestic_split(year, "Split 1", date(year, 1, 15), date(year, 4, 15))
        world.international("MSI", f"MSI {year}", MSI_SLOTS, date(year, 6, 27), 12, 2, [5])
        world.domestic_split(year, "Split 2", date(year, 5, 1), date(year, 8, 31))
        if year == 2025:
            world.international("WLDs", "Worlds 2025", WORLDS_SLOTS, date(2025, 10, 1), 30, 3, [1, 3, 3, 5])
    # Worlds 2026 so far: play-ins and the first Swiss rounds.
    return world.international("WLDs", "Worlds 2026", WORLDS_SLOTS, date(2026, 10, 1), 13, 4, [1, 3])


def book_odds(rng: np.random.Generator, p_true: float, margin: float, noise: float) -> tuple[float, float]:
    """Price a two-way market: the book's own (noisy) view plus its margin."""
    logit = np.log(p_true / (1 - p_true)) + rng.normal(0, noise)
    p_book = 1 / (1 + np.exp(-logit))
    total = 1 + margin
    return round(1 / (p_book * total), 2), round(1 / ((1 - p_book) * total), 2)


def build_odds(world: World, teams: list[str], slate_start: date) -> pd.DataFrame:
    rng = world.rng
    rows = []
    slate = [(0, 4, "Swiss Round 4"), (1, 4, "Swiss Round 4"), (2, 3, "Swiss Round 5")]
    hours = [7, 10, 13, 16]
    for day_offset, n_matches, stage in slate:
        day = slate_start + timedelta(days=day_offset)
        order = rng.permutation(teams)
        for k in range(n_matches):
            a, b = str(order[2 * k]), str(order[2 * k + 1])
            start = datetime(day.year, day.month, day.day, hours[k], tzinfo=AEDT)
            p_map = p_win(world.strength[a], world.strength[b])
            p_series = p_map**2 * (3 - 2 * p_map)
            match_id = f"W26-{day:%m%d}-{k + 1}"
            for book, margin in (("Sportsbet", 0.055), ("PointsBet", 0.06)):
                odds_a, odds_b = book_odds(rng, p_series, margin, noise=0.15)
                rows.append(
                    {
                        "match_id": match_id,
                        "start_time": start.isoformat(),
                        "event": "Worlds 2026",
                        "stage": stage,
                        "best_of": 3,
                        "team_a": a,
                        "team_b": b,
                        "bookmaker": book,
                        "odds_a": odds_a,
                        "odds_b": odds_b,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slate-start", default="2026-10-15", help="first date (AEDT) of the sample odds slate")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    world = World(args.seed)
    worlds_teams = build_matches(world)
    matches = pd.DataFrame(world.rows)
    odds = build_odds(world, worlds_teams, date.fromisoformat(args.slate_start))

    OUT.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUT / "matches.csv", index=False)
    odds.to_csv(OUT / "odds.csv", index=False)
    print(f"wrote {len(matches):,} synthetic games and {len(odds)} odds rows to {OUT}")


if __name__ == "__main__":
    main()
