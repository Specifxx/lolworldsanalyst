"""Download real results and fixtures from the official LoL Esports API.

This is the API behind lolesports.com. It returns series scores (not per-game
stats or sides), so each series is expanded into individual games with
``sides_known = False``.

Run::

    python -m lolworlds.fetch                 # results since 2025-01-01 + upcoming Worlds fixtures
    python -m lolworlds.fetch --since 2024-01-01
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import load_settings

API = "https://esports-api.lolesports.com/persisted/gw"
# Public key that the lolesports.com website itself sends with every request.
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"

# Top-flight leagues and international events. slug -> (league id, code used in our
# data). LTA North/South map back to their LCS/CBLOL regions; cross-region events get
# their own codes (listed as international in config.toml).
LEAGUES = {
    "worlds": ("98767975604431411", "WLDs"),
    "msi": ("98767991325878492", "MSI"),
    "first_stand": ("113464388705111224", "FST"),
    "ewc_lol": ("116838530616006090", "EWC"),
    "americas_cup": ("116096325848746167", "AMC"),
    "lta_cross": ("113475149040947852", "LTA"),
    "lck": ("98767991310872058", "LCK"),
    "lpl": ("98767991314006698", "LPL"),
    "lec": ("98767991302996019", "LEC"),
    "lcs": ("98767991299243165", "LCS"),
    "lta_n": ("113470291645289904", "LCS"),
    "cblol-brazil": ("98767991332355509", "CBLOL"),
    "lta_s": ("113475181634818701", "CBLOL"),
    "lcp": ("113476371197627891", "LCP"),
}
# Second-tier leagues (VCS, PCS, LJL feed the LCP since 2025) are left out: their teams
# rarely play anyone outside their league, so their ratings float free and distort.
FIXTURE_LEAGUES = ("worlds",)


def _get(endpoint: str, **params) -> dict:
    url = f"{API}/{endpoint}?" + urllib.parse.urlencode({"hl": "en-US", **params})
    req = urllib.request.Request(url, headers={"x-api-key": API_KEY, "User-Agent": "lolworldsanalyst"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)["data"]
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def league_events(league_id: str, since: str) -> list[dict]:
    """All events for a league from ``since`` (ISO date) onwards, oldest first."""
    first = _get("getSchedule", leagueId=league_id)["schedule"]
    events, pages = list(first["events"]), first["pages"]
    older, newer = pages.get("older"), pages.get("newer")
    while older and (not events or events[0]["startTime"] >= since):
        page = _get("getSchedule", leagueId=league_id, pageToken=older)["schedule"]
        events = page["events"] + events
        older = page["pages"].get("older")
    while newer:
        page = _get("getSchedule", leagueId=league_id, pageToken=newer)["schedule"]
        events += page["events"]
        newer = page["pages"].get("newer")
    return [e for e in events if e.get("type") == "match" and e["startTime"] >= since]


def _game_order(winner_games: int, loser_games: int) -> list[bool]:
    """A plausible game order (True = series winner won): loser's wins spread out, winner takes the last."""
    order = []
    for i in range(winner_games + loser_games - 1):
        loser_turn = loser_games > 0 and i % 2 == 1
        order.append(not loser_turn)
        loser_games -= loser_turn
    return order + [True]


def series_to_games(event: dict, code: str) -> list[dict]:
    """Expand a completed series into one row per game (sides unknown)."""
    match = event.get("match") or {}
    teams = match.get("teams") or []
    if event.get("state") != "completed" or len(teams) != 2:
        return []
    wins = [(t.get("result") or {}).get("gameWins") or 0 for t in teams]
    if sum(wins) == 0 or wins[0] == wins[1]:
        return []  # forfeits / no games played
    w, l = (0, 1) if wins[0] > wins[1] else (1, 0)
    start = datetime.fromisoformat(event["startTime"].replace("Z", "+00:00"))
    year = start.year
    rows = []
    for n, winner_won in enumerate(_game_order(wins[w], wins[l]), start=1):
        rows.append(
            {
                "game_id": f"{match['id']}-{n}",
                "date": (start + timedelta(minutes=45 * (n - 1))).isoformat(),
                "league": code,
                "event": f"{event['league']['name']} {year}",
                "series_id": match["id"],
                "game_no": n,
                "best_of": (match.get("strategy") or {}).get("count"),
                "blue_team": teams[0]["name"],
                "red_team": teams[1]["name"],
                "winner": teams[w]["name"] if winner_won else teams[l]["name"],
                "sides_known": False,
            }
        )
    return rows


def fixture_row(event: dict) -> dict | None:
    match = event.get("match") or {}
    teams = match.get("teams") or []
    if event.get("state") == "completed" or len(teams) != 2 or any(t.get("name") in (None, "TBD") for t in teams):
        return None
    return {
        "match_id": match["id"],
        "start_time": event["startTime"],
        "event": f"{event['league']['name']} {event['startTime'][:4]}",
        "stage": event.get("blockName") or "",
        "best_of": (match.get("strategy") or {}).get("count"),
        "team_a": teams[0]["name"],
        "team_b": teams[1]["name"],
    }


def fetch(since: str = "2025-01-01", verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    games, fixtures = [], []
    for slug, (league_id, code) in LEAGUES.items():
        events = league_events(league_id, since)
        before = len(games)
        for e in events:
            games += series_to_games(e, code)
            if slug in FIXTURE_LEAGUES and (row := fixture_row(e)):
                fixtures.append(row)
        if verbose:
            print(f"{slug:14s} {len(events):4d} series -> {len(games) - before:5d} games", flush=True)
    games_df = pd.DataFrame(games).sort_values("date", ignore_index=True)
    fixtures_df = pd.DataFrame(fixtures, columns=["match_id", "start_time", "event", "stage", "best_of", "team_a", "team_b"])
    return games_df, fixtures_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download real LoL esports results and Worlds fixtures.")
    parser.add_argument("--since", default="2025-01-01", help="earliest match date to download (YYYY-MM-DD)")
    parser.add_argument("--config", help="path to config.toml")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    games, fixtures = fetch(args.since)
    settings.paths.matches.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(settings.paths.matches, index=False)
    fixtures.to_csv(settings.paths.fixtures, index=False)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{len(games):,} games -> {settings.paths.matches}")
    print(f"{len(fixtures)} upcoming Worlds fixtures -> {settings.paths.fixtures}  (fetched {stamp})")


if __name__ == "__main__":
    main()
