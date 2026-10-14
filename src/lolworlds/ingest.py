"""Data ingestion: load historical games, compute team metrics and Elo ratings.

Accepts either of two CSV layouts:

* **Native** - one row per game: ``date, league, blue_team, red_team, winner`` plus
  optional ``event, series_id, game_no, best_of, duration_min, gd15, blue_kills,
  red_kills, blue_towers, red_towers, blue_dragons, red_dragons, blue_barons,
  red_barons``. ``winner`` may be a team name or ``blue``/``red``; ``gd15`` is blue's
  gold lead at 15 minutes.
* **Oracle's Elixir** - the public per-player/per-team match data export
  (detected automatically; only the ``position == "team"`` rows are used).

Run::

    python -m lolworlds.ingest                       # uses paths from config.toml
    python -m lolworlds.ingest --input data/raw/2026_LoL_esports_match_data.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .elo import EloModel, calibration_summary

NATIVE_REQUIRED = ["date", "league", "blue_team", "red_team", "winner"]
METRIC_COLUMNS = [
    "duration_min", "gd15",
    "blue_kills", "red_kills", "blue_towers", "red_towers",
    "blue_dragons", "red_dragons", "blue_barons", "red_barons",
]
OE_REQUIRED = {"gameid", "teamname", "position", "side", "result", "date", "league"}


def is_oracles_elixir(df: pd.DataFrame) -> bool:
    return OE_REQUIRED <= set(df.columns)


def from_oracles_elixir(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert Oracle's Elixir rows (10 players + 2 team rows per game) to native games."""
    teams = raw[raw["position"].astype(str).str.lower() == "team"].copy()
    teams["side"] = teams["side"].astype(str).str.lower()
    teams = teams.drop_duplicates(["gameid", "side"])
    blue = teams[teams.side == "blue"].set_index("gameid")
    red = teams[teams.side == "red"].set_index("gameid")
    ids = blue.index.intersection(red.index)
    blue, red = blue.loc[ids], red.loc[ids]

    def col(frame: pd.DataFrame, name: str) -> pd.Series:
        return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)

    event = blue["league"].astype(str)
    for extra in ("year", "split"):
        if extra in blue:
            event = event + " " + blue[extra].fillna("").astype(str)
    games = pd.DataFrame(
        {
            "game_id": ids,
            "date": blue["date"],
            "league": blue["league"],
            "event": event.str.strip(),
            "game_no": col(blue, "game"),
            "blue_team": blue["teamname"],
            "red_team": red["teamname"],
            "winner": np.where(blue["result"].astype(int) == 1, blue["teamname"], red["teamname"]),
            "duration_min": col(blue, "gamelength") / 60.0,
            "gd15": col(blue, "golddiffat15"),
            "blue_kills": col(blue, "kills"),
            "red_kills": col(red, "kills"),
            "blue_towers": col(blue, "towers"),
            "red_towers": col(red, "towers"),
            "blue_dragons": col(blue, "dragons"),
            "red_dragons": col(red, "dragons"),
            "blue_barons": col(blue, "barons"),
            "red_barons": col(red, "barons"),
        }
    ).reset_index(drop=True)
    day = pd.to_datetime(games["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pair = [" vs ".join(sorted(p)) for p in zip(games.blue_team, games.red_team)]
    games["series_id"] = day + "|" + games["league"].astype(str) + "|" + pd.Series(pair)
    return games


def clean_games(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise native-format games, sorted for Elo replay."""
    missing = [c for c in NATIVE_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"match data is missing required columns: {missing}")
    g = df.copy()
    for name in ["event", "series_id", "game_id"]:
        if name not in g:
            g[name] = None
    if "game_no" not in g:
        g["game_no"] = 1
    if "best_of" not in g:
        g["best_of"] = np.nan
    for name in METRIC_COLUMNS:
        g[name] = pd.to_numeric(g[name], errors="coerce") if name in g else np.nan

    g["date"] = pd.to_datetime(g["date"], errors="coerce", utc=True)
    for name in ["blue_team", "red_team", "winner", "league"]:
        g[name] = g[name].astype("string").str.strip()
    g = g.dropna(subset=["date", "blue_team", "red_team", "winner", "league"])
    g = g[g.blue_team != g.red_team]

    winner = g["winner"].str.lower()
    blue_won = (winner == "blue") | (winner == g.blue_team.str.lower())
    red_won = (winner == "red") | (winner == g.red_team.str.lower())
    g = g[blue_won | red_won].copy()
    g["blue_win"] = blue_won[g.index].astype(int)
    g["winner"] = np.where(g.blue_win == 1, g.blue_team, g.red_team)

    g["game_no"] = pd.to_numeric(g["game_no"], errors="coerce").fillna(1).astype(int)
    fallback_id = (
        g.date.dt.strftime("%Y%m%d%H%M") + "-" + g.blue_team + "-" + g.red_team + "-" + g.game_no.astype(str)
    )
    g["game_id"] = g["game_id"].fillna(fallback_id).astype(str)
    g = g.drop_duplicates("game_id")
    return g.sort_values(["date", "game_no"], kind="stable", ignore_index=True)


def load_games(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    if is_oracles_elixir(raw):
        raw = from_oracles_elixir(raw)
    return clean_games(raw)


def team_perspective(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game) with stats signed from that team's point of view."""
    sides = []
    for side, opp, sign in (("blue", "red", 1), ("red", "blue", -1)):
        sides.append(
            pd.DataFrame(
                {
                    "team": games[f"{side}_team"],
                    "opponent": games[f"{opp}_team"],
                    "date": games["date"],
                    "league": games["league"],
                    "side": side,
                    "win": games["blue_win"] if sign == 1 else 1 - games["blue_win"],
                    "gd15": sign * games["gd15"],
                    "kill_diff": games[f"{side}_kills"] - games[f"{opp}_kills"],
                    "duration_min": games["duration_min"],
                }
            )
        )
    return pd.concat(sides, ignore_index=True).sort_values("date", kind="stable", ignore_index=True)


def infer_home_leagues(games: pd.DataFrame, international: tuple[str, ...]) -> dict[str, str]:
    """A team's home league is the domestic league it has played most games in."""
    long = team_perspective(games)
    domestic = long[~long.league.isin(international)]
    if domestic.empty:
        return {}
    counts = domestic.groupby(["team", "league"]).size().reset_index(name="n")
    best = counts.sort_values("n", ascending=False).drop_duplicates("team")
    return dict(zip(best.team, best.league))


def team_metrics(games: pd.DataFrame, last_n: int = 10) -> pd.DataFrame:
    long = team_perspective(games)
    recent = long.groupby("team").tail(last_n)
    overall = long.groupby("team").agg(
        avg_gd15=("gd15", "mean"),
        avg_kill_diff=("kill_diff", "mean"),
        avg_game_min=("duration_min", "mean"),
    )
    by_side = long.groupby(["team", "side"]).win.mean().unstack()
    overall = overall.join(by_side.add_suffix("_win_rate"))
    form = recent.groupby("team").agg(
        form_win_rate=("win", "mean"),
        form_wins=("win", "sum"),
        form_games=("win", "size"),
        form_gd15=("gd15", "mean"),
    )
    out = overall.join(form)
    out["form"] = out.form_wins.astype(int).astype(str) + "-" + (out.form_games - out.form_wins).astype(int).astype(str)
    return out.drop(columns=["form_wins", "form_games"]).reset_index()


@dataclass
class IngestResult:
    games: pd.DataFrame
    history: pd.DataFrame
    model: EloModel
    ratings: pd.DataFrame
    calibration: dict
    source: Path
    is_sample: bool


def build_ratings(settings: Settings, input_path: str | Path | None = None) -> IngestResult:
    if input_path is not None:
        source, is_sample = Path(input_path), False
    else:
        source, is_sample = settings.paths.matches_source()
    games = load_games(source)
    if games.empty:
        raise ValueError(f"no usable games found in {source}")

    model = EloModel(settings.elo, infer_home_leagues(games, settings.elo.international_leagues))
    history = model.fit(games)
    ratings = model.table().merge(team_metrics(games), on="team", how="left")
    return IngestResult(
        games=games,
        history=history,
        model=model,
        ratings=ratings,
        calibration=calibration_summary(history),
        source=source,
        is_sample=is_sample,
    )


def save_outputs(result: IngestResult, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "elo_ratings.csv", out / "elo_history.csv"]
    result.ratings.to_csv(written[0], index=False, float_format="%.4f")
    result.history.to_csv(written[1], index=False, float_format="%.4f")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load historical LoL games and compute Elo ratings.")
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("--input", help="match CSV (native or Oracle's Elixir format)")
    parser.add_argument("--output-dir", help="where to write elo_ratings.csv / elo_history.csv")
    parser.add_argument("--top", type=int, default=20, help="how many teams to print")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    result = build_ratings(settings, args.input)
    written = save_outputs(result, args.output_dir or settings.paths.processed_dir)

    label = " (bundled SAMPLE data - synthetic)" if result.is_sample else ""
    print(f"Loaded {len(result.games):,} games from {result.source}{label}")
    print(f"Date range: {result.games.date.min():%Y-%m-%d} to {result.games.date.max():%Y-%m-%d}")
    cal = result.calibration
    print(
        f"Pre-game prediction quality over {cal['games']:,} games: "
        f"log loss {cal['log_loss']:.4f}, Brier {cal['brier']:.4f}, accuracy {cal['accuracy']:.1%}"
    )
    cols = ["rank", "team", "league", "rating", "games", "win_rate", "form", "avg_gd15"]
    print()
    print(result.ratings[cols].head(args.top).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print()
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
