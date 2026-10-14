import pandas as pd
import pytest

from lolworlds.ingest import (
    build_ratings,
    clean_games,
    from_oracles_elixir,
    infer_home_leagues,
    is_oracles_elixir,
    team_metrics,
)


def native(rows):
    base = {"date": "2026-03-01", "league": "LCK", "blue_team": "A", "red_team": "B", "winner": "A"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_winner_accepts_team_name_or_side():
    g = clean_games(native([{"winner": "A"}, {"winner": "red", "date": "2026-03-02"}, {"winner": "b", "date": "2026-03-03"}]))
    assert g.blue_win.tolist() == [1, 0, 0]
    assert g.winner.tolist() == ["A", "B", "B"]


def test_invalid_rows_are_dropped():
    g = clean_games(native([
        {"winner": "Someone else"},
        {"red_team": "A", "date": "2026-03-02"},      # team vs itself
        {"date": "not a date"},
        {"date": "2026-03-04"},                       # the only valid row
    ]))
    assert len(g) == 1


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="winner"):
        clean_games(native([{}]).drop(columns="winner"))


def test_games_are_sorted_and_deduplicated():
    g = clean_games(native([
        {"date": "2026-03-05", "game_id": "x"},
        {"date": "2026-03-01", "game_id": "y"},
        {"date": "2026-03-05", "game_id": "x"},
    ]))
    assert g.game_id.tolist() == ["y", "x"]


def oe_rows(gameid, blue, red, blue_won, league="LCK", gd15=1200):
    rows = []
    for side, team, won, sign in (("Blue", blue, blue_won, 1), ("Red", red, not blue_won, -1)):
        for pos in ("top", "jng", "mid", "bot", "sup"):
            rows.append({"gameid": gameid, "league": league, "date": "2026-03-01 08:00:00", "side": side,
                         "position": pos, "teamname": team, "result": int(won), "kills": 1})
        rows.append({"gameid": gameid, "league": league, "date": "2026-03-01 08:00:00", "side": side,
                     "position": "team", "teamname": team, "result": int(won), "kills": 15 if won else 5,
                     "gamelength": 1800, "golddiffat15": sign * gd15, "game": 1})
    return rows


def test_oracles_elixir_export_is_converted():
    raw = pd.DataFrame(oe_rows("g1", "T1", "Gen.G", True) + oe_rows("g2", "Gen.G", "T1", False))
    assert is_oracles_elixir(raw)
    g = clean_games(from_oracles_elixir(raw))
    assert len(g) == 2
    first = g.iloc[0]
    assert (first.blue_team, first.red_team, first.winner) == ("T1", "Gen.G", "T1")
    assert first.duration_min == 30
    assert first.gd15 == 1200
    assert (first.blue_kills, first.red_kills) == (15, 5)


def test_home_league_ignores_international_events(settings):
    g = clean_games(native([
        {"league": "WLDs", "date": "2026-10-01"},
        {"league": "WLDs", "date": "2026-10-02"},
        {"league": "LEC", "date": "2026-03-01", "red_team": "C"},
        {"league": "LCK", "date": "2026-03-02", "blue_team": "B", "red_team": "D", "winner": "B"},
    ]))
    home = infer_home_leagues(g, settings.elo.international_leagues)
    assert home["A"] == "LEC"
    assert home["B"] == "LCK"


def test_team_metrics_signs_stats_per_team():
    g = clean_games(native([{"gd15": 2000, "blue_kills": 20, "red_kills": 5}]))
    m = team_metrics(g).set_index("team")
    assert m.loc["A", "avg_gd15"] == 2000 and m.loc["B", "avg_gd15"] == -2000
    assert m.loc["A", "avg_kill_diff"] == 15
    assert m.loc["A", "form"] == "1-0" and m.loc["B", "form"] == "0-1"


def test_build_ratings_end_to_end(settings, tmp_path):
    rows = []
    for i in range(60):
        strong_wins = i % 3 != 0
        rows.append({"date": f"2026-0{1 + i // 20}-{1 + i % 20:02d}", "league": "LCK" if i % 2 else "LEC",
                     "blue_team": "Strong", "red_team": "Weak", "winner": "Strong" if strong_wins else "Weak",
                     "sides_known": False})
    path = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = build_ratings(settings, path)
    assert len(result.games) == 60
    assert not result.games.sides_known.any()  # "False" strings from CSV parsed as booleans
    assert result.ratings.team.tolist() == ["Strong", "Weak"]


def test_missing_match_file_points_to_fetch(settings, tmp_path):
    with pytest.raises(FileNotFoundError, match="lolworlds.fetch"):
        build_ratings(settings, tmp_path / "nope.csv")
