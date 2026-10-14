import pytest

from lolworlds.fetch import _game_order, fixture_row, series_to_games


@pytest.mark.parametrize("w,l,expected", [
    (1, 0, [True]),
    (2, 0, [True, True]),
    (2, 1, [True, False, True]),
    (3, 1, [True, False, True, True]),
    (3, 2, [True, False, True, False, True]),
])
def test_game_order_ends_with_the_series_winner(w, l, expected):
    assert _game_order(w, l) == expected


def event(state="completed", wins=(1, 3), names=("Cloud9 Kia", "Karmine Corp"), count=5):
    return {
        "startTime": "2026-10-15T18:00:00Z", "state": state, "type": "match", "blockName": "Play-Ins",
        "league": {"name": "Worlds", "slug": "worlds"},
        "match": {"id": "123", "strategy": {"type": "bestOf", "count": count},
                  "teams": [{"name": n, "result": {"gameWins": g} if state == "completed" else None}
                            for n, g in zip(names, wins)]},
    }


def test_series_expands_to_games():
    games = series_to_games(event(), "WLDs")
    assert len(games) == 4
    assert [g["winner"] for g in games].count("Karmine Corp") == 3
    assert games[-1]["winner"] == "Karmine Corp"
    assert all(not g["sides_known"] and g["league"] == "WLDs" and g["best_of"] == 5 for g in games)
    assert games[0]["event"] == "Worlds 2026"
    assert games[1]["date"] > games[0]["date"]


@pytest.mark.parametrize("ev", [event(state="unstarted"), event(wins=(0, 0)), event(wins=(1, 1))])
def test_unplayed_or_odd_series_are_skipped(ev):
    assert series_to_games(ev, "WLDs") == []


def test_fixtures_need_both_teams_confirmed():
    row = fixture_row(event(state="unstarted"))
    assert row["team_a"] == "Cloud9 Kia" and row["best_of"] == 5 and row["stage"] == "Play-Ins"
    assert fixture_row(event(state="unstarted", names=("TBD", "Karmine Corp"))) is None
    assert fixture_row(event()) is None  # already played
