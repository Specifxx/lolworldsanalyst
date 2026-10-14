import math

import pandas as pd
import pytest

from lolworlds.elo import EloModel, calibration_summary, expected_score, series_win_prob


def test_expected_score_is_symmetric():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1600, 1500) + expected_score(1500, 1600) == pytest.approx(1.0)
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)  # 400 points = 10:1


@pytest.mark.parametrize("p", [0.3, 0.5, 0.62, 0.9])
def test_series_probability_matches_closed_forms(p):
    assert series_win_prob(p, 1) == pytest.approx(p)
    assert series_win_prob(p, 3) == pytest.approx(p**2 * (3 - 2 * p))
    assert series_win_prob(p, 5) == pytest.approx(p**3 * (10 - 15 * p + 6 * p**2))


def test_longer_series_favour_the_stronger_team():
    assert series_win_prob(0.6, 1) < series_win_prob(0.6, 3) < series_win_prob(0.6, 5)
    assert series_win_prob(0.5, 5) == pytest.approx(0.5)


@pytest.mark.parametrize("best_of", [0, 2, 4, -3])
def test_series_rejects_even_lengths(best_of):
    with pytest.raises(ValueError):
        series_win_prob(0.5, best_of)


def update(model, blue="A", red="B", blue_win=True, league="LCK", date="2026-03-01", **kw):
    return model.update(blue=blue, red=red, blue_win=blue_win, date=pd.Timestamp(date, tz="UTC"), league=league, **kw)


def test_winner_gains_and_loser_loses(settings):
    model = EloModel(settings.elo, {"A": "LCK", "B": "LCK"})
    update(model)
    assert model.rating("A") > settings.elo.region_priors["LCK"] > model.rating("B")


def test_region_priors_seed_new_teams(settings):
    model = EloModel(settings.elo, {"A": "LCK", "B": "LCS"})
    assert model.rating("A") == settings.elo.region_priors["LCK"]
    assert model.rating("B") == settings.elo.region_priors["LCS"]
    assert model.rating("Unknown") == settings.elo.default_rating


def test_upsets_move_ratings_more_than_expected_wins(settings):
    fav_win = EloModel(settings.elo, {"Fav": "LCK", "Dog": "LCS"})
    rec = update(fav_win, blue="Fav", red="Dog", blue_win=True)
    upset = EloModel(settings.elo, {"Fav": "LCK", "Dog": "LCS"})
    rec_upset = update(upset, blue="Fav", red="Dog", blue_win=False)
    assert abs(rec_upset["delta_blue"]) > abs(rec["delta_blue"])


def test_international_games_use_a_bigger_k(settings):
    dom = EloModel(settings.elo, {"A": "LCK", "B": "LCK"})
    intl = EloModel(settings.elo, {"A": "LCK", "B": "LCK"})
    d = update(dom, league="LCK")["delta_blue"]
    i = update(intl, league="WLDs")["delta_blue"]
    assert i == pytest.approx(d * settings.elo.international_k_multiplier)


def test_dominant_wins_move_ratings_more(settings):
    close = EloModel(settings.elo, {"A": "LCK", "B": "LCK"})
    stomp = EloModel(settings.elo, {"A": "LCK", "B": "LCK"})
    d_close = update(close, gd15=100, kill_diff=1)["delta_blue"]
    d_stomp = update(stomp, gd15=6000, kill_diff=20)["delta_blue"]
    d_missing = update(EloModel(settings.elo, {"A": "LCK", "B": "LCK"}), gd15=math.nan)["delta_blue"]
    assert d_stomp > d_close >= d_missing > 0


def test_new_season_regresses_toward_prior(settings):
    model = EloModel(settings.elo, {"A": "LCK", "B": "LCK"})
    for _ in range(30):
        update(model, date="2025-05-01")
    before = model.rating("A")
    prior = settings.elo.region_priors["LCK"]
    record = update(model, date="2026-01-20")  # first game of the new year
    assert before > prior
    assert record["blue_rating_pre"] == pytest.approx(prior + settings.elo.season_carryover * (before - prior))


def test_fit_ranks_the_stronger_team_higher(settings):
    rows = []
    for i in range(40):
        a_wins = i % 4 != 0  # A wins 75%
        rows.append({"date": pd.Timestamp("2026-02-01", tz="UTC") + pd.Timedelta(days=i), "league": "LCK",
                     "blue_team": "A" if i % 2 else "B", "red_team": "B" if i % 2 else "A",
                     "blue_win": int(a_wins == bool(i % 2)), "gd15": float("nan"),
                     "blue_kills": float("nan"), "red_kills": float("nan")})
    games = pd.DataFrame(rows)
    model = EloModel(settings.elo, {"A": "LCK", "B": "LCK"})
    history = model.fit(games)
    assert model.rating("A") > model.rating("B")
    assert len(history) == 40 and "p_blue" in history
    assert calibration_summary(history)["games"] > 0


def test_side_unknown_games_get_no_blue_bonus(settings):
    sided = update(EloModel(settings.elo, {"A": "LCK", "B": "LCK"}))
    neutral = update(EloModel(settings.elo, {"A": "LCK", "B": "LCK"}), neutral=True)
    assert neutral["p_blue"] == pytest.approx(0.5)
    assert sided["p_blue"] > 0.5
