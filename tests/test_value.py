import pandas as pd
import pytest

from lolworlds.elo import EloModel
from lolworlds.value import (
    canonical_bookmaker,
    clean_odds,
    evaluate_sides,
    expected_value,
    find_value,
    market_board,
    overround,
    remove_vig,
    resolve_team,
    select_value_bets,
)


def test_overround_and_ev():
    assert overround([1.90, 1.90]) == pytest.approx(2 / 1.9 - 1)
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert expected_value(0.40, 2.0) == pytest.approx(-0.20)


@pytest.mark.parametrize("method", ["multiplicative", "power"])
def test_remove_vig_returns_a_fair_book(method):
    probs = remove_vig([1.30, 3.40], method)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1]


def test_power_method_shades_the_longshot_more():
    mult = remove_vig([1.20, 4.50], "multiplicative")
    power = remove_vig([1.20, 4.50], "power")
    assert power[1] < mult[1]


def test_bookmaker_names_are_normalised(settings):
    assert canonical_bookmaker("pointsbet", settings.bookmakers) == "PointsBet"
    assert canonical_bookmaker("Points Bet", settings.bookmakers) == "PointsBet"
    assert canonical_bookmaker("SPORTSBET", settings.bookmakers) == "Sportsbet"
    assert canonical_bookmaker("Ladbrokes", settings.bookmakers) is None


def test_team_resolution_uses_aliases_and_case():
    known = {"Gen.G", "T1"}
    aliases = {"GEN": "Gen.G"}
    assert resolve_team("GEN", known, aliases) == "Gen.G"
    assert resolve_team("gen", known, aliases) == "Gen.G"
    assert resolve_team("t1", known, aliases) == "T1"
    assert resolve_team("Nobody", known, aliases) is None


def odds_frame(settings, rows):
    base = {"match_id": "M1", "start_time": "2026-10-15T19:00:00+11:00", "event": "Worlds", "stage": "Swiss",
            "best_of": 3, "team_a": "Strong", "team_b": "Weak"}
    return clean_odds(pd.DataFrame([{**base, **r} for r in rows]), settings)


def test_board_picks_the_best_price_per_side(settings):
    odds = odds_frame(settings, [
        {"bookmaker": "Sportsbet", "odds_a": 1.50, "odds_b": 2.55},
        {"bookmaker": "pointsbet", "odds_a": 1.55, "odds_b": 2.45},
        {"bookmaker": "Unlisted Book", "odds_a": 9.0, "odds_b": 9.0},  # ignored
    ])
    board = market_board(odds, settings).iloc[0]
    assert (board.best_odds_a, board.best_book_a) == (1.55, "PointsBet")
    assert (board.best_odds_b, board.best_book_b) == (2.55, "Sportsbet")
    assert 0.5 < board.market_prob_a < 0.7


def model_with(settings, strong=1700.0, weak=1500.0):
    model = EloModel(settings.elo)
    model._state("Strong").rating = strong
    model._state("Weak").rating = weak
    return model


def test_value_bet_found_when_price_beats_model(settings, aggressive):
    profile = aggressive.with_overrides(max_prob_gap=0.5)
    odds = odds_frame(settings, [{"bookmaker": "Sportsbet", "odds_a": 1.60, "odds_b": 2.30}])
    board, sides, bets = find_value(odds, model_with(settings), settings, profile)
    assert len(sides) == 2
    assert bets.selection.tolist() == ["Strong"]
    bet = bets.iloc[0]
    expected_prob = profile.model_weight * board.model_prob_a[0] + (1 - profile.model_weight) * board.market_prob_a[0]
    assert bet.bet_prob == pytest.approx(expected_prob)
    assert bet.ev == pytest.approx(expected_prob * 1.60 - 1)


def test_filters_explain_why_a_side_is_skipped(settings, aggressive):
    odds = odds_frame(settings, [{"bookmaker": "Sportsbet", "odds_a": 1.60, "odds_b": 2.30}])
    strict = aggressive.with_overrides(max_prob_gap=0.05, max_odds=2.0)
    _, sides, bets = find_value(odds, model_with(settings), settings, strict)
    assert bets.empty
    reasons = dict(zip(sides.selection, sides.reason))
    assert "gap" in reasons["Strong"]
    assert "odds outside" in reasons["Weak"] and "EV" in reasons["Weak"]


def test_unrated_teams_are_never_bet(settings, aggressive):
    odds = odds_frame(settings, [{"bookmaker": "Sportsbet", "odds_a": 5.0, "odds_b": 5.0, "team_a": "Mystery"}])
    _, sides, bets = find_value(odds, model_with(settings), settings, aggressive)
    assert bets.empty
    assert all("not in rating data" in r for r in sides.reason)


def test_only_one_side_per_match_is_selected():
    sides = pd.DataFrame({
        "match_id": ["M1", "M1", "M2"], "start_time": pd.to_datetime(["2026-10-15"] * 3),
        "selection": ["A", "B", "C"], "ev": [0.05, 0.08, 0.04], "qualifies": [True, True, True],
    })
    picked = select_value_bets(sides)
    assert picked.selection.tolist() == ["B", "C"]


def test_bundled_sample_odds_load(settings):
    from lolworlds.value import load_odds

    odds = load_odds(settings.paths.sample_odds, settings)
    assert set(odds.bookmaker) == {"Sportsbet", "PointsBet"}
    assert (odds.odds_a > 1).all() and (odds.odds_b > 1).all()
    assert str(odds.start_time.dt.tz) == settings.timezone


def test_blank_stage_and_event_become_empty_strings(settings):
    odds = odds_frame(settings, [{"bookmaker": "Sportsbet", "odds_a": 1.8, "odds_b": 2.0,
                                  "stage": float("nan"), "event": None}])
    assert odds.stage.iloc[0] == "" and odds.event.iloc[0] == ""
