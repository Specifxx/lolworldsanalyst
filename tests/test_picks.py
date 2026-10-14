import pytest

from lolworlds.bankroll import SizingContext
from lolworlds.elo import fair_odds
from lolworlds.picks import price_ladder, single_price_check


def test_minimum_price_is_above_fair_odds(aggressive):
    ctx = SizingContext(bankroll=500, peak_bankroll=500)
    ladder = price_ladder(0.70, aggressive, ctx)
    assert ladder[0]["odds"] > fair_odds(0.70)
    assert ladder[0]["ev"] >= aggressive.min_ev


def test_better_prices_never_lower_the_stake(aggressive):
    ladder = price_ladder(0.70, aggressive, SizingContext(bankroll=500, peak_bankroll=500))
    stakes = [r["stake"] for r in ladder]
    assert stakes == sorted(stakes)
    assert max(stakes) <= 0.15 * 500


def test_big_model_market_gap_is_not_bet(aggressive):
    ok, _, ev = single_price_check(0.90, 3.0, aggressive)  # market implies ~32%, model says 90%
    assert ev > 0 and not ok


def test_no_ladder_for_a_hopeless_side(aggressive):
    assert price_ladder(0.03, aggressive, SizingContext(bankroll=500, peak_bankroll=500)) == []
