import itertools

import numpy as np
import pandas as pd
import pytest

from lolworlds.bankroll import (
    SizingContext,
    _project,
    expected_log_growth,
    kelly_fraction,
    kelly_multiplier,
    recommend_stakes,
    simulate_bankroll,
    simultaneous_kelly,
    suggest_multiplier,
)
from lolworlds.config import RiskProfile


def test_single_bet_kelly():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    assert kelly_fraction(0.5, 3.0) == pytest.approx(0.25)
    assert kelly_fraction(0.4, 2.0) == 0.0  # negative EV -> no bet


def test_projection_respects_the_cap():
    x = _project(np.array([0.6, 0.5, -0.1]), 0.8)
    assert x.min() >= 0 and x.sum() == pytest.approx(0.8)
    assert np.array_equal(_project(np.array([0.1, 0.2]), 0.8), np.array([0.1, 0.2]))


def test_simultaneous_kelly_matches_single_kelly_for_one_bet():
    assert simultaneous_kelly([0.6], [2.0])[0] == pytest.approx(0.2)


def test_simultaneous_kelly_beats_naive_stacking():
    p, o = [0.6, 0.55, 0.5], [2.0, 2.1, 2.3]
    single = np.array([kelly_fraction(pi, oi) for pi, oi in zip(p, o)])
    joint = simultaneous_kelly(p, o)
    assert np.all(joint <= single + 1e-9)
    assert joint.sum() < single.sum()
    assert expected_log_growth(joint, p, o) > expected_log_growth(single, p, o)


def test_simultaneous_kelly_matches_brute_force_grid():
    p, o = [0.62, 0.48], [1.9, 2.4]
    grid = np.linspace(0, 0.5, 101)
    best = max(itertools.product(grid, grid), key=lambda f: expected_log_growth(f, p, o))
    assert simultaneous_kelly(p, o) == pytest.approx(np.array(best), abs=0.006)


def profile(**overrides):
    base = dict(name="test", kelly_fraction=0.5, model_weight=0.75, min_ev=0.03, max_prob_gap=0.2,
                min_odds=1.2, max_odds=5.0, max_stake_pct=0.15, max_exposure_pct=0.5,
                drawdown_brake_pct=0.4, drawdown_brake_multiplier=0.5)
    return RiskProfile(**{**base, **overrides})


def bets(*rows):
    return pd.DataFrame([{"bet_prob": p, "odds": o, "ev": p * o - 1} for p, o in rows])


def test_half_kelly_stake_rounded_down():
    out = recommend_stakes(bets((0.6, 2.0)), profile(), SizingContext(bankroll=500, peak_bankroll=500))
    # full Kelly 20% -> half Kelly 10% of $500 = $50
    assert out.stake[0] == 50.0
    assert out.to_return[0] == 100.0
    assert out.expected_profit[0] == pytest.approx(50 * 0.2)


def test_rounding_to_fifty_cents():
    out = recommend_stakes(bets((0.6, 2.0)), profile(), SizingContext(bankroll=503.9, peak_bankroll=503.9))
    assert out.stake[0] == 50.0  # 50.39 rounds down to the 50c increment


def test_per_bet_cap():
    out = recommend_stakes(bets((0.8, 2.0)), profile(), SizingContext(bankroll=500, peak_bankroll=500))
    assert out.stake[0] == 75.0  # half Kelly would be 30%, capped at 15%
    assert "capped" in out.notes[0]


def test_exposure_cap_counts_open_bets():
    ctx = SizingContext(bankroll=500, peak_bankroll=500, open_exposure=200)
    out = recommend_stakes(bets((0.7, 2.0), (0.7, 2.1)), profile(), ctx)
    assert out.stake.sum() <= 0.5 * 500 - 200 + 1e-9
    assert all("exposure" in n for n in out.notes)


def test_drawdown_brake_halves_stakes():
    calm = recommend_stakes(bets((0.6, 2.0)), profile(), SizingContext(bankroll=500, peak_bankroll=500))
    braked = recommend_stakes(bets((0.6, 2.0)), profile(), SizingContext(bankroll=500, peak_bankroll=1000))
    assert kelly_multiplier(profile(), 500, 1000) == (0.25, True)
    assert braked.stake[0] == pytest.approx(calm.stake[0] / 2)
    assert "brake" in braked.notes[0]


def test_tiny_stakes_are_dropped():
    out = recommend_stakes(bets((0.6, 2.0)), profile(), SizingContext(bankroll=8, peak_bankroll=8, min_stake=1.0))
    assert out.stake[0] == 0.0  # 10% of $8 = $0.80 < $1 minimum
    assert "minimum" in out.notes[0]


def test_empty_slate():
    out = recommend_stakes(bets(), profile(), SizingContext(bankroll=500, peak_bankroll=500))
    assert out.empty and "stake" in out


def test_profile_rejects_more_than_full_kelly():
    with pytest.raises(ValueError, match="full"):
        profile(kelly_fraction=1.5)


def test_simulation_is_deterministic_and_ordered():
    kwargs = dict(start=500, profile=profile(), market_probs=[0.5, 0.45], n_slates=10, n_paths=500)
    a = simulate_bankroll([0.58, 0.52], [2.0, 2.2], **kwargs)
    b = simulate_bankroll([0.58, 0.52], [2.0, 2.2], **kwargs)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.multiplier) == sorted(a.multiplier)
    # Bigger multipliers mean wider outcome ranges.
    spread = a.p90_final - a.p10_final
    assert spread.iloc[-1] > spread.iloc[0]


def test_no_real_edge_means_losses():
    sim = simulate_bankroll([0.6], [1.9], start=500, profile=profile(), market_probs=[0.5],
                            edge_realisation=0.0, n_slates=40, n_paths=2000)
    assert (sim.median_final < 500).all()


def test_suggestion_respects_drawdown_tolerance():
    sim = pd.DataFrame({"multiplier": [0.25, 0.5, 1.0], "median_final": [600, 700, 800],
                        "prob_drawdown_50": [0.01, 0.2, 0.6]})
    assert suggest_multiplier(sim, 0.25) == 0.5
    assert suggest_multiplier(sim, 0.7) == 1.0
    assert suggest_multiplier(sim, 0.001) is None
