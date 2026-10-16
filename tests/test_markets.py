import numpy as np
import pandas as pd
import pytest

from lolworlds.bankroll import kelly_fraction
from lolworlds.elo import series_win_prob
from lolworlds.markets import (
    MatchMarkets,
    Selection,
    implied_map_prob,
    joint_kelly,
    plan_bets,
    price_match,
    returns_matrix,
    sequence_probs,
    sequences,
    settle,
    win_void_prob,
)


@pytest.mark.parametrize("best_of,count", [(1, 2), (3, 6), (5, 20)])
@pytest.mark.parametrize("kappa", [0.0, 13.0])
def test_sequences_cover_every_outcome(best_of, count, kappa):
    seqs = sequences(best_of)
    assert len(seqs) == count
    probs = sequence_probs(0.62, seqs, kappa)
    assert probs.sum() == pytest.approx(1.0)
    a_wins = sum(p for p, s in zip(probs, seqs) if sum(s) > len(s) - sum(s))
    assert a_wins == pytest.approx(series_win_prob(0.62, best_of, kappa))


def test_linked_maps_mean_more_sweeps_and_a_closer_series():
    sweep = (True, True, True)
    assert sequence_probs(0.6, [sweep], 13.0)[0] > sequence_probs(0.6, [sweep], 0.0)[0]
    # A favourite's series edge shrinks toward its map edge when maps are linked.
    assert 0.6 < series_win_prob(0.6, 5, 13.0) < series_win_prob(0.6, 5, 0.0)


def sel(kind, **kw):
    return Selection(market=kind, kind=kind, label=kind, odds=kw.pop("odds", 2.0), **kw)


def test_settlement_rules():
    a_31 = (True, False, True, True)
    assert settle(sel("match_winner", side="a"), a_31)
    assert not settle(sel("match_winner", side="b"), a_31)
    assert settle(sel("map_handicap", side="a", line=-1.5), a_31)        # 3-1 covers -1.5
    assert not settle(sel("map_handicap", side="a", line=-2.5), a_31)
    assert settle(sel("map_handicap", side="b", line=2.5), a_31)         # loser took a map
    assert settle(sel("total_maps", over=True, line=3.5), a_31)
    assert not settle(sel("total_maps", over=False, line=3.5), a_31)
    assert settle(sel("correct_score", side="a", score=(3, 1)), a_31)
    assert settle(sel("correct_score", side="b", score=(1, 3)), a_31)
    assert settle(sel("map_winner", side="b", map_no=2), a_31)
    assert settle(sel("map_winner", side="a", map_no=5), a_31) is None   # never played -> void


def test_map_five_winner_can_void():
    win, void = win_void_prob(sel("map_winner", side="a", map_no=5), 0.5, 5)
    assert void > 0 and win + void < 1


@pytest.mark.parametrize("kappa", [0.0, 13.0])
def test_implied_map_prob_inverts_series_prob(kappa):
    for p in (0.3, 0.55, 0.8):
        assert implied_map_prob(series_win_prob(p, 5, kappa), 5, kappa) == pytest.approx(p, abs=1e-6)


def match_with(prices: dict[str, float], best_of=5) -> MatchMarkets:
    """prices: 'a'/'b' match-result odds plus optional handicap/score selections."""
    sels = [Selection(market="Match Result", kind="match_winner", label=s, odds=o, side=s, market_id="mr")
            for s, o in prices.items() if s in "ab"]
    if "a-1.5" in prices:
        sels += [Selection(market="Maps Handicap", kind="map_handicap", label="A -1.5", odds=prices["a-1.5"],
                           side="a", line=-1.5, market_id="hc"),
                 Selection(market="Maps Handicap", kind="map_handicap", label="B +1.5", odds=prices["b+1.5"],
                           side="b", line=1.5, market_id="hc")]
    sels.append(Selection(market="Map 1 Race to 10 Kills", kind="unpriced", label="A", odds=1.8, side="a",
                          market_id="k", note="needs per-game kill/objective data"))
    return MatchMarkets("M1", pd.Timestamp("2026-10-20T10:00Z"), "A", "B", best_of, "PointsBet", sels)


def fair_odds_for(p_map, kappa=13.0, bo=5, margin=0.0):
    p = series_win_prob(p_map, bo, kappa)
    return 1 / (p * (1 + margin)), 1 / ((1 - p) * (1 + margin))


def test_fairly_priced_match_has_no_bets(aggressive):
    oa, ob = fair_odds_for(0.6, margin=0.04)
    table, info = price_match(match_with({"a": oa, "b": ob}), 0.6, aggressive, concentration=13.0)
    assert not table.qualifies.any()
    assert info["model_series"] == pytest.approx(series_win_prob(0.6, 5, 13.0))
    assert "kill" in table[table.kind == "unpriced"].reason.iloc[0]


def test_mispriced_side_is_flagged(aggressive):
    oa, ob = fair_odds_for(0.55)  # market thinks A is 55% per map
    table, _ = price_match(match_with({"a": oa, "b": ob}), 0.62, aggressive, concentration=13.0)
    assert table.set_index("selection").loc["a", "qualifies"]
    assert not table.set_index("selection").loc["b", "qualifies"]


def test_big_gap_blocks_the_whole_match(aggressive):
    oa, ob = fair_odds_for(0.35)
    table, info = price_match(match_with({"a": oa, "b": ob}), 0.75, aggressive, concentration=13.0)
    assert "gap" in info["blocked"] and not table.qualifies.any()


def test_linked_bets_on_one_match_are_sized_down():
    seqs = sequences(5)
    probs = sequence_probs(0.65, seqs, 13.0)
    bets = [sel("match_winner", side="a", odds=1.75), sel("map_handicap", side="a", line=-1.5, odds=2.6)]
    full = joint_kelly([(probs, returns_matrix(bets, seqs))])[0]
    singles = [kelly_fraction(win_void_prob(b, 0.65, 5, 13.0)[0], b.odds) for b in bets]
    assert full.sum() < sum(singles)


def test_independent_matches_combine():
    seqs = sequences(3)
    probs = sequence_probs(0.6, seqs)
    r = returns_matrix([sel("match_winner", side="a", odds=1.9)], seqs)
    one = joint_kelly([(probs, r)])[0]
    two = joint_kelly([(probs, r), (probs, r)])
    assert len(two) == 2 and two[0][0] == pytest.approx(two[1][0], abs=1e-6)
    assert two[0][0] <= one[0] + 1e-9


def test_plan_bets_end_to_end(aggressive):
    oa, ob = fair_odds_for(0.55)
    m = match_with({"a": oa, "b": ob})
    priced = {m.match_id: price_match(m, 0.62, aggressive, concentration=13.0)}
    bets, full = plan_bets([m], priced)
    assert bets.selection.tolist() == ["a"] and full[0] > 0
