import pandas as pd
import pytest

from lolworlds.ledger import Ledger, bankroll_history, bankroll_state


def bet(selection, odds, stake, match_id="M"):
    return {"match_id": match_id, "selection": selection, "opponent": "X", "bookmaker": "Sportsbet",
            "odds": odds, "stake": stake}


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.csv")


def test_empty_ledger_is_the_starting_bankroll(ledger):
    state = bankroll_state(ledger.load(), 500)
    assert (state.bankroll, state.open_exposure, state.settled_bets) == (500, 0, 0)


def test_settling_bets_updates_bankroll(ledger):
    ids = ledger.add_bets(pd.DataFrame([bet("A", 2.0, 50), bet("B", 3.0, 20), bet("C", 1.5, 40), bet("D", 1.8, 10)]))
    ledger.settle(ids[0], "won")   # +50
    ledger.settle(ids[1], "lost")  # -20
    ledger.settle(ids[2], "void")  # 0, stake back
    state = bankroll_state(ledger.load(), 500)
    assert state.bankroll == pytest.approx(530)
    assert state.profit == pytest.approx(30)
    assert state.open_bets == 1 and state.open_exposure == 10
    assert state.available == pytest.approx(520)
    assert (state.wins, state.settled_bets) == (1, 2)  # voids don't count
    assert state.roi == pytest.approx(30 / 70)


def test_history_and_drawdown(ledger):
    ids = ledger.add_bets(pd.DataFrame([bet("A", 2.0, 100), bet("B", 2.0, 100)]))
    ledger.settle(ids[0], "won")
    ledger.settle(ids[1], "lost")
    df = ledger.load()
    history = bankroll_history(df, 500)
    assert history.bankroll.tolist() == [500, 600, 500]
    state = bankroll_state(df, 500)
    assert state.peak == 600
    assert state.drawdown == pytest.approx(1 / 6)


def test_remove_and_bad_input(ledger):
    (bet_id,) = ledger.add_bets(pd.DataFrame([bet("A", 2.0, 10)]))
    with pytest.raises(ValueError):
        ledger.settle(bet_id, "maybe")
    with pytest.raises(KeyError):
        ledger.settle("nope", "won")
    ledger.remove(bet_id)
    assert ledger.load().empty
