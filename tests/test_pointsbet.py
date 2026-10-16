import pytest

from lolworlds.pointsbet import PointsBetError, link_key, parse_event


def test_link_key():
    assert link_key("https://pointsbet.com.au/sports/e-sports/[LoL]-World-Championship/2930260") == "2930260"
    assert link_key("https://pointsbet.com.au/sports/e-sports/LOL/63455/") == "63455"
    assert link_key("2930260") == "2930260"
    with pytest.raises(PointsBetError):
        link_key("https://pointsbet.com.au/sports/e-sports")


def market(key, name, outcomes):
    return {"key": key, "eventName": name, "isOpenForBetting": True,
            "outcomes": [{"name": n, "price": p, "isOpenForBetting": True} for n, p in outcomes]}


EVENT = {
    "key": "2930260",
    "name": "MVK Esports v Karmine Corp (Bo5)",
    "homeTeam": "MVK Esports",
    "awayTeam": "Karmine Corp (Bo5)",
    "startsAt": "2026-10-16T18:00:00Z",
    "fixedOddsMarkets": [
        market("1", "Match Result", [("MVK Esports", 4.6), ("Karmine Corp", 1.1539)]),
        market("2", "Maps Handicap", [("MVK Esports +1.5", 2.55), ("Karmine Corp -1.5", 1.4445)]),
        market("3", "Total Maps", [("Over +3.5", 1.5406), ("Under +3.5", 2.3)]),
        market("4", "Best of 5 Correct Score", [("Karmine Corp 3-0", 2.55), ("MVK Esports 3-2", 9.0)]),
        market("5", "Map 1 Winner", [("MVK Esports", 2.5), ("Karmine Corp", 1.5)]),
        market("6", "Map 1 Race to 5 Kills", [("MVK Esports", 2.4), ("Karmine Corp", 1.5)]),
        market("7", "Map 1 Total Kills Odd/Even", [("Even", 1.83), ("Odd", 1.83)]),
        {**market("8", "Closed Market", [("MVK Esports", 3.0)]), "isOpenForBetting": False},
    ],
}


def test_parse_event_reads_every_market():
    m = parse_event(EVENT)
    assert (m.team_a, m.team_b, m.best_of, m.match_id) == ("MVK Esports", "Karmine Corp", 5, "PB-2930260")
    by_label = {(s.market, s.label): s for s in m.selections}
    assert len(m.selections) == 14  # 7 open markets x 2 outcomes; the closed market is skipped

    mr = by_label[("Match Result", "Karmine Corp")]
    assert (mr.kind, mr.side, mr.odds) == ("match_winner", "b", 1.1539)
    hc = by_label[("Maps Handicap", "MVK Esports +1.5")]
    assert (hc.kind, hc.side, hc.line) == ("map_handicap", "a", 1.5)
    tot = by_label[("Total Maps", "Under +3.5")]
    assert (tot.kind, tot.over, tot.line) == ("total_maps", False, 3.5)
    cs = by_label[("Best of 5 Correct Score", "MVK Esports 3-2")]
    assert (cs.kind, cs.side, cs.score) == ("correct_score", "a", (3, 2))
    mw = by_label[("Map 1 Winner", "Karmine Corp")]
    assert (mw.kind, mw.side, mw.map_no) == ("map_winner", "b", 1)
    kills = by_label[("Map 1 Race to 5 Kills", "MVK Esports")]
    assert kills.kind == "unpriced" and "kill" in kills.note
    assert by_label[("Map 1 Total Kills Odd/Even", "Even")].kind == "unpriced"


def test_best_of_from_correct_score_market_when_name_lacks_it():
    event = {**EVENT, "name": "MVK Esports v Karmine Corp", "awayTeam": "Karmine Corp"}
    assert parse_event(event).best_of == 5
