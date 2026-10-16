"""Pull every League of Legends market from PointsBet.

Reads the same JSON the pointsbet.com.au website loads. That API is unofficial
and undocumented, so it can change or be blocked without notice. With no link,
the LoL competitions are found automatically; a pasted link to a match or
competition page works too, e.g.
https://pointsbet.com.au/sports/e-sports/[LoL]-World-Championship/2930260

    python -m lolworlds.pointsbet                    # all LoL Worlds matches
    python -m lolworlds.pointsbet <pointsbet link>   # one match / competition
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request

import pandas as pd

from .markets import MatchMarkets, Selection

API = "https://api.au.pointsbet.com"
BOOKMAKER = "PointsBet"
HEADERS = {"User-Agent": "Mozilla/5.0", "Origin": "https://pointsbet.com.au", "Accept": "application/json"}
DEFAULT_COMPETITION = "world championship"


class PointsBetError(RuntimeError):
    pass


def _get(path: str) -> dict | None:
    req = urllib.request.Request(API + path, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise PointsBetError(f"PointsBet returned HTTP {err.code} for {path}") from err
    except OSError as err:
        raise PointsBetError(f"could not reach PointsBet ({err})") from err


def link_key(link: str) -> str:
    """The numeric id at the end of a PointsBet page link (or a bare id)."""
    match = re.search(r"(\d+)/?(?:[?#].*)?$", link.strip())
    if not match:
        raise PointsBetError(f"no match or competition id found in {link!r}")
    return match.group(1)


def lol_competitions() -> list[dict]:
    """PointsBet's current LoL competitions: [{key, name, events}]."""
    data = _get("/api/v2/sports/e-sports/competitions") or {}
    out = []
    for locale in data.get("locales", []):
        for comp in locale.get("competitions", []):
            if "[lol]" in comp.get("name", "").lower() or locale.get("name", "").upper() == "LOL":
                out.append({"key": comp["key"], "name": comp["name"], "events": comp.get("numberOfEvents", 0)})
    return out


def competition_event_keys(competition_key: str) -> list[str]:
    data = _get(f"/api/v2/competitions/{competition_key}/events/featured?includeLive=false&page=1") or {}
    return [e["key"] for e in data.get("events", [])]


# -- parsing ------------------------------------------------------------------
_BO = re.compile(r"\(\s*bo\s*(\d)\s*\)", re.I)
_LINE = re.compile(r"([+-]\d+(?:\.\d+)?)\s*$")
_SCORE = re.compile(r"(\d+)\s*-\s*(\d+)\s*$")
_MAP_WINNER = re.compile(r"^map (\d+) (?:winner|result|match result|to win)$", re.I)


def _side(label: str, team_a: str, team_b: str) -> str | None:
    low = label.lower()
    # Longest name first so "Team WE" doesn't match "Team WE Academy" the wrong way round.
    for name, side in sorted(((team_a, "a"), (team_b, "b")), key=lambda t: -len(t[0])):
        if low.startswith(name.lower()):
            return side
    return None


def parse_selection(market: dict, outcome: dict, team_a: str, team_b: str) -> Selection:
    name = (market.get("eventName") or market.get("name") or "").strip()
    label = (outcome.get("name") or "").strip()
    low = name.lower()
    base = dict(market=name, label=label, odds=float(outcome["price"]), market_id=str(market.get("key", name)))
    side = _side(label, team_a, team_b)

    if low in ("match result", "match winner", "head to head", "match betting", "winner"):
        return Selection(kind="match_winner", side=side, **base)
    if "handicap" in low and "map" in low and "kill" not in low:
        line = _LINE.search(label)
        if side and line:
            return Selection(kind="map_handicap", side=side, line=float(line.group(1)), **base)
    if low.startswith("total maps"):
        line = _LINE.search(label)
        if line and label.lower().startswith(("over", "under")):
            return Selection(kind="total_maps", over=label.lower().startswith("over"),
                             line=abs(float(line.group(1))), **base)
    if "correct score" in low:
        score = _SCORE.search(label)
        if side and score:
            return Selection(kind="correct_score", side=side, score=(int(score.group(1)), int(score.group(2))), **base)
    if m := _MAP_WINNER.match(name):
        if side:
            return Selection(kind="map_winner", side=side, map_no=int(m.group(1)), **base)

    needs_kills = any(w in low for w in ("kill", "blood", "tower", "dragon", "baron", "inhibitor"))
    return Selection(kind="unpriced", side=side,
                     note="needs per-game kill/objective data" if needs_kills else "market type not modelled", **base)


def parse_event(data: dict) -> MatchMarkets:
    team_a = re.sub(_BO, "", data.get("homeTeam") or "").strip()
    team_b = re.sub(_BO, "", data.get("awayTeam") or "").strip()
    bo = _BO.search(data.get("name") or "")
    best_of = int(bo.group(1)) if bo else None
    selections = []
    for market in data.get("fixedOddsMarkets") or []:
        if not market.get("isOpenForBetting", True):
            continue
        if best_of is None and (m := re.search(r"best of (\d)", market.get("eventName", ""), re.I)):
            best_of = int(m.group(1))
        for outcome in market.get("outcomes") or []:
            if outcome.get("price") and outcome.get("isOpenForBetting", True) and not outcome.get("isHidden"):
                selections.append(parse_selection(market, outcome, team_a, team_b))
    return MatchMarkets(
        match_id=f"PB-{data['key']}",
        start_time=pd.Timestamp(data["startsAt"]),
        team_a=team_a,
        team_b=team_b,
        best_of=best_of,
        bookmaker=BOOKMAKER,
        selections=selections,
        name=data.get("name", ""),
    )


def fetch_event(event_key: str) -> MatchMarkets | None:
    data = _get(f"/api/mes/v3/events/{event_key}")
    return parse_event(data) if data and data.get("key") else None


def fetch_link(link: str) -> list[MatchMarkets]:
    """A match link gives that match; a competition link gives all its matches."""
    key = link_key(link)
    event = fetch_event(key)
    if event is not None:
        return [event]
    keys = competition_event_keys(key)
    if not keys:
        raise PointsBetError(f"no open matches found for {link!r}")
    return [e for k in keys if (e := fetch_event(k)) is not None]


def fetch_worlds(name_contains: str = DEFAULT_COMPETITION) -> list[MatchMarkets]:
    """Every open match in PointsBet's LoL competitions whose name contains ``name_contains``."""
    comps = [c for c in lol_competitions() if name_contains in c["name"].lower()]
    if not comps:
        raise PointsBetError(f"PointsBet has no LoL competition matching {name_contains!r} right now")
    matches = []
    for comp in comps:
        matches += [e for k in competition_event_keys(comp["key"]) if (e := fetch_event(k)) is not None]
    return matches


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    matches = fetch_link(args[0]) if args else fetch_worlds()
    for m in matches:
        print(f"{m.name}  ({m.start_time:%Y-%m-%d %H:%M} UTC, Bo{m.best_of})")
        for s in m.selections:
            print(f"   {s.kind:13s} {s.market[:45]:45s} {s.label:28s} {s.odds:7.2f}")


if __name__ == "__main__":
    main()
