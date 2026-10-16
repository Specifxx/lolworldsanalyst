"""Streamlit dashboard: the day's Worlds slate, +EV bets and Kelly stake sizes.

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from dataclasses import astuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from lolworlds.bankroll import (  # noqa: E402
    SizingContext,
    kelly_multiplier,
    recommend_stakes,
    simulate_bankroll,
    suggest_multiplier,
)
from lolworlds.config import RiskProfile, load_settings  # noqa: E402
from lolworlds.elo import calibration_table, fair_odds  # noqa: E402
from lolworlds.fetch import fetch  # noqa: E402
from lolworlds.ingest import build_ratings  # noqa: E402
from lolworlds import pointsbet  # noqa: E402
from lolworlds.ledger import Ledger, bankroll_history, bankroll_state  # noqa: E402
from lolworlds.markets import KIND_LABELS  # noqa: E402
from lolworlds.picks import analyse_markets  # noqa: E402
from lolworlds.value import ODDS_REQUIRED, find_value, load_odds  # noqa: E402

st.set_page_config(page_title="LoL Worlds Value Finder", layout="wide")

SERIES = {"light": "#2a78d6", "dark": "#3987e5"}
MUTED = "#898781"
pct = st.column_config.NumberColumn(format="%.1f%%")  # use with x100()
dollars = st.column_config.NumberColumn(format="dollar")
price = st.column_config.NumberColumn(format="%.2f")


def x100(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Copy of ``df`` with probability columns scaled to percentages for display."""
    out = df.copy()
    present = [c for c in cols if c in out]
    out[present] = out[present].astype(float) * 100
    return out


def money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def series_color() -> str:
    try:
        return SERIES["dark"] if st.context.theme.type == "dark" else SERIES["light"]
    except Exception:
        return SERIES["light"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), showlegend=False, hovermode="closest")
    return fig


# -- data -------------------------------------------------------------------
@st.cache_resource
def get_settings():
    return load_settings()


@st.cache_resource(show_spinner="Computing Elo ratings...")
def get_ratings(path: str, mtime: float):
    return build_ratings(get_settings(), path)


@st.cache_data(show_spinner="Simulating bankroll paths...")
def run_simulation(probs: tuple, odds: tuple, market: tuple, start: float, profile_fields: tuple,
                   slates: int, realisation: float) -> pd.DataFrame:
    return simulate_bankroll(
        probs, odds, start=start, profile=RiskProfile(*profile_fields), market_probs=market,
        edge_realisation=realisation, n_slates=slates,
    )


def download_results() -> None:
    with st.spinner("Downloading real results from the LoL Esports API..."):
        games, fixtures = fetch(verbose=False)
    settings.paths.matches.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(settings.paths.matches, index=False)
    fixtures.to_csv(settings.paths.fixtures, index=False)
    st.cache_resource.clear()


@st.cache_data(ttl=300, show_spinner="Loading PointsBet markets...")
def load_pointsbet(link: str) -> tuple[list, str]:
    """All open LoL Worlds matches on PointsBet (or just the linked one). Cached for 5 minutes."""
    try:
        return (pointsbet.fetch_link(link) if link else pointsbet.fetch_worlds()), ""
    except pointsbet.PointsBetError as err:
        return [], str(err)


settings = get_settings()
matches_path, odds_path = settings.paths.matches, settings.paths.odds
if not matches_path.exists():
    download_results()
ratings = get_ratings(str(matches_path), matches_path.stat().st_mtime)
odds_all = load_odds(odds_path, settings) if odds_path.exists() else pd.DataFrame(columns=ODDS_REQUIRED)
ledger = Ledger(settings.paths.ledger)
ledger_df = ledger.load()
state = bankroll_state(ledger_df, settings.starting_bankroll)
st.session_state.setdefault("log_version", 0)

# -- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("Bankroll")
    st.metric(
        "Ledger bankroll",
        money(state.bankroll),
        delta=f"{money(state.profit)} since start" if state.settled_bets else None,
        help=f"Starting bankroll {money(settings.starting_bankroll)} plus settled profit/loss.",
    )
    manual = st.toggle("Override bankroll", help="Size stakes off a different figure, e.g. your live Sportsbet + PointsBet balances.")
    bankroll = st.number_input(
        f"Bankroll for sizing ({settings.currency})", min_value=0.0, value=float(round(state.bankroll, 2)),
        step=10.0, disabled=not manual,
    ) if manual else state.bankroll
    peak = max(state.peak, bankroll)

    st.header("Risk")
    names = list(settings.risk_profiles)
    name = st.selectbox("Risk profile", names, index=names.index(settings.risk_profile), format_func=str.title)
    base = settings.risk_profiles[name]
    with st.expander("Fine-tune profile"):
        kf = st.slider("Kelly multiplier", 0.05, 1.0, base.kelly_fraction, 0.05, format="%.2fx", key=f"kf_{name}",
                       help="Share of the full-Kelly stake to bet. Above 1.0 lowers growth and raises ruin risk, so it is not offered.")
        mw = st.slider("Model weight", 0, 100, int(base.model_weight * 100), 5, format="%d%%", key=f"mw_{name}",
                       help="How much to trust the Elo model versus the bookmakers' fair price when estimating win probability.")
        min_ev = st.slider("Minimum EV", 0.0, 20.0, base.min_ev * 100, 0.5, format="%.1f%%", key=f"ev_{name}")
        gap = st.slider("Max model vs market gap", 5, 50, int(round(base.max_prob_gap * 100)), 1, format="%d%%", key=f"gap_{name}",
                        help="Skip bets where the model and the market disagree by more than this. Big gaps usually mean missing news.")
        max_stake = st.slider("Max stake per bet", 1, 50, int(round(base.max_stake_pct * 100)), 1, format="%d%%", key=f"ms_{name}")
        max_exp = st.slider("Max total exposure", 5, 100, int(round(base.max_exposure_pct * 100)), 5, format="%d%%", key=f"mx_{name}")
        odds_range = st.slider("Odds range", 1.01, 15.0, (base.min_odds, base.max_odds), 0.01, key=f"or_{name}")
    try:
        profile = base.with_overrides(
            kelly_fraction=kf, model_weight=mw / 100, min_ev=min_ev / 100, max_prob_gap=gap / 100,
            max_stake_pct=max_stake / 100, max_exposure_pct=max(max_exp, max_stake) / 100,
            min_odds=odds_range[0], max_odds=odds_range[1],
        )
    except ValueError as err:
        st.error(str(err))
        profile = base

    st.header("Data")
    st.caption(f"Results: {len(ratings.games):,} games to {ratings.games.date.max():%d %b %Y}  \n"
               f"Odds: `{rel(odds_path)}`  \n"
               f"Ledger: `{rel(ledger.path)}`")
    if st.button("Download latest results", width="stretch"):
        download_results()
        st.cache_data.clear()
        st.rerun()

# -- header -----------------------------------------------------------------
st.title("LoL Worlds Value Finder")
st.caption(f"Elo model vs {' / '.join(settings.bookmakers)} - {profile.name.title()} profile, "
           f"{profile.kelly_fraction:.2f}x Kelly - all times {settings.timezone}")

# -- today's slate ----------------------------------------------------------
today = pd.Timestamp.now(tz=settings.timezone).date()
days = sorted(odds_all.start_time.dt.date.unique()) if not odds_all.empty else []
open_bets = ledger_df[ledger_df.status == "open"]
board = sides = value_bets = stakes = pd.DataFrame()
day = None
if days:
    default_day = today if today in days else next((d for d in days if d > today), days[-1])
    day = st.selectbox(
        "Match day", days, index=days.index(default_day),
        format_func=lambda d: d.strftime("%a %d %b %Y") + ("  (today)" if d == today else ""),
    )
    day_odds = odds_all[odds_all.start_time.dt.date == day]
    board, sides, value_bets = find_value(day_odds, ratings.model, settings, profile)
    already = set(open_bets.match_id.astype(str))
    fresh = value_bets[~value_bets.match_id.astype(str).isin(already)] if not value_bets.empty else value_bets
    stakes = recommend_stakes(
        fresh, profile,
        SizingContext(bankroll=bankroll, peak_bankroll=peak, open_exposure=state.open_exposure,
                      stake_rounding=settings.stake_rounding, min_stake=settings.min_stake),
    )

tab_pb, tab_slate, tab_elo, tab_bank, tab_sim = st.tabs(
    ["PointsBet markets", "Manual odds", "Elo ratings", "Bankroll & bets", "Risk simulator"])

with tab_pb:
    c = st.columns([5, 1], vertical_alignment="bottom")
    link = c[0].text_input("PointsBet link (optional)", key="pb_link",
                           placeholder="Leave empty for every LoL Worlds match, or paste a match / competition link")
    if c[1].button("Refresh odds", width="stretch"):
        load_pointsbet.clear()
        st.rerun()
    pb_matches, pb_error = load_pointsbet(link.strip())
    pb_matches = [m for m in pb_matches if m.start_time >= pd.Timestamp.now(tz="UTC")]
    if pb_error:
        st.error(f"Couldn't load PointsBet: {pb_error}")
    elif not pb_matches:
        st.info("PointsBet has no open LoL Worlds matches right now.")
    else:
        pb_ctx = SizingContext(bankroll=bankroll, peak_bankroll=peak, open_exposure=state.open_exposure,
                               stake_rounding=settings.stake_rounding, min_stake=settings.min_stake)
        priced, pb_stakes = analyse_markets(pb_matches, ratings.model, settings, profile, pb_ctx,
                                            skip_match_ids=set(open_bets.match_id.astype(str)))
        to_bet = pb_stakes[pb_stakes.stake > 0] if not pb_stakes.empty else pb_stakes
        n_priced = sum(int(tbl.kind.isin(KIND_LABELS).sum()) for tbl, _ in priced.values())
        k = st.columns(4)
        k[0].metric("Matches", len(pb_matches), border=True)
        k[1].metric("Prices checked", n_priced, border=True,
                    help="Match result, maps handicap, total maps, correct score and map winner selections")
        k[2].metric("Bets to place", len(to_bet), border=True)
        k[3].metric("Total stake", money(float(to_bet.stake.sum()) if not to_bet.empty else 0.0), border=True,
                    delta=f"exp. profit {money(float(to_bet.expected_profit.sum()))}" if not to_bet.empty else None,
                    delta_color="off", delta_arrow="off")

        st.subheader("Bets to place")
        if to_bet.empty:
            st.info("No PointsBet price clears your profile's filters right now. Every market is listed below.")
        else:
            st.dataframe(
                x100(to_bet.assign(time=to_bet.start_time.dt.tz_convert(settings.timezone).dt.strftime("%a %H:%M")),
                     ["model_prob", "market_prob", "bet_prob", "ev"]),
                hide_index=True,
                column_order=["time", "match", "market", "selection", "odds", "model_prob", "market_prob", "ev",
                              "stake", "to_return", "notes"],
                column_config={
                    "time": "Start", "match": "Match", "market": "Market", "selection": "Bet",
                    "odds": st.column_config.NumberColumn("Odds", format="%.2f"),
                    "model_prob": st.column_config.NumberColumn("Model", format="%.1f%%"),
                    "market_prob": st.column_config.NumberColumn("Market", format="%.1f%%"),
                    "ev": st.column_config.NumberColumn("EV", format="%.1f%%"),
                    "stake": st.column_config.NumberColumn("Stake", format="dollar"),
                    "to_return": st.column_config.NumberColumn("Returns", format="dollar"),
                    "notes": "Notes",
                },
            )
            st.caption("Bets on the same match are sized together, because they win and lose together.")
            with st.container(border=True):
                st.markdown("**Log bets you placed** - tick them, adjust to the price/stake you got, then save.")
                plog = to_bet[["match_id", "start_time", "match", "market", "selection", "team_a", "team_b",
                               "odds", "stake", "bet_prob", "ev"]].copy()
                plog.insert(0, "placed", False)
                pedit = st.data_editor(
                    plog, hide_index=True, key=f"pblog_{st.session_state.log_version}",
                    column_order=["placed", "match", "market", "selection", "odds", "stake"],
                    disabled=["match", "market", "selection"],
                    column_config={
                        "placed": st.column_config.CheckboxColumn("Placed"),
                        "match": "Match", "market": "Market", "selection": "Bet",
                        "odds": st.column_config.NumberColumn("Odds taken", min_value=1.01, format="%.2f"),
                        "stake": st.column_config.NumberColumn("Stake", min_value=0.0, format="dollar"),
                    },
                )
                if st.button("Add ticked bets to ledger", type="primary", disabled=not pedit.placed.any(), key="pb_add"):
                    chosen = pedit[pedit.placed]
                    ledger.add_bets(pd.DataFrame({
                        "match_id": chosen.match_id, "start_time": chosen.start_time.astype(str),
                        "event": chosen.market, "selection": chosen.selection, "opponent": chosen.match,
                        "bookmaker": "PointsBet", "odds": chosen.odds, "stake": chosen.stake,
                        "bet_prob": chosen.bet_prob, "ev": chosen.ev,
                    }))
                    st.session_state.log_version += 1
                    st.toast(f"Logged {len(chosen)} bet(s).")
                    st.rerun()

        st.subheader("Every market")
        staked = {}
        if not to_bet.empty:
            staked = dict(zip(to_bet.match_id + "|" + to_bet.market + "|" + to_bet.selection, to_bet.stake))
        for m in sorted(pb_matches, key=lambda m: m.start_time):
            table, info = priced[m.match_id]
            start = m.start_time.tz_convert(settings.timezone)
            with st.expander(f"{m.team_a} vs {m.team_b} - {start:%a %d %b %H:%M} - Bo{m.best_of}", expanded=True):
                if info.get("model_series") is not None:
                    mkt = info["market_series"]
                    st.markdown(f"{m.team_a} to win: model **{info['model_series']:.0%}** (map {info['model_map']:.0%})"
                                + (f" vs market **{mkt:.0%}**" if mkt is not None else ""))
                if info.get("blocked"):
                    st.warning(f"No bets on this match: {info['blocked']}.")
                shown = table[table.kind.isin(KIND_LABELS)].copy()
                if not shown.empty and info.get("model_series") is not None:
                    shown["type"] = shown.kind.map(KIND_LABELS)
                    shown["stake"] = [staked.get(f"{m.match_id}|{r.market}|{r.selection}", 0.0) for r in shown.itertuples()]
                    st.dataframe(
                        x100(shown, ["model_prob", "market_prob", "ev"]), hide_index=True,
                        height=35 * (len(shown) + 1) + 3,
                        column_order=["type", "selection", "odds", "model_prob", "market_prob", "ev", "stake", "reason"],
                        column_config={
                            "type": "Market", "selection": "Selection",
                            "odds": st.column_config.NumberColumn("Odds", format="%.2f"),
                            "model_prob": st.column_config.NumberColumn("Model", format="%.1f%%"),
                            "market_prob": st.column_config.NumberColumn("Market", format="%.1f%%"),
                            "ev": st.column_config.NumberColumn("EV", format="%+.1f%%"),
                            "stake": st.column_config.NumberColumn("Stake", format="dollar"),
                            "reason": "Not bet because",
                        },
                    )
                n_unpriced = table[~table.kind.isin(KIND_LABELS)].market.nunique()
                if n_unpriced:
                    st.caption(f"{n_unpriced} more markets (kills, first blood, odd/even) aren't priced: "
                               "the model has no per-game kill data yet.")

with tab_slate:
    if not days:
        st.info("No odds loaded yet. Use **Add a match** below or create `data/odds/odds.csv`.")
    else:
        total_stake = float(stakes.stake.sum()) if not stakes.empty else 0.0
        exp_profit = float(stakes.expected_profit.sum()) if not stakes.empty else 0.0
        n_bets = int((stakes.stake > 0).sum()) if not stakes.empty else 0
        k = st.columns(5)
        k[0].metric("Bankroll", money(bankroll), border=True)
        k[1].metric("Open exposure", money(state.open_exposure), border=True,
                    help="Stakes on bets you've logged but not settled yet.")
        k[2].metric("Recommended stakes", money(total_stake), border=True,
                    delta=f"{total_stake / bankroll:.0%} of bankroll" if bankroll else None, delta_color="off", delta_arrow="off")
        k[3].metric("Value bets", f"{n_bets} / {len(board)}", border=True, help="Matches with a recommended bet / matches on this day")
        k[4].metric("Expected profit", money(exp_profit), border=True)

        mult, braked = kelly_multiplier(profile, bankroll, peak)
        if braked:
            st.warning(f"Drawdown brake on: bankroll is {1 - bankroll / peak:.0%} below its peak, so stakes are cut to "
                       f"{mult:.2f}x Kelly until it recovers.", icon=":material/trending_down:")

        st.subheader("Matches")
        show = pd.DataFrame({
            "Time": board.start_time.dt.strftime("%H:%M"),
            "Stage": (board.stage.astype(str) + " - Bo" + board.best_of.astype(str)).str.strip(" -"),
            "Team A": board.team_a,
            "Team B": board.team_b,
            "Model P(A)": board.model_prob_a * 100,
            "Market P(A)": board.market_prob_a * 100,
        })
        book_cols = {side: [f"{book} {side}" for book in settings.bookmakers] for side in "AB"}
        for side in "AB":
            for book in settings.bookmakers:
                show[f"{book} {side}"] = board.get(f"{book}_{side.lower()}")

        def best_price(row: pd.Series) -> list[str]:
            styles = pd.Series("", index=row.index)
            for cols in book_cols.values():
                prices = pd.to_numeric(row[cols], errors="coerce")
                if prices.notna().any():
                    styles[prices.idxmax()] = "background-color: rgba(42,120,214,0.18); font-weight: 600"
            return styles.tolist()

        styled = (
            show.style.apply(best_price, axis=1)
            .format({"Model P(A)": "{:.1f}%", "Market P(A)": "{:.1f}%",
                     **{c: "{:.2f}" for cols in book_cols.values() for c in cols}}, na_rep="-")
        )
        st.dataframe(styled, hide_index=True)
        st.caption("Highlighted = best available price for that team. Market P(A) is the bookmakers' price with the "
                   "margin removed, averaged across books.")

        st.subheader("Recommended bets")
        placed_matches = len(set(value_bets.match_id.astype(str)) & set(open_bets.match_id.astype(str))) if not value_bets.empty else 0
        if placed_matches:
            st.caption(f"{placed_matches} value match(es) already have open bets in your ledger and are left out.")
        if stakes.empty:
            st.info("No bets clear the profile's filters on this day. See *All sides* below for why.")
        else:
            rec = stakes.assign(time=stakes.start_time.dt.strftime("%H:%M"))
            st.dataframe(
                x100(rec, ["model_prob", "market_prob", "bet_prob", "ev", "full_kelly", "stake_pct"]), hide_index=True,
                column_order=["time", "selection", "opponent", "bookmaker", "odds", "model_prob", "market_prob",
                              "bet_prob", "ev", "full_kelly", "stake_pct", "stake", "to_return", "expected_profit", "notes"],
                column_config={
                    "time": "Time", "selection": "Bet", "opponent": "vs", "bookmaker": "Book",
                    "odds": st.column_config.NumberColumn("Odds", format="%.2f"),
                    "model_prob": st.column_config.NumberColumn("Model", format="%.1f%%"),
                    "market_prob": st.column_config.NumberColumn("Market", format="%.1f%%"),
                    "bet_prob": st.column_config.NumberColumn("Bet prob", format="%.1f%%",
                                                              help="Model blended with the market by the model weight"),
                    "ev": st.column_config.NumberColumn("EV", format="%.1f%%", help="Expected profit per $1 at the best price"),
                    "full_kelly": st.column_config.NumberColumn("Full Kelly", format="%.1f%%",
                                                                help="Joint full-Kelly fraction across today's bets"),
                    "stake_pct": st.column_config.NumberColumn("Stake %", format="%.1f%%"),
                    "stake": st.column_config.NumberColumn("Stake", format="dollar"),
                    "to_return": st.column_config.NumberColumn("Returns", format="dollar"),
                    "expected_profit": st.column_config.NumberColumn("Exp. profit", format="dollar"),
                    "notes": "Notes",
                },
            )

            with st.container(border=True):
                st.markdown("**Log bets you placed** - tick them, adjust the price or stake you actually got, then save.")
                log = stakes[stakes.stake > 0][["match_id", "start_time", "event", "stage", "selection", "opponent",
                                                "bookmaker", "odds", "stake", "bet_prob", "ev"]].copy()
                log.insert(0, "placed", False)
                edited = st.data_editor(
                    log, hide_index=True, key=f"log_{day}_{st.session_state.log_version}",
                    column_order=["placed", "selection", "opponent", "bookmaker", "odds", "stake"],
                    disabled=["selection", "opponent"],
                    column_config={
                        "placed": st.column_config.CheckboxColumn("Placed"),
                        "selection": "Bet", "opponent": "vs",
                        "bookmaker": st.column_config.SelectboxColumn("Book", options=list(settings.bookmakers), required=True),
                        "odds": st.column_config.NumberColumn("Odds taken", min_value=1.01, format="%.2f"),
                        "stake": st.column_config.NumberColumn("Stake", min_value=0.0, format="dollar"),
                    },
                )
                if st.button("Add ticked bets to ledger", type="primary", disabled=not edited.placed.any()):
                    chosen = edited[edited.placed].drop(columns="placed")
                    chosen["event"] = (chosen.event.astype(str) + " " + chosen.stage.astype(str)).str.strip()
                    chosen["start_time"] = chosen.start_time.astype(str)
                    ledger.add_bets(chosen.drop(columns="stage"))
                    st.session_state.log_version += 1
                    st.toast(f"Logged {len(chosen)} bet(s).")
                    st.rerun()

        with st.expander("All sides - model vs market and why bets were skipped"):
            st.dataframe(
                x100(sides, ["implied_prob", "market_prob", "model_prob", "bet_prob", "edge", "ev"]), hide_index=True,
                column_order=["selection", "opponent", "bookmaker", "odds", "implied_prob", "market_prob", "model_prob",
                              "bet_prob", "edge", "ev", "qualifies", "reason"],
                column_config={
                    "selection": "Team", "opponent": "vs", "bookmaker": "Best book", "odds": price,
                    "implied_prob": st.column_config.NumberColumn("Implied", format="%.1f%%"),
                    "market_prob": st.column_config.NumberColumn("Market", format="%.1f%%"),
                    "model_prob": st.column_config.NumberColumn("Model", format="%.1f%%"),
                    "bet_prob": st.column_config.NumberColumn("Bet prob", format="%.1f%%"),
                    "edge": st.column_config.NumberColumn("Edge", format="%.1f%%"),
                    "ev": st.column_config.NumberColumn("EV", format="%.1f%%"),
                    "qualifies": "Bet?", "reason": "Skipped because",
                },
            )

    with st.expander("Add a match", icon=":material/add:"):
        st.caption(f"Saves to `{rel(settings.paths.odds)}`.")
        teams = sorted(ratings.model.teams)
        with st.form("add_match", clear_on_submit=True):
            c = st.columns([1, 1, 1, 1])
            match_day = c[0].date_input("Date", value=today)
            match_time = c[1].time_input("Start (local)", value=pd.Timestamp("19:00").time())
            best_of = c[2].selectbox("Best of", [1, 3, 5], index=1)
            stage = c[3].text_input("Stage", value="")
            c = st.columns(2)
            team_a = c[0].selectbox("Team A", teams, index=None, accept_new_options=True, placeholder="Choose or type")
            team_b = c[1].selectbox("Team B", teams, index=None, accept_new_options=True, placeholder="Choose or type")
            prices = {}
            for book in settings.bookmakers:
                c = st.columns(2)
                prices[book] = (
                    c[0].number_input(f"{book} odds - Team A", min_value=0.0, value=0.0, step=0.01, format="%.2f"),
                    c[1].number_input(f"{book} odds - Team B", min_value=0.0, value=0.0, step=0.01, format="%.2f"),
                )
            if st.form_submit_button("Save match", type="primary"):
                quoted = {b: p for b, p in prices.items() if p[0] > 1 and p[1] > 1}
                if not team_a or not team_b or team_a == team_b:
                    st.error("Pick two different teams.")
                elif not quoted:
                    st.error("Enter both prices (above 1.00) for at least one bookmaker.")
                else:
                    start = pd.Timestamp.combine(match_day, match_time).tz_localize(settings.timezone)
                    match_id = f"{start:%Y%m%d-%H%M}-{team_a}-{team_b}".replace(" ", "")
                    rows = pd.DataFrame([
                        {"match_id": match_id, "start_time": start.isoformat(), "event": "Worlds", "stage": stage,
                         "best_of": best_of, "team_a": team_a, "team_b": team_b, "bookmaker": book,
                         "odds_a": a, "odds_b": b}
                        for book, (a, b) in quoted.items()
                    ])
                    path = settings.paths.odds
                    path.parent.mkdir(parents=True, exist_ok=True)
                    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=rows.columns)
                    pd.concat([existing, rows], ignore_index=True).to_csv(path, index=False)
                    st.toast(f"Saved {team_a} vs {team_b}.")
                    st.rerun()

# -- Elo ratings --------------------------------------------------------------
with tab_elo:
    cal = ratings.calibration
    k = st.columns(4)
    k[0].metric("Games rated", f"{len(ratings.games):,}", border=True)
    k[1].metric("Accuracy", f"{cal['accuracy']:.1%}", border=True, help="Pre-game favourite won (teams with 5+ prior games)")
    k[2].metric("Log loss", f"{cal['log_loss']:.4f}", border=True, help="Lower is better; 0.6931 = coin flip")
    k[3].metric("Brier score", f"{cal['brier']:.4f}", border=True, help="Lower is better; 0.25 = coin flip")

    st.subheader("Ratings")
    leagues = sorted(ratings.ratings.league.unique())
    pick = st.multiselect("Leagues", leagues, default=leagues)
    table = ratings.ratings[ratings.ratings.league.isin(pick)]
    st.dataframe(
        x100(table, ["win_rate"]), hide_index=True, height=420,
        column_order=[c for c in ["rank", "team", "league", "rating", "games", "wins", "losses", "win_rate", "form",
                                  "avg_gd15", "avg_kill_diff", "avg_game_min"] if table[c].notna().any()],
        column_config={
            "rank": "#", "team": "Team", "league": "League",
            "rating": st.column_config.NumberColumn("Elo", format="%.0f"),
            "games": "Games", "wins": "W", "losses": "L",
            "win_rate": st.column_config.NumberColumn("Win %", format="%.1f%%"),
            "form": st.column_config.TextColumn("Last 10", help="Record over the last 10 games"),
            "avg_gd15": st.column_config.NumberColumn("Gold diff @15", format="%+.0f"),
            "avg_kill_diff": st.column_config.NumberColumn("Kill diff", format="%+.1f"),
            "avg_game_min": st.column_config.NumberColumn("Avg length (min)", format="%.1f"),
        },
    )

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("Matchup calculator")
        teams = list(ratings.ratings.team)
        a = st.selectbox("Team A", teams, index=0, key="calc_a")
        b = st.selectbox("Team B", teams, index=1, key="calc_b")
        bo = st.segmented_control("Series", [1, 3, 5], default=3, format_func=lambda n: f"Bo{n}", required=True)
        if a == b:
            st.info("Pick two different teams.")
        else:
            p_map = ratings.model.predict_map(a, b)
            p = ratings.model.predict_series(a, b, bo)
            c = st.columns(2)
            c[0].metric(a, f"{p:.1%}", delta=f"fair odds {fair_odds(p):.2f}", delta_color="off", delta_arrow="off")
            c[1].metric(b, f"{1 - p:.1%}", delta=f"fair odds {fair_odds(1 - p):.2f}", delta_color="off", delta_arrow="off")
            st.caption(f"Single-map win probability for {a}: {p_map:.1%}. Any price above the fair odds is +EV "
                       "if you trust the model fully.")

    with right:
        st.subheader("Top 20 by Elo")
        top = ratings.ratings.head(20).iloc[::-1]
        fig = go.Figure(go.Scatter(
            x=top.rating, y=top.team, mode="markers",
            marker=dict(size=10, color=series_color()),
            customdata=top[["league", "games"]],
            hovertemplate="<b>%{y}</b><br>Elo %{x:.0f}<br>%{customdata[0]} - %{customdata[1]} games<extra></extra>",
        ))
        fig.update_xaxes(title="Elo rating")
        st.plotly_chart(style_fig(fig, 520), config={"displayModeBar": False})

    with st.expander("Calibration - predicted vs actual win rate"):
        st.dataframe(
            x100(calibration_table(ratings.history), ["predicted", "actual"]), hide_index=True,
            column_config={"bucket": "Predicted range", "games": "Games",
                           "predicted": st.column_config.NumberColumn("Avg predicted", format="%.1f%%"),
                           "actual": st.column_config.NumberColumn("Actual win rate", format="%.1f%%")},
        )

# -- bankroll & ledger --------------------------------------------------------
with tab_bank:
    k = st.columns(5)
    k[0].metric("Bankroll", money(state.bankroll), delta=money(state.profit) if state.settled_bets else None, border=True)
    k[1].metric("ROI", f"{state.roi:+.1%}", border=True, help="Profit divided by total settled stakes")
    k[2].metric("Record", f"{state.wins}-{state.settled_bets - state.wins}", border=True,
                delta=f"{state.win_rate:.0%} won" if state.settled_bets else None, delta_color="off", delta_arrow="off")
    k[3].metric("Drawdown from peak", f"{state.drawdown:.0%}", border=True,
                help=f"Peak {money(state.peak)}. The drawdown brake engages at {profile.drawdown_brake_pct:.0%}.")
    k[4].metric("Open bets", f"{state.open_bets} - {money(state.open_exposure)}", border=True)

    history = bankroll_history(ledger_df, settings.starting_bankroll)
    st.subheader("Bankroll over settled bets")
    if len(history) < 2:
        st.info("Settle some bets to see your bankroll curve.")
    else:
        fig = go.Figure(go.Scatter(
            x=history.bet_no, y=history.bankroll, mode="lines+markers",
            line=dict(width=2, color=series_color()), marker=dict(size=8),
            customdata=history[["label", "profit"]],
            hovertemplate="%{customdata[0]}<br>P/L %{customdata[1]:+,.2f}<br>Bankroll $%{y:,.2f}<extra></extra>",
        ))
        fig.add_hline(y=settings.starting_bankroll, line_dash="dot", line_color=MUTED,
                      annotation_text="Start", annotation_position="top left")
        fig.update_xaxes(title="Settled bet #", tick0=0, dtick=max(1, len(history) // 10))
        fig.update_yaxes(title="Bankroll (AUD)", tickprefix="$")
        st.plotly_chart(style_fig(fig), config={"displayModeBar": False})

    st.subheader("Open bets")
    if open_bets.empty:
        st.caption("No open bets.")
    for bet in open_bets.itertuples(index=False):
        c = st.columns([5, 1, 1, 1, 1], vertical_alignment="center")
        c[0].markdown(f"**{bet.selection}** vs {bet.opponent} - {bet.bookmaker} @ {float(bet.odds):.2f} - "
                      f"stake {money(float(bet.stake))} - {bet.event}")
        for col, result in zip(c[1:4], ("won", "lost", "void")):
            if col.button(result.title(), key=f"{result}_{bet.bet_id}", width="stretch"):
                ledger.settle(bet.bet_id, result)
                st.rerun()
        if c[4].button("Delete", key=f"del_{bet.bet_id}", width="stretch", type="tertiary"):
            ledger.remove(bet.bet_id)
            st.rerun()

    with st.expander("Full ledger"):
        st.dataframe(x100(ledger_df, ["bet_prob", "ev"]), hide_index=True, column_config={
            "odds": price, "stake": dollars, "payout": dollars, "bet_prob": pct, "ev": pct})
        st.download_button("Download ledger CSV", ledger_df.to_csv(index=False), file_name="ledger.csv",
                           mime="text/csv", disabled=ledger_df.empty)

# -- risk simulator -------------------------------------------------------------
with tab_sim:
    st.markdown("Replays the selected day's recommended bets over and over to show how different Kelly multipliers "
                "trade growth against drawdowns. Results assume the edges are as large as the *edge realised* "
                "setting says - real edges are usually smaller than models think.")
    sim_bets = stakes[stakes.stake > 0] if not stakes.empty else stakes
    if sim_bets.empty:
        st.info("No recommended bets on the selected day to simulate.")
    else:
        c = st.columns(3)
        slates = c[0].slider("Betting days to simulate", 5, 60, 30, 5)
        realisation = c[1].slider("Edge realised", 0, 100, 75, 5, format="%d%%",
                                  help="100% = the bet probabilities are exactly right; 0% = the market is right and you have no edge.")
        tolerance = c[2].slider("Max chance of a 50% drawdown", 5, 60, 25, 5, format="%d%%",
                                help="Your pain threshold. The suggestion is the best median outcome within it.")
        sim = run_simulation(
            tuple(sim_bets.bet_prob), tuple(sim_bets.odds), tuple(sim_bets.market_prob), float(bankroll),
            astuple(profile), slates, realisation / 100,
        )
        best = suggest_multiplier(sim, tolerance / 100)
        current = profile.kelly_fraction
        if best is None:
            st.warning("Every multiplier breaches your drawdown tolerance for this slate - consider a smaller profile.")
        else:
            verdict = "matches" if abs(best - current) < 1e-9 else ("is above" if current > best else "is below")
            st.success(f"Suggested Kelly multiplier: **{best:.2f}x**. Your current setting ({current:.2f}x) {verdict} it.")

        color = series_color()
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Bankroll after {slates} days** - median with 10th-90th percentile band")
            fig = go.Figure([
                go.Scatter(x=sim.multiplier, y=sim.p90_final, mode="lines", line=dict(width=0), hoverinfo="skip"),
                go.Scatter(x=sim.multiplier, y=sim.p10_final, mode="lines", line=dict(width=0), fill="tonexty",
                           fillcolor="rgba(57,135,229,0.18)", hoverinfo="skip"),
                go.Scatter(x=sim.multiplier, y=sim.median_final, mode="lines+markers", line=dict(width=2, color=color),
                           marker=dict(size=8), customdata=sim[["p10_final", "p90_final"]],
                           hovertemplate="%{x:.2f}x Kelly<br>Median $%{y:,.0f}<br>10th-90th $%{customdata[0]:,.0f} - $%{customdata[1]:,.0f}<extra></extra>"),
            ])
            fig.add_hline(y=bankroll, line_dash="dot", line_color=MUTED, annotation_text="Today",
                          annotation_position="top left")
            fig.add_vline(x=current, line_dash="dash", line_color=MUTED)
            fig.update_xaxes(title="Kelly multiplier")
            fig.update_yaxes(title="Bankroll (AUD)", tickprefix="$", rangemode="tozero")
            st.plotly_chart(style_fig(fig), config={"displayModeBar": False})
        with right:
            st.markdown("**Chance of a 50% drawdown at some point**")
            fig = go.Figure(go.Scatter(
                x=sim.multiplier, y=sim.prob_drawdown_50, mode="lines+markers",
                line=dict(width=2, color=color), marker=dict(size=8),
                hovertemplate="%{x:.2f}x Kelly<br>%{y:.1%} chance<extra></extra>",
            ))
            fig.add_hline(y=tolerance / 100, line_dash="dot", line_color=MUTED, annotation_text="Your tolerance",
                          annotation_position="top left")
            fig.add_vline(x=current, line_dash="dash", line_color=MUTED)
            fig.update_xaxes(title="Kelly multiplier")
            fig.update_yaxes(title="Probability", tickformat=".0%", rangemode="tozero")
            st.plotly_chart(style_fig(fig), config={"displayModeBar": False})
        st.caption("Dashed vertical line = your current multiplier. Caps from the profile are applied; "
                   "rounding, minimum stakes and the drawdown brake are not.")
        st.dataframe(x100(sim, ["prob_profit", "prob_halved", "prob_drawdown_50", "median_max_drawdown"]),
                     hide_index=True, column_config={
            "multiplier": st.column_config.NumberColumn("Kelly x", format="%.2f"),
            "median_final": st.column_config.NumberColumn("Median", format="dollar"),
            "p10_final": st.column_config.NumberColumn("10th pct", format="dollar"),
            "p90_final": st.column_config.NumberColumn("90th pct", format="dollar"),
            "prob_profit": st.column_config.NumberColumn("P(profit)", format="%.1f%%"),
            "prob_halved": st.column_config.NumberColumn("P(end below half)", format="%.1f%%"),
            "prob_drawdown_50": st.column_config.NumberColumn("P(50% drawdown)", format="%.1f%%"),
            "median_max_drawdown": st.column_config.NumberColumn("Median max drawdown", format="%.1f%%"),
        })

st.divider()
st.caption("Model output is an estimate, not a guarantee - only stake what you can afford to lose. "
           "Set deposit limits in your Sportsbet and PointsBet accounts. Free, confidential support: "
           "Gambling Help Online, 1800 858 858 (gamblinghelponline.org.au).")
