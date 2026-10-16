"""Load project settings from ``config.toml``."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class RiskProfile:
    name: str
    kelly_fraction: float
    model_weight: float
    min_ev: float
    max_prob_gap: float
    min_odds: float
    max_odds: float
    max_stake_pct: float
    max_exposure_pct: float
    drawdown_brake_pct: float
    drawdown_brake_multiplier: float

    def __post_init__(self) -> None:
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError(
                f"{self.name}: kelly_fraction must be in (0, 1]; betting above full "
                "Kelly lowers long-run growth and raises risk of ruin"
            )
        if not 0 <= self.model_weight <= 1:
            raise ValueError(f"{self.name}: model_weight must be in [0, 1]")
        if not 0 < self.max_prob_gap <= 1:
            raise ValueError(f"{self.name}: max_prob_gap must be in (0, 1]")
        if not 0 < self.max_stake_pct <= self.max_exposure_pct <= 1:
            raise ValueError(
                f"{self.name}: need 0 < max_stake_pct <= max_exposure_pct <= 1"
            )
        if not 1 < self.min_odds < self.max_odds:
            raise ValueError(f"{self.name}: need 1 < min_odds < max_odds")

    def with_overrides(self, **changes: float) -> "RiskProfile":
        return replace(self, **changes)


@dataclass(frozen=True)
class EloSettings:
    base_k: float
    international_k_multiplier: float
    international_leagues: tuple[str, ...]
    new_team_games: int
    new_team_k_multiplier: float
    blue_side_bonus: float
    season_carryover: float
    mov_gold_weight: float
    mov_kill_weight: float
    default_rating: float
    region_priors: dict[str, float]
    series_concentration: float = 0.0

    def prior_for(self, league: str | None) -> float:
        if league is None:
            return self.default_rating
        return self.region_priors.get(league, self.default_rating)


@dataclass(frozen=True)
class Paths:
    matches: Path
    fixtures: Path
    odds: Path
    processed_dir: Path
    ledger: Path


@dataclass(frozen=True)
class Settings:
    starting_bankroll: float
    currency: str
    risk_profile: str
    stake_rounding: float
    min_stake: float
    timezone: str
    bookmakers: tuple[str, ...]
    devig_method: str
    risk_profiles: dict[str, RiskProfile]
    team_aliases: dict[str, str]
    elo: EloSettings
    paths: Paths

    @property
    def profile(self) -> RiskProfile:
        return self.risk_profiles[self.risk_profile]


def _resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "rb") as fh:
        raw = tomllib.load(fh)
    root = config_path.resolve().parent

    profiles = {
        name: RiskProfile(name=name, **values)
        for name, values in raw["risk_profiles"].items()
    }
    bankroll = raw["bankroll"]
    if bankroll["risk_profile"] not in profiles:
        raise ValueError(
            f"risk_profile '{bankroll['risk_profile']}' not in {sorted(profiles)}"
        )

    elo_raw = dict(raw["elo"])
    elo = EloSettings(
        region_priors={k: float(v) for k, v in elo_raw.pop("region_priors").items()},
        international_leagues=tuple(elo_raw.pop("international_leagues")),
        **elo_raw,
    )

    paths = Paths(**{k: _resolve(v, root) for k, v in raw["paths"].items()})
    market = raw["market"]
    if market["devig_method"] not in {"multiplicative", "power"}:
        raise ValueError("devig_method must be 'multiplicative' or 'power'")

    return Settings(
        starting_bankroll=float(bankroll["starting_bankroll"]),
        currency=bankroll["currency"],
        risk_profile=bankroll["risk_profile"],
        stake_rounding=float(bankroll["stake_rounding"]),
        min_stake=float(bankroll["min_stake"]),
        timezone=market["timezone"],
        bookmakers=tuple(market["bookmakers"]),
        devig_method=market["devig_method"],
        risk_profiles=profiles,
        team_aliases=dict(raw.get("team_aliases", {})),
        elo=elo,
        paths=paths,
    )
