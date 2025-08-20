import os
import logging
from typing import Dict, Set, Optional, Tuple, List, Iterable
from collections import defaultdict

import requests
import pandas as pd
from tqdm import tqdm

# nba_api
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

# Optional fuzzy matching (falls back to difflib if not installed)
try:
    from fuzzywuzzy import fuzz  # type: ignore
    _HAS_FUZZY = True
except Exception:  # pragma: no cover
    from difflib import SequenceMatcher
    _HAS_FUZZY = False

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Odds API fetcher (current + historical)
# ---------------------------------------------------------------------
class LiveLineFetcher:
    """
    A small helper around The Odds API endpoints for NBA player props.

    Supports:
      • Live odds snapshot (for quick availability checks) via /v4/sports/.../odds
      • Historical event list & historical event odds (DFS platforms) via /v4/historical/...

    Notes:
      • Historical access requires a paid plan on The Odds API.
      • This class does not implement local caching. Consider wrapping calls
        so you don't re-fetch the same JSON repeatedly during backtests.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            logger.warning("ODDS_API_KEY not set; requests will fail without an API key.")
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "basketball_nba"

    # -------------------- helpers --------------------
    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize a player name as it often appears in DFS markets.
        """
        return (
            name.replace(".", "")
            .replace("-", " ")
            .replace(" Jr", " Jr.")
            .replace(" Ii", " II")
            .replace(" Iii", " III")
            .strip()
            .title()
        )

    # -------------------- live odds --------------------
    def get_active_lines(self) -> Dict[str, Set[str]]:
        """
        Fetch a simple "who has lines right now" snapshot (US DFS, PP & UD).
        Returns:
          dict[player_name] -> set({STAT_ABBREV, ...})
        where STAT_ABBREV ∈ {'PTS','REB','AST','PRA','FG3M','STOCKS','TOV'} (when present).

        This is *not* historical—it's a live snapshot.
        """
        params = {
            "apiKey": self.api_key,
            "regions": "us_dfs",
            "bookmakers": "prizepicks,underdog",
            "markets": ",".join([
                "player_points",
                "player_rebounds",
                "player_assists",
                "player_points_rebounds_assists",
                "player_threes",
                "player_blocks_steals",
                "player_turnovers",
            ]),
            "oddsFormat": "decimal",
        }

        try:
            url = f"{self.base_url}/sports/{self.sport}/odds"
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            return self._process_odds_response(response.json())
        except Exception as e:  # pragma: no cover
            logger.error(f"OddsAPI Error: {str(e)}")
            return {}

    # -------------------- historical --------------------
    def historical_events(self, iso_ts: str) -> List[dict]:
        """
        List NBA events at a snapshot timestamp (ISO8601 '...Z').
        Endpoint: /v4/historical/sports/{sport}/events?date=<iso_ts>
        Returns a list of event dicts (each contains id, home_team, away_team, commence_time, etc.)
        """
        url = f"{self.base_url}/historical/sports/{self.sport}/events"
        params = {"apiKey": self.api_key, "date": iso_ts}
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        # Some clients return {'data': [...]}; support both
        return data.get("data", data)

    def historical_event_props(
        self,
        event_id: str,
        iso_ts: str,
        markets: str,
        bookmakers: str = "prizepicks,underdog",
    ) -> dict:
        """
        For a single event (game) at a given snapshot, return the DFS player prop markets.

        Endpoint:
          /v4/historical/sports/{sport}/events/{event_id}/odds?date=<iso_ts>&regions=us_dfs&bookmakers=...&markets=...

        Returns raw JSON blob (possibly wrapped under 'data').
        """
        url = f"{self.base_url}/historical/sports/{self.sport}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "date": iso_ts,
            "regions": "us_dfs",
            "bookmakers": bookmakers,
            "markets": markets,
            "oddsFormat": "decimal",
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("data", data)

    # -------------------- internal --------------------
    def _process_odds_response(self, data: list) -> Dict[str, Set[str]]:
        """
        Convert a live odds JSON into {player_name -> set(stats_abbrev)}.
        """
        # Map Odds API market keys -> short stat names we use
        stat_map = {
            "player_points": "PTS",
            "player_rebounds": "REB",
            "player_assists": "AST",
            "player_points_rebounds_assists": "PRA",
            "player_threes": "FG3M",
            "player_blocks_steals": "STOCKS",
            "player_turnovers": "TOV",
        }

        active_players: Dict[str, Set[str]] = defaultdict(set)
        for game in data or []:
            for bookmaker in game.get("bookmakers", []):
                # Only parse PP/UD for DFS context; ignore others if present
                if bookmaker.get("key") not in {"prizepicks", "underdog"}:
                    continue
                for market in bookmaker.get("markets", []):
                    mkey = market.get("key")
                    if mkey not in stat_map:
                        continue
                    stat = stat_map[mkey]
                    for outcome in market.get("outcomes", []):
                        name = self.normalize_name(outcome.get("description", ""))
                        if not name:
                            continue
                        active_players[name].add(stat)

        return dict(active_players)


# ---------------------------------------------------------------------
# NBA Player ID resolution & game log loading
# ---------------------------------------------------------------------
_all_players = nba_players.get_players()
_NAME_TO_ID = {p["full_name"].lower(): p["id"] for p in _all_players}


def _fuzzy_ratio(a: str, b: str) -> float:
    if _HAS_FUZZY:
        try:
            return float(fuzz.ratio(a, b))
        except Exception:  # pragma: no cover
            pass
    # fallback
    return 100.0 * SequenceMatcher(None, a, b).ratio()


def find_player_id(name: str) -> Optional[int]:
    """
    Resolve a (possibly noisy) name -> NBA player_id.

    Strategy:
      1) exact lowercase match against nba_api roster
      2) fuzzy best-match above a reasonable cutoff
    """
    key = name.strip().lower()
    if key in _NAME_TO_ID:
        return _NAME_TO_ID[key]

    # fuzzy
    best_id, best_score = None, -1.0
    for nm, pid in _NAME_TO_ID.items():
        s = _fuzzy_ratio(key, nm)
        if s > best_score:
            best_id, best_score = pid, s

    # Cutoff can be tuned; 70 is forgiving for punctuation/suffixes
    return best_id if best_score >= 70.0 else None


def fetch_player_data(
    player_id: int,
    season: str = "2024-25",
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """
    Pull a player's game log for the given season.

    Returns a raw nba_api DataFrame.
    """
    gl = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
    )
    dfs = gl.get_data_frames()
    return dfs[0] if dfs else pd.DataFrame()


def clean_gamelog(df: pd.DataFrame, min_minutes: int = 10) -> pd.DataFrame:
    """
    Coerce numeric types, derive PRA/STOCKS, keep useful fields,
    and filter out very low-minute games.

    Returned columns:
      ['GAME_DATE','PTS','REB','AST','PRA','FG3M','STOCKS','TOV','MIN']
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["GAME_DATE", "PTS", "REB", "AST", "PRA", "FG3M", "STOCKS", "TOV", "MIN"]
        )

    df = df.copy()

    # Ensure expected columns exist and are numeric
    for col in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN", "FG3M"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Dates as date objects
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    else:  # pragma: no cover
        df["GAME_DATE"] = pd.NaT

    # Derived stats
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["STOCKS"] = df["STL"] + df["BLK"]

    # Filter by minutes to reduce noise
    df = df[df["MIN"] > min_minutes]

    return df[["GAME_DATE", "PTS", "REB", "AST", "PRA", "FG3M", "STOCKS", "TOV", "MIN"]].reset_index(drop=True)


def load_active_player_data(
    active_lines: Dict[str, Set[str]],
    season: str = "2024-25",
    min_minutes: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Pull and clean game logs for the set of names in `active_lines`.

    Returns:
      dict[player_name] -> cleaned DataFrame
    """
    player_data: Dict[str, pd.DataFrame] = {}

    for name in tqdm(active_lines.keys(), desc="Loading player data"):
        pid = find_player_id(name)
        if not pid:
            logger.warning(f"Could not resolve player: {name}")
            continue
        try:
            raw = fetch_player_data(pid, season=season)
            cleaned = clean_gamelog(raw, min_minutes=min_minutes)
            if not cleaned.empty:
                player_data[name] = cleaned
        except Exception as e:  # pragma: no cover
            logger.error(f"Error loading {name} ({pid}): {e}")

    return player_data


def load_players_by_names(
    names: Iterable[str],
    season: str = "2024-25",
    min_minutes: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience for loading specific names (e.g., from a historical snapshot roster).
    """
    player_data: Dict[str, pd.DataFrame] = {}
    for name in tqdm(list(names), desc="Loading player data"):
        pid = find_player_id(name)
        if not pid:
            logger.warning(f"Could not resolve player: {name}")
            continue
        try:
            raw = fetch_player_data(pid, season=season)
            cleaned = clean_gamelog(raw, min_minutes=min_minutes)
            if not cleaned.empty:
                player_data[name] = cleaned
        except Exception as e:  # pragma: no cover
            logger.error(f"Error loading {name} ({pid}): {e}")

    return player_data
