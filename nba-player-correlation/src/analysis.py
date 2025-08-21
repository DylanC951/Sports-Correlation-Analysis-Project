from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Tuple, Set
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

# --- Quiet Pearson constant-input warnings (SciPy version–agnostic) ---
_ConstWarn = None
try:
    # SciPy >= 1.11
    from scipy.stats import PearsonRConstantInputWarning as _ConstWarn  # type: ignore
except Exception:
    try:
        # Older SciPy
        from scipy.stats import ConstantInputWarning as _ConstWarn  # type: ignore
    except Exception:
        _ConstWarn = None

if _ConstWarn is not None:
    warnings.filterwarnings("ignore", category=_ConstWarn)


class ParlayAnalyzer:
    """
    High-confidence correlation utilities for same-game players.

    Key filters:
      • min_games:    minimum shared games in the (rolling) window
      • window_games: use the most recent N games BEFORE cutoff_date
      • min_corr:     minimum correlation magnitude to include
      • use_abs:      filter on absolute value of correlation (True = |r| >= min_corr)
      • max_p:        maximum Pearson p-value for significance

    APIs kept compatible with your backtest scripts:
      • analyze_same_game_pairs(...)    -> SAME stat on both players
      • analyze_cross_market_pairs(...) -> ANY stat1 (p1) vs ANY stat2 (p2)

    Returned columns (superset of old):
      Same-market: ['event_id','player1','player2','stat','correlation','p_value','n_games']
      Cross-mkt:   ['event_id','player1','stat1','player2','stat2','correlation','p_value','n_games']
    """

    def __init__(
        self,
        min_games: int = 15,
        min_corr: float = 0.50,
        max_p: float = 0.05,
        window_games: int = 20,
        use_abs: bool = True,
        stats_whitelist: Set[str] | None = None,
        min_std: float = 1e-8,  # treat near-constant series as invalid
    ):
        self.min_games = int(min_games)
        self.min_corr = float(min_corr)
        self.max_p = float(max_p)
        self.window_games = int(window_games)
        self.use_abs = bool(use_abs)
        self.min_std = float(min_std)
        # Default to core, stabler markets; you can expand to {"FG3M","STOCKS","TOV"} if desired.
        self.track_stats: Set[str] = stats_whitelist or {"PTS", "REB", "AST", "PRA"}

    # -------------------- helpers --------------------
    def _pearson(self, x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """
        Returns (r, p). Robust to degenerate inputs:
        - If either side length < 2, returns (0,1)
        - If either side is (near-)constant, returns (0,1) without calling pearsonr
        - If pearsonr returns NaN or errors, returns (0,1)
        """
        # Drop NaNs to compute variance safely
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()

        if len(x) < 2 or len(y) < 2:
            return 0.0, 1.0

        # Constant/near-constant guard (prevents warnings & meaningless r)
        if np.nanstd(x.values) < self.min_std or np.nanstd(y.values) < self.min_std:
            return 0.0, 1.0

        try:
            r, p = pearsonr(x.values, y.values)
            if not np.isfinite(r) or not np.isfinite(p):
                return 0.0, 1.0
            return float(r), float(p)
        except Exception:
            return 0.0, 1.0

    def _window_before(
        self, df: pd.DataFrame, stat: str, cutoff_date
    ) -> pd.DataFrame:
        """
        Most recent `window_games` rows before cutoff_date for a single stat.
        Returns columns ['GAME_DATE','val'].
        """
        if stat not in df.columns:
            raise KeyError(f"Stat column missing: {stat}")
        sub = df[df["GAME_DATE"] < cutoff_date][["GAME_DATE", stat]].dropna()
        if sub.empty:
            return sub.rename(columns={stat: "val"})
        # Take most recent N (sort descending, head N), then resort ascending by date for merge stability
        sub = sub.sort_values("GAME_DATE", ascending=False).head(self.window_games)
        sub = sub.sort_values("GAME_DATE").rename(columns={stat: "val"})
        return sub

    def _pass_filters(self, r: float, p: float, n: int) -> bool:
        if n < self.min_games:
            return False
        val = abs(r) if self.use_abs else r
        if val < self.min_corr:
            return False
        if p > self.max_p:
            return False
        return True

    # -------------------- SAME-MARKET same-game analysis --------------------
    def corr_on_or_before(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        stat: str,
        cutoff_date,  # datetime.date
    ) -> Tuple[float, float, int]:
        """
        Pearson for the SAME stat on both players, using a rolling window before cutoff_date.
        Returns (r, p, N_common_games).
        """
        a = self._window_before(df1, stat, cutoff_date).rename(columns={"val": "x"})
        b = self._window_before(df2, stat, cutoff_date).rename(columns={"val": "y"})
        m = pd.merge(a, b, on="GAME_DATE", how="inner").dropna()
        if m.empty:
            return 0.0, 1.0, 0
        r, p = self._pearson(m["x"], m["y"])
        return r, p, len(m)

    def analyze_same_game_pairs(
        self,
        game_players: Dict[str, Dict[str, Set[str]]],
        player_data: Dict[str, pd.DataFrame],
        cutoff_date,  # datetime.date
    ) -> pd.DataFrame:
        rows: List[Dict] = []

        for event_id, roster in game_players.items():
            names = list(roster.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    p1, p2 = names[i], names[j]
                    offered = (roster[p1] & roster[p2]) & self.track_stats
                    if not offered:
                        continue

                    df1 = player_data.get(p1)
                    df2 = player_data.get(p2)
                    if df1 is None or df2 is None or df1.empty or df2.empty:
                        continue

                    for stat in offered:
                        try:
                            r, p, n = self.corr_on_or_before(df1, df2, stat, cutoff_date)
                        except KeyError:
                            continue
                        if self._pass_filters(r, p, n):
                            rows.append(
                                {
                                    "event_id": event_id,
                                    "player1": p1,
                                    "player2": p2,
                                    "stat": stat,
                                    "correlation": float(r),
                                    "p_value": float(p),
                                    "n_games": int(n),
                                }
                            )

        cols = ["event_id", "player1", "player2", "stat", "correlation", "p_value", "n_games"]
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df
        # Sort by absolute correlation, then p-value ascending, then n descending
        return (
            df.assign(abs_corr=df["correlation"].abs())
              .sort_values(["abs_corr", "p_value", "n_games"], ascending=[False, True, False])
              .drop(columns="abs_corr")
              .reset_index(drop=True)
        )

    # -------------------- CROSS-MARKET same-game analysis --------------------
    def corr_cross_stats_before(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        stat1: str,
        stat2: str,
        cutoff_date,  # datetime.date
    ) -> Tuple[float, float, int]:
        """
        Pearson for DIFFERENT stats stat1 (p1) vs stat2 (p2), rolling window before cutoff_date.
        Returns (r, p, N_common_games).
        """
        a = self._window_before(df1, stat1, cutoff_date).rename(columns={"val": "x"})
        b = self._window_before(df2, stat2, cutoff_date).rename(columns={"val": "y"})
        m = pd.merge(a, b, on="GAME_DATE", how="inner").dropna()
        if m.empty:
            return 0.0, 1.0, 0
        r, p = self._pearson(m["x"], m["y"])
        return r, p, len(m)

    def analyze_cross_market_pairs(
        self,
        game_players: Dict[str, Dict[str, Set[str]]],
        player_data: Dict[str, pd.DataFrame],
        cutoff_date,  # datetime.date
    ) -> pd.DataFrame:
        rows: List[Dict] = []

        for event_id, roster in game_players.items():
            names = list(roster.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    p1, p2 = names[i], names[j]
                    s1s = (roster[p1] & self.track_stats)
                    s2s = (roster[p2] & self.track_stats)
                    if not s1s or not s2s:
                        continue

                    df1 = player_data.get(p1)
                    df2 = player_data.get(p2)
                    if df1 is None or df2 is None or df1.empty or df2.empty:
                        continue

                    for stat1, stat2 in product(s1s, s2s):
                        try:
                            r, p, n = self.corr_cross_stats_before(df1, df2, stat1, stat2, cutoff_date)
                        except KeyError:
                            continue
                        if self._pass_filters(r, p, n):
                            rows.append(
                                {
                                    "event_id": event_id,
                                    "player1": p1,
                                    "stat1": stat1,
                                    "player2": p2,
                                    "stat2": stat2,
                                    "correlation": float(r),
                                    "p_value": float(p),
                                    "n_games": int(n),
                                }
                            )

        cols = ["event_id", "player1", "stat1", "player2", "stat2", "correlation", "p_value", "n_games"]
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df
        return (
            df.assign(abs_corr=df["correlation"].abs())
              .sort_values(["abs_corr", "p_value", "n_games"], ascending=[False, True, False])
              .drop(columns="abs_corr")
              .reset_index(drop=True)
        )

    # -------------------- optional helpers for parlay assembly (unchanged API) --------------------
    @staticmethod
    def _corr_matrix(corr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a symmetric correlation matrix across players, ignoring which stat it came from.
        If duplicates exist, keeps the first (assuming corr_df is pre-sorted).
        """
        if corr_df.empty:
            return pd.DataFrame()

        mat = corr_df.pivot_table(
            index="player1", columns="player2", values="correlation", aggfunc="first"
        ).fillna(0.0)
        mat = mat.add(mat.T, fill_value=0.0)  # enforce symmetry
        return mat.fillna(0.0)

    @staticmethod
    def _val(cm: pd.DataFrame, r: str, c: str) -> float:
        if r in cm.index and c in cm.columns:
            v = cm.loc[r, c]
            return float(v) if pd.notna(v) else 0.0
        return 0.0

    def _group_correlation(self, players: List[str], cm: pd.DataFrame) -> float:
        """
        Average pairwise correlation among a set of players (stat-agnostic).
        """
        total, count = 0.0, 0
        for p1, p2 in combinations(players, 2):
            total += max(self._val(cm, p1, p2), self._val(cm, p2, p1))
            count += 1
        return total / count if count else 0.0

    def _common_stat(self, corr_df: pd.DataFrame, players: List[str]) -> str:
        """
        Choose the most frequent stat among pairwise entries for this group (best-effort).
        Works with same-market DF. With cross-market DF, this is less meaningful.
        """
        stats: List[str] = []
        for p1, p2 in combinations(players, 2):
            mask = (
                ((corr_df.get("player1") == p1) & (corr_df.get("player2") == p2))
                | ((corr_df.get("player1") == p2) & (corr_df.get("player2") == p1))
            )
            col = "stat" if "stat" in corr_df.columns else "stat1"
            stats.extend(corr_df.loc[mask, col].tolist())
        if not stats:
            return "MIXED"
        return max(set(stats), key=stats.count)

    def _min_pair_corr(self, players: List[str], cm: pd.DataFrame) -> float:
        """
        Minimum pairwise correlation within the group (stat-agnostic).
        """
        mc = 1.0
        for p1, p2 in combinations(players, 2):
            cur = max(self._val(cm, p1, p2), self._val(cm, p2, p1))
            mc = min(mc, cur)
        return mc

    def find_top_parlays(self, corr_df: pd.DataFrame, n_legs: int = 3) -> List[Dict]:
        """
        Naive assembly: choose groups of `n_legs` distinct players with high average
        pairwise correlation (ignoring which stat yielded it). Useful for shortlisting.

        Returns:
          {'players': tuple, 'stat': str, 'avg_correlation': float, 'worst_pair': float}
        """
        if corr_df.empty:
            return []

        cm = self._corr_matrix(corr_df)
        if cm.empty:
            return []

        players = list(set(corr_df["player1"]).union(set(corr_df["player2"])))
        out: List[Dict] = []
        for group in combinations(players, n_legs):
            g_corr = self._group_correlation(list(group), cm)
            val = abs(g_corr) if self.use_abs else g_corr
            if val >= self.min_corr:
                out.append(
                    {
                        "players": group,
                        "stat": self._common_stat(corr_df, list(group)),
                        "avg_correlation": float(g_corr),
                        "worst_pair": float(self._min_pair_corr(list(group), cm)),
                    }
                )
        return sorted(out, key=lambda x: -abs(x["avg_correlation"]))
