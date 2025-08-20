from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Set, Iterable
from itertools import combinations

import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ParlayAnalyzer
# ---------------------------------------------------------------------
class ParlayAnalyzer:
    """
    Tools for computing correlations between same-game player stats and
    assembling candidate parlays.

    Key points:
      • track_stats: which box-score columns we consider for correlation.
      • All correlations are computed *up to a cutoff date* to avoid leakage.
      • Use analyze_same_game_pairs(...) to restrict to pairs in the same event.
    """

    def __init__(self, min_games: int = 10, min_corr: float = 0.30):
        self.min_games = int(min_games)
        self.min_corr = float(min_corr)
        # Standardized stat names that should exist in cleaned gamelog:
        self.track_stats: Set[str] = {"PTS", "REB", "AST", "PRA", "FG3M", "STOCKS", "TOV"}

    # -------------------- correlation primitives --------------------
    @staticmethod
    def _pearson(x: pd.Series, y: pd.Series) -> float:
        if len(x) < 2 or len(y) < 2:
            return 0.0
        try:
            r, _ = pearsonr(x, y)
            # handle nan edge cases
            return float(r) if pd.notna(r) else 0.0
        except Exception:  # pragma: no cover
            return 0.0

    def corr_on_or_before(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        stat: str,
        cutoff_date,  # datetime.date
    ) -> Tuple[float, int]:
        """
        Compute Pearson correlation between two players' time series for a given stat,
        using only games strictly before `cutoff_date`.
        Returns (corr, N_common_games).
        """
        a = (
            df1[df1["GAME_DATE"] < cutoff_date][["GAME_DATE", stat]]
            .rename(columns={stat: "x"})
            .dropna()
        )
        b = (
            df2[df2["GAME_DATE"] < cutoff_date][["GAME_DATE", stat]]
            .rename(columns={stat: "y"})
            .dropna()
        )
        merged = pd.merge(a, b, on="GAME_DATE", how="inner").dropna()
        if len(merged) < self.min_games:
            return 0.0, len(merged)
        return self._pearson(merged["x"], merged["y"]), len(merged)

    # -------------------- same-game analysis --------------------
    def analyze_same_game_pairs(
        self,
        game_players: Dict[str, Dict[str, Set[str]]],
        player_data: Dict[str, pd.DataFrame],
        cutoff_date,  # datetime.date
    ) -> pd.DataFrame:
        """
        Build a correlation table for pairs of players who are listed in the same event_id.

        Args:
          game_players: event_id -> { player_name -> set(stats_offered_that_snapshot) }
          player_data:  player_name -> cleaned gamelog DataFrame
          cutoff_date:  grade date; only historical games BEFORE this date are used for corr

        Returns a DataFrame with columns:
          ['event_id','player1','player2','stat','correlation','n_games']
        """
        rows: List[Dict] = []

        for event_id, roster in game_players.items():
            names = list(roster.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    p1, p2 = names[i], names[j]
                    # stats we can actually bet on for both players & we track historically
                    offered = (roster[p1] & roster[p2]) & self.track_stats
                    if not offered:
                        continue

                    df1 = player_data.get(p1)
                    df2 = player_data.get(p2)
                    if df1 is None or df2 is None or df1.empty or df2.empty:
                        continue

                    for stat in offered:
                        c, n = self.corr_on_or_before(df1, df2, stat, cutoff_date)
                        if n >= self.min_games and abs(c) >= self.min_corr:
                            rows.append(
                                {
                                    "event_id": event_id,
                                    "player1": p1,
                                    "player2": p2,
                                    "stat": stat,
                                    "correlation": float(c),
                                    "n_games": int(n),
                                }
                            )

        df = pd.DataFrame(rows)
        return df.sort_values("correlation", ascending=False).reset_index(drop=True)

    # -------------------- multi-leg assembly helpers --------------------
    @staticmethod
    def _corr_matrix(corr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a symmetric correlation matrix across players, ignoring stat differences.
        If duplicates exist, keeps the first (largest since corr_df is sorted).
        """
        if corr_df.empty:
            return pd.DataFrame()

        # pivot both ways then combine to ensure symmetry
        mat = corr_df.pivot_table(
            index="player1", columns="player2", values="correlation", aggfunc="first"
        ).fillna(0.0)
        # make symmetric by adding its transpose (prefers larger magnitude)
        mat = mat.add(mat.T, fill_value=0.0)
        return mat.fillna(0.0)

    @staticmethod
    def _val(cm: pd.DataFrame, r: str, c: str) -> float:
        if r in cm.index and c in cm.columns:
            v = cm.loc[r, c]
            return float(v) if pd.notna(v) else 0.0
        return 0.0

    def _group_correlation(self, players: List[str], cm: pd.DataFrame) -> float:
        """
        Average pairwise correlation among a set of players.
        """
        total, count = 0.0, 0
        for p1, p2 in combinations(players, 2):
            total += max(self._val(cm, p1, p2), self._val(cm, p2, p1))
            count += 1
        return total / count if count else 0.0

    def _common_stat(self, corr_df: pd.DataFrame, players: List[str]) -> str:
        """
        Choose the most frequent stat used among pairwise entries for this group.
        """
        stats: List[str] = []
        for p1, p2 in combinations(players, 2):
            mask = (
                ((corr_df["player1"] == p1) & (corr_df["player2"] == p2))
                | ((corr_df["player1"] == p2) & (corr_df["player2"] == p1))
            )
            stats.extend(corr_df.loc[mask, "stat"].tolist())
        if not stats:
            return "MIXED"
        return max(set(stats), key=stats.count)

    def _min_pair_corr(self, players: List[str], cm: pd.DataFrame) -> float:
        """
        Minimum pairwise correlation within the group.
        """
        mc = 1.0
        for p1, p2 in combinations(players, 2):
            cur = max(self._val(cm, p1, p2), self._val(cm, p2, p1))
            mc = min(mc, cur)
        return mc

    def find_top_parlays(self, corr_df: pd.DataFrame, n_legs: int = 3) -> List[Dict]:
        """
        Naive assembly: choose groups of `n_legs` distinct players with high average
        pairwise correlation (ignoring stats). Useful for a quick shortlist.

        Returns sorted list of:
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
            if g_corr >= self.min_corr:
                out.append(
                    {
                        "players": group,
                        "stat": self._common_stat(corr_df, list(group)),
                        "avg_correlation": float(g_corr),
                        "worst_pair": float(self._min_pair_corr(list(group), cm)),
                    }
                )
        return sorted(out, key=lambda x: -x["avg_correlation"])
