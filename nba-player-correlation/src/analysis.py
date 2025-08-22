from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import rankdata

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


def _to_num(s: pd.Series) -> pd.Series:
    """Coerce to numeric, keeping index/shape predictable."""
    return pd.to_numeric(pd.Series(s), errors="coerce")


def _winsorize(x: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    """Clip extremes to reduce outlier influence (simple winsorization)."""
    if x.size == 0 or not np.isfinite(x).any():
        return x
    lo = np.nanpercentile(x, q_low * 100.0)
    hi = np.nanpercentile(x, q_high * 100.0)
    return np.clip(x, lo, hi)


def _weighted_corr(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    min_std: float = 1e-8,
) -> float:
    """Weighted Pearson correlation."""
    m_w = np.sum(w)
    if m_w <= 0:
        return 0.0
    x_bar = np.sum(w * x) / m_w
    y_bar = np.sum(w * y) / m_w
    dx = x - x_bar
    dy = y - y_bar
    cov = np.sum(w * dx * dy) / m_w
    vx = np.sum(w * dx * dx) / m_w
    vy = np.sum(w * dy * dy) / m_w
    if vx < min_std or vy < min_std:
        return 0.0
    r = cov / np.sqrt(vx * vy)
    # Guard against numeric drift
    return float(np.clip(r, -1.0, 1.0))


def _effective_n(w: np.ndarray) -> float:
    """Kish effective sample size for weights."""
    s1 = np.sum(w)
    s2 = np.sum(w * w)
    if s2 <= 0:
        return 0.0
    return float((s1 * s1) / s2)


def _pearson_pvalue_from_r(r: float, n_eff: float) -> float:
    """Approximate two-sided p-value from r using effective N."""
    if n_eff <= 2 or not np.isfinite(r):
        return 1.0
    df = max(n_eff - 2.0, 1.0)
    denom = max(1e-12, 1.0 - r * r)
    t = r * np.sqrt(df / denom)
    # two-sided
    p = 2.0 * (1.0 - student_t.cdf(abs(t), df))
    return float(np.clip(p, 0.0, 1.0))


def _shrink_r_fisher(r: float, n_eff: float, tau: float = 10.0) -> float:
    """
    Fisher z-shrinkage: scales z toward 0 based on n_eff and tau (prior strength).
    """
    r = float(np.clip(r, -0.999999, 0.999999))
    if n_eff <= 3:
        return 0.0
    z = np.arctanh(r)
    factor = max(0.0, (n_eff - 3.0) / (n_eff - 3.0 + tau))
    z_shrunk = z * factor
    return float(np.tanh(z_shrunk))


class ParlayAnalyzer:
    """
    High-confidence correlation utilities for same-game players.

    Upgrades over the basic version:
      • Recency-weighted correlations (exponential decay)
      • Fisher z-shrinkage by effective sample size
      • Optional partial correlation (control for MIN)
      • Rank-correlation blend for robustness
      • Winsorization to tame outliers

    Public API matches your scripts:
      • analyze_same_game_pairs(...)
      • analyze_cross_market_pairs(...)

    Returned columns (superset):
      Same-market: ['event_id','player1','player2','stat','correlation','p_value','n_games', ... extras]
      Cross-mkt:   ['event_id','player1','stat1','player2','stat2','correlation','p_value','n_games', ... extras]
    """

    def __init__(
        self,
        min_games: int = 15,
        min_corr: float = 0.50,
        max_p: float = 0.05,
        window_games: int = 20,
        use_abs: bool = True,
        stats_whitelist: Optional[Set[str]] = None,
        min_std: float = 1e-8,
        # --- New knobs ---
        half_life_weeks: float = 4.0,     # recency half-life (weeks)
        shrink_tau: float = 10.0,         # Fisher shrink prior strength
        blend_rank: float = 0.25,         # 0..1 : weight of rank corr in final
        winsor_low: float = 0.02,         # 2% / 98% winsorization by default
        winsor_high: float = 0.98,
        use_partial_minutes: bool = True, # control for MIN if available
    ):
        self.min_games = int(min_games)
        self.min_corr = float(min_corr)
        self.max_p = float(max_p)
        self.window_games = int(window_games)
        self.use_abs = bool(use_abs)
        self.min_std = float(min_std)

        self.half_life_weeks = float(half_life_weeks)
        self.shrink_tau = float(shrink_tau)
        self.blend_rank = float(np.clip(blend_rank, 0.0, 1.0))
        self.winsor_low = float(np.clip(winsor_low, 0.0, 0.5))
        self.winsor_high = float(np.clip(winsor_high, 0.5, 1.0))
        self.use_partial_minutes = bool(use_partial_minutes)

        # Default to core, stabler markets; expand if you wish.
        self.track_stats: Set[str] = stats_whitelist or {"PTS", "REB", "AST", "PRA"}

        # Precompute decay constant (per day)
        # half-life (weeks) -> lambda per day
        # weight = exp(-lambda_days * age_days)
        if self.half_life_weeks <= 0:
            self._lambda_per_day = 0.0
        else:
            self._lambda_per_day = np.log(2.0) / (7.0 * self.half_life_weeks)

    # -------------------- helpers --------------------
    def _window_before(
        self, df: pd.DataFrame, stat: str, cutoff_date
    ) -> pd.DataFrame:
        """
        Most recent `window_games` rows before cutoff_date for a single stat.
        Returns columns ['GAME_DATE','val','MIN'(optional)].
        """
        if stat not in df.columns:
            raise KeyError(f"Stat column missing: {stat}")
        cols = ["GAME_DATE", stat]
        if "MIN" in df.columns:  # minutes column (from your clean_gamelog)
            cols.append("MIN")
        sub = df[df["GAME_DATE"] < cutoff_date][cols].dropna(subset=[stat])
        if sub.empty:
            return sub.rename(columns={stat: "val"})
        # Take most recent N (sort desc, head N), then resort asc by date
        sub = sub.sort_values("GAME_DATE", ascending=False).head(self.window_games)
        sub = sub.sort_values("GAME_DATE").rename(columns={stat: "val"})
        return sub

    def _decay_weights(self, dates: pd.Series, cutoff_date) -> np.ndarray:
        """
        Exponential recency weights by age (in days) relative to cutoff_date.
        """
        if dates.empty or self._lambda_per_day == 0.0:
            return np.ones(len(dates), dtype=float)
        ages = (pd.to_datetime(cutoff_date) - pd.to_datetime(dates)).dt.days.clip(lower=0).astype(float).values
        return np.exp(-self._lambda_per_day * ages)

    def _residualize_on_minutes(
        self, y: np.ndarray, minutes: Optional[np.ndarray], w: np.ndarray
    ) -> np.ndarray:
        """
        Weighted least squares residualization of y on [const, minutes].
        If minutes missing or degenerate, returns y unchanged.
        """
        if (minutes is None) or (minutes.size != y.size) or (not np.isfinite(minutes).any()):
            return y
        # Check variability
        if np.nanstd(minutes) < self.min_std:
            return y

        # Design matrix with intercept
        X = np.column_stack([np.ones_like(minutes), minutes])
        # Apply sqrt weights for WLS
        sw = np.sqrt(np.clip(w, 0.0, None))
        Xw = X * sw[:, None]
        yw = y * sw
        try:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            y_hat = X @ beta
            resid = y - y_hat
            return resid
        except Exception:
            return y

    def _compute_corr_pack(
        self,
        a: pd.DataFrame,  # ['GAME_DATE','val','MIN'?]
        b: pd.DataFrame,  # ['GAME_DATE','val','MIN'?]
        cutoff_date,
    ) -> Tuple[float, float, int, float, float, float]:
        """
        Merge two windows, build weights, winsorize, partial-out minutes,
        compute weighted Pearson (raw), weighted Spearman (approx via ranks),
        shrink with Fisher, and blend.
        Returns:
          r_final, p_value, n_common, n_eff, r_raw_shrunk, r_rank_shrunk
        """
        m = pd.merge(
            a.rename(columns={"val": "x", "MIN": "MIN_x"}),
            b.rename(columns={"val": "y", "MIN": "MIN_y"}),
            on="GAME_DATE",
            how="inner",
        ).dropna(subset=["x", "y"])
        n_common = int(len(m))
        if n_common < 2:
            return 0.0, 1.0, n_common, 0.0, 0.0, 0.0

        w = self._decay_weights(m["GAME_DATE"], cutoff_date)
        # Winsorize x,y (independently)
        x = _winsorize(m["x"].to_numpy(dtype=float), self.winsor_low, self.winsor_high)
        y = _winsorize(m["y"].to_numpy(dtype=float), self.winsor_low, self.winsor_high)

        # Optional partial-out minutes (per side)
        if self.use_partial_minutes:
            min_x = m["MIN_x"].to_numpy(dtype=float) if "MIN_x" in m.columns else None
            min_y = m["MIN_y"].to_numpy(dtype=float) if "MIN_y" in m.columns else None
            x = self._residualize_on_minutes(x, min_x, w)
            y = self._residualize_on_minutes(y, min_y, w)

        # Weighted Pearson (raw)
        r_raw = _weighted_corr(x, y, w, self.min_std)
        n_eff = _effective_n(w)

        # Weighted Spearman approx: Pearson on ranks (with same weights)
        # (rankdata handles ties; method='average' like SciPy default)
        rx = rankdata(x, method="average")
        ry = rankdata(y, method="average")
        r_rank = _weighted_corr(rx.astype(float), ry.astype(float), w, self.min_std)

        # Fisher shrink per component
        r_raw_shrunk = _shrink_r_fisher(r_raw, n_eff, self.shrink_tau)
        r_rank_shrunk = _shrink_r_fisher(r_rank, n_eff, self.shrink_tau)

        # Blend: final r used for filtering/sorting
        r_final = (1.0 - self.blend_rank) * r_raw_shrunk + self.blend_rank * r_rank_shrunk
        r_final = float(np.clip(r_final, -1.0, 1.0))

        # p-value from final r with effective N
        p_val = _pearson_pvalue_from_r(r_final, n_eff)

        return r_final, p_val, n_common, float(n_eff), r_raw_shrunk, r_rank_shrunk

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
    ) -> Tuple[float, float, int, float, float, float]:
        """
        Recency-weighted & shrunk (optionally partialed) Pearson/Spearman blend
        for the SAME stat on both players. Returns:
          r_final, p, n_common, n_eff, r_raw_shrunk, r_rank_shrunk
        """
        a = self._window_before(df1, stat, cutoff_date)
        b = self._window_before(df2, stat, cutoff_date)
        return self._compute_corr_pack(a, b, cutoff_date)

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
                            r_final, p, n, n_eff, r_raw_s, r_rank_s = self.corr_on_or_before(df1, df2, stat, cutoff_date)
                        except KeyError:
                            continue
                        if self._pass_filters(r_final, p, n):
                            rows.append(
                                {
                                    "event_id": event_id,
                                    "player1": p1,
                                    "player2": p2,
                                    "stat": stat,
                                    "correlation": float(r_final),
                                    "p_value": float(p),
                                    "n_games": int(n),
                                    # extras for debugging/tuning
                                    "r_raw": float(r_raw_s),
                                    "r_rank": float(r_rank_s),
                                    "n_eff": float(n_eff),
                                }
                            )

        cols = ["event_id", "player1", "player2", "stat", "correlation", "p_value", "n_games", "r_raw", "r_rank", "n_eff"]
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
    ) -> Tuple[float, float, int, float, float, float]:
        """
        Recency-weighted & shrunk (optionally partialed) Pearson/Spearman blend
        for DIFFERENT stats stat1 (p1) vs stat2 (p2). Returns:
          r_final, p, n_common, n_eff, r_raw_shrunk, r_rank_shrunk
        """
        a = self._window_before(df1, stat1, cutoff_date)
        b = self._window_before(df2, stat2, cutoff_date)
        return self._compute_corr_pack(a, b, cutoff_date)

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

                    for stat1 in s1s:
                        for stat2 in s2s:
                            try:
                                r_final, p, n, n_eff, r_raw_s, r_rank_s = self.corr_cross_stats_before(
                                    df1, df2, stat1, stat2, cutoff_date
                                )
                            except KeyError:
                                continue
                            if self._pass_filters(r_final, p, n):
                                rows.append(
                                    {
                                        "event_id": event_id,
                                        "player1": p1,
                                        "stat1": stat1,
                                        "player2": p2,
                                        "stat2": stat2,
                                        "correlation": float(r_final),
                                        "p_value": float(p),
                                        "n_games": int(n),
                                        # extras
                                        "r_raw": float(r_raw_s),
                                        "r_rank": float(r_rank_s),
                                        "n_eff": float(n_eff),
                                    }
                                )

        cols = ["event_id", "player1", "stat1", "player2", "stat2", "correlation", "p_value", "n_games", "r_raw", "r_rank", "n_eff"]
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
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                p1, p2 = players[i], players[j]
                total += max(self._val(cm, p1, p2), self._val(cm, p2, p1))
                count += 1
        return total / count if count else 0.0

    def _common_stat(self, corr_df: pd.DataFrame, players: List[str]) -> str:
        """
        Choose the most frequent stat among pairwise entries for this group (best-effort).
        Works with same-market DF. With cross-market DF, this is less meaningful.
        """
        stats: List[str] = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                p1, p2 = players[i], players[j]
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
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                p1, p2 = players[i], players[j]
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
        for idx_i in range(len(players)):
            for idx_j in range(idx_i + 1, len(players)):
                for idx_k in range(idx_j + 1, len(players)):
                    group = [players[idx_i], players[idx_j], players[idx_k]]
                    g_corr = self._group_correlation(group, cm)
                    val = abs(g_corr) if self.use_abs else g_corr
                    if val >= self.min_corr:
                        out.append(
                            {
                                "players": tuple(group),
                                "stat": self._common_stat(corr_df, group),
                                "avg_correlation": float(g_corr),
                                "worst_pair": float(self._min_pair_corr(group, cm)),
                            }
                        )
        return sorted(out, key=lambda x: -abs(x["avg_correlation"]))
