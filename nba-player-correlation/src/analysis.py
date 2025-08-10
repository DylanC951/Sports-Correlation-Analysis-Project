import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, List

class CorrelationAnalyzer:
    def __init__(self, min_games: int = 10, window_size: int = 5):  # Lowered min_games for testing
        self.min_games = min_games
        self.window_size = window_size

    def calculate_correlation(self, df1, df2, stat1, stat2):
        """Safe correlation calculation with proper column handling"""
        try:
            print(f"\nCalculating {stat1} vs {stat2}")
            
            # Ensure consistent date formatting
            df1['GAME_DATE'] = pd.to_datetime(df1['GAME_DATE']).dt.date
            df2['GAME_DATE'] = pd.to_datetime(df2['GAME_DATE']).dt.date
            
            # Merge with proper suffixes
            merged = pd.merge(
                df1[['GAME_DATE', stat1]].rename(columns={stat1: f"{stat1}_1"}),
                df2[['GAME_DATE', stat2]].rename(columns={stat2: f"{stat2}_2"}),
                on='GAME_DATE',
                how='inner'
            ).dropna()
            
            print(f"Merged games found: {len(merged)}")
            if len(merged) > 0:
                print("Sample merged data:")
                print(merged.head(3))
            
            if len(merged) < self.min_games:
                print(f"Not enough games: {len(merged)} < {self.min_games}")
                return {'pearson': (0.0, 1.0), 'spearman': (0.0, 1.0), 'n_games': 0}
            
            # Calculate correlations - extract just the values we need
            col1 = f"{stat1}_1"
            col2 = f"{stat2}_2"
            pearson_result = pearsonr(merged[col1], merged[col2])
            spearman_result = spearmanr(merged[col1], merged[col2])
            
            # Format the output cleanly
            print(f"Pearson r: {pearson_result.statistic:.3f} (p={pearson_result.pvalue:.4f})")
            print(f"Spearman ρ: {spearman_result.statistic:.3f} (p={spearman_result.pvalue:.4f})")
            
            return {
                'pearson': (pearson_result.statistic, pearson_result.pvalue),
                'spearman': (spearman_result.statistic, spearman_result.pvalue),
                'n_games': len(merged)
            }
        except Exception as e:
            print(f"Correlation failed for {stat1}/{stat2}: {str(e)}")
            return {'pearson': (0.0, 1.0), 'spearman': (0.0, 1.0), 'n_games': 0}

    def analyze_priority_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """Analyze prioritized stat combinations between two players"""
        priority_pairs = [
            ('PTS', 'AST'),   # Scorer to playmaker
            ('PTS', 'REB'),   # Scorer to rebounder
            ('AST', 'REB'),   # Playmaker to rebounder
            ('PTS', 'PTS'),   # Co-scoring
            ('AST', 'AST'),   # Co-playmaking
            ('REB', 'REB')    # Co-rebounding
        ]
        
        results = {}
        for stat1, stat2 in priority_pairs:
            if stat1 in df1.columns and stat2 in df2.columns:
                results[(stat1, stat2)] = self.calculate_correlation(df1, df2, stat1, stat2)
        
        return results

    def filter_significant(self, results: Dict, p_threshold: float = 0.1, corr_threshold: float = 0.15) -> Dict:
        """Filter results with relaxed thresholds for debugging"""
        return {
            k: v for k, v in results.items()
            if (abs(v['pearson'][0]) >= corr_threshold and 
                v['pearson'][1] < p_threshold)
        }