import pandas as pd
from scipy.stats import pearsonr
from typing import Dict, Tuple

def calculate_correlation(df1, df2, stat1, stat2):
    """Calculate correlation between any two stats"""
    merged = pd.merge(df1, df2, on='GAME_DATE', suffixes=('_1', '_2'))
    return pearsonr(merged[f'{stat1}_1'], merged[f'{stat2}_2'])

def calculate_seasonal_correlations(df1, df2, stat1, stat2):
    """Calculate per-season correlations"""
    merged = pd.merge(df1, df2, on=['SEASON', 'GAME_DATE'], suffixes=('_1', '_2'))
    return {
        season: pearsonr(group[f'{stat1}_1'], group[f'{stat2}_2'])
        for season, group in merged.groupby('SEASON')
    }

def analyze_stat_combinations(df1, df2, stats):
    """Analyze all possible stat combinations"""
    results = {}
    stat_pairs = [(s1, s2) for s1 in stats for s2 in stats if s1 != s2]
    
    for stat1, stat2 in stat_pairs:
        try:
            corr, p_value = calculate_correlation(df1, df2, stat1, stat2)
            seasonal = calculate_seasonal_correlations(df1, df2, stat1, stat2)
            results[(stat1, stat2)] = {
                'overall': (corr, p_value),
                'seasonal': seasonal
            }
        except Exception as e:
            print(f"Error analyzing {stat1} vs {stat2}: {str(e)}")
    
    return results