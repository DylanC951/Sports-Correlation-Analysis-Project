# Add this at the VERY TOP of visualization.py
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, Tuple, Optional
from config.players import PLAYER_NAMES, ALL_STATS

# Stat display names for plots
STAT_NAMES = {
    'PTS': 'Points',
    'AST': 'Assists',
    'REB': 'Rebounds',
    'STL': 'Steals',
    'BLK': 'Blocks',
    'FG3M': '3PM',
    'FG3A': '3PA',
    'FG3_PCT': '3P%',
    'FGM': 'FGM',
    'FGA': 'FGA',
    'FTM': 'FTM',
    'FTA': 'FTA',
    'OREB': 'Off Reb',
    'DREB': 'Def Reb',
    'TOV': 'Turnovers',
    'PF': 'Fouls',
    'PLUS_MINUS': '+/-',
    'MIN': 'Minutes'
}

def sanitize_filename(name):
    return re.sub(r'[^\w-]', '', name.replace(' ', '_'))

def plot_correlation(df1, df2, stat1, stat2, player1_id, player2_id, corr_value=None):
    """Enhanced correlation plot for any stat combination"""
    # Setup
    player1_name = PLAYER_NAMES.get(player1_id, f"Player {player1_id}")
    player2_name = PLAYER_NAMES.get(player2_id, f"Player {player2_id}")
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Create filename
    filename = f"{sanitize_filename(player1_name)}_{stat1}_vs_{sanitize_filename(player2_name)}_{stat2}.png"
    output_path = output_dir / filename
    
    # Prepare data
    merged = pd.merge(df1, df2, on='GAME_DATE', suffixes=('_1', '_2'))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    ax = sns.regplot(
        x=merged[f'{stat1}_1'],
        y=merged[f'{stat2}_2'],
        scatter_kws={'alpha': 0.6, 's': 60},
        line_kws={'color': 'red', 'linestyle': '--'}
    )
    
    # Add annotations
    season_range = get_season_range(merged)
    plt.title(
        f"{player1_name} {STAT_NAMES.get(stat1, stat1)} vs\n"
        f"{player2_name} {STAT_NAMES.get(stat2, stat2)}\n"
        f"{season_range} | {len(merged)} games",
        fontsize=14
    )
    
    if corr_value:
        plt.annotate(
            f"r = {corr_value:.2f}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle='round', fc='white')
        )  # This was the missing parenthesis
    
    plt.xlabel(f"{player1_name} {STAT_NAMES.get(stat1, stat1)}")
    plt.ylabel(f"{player2_name} {STAT_NAMES.get(stat2, stat2)}")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_seasonal_trends(seasonal_data, player1_id, player2_id, stat1, stat2):
    """Plot correlation trends across seasons"""
    player1_name = PLAYER_NAMES.get(player1_id, f"Player {player1_id}")
    player2_name = PLAYER_NAMES.get(player2_id, f"Player {player2_id}")
    output_dir = Path(__file__).parent.parent / "trends"
    output_dir.mkdir(exist_ok=True)
    
    # Prepare data
    trends = pd.DataFrame([
        (season, corr, pval) 
        for season, (corr, pval) in seasonal_data.items()
    ], columns=['Season', 'Correlation', 'P-value'])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=trends, x='Season', y='Correlation', palette='viridis')
    
    # Add significance markers
    for i, row in trends.iterrows():
        if row['P-value'] < 0.05:
            ax.text(i, row['Correlation'] + 0.05, "*", ha='center', fontsize=14)
    
    plt.title(
        f"Seasonal Correlation: {player1_name} {stat1} vs {player2_name} {stat2}",
        fontsize=12
    )
    plt.ylim(-1, 1)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    # Save
    filename = f"Trend_{sanitize_filename(player1_name)}_{stat1}_vs_{sanitize_filename(player2_name)}_{stat2}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def get_season_range(df):
    """Helper to format season range string"""
    if 'SEASON' not in df.columns:
        return ""
    seasons = df['SEASON'].unique()
    return f"{min(seasons)} to {max(seasons)}"