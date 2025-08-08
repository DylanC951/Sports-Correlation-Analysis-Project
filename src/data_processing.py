from nba_api.stats.endpoints import playergamelog
import pandas as pd
from pathlib import Path
from config.players import SEASONS, ALL_STATS

# Set up paths
base_dir = Path(__file__).parent.parent
raw_dir = base_dir / "data/raw"
processed_dir = base_dir / "data/processed"

# Create directories
raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

def fetch_player_multiseason(player_id, seasons=None):
    """Fetch game logs for multiple seasons with all stats"""
    if seasons is None:
        seasons = SEASONS
    
    all_data = []
    for season in seasons:
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season=season,
                season_type_all_star='Regular Season'
            )
            df = gamelog.get_data_frames()[0]
            df['SEASON'] = season
            all_data.append(df)
        except Exception as e:
            print(f"Error fetching {season} for player {player_id}: {str(e)}")
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def clean_gamelog(df):
    """Clean and keep all relevant stats"""
    if df.empty:
        return df
        
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Keep all stats plus metadata
    keep_cols = ['SEASON', 'GAME_DATE', 'MATCHUP', 'MIN'] + ALL_STATS
    return df[[c for c in keep_cols if c in df.columns]].sort_values('GAME_DATE')

def filter_min_games(df, min_games=20):
    """Filter to seasons with sufficient data"""
    if 'SEASON' not in df.columns:
        return df
    return df.groupby('SEASON').filter(lambda x: len(x) >= min_games)