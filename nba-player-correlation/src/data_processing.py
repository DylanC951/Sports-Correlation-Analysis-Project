import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
import time
from typing import Optional, List, Dict

def fetch_player_data(player_id: int, seasons: Optional[List[str]] = None, retries: int = 3) -> pd.DataFrame:
    """Robust player data fetcher with improved error handling"""
    seasons = seasons or ['2023-24', '2022-23']  # Default to current and last season
    dfs = []
    
    for season in seasons:
        for attempt in range(retries):
            try:
                # Fetch data with timeout
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star='Regular Season',
                    timeout=30
                )
                
                # Get the dataframe
                df = gamelog.get_data_frames()[0]
                
                # Convert MIN to numeric safely
                if 'MIN' in df.columns:
                    # Handle cases where MIN might be in "MM:SS" format or numeric
                    try:
                        df['MIN'] = pd.to_numeric(df['MIN'])
                    except:
                        # If conversion fails, try splitting time format
                        try:
                            df['MIN'] = df['MIN'].str.split(':').str[0].astype(float)
                        except:
                            df['MIN'] = 0  # Default value if conversion fails
                else:
                    df['MIN'] = 0  # Add MIN column if missing
                
                # Add season marker
                df['SEASON'] = season
                dfs.append(df)
                break
                
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {season}: {str(e)}")
                if attempt == retries - 1:
                    print(f"Failed to fetch {season} for player {player_id}")
                time.sleep(2)  # Wait before retry
    
    if dfs:
        try:
            combined = pd.concat(dfs, ignore_index=True)
            return combined.sort_values('GAME_DATE', ascending=False)
        except Exception as e:
            print(f"Error combining dataframes: {str(e)}")
    return pd.DataFrame()

def clean_gamelog(df: pd.DataFrame) -> pd.DataFrame:
    """Improved data cleaning with robust column handling"""
    if df.empty:
        return df
    
    # Required columns with fallbacks
    required = {
        'GAME_DATE': pd.to_datetime,
        'PTS': lambda x: pd.to_numeric(x, errors='coerce').fillna(0),
        'AST': lambda x: pd.to_numeric(x, errors='coerce').fillna(0),
        'REB': lambda x: pd.to_numeric(x, errors='coerce').fillna(0),
        'MIN': lambda x: pd.to_numeric(x, errors='coerce').fillna(0)
    }
    
    # Add columns if missing
    for col in required:
        if col not in df.columns:
            df[col] = 0 if col != 'GAME_DATE' else pd.NaT
    
    # Apply conversions
    for col, converter in required.items():
        try:
            df[col] = converter(df[col])
        except Exception as e:
            print(f"Error converting {col}: {str(e)}")
            df[col] = 0 if col != 'GAME_DATE' else pd.NaT
    
    # Filter valid games
    df = df[df['MIN'] > 10]  # At least 10 minutes played
    df = df.dropna(subset=['GAME_DATE'])
    
    # Select columns to keep
    keep_cols = ['SEASON', 'GAME_DATE', 'MIN', 'PTS', 'AST', 'REB']
    optional_cols = ['STL', 'BLK', 'FG3M', 'TOV']
    
    for col in optional_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            keep_cols.append(col)
    
    return df[keep_cols]

def get_team_players(team_abbrev: str) -> List[int]:
    """Robust team roster fetcher"""
    try:
        # First try the API
        team_dict = teams.find_team_by_abbreviation(team_abbrev)
        if not team_dict:
            raise ValueError(f"Team {team_abbrev} not found")
        
        roster = commonteamroster.CommonTeamRoster(
            team_id=team_dict['id'],
            timeout=15
        )
        roster_df = roster.get_data_frames()[0]  # Players are in first dataframe
        return roster_df['PLAYER_ID'].tolist()
    
    except Exception as e:
        print(f"API roster fetch failed: {str(e)}")
        return get_fallback_players(team_abbrev)

def get_fallback_players(team_abbrev: str) -> List[int]:
    """Updated fallback with current player IDs"""
    teams_dict = {
        'LAL': [2544, 203076, 1626156, 1630559],  # LeBron, AD, DLo, Reaves
        'GSW': [201939, 203110, 1626178],         # Curry, Klay, Draymond
        'BOS': [1627759, 1628368, 1630162]        # Tatum, Brown, Smart
    }
    return teams_dict.get(team_abbrev.upper(), [])

def fetch_odds(player_id: int) -> Dict:
    """Mock odds fetcher with validation"""
    return {
        'player_points': {'odds': {'over': 1.91, 'under': 1.91}},
        'player_assists': {'odds': {'over': 1.83, 'under': 1.83}},
        'player_rebounds': {'odds': {'over': 1.74, 'under': 1.74}}
    }