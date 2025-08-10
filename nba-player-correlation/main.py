import pandas as pd
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import project modules
from src.data_processing import fetch_player_data, clean_gamelog as clean_data, fetch_odds, get_team_players
from src.analysis import CorrelationAnalyzer
from src.betting import BettingEngine
from src.visualization import plot_correlation_matrix
from config.players import PLAYER_IDS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame, player_name: str) -> bool:
    """Validate DataFrame has required columns and data"""
    required_cols = ['PTS', 'AST', 'REB', 'GAME_DATE', 'MIN']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Skipping {player_name} - missing required columns")
        return False
    if len(df) < 10:
        logger.warning(f"Skipping {player_name} - only {len(df)} games")
        return False
    return True

def load_player_data(player_ids: List[int], player_names: Dict[int, str]) -> Dict[int, pd.DataFrame]:
    """Load and validate data for all players"""
    all_data = {}
    for pid in tqdm(player_ids, desc="Fetching player stats"):
        try:
            raw_data = fetch_player_data(pid)
            df = clean_data(raw_data)
            
            if validate_dataframe(df, player_names.get(pid, str(pid))):
                df = df[df['MIN'] > 15]
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
                all_data[pid] = df
            else:
                logger.warning(f"Invalid dataframe for player {pid}")
        except Exception as e:
            logger.error(f"Error processing player {pid}: {str(e)}")
            continue
    return all_data

def analyze_player_pair(df1: pd.DataFrame, df2: pd.DataFrame, 
                      pid1: int, pid2: int, player_names: Dict[int, str],
                      min_corr: float) -> Optional[Tuple[Dict, Tuple[int, int, Tuple[str, str]]]]:
    """Analyze correlation between two players with debugging"""
    
    print(f"\nAnalyzing {player_names.get(pid1, 'Player 1')} vs {player_names.get(pid2, 'Player 2')}")
    print(f"Date range {player_names.get(pid1, 'Player 1')}: {df1['GAME_DATE'].min()} to {df1['GAME_DATE'].max()}")
    print(f"Date range {player_names.get(pid2, 'Player 2')}: {df2['GAME_DATE'].min()} to {df2['GAME_DATE'].max()}")
    
    common_dates = set(df1['GAME_DATE']).intersection(set(df2['GAME_DATE']))
    print(f"Common games: {len(common_dates)}")
    
    analyzer = CorrelationAnalyzer(min_games=5)
    corrs = analyzer.analyze_priority_pairs(df1, df2)
    filtered = analyzer.filter_significant(corrs)
    
    if not filtered:
        print("No significant correlations found")
        return None
    
    best_stat_pair, best_corr_data = max(
        filtered.items(),
        key=lambda x: abs(x[1]['pearson'][0])
    )
    
    result = {
        'Player 1': player_names.get(pid1, 'Player 1'),
        'Player 2': player_names.get(pid2, 'Player 2'),
        'Stat Pair': f"{best_stat_pair[0]} vs {best_stat_pair[1]}",
        'Correlation': best_corr_data['pearson'][0],
        'P-value': best_corr_data['pearson'][1],
        'Overlap Games': best_corr_data['n_games'],
        'Direction': 'Positive' if best_corr_data['pearson'][0] > 0 else 'Negative'
    }
    
    return result, (pid1, pid2, best_stat_pair)

def analyze_all_pairs(team_abbrev: str = 'LAL', min_corr: float = 0.15) -> List[Dict]:
    """Analyze all player pairs on a team for correlated betting opportunities
    
    Args:
        team_abbrev: 3-letter team abbreviation (e.g. 'LAL')
        min_corr: Minimum correlation coefficient to consider (default: 0.15)
        
    Returns:
        List of dictionaries containing correlation results for significant pairs
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING {team_abbrev} PLAYER PAIRS")
    print(f"{'='*50}")
    
    try:
        # 1. Load all player data
        print("\n[1/4] LOADING PLAYER DATA...")
        player_ids = get_team_players(team_abbrev)
        player_names = {v: k for k, v in PLAYER_IDS.items()}
        all_data = load_player_data(player_ids, player_names)
        
        if not all_data:
            print("\nNo valid player data found!")
            return []

        # 2. Analyze all possible pairs
        print("\n[2/4] ANALYZING PAIRWISE CORRELATIONS...")
        results = []
        valid_pairs = []
        
        for pid1, pid2 in combinations(all_data.keys(), 2):
            try:
                pair_result = analyze_player_pair(
                    all_data[pid1],
                    all_data[pid2],
                    pid1,
                    pid2,
                    player_names,
                    min_corr
                )
                if pair_result:
                    result, pair_info = pair_result
                    results.append(result)
                    valid_pairs.append(pair_info)
            except Exception as e:
                logger.error(f"Error processing pair {pid1}-{pid2}: {str(e)}")
                continue

        # 3. Display results with improved formatting
        print("\n[3/4] CORRELATION RESULTS")
        if not results:
            print("\nNo significant correlations found above threshold")
            return []
        
        # Create results dataframe with formatted numbers
        results_df = pd.DataFrame(results)
        
        # Format numeric columns
        results_df['Correlation'] = results_df['Correlation'].apply(lambda x: f"{x:.3f}")
        results_df['P-value'] = results_df['P-value'].apply(lambda x: f"{x:.4f}")
        
        # Sort by absolute correlation value
        results_df['abs_corr'] = results_df['Correlation'].astype(float).abs()
        results_df = results_df.sort_values('abs_corr', ascending=False).drop('abs_corr', axis=1)
        
        print(f"\nTop correlations (min r = {min_corr}):")
        print(results_df.to_string(index=False))
        
        # 4. Generate betting recommendations
        print("\n[4/4] GENERATING BETTING RECOMMENDATIONS...")
        if valid_pairs:
            analyzer = CorrelationAnalyzer()
            engine = BettingEngine(edge_threshold=0.02)
            all_bets = []
            
            for pid1, pid2, stat_pair in valid_pairs:
                try:
                    odds1 = fetch_odds(pid1)
                    odds2 = fetch_odds(pid2)
                    corr_data = analyzer.calculate_correlation(
                        all_data[pid1],
                        all_data[pid2],
                        stat_pair[0],
                        stat_pair[1]
                    )
                    
                    bets = engine.generate_recommendations(
                        {stat_pair: corr_data},
                        odds1,
                        odds2
                    )
                    all_bets.extend(bets)
                except Exception as e:
                    logger.error(f"Error generating bets for {pid1}-{pid2}: {str(e)}")
                    continue
            
            if all_bets:
                # Create and format bets dataframe
                bets_df = pd.DataFrame(all_bets)
                
                # Format numeric columns
                bets_df['correlation'] = bets_df['correlation'].apply(lambda x: f"{x:.3f}")
                bets_df['edge'] = bets_df['edge'].apply(lambda x: f"{x:.4f}")
                
                # Format odds display
                bets_df['odds'] = bets_df['odds'].apply(
                    lambda x: f"{x['over']:.2f}/{x['under']:.2f}" if isinstance(x, dict) else str(x))
                
                print("\nTOP BETTING OPPORTUNITIES:")
                print(bets_df.to_string(index=False))
                
                # Ensure results directory exists
                Path("results").mkdir(exist_ok=True)
                
                # Save with timestamp
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                results_df.to_csv(f"results/{team_abbrev}_correlations_{timestamp}.csv", index=False)
                bets_df.to_csv(f"results/{team_abbrev}_bets_{timestamp}.csv", index=False)
                print(f"\nResults saved to results/{team_abbrev}_*_{timestamp}.csv")
            else:
                print("No qualifying bets found")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error in analyze_all_pairs: {str(e)}")
        return []

if __name__ == "__main__":
    analyze_all_pairs(team_abbrev='LAL', min_corr=0.1)