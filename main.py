from src.data_processing import fetch_player_multiseason, clean_gamelog, filter_min_games
from src.analysis import analyze_stat_combinations
from src.visualization import plot_correlation, plot_seasonal_trends
from config.players import PLAYER_NAMES, ALL_STATS, SEASONS
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import time
import json

def analyze_players(player1_id, player2_id, stats=None, seasons=None, min_games=20):
    """Main analysis function for any two players"""
    # Initialize
    stats = stats or ALL_STATS
    seasons = validate_seasons(seasons)
    results = {
        'players': (player1_id, player2_id),
        'seasons': seasons,
        'stats': stats,
        'correlations': {},
        'plots': []
    }
    
    # Fetch data
    print(f"\nFetching data for {get_player_name(player1_id)} vs {get_player_name(player2_id)}...")
    df1 = clean_gamelog(fetch_player_multiseason(player1_id, seasons))
    df2 = clean_gamelog(fetch_player_multiseason(player2_id, seasons))
    
    # Filter
    df1 = filter_min_games(df1, min_games)
    df2 = filter_min_games(df2, min_games)
    
    if df1.empty or df2.empty:
        print("Insufficient data after filtering")
        return None
    
    # Analyze all stat combinations
    print("Calculating correlations...")
    start_time = time.time()
    stat_results = analyze_stat_combinations(df1, df2, stats)
    
    # Generate visualizations
    print("Creating plots...")
    with ThreadPoolExecutor() as executor:
        futures = []
        for (stat1, stat2), data in stat_results.items():
            futures.append(executor.submit(
                create_visualizations,
                df1, df2, stat1, stat2,
                player1_id, player2_id,
                data
            ))
        
        for future in futures:
            plot_data = future.result()
            if plot_data:
                results['correlations'][plot_data['stat_pair']] = plot_data['correlation']
                results['plots'].append(plot_data['paths'])
    
    # Save results
    save_path = save_analysis_results(results)
    print(f"\nAnalysis completed in {time.time()-start_time:.1f}s")
    print(f"Results saved to: {save_path}")
    
    return results

def create_visualizations(df1, df2, stat1, stat2, p1_id, p2_id, data):
    """Helper to parallelize plot generation"""
    try:
        corr_plot = plot_correlation(
            df1, df2, stat1, stat2,
            p1_id, p2_id, data['overall'][0]
        )
        trend_plot = plot_seasonal_trends(
            data['seasonal'], p1_id, p2_id, stat1, stat2
        )
        return {
            'stat_pair': (stat1, stat2),
            'correlation': data['overall'],
            'paths': {
                'correlation': str(corr_plot),
                'trend': str(trend_plot)
            }
        }
    except Exception as e:
        print(f"Error visualizing {stat1} vs {stat2}: {str(e)}")
        return None

def validate_seasons(seasons):
    """Ensure valid seasons"""
    if seasons is None:
        return SEASONS
    return [s for s in (seasons if isinstance(seasons, list) else [seasons]) if s in SEASONS]

def get_player_name(player_id):
    return PLAYER_NAMES.get(player_id, f"Player_{player_id}")

def save_analysis_results(results):
    """Save all results to JSON and CSV"""
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # JSON output
    p1, p2 = results['players']
    json_path = output_dir / f"{p1}_{p2}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # CSV output
    csv_data = []
    for (stat1, stat2), data in results['correlations'].items():
        csv_data.append({
            'player1': get_player_name(p1),
            'player2': get_player_name(p2),
            'stat1': stat1,
            'stat2': stat2,
            'correlation': data[0],
            'p_value': data[1],
            'seasons': len(results['seasons'])
        })
    
    csv_path = output_dir / f"{p1}_{p2}_results.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    
    return json_path

if __name__ == "__main__":
    # Example analysis
    results = analyze_players(
        player1_id=2544,  # LeBron
        player2_id=203076,  # AD
        stats=['PTS', 'AST', 'REB', 'FG3M', 'BLK'],  # Or None for all stats
        seasons=['2023-24', '2022-23', '2021-22', '2020-21', '2019-20'],
        min_games=15
    )
    
    # Print top correlations
    if results:
        print("\nTop Correlations:")
        sorted_results = sorted(
            results['correlations'].items(),
            key=lambda x: abs(x[1][0]),
            reverse=True
        )[:10]
        
        for (stat1, stat2), (corr, pval) in sorted_results:
            print(f"{stat1} vs {stat2}: {corr:.2f} (p={pval:.4f})")