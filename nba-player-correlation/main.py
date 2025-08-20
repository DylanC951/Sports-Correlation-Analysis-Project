from src.data_processing import LiveLineFetcher, load_active_player_data
from src.analysis import ParlayAnalyzer
from tqdm import tqdm
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Get active lines
    logger.info("Fetching active lines...")
    fetcher = LiveLineFetcher()
    active_lines = fetcher.get_active_lines()
    
    if not active_lines:
        logger.error("No active lines found. Check API key and connection.")
        return
    
    logger.info(f"Found {len(active_lines)} active players")
    
    # 2. Load player data
    logger.info("Loading player data...")
    player_data = load_active_player_data(active_lines)
    
    if not player_data:
        logger.error("No player data loaded. Exiting.")
        return
    
    # 3. Analyze correlations
    logger.info("Analyzing player pairs...")
    analyzer = ParlayAnalyzer(min_games=10, min_corr=0.35)
    corr_results = analyzer.analyze_active_pairs(player_data, active_lines)
    
    if corr_results.empty:
        logger.warning("No significant correlations found.")
        return
    
    print("\nTop 2-player correlations:")
    print(corr_results.sort_values('correlation', ascending=False).head(10))
    
    # 4. Find parlays
    logger.info("Finding best parlays...")
    for n in range(2, 5):  # 2-4 leg parlays
        parlays = analyzer.find_top_parlays(corr_results, n_legs=n)
        if parlays:
            print(f"\nTop {n}-leg parlays:")
            for i, parlay in enumerate(parlays[:5], 1):
                print(f"{i}. {parlay['avg_correlation']:.2f}: {', '.join(parlay['players'])} - {parlay['stat']}")

if __name__ == "__main__":
    main()