import pandas as pd
import requests
import logging
from datetime import datetime
from typing import Dict, List
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrizePicksAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
        
    def get_active_lines(self):
        """Fetch current player props from PrizePicks"""
        try:
            params = {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": "player_points,player_assists,player_rebounds",
                "bookmakers": "prizepicks"
            }
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return None

    def analyze_players(self):
        """Main analysis workflow"""
        data = self.get_active_lines()
        if not data:
            print("No active NBA lines found. Possible reasons:")
            print("- Offseason period")
            print("- No games today")
            print("- PrizePicks maintenance")
            return

        # Process players with active lines
        active_players = self.process_players(data)
        if not active_players:
            print("No players with qualifying stats found")
            return

        # Generate and save correlations
        results = self.find_correlations(active_players)
        if not results.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results.to_csv(f"prizepicks_results_{timestamp}.csv", index=False)
            print(f"\nTop Correlations Found:\n{results.head(10).to_string(index=False)}")
            print(f"\nFull results saved to prizepicks_results_{timestamp}.csv")

    def process_players(self, data):
        """Extract players with active lines"""
        players = {}
        for game in data:
            for market in game['markets']:
                stat_type = market['key'].replace('player_', '')
                for outcome in market['outcomes']:
                    player_name = outcome['description']
                    if player_name not in players:
                        players[player_name] = {
                            'stats': set(),
                            'lines': {}
                        }
                    players[player_name]['stats'].add(stat_type)
                    players[player_name]['lines'][stat_type] = {
                        'over': outcome.get('over', {}).get('price'),
                        'under': outcome.get('under', {}).get('price')
                    }
        return players

    def find_correlations(self, players):
        """Find correlated player pairs (mock implementation)"""
        results = []
        player_list = list(players.items())
        
        for i, (p1_name, p1_data) in enumerate(player_list):
            for p2_name, p2_data in player_list[i+1:]:
                common_stats = p1_data['stats'] & p2_data['stats']
                if common_stats:
                    # Replace this with your actual correlation calculation
                    correlation = 0.5  # Mock value
                    p_value = 0.01    # Mock value
                    
                    results.append({
                        'Player 1': p1_name,
                        'Player 2': p2_name,
                        'Stats': ', '.join(common_stats),
                        'Correlation': correlation,
                        'P-value': p_value,
                        'Suggested Bet': self.generate_bet(p1_name, p1_data, correlation)
                    })
        
        return pd.DataFrame(results).sort_values('Correlation', ascending=False)

    def generate_bet(self, player_name, player_data, correlation):
        """Generate betting suggestion (simplified example)"""
        stat = next(iter(player_data['stats']))  # Take first available stat
        line = player_data['lines'][stat]
        direction = "OVER" if correlation > 0 else "UNDER"
        price = line['over'] if direction == "OVER" else line['under']
        return f"{direction} {price} {stat.upper()}"

    # Add the other helper methods here (process_players, find_correlations etc.)
    # [I'll provide these in the next steps]

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("ODDS_API_KEY")
    
    if not api_key:
        print("ERROR: Missing API key. Create a .env file with ODDS_API_KEY=your_key")
    else:
        analyzer = PrizePicksAnalyzer(api_key)
        analyzer.analyze_players()