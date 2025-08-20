import requests
import json
from datetime import datetime, timedelta
import time
from tqdm import tqdm

# Configuration
API_KEY = '83e6497c8f87c47af8ea4bf172fcbc83'
SPORT = 'basketball_nba'
REGIONS = 'us_dfs'
ODDS_FORMAT = 'american'
MAX_REQUESTS = 50  # Very conservative limit
REQUEST_DELAY = 1

def get_december_dates():
    """Get only key December dates (weekends + Christmas)"""
    key_dates = [
        '2024-12-01', '2024-12-07', '2024-12-08',  # Weekends
        '2024-12-14', '2024-12-15', '2024-12-21',
        '2024-12-22', '2024-12-25', '2024-12-28',
        '2024-12-29'
    ]
    return [f"{date}T00:00:00Z" for date in key_dates]

def check_all_props_in_one_request(game_id, date):
    """Check all three prop types in a single API call"""
    try:
        url = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/events/{game_id}/odds'
        params = {
            'apiKey': API_KEY,
            'date': date,
            'regions': REGIONS,
            'oddsFormat': ODDS_FORMAT,
            'markets': 'player_points,player_rebounds,player_assists'  # All markets in one request
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check which markets are available
        bookmakers = data.get('data', {}).get('bookmakers', [])
        available_markets = set()
        
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                available_markets.add(market['key'])
        
        return {
            'player_points': 'player_points' in available_markets,
            'player_rebounds': 'player_rebounds' in available_markets,
            'player_assists': 'player_assists' in available_markets
        }
        
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Error checking game {game_id}: {str(e)}")
        return {m: False for m in ['player_points', 'player_rebounds', 'player_assists']}

def efficient_scan():
    """Highly optimized scan using minimal requests"""
    dates = get_december_dates()
    results = []
    request_count = 0
    
    print("Running efficient scan (max {} requests)...".format(MAX_REQUESTS))
    
    try:
        for date in tqdm(dates, desc="Checking dates"):
            if request_count >= MAX_REQUESTS:
                break
                
            # Get games for date (1 request)
            games_url = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/events'
            params = {
                'apiKey': API_KEY,
                'regions': REGIONS,
                'date': date
            }
            games_response = requests.get(games_url, params=params, timeout=10)
            games_response.raise_for_status()
            games = games_response.json().get('data', [])
            request_count += 1
            time.sleep(REQUEST_DELAY)
            
            if not games:
                continue
                
            # Check first 2 games per date to save requests
            for game in games[:2]:
                if request_count >= MAX_REQUESTS:
                    break
                    
                props = check_all_props_in_one_request(game['id'], date)
                request_count += 1
                time.sleep(REQUEST_DELAY)
                
                results.append({
                    'date': date[:10],
                    'home': game['home_team'],
                    'away': game['away_team'],
                    **props
                })
                
    except Exception as e:
        print(f"Scan error: {str(e)}")
    
    # Generate report
    print("\n=== RESULTS ===")
    print(f"Used {request_count} requests")
    print(f"Checked {len(results)} games")
    
    if results:
        # Calculate availability percentages
        total = len(results)
        points = sum(1 for r in results if r['player_points'])
        rebounds = sum(1 for r in results if r['player_rebounds'])
        assists = sum(1 for r in results if r['player_assists'])
        all_three = sum(1 for r in results if all(r[m] for m in ['player_points', 'player_rebounds', 'player_assists']))
        
        print("\nProp Availability:")
        print(f"Points: {points}/{total} ({points/total:.1%})")
        print(f"Rebounds: {rebounds}/{total} ({rebounds/total:.1%})")
        print(f"Assists: {assists}/{total} ({assists/total:.1%})")
        print(f"All Three: {all_three}/{total} ({all_three/total:.1%})")
        
        # Save results
        with open('prop_availability.csv', 'w') as f:
            f.write("Date,Home Team,Away Team,Points,Rebounds,Assists\n")
            for r in results:
                f.write(f"{r['date']},{r['home']},{r['away']},"
                       f"{'Y' if r['player_points'] else 'N'},"
                       f"{'Y' if r['player_rebounds'] else 'N'},"
                       f"{'Y' if r['player_assists'] else 'N'}\n")
        print("\nSaved results to prop_availability.csv")

if __name__ == '__main__':
    efficient_scan()