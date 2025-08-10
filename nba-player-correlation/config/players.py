# Player Database
PLAYER_IDS = {
    "LeBron James": 2544,
    "Stephen Curry": 201939,
    "Anthony Davis": 203076,
    "Nikola Jokic": 203999,
    "Luka Doncic": 1629029,
    "D'Angelo Russell": 1626156,
    "Austin Reaves": 1630559,
    "Rui Hachimura": 1629060
}

TEAM_ABBREVIATIONS = {
    'LAL': 'Los Angeles Lakers',
    'GSW': 'Golden State Warriors'
}

# Seasons to analyze (last 4 years)
SEASONS = [f"{year}-{str(year+1)[-2:]}" for year in range(2020, 2024)]

# All bettable stats (from major sportsbooks)
BETTABLE_STATS = [
    # Scoring
    'PTS', 'FG3M', 'FG3A', 'FGM', 'FGA', 'FTM', 'FTA',
    # Playmaking
    'AST', 'TOV',
    # Rebounding
    'REB', 'OREB', 'DREB',
    # Defense
    'STL', 'BLK',
    # Miscellaneous
    'PLUS_MINUS', 'PF', 'MIN'
]

# Stats with liquid betting markets
PRIMARY_BET_STATS = ['PTS', 'AST', 'REB', 'FG3M', 'STL', 'BLK', 'TOV']  # Added more stats

# Minimum games for statistical significance
MIN_GAMES = 25