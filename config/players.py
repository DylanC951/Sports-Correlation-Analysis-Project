# config/players.py

# All available NBA stats we want to analyze
ALL_STATS = [
    'PTS',    # Points
    'AST',    # Assists
    'REB',    # Rebounds
    'STL',    # Steals
    'BLK',    # Blocks
    'TOV',    # Turnovers
    'FGM',    # Field Goals Made
    'FGA',    # Field Goals Attempted
    'FG3M',   # 3-Pointers Made
    'FG3A',   # 3-Pointers Attempted
    'FTM',    # Free Throws Made
    'FTA',    # Free Throws Attempted
    'OREB',   # Offensive Rebounds
    'DREB',   # Defensive Rebounds
    'PF',     # Personal Fouls
    'PLUS_MINUS',  # Plus-Minus
    'MIN'     # Minutes Played
]

# Seasons to analyze (format: 'YYYY-YY')
SEASONS = [
    '2023-24',
    '2022-23',
    '2021-22',
    '2020-21',
    '2019-20',
    '2018-19',
    '2017-18',
    '2016-17',
    '2015-16',
]

# Player IDs and names
PLAYER_NAMES = {
    2544: "LeBron James",
    203076: "Anthony Davis",
    201939: "Stephen Curry",
    202691: "Klay Thompson",
    203999: "Nikola Jokic",
    # Add more players as needed
}

# Team rosters (list of player IDs)
TEAM_ROSTERS = {
    "Lakers": [2544, 203076],  # LeBron, AD
    "Warriors": [201939, 202691],  # Curry, Thompson
    "Nuggets": [203999],  # Jokic
    # Add more teams as needed
}