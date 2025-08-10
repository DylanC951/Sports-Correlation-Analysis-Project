import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List  # Added List here
from config.players import PLAYER_IDS

plt.style.use('ggplot')

def plot_correlation_matrix(correlations: Dict, player1_id: int, player2_id: int):
    """Heatmap of all stat correlations"""
    player1_name = next(k for k,v in PLAYER_IDS.items() if v == player1_id)
    player2_name = next(k for k,v in PLAYER_IDS.items() if v == player2_id)
    
    # Prepare matrix
    stats = sorted({k[0] for k in correlations.keys()})
    matrix = pd.DataFrame(index=stats, columns=stats)
    
    for (stat1, stat2), data in correlations.items():
        matrix.loc[stat1, stat2] = data['pearson'][0]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix.astype(float), annot=True, cmap='coolwarm', center=0)
    plt.title(f"{player1_name} vs {player2_name}\nStat Correlation Matrix")
    
    # Save
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    filename = f"corr_matrix_{player1_id}_{player2_id}.png"
    plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_betting_opportunities(bets: List[Dict]):
    """Visualize top betting recommendations"""
    if not bets:
        return
        
    df = pd.DataFrame(bets)
    df = df.sort_values('edge', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='stat', y='edge', hue='direction')
    plt.title("Top Betting Opportunities by Expected Edge")
    plt.ylabel("Expected Value (%)")
    plt.axhline(0, color='black', linestyle='--')
    
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "top_bets.png", bbox_inches='tight', dpi=300)
    plt.close()