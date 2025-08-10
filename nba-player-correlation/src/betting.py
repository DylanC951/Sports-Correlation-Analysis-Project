from typing import Dict, List, Optional  # Added Optional here
import numpy as np

class BettingEngine:
    def __init__(self, edge_threshold: float = 0.05):
        self.edge_threshold = edge_threshold

    def generate_recommendations(self, 
                               correlations: Dict, 
                               player1_odds: Dict, 
                               player2_odds: Dict) -> List[Dict]:
        """Convert correlations into betting recommendations"""
        bets = []
        
        for (stat1, stat2), corr_data in correlations.items():
            # Get odds for both players
            odds1 = self._extract_odds(player1_odds, stat1)
            odds2 = self._extract_odds(player2_odds, stat2)
            
            if not odds1 or not odds2:
                continue
                
            # Calculate expected value
            edge1 = self._calculate_edge(corr_data['pearson'][0], odds1)
            edge2 = self._calculate_edge(corr_data['pearson'][0], odds2)
            
            if edge1 > self.edge_threshold:
                bets.append(self._format_bet(
                    player=1, stat=stat1, 
                    correlation=corr_data['pearson'][0],
                    edge=edge1, odds=odds1
                ))
                
            if edge2 > self.edge_threshold:
                bets.append(self._format_bet(
                    player=2, stat=stat2,
                    correlation=corr_data['pearson'][0],
                    edge=edge2, odds=odds2
                ))
        
        return sorted(bets, key=lambda x: -x['edge'])

    def _extract_odds(self, odds_data: Dict, stat: str) -> Optional[Dict]:
        """Convert API odds to standardized format"""
        stat_map = {
            'PTS': 'player_points',
            'AST': 'player_assists',
            'REB': 'player_rebounds'
        }
        market = stat_map.get(stat)
        return odds_data.get(market, {}).get('odds') if market else None

    def _calculate_edge(self, corr: float, odds: Dict) -> float:
        """Calculate expected value based on correlation"""
        implied_prob = 1 / odds['over']
        predicted_prob = 0.5 + (corr * 0.3)  # Adjusted correlation-to-prob formula
        return predicted_prob - implied_prob

    def _format_bet(self, player: int, stat: str, 
                   correlation: float, edge: float, 
                   odds: Dict) -> Dict:
        return {
            'player': player,
            'stat': stat,
            'direction': 'OVER' if correlation > 0 else 'UNDER',
            'correlation': round(correlation, 3),
            'edge': round(edge, 3),
            'odds': odds,
            'confidence': self._get_confidence_level(abs(correlation))
        }

    def _get_confidence_level(self, corr: float) -> str:
        if corr > 0.7: return "HIGH"
        elif corr > 0.5: return "MEDIUM"
        return "LOW"