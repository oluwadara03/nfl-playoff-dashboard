# Predictor - predictor.py

"""
This file uses the trained machine learning models to predict NFL playoff probabilities and projected wins 
for each team.
It uses team statistics from the 2025 season to generate the 2026 playoff outlook without using 2026 
game results.
"""

# Importing important packages and libraries
import joblib
import numpy as np
import pandas as pd

# Import necessary data
from src.data_processor import FEATURES, LAST_SEASON, MODEL_DIR, PROCESSED_DIR



class Predictor:

    # Constructor
    def __init__(self):
        # Load the saved playoff and wins prediction models
        self.playoff_model = joblib.load(MODEL_DIR / "best_model.pkl")
        self.wins_prediction_model = joblib.load(MODEL_DIR / "wins_model.pkl")


    # Predict based on team statistics
    def predict_team(self, input_features):
        # Predict playoff probabilities and expected wins
        playoff_probability = self.playoff_model.predict_proba(input_features[FEATURES])[:, 1]
        projected_wins = np.clip(self.wins_prediction_model.predict(input_features[FEATURES]), 0, 17)
        
        # Return probabilities and projected wins
        return playoff_probability, projected_wins


    # Predict playoff chances and wins for a single team
    def predict(self, win_percentage, points_per_game, points_allowed_per_game, made_playoffs):
        # Format team statistics in proper format for the models
        input_features = pd.DataFrame([{"win_pct": win_percentage, "ppg": points_per_game, "papg": points_allowed_per_game, "diff_pg": points_per_game - points_allowed_per_game, "made_playoffs": float(made_playoffs)}])

        # Predict playoff probability and wins
        playoff_probability, projected_wins = self.predict_team(input_features)
        return float(playoff_probability[0]), float(projected_wins[0])

    # Generate the 2026 playoff outlook for all teams
    def outlook(self):
        """Create the 2026 outlook for all 32 teams using only 2025-and-earlier data."""

        # Load team statistics and select the latest season
        team_seasons_table = pd.read_csv(PROCESSED_DIR / "team_seasons.csv")
        latest_season_teams = team_seasons_table[team_seasons_table.season == LAST_SEASON].reset_index(drop=True)

        # Predict each team's playoff chances and projected wins
        playoff_probability, projected_wins = self.predict_team(latest_season_teams)

        # Build the outlook table with team records and predictions
        outlook_table = pd.DataFrame({"Team": latest_season_teams.team, "2025 record": [f"{wins:g}-{games - wins:g}" for wins, games in zip(latest_season_teams.wins, latest_season_teams.games)], "P(playoffs 2026)": playoff_probability, "Projected wins": projected_wins, "Projected losses": 17 - projected_wins})
        outlook_table = outlook_table.sort_values("P(playoffs 2026)", ascending=False).reset_index(drop=True)
        
        # Ranking teams in order, while selecting the top 14 playoff predictions
        outlook_table.insert(0, "Rank", outlook_table.index + 1)
        # Only 14 teams make the NFL playoffs
        outlook_table["Playoff pick (top 14)"] = np.where(outlook_table.Rank <= 14, "Yes", "")

        # Return the outlook table and team data for the dashboard
        return outlook_table, latest_season_teams