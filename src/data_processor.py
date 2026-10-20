# Data Processor - data_processor.py

"""
This file imports the data from nflverse (games.csv) to use in the project. 
It also contains functions to clean and process the data for analysis.
Only data from the 2023, 2024 and 2025 NFL seasons are used in this project. All 2026 NFL data is excluded.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

# Setting paths used to read and save project files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" /"2023_to_2025.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
NFL_GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
FIRST_SEASON = 2023
LAST_SEASON = 2025

# Features used in the model
FEATURES = ["win_pct", "ppg", "papg", "diff_pg", "made_playoffs"]



class DataProcessor:
   
    # Constructor
    def __init__(self):
        # Stores information about the data
        self.report = {}
    
    
    # Loads the data
    def load(self):
        # Download the NFL schedule if raw data file does is not present
        if not RAW_PATH.exists():
            # Creates directory for the file
            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # Reads the file from the URL
            games = pd.read_csv(NFL_GAMES_URL)
            self.report["downloaded_2026_games_dropped"] = int((games.season > LAST_SEASON).sum())
            games = games[games.season.between(FIRST_SEASON, LAST_SEASON)]
            games.to_csv(RAW_PATH, index=False)   
             
        # Loads NFL data into a DataFrame
        return pd.read_csv(RAW_PATH)
    
    
    # Cleans up the data
    def clean (self, games):
       # Keeps raw data before removing duplicates and games without scores
        cleaning_report = self.report
        cleaning_report["raw_shape"] = list(games.shape)
        cleaning_report["games_per_season"] = games.groupby("season").size().to_dict()
        cleaning_report["missing_scores"] = int(games[["home_score", "away_score"]].isna().any(axis=1).sum())
        cleaning_report["duplicate_games"] = int(games.duplicated(subset=["game_id"]).sum())
        
        # Remove duplicate games and rows missing scores
        games = games.drop_duplicates("game_id").dropna(subset=["home_score", "away_score"])
        
        # Restrict dataset to include only NFL seasons of 2023 through to 2025
        games = games[games.season.between(FIRST_SEASON, LAST_SEASON)]
        assert games.season.max() <= LAST_SEASON, "Do not include 2026 data"
        
        # Save the number of rows remaining after cleaning
        cleaning_report["clean_shape"] = list(games.shape)
        
        # Return cleaned dataset of selected NFL games
        return games


    # Calculates statistics of teams in specified seasons 
    def team_seasons(self, games):
        # Only regular season games are being used, no playoff games
        regular_season_games = games[games.game_type == "REG"]
        # Home team games, showing points scored and points allowed
        home_team_games = pd.DataFrame({"season": regular_season_games.season, "team": regular_season_games.home_team, "pf": regular_season_games.home_score, "pa": regular_season_games.away_score})
        # Away team games, showing points scored and points allowed
        away_team_games = pd.DataFrame({"season": regular_season_games.season, "team": regular_season_games.away_team, "pf": regular_season_games.away_score, "pa": regular_season_games.home_score})
        
        # Combine team statistics accurately
        team_game_rows = pd.concat([home_team_games, away_team_games])
        # Teams get one point for a win, zero for a loss and 0.5 for a tie
        team_game_rows["win"] = np.where(team_game_rows.pf > team_game_rows.pa, 1.0, np.where(team_game_rows.pf < team_game_rows.pa, 0.0, 0.5))
        
        # Aggregate game records by season and team to calculate overall performance
        team_seasons_table = team_game_rows.groupby(["season", "team"]).agg(games=("win", "size"), wins=("win", "sum"), pf=("pf", "sum"), pa=("pa", "sum")).reset_index()
       
        # Calculation of performance metrics
        win_percentage = team_seasons_table.wins / team_seasons_table.games
        points_per_game = team_seasons_table.pf / team_seasons_table.games
        points_allowed_per_game = team_seasons_table.pa / team_seasons_table.games
        point_differential_per_game = points_per_game - points_allowed_per_game
    
        # Add performance metrics to the required table
        team_seasons_table["win_pct"] = win_percentage
        team_seasons_table["ppg"] = points_per_game
        team_seasons_table["papg"] = points_allowed_per_game
        team_seasons_table["diff_pg"] = point_differential_per_game

        # Identify teams that appeared in postseason games to determine they made the playoffs
        postseason_games = games[games.game_type != "REG"]
        playoff_team_seasons = pd.concat([postseason_games[["season", "home_team"]].set_axis(["season", "team"], axis=1), postseason_games[["season", "away_team"]].set_axis(["season", "team"], axis=1)])
        playoff_team_seasons.drop_duplicates(inplace=True)

        # Binominal classification: postseason teams get 1 and non-playoff teams get 0
        playoff_team_seasons["made_playoffs"] = 1
        team_seasons_table = team_seasons_table.merge(playoff_team_seasons, on=["season", "team"], how="left").fillna({"made_playoffs": 0})

        # Record outliers and missing values
        point_differential_z_scores = np.abs((team_seasons_table.diff_pg - team_seasons_table.diff_pg.mean()) / team_seasons_table.diff_pg.std())
        self.report["outliers_kept_diff_pg_z_gt_3"] = int((point_differential_z_scores > 3).sum())
        self.report["missing_in_team_table"] = int(team_seasons_table.isna().sum().sum())
        self.report["playoff_teams_per_season"] = team_seasons_table.groupby("season")["made_playoffs"].sum().astype(int).to_dict()

        # Return the team-season table
        return team_seasons_table


    # Creating pairs based on season results
    def make_pairs(self, ts):
        # next season prediction is the target
        future = ts[["season", "team", "wins", "made_playoffs"]].copy()
        future["season"] -= 1
        future = future.rename(columns={"wins": "next_wins", "made_playoffs": "next_playoffs"})
        pairs = ts.merge(future, on=["season", "team"])
        
        # Record number fo training pairs
        self.report["pairs_per_feature_season"] = pairs.groupby("season").size().to_dict()
        
        # Return the training data
        return pairs


    # Runs the data processing
    def run(self):
        # Check folders exist
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Load, clean, summarize, and pair the NFL data in order
        games = self.clean(self.load())
        team_seasons_table = self.team_seasons(games)
        training_pairs = self.make_pairs(team_seasons_table)

        # Save processed files and cleaning report
        team_seasons_table.to_csv(PROCESSED_DIR / "team_seasons.csv", index=False)
        training_pairs.to_csv(PROCESSED_DIR / "pairs.csv", index=False)
        (PROCESSED_DIR / "cleaning_report.json").write_text(json.dumps(self.report, indent=2, default=str))
        
        # Return team statistics and training pairs
        return team_seasons_table, training_pairs