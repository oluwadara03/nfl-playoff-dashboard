# Test if correct data - test.py

"""
This file tests that no 2026 NFL game data is being ued in the model training data.
"""

# Import libraries for system paths, file paths, and data handling
import sys
from pathlib import Path
import pandas as pd

# Add project root to import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import necessary data
from src.data_processor import PROCESSED_DIR, RAW_PATH, LAST_SEASON


# Ensure no NFL data from 2026 is included
def no_2026_games():
    assert pd.read_csv(RAW_PATH).season.max() <= LAST_SEASON
    assert pd.read_csv(PROCESSED_DIR / "team_seasons.csv").season.max() <= LAST_SEASON
    assert pd.read_csv(PROCESSED_DIR / "pairs.csv").season.max() < LAST_SEASON

# Confirm no games from 2026 are added
if __name__ == "__main__":
    no_2026_games()
    print("Test passed!")