# Visual Dashboard - dashboard.py

"""
This file creates an interactive dashboard for exploring NFL team statistics, viewing playoff predictions,
and evaluating model performance.
It displays data from the 2023, 2024 and 2025 seasons and uses the trained models to generate predictions 
for the 2026 NFL season.

This file was created with some help from AI (Claude), not the entirety. Mainly, the layout, some graphs 
and the logic to organize things properly.
"""

# Importing important packages and libraries
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_processor import FEATURES, MODEL_DIR, PROCESSED_DIR
from src.predictor import Predictor
# Configure the dashboard to use the full browser width
st.set_page_config(page_title="NFL Playoff Predictor", layout="wide")

# Check for the trained model and run the full pipeline automatically if this is the first launch
if not (MODEL_DIR / "best_model.pkl").exists():
    from run_pipeline import run
    with st.spinner("First run: loading nflverse data and training models..."):
        run()


# Cache the processed tables so switching dashboard pages does not repeatedly read the CSV files
@st.cache_data
def load_tables():
    return pd.read_csv(PROCESSED_DIR / "team_seasons.csv"), pd.read_csv(PROCESSED_DIR / "pairs.csv")


# Cache the loaded model objects so they remain available while the dashboard is running
@st.cache_resource
def load_predictor():
    return Predictor()


# Load the processed team-season data, training pairs, and saved model evaluation results
team_seasons_table, training_pairs = load_tables()
model_results = joblib.load(MODEL_DIR / "results.pkl")
selected_page = st.sidebar.radio("Page", ["Data overview", "Explore", "Predict", "Model performance"])
st.title("NFL Playoff Predictor: 2026 Season")
st.caption("Trained on nflverse games from the 2023, 2024 and 2025 seasons. No 2026 games are used.")



# Data Overview page

# Show the dataset counts, cleaning report, summary statistics, and playoff totals
if selected_page == "Data overview":
    cleaning_report = json.loads((PROCESSED_DIR / "cleaning_report.json").read_text())
    first_column, second_column, third_column, fourth_column = st.columns(4)
    first_column.metric("Games used", cleaning_report["clean_shape"][0])
    second_column.metric("Team-seasons", len(team_seasons_table))
    third_column.metric("Training pairs", len(training_pairs))
    fourth_column.metric("Playoff teams / season", 14)
    st.write(f"Games per season (regular season + playoffs): {cleaning_report['games_per_season']}. Missing scores: {cleaning_report['missing_scores']}, duplicates: {cleaning_report['duplicate_games']}, outliers found: {cleaning_report['outliers_kept_diff_pg_z_gt_3']} (real values, kept).")
    st.subheader("Team-season table (preview)")
    st.dataframe(team_seasons_table.drop(columns=["pf", "pa"]).head(20))
    st.subheader("Summary statistics")
    st.dataframe(team_seasons_table[["wins", "win_pct", "ppg", "papg", "diff_pg", "made_playoffs"]].describe())
    st.subheader("Target: teams that made the playoffs")
    figure, axes = plt.subplots()
    playoff_counts = team_seasons_table.groupby("season").made_playoffs.sum()
    axes.bar(playoff_counts.index.astype(str), playoff_counts.values, color="steelblue")
    axes.set(title="Playoff teams per season", xlabel="Season", ylabel="Teams")
    st.pyplot(figure)



# Explore page

# Let users filter the data and explore the relationship between season statistics and later outcomes
elif selected_page == "Explore":
    st.sidebar.header("Filters")
    available_seasons = sorted(training_pairs.season.unique())
    selected_seasons = st.sidebar.multiselect("Stats season", available_seasons, available_seasons)
    previous_playoff_status = st.sidebar.multiselect("Made playoffs that season?", [0, 1], [0, 1])
    filtered_data = training_pairs[training_pairs.season.isin(selected_seasons) & training_pairs.made_playoffs.isin(previous_playoff_status)]
    st.write(f"**{len(filtered_data)}** team-seasons match (stats from season N, outcome in season N+1)")
    feature_display_names = {"win_pct": "Win %", "ppg": "Points for / game", "papg": "Points against / game", "diff_pg": "Point differential / game"}

    first_column, second_column = st.columns(2)
    
    # Chart 1: histogram split by next-season outcome
    with first_column:   
        selected_histogram_feature = st.selectbox("Histogram feature", list(feature_display_names), format_func=feature_display_names.get)
        figure, axes = plt.subplots()
        axes.hist([filtered_data[filtered_data.next_playoffs == 0][selected_histogram_feature], filtered_data[filtered_data.next_playoffs == 1][selected_histogram_feature]], bins=12, label=["Missed playoffs next year", "Made playoffs next year"], color=["tab:gray", "tab:green"])
        axes.set(title=f"{feature_display_names[selected_histogram_feature]} by next-season outcome", xlabel=feature_display_names[selected_histogram_feature], ylabel="Teams")
        axes.legend()
        st.pyplot(figure)
        
    # Chart 2: scatter vs next-season wins
    with second_column: 
        selected_scatter_feature = st.selectbox("Scatter x-axis", list(feature_display_names), index=3, format_func=feature_display_names.get)
        figure, axes = plt.subplots()
        for playoff_outcome, color, label in [(0, "tab:gray", "Missed"), (1, "tab:green", "Made")]:
            filtered_outcome_data = filtered_data[filtered_data.next_playoffs == playoff_outcome]
            axes.scatter(filtered_outcome_data[selected_scatter_feature], filtered_outcome_data.next_wins, c=color, alpha=0.7, label=f"{label} playoffs next year")
        axes.set(title=f"{feature_display_names[selected_scatter_feature]} vs next-season wins", xlabel=feature_display_names[selected_scatter_feature], ylabel="Next-season wins")
        axes.legend()
        st.pyplot(figure)

    third_column, fourth_column = st.columns(2)
    with third_column:   # Chart 3: box plot
        selected_boxplot_feature = st.selectbox("Box plot feature", list(feature_display_names), index=3, format_func=feature_display_names.get)
        figure, axes = plt.subplots()
        axes.boxplot([filtered_data[filtered_data.next_playoffs == 0][selected_boxplot_feature], filtered_data[filtered_data.next_playoffs == 1][selected_boxplot_feature]])
        axes.set_xticklabels(["Missed next year", "Made next year"])
        axes.set(title=f"{feature_display_names[selected_boxplot_feature]} by next-season outcome", ylabel=feature_display_names[selected_boxplot_feature])
        st.pyplot(figure)
    with fourth_column:   # Chart 4: correlation heatmap
        columns = FEATURES + ["next_wins", "next_playoffs"]
        correlation_matrix = filtered_data[columns].corr().to_numpy()
        figure, axes = plt.subplots()
        heatmap = axes.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axes.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
        axes.set_yticks(range(len(columns)), columns)
        for row_index in range(len(columns)):
            for column_index in range(len(columns)):
                axes.text(column_index, row_index, f"{correlation_matrix[row_index, column_index]:.2f}", ha="center", va="center", fontsize=7)
        figure.colorbar(heatmap)
        axes.set_title("Correlation heatmap")
        st.pyplot(figure)
        
        

# Prediction page

# Show the 2026 predictions and allow users to test their own custom team profile
elif selected_page == "Predict":
    predictor = load_predictor()
    playoff_outlook, latest_season_teams = predictor.outlook()
    st.subheader("2026 outlook for all 32 teams (based on 2025 season stats)")
    st.dataframe(playoff_outlook.style.format({"P(playoffs 2026)": "{:.1%}", "Projected wins": "{:.1f}", "Projected losses": "{:.1f}"}), height=400)
    outlook_sorted_for_chart = playoff_outlook.iloc[::-1]
    figure, axes = plt.subplots(figsize=(7, 8))
    axes.barh(outlook_sorted_for_chart.Team, outlook_sorted_for_chart["P(playoffs 2026)"], color=np.where(outlook_sorted_for_chart.Rank <= 14, "tab:green", "tab:gray"))
    axes.set(title="Predicted probability of making the 2026 playoffs (green = top 14)", xlabel="Probability", ylabel="Team")
    st.pyplot(figure)

    st.subheader("What-if: try your own team")
    selected_team = st.selectbox("Start from team (2025 stats)", list(latest_season_teams.team))
    selected_team_row = latest_season_teams[latest_season_teams.team == selected_team].iloc[0]
    first_column, second_column = st.columns(2)
    win_percentage = first_column.slider("Win %", 0.0, 1.0, float(round(selected_team_row.win_pct, 2)), 0.01, key=f"w{selected_team}")
    points_per_game = first_column.slider("Points for / game", 10.0, 35.0, float(round(selected_team_row.ppg, 1)), 0.1, key=f"f{selected_team}")
    points_allowed_per_game = second_column.slider("Points against / game", 10.0, 35.0, float(round(selected_team_row.papg, 1)), 0.1, key=f"a{selected_team}")
    made_playoffs_last_season = second_column.checkbox("Made playoffs last season", bool(selected_team_row.made_playoffs), key=f"m{selected_team}")
    if st.button("Predict"):
        playoff_probability, projected_wins = predictor.predict(win_percentage, points_per_game, points_allowed_per_game, made_playoffs_last_season)
        first_column, second_column, third_column = st.columns(3)
        first_column.metric("P(makes playoffs)", f"{playoff_probability:.1%}")
        second_column.metric("P(misses playoffs)", f"{1 - playoff_probability:.1%}")
        third_column.metric("Projected record", f"{projected_wins:.1f} - {17 - projected_wins:.1f}")



# Model Performance page

# Show model evaluation, confusion matrix, feature weights, and tuning results
else:
    test_season = model_results["test_season"]
    st.subheader(f"Model comparison (test set: {test_season} stats -> {test_season + 1} outcome)")
    st.dataframe(model_results["table"].round(3))
    st.success(f"Selected model (best cross-validation score): {model_results['best_name']}. Projected-wins model test MAE: {model_results['wins_mae']:.2f} wins.")
    first_column, second_column = st.columns(2)
    with first_column:
        confusion_matrix_values = model_results["cm"]
        figure, axes = plt.subplots()
        axes.imshow(confusion_matrix_values, cmap="Blues")
        for row_index in range(2):
            for column_index in range(2):
                axes.text(column_index, row_index, confusion_matrix_values[row_index, column_index], ha="center", va="center", fontsize=16)
        axes.set_xticks([0, 1], ["Missed", "Made"])
        axes.set_yticks([0, 1], ["Missed", "Made"])
        axes.set(title="Confusion matrix (test set)", xlabel="Predicted", ylabel="Actual")
        st.pyplot(figure)
    with second_column:
        sorted_model_weights = model_results["weights"].sort_values()
        figure, axes = plt.subplots()
        axes.barh(sorted_model_weights.index, sorted_model_weights.values, color=np.where(sorted_model_weights.values > 0, "tab:green", "tab:red"))
        axes.set(title=f"Selected model: {model_results['weights_label']}", xlabel="Weight")
        st.pyplot(figure)
    st.subheader("Error analysis: test-set teams the selected model got wrong")
    error_table = model_results["errors"].assign(Type=lambda error_rows: np.where(error_rows.Predicted == 1, "False positive", "False negative"))
    st.dataframe(error_table.reset_index(drop=True))
    st.subheader("Hyperparameter tuning (4-fold CV on training seasons)")
    st.dataframe(model_results["tuning"].round(3))
    st.caption("The test set has only 32 teams, so each team moves accuracy by about 3 points and differences between models are within noise.")