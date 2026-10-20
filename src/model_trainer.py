# Model Trainer - model_trainer.py

"""
This file trains and evaluates machine learning models to predict NFL playoff qualification and team wins.
It compares logistic regression and random forest models against a baseline, then saves the best model and 
evaluation results for use in the dashboard.
"""

# Importing the libraries used for training, tuning, evaluating, and saving the models
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Importing the shared feature list and model folder from the data processor
from src.data_processor import FEATURES, MODEL_DIR

# Random seed and feature constraints for a reproducible model
SEED = 42
MONOTONIC = {"win_pct": 1, "ppg": 1, "papg": -1, "diff_pg": 1, "made_playoffs": 1}


# ML pipeline
def model_pipeline(model, columns):
    # Scale the selected features before fitting the model
    return Pipeline([("prep", ColumnTransformer([("scale", StandardScaler(), columns)])), ("model", model)])


# Classification performance metrics
def calculate_scores(actual_values, predicted_values):
    # Calculate accuracy, precision, recall, and F1 score
    return {"Accuracy": accuracy_score(actual_values, predicted_values), "Precision": precision_score(actual_values, predicted_values, zero_division=0), "Recall": recall_score(actual_values, predicted_values, zero_division=0), "F1": f1_score(actual_values, predicted_values, zero_division=0)}
    
    

class ModelTrainer:
    
    # Constructor
    def __init__(self, training_pairs):
        # Stores training data
        self.training_pairs = training_pairs
        
    # Training and evaluation
    def run(self):
        # Latest season is the test set, split from remaining data
        test_season = self.training_pairs.season.max()
        training_data, test_data = self.training_pairs[self.training_pairs.season < test_season], self.training_pairs[self.training_pairs.season == test_season]

        # Playoff labels are used for training and testing
        training_labels, test_labels = training_data.next_playoffs.astype(int), test_data.next_playoffs.astype(int)

        # Monotonic direction for each feature
        monotonic_constraints = [MONOTONIC[feature_name] for feature_name in FEATURES]

        # Definitions for the model configurations for comparison
        model_families = {
            "Logistic Regression": (["diff_pg"], [(f"point diff only, C={regularization_strength}", lambda regularization_strength=regularization_strength: LogisticRegression(C=regularization_strength)) for regularization_strength in [0.1, 1, 10]]),
            "Random Forest (monotonic)": (FEATURES, [(f"200 trees, max depth {maximum_depth}", lambda maximum_depth=maximum_depth: RandomForestClassifier(200, max_depth=maximum_depth, min_samples_leaf=3, monotonic_cst=monotonic_constraints, random_state=SEED)) for maximum_depth in [2, 3, 4]]),
        }

        # Compare model configurations using four-fold cross-validation
        cross_validation = StratifiedKFold(4, shuffle=True, random_state=SEED)
        tuning_results, best_models = [], {}
        for model_family, (selected_columns, model_options) in model_families.items():
            for configuration_name, build_model in model_options:
                cross_validation_accuracy = cross_val_score(model_pipeline(build_model(), selected_columns), training_data[FEATURES], training_labels, cv=cross_validation, scoring="accuracy").mean()
                tuning_results.append({"Model": model_family, "Configuration": configuration_name, "CV accuracy": cross_validation_accuracy})
                if model_family not in best_models or cross_validation_accuracy > best_models[model_family][0]:
                    best_models[model_family] = (cross_validation_accuracy, configuration_name, build_model, selected_columns)

        # Assess the best configuration from each model family on the test set
        model_results, test_predictions = [], {}
        for model_family, (cross_validation_accuracy, configuration_name, build_model, selected_columns) in best_models.items():
            training_start_time = time.time()
            trained_model = model_pipeline(build_model(), selected_columns).fit(training_data[FEATURES], training_labels)
            training_duration = time.time() - training_start_time
            test_predictions[model_family] = trained_model.predict(test_data[FEATURES])
            model_results.append({"Model": model_family, "Configuration": configuration_name, **calculate_scores(test_labels, test_predictions[model_family]), "Training time (s)": training_duration, "_cv": cross_validation_accuracy})

        # Use the previous season's playoff status as a comparison baseline
        baseline_predictions = test_data["made_playoffs"].astype(int).to_numpy()
        model_results.append({"Model": "Naive: repeat last year", "Configuration": "No model", **calculate_scores(test_labels, baseline_predictions), "Training time (s)": 0.0, "_cv": np.nan})
        test_predictions["Naive: repeat last year"] = baseline_predictions
        comparison_table = pd.DataFrame(model_results)

        # Selecting best the model based on cross-validation accuracy
        best_model = comparison_table[comparison_table.Model.isin(best_models)].sort_values("_cv", ascending=False).iloc[0]["Model"]
        comparison_table = comparison_table.drop(columns="_cv")

        # Train a linear-regression model on point differential to measure how accurately wins can be projected
        wins_test_model = model_pipeline(LinearRegression(), ["diff_pg"]).fit(training_data[FEATURES], training_data.next_wins)
        wins_mean_absolute_error = float(np.abs(wins_test_model.predict(test_data[FEATURES]) - test_data.next_wins).mean())

        # Retrain the selected playoff model and wins model on all available data
        _, _, build_best_model, best_model_columns = best_models[best_model]
        final_playoff_model = model_pipeline(build_best_model(), best_model_columns).fit(self.training_pairs[FEATURES], self.training_pairs.next_playoffs.astype(int))
        final_wins_model = model_pipeline(LinearRegression(), ["diff_pg"]).fit(self.training_pairs[FEATURES], self.training_pairs.next_wins)

        # Extract feature coefficients or importances from the selected model
        selected_classifier = final_playoff_model.named_steps["model"]
        is_logistic_regression = hasattr(selected_classifier, "coef_")

        # Preparing results to be shown on the dashboard
        results = {
            "table": comparison_table,
            "best_name": best_model,
            "tuning": pd.DataFrame(tuning_results),
            "cm": confusion_matrix(test_labels, test_predictions[best_model]),
            "wins_mae": wins_mean_absolute_error,
            "weights": pd.Series(selected_classifier.coef_[0] if is_logistic_regression else selected_classifier.feature_importances_, index=best_model_columns),
            "weights_label": "coefficient (scaled)" if is_logistic_regression else "feature importance",
            "test_season": int(test_season),
            "errors": pd.DataFrame({"Team": test_data.team.to_numpy(), "Actual (made playoffs)": test_labels.to_numpy(), "Predicted": test_predictions[best_model]}).query("`Actual (made playoffs)` != Predicted"),
        }

        #  Save the trained models and evaluation results
        joblib.dump(final_playoff_model, MODEL_DIR / "best_model.pkl")
        joblib.dump(final_wins_model, MODEL_DIR / "wins_model.pkl")
        joblib.dump(results, MODEL_DIR / "results.pkl")
        
        # Return model evaluation results for the dashboard
        return results