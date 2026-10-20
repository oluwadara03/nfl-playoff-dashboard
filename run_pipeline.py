# Run Pipeline - run_pipeline.py

"""
This file runs the complete NFL Playoff Predictor pipeline, from processing the game data to training 
and evaluating the machine learning models, and saving the results.
It saves the processed data, trained models, and evaluation results needed by the dashboard.
"""

# Import the data processing and model training classes
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer


# Run the complete NFL Playoff Predictor pipeline
def run():
    # Process the game data and prepare the training pairs
    team_seasons_table, training_pairs = DataProcessor().run()

    # Training and evaluating the models
    return ModelTrainer(training_pairs).run()


# Main method, runs the pipeline
if __name__ == "__main__":
    # Print the model comparison and selected model details when the pipeline is run directly
    results = run()
    print(results["table"].round(3).to_string(index=False))
    
    # Show the best model, and wins/losses prediction errors
    print("Best model:", results["best_name"], "| wins-model test Mean Absolute Error:", round(results["wins_mae"], 2))