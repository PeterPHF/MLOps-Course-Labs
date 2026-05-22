"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

<<<<<<< HEAD
from pathlib import Path
import warnings
import joblib
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
=======
# TODO 1: Load your serialized churn model (and preprocessor if any) from data/
model = ...
>>>>>>> 0aae8b7e10987164221cf76b5e30ddfedf7c8b29

# TODO 1: Load your serialized churn model from data/model.joblib
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
MODEL_PATH = PROJECT_ROOT / "bin" / "model.pkl"
# PREPROCESSOR_PATH = PROJECT_ROOT / "preprocessor.pkl"

model = joblib.load(MODEL_PATH)
# preprocessor = joblib.load(PREPROCESSOR_PATH)

def preprocess(features: list[float]) -> list[float]:
    """
    Takes raw features and applies necessary preprocessing (e.g. scaling).
    """
    # TODO 2: Apply any preprocessing steps here (if applicable)
    return features


def predict_churn(features: list[float]) -> int:
    """
    Takes a list of raw feature values and returns a churn prediction (0 or 1).
    """
    # TODO 3: Preprocess the features
    processed_features = preprocess(features)
    
    # TODO 4: Use model.predict() on processed_features to get a prediction and return it as an int
    #         Hint: model.predict() expects a 2D array
    prediction = model.predict([features])
    return int(prediction[0])


if __name__ == "__main__":
<<<<<<< HEAD
    # TODO 3: Replace with sample features that match your model
    sample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.2]
=======
    # TODO 5: Replace with sample features that match your model
    sample = []
>>>>>>> 0aae8b7e10987164221cf76b5e30ddfedf7c8b29
    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
