"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

from pathlib import Path
import warnings
import joblib
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# TODO 1: Load your serialized churn model from data/model.joblib
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
MODEL_PATH = PROJECT_ROOT / "bin" / "model.pkl"
# PREPROCESSOR_PATH = PROJECT_ROOT / "preprocessor.pkl"

model = joblib.load(MODEL_PATH)
# preprocessor = joblib.load(PREPROCESSOR_PATH)

def predict_churn(features: list[float]) -> int:
    """
    Takes a list of feature values and returns a churn prediction (0 or 1).
    """
    # TODO 2: Use model.predict() to get a prediction and return it as an int
    #         Hint: model.predict() expects a 2D array
    prediction = model.predict([features])
    return int(prediction[0])


if __name__ == "__main__":
    # TODO 3: Replace with sample features that match your model
    sample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.2]
    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
