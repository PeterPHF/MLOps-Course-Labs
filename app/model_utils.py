"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

# TODO 1: Load your serialized churn model from data/model.joblib
model = ...


def predict_churn(features: list[float]) -> int:
    """
    Takes a list of feature values and returns a churn prediction (0 or 1).
    """
    # TODO 2: Use model.predict() to get a prediction and return it as an int
    #         Hint: model.predict() expects a 2D array
    pass


if __name__ == "__main__":
    # TODO 3: Replace with sample features that match your model
    sample = []
    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
