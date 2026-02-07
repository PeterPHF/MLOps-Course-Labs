"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import tempfile
import os

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "column_transformer.pkl")
        pd.to_pickle(col_transf, path)
        mlflow.log_artifact(path, artifact_path="preprocessing")

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    signature = infer_signature(X_train, log_reg.predict(X_train))

    # Log model
    mlflow.sklearn.log_model(sk_model=log_reg,artifact_path="model",signature=signature,input_example=X_train.iloc[:1],)

    ### Log the data
    mlflow.log_param("model_type" , "LogisticRegression")

    return log_reg

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)

    signature = infer_signature(X_train, rf.predict(X_train))

    mlflow.sklearn.log_model(
        rf,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:1]
    )

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)

    return rf


from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(X_train, y_train):
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)

    signature = infer_signature(X_train, gb.predict(X_train))

    mlflow.sklearn.log_model(
        gb,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:1]
    )

    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.05)

    return gb


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    ### Set the experiment name
    mlflow.set_experiment("Churn_Prediction")

        ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run():

        df = pd.read_csv("D:\\ITI\\mlflow\\MLOps-Course-Labs\\dataset\\Churn_Modelling.csv") 
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the max_iter parameter
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("rebalance", True)

        model = train(X_train, y_train)

        y_pred = model.predict(X_test)

        ### Log metrics after calculating them
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        ### Log tag
        mlflow.set_tag("developer", "Begol Osama")
        mlflow.set_tag("stage", "research")

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()

        # Log the image as an artifact in MLflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = os.path.join(tmp_dir, "confusion_matrix.png")
            plt.savefig(img_path)
            mlflow.log_artifact(img_path, artifact_path="plots")

        plt.show()


    with mlflow.start_run(run_name="Random_Forest"):
        model = train_random_forest(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))


    with mlflow.start_run(run_name="Gradient_Boosting"):
        model = train_gradient_boosting(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))


if __name__ == "__main__":
    main()