# Bank Consumer Churn Prediction

This project demonstrates the use of MLflow for tracking experiments, logging artifacts, and managing machine learning models. The goal of this project is to predict bank customer churn using a logistic regression model.

## Project Structure

```
MLOps-Course-Labs/
├── README.md
├── requirements.txt
├── dataset/
│   ├── Churn_Modelling.csv
│   └── note.txt
├── src/
│   ├── train.py
│   └── train2.py
```

- **README.md**: This file provides an overview of the project.
- **requirements.txt**: Contains the list of dependencies required to run the project.
- **dataset/**: Contains the dataset used for training and testing the model.
- **src/**: Contains the source code for preprocessing, training, and logging with MLflow.

## Features

- **Data Preprocessing**: Handles data balancing, scaling, and encoding.
- **Model Training**: Trains a logistic regression model to predict customer churn.
- **MLflow Integration**:
  - Tracks experiments and logs parameters, metrics, and artifacts.
  - Logs the preprocessor and trained model as artifacts.
  - Logs evaluation metrics such as accuracy, precision, recall, and F1-score.
  - Logs confusion matrix as an artifact.

## Getting Started

### Prerequisites

- Python 3.12
- MLflow
- Required Python libraries (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MLOps-Course-Labs
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. Start the MLflow tracking server:
   ```bash
   mlflow ui
   ```
   This will start the MLflow UI at `http://localhost:5000`.

2. Run the training script:
   ```bash
   python src/train.py
   ```

3. View the experiment results and artifacts in the MLflow UI.

## Dataset

The dataset used in this project is `Churn_Modelling.csv`, which contains information about bank customers and whether they exited the bank. The dataset includes features such as:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited (target variable)

## MLflow Tracking

This project uses MLflow to:

- Log parameters such as `max_iter`.
- Log metrics including accuracy, precision, recall, and F1-score.
- Log artifacts such as the preprocessor, trained model, and confusion matrix.
- Track experiments and runs for better reproducibility and model management.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
