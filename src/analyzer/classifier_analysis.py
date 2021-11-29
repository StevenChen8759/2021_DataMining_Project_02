import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix, classification_report

def iris_result_analysis(
    y_ground_truth: pd.DataFrame,
    y_prediction_outcome: pd.DataFrame,
    model_name: str,
):
    logger.debug(f"Iris Classification Result Analysis, Model: {model_name}")

    cfmx = confusion_matrix(
        y_ground_truth,
        y_prediction_outcome,
    )

    logger.debug(f"Confusion Matrix:\n(Iris-setosa | Iris-versicolor | Iris-virginica):\n{cfmx}")

    crpt = classification_report(
        y_ground_truth,
        y_prediction_outcome,
    )

    logger.debug(f"Classification Report\n(0: Iris-setosa | 1: Iris-versicolor | 2: Iris-virginica):\n{crpt}")
