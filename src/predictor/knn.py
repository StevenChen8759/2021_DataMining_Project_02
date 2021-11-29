import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def iris_classify_knn(input_df: pd.DataFrame, k_value: int) -> pd.DataFrame:
    # Copy Data ID
    output_df = input_df[["Id"]].copy()

    # Get KNeighborsClassifier object
    knn = KNeighborsClassifier(n_neighbors=k_value)

    # Train KNeighborsClassifier
    knn.fit(
        input_df[
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ]
        ],
        input_df["Species_num"]
    )

    # Predict Iris by KNeighborsClassifier
    predict_res = knn.predict(
        input_df[
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ]
        ],
    )

    # Append to output dataframe and return
    output_df["Classified_Species_num"] = pd.Series(predict_res)

    return output_df
