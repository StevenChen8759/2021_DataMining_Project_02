import numpy as np
import pandas as pd
from sklearn.svm import SVC


def iris_classify_svm(input_df: pd.DataFrame) -> pd.DataFrame:
    # Copy Data ID
    output_df = input_df[["Id"]].copy()

    # Get Support Vector Classifier (SVC) object
    svc = SVC()

    # Train SVC
    svc.fit(
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

    # Predict Iris by SVC
    predict_res = svc.predict(
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
