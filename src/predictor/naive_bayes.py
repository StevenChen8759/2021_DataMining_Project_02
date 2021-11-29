import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB


def iris_classify_nb(input_df: pd.DataFrame) -> pd.DataFrame:
    # Copy Data ID
    output_df = input_df[["Id"]].copy()

    # Get CategoricalNB object
    cnb = CategoricalNB()

    # Train CategoricalNB
    cnb.fit(
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

    # Predict Iris by CategoricalNB
    predict_res = cnb.predict(
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
