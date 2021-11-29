import numpy as np
import pandas as pd
from loguru import logger


def iris_classify_heuristic_v1(input_df: pd.DataFrame) -> pd.DataFrame:

    logger.debug("Iris-Heuristic V1. - Single Feature Threshold")

    # Prepare Output Dataframe
    output_df = input_df[["Id"]].copy()
    output_df["Classified_Species_num"] = np.nan

    # Query dataframe to classify Iris-Setosa
    temp_df = input_df.query("PetalLengthCm < 2.2")
    output_df.loc[temp_df.index, 'Classified_Species_num'] = 0

    # Filter classified data
    layer2_query = input_df.query("PetalLengthCm >= 2.2")

    # Query dataframe to classify Iris-Versicolor
    temp_df = layer2_query.query("PetalWidthCm <= 1.8")
    output_df.loc[temp_df.index, 'Classified_Species_num'] = 1

    # The rest of field is Iris-Virginica, replace all nan to specific label
    output_df.fillna(2, inplace=True)

    return output_df


def iris_classify_heuristic_v2(input_df: pd.DataFrame) -> pd.DataFrame:

    logger.debug("Iris-Heuristic V2. Multiple Feature Inequation.")

    # Prepare Output Dataframe
    output_df = input_df[["Id"]].copy()
    output_df["Classified_Species_num"] = np.nan

    # Query dataframe to classify Iris-Setosa
    temp_df = input_df.query("PetalLengthCm < 2.2")
    output_df.loc[temp_df.index, 'Classified_Species_num'] = 0

    # Filter classified data
    layer2_query = input_df.query("PetalLengthCm >= 2.2")

    # Query dataframe to classify Iris-Versicolor by two-feature inequation
    temp_df = layer2_query.query("PetalLengthCm * 0.5867 + PetalWidthCm <= 4.4")
    output_df.loc[temp_df.index, 'Classified_Species_num'] = 1

    # The rest of field is Iris-Virginica, replace all nan to specific label
    output_df.fillna(2, inplace=True)

    return output_df


