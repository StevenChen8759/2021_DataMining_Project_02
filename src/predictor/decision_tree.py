import pandas as pd
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


def iris_classify_decision_tree(input_df: pd.DataFrame) -> pd.DataFrame:

    # Copy Data ID
    output_df = input_df[["Id"]].copy()

    # Get DecisionTreeClassifier object
    dtc = DecisionTreeClassifier()

    # Train DecisionTreeClassifier
    dtc.fit(
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

    # Predict Iris by DecisionTreeClassifier
    predict_res = dtc.predict(
        input_df[
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ]
        ],
    )

    dtc_graph = tree.export_graphviz(
        dtc,
        out_file=None,
        feature_names=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
        ],
        class_names=[
            "Iris-setosa",
            "Iris-versicolor",
            "Iris-virginica",
        ],
        filled=True
    )

    graph = graphviz.Source(dtc_graph, format="jpg")
    graph.render(filename="decision_tree_rule", directory="./output/iris_analyze")

    # Append to output dataframe and return
    output_df["Classified_Species_num"] = pd.Series(predict_res)

    return output_df
