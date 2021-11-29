from loguru import logger
import pandas as pd

from utils import mkdir_conditional
from analyzer.dataset_analysis import iris_pre_analyze
from analyzer.classifier_analysis import iris_result_analysis

from predictor.heuristic_decision import iris_classify_heuristic_v1, iris_classify_heuristic_v2
from predictor.decision_tree import iris_classify_decision_tree
from predictor.knn import iris_classify_knn
from predictor.naive_bayes import iris_classify_nb
from predictor.svm import iris_classify_svm

if __name__ == "__main__":

    # Initialize - prepare output dir
    logger.info("Initialize...")
    mkdir_conditional("./output")

    logger.info("Load Datasets")
    iris_data = pd.read_csv("./dataset/IRIS/Iris.csv")

    logger.info("Pre-analyze datasets")
    iris_pre_analyze(iris_data)

    logger.info("Run classification model fitting and prediction")
    iris_heur_v1_result = iris_classify_heuristic_v1(iris_data)
    iris_heur_v2_result = iris_classify_heuristic_v2(iris_data)

    iris_dt_result = iris_classify_decision_tree(iris_data)

    iris_3nn_result = iris_classify_knn(iris_data, 3)
    iris_1nn_result = iris_classify_knn(iris_data, 1)
    iris_nb_result = iris_classify_nb(iris_data)
    iris_svm_result = iris_classify_svm(iris_data)

    iris_result_analysis(
        iris_data["Species_num"],
        iris_heur_v1_result["Classified_Species_num"],
        "Heuristic_V1",
    )

    iris_result_analysis(
        iris_data["Species_num"],
        iris_heur_v2_result["Classified_Species_num"],
        "Heuristic_V2",
    )

    iris_result_analysis(
        iris_data["Species_num"],
        iris_dt_result["Classified_Species_num"],
        "Decision_Tree",
    )

    iris_result_analysis(
        iris_data["Species_num"],
        iris_3nn_result["Classified_Species_num"],
        "knn_k=3",
    )

    iris_result_analysis(
        iris_data["Species_num"],
        iris_1nn_result["Classified_Species_num"],
        "knn_k=1",
    )

    iris_result_analysis(
        iris_data["Species_num"],
        iris_nb_result["Classified_Species_num"],
        "Naive_Bayes",
    )

    iris_result_analysis(
        iris_data["Species_num"],
        iris_svm_result["Classified_Species_num"],
        "svm",
    )
