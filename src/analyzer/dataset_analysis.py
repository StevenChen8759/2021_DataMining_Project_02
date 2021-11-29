from re import T
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.figure_factory as ff
from loguru import logger
from plotly.subplots import make_subplots

from utils import mkdir_conditional


def iris_pre_analyze(
    iris_dataset: pd.DataFrame,
) -> None:

    logger.debug("Iris Dataset Pre-analysis")
    mkdir_conditional("./output/iris_analyze")

    #====================================================================================

    logger.debug(f"Data description:\n{iris_dataset.describe().drop(columns='Id')}")
    logger.debug(f"Species labels: {iris_dataset['Species'].unique()}")

    #====================================================================================

    logger.debug("Generate Class-integer mapping")
    species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    iris_dataset['Species_num'] = iris_dataset['Species'].map(species_map)

    logger.debug(f"Pearson Correlation Matrix:\n{iris_dataset.corr()[['Species_num']]}")

    #====================================================================================

    logger.debug("Plot Pearson Correlation Heatmap")
    fig = ff.create_annotated_heatmap(
        np.around(iris_dataset.drop(columns="Id").corr().drop(index="Species_num")[["Species_num"]].values, 2),
        colorscale='Viridis',
        x=["Species_num"],
        y=["SpealLengthCm", "SepalWidthCm",  "PetalLengthCm",  "PetalWidthCm"],
    )
    fig.update_layout(
        title_text="Pearson Correlation Heatmap of Iris Dataset Features",
        title_x=0.5,
        showlegend=False,
    )

    fig.write_image("./output/iris_analyze/iris_heatmap.jpg")

    #====================================================================================

    logger.debug("Plot Data-Label Distribution")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm")
    )

    color_map = {0: 'red', 1: 'blue', 2: 'green'}
    color_list = [color_map[iris_dataset['Species_num'][iterator]] for iterator in range(len(iris_dataset))]

    iris_features = [
        ['SepalLengthCm', 'SepalWidthCm'],
        ['PetalLengthCm', 'PetalWidthCm'],
    ]

    for i in range(2):
        for j in range(2):
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(len(iris_dataset))],
                    y=iris_dataset[iris_features[i][j]].values,
                    mode='markers',
                    marker_color=color_list,
                    name=iris_features[i][j],
                ),
                row=i + 1,
                col=j + 1,
            )

    for suffix in [""] + [f"{i}" for i in range(2,5)]:
        fig['layout'][f'xaxis{suffix}']['title'] = "Index"
        fig['layout'][f'yaxis{suffix}']['title'] = "Value"

    fig.update_layout(
        title_text="Data Distribution Plot<br>Red: Iris-setosa, Blue: Iris-versicolor, Green: Iris-virginica",
        title_xanchor='center',
        title_x=0.5,
        showlegend=False,
    )

    fig.write_image("./output/iris_analyze/data_distribution_index_value.jpg")

    #====================================================================================

    logger.debug("Plot Data-Label Distribution on Petal Distribution, without iris-setosa")
    fig = go.Figure(
        layout_xaxis_range=[2.7, 7.3],
        layout_yaxis_range=[0.7, 2.9]
    )

    iris_dataset_no_setosa = iris_dataset.query("Species_num != 0").reset_index(drop=True)

    color_list = [color_map[iris_dataset_no_setosa['Species_num'][iterator]] for iterator in range(len(iris_dataset_no_setosa))]

    fig.add_trace(
        go.Scatter(
            x=iris_dataset_no_setosa[iris_features[1][0]],
            y=iris_dataset_no_setosa[iris_features[1][1]],
            mode='markers',
            marker_color=color_list,
            name="Petal",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 7.5],
            y=[4.4, 0],
            mode='lines+markers',
            name="Split",
        ),
    )

    fig['layout']['xaxis']['title'] = iris_features[1][0]
    fig['layout']['yaxis']['title'] = iris_features[1][1]

    fig.update_layout(
        title_text="Data Distribution Plot on Petal Features<br>Blue: Iris-versicolor, Green: Iris-virginica",
        title_xanchor='center',
        title_x=0.5,
        showlegend=True,
    )

    fig.write_image("./output/iris_analyze/data_distribution_petal_len_width.jpg")


    #====================================================================================

    logger.debug("Plot Data-Label Distribution on Sepal Distribution, without iris-setosa")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=iris_dataset_no_setosa[iris_features[0][0]],
            y=iris_dataset_no_setosa[iris_features[0][1]],
            mode='markers',
            marker_color=color_list,
            name="Petal",
        ),
    )

    fig['layout']['xaxis']['title'] = iris_features[0][0]
    fig['layout']['yaxis']['title'] = iris_features[0][1]

    fig.update_layout(
        title_text="Data Distribution Plot on Sepal Features<br>Blue: Iris-versicolor, Green: Iris-virginica",
        title_xanchor='center',
        title_x=0.5,
        showlegend=False,
    )

    fig.write_image("./output/iris_analyze/data_distribution_sepal_len_width.jpg")
