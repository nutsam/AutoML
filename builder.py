from __future__ import annotations

import os
from typing import List

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from multilabel import MultilabelPredictor


def build_dataset(
    data_path: str,
    label_column: str,
    label_type: str,
    task_type: str,
    feature_list: List[str],
    categorical_features: List[str],
    selected_folds: List[int],
) -> TabularDataset:
    """
    Build a tabular dataset based on the provided data and configuration.

    Args:
        data_path (str): The path to the CSV file containing the dataset.
        label_column (str): column name representing the labels in the dataset.
        label_type (str): The type of the label. Should be one of ["binary", "multiclass", "multilabel"]
        task_type (str): The type of the prediction problem. Should be one of ["classification", "regression"].
        feature_list (List[str]): A list of column names representing the features to include in the dataset.
        categorical_features (List[str]): A list of column names representing categorical features.
        selected_folds (List[int]): A list of fold indices to include in the dataset.

    Returns:
        TabularDataset: A tabular dataset containing selected folds and specified features from the dataset.
    """

    assert os.path.exists(data_path), f"{data_path} does not exist"
    df = pd.read_csv(data_path)

    assert "fold_index" in df.columns, "fold_index column must be present in the dataframe."
    dataset = df[df["fold_index"].isin(selected_folds)]
    assert all(
        feature in dataset.columns for feature in feature_list
    ), "All features in feature_list must exist in the dataframe."
    dataset = dataset[feature_list + [label_column]]

    if label_type in ["multilabel", "multiple"]:
        label_length = len(dataset[label_column].iloc[0].strip("[]").split(","))
        new_label_column = [f"label_{i}" for i in range(label_length)]
        new_label_column = [f"label_{i}" for i in range(label_length)]
        dtype = float if task_type == "regression" else int
        dataset[new_label_column] = (
            dataset[label_column].str.strip("[]").str.split(",", expand=True).astype(dtype)
        )
        dataset = dataset.drop(columns=label_column)

    assert all(
        feature in dataset.columns for feature in categorical_features
    ), "All features in categorical_features must exist in the dataframe."
    for col in categorical_features:
        dataset[col] = dataset[col].astype("category")

    return TabularDataset(dataset)


def build_model(
    task_type: str,
    label_type: str,
    label_column: str,
    metric: str,
    output_dir: str,
    num_classes: int,
) -> TabularPredictor:
    """
    Build a tabular predictor model based on the specified configuration.

    Args:
        task_type (str): The type of the prediction problem. Should be one of ["classification", "regression"].
        label_type (str): The type of the label. Should be one of ["binary", "multiclass", "multilabel"]
        label_column (str): The column in the dataset containing the target variable.
        metric (str): The evaluation metric to be used during training and prediction.
        output_dir (str): The directory where the model and related files will be saved.
        num_classes (int): The number of classes for the `multilabel` task.

    Returns:
        TabularPredictor: An instance of the tabular predictor model.
    """
    if label_type in ["multilabel", "multiple"]:
        problem_type = "binary" if task_type == "classification" else "regression"
        predictor = MultilabelPredictor(
            labels=[f"label_{i}" for i in range(num_classes)],
            path=output_dir,
            problem_types=[problem_type] * num_classes,
            eval_metrics=[metric] * num_classes,
        )
    else:
        problem_type = "regression" if task_type == "regression" else label_type
        predictor = TabularPredictor(
            label=label_column, path=output_dir, problem_type=problem_type, eval_metric=metric
        )

    return predictor
