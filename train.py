from __future__ import annotations

import argparse
import json
import os
import pprint

import numpy as np

import yaml
from builder import build_dataset, build_model
from metrics import EvaluationMetrics
from utilities.yaml_loader import LOADER


def get_config_variable(config, section, variable, default=None):
    return config.get(section, {}).get(variable, default)


def train(cfg, unparsed=[], **kwargs):
    # 1. Parse the configuration
    var_config = cfg.get("var")
    exp_config = cfg.get("exp")
    dataset_config = cfg.get("dataset")
    model_config = cfg.get("model")

    # var
    task_type = var_config.get("task_type")
    label_type = var_config.get("label_type")
    model_name = var_config.get("model_name")
    num_classes = var_config.get("num_classes")
    feature_list = var_config.get("feature_list")
    categorical_features = var_config.get("categorical_features")
    label_column = var_config.get("label_column")

    # advanced
    weighted_ensemble = var_config.get("weighted_ensemble", False)
    presets = var_config.get("presets", "best_quality")

    # exp
    exp_name = exp_config.get("exp_name", "exp")
    default_root_path = exp_config.get("default_root_path")
    default_root_dir = exp_config.get("default_root_dir")

    # dataset
    dataset_folder = dataset_config.get("dataset_folder")
    annotation_filename = dataset_config.get("annotation_filename")
    train_selected_folds = dataset_config.get("train").get("selected_folds", [1, 2, 3, 4, 5])
    val_selected_folds = dataset_config.get("val").get("selected_folds", [5])

    # model
    metric_monitors = model_config.get("metrics").get("monitors")
    metric_evaluation = model_config.get("metrics").get("evaluation")

    # 2. Build the dataset
    data_path = os.path.join(dataset_folder, annotation_filename)
    train_data = build_dataset(
        data_path,
        label_column,
        label_type,
        task_type,
        feature_list,
        categorical_features,
        train_selected_folds,
    )
    val_data = build_dataset(
        data_path,
        label_column,
        label_type,
        task_type,
        feature_list,
        categorical_features,
        val_selected_folds,
    )

    # 3. Build the model
    output_model_path = os.path.join(default_root_path, default_root_dir + "_exp", exp_name)
    os.makedirs(output_model_path, exist_ok=True)
    predictor = build_model(
        task_type, label_type, label_column, metric_monitors, output_model_path, num_classes
    )
    predictor.fit(
        train_data,
        val_data,
        use_bag_holdout=True,
        fit_weighted_ensemble=weighted_ensemble,
        presets=presets,
        hyperparameters={model_name: {}},
    )

    # 4. Evaluation
    label_column = (
        [f"label_{i}" for i in range(num_classes)]
        if label_type in ["multilabel", "multiple"]
        else label_column
    )
    y_true = val_data[label_column].values

    if task_type == "regression":
        y_prob = predictor.predict(val_data).values
    elif label_type == "binary":
        y_prob = predictor.predict_proba(val_data, as_multiclass=False).values
    elif label_type == "multiclass":
        y_prob = predictor.predict_proba(val_data, as_multiclass=True).values
    elif label_type == "multilabel":
        y_prob = np.array(list(predictor.predict_proba(val_data).values())).T

    metrics_report = EvaluationMetrics.calculate_metrics(y_true, y_prob, metric_evaluation)
    output_metric_path = os.path.join(
        default_root_path, default_root_dir + "_exp", exp_name, "metrics"
    )
    os.makedirs(output_metric_path, exist_ok=True)
    with open(os.path.join(output_metric_path, "metrics_report.json"), "w") as f:
        json.dump(metrics_report, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--config-path", type=str, help="Yaml name")
    args, unparsed = parser.parse_known_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=LOADER)

    pprint.pprint(cfg)

    train(cfg, unparsed=unparsed, **vars(args))


if __name__ == "__main__":
    main()
