AutoML with AutoGluon for Classification/Regression Prediction
=====================================================

Overview
--------

This repository contains code for training a machine learning model with different type of task using [AutoGluon](https://autogluon.mxnet.io/). AutoGluon is an open-source AutoML library that automates the process of training and deploying machine learning models.

Table of Contents
-----------------

*   [Requirements](#requirements)
*   [Usage](#usage)
    *   [Configuration](#configuration)
    *   [Training](#training)
*   [Dataset](#dataset)
*   [Model](#model)
*   [Evaluation Metrics](#evaluation-metrics)
*   [Results](#results)
*   [Contributing](#contributing)
*   [License](#license)

Requirements
------------

Make sure you have the following dependencies installed:

*   Python 3.x
*   AutoGluon

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Usage
-----

### Configuration

Adjust the configuration in the `config.yaml` file to specify the task type, label type, model name, and other parameters for your AutoGluon training.


```yaml
# Add your configuration details here
var:
  task_type: "classification"
  label_type: "binary"
  model_name: "XT"
  num_classes: 1
  feature_list: ["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
  categorical_features: ["Sex", "Parch", "Embarked"]
  categorical_dimensions: [2, 2, 3]
  label_column: "Survived"
  default_root_path: "./output"
  weighted_ensemble: True
  presets: best_quality
  exp_name: "exp"

dataset:
  # Add dataset details here
  ...

model:
  # Add model details here
  ...

metrics:
  # Add metrics details here
  ...
```

### Training

To train the model, run the following command:


```bash
python train.py -o yaml/exp_config/model_zoo/your_config.yaml
```

This will initiate the training process using AutoGluon based on the configurations provided.

Dataset
-------

The dataset used for training is located in the `dataset` folder. Ensure that you have the necessary data files, including `annotation_train.csv`, before running the training script.

Model
-----

The trained model will be saved in the specified output path. You can find the model files and logs in this directory.

Evaluation Metrics
------------------

The model is evaluated based on various metrics, including accuracy, AUROC, average precision, F1 score, precision, recall, and specificity.

Results
-------

Details about the training process, results, and model performance can be found in the output directory.


License
-------

This project is licensed under the [Apache License 2.0](LICENSE).
