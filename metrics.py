from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
import torchmetrics


class EvaluationMetrics:
    @staticmethod
    def calculate_metrics(
        y_true: Union[torch.Tensor, np.ndarray],
        y_prob: Union[torch.Tensor, np.ndarray],
        cfg_metrics: List[dict],
        device: str = "cpu",
    ) -> dict:
        """
        Calculate evaluation metrics based on the provided configuration.

        Args:
            y_true (torch.Tensor or np.ndarray): Ground truth labels.
            y_prob (torch.Tensor or np.ndarray): Predicted probabilities.
            cfg_metrics (List[dict]): Configuration for evaluation metrics.
            device (str): Device for tensor operations ("cuda" or "cpu").

        Returns:
            dict: Dictionary containing computed metric results.
        """
        y_true = torch.tensor(y_true, device=device)
        y_prob = torch.tensor(y_prob, device=device)

        metrics_report = {}
        for metric_config in cfg_metrics:
            metric_name = metric_config["name"]
            log_name = metric_config["log_name"]
            args = metric_config["args"]
            metric_class = getattr(torchmetrics, metric_name.split(".")[-1])
            metric = metric_class(**args)
            result = metric(y_prob, y_true)
            if isinstance(result, list) or (
                isinstance(result, torch.Tensor) and result.numel() > 1
            ):
                result = [x.cpu().detach().numpy().tolist() for x in result]
                for i in range(len(result)):
                    metrics_report[f"{log_name}_{i}"] = result[i]
            elif isinstance(result, torch.Tensor):
                result = result.cpu().detach().numpy().tolist()
                metrics_report[log_name] = result

        return metrics_report
