from typing import (
    Any,
    Optional,
)

import numpy as np
import torch

from monai.utils.type_conversion import convert_to_numpy
from scipy import ndimage
from skimage.morphology import skeletonize_3d
from torch import Tensor
from torchmetrics import classification
from torchmetrics.metric import Metric


class BinaryRecall(classification.BinaryRecall):
    def __init__(self, sigmoid: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sigmoid = sigmoid

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.sigmoid:
            preds = torch.sigmoid(preds)
        return super().update(preds, target)


class BinaryPrecision(classification.BinaryPrecision):
    def __init__(self, sigmoid: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sigmoid = sigmoid

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.sigmoid:
            preds = torch.sigmoid(preds)
        return super().update(preds, target)


class Dice(classification.Dice):
    def __init__(self, sigmoid: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sigmoid = sigmoid

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.sigmoid:
            preds = torch.sigmoid(preds)
        return super().update(preds, target)


class ConfusionMatrix(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        slack: int = 3,
        pred_threshold: int = 3,
        early_check_value: float = 0.1,
        relu: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.slack = slack
        self.pred_threshold = pred_threshold
        self.early_check_value = early_check_value
        self.relu = relu

        for s in ('true_positives_gt', 'false_positives', 'true_positives_pred', 'false_negatives'):
            self.add_state(s, default=torch.zeros(1, dtype=torch.float), dist_reduce_fx='sum')

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        if self.relu:
            pred = torch.relu(pred)

        pred = convert_to_numpy(pred)
        target = convert_to_numpy(target)

        pred = pred.squeeze()
        target = target.squeeze()

        if len(pred.shape) == 3:
            pred, target = pred[None], target[None]

        for pred_sample, target_sample in zip(pred, target):
            pred_sample = pred_sample < self.pred_threshold

            if pred_sample.sum() / pred_sample.size > self.early_check_value:
                continue

            pred_skeleton = skeletonize_3d(pred_sample).astype(np.uint8)
            target_skeleton = (target_sample == 0).astype(np.uint8)
            distances_pred = ndimage.distance_transform_edt((np.logical_not(pred_skeleton)))

            true_pos_area_gt = target_sample <= self.slack
            false_pos_area = target_sample > self.slack
            true_pos_area_pred = distances_pred <= self.slack
            false_neg_area = distances_pred > self.slack

            self.true_positives_gt += np.logical_and(true_pos_area_gt, pred_skeleton).sum()
            self.false_positives += np.logical_and(false_pos_area, pred_skeleton).sum()
            self.true_positives_pred += np.logical_and(true_pos_area_pred, target_skeleton).sum()
            self.false_negatives += np.logical_and(false_neg_area, target_skeleton).sum()

    def compute(self) -> torch.Tensor:
        return (
            self.true_positives_gt,
            self.false_positives,
            self.true_positives_pred,
            self.false_negatives,
        )


class Correctness(ConfusionMatrix):
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def __init__(self, eps: float = 1e-12, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    def compute(self):
        return self.true_positives_gt / (self.true_positives_gt + self.false_positives + self.eps)


class Completeness(ConfusionMatrix):
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def __init__(self, eps: float = 1e-12, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    def compute(self):
        return self.true_positives_pred / (self.true_positives_pred + self.false_negatives + self.eps)


class Quality(ConfusionMatrix):
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def __init__(self, eps: float = 1e-12, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    def compute(self):
        corr = self.true_positives_gt / (self.true_positives_gt + self.false_positives + self.eps)
        comp = self.true_positives_pred / (self.true_positives_pred + self.false_negatives + self.eps)

        return (comp * corr) / (comp - comp * corr + corr + self.eps)
