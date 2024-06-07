from pathlib import Path
from typing import Iterable

import torch

from lightning.pytorch.loggers import Logger, TensorBoardLogger
from monai.data.meta_tensor import MetaTensor
from monai.transforms import AsDiscrete
from monai.visualize.img2tensorboard import plot_2d_or_3d_image


def log_3d_images(
    loggers: Iterable[Logger], y_hat: MetaTensor, multiclass: bool = False, step: int = 0, frame_dim: int = -3
):
    tb_logger: TensorBoardLogger = None
    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            tb_logger = logger.experiment
            break

    if tb_logger is None:
        raise ValueError('TensorBoard Logger not found')

    if multiclass:
        y_hat = AsDiscrete(argmax=True, to_onehot=y_hat.shape[1], dim=1)(y_hat)
    else:
        y_hat = torch.sigmoid(y_hat)
        y_hat = AsDiscrete(threshold=0.5)(y_hat)

    file_paths = y_hat.meta.get('filename_or_obj', ['unknown'] * int(y_hat.shape[0]))

    for index, file_path in zip(range(y_hat.shape[0]), file_paths):
        filename = Path(file_path).name.removesuffix('.nii.gz')

        plot_2d_or_3d_image(
            y_hat[index][None],
            step=step,
            writer=tb_logger,
            frame_dim=frame_dim,
            tag=filename,
            max_channels=y_hat.shape[1],
        )
