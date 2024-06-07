from typing import List, Tuple

import hydra

from lightning import Callback
from lightning.pytorch.loggers import Logger
from monai.transforms import Compose as MonaiCompose, MapTransform
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection
from torchvision.transforms.v2 import Compose as TorchvisionCompose

from masters.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

ComposeType = MonaiCompose | TorchvisionCompose


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning('No callback configs found! Skipping..')
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError('Callbacks config must be a DictConfig!')

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and '_target_' in cb_conf:
            log.info(f'Instantiating callback <{cb_conf._target_}>')
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning('No logger configs found! Skipping...')
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError('Logger config must be a DictConfig!')

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and '_target_' in lg_conf:
            log.info(f'Instantiating logger <{lg_conf._target_}>')
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_transforms(transforms_cfg: DictConfig) -> Tuple[ComposeType, ComposeType]:
    """Instantiates transforms from config.

    :param transforms_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated transforms.
    """

    if not transforms_cfg:
        log.warning('No transforms configs found! Skipping..')
        return None, None

    if not isinstance(transforms_cfg, DictConfig):
        raise TypeError('Transforms config must be a DictConfig!')

    compose_type = hydra.utils.get_class(transforms_cfg.get('compose_type'))

    transforms = [None, None]

    for index, split in enumerate(('train', 'test')):
        split_transforms: List[MapTransform] = []
        split_transforms_cfg = transforms_cfg[split]

        for transform_conf in split_transforms_cfg:
            if isinstance(transform_conf, DictConfig) and '_target_' in transform_conf:
                log.info(f'Instantiating transform <{transform_conf._target_}>')
                split_transforms.append(hydra.utils.instantiate(transform_conf))
        transforms[index] = compose_type(split_transforms)
    return transforms


def instantiate_metrics(metrics_cfg: DictConfig) -> Tuple[MetricCollection, MetricCollection]:
    """Instantiates metrics from config.

    :param metrics_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated metrics.
    """
    if not metrics_cfg:
        log.warning('No metrics configs found! Skipping..')
        return None

    if not isinstance(metrics_cfg, DictConfig):
        raise TypeError('Metrics config must be a DictConfig!')

    metrics = [None, None]

    for index, split in enumerate(('train', 'test')):
        split_metrics: List[Metric] = []
        split_metrics_cfg = metrics_cfg[split]

        for metric_cfg in split_metrics_cfg:
            if isinstance(metric_cfg, DictConfig) and '_target_' in metric_cfg:
                log.info(f'Instantiating metric <{metric_cfg._target_}>')
                split_metrics.append(hydra.utils.instantiate(metric_cfg))
        metrics[index] = MetricCollection(split_metrics)
    return metrics


def instantiate_preprocessing(preprocessing_cfg: DictConfig) -> ComposeType:
    """Instantiates preprocessing transforms from config.

    :param transforms_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated preprocessing transforms.
    """

    if not preprocessing_cfg:
        log.warning('No transforms configs found! Skipping..')
        return None, None

    if not isinstance(preprocessing_cfg, DictConfig):
        raise TypeError('Preprocessing config must be a DictConfig!')

    compose_type = hydra.utils.get_class(preprocessing_cfg.get('compose_type'))

    transforms: List[MapTransform] = []
    transforms_cfg = preprocessing_cfg['transforms']

    for transform_conf in transforms_cfg:
        if isinstance(transform_conf, DictConfig) and '_target_' in transform_conf:
            log.info(f'Instantiating transform <{transform_conf._target_}>')
            transforms.append(hydra.utils.instantiate(transform_conf))
    return compose_type(transforms)
