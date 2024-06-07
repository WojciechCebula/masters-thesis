import hydra
import lightning as L

from omegaconf import DictConfig
from pqdm.processes import pqdm

from masters.data.components.paths import DataPathsCollector
from masters.utils import (
    RankedLogger,
    instantiate_preprocessing,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def preprocess(cfg: DictConfig) -> None:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get('seed'):
        L.seed_everything(cfg.seed, workers=True)

    log.info('Instantiating preprocessing transforms')

    transforms = instantiate_preprocessing(cfg.preprocess)
    data_paths_collector: DataPathsCollector = hydra.utils.instantiate(cfg.data_paths_collector)
    paths = data_paths_collector.get_paths()
    pqdm(paths, transforms, n_jobs=cfg.n_jobs)


@hydra.main(version_base='1.3', config_path='../configs', config_name='preprocess.yaml')
def main(cfg: DictConfig) -> None:
    preprocess(cfg)
    return None


if __name__ == '__main__':
    main()
