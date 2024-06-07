from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

from lightning import LightningDataModule
from monai.data import list_data_collate
from monai.data.dataloader import DataLoader
from monai.data.dataset import CacheDataset
from monai.transforms import (
    Compose,
)

from masters.data.components.paths import DataPathsCollector


class ImageCasDataPathsCollectors(DataPathsCollector):
    def get_paths(self, split: str | None = None) -> Sequence[Dict[str, Any]]:
        images_path = self.root_dir / 'images'
        labels_path = self.root_dir / 'labels'

        results = []
        for image_path, label_path in zip(sorted(images_path.iterdir()), sorted(labels_path.iterdir())):
            stem = image_path.name.split('.')[0]
            if split and not any(str(name) == stem for name in self.split_files[split]):
                continue
            results.append({'image': image_path, 'label': label_path, 'name': stem})
        return results

    def load_split_file(self, split_path: str | Path) -> Dict[str, Any]:
        split_path = Path(split_path) if split_path is not None else self.root_dir / 'split.yaml'
        with open(split_path, 'r') as file:
            return yaml.safe_load(file)


class ImageCasDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str | Path,
        train_transforms: Compose,
        test_transforms: Compose,
        split_path: str | Path | None = None,
        batch_size: int = 4,
        num_workers: int = 4,
        cache_rate: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.num_workers = num_workers
        self.data_paths_collector = ImageCasDataPathsCollectors(self.dataset_path, split_path)
        self.cache_rate = cache_rate

    def setup(self, stage: str):
        if stage == 'fit':
            train_data_paths = self.data_paths_collector.get_paths('train')
            self.train_ds = CacheDataset(
                data=train_data_paths, transform=self.train_transforms, cache_rate=self.cache_rate, num_workers=8
            )

            val_data_paths = self.data_paths_collector.get_paths('val')
            self.val_ds = CacheDataset(
                data=val_data_paths, transform=self.test_transforms, cache_rate=self.cache_rate, num_workers=4
            )
        elif stage == 'test':
            test_data_paths = self.data_paths_collector.get_paths('test')
            self.test_ds = CacheDataset(
                data=test_data_paths, transform=self.test_transforms, cache_rate=self.cache_rate, num_workers=4
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=list_data_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers // 2,
            shuffle=False,
            collate_fn=list_data_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers // 2,
            shuffle=False,
            collate_fn=list_data_collate,
        )
