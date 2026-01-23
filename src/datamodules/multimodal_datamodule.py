"""
LightningDataModule for BigEarthNet-v2 and SEN12MS datasets.

This module handles dataset loading, transforms, and dataloaders for
SAR-Optical multimodal training.
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig
from typing import Optional
import random
import numpy as np

from configilm.extra.DataSets import BENv2_DataSet
from dataloaders.sen12ms_dataloader import SEN12MSDataset
from utils.transforms import make_transform


def seed_worker(worker_id):
    """Worker init function for reproducible data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MultimodalDataModule(pl.LightningDataModule):
    """
    DataModule for SAR-Optical multimodal datasets.

    Supports:
    - BigEarthNet-v2 (benv2): LMDB format with 19-class land cover labels
    - SEN12MS: Directory-based format with 11-class labels

    Args:
        cfg: OmegaConf configuration object
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg.data.dataset
        self.train_dataset = None
        self.val_dataset = None
        self.pos_weight = None

        # Generator for reproducible data loading
        self.g = torch.Generator()
        self.g.manual_seed(cfg.experiment.seed)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation."""
        if stage == "fit" or stage is None:
            self._setup_transforms()
            self._setup_datasets()

            # Handle overfit mode
            if self.cfg.debug.overfit_batches > 0:
                batch_size = self.cfg.data.dataloader.batch_size
                base_indices = list(range(min(batch_size, len(self.train_dataset))))
                replicated_indices = base_indices * 10000
                self.train_dataset = Subset(self.train_dataset, replicated_indices)

    def _setup_transforms(self):
        """Setup data transforms for training and validation."""
        resize_size = self.cfg.data.preprocessing.resize_size
        dataset = self.dataset_name

        self.transform_train = {
            "opt": make_transform(
                resize_size=resize_size,
                data_type="opt",
                is_train=True,
                train_datatype="opt",
                dataset=dataset
            ),
            "sar": make_transform(
                resize_size=resize_size,
                data_type="sar",
                is_train=True,
                train_datatype="sar",
                dataset=dataset
            )
        }

        self.transform_val = {
            "opt": make_transform(
                resize_size=resize_size,
                data_type="opt",
                is_train=False,
                train_datatype="opt",
                dataset=dataset
            ),
            "sar": make_transform(
                resize_size=resize_size,
                data_type="sar",
                is_train=False,
                train_datatype="sar",
                dataset=dataset
            )
        }

    def _setup_datasets(self):
        """Initialize datasets based on configuration."""
        if self.dataset_name == "benv2":
            self._setup_benv2()
        elif self.dataset_name == "sen12ms":
            self._setup_sen12ms()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Get class weights for handling imbalance
        self.pos_weight = self.train_dataset.get_class_weights()

    def _setup_benv2(self):
        """Setup BigEarthNet-v2 datasets."""
        cfg = self.cfg.data.benv2
        datapath = {
            "images_lmdb": cfg.images_lmdb,
            "metadata_parquet": cfg.metadata_parquet,
            "metadata_snow_cloud_parquet": cfg.metadata_snow_cloud_parquet,
        }

        self.train_dataset = BENv2_DataSet.BENv2DataSet(
            data_dirs=datapath,
            img_size=tuple(cfg.img_size),
            split='train',
            transform=self.transform_train,
            merge_patch=cfg.merge_patch
        )

        self.val_dataset = BENv2_DataSet.BENv2DataSet(
            data_dirs=datapath,
            img_size=tuple(cfg.img_size),
            split='test',
            transform=self.transform_val,
            merge_patch=cfg.merge_patch,
            get_labels_name=True
        )

    def _setup_sen12ms(self):
        """Setup SEN12MS datasets."""
        cfg = self.cfg.data.sen12ms

        self.train_dataset = SEN12MSDataset(
            root_dir=cfg.root_dir,
            subset="train",
            seed=self.cfg.experiment.seed,
            transform=self.transform_train
        )

        self.val_dataset = SEN12MSDataset(
            root_dir=cfg.root_dir,
            subset="test",
            seed=self.cfg.experiment.seed,
            transform=self.transform_val
        )

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.dataloader.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.dataloader.num_workers,
            pin_memory=self.cfg.data.dataloader.pin_memory,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=self.g
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.dataloader.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.dataloader.num_workers,
            pin_memory=self.cfg.data.dataloader.pin_memory
        )
