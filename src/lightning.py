import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import wandb
import pytorch_lightning as pl
from typing import Any, Dict, List, Tuple

from .collate import Batch, LJSpeechCollator
from .model import Discriminator, Generator
from .featurizer import MelSpectrogram, MelSpectrogramConfig


class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset: Dataset,
                       train_batch_size: int,
                       train_num_workers: int,
                       val_dataset: Dataset,
                       val_batch_size: int,
                       val_num_workers: int):
        super().__init__()
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

        self._train_dataloader_kwargs = {
                'batch_size': train_batch_size,
                'num_workers': train_num_workers,
                'collate_fn': LJSpeechCollator(),
            }
        self._val_dataloader_kwargs = {
                'batch_size': val_batch_size,
                'num_workers': val_num_workers,
                'collate_fn': LJSpeechCollator(),
            }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_dataset, **self._train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_dataset, **self._val_dataloader_kwargs)


class Module(pl.LightningModule):

    def __init__(self, opt_learning_rate: float,
                       opt_beta_1: float,
                       opt_beta_2: float,
                       lr_sched_gamma: float):
        super().__init__()

        self._opt_parameters = {
                'lr': opt_learning_rate,
                'betas': (opt_beta_1, opt_beta_2),
            }

        self._lr_sched_parameters = {
                'gamma': lr_sched_gamma,
            }

        self.featurizer = MelSpectrogram(MelSpectrogramConfig())
        self.generator = Generator()
        self.discriminator = Discriminator()

    def configure_optimizers(self) -> Tuple[List, List]:
        gen_optim = torch.optim.AdamW(self.generator.parameters(), **self._opt_parameters)
        gen_sched = torch.optim.lr_scheduler.ExponentialLR(gen_optim, **self._lr_sched_parameters)

        dis_optim = torch.optim.AdamW(self.generator.parameters(), **self._opt_parameters)
        dis_sched = torch.optim.lr_scheduler.ExponentialLR(dis_optim, **self._lr_sched_parameters)

        return [gen_optim, dis_optim], [gen_sched, dis_sched]

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int) -> Dict[str, Any]:
        pass

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        pass
