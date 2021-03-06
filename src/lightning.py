import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import wandb
import itertools
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

    def __init__(self, lambda_recon_wav: float,
                       lambda_recon_mel: float,
                       lambda_feature: float,
                       loss_func: str,
                       opt_learning_rate: float,
                       opt_beta_1: float,
                       opt_beta_2: float,
                       lr_sched_gamma: float):
        super().__init__()

        losses = {
                'mse': F.mse_loss,
                'mae': F.l1_loss,
            }

        self._lambda_recon_wav = lambda_recon_wav
        self._lambda_recon_mel = lambda_recon_mel
        self._lambda_feature = lambda_feature
        self._loss_func = losses[loss_func]

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
        dis_optim = torch.optim.Adam(self.generator.parameters(), **self._opt_parameters)
        dis_sched = torch.optim.lr_scheduler.ExponentialLR(dis_optim, **self._lr_sched_parameters)

        gen_optim = torch.optim.Adam(self.generator.parameters(), **self._opt_parameters)
        gen_sched = torch.optim.lr_scheduler.ExponentialLR(gen_optim, **self._lr_sched_parameters)

        return [dis_optim, gen_optim], [dis_sched, gen_sched]

    def _get_mel_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        X = self.featurizer(wav)
        #  X = X.clamp(min=1e-6).log()
        return X

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int) -> Dict[str, Any]:
        real_wavs = batch.waveform
        real_mels = self._get_mel_spectrogram(real_wavs)
        real_wavs = torch.unsqueeze(real_wavs, dim=1)

        fake_wavs = self.generator(real_mels)
        fake_wavs = fake_wavs[:, :, :real_wavs.shape[2]]
        fake_mels = self._get_mel_spectrogram(fake_wavs.squeeze(dim=1))

        if optimizer_idx == 0: # discriminator
            real_disc_out, _ = self.discriminator(real_wavs.detach())
            fake_disc_out, _ = self.discriminator(fake_wavs.detach())

            real_loss = sum(torch.mean((x - 1) ** 2) for x in real_disc_out)
            fake_loss = sum(torch.mean(x ** 2) for x in fake_disc_out)
            disc_loss = real_loss + fake_loss

            self.log('disc_real_loss', real_loss.item())
            self.log('disc_fake_loss', fake_loss.item())
            self.log('disc_all_loss', disc_loss.item())

            return {
                    'loss': disc_loss,
                }

        elif optimizer_idx == 1: # generator
            _,             real_disc_maps = self.discriminator(real_wavs)
            fake_disc_out, fake_dics_maps = self.discriminator(fake_wavs)

            recon_mel_loss = self._loss_func(fake_mels, real_mels)
            recon_wav_loss = self._loss_func(fake_wavs, real_wavs)

            fake_loss = 0
            feature_loss = 0

            for disc_out in fake_disc_out:
                fake_loss += torch.mean((disc_out - 1) ** 2)

            for real_maps, fake_maps in zip(real_disc_maps, fake_dics_maps):
                for real_map, fake_map in zip(real_maps, fake_maps):
                    feature_loss += self._loss_func(fake_map, real_map)

            gen_loss = fake_loss + self._lambda_recon_mel * recon_mel_loss \
                                 + self._lambda_recon_wav * recon_wav_loss \
                                 + self._lambda_feature * feature_loss

            self.log('gen_recon_mel_loss', recon_mel_loss.item())
            self.log('gen_recon_wav_loss', recon_wav_loss.item())
            self.log('gen_fake_loss', fake_loss.item())
            self.log('gen_feature_loss', feature_loss.item())
            self.log('gen_all_loss', gen_loss.item())

            return {
                    'loss': gen_loss,
                }

        else: # undefined
            raise ValueError('undefined optimizer')


    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        real_wavs = batch.waveform
        real_mels = self._get_mel_spectrogram(real_wavs)
        fake_wavs = self.generator(real_mels)[:, :, :-99]
        fake_wavs = fake_wavs.squeeze(dim=1)

        return {
                'real': real_wavs.detach().cpu().to(torch.float32),
                'fake': fake_wavs.detach().cpu().to(torch.float32),
            }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]):
        reals = itertools.chain(*[x['real'] for x in outputs])
        fakes = itertools.chain(*[x['fake'] for x in outputs])

        table_lines = []
        table_name = f'examples_{self.current_epoch}'

        for real, fake in zip(reals, fakes):
            line = [
                    wandb.Audio(real, sample_rate=22050),
                    wandb.Audio(fake, sample_rate=22050),
                ]
            table_lines.append(line)

        self.logger.experiment.log({
                table_name: wandb.Table(
                        columns=['real', 'fake'],
                        data=table_lines,
                    )
            })
