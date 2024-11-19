from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl

from . import hparams
from . import utils
from . import models
from . import audio



def pseudo_huber_constant(dim):
    # taken from section 3.3 in the paper "improving the training of consistency models"
    return 0.00054*math.sqrt(dim)

# pseudo-huber loss, section 3.3
# note that returns loss per batch item for loss scaling
def pseudo_huber(x, y):
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)

    c = pseudo_huber_constant(x.shape[1])
    mse = F.mse_loss(x, y, reduction='none').sum(dim=-1)

    loss = (mse + c*c).sqrt() - c
    return loss


from pathlib import Path
import soundfile as sf

fma_path = Path('/Users/shenberg/Documents/work/whitebalance/music_detection/fma/fma_large')
phi_minus_1 = (1+(5**0.5))/2 - 1

def get_train_val_datasets():
    # TODO: split by artist via metadata
    files = sorted(fma_path.glob('*/*mp3'))
    files = np.random.permutation(files)
    # TODO: hparam
    train_proportion = 0.9

    train_split_point = int(len(files)*train_proportion) + 1
    train_files = files[:train_split_point]
    val_files = files[train_split_point:]

    return FileRandomDataset(train_files), FileRandomDataset(val_files)

class FileRandomDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, slice_size = 34304):
        self.files = file_list
        self.relative_offsets = None
        self.slice_size = slice_size


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        audio_path = self.files[index]
        audio_data, sr = sf.read(audio_path, dtype='float32', always_2d=True)
        # TODO: deal with stereo, right now averaging channels
        audio_data = audio_data.transpose([1,0]).mean(0, keepdims=True)
        audio_data = torch.from_numpy(audio_data)
        # lazy init so we generate relative_offsets separately in each worker
        if self.relative_offsets is None:
            self.relative_offsets = torch.rand(len(self.files))
        offset = int(self.relative_offsets[index]*(audio_data.shape[-1] - self.slice_size))
        self.relative_offsets[index] = (self.relative_offsets[index] + phi_minus_1) % 1
        audio_slice = audio_data[..., offset : offset + self.slice_size]
        return audio.to_representation_encoder(audio_slice).squeeze(0)


    def randomize_offsets(self):
        self.relative_offsets = torch.rand(len(self.files))


class ConsistencyAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.UNet()


    def setup(self, stage):
        if stage == 'fit':
            # TODO: deal with resuming training
            self.current_training_step = 0
            self.total_steps = self.trainer.max_steps


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def step_generic(self, batch, batch_idx):
        # to match the paper syntax
        x0 = batch

        # TODO: move to hparams
        # total iterations also taken from paper
        # taken from section 4.4 implementation details of music2latent
        delta_t0 = 0.1
        # maximum step exponent
        ek = 3

        # eq. (13) section 4.3
        dt = math.pow(delta_t0, (self.current_training_step / self.total_steps) * (ek - 1) + 1)
        
        self.log('dt', dt)

        t_plus_1 = torch.rand(x0.shape[0], device=x0.device)
        t = torch.clamp(t_plus_1 - dt, min=0)
        sigma_plus_1 = utils.get_sigma_continuous(t_plus_1)
        sigma = utils.get_sigma_continuous(t)

        initial_noise = torch.randn_like(x0)

        x_t = utils.add_noise(x0, initial_noise, sigma)
        x_t_plus_1 = utils.add_noise(x0, initial_noise, sigma_plus_1)

        # eq. (10) in the paper
        latents = self.model.encoder(x0)
        y0 = self.model.decoder(latents)

        denoised_t_plus_1 = self.model(latents, x_t_plus_1, sigma_plus_1, pyramid_latents=y0)
        # note detach(), to correspond to f_theta_minus ("teacher" output)
        denoised_t = self.model(latents, x_t, sigma, pyramid_latents=y0).detach()

        # TODO: log denoised_t_plus_1

        # eq. (12)
        loss_scale = 1 / (sigma_plus_1 - sigma)

        loss = (loss_scale * pseudo_huber(denoised_t_plus_1, denoised_t)).mean()

        # TODO: deal with gradient accumulation - step should be incremented every optimizer step and not every
        #       call to this function
        self.current_training_step += 1

        return loss


    def validation_step(self, batch, batch_idx):
        return self.step_generic(batch, batch_idx)


    def training_step(self, batch, batch_idx):
        return self.step_generic(batch, batch_idx)


    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.RAdam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.total_steps,
            # TODO: move into hparams.py
            eta_min=1e-6,
            )
        return {
            # TODO: move optimizer args to hparams.py
            "optimizer" : optimizer,
            "lr_scheduler" : {
                "scheduler" : lr_scheduler,
                "interval" : "step",
            },
        }


def main():
    device = 'mps'
    # TODO: utils somewhere
    pl.seed_everything(42, workers=True)

    consistency_model = ConsistencyAE()

    train_dataset, val_dataset = get_train_val_datasets()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # TODO: hparam
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )


    trainer = pl.Trainer(
        accelerator=device,
        logger=True,
        # TODO: make hparam or argument or both
        max_steps=800000,
        )
    trainer.fit(consistency_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__=='__main__':
    main()