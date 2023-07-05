import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from fache.model import FacheModel

def train(
        model: FacheModel,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        model_dir: str,
        n_workers: int,
        max_epochs: int
    ):
    model_checkpoint = ModelCheckpoint(
        dirpath=model_dir,
        verbose=True,
        save_last=True,
        filename='best',
        mode='max',
        monitor='val_acc'
    )
    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        strategy='ddp',
        devices=n_workers,
        precision=16,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
        max_epochs=max_epochs,
        callbacks=[model_checkpoint],
        gradient_clip_val=5.0
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=dev_dataloader
    )