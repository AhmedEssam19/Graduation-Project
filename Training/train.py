import torch
import wandb

import torchvision.transforms as transforms
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from Training.model import Model
from Training.dataset import CombinedDataset, DistractionDataset


PATH = './'


def main(config=None):
    torch.cuda.empty_cache()
    # Initialize a new wandb run
    with wandb.init(job_type="train", config=config) as run:
        config = run.config

        wandb_logger = WandbLogger(project="Driver-Distraction", entity='graduation-project', config=config,
                                   experiment=run, log_model=True)

        # prepare data
        transformers_test = transforms.Compose([
            transforms.Resize(config.input_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transformers_train = transforms.Compose([
            transforms.Resize(config.input_shape),
            transforms.ColorJitter(brightness=config.brightness, contrast=config.contrast,
                                   saturation=config.saturation),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train1_df = pd.read_csv(PATH + "data/train.csv")
        train2_df = pd.read_csv(PATH + "data/new_train.csv")
        val_df = pd.read_csv(PATH + "data/val.csv")
        test_df = pd.read_csv(PATH + "data/test.csv")

        train1_dataset = DistractionDataset(train1_df, transformers_train)
        train2_dataset = DistractionDataset(train2_df, transformers_train)
        train_dataset = CombinedDataset(train1_dataset, train2_dataset)
        test_dataset = DistractionDataset(test_df, transformers_test)
        val_dataset = DistractionDataset(val_df, transformers_test)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size // 2, shuffle=True,
                                      num_workers=2)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        # setup model
        model = Model(config.num_classes, config.learning_rate, config.decay, config.frozen_blocks)

        callbacks = [
            pl.callbacks.ModelCheckpoint(monitor='val_acc', dirpath=PATH, verbose=True, mode='max',
                                         filename='resnet50-{val_acc:.4f}'),
            pl.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=True, mode='max')
        ]

        # setup Trainer
        trainer = pl.Trainer(
            logger=wandb_logger,
            gpus=1,
            max_epochs=config.epochs,
            callbacks=callbacks
        )

        # train
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.validate(dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
