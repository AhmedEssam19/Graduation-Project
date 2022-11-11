import torch
import wandb

import torchvision.transforms as transforms
import pandas as pd
import pytorch_lightning as pl

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from model import Model
from dataset import CombinedDataset, DistractionDataset


def main(config):
    torch.cuda.empty_cache()
    # Initialize a new wandb run

    wandb_logger = WandbLogger(project="Driver-Distraction", entity='graduation-project', config=config, log_model=True)

    # prepare data
    transformers_test = transforms.Compose([
        transforms.Resize(config.input_shape),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transformers_train = transforms.Compose([
        transforms.Resize(config.input_shape),
        transforms.ColorJitter(brightness=config.brightness, contrast=config.contrast, saturation=config.saturation),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train1_df = pd.read_csv(config.data_path + "/train.csv")
    train2_df = pd.read_csv(config.data_path + "/new_train.csv")
    val_df = pd.read_csv(config.data_path + "/val.csv")
    test_df = pd.read_csv(config.data_path + "/test.csv")

    train1_dataset = DistractionDataset(train1_df, transformers_train)
    train2_dataset = DistractionDataset(train2_df, transformers_train)
    train_dataset = CombinedDataset(train1_dataset, train2_dataset)
    test_dataset = DistractionDataset(test_df, transformers_test)
    val_dataset = DistractionDataset(val_df, transformers_test)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size // 2, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=6)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=6)
    # setup model
    model = Model(config.num_classes, config.learning_rate, config.decay, config.frozen_blocks)

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='val_acc', dirpath=config.ckpt_path, verbose=True, mode='max',
                                     filename='resnet50-{val_acc:.4f}', save_top_k=1),
        pl.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=True, mode='max')
    ]

    # setup Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        devices=1,
        max_epochs=config.epochs,
        callbacks=callbacks,
    )

    # train
    wandb.define_metric('val_acc', summary='max')
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(dataloaders=test_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input-shape', default=[500, 500], type=lambda x: list(map(int, x[1:-1].split(','))),
                        help='The input shape of the image to be resized to')
    parser.add_argument('--brightness', default=0.7399, type=float)
    parser.add_argument('--contrast', default=0.0429, type=float)
    parser.add_argument('--saturation', default=0.0325, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-classes', default=10, type=int)
    parser.add_argument('--learning-rate', default=5e-5, type=float)
    parser.add_argument('--decay', default=0.7083, type=float)
    parser.add_argument('--frozen-blocks', default=2, type=int,
                        help='The number of blocks of ResNet50 to freeze during fine-tuning')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--data-path', default='./data',
                        help='The path to the directory that contains the training, validation, and test CSV files')
    parser.add_argument('--ckpt-path', default='./checkpoints',
                        help='The path to save the model checkpoints')

    main(parser.parse_args())
