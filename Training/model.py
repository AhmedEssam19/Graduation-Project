import pytorch_lightning as pl
import torchmetrics
from torch import nn
import torch
import torchvision


class Model(pl.LightningModule):
    def __init__(self, output_units, learning_rate, weight_decay, frozen_blocks):
        super().__init__()
        self.base_model = torchvision.models.resnet50(pretrained=True)
        self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features, out_features=output_units)

        freezing_layers = [
            self.base_model.conv1,
            self.base_model.bn1,
            *[getattr(self.base_model, f'layer{i}') for i in range(1, frozen_blocks + 1)]
        ]

        for layer in freezing_layers:
            for param in layer.parameters():
                param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, input_data):
        return self.base_model(input_data)

    def training_step(self, batch, batch_nb):
        input_data = torch.cat([sub_batch[0] for sub_batch in batch], dim=0)
        targets = torch.cat([sub_batch[1] for sub_batch in batch], dim=0)
        preds = self(input_data)
        loss = self.criterion(preds, targets)
        self.log('train_loss', loss)
        self.train_acc(preds, targets)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        self._evaluate(batch, 'val')

    def test_step(self, batch, batch_nb):
        self._evaluate(batch, 'test')

    def _evaluate(self, batch, name):
        input_data, targets = batch
        preds = self(input_data)
        loss = self.criterion(preds, targets)
        self.log(f'{name}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc(preds, targets)
        self.log(f'{name}_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_nb):
        if type(batch) == tuple:
            input_data, _ = batch
        else:
            input_data = batch

        preds = self(input_data)
        return torch.argmax(preds, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.learning_rate, max_lr=1e-4,
                                                      cycle_momentum=False)
        return [optimizer], [scheduler]
