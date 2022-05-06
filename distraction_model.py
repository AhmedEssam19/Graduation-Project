import torch
import torchvision
from torch import nn


class DistractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.resnet50(pretrained=True)
        self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features, out_features=10)


    def forward(self, input_data):
        return self.base_model(input_data)
