import torch
import torchvision

import cv2 as cv
import numpy as np

from torchvision import transforms
from torch import nn
from utils import convert2trt


class DistractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.resnet50(pretrained=True)
        self.base_model.fc = torch.nn.Linear(in_features=self.base_model.fc.in_features, out_features=10)

    def forward(self, input_data):
        return self.base_model(input_data)


class WrapperDistractionModel:
    def __init__(self):
        super(WrapperDistractionModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape = (480, 640)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(input_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = DistractionModel()
        self.model.load_state_dict(torch.load('distraction_model.pth', map_location=self.device))
        self.model = convert2trt(self.model, (1, 3, *input_shape))
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, frame: np.ndarray):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = self.transforms(frame)
        frame = torch.unsqueeze(frame.float(), 0)
        frame = frame.to(self.device)

        return frame

    def predict(self, frame):
        preprocessed_frame = self._preprocess(frame)
        scores = self.model(preprocessed_frame)
        prediction = self._postprocess(scores)
        return prediction

    @staticmethod
    def _postprocess(scores):
        prediction = torch.argmax(scores).item()
        return prediction
