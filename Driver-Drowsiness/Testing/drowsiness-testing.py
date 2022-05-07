import torch
import torch.nn as nn
import torch.optim as optim  
import torchvision.transforms as transforms
import torchvision
import os
from os.path import dirname, join
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,models
from torchvision.transforms import ToTensor
import torchmetrics
from torch import nn
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import random
import glob
from PIL import Image
import cv2

#test_imgs_path = '../input/drowsiness-test-data'
test_videos_path = './Drowsiness Test/Videos/'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.residual_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2,
                                       bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)

        self.sepConv1 = SeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=False,
                                        padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.relu = nn.ReLU()

        self.sepConv2 = SeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False,
                                        padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.residual_bn(res)
        x = self.sepConv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.sepConv2(x)
        x = self.bn2(x)
        x = self.maxp(x)
        return res + x

class Model(pl.LightningModule):
    def __init__(self, output_units):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8, affine=True, momentum=0.99, eps=1e-3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(8, momentum=0.99, eps=1e-3)
        self.relu2 = nn.ReLU()

        self.module1 = ResidualBlock(in_channels=8, out_channels=16)
        self.module2 = ResidualBlock(in_channels=16, out_channels=32)
        self.module3 = ResidualBlock(in_channels=32, out_channels=64)
        self.module4 = ResidualBlock(in_channels=64, out_channels=128)

        self.last_conv = nn.Conv2d(in_channels=128, out_channels=output_units, kernel_size=3, padding=1)
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        
        self.learning_rate = 0.001
        self.save_hyperparameters()
        
    def forward(self, input_data):
        x = input_data
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.last_conv(x)
        x = self.avgp(x)
        x = x.view((x.shape[0], -1))
        return x
        
    def training_step(self, batch, batch_idx):
        input_data, targets = batch
        preds = self(input_data)
        loss = self.criterion(preds, targets)
        self.log('train_loss', loss)
        self.train_acc(preds, targets)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
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
        
    def predict_step(self, batch, batch_idx):
        input_data, targets = batch
        preds = self(input_data)
        return torch.argmax(preds, dim=1)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.learning_rate, max_lr=1e-4, cycle_momentum=False)
        return [optimizer],[scheduler]

class BaseWrapper(object):
    """
    Please modify forward() and backward() according to your task.
    """

    def __init__(self, model):
        super(BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        """
        Class-specific backpropagation
        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """

        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

class BackPropagation(BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient

class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))

class GradCAM(BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps
                self.fmap_pool[key] = output.detach()

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image):
        self.image_shape = image.shape[2:]
        return super(GradCAM, self).forward(image)

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

#load the model
loaded_model = Model.load_from_checkpoint('./Drowsiness-val_acc=1.0000.ckpt')

shape = (24,24)
classes = [
    'Closed',
    'Open',
]

eyess=[]
detected_face=0
"""
Depending on the quality of your camera, this number can vary 
between 10 and 40, since this is the "sensitivity" to detect the eyes.
"""
sensi=20

def preprocess_img(img):
    global eyess
    global detected_face
    eyes_list = []
    rectangle = ()
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)
    if len(faces) == 0:
        #no face detection
        detected_face = 0
        ...
    else:
        detected_face = 1
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 5)
            rectangle = (x,y,w,h)
            roi_gray = gray_img[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 5)
                eye = roi_gray[ey:ey+eh, ex:ex+ew]
                eye = cv2.resize(eye, shape)
                eyes_list.append([transform_test(Image.fromarray(eye))])
    return img, eyes_list, rectangle

def eye_status(image):
    img = torch.stack(image)
    bp = BackPropagation(model=loaded_model)
    probs, ids = bp.forward(img)
    actual_status = ids[:, 0]
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:,1]
    return classes[actual_status.data]

def drow(img):
    global eyess
    global detected_face
    eyes = []
    loaded_model.eval()
    left_status=""
    right_status=""
    preprocessed_img, eyes, rectangle = preprocess_img(img)
    print(len(eyes))
    if detected_face == 0:
        # if the preprocessing does not see any faces, we indicate that we have not seen a face.
        preprocessed_img = cv2.putText(preprocessed_img, 'No face Detected', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 0, 0), 10, cv2.LINE_AA)
    elif(len(eyes)==2):
        # If we have seen eyes in the image, we run them through the neural network to determine whether the eyes are distracted or not.
        right_status = eye_status(eyes[0])
        left_status = eye_status(eyes[1])
        print("right = ",right_status," , left = ",left_status)
        if(right_status == "Closed" or left_status == "Closed"):
            preprocessed_img = cv2.putText(preprocessed_img, 'Closed eyes', (int(rectangle[0]+rectangle[2]/8), int(rectangle[1]-15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            preprocessed_img = cv2.putText(preprocessed_img, 'Open eyes', (int(rectangle[0]+rectangle[2]/8), int(rectangle[1]-15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    
    return preprocessed_img


# from moviepy.editor import VideoFileClip
# video = VideoFileClip(test_videos_path+'v6.mp4')
# output_video = './v6_output.mp4'
# clip = video.fl_image(drow)
# clip.write_videofile(output_video, audio=False, fps=20)
# video.reader.close()
# video.close()

def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while(video.isOpened()):
        ret, frame = video.read()
        output = drow(frame)
        cv2.imshow('Output',output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

main()