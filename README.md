# Graduation-Project: Driver Distraction and Drowsiness Detection
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

## Description
In our project, we will detect Distraction and Drowsiness by Plug two cameras into the car
cabin one to detect the distraction and the other to detect the Drowsiness, and then forward these
images to our chip to decide whether the driver is distracted or drowsy or neither based
on our models (Distraction and Drowsiness).

We have two scenarios:-
* If the driver isn't distracted or drowsy nothing will happen.
* If the driver is distracted or drowsy for 2 seconds, we will get his attention
back by alerting him with a loud sound using (Buzzer)

So we used a CNN model for distraction detection
that achieved 90.12% Macro F1-Score and 90.11% Accuracy. Another model uses the Eye Aspect Ratio (EAR) for drowsiness detection. The
two models are simultaneously deployed on Nvidia Jetson Nano and work at 4 FPS.

## Demo: [link](https://drive.google.com/file/d/1HXVygLLPKvDu3HmLsZsyhBqEgVE7N42e/view?usp=share_link)

## Datasets
* AUC Distracted Driver Dataset: [link](https://abouelnaga.io/projects/auc-distracted-driver-dataset/)
* StateFarm Dataset: [link](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/overview)
* Private Collected Dataset.

## Contents
The main structure of the project is as follows:

```bash  
.
├── data
│   ├── train.csv
│   ├── new_train.csv
│   ├── val.csv
│   └── test.csv
├── data_analysis
│   ├── Data_Wrangling.ipynb
│   ├── DW&EDA2.ipynb
│   ├── EDA.ipynb
│   └── new_images_annotation.py
├── deployment
│   ├── distraction_model.py
│   ├── drowsiness_model.py
│   ├── inference.py
│   └── utils.py
├── error_analysis
│   └── Error_Analysis.ipynb
├── training
│   ├── config.yaml
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── README.md
├── requirements.txt
└── Dockerfile
```

## Usage
### 1. Training
- ###### Create new environment
```shell
  ~$ git clone https://github.com/AhmedEssam19/Graduation-Project.git
  ~$ cd graduation_project
  ~$ conda create -n gp python=3.9
  ~$ conda activate gp
```
- ###### Install dependencies
```shell
  (gp):~$ conda install cudatoolkit=11.3.1 -c conda-forge
  (gp):~$ pip3 install --no-deps -r requirements.txt
  (gp):~$ wandb login
```
- ###### Run training script
```shell
  (gp):~$ python3 training/train.py --help
  
usage: train.py [-h] [--input-shape INPUT_SHAPE] [--brightness BRIGHTNESS]
                [--contrast CONTRAST] [--saturation SATURATION]
                [--batch-size BATCH_SIZE] [--num-classes NUM_CLASSES]
                [--learning-rate LEARNING_RATE] [--decay DECAY]
                [--frozen-blocks FROZEN_BLOCKS] [--epochs EPOCHS]
                [--data-path DATA_PATH] [--ckpt-path CKPT_PATH]
                [--entity ENTITY] [--project PROJECT]

optional arguments:
  -h, --help            show this help message and exit
  --input-shape INPUT_SHAPE
                        The input shape of the image to be resized to
  --brightness BRIGHTNESS
  --contrast CONTRAST
  --saturation SATURATION
  --batch-size BATCH_SIZE
  --num-classes NUM_CLASSES
  --learning-rate LEARNING_RATE
  --decay DECAY
  --frozen-blocks FROZEN_BLOCKS
                        The number of blocks of ResNet50 to freeze during
                        fine-tuning
  --epochs EPOCHS
  --data-path DATA_PATH
                        The path to the directory that contains the training,
                        validation, and test CSV files
  --ckpt-path CKPT_PATH
                        The path to save the model checkpoints
  --entity ENTITY       The desired entity on Weights&Biases
  --project PROJECT     The desired project on Weights&Biases
```
- ###### Run hyperparameter-tuning script
```shell
  (gp):~$ wandb sweep --entity [Weights&Biases Entity] --project [Weights&Biases Project] training/config.yaml
  (gp):~$ wandb agent enity/project/sweep_ID
```
### 2. Deployment on Jetson Nano
- ###### Create Docker image
```shell
  (gp):~$ docker build -t gp .
```
- ###### Run Docker Container
```shell
  (gp):~$ docker run -it --rm --runtime nvidia --network host gp [INDEX_OF_FIRST_CAMERA] [INDEX_OF_SECOND_CAMERA]
```

## Requirements
- Weights&Biases Account
- Docker
- Conda
- Jetson Nano for deployment

## Authors
<a href="https://github.com/ahmedessam19/Graduation-Project/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ahmedessam19/Graduation-Project"  alt="authors"/>
</a>