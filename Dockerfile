FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3

# Copy Code
COPY deployment/ /home/app
WORKDIR /home/app

# Install Dependencies
RUN apt-get install libpng-dev libjpeg-dev
RUN pip3 install -U pip
RUN python3 -m pip install cmake packaging
RUN python3 -m pip install python-can dlib==19.23.0 imutils scipy opencv-python opencv-contrib-python Jetson.GPIO
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR /home/app/torch2trt
RUN python3 setup.py install
WORKDIR /home/app
RUN rm -r torch2trt

# Download Models Weights
RUN python3 -m pip install gdown
RUN gdown 1wea08KIZ1VpLZ8rhLV6__tD0jZB2aLzx
RUN gdown 1_V0BAZU8lDps3YYgmqRYLrUiKNQR0E6a

ENTRYPOINT ["python3", "inference.py"]