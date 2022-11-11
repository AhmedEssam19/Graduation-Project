apt update
apt-get install libpng-dev libjpeg-dev
pip3 install -U pip
python3 -m pip install cmake packaging
python3 -m pip install python-can dlib==19.23.0 imutils scipy opencv-python opencv-contrib-python Jetson.GPIO
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python3 setup.py install
cd ..
rm -r torch2trt
