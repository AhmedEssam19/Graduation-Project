import time

import numpy as np

import torch
import torch_tensorrt
from distraction_model import DistractionModel
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DistractionModel()
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    model.eval()
    print(device)
    trt_model = torch_tensorrt.compile(model,
                                       inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
                                       enabled_precisions={torch.half}  # Run with FP16
                                       )
    benchmark(trt_model, device, dtype='fp16')


def benchmark(model, device, input_shape=(1, 3, 224, 224), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)
    if dtype == 'fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            _ = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            _ = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print('Iteration %d/%d, avg batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))


if __name__ == "__main__":
    main()
