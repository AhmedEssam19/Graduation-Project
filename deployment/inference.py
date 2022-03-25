import time

import numpy as np

import torch
from torch2trt.torch2trt import TRTModule
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('trt_model.pt', map_location=device))
    model_trt.to(device)
    model_trt.eval()
    print(device)
    shape = (1, 3, 480, 640)
    benchmark(model_trt.half(), device, shape)


def benchmark(model, device, input_shape, dtype='fp32', nwarmup=50, nruns=1000):
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
