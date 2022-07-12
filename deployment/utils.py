import torch
import time

import torch.backends.cudnn as cudnn
import numpy as np

from torch2trt import torch2trt, TRTModule
from Training.model import Model


def convert2trt(model: torch.nn.Module, shape: tuple) -> TRTModule:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    x = torch.ones(shape).to(device)
    trt_model = torch2trt(model, [x], fp16_mode=True)

    return trt_model


def benchmark(model, device, input_shape, dtype='fp32', nwarmup=50, nruns=1000):
    cudnn.benchmark = True
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


def pl_to_torch(ckpt_path, output_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Model.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=device)
    torch.save(model.state_dict(), output_path)
