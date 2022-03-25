import torch
from torch2trt import torch2trt
from distraction_model import DistractionModel


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DistractionModel()
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    model.eval()
    print(device)
    shape = (1, 3, 480, 640)
    x = torch.ones(shape).to(device)

    trt_model = torch2trt(model, [x], fp16_mode=True)
    torch.save(trt_model.state_dict(), 'trt_model.pt')


if __name__ == "__main__":
    main()
