import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class DistractionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index, 0]
        image = read_image(img_path, mode=ImageReadMode.RGB) / 255.0
        label = self.df.iloc[index, 1]

        if self.transform:
            image = self.transform(image)

        if self.df.iloc[index, 3] == "Camera 2":
            image = torch.flip(image, dims=[1])
            if label == 4 or label == 3:
                label -= 2
            elif label == 1 or label == 2:
                label += 2

        return image, int(label)


class CombinedDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        return tuple(dataset[idx % len(dataset)] for dataset in self.datasets)
