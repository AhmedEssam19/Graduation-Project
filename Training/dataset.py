from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

import torchvision.transforms as transforms

PATH = '../'


class CreateDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index, 0]
        image = read_image(PATH + img_path, mode=ImageReadMode.GRAY) / 255.0
        image = image.repeat((3, 1, 1))
        label = self.df.iloc[index, 1]

        if self.transform:
            image = self.transform(image)

        if self.df.iloc[index, 3] == "Camera 2":
            image = transforms.RandomHorizontalFlip(p=1.0)(image)
            if label == 4 or label == 3:
                label -= 2
            elif label == 1 or label == 2:
                label += 2

        return image, int(label)


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        return tuple(dataset[idx % len(dataset)] for dataset in self.datasets)
