import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from watermark import watermark


class CustomDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path)
        self._img_dir = img_dir
        self._transform = transform

        self._img_names = df["filepath"]
        self._labels = df["label"]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self._img_dir, self._img_names[idx]))
        if self._transform is not None:
            img = self._transform(img)

        label = self._labels[idx]
        return img, label

    def __len__(self):
        return self._labels.shape[0]


if __name__ == "__main__":
    print(watermark(packages="torch", python=True))

    data_transforms = {
        "train":  transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5,)),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
    }

    train_dataset = CustomDataset(
        csv_path="mnist-pngs/new_train.csv",
        img_dir="mnist-pngs",
        transform=data_transforms["train"],
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    val_dataset = CustomDataset(
        csv_path="mnist-pngs/new_val.csv",
        img_dir="mnist-pngs",
        transform=data_transforms["test"],
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    test_dataset = CustomDataset(
        csv_path="mnist-pngs/test.csv",
        img_dir="mnist-pngs",
        transform=data_transforms["test"],
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    num_epoch = 1
    for epoch in range(num_epoch):

        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= 3:
                break

            print(f"Batch index: {batch_idx} | Batch size: {y.shape[0]} | x shape: {x.shape} | y shape: {y.shape}")

    print(f"Labels for current batch: {y}")
