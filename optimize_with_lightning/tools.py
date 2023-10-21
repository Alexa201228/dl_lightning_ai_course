import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from typing import Any

import lightning as L
import torch.nn.functional as F
import torchmetrics

class LightningModel(L.LightningModule):

    def __init__(self, model, learning_rate):

        super().__init__()

        self._learning_rate = learning_rate
        self._model = model
        self.save_hyperparameters(ignore=["model"])
        self._train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self._val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self._test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)

        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels
    def forward(self, x) -> Any:
        return self._model(x)

    def training_step(self, batch, batch_idx):

        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        self._train_acc(predicted_labels, true_labels)
        self.log("train_acc", self._train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self._val_acc(predicted_labels, true_labels)
        self.log("val_acc", self._val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self._test_acc(predicted_labels, true_labels)
        self.log("test_acc", self._test_acc, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self._learning_rate)
        return optimizer


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(25, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./mnist", batch_size=64):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size

    def prepare_data(self):
        # download
        datasets.MNIST(self._data_dir, train=True, download=True)
        datasets.MNIST(self._data_dir, train=False, download=True)

    def setup(self, stage: str):
        self.mnist_test = datasets.MNIST(
            self._data_dir, transform=transforms.ToTensor(), train=False
        )
        self.mnist_predict = datasets.MNIST(
            self._data_dir, transform=transforms.ToTensor(), train=False
        )
        mnist_full = datasets.MNIST(
            self._data_dir, transform=transforms.ToTensor(), train=True
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self._batch_size, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self._batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self._batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self._batch_size, shuffle=False)


def get_dataset_loaders():
    train_dataset = datasets.MNIST(
        root="./mnist", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = datasets.MNIST(
        root="./mnist", train=False, transform=transforms.ToTensor()
    )

    train_dataset, val_dataset = random_split(train_dataset, lengths=[55000, 5000])

    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

def compute_accuracy(model, dataloader, device=None):

    if device is None:
        device = torch.device("cpu")
    model = model.eval()

    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples
