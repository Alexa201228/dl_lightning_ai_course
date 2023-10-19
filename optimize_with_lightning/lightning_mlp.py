from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning import Trainer
import torchmetrics

from tools import PyTorchMLP, get_dataset_loaders


class LightningModel(L.LightningModule):

    def __init__(self, model, learning_rate):

        super().__init__()

        self._learning_rate = learning_rate
        self._model = model
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
        self._train_acc(predicted_labels, true_labels)
        self.log("train_acc", self._train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self._val_acc(predicted_labels, true_labels)
        self.log("val_acc", self._val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self._test_acc(predicted_labels, true_labels)
        self.log("test_acc", self._test_acc, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self._learning_rate)
        return optimizer


if __name__ == "__main__":
    print(f"Torch CUDA is available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataset_loaders()

    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)

    lightning_model = LightningModel(pytorch_model, learning_rate=0.05)
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    train_acc = trainer.test(dataloaders=train_loader)[0]["test_acc"]
    val_acc = trainer.test(dataloaders=val_loader)[0]["test_acc"]
    test_acc = trainer.test(dataloaders=test_loader)[0]["test_acc"]

    print(f"Train acc {train_acc * 100:.2f}%")
    print(f"Val acc {val_acc * 100:.2f}%")
    print(f"Test acc {test_acc * 100:.2f}%")


