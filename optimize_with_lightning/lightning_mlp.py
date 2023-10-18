from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning import Trainer

from tools import PyTorchMLP, compute_accuracy, get_dataset_loaders


class LightningModel(L.LightningModule):

    def __init__(self, model, learning_rate):

        super().__init__()

        self._learning_rate = learning_rate
        self._model = model

    def forward(self, x) -> Any:
        return self._model(x)

    def training_step(self, batch, batch_idx):

        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        self.log("val_loss", loss, prog_bar=True)

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

    train_acc = compute_accuracy(pytorch_model, train_loader, device)
    val_acc = compute_accuracy(pytorch_model, val_loader, device)
    test_acc = compute_accuracy(pytorch_model, test_loader, device)

    print(f"Train acc: {train_acc * 100:.2f}%")
    print(f"Val acc: {val_acc * 100:.2f}%")
    print(f"Test acc: {test_acc * 100:.2f}%")

