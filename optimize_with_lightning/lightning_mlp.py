import torch
from lightning import Trainer


from tools import PyTorchMLP, get_dataset_loaders, LightningModel, MNISTDataModule

if __name__ == "__main__":
    print(f"Torch CUDA is available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    train_loader, val_loader, test_loader = get_dataset_loaders()

    data_module = MNISTDataModule()
    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)

    lightning_model = LightningModel(pytorch_model, learning_rate=0.05)
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        deterministic=True,
    )

    # trainer.fit(
    #     model=lightning_model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=val_loader,
    # )

    trainer.fit(model=lightning_model, datamodule=data_module)

    train_acc = trainer.validate(dataloaders=data_module.train_dataloader())[0]["val_acc"]
    val_acc = trainer.validate(dataloaders=val_loader)[0]["val_acc"]
    test_acc = trainer.test(dataloaders=test_loader)[0]["test_acc"]

    print(f"Train acc {train_acc * 100:.2f}%")
    print(f"Val acc {val_acc * 100:.2f}%")
    print(f"Test acc {test_acc * 100:.2f}%")


