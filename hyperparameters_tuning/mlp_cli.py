from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from tools import CustomDataModule, LightningModel, PyTorchMLP
from watermark import watermark


if __name__ == "__main__":

    print(watermark(packages="torch, lightning"))

    cli = LightningCLI(
        model_class=LightningModel,
        datamodule_class=CustomDataModule,
        run=False,
        save_config_callback=None,
        seed_everything_default=23,
        trainer_defaults={
            "max_epochs": 10,
            "accelerator": "gpu",
            "callbacks": ModelCheckpoint(monitor="val_acc", mode="max")
        }
    )

    lt_model = LightningModel(model=None,
                              learning_rate=cli.model.learning_rate,
                              hidden_units=cli.model.hidden_units,
                              num_features=100,
                              num_classes=2,
                              )

    cli.trainer.fit(lt_model, datamodule=cli.datamodule)
    cli.trainer.test(lt_model, datamodule=cli.datamodule)
