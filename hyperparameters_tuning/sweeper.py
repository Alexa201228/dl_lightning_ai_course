import os

from lightning import LightningApp

from lightning_training_studio import Sweep
from lightning_training_studio.algorithm import RandomSearch
from lightning_training_studio.distributions import (
    Categorical,
    IntUniform,
    LogUniform
)

app = LightningApp(
    Sweep(
        script_path=os.path.join(os.path.dirname(__file__), "./mlp_cli.py"),
        total_experiments=3,
        parallel_experiments=1,
        algorithm=RandomSearch(distributions=({
            "--model.learning_rate": LogUniform(0.001, 0.1),
            "--model.hidden_units": Categorical(["[50, 100]",
                                                 "[100, 200]"]),
            "--data.batch_size": Categorical([32, 64]),
            "--trainer.max_epochs": IntUniform(1, 3),
        }))
    )
)
