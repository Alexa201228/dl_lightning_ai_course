import os
from functools import partial
from unittest import mock

from lightning.app.testing import application_testing, LightningTestApp


class LightningHPOTestApp(LightningTestApp):
    def __init__(self, root, total_experiments=2, **kwargs):
        root.total_experiments = total_experiments
        super().__init__(root, **kwargs)

    def on_before_run_once(self):
        if self.root.total_experiments_done == self.root.total_experiments:
            return True


@mock.patch.dict(os.environ, {"WANDB_ENTITY": "thomas-chaton"})
@mock.patch.dict(os.environ, {"WANDB_API_KEY": "eabe6ced59f74db139187745544572f81ef76162"})
def test_custom_objective_sweep_streamlit():
    command_line = [
        os.path.join(os.getcwd(), "sweep_examples/1_app_agnostic.py"),
        "--open-ui",
        "False",
    ]
    result = application_testing(LightningHPOTestApp, command_line)
    assert result.exit_code == 0, result.__dict__


@mock.patch.dict(os.environ, {"WANDB_ENTITY": "thomas-chaton"})
@mock.patch.dict(os.environ, {"WANDB_API_KEY": "eabe6ced59f74db139187745544572f81ef76162"})
def test_pytorch_lightning_objective_sweep_wandb():
    command_line = [
        os.path.join(os.getcwd(), "sweep_examples/2_app_pytorch_lightning.py"),
        "--open-ui",
        "False",
    ]
    result = application_testing(partial(LightningHPOTestApp, total_experiments=1), command_line)
    assert result.exit_code == 0, result.__dict__


def test_pytorch_lightning_custom_objective_sweep():
    command_line = [
        os.path.join(os.getcwd(), "sweep_examples/3_app_sklearn.py"),
        "--open-ui",
        "False",
    ]
    result = application_testing(partial(LightningHPOTestApp, total_experiments=1), command_line)
    assert result.exit_code == 0, result.__dict__
