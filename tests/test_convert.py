import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.convert import convert
from src.train import train


@pytest.mark.slow
def test_train_convert(tmp_path, cfg_train, cfg_eval):
    """Train for 1 epoch with `train.py` and convert to SNN with `convert.py`"""
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    # we only test if the correct output was generated
    test_metric_dict, _ = convert(cfg_eval)
    save_path = os.path.join(os.path.split(cfg_eval.ckpt_path)[0], "converted_snn.pt")
    assert os.path.exists(save_path)
