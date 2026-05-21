"""
licence: https://github.com/Rafiul1999/Bangla_LLM/blob/main/LICENSE
author: Amzad Hossain Rafi
date: 2026-05-12
version: 0.1.0



change log :

    v1.0.0 : Initial release
    branch : main

    date : 2026-05-12
    add: configuration file loading -->> load_project_config

"""

import yaml
import torch
# CONFIG_PATH = pathlib.Path(__file__).parent / "config" / "config.yaml"


def load_project_config(config_path="/home/rafi/Desktop/BLLMC/config/config.yaml"):
    """Load YAML and return typed config objects."""
    from BLLMC.components.config import (
        ModelConfig,
        TrainingConfig,
        DataConfig,
        RoPEConfig,
    )

    raw = read_config(config_path)

    model_raw = raw.get("model", {})
    rope_raw = model_raw.pop("rope_freq_config", {})
    if "rope_base" in model_raw:
        rope_raw["rope_base"] = model_raw.pop("rope_base")

    rope_cfg = RoPEConfig(**rope_raw)
    model_cfg = ModelConfig(**model_raw, rope=rope_cfg)
    train_cfg = TrainingConfig(**raw.get("training", {}))
    data_cfg = DataConfig(**raw.get("data", {}))

    return model_cfg, train_cfg, data_cfg


def read_config(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

