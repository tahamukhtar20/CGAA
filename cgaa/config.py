import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(experiment_path: str | os.PathLike | None = None) -> dict:
    """Load default config and optionally overlay an experiment YAML.

    Also ensures ``results_dir`` and ``cav_dir`` exist.
    """
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    if experiment_path is not None:
        with open(experiment_path) as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["cav_dir"], exist_ok=True)
    return cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
