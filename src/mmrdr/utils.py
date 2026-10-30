#reproducibility, configuration and logging helpers

from __future__ import annotations

import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def seed_everything(seed: int, deterministic: bool = False) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    except ImportError:
        pass


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        base[k] = _deep_update(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = _deep_update(deepcopy(base), cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_scalar(value)
    return cfg

                   #yaml parser
def _parse_scalar(value: str) -> Any:
    
    v = yaml.safe_load(value)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def get_logger(name: str = "mmrdr", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
