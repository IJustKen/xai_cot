"""
This file is mainly to read and store the config as a class's method 
so that it can be passed as arg to functions.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml


@dataclass
class Config:
    seed: int
    model: Dict[str, Any]
    generation: Dict[str, Any]
    data: Dict[str, Any]
    attribution: Dict[str, Any]
    sensitivity: Dict[str, Any]
    paths: Dict[str, str]

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

        cfg = cls(
            seed=cfg_dict.get("seed", 42),
            model=cfg_dict.get("model", {}),
            generation=cfg_dict.get("generation", {}),
            data=cfg_dict.get("data", {}),
            attribution=cfg_dict.get("attribution", {}),
            sensitivity=cfg_dict.get("sensitivity", {}),
            paths=cfg_dict.get("paths", {}),
        )

        for p in [
            cfg.data.get("raw_dir"),
            cfg.data.get("processed_dir"),
            os.path.dirname(cfg.paths.get("generations_file", "results/")),
            cfg.paths.get("attributions_dir"),
            cfg.paths.get("figures_dir"),
        ]:
            if p:
                os.makedirs(p, exist_ok=True)

        return cfg
