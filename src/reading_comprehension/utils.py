import logging
import os
import random
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import yaml

from src.get_root import get_root


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if len(logger.handlers) == 0:
        logger.addHandler(handler)

    return logger


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_val(dictionary: Dict[str, Any]) -> Any:
    values = list(dictionary.values())
    assert len(values) == 1
    return values[0]


def get_key(dictionary: Dict[str, Any]) -> Any:
    keys = list(dictionary.keys())
    assert len(keys) == 1
    return keys[0]


class YamlConfig:

    def __init__(self, config_path: str, additional_args: Dict[str, str] = None):
        self.config_dict = AttrDict()
        self._read_yml(os.path.join(get_root(), config_path))
        if additional_args:
            self._update(additional_args)

    def _read_yml(self, config_path: str) -> None:
        with open(config_path, 'r') as yml_file:
            yaml_raw = yaml.load(yml_file, Loader=yaml.FullLoader)

        self._update(yaml_raw)

    def _update(self, config: {Dict[str, str], List[Dict[str, str]]}) -> None:
        assert type(config) in [dict, list], f"type(config) is {type(config)}, should be Namespace or list"

        if type(config) is dict:
            for key, value in config.items():
                self.config_dict.setdefault(key, value)

        else:
            for single_conf in config:
                self.config_dict.setdefault(get_key(single_conf), get_val(single_conf))

    def get(self):
        return self.config_dict


class Result:
    """
    Constructs a SemEvalResult which can be used to evaluate a model's output on the SemEval dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


class Prediction:

    def __init__(self, start_logit: float, end_logit: float, feature_index: Optional[int] = None,
                 start_index: Optional[int] = None, end_index: Optional[int] = None, text: Optional[str] = None):

        self.text = text
        self.probability = None

        self.start_logit = start_logit
        self.end_logit = end_logit

        self.feature_index = feature_index
        self.start_index = start_index
        self.end_index = end_index

    def __iter__(self):
        for key, value in self.__dict__.items():
            if value is not None or key == "text":
                yield key, value

    def set_probability(self, value: float):
        self.probability = value


def set_seed(n_gpu, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)