# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import itertools
import os
import random
import sys
from enum import Enum
from typing import Dict, List

import numpy as np
import torch


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LogDirType(Enum):
    CHECKPOINT = "checkpoint"
    FINAL = "final"
    ROLLING = "rolling"


class LogItemType(Enum):
    CURRENT_EPOCH = "current_epoch"
    MODEL_STATE_DICT = "model_state_dict"
    OPTIMIZER_STATE_DICT = "optimizer_state_dict"


class DatasetItemType(Enum):
    OBSERVATIONS = "observations"
    ACTIONS = "actions"
    REWARDS = "rewards"
    DONES = "dones"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def permutate_params_and_merge(grid: Dict[str, List], defaults={}) -> List[Dict[str, any]]:
    all_dynamic_combinations = permutate_values_from(grid)
    return [merge_two_dicts(defaults, dynamic_params) for dynamic_params in all_dynamic_combinations]


def permutate_values_from(grid: Dict[str, List]) -> List[Dict[str, any]]:
    """
    Example:
        >>> grid = {k_0: [v_00, v_01, v_02], k_2: [v_21, v_22]}
        >>> result = permutate_param_values(grid)
        >>> result
        [{k_0: v_00, k_2: v_21}, {k_0: v_00, k_2: v_22},
         {k_0: v_01, k_2: v_21}, {k_0: v_01, k_2: v_22},
         {k_0: v_02, k_2: v_21}, {k_0: v_02, k_2: v_22},]
    """
    params, choice_lists = zip(*grid.items())
    return [dict(zip(params, choices)) for choices in itertools.product(*choice_lists)]


def merge_two_dicts(a: Dict, b: Dict) -> Dict:
    """
    Example:
        >>> a = {k_0: v_0}
        >>> b = {k_1: v_1, k_0: v_2}
        >>> c = merge_two_dicts(a, b)
        >>> c
        {k_0: v_2, k_1: v_1}
    """
    python_version = sys.version_info
    if python_version[0] >= 3 and python_version[1] >= 9:
        return a | b
    elif python_version[0] >= 3 and python_version[1] >= 5:
        return {**a, **b}
    else:
        c = a.copy()
        c.update(b)
        return c
