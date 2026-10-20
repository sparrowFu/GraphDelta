import random

import numpy as np
import torch


def set_seed(seed: int):
    """只用于 sampling / bootstrap / nuisance perturbation，不涉及训练。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
