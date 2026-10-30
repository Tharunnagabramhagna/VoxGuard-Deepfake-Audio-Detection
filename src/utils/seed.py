"""Reproducibility utilities for VoxGuard."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries to guarantee deterministic reproduction.

    Args:
        seed: The integer random seed (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

