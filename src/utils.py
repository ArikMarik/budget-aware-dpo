"""Reproducibility utilities: fix random seeds for all libraries."""

import random
import os


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across torch, numpy, transformers."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import transformers
        transformers.set_seed(seed)
    except (ImportError, AttributeError):
        pass
