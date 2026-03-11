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


def approx_tokens(text: str) -> int:
    """Word count as proxy for token count. Deprecated: prefer count_tokens_tiktoken for preprocessing."""
    return max(1, len(str(text).split()))


def count_tokens_tiktoken(text: str) -> int:
    """Token count using tiktoken cl100k_base (GPT-4/Claude compatible)."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(str(text) if text else ""))
