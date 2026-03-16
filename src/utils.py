"""Reproducibility utilities: fix random seeds for all libraries."""

import logging
import os
import random
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

from transformers import AutoTokenizer

from src.config import MODEL_NAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "cli"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_loggers = {}


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with console and file handlers."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        _loggers[name] = logger
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = name.replace(".", "_").replace("/", "_")
    log_file = LOG_DIR / f"{safe_name}.log"

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=2,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _loggers[name] = logger
    return logger


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
    """Word count as proxy for token count. Deprecated: prefer count_tokens_qwen for preprocessing."""
    return max(1, len(str(text).split()))


@lru_cache(maxsize=1)
def _get_model_tokenizer():
    """Get cached Qwen tokenizer (lazy loading)."""
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def count_tokens(text: str) -> int:
    """Token count using Qwen2.5-0.5B tokenizer.

    Note: Only works for Qwen model tokenization. Uses model's tokenizer
    to ensure token counts match what the model sees during training/inference.
    """
    tokenizer = _get_model_tokenizer()
    return len(tokenizer.encode(str(text) if text else "", add_special_tokens=False))
