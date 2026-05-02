"""Reproducibility utilities: fix random seeds for all libraries."""

import atexit
import logging
import os
import random
import sys
import traceback
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from src.config import CHOSEN_ENCODINGS_PATH, MODEL_NAME, PROCESSED_PAIRS_INFO_PATH, REJECTED_ENCODINGS_PATH, SEED

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "cli"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"

_loggers = {}
_exception_handler_installed = False


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with console and file handlers."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
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
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _loggers[name] = logger
    return logger


def setup_global_exception_handler(logger_name: str = "main") -> None:
    """Install a global exception handler that logs all uncaught exceptions before crash."""
    global _exception_handler_installed
    if _exception_handler_installed:
        return
    _exception_handler_installed = True

    logger = get_logger(logger_name)

    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.critical("Unhandled exception:\n%s", tb)

    sys.excepthook = exception_handler
    atexit.register(lambda: logger.info("Script exited normally"))


def set_seed(seed: int = SEED) -> None:
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
def get_model_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizer:
    """Get cached Qwen tokenizer (lazy loading)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_tokens(text: str, tokenizer: PreTrainedTokenizer | None = None) -> int:
    """Token count using Qwen2.5-0.5B tokenizer.

    Note: Only works for Qwen model tokenization. Uses model's tokenizer
    to ensure token counts match what the model sees during training/inference.
    """
    if tokenizer is None:
        tokenizer = get_model_tokenizer()
    return len(tokenizer.encode(str(text) if text else "", add_special_tokens=False))


def load_and_combine_pairs_tokens_info(chosen_path: Path = CHOSEN_ENCODINGS_PATH, rejected_path: Path = REJECTED_ENCODINGS_PATH, info_path: Path = PROCESSED_PAIRS_INFO_PATH) -> dict:
    """Load tokenized prompts and problem info, and combine into single dict."""
    chosen_encodings: BatchEncoding = torch.load(chosen_path) # TODO - try if works with weights_only=True
    rejected_encodings: BatchEncoding = torch.load(rejected_path) # TODO - try if works with weights_only=True
    pairs_info: dict = torch.load(info_path) # TODO - try if works with weights_only=True

    assert len(chosen_encodings["input_ids"]) == len(rejected_encodings["input_ids"]) == len(pairs_info["prompt_length"]), "Mismatched lengths of encodings and info"

    combined = pairs_info
    combined["chosen_input_ids"] = chosen_encodings["input_ids"],
    combined["rejected_input_ids"] = rejected_encodings["input_ids"]
    if "attention_mask" in chosen_encodings and "attention_mask" in rejected_encodings:
        combined["chosen_attention_mask"] = chosen_encodings["attention_mask"]
        combined["rejected_attention_mask"] = rejected_encodings["attention_mask"]

    return combined
