"""
Shared utilities for parallel data loading and processing.
Provides batch processing helpers.
"""

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import PreTrainedTokenizer

from src.utils import get_logger, count_tokens, _get_model_tokenizer


logger = get_logger(__name__)


def count_tokens_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizer | None = None,
    internal_batch_size: int = 20_000,
    show_progress: bool = True,
) -> list[int]:
    """
    Batch tokenize multiple texts efficiently with internal chunking.

    Args:
        texts: List of text strings to tokenize
        tokenizer: Optional tokenizer (uses _tokenizer if None)
        internal_batch_size: Number of texts to process per chunk (default: 1000)
        show_progress: Show progress bar (default: True)

    Returns:
        List of token counts, one per input text
    """
    if not texts:
        return []

    if tokenizer is None:
        tokenizer = _get_model_tokenizer()
    if tokenizer is None:
        return [count_tokens(t) for t in tqdm(texts, desc="Tokenizing texts (one by one)")]

    results = []
    num_batches = (len(texts) + internal_batch_size - 1) // internal_batch_size

    iterator = range(0, len(texts), internal_batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc=f"Tokenizing {len(texts)} texts", unit="batch")

    for start in iterator:
        end = min(start + internal_batch_size, len(texts))
        chunk = texts[start:end]
        encodings = tokenizer(
            chunk,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        chunk_results = [len(enc) for enc in encodings["input_ids"]]
        results.extend(chunk_results)

    return results