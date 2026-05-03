# IPC Deadlock Bug — `tokenize_dpo_pairs_parallel`

**Date:** 2026-04-30  
**File:** `src/data/worker_utils.py`  
**Status:** Option A implemented — revealed second bug (see below)

---

## The Bug

When `tokenize_dpo_pairs_parallel` ran with `num_workers=32` and `batch_size=10_000`, the pipeline appeared to deadlock in step [3/4] — all 32 worker processes stayed alive (sleeping in `futex_wait_queue_me`) for 50+ minutes with no output, CPU at 0.1–0.4%, and no output file being written.

### What actually happened

1. Workers spawned and tokenized their shards (~3 min, normal)
2. Each worker called `return result_dict` where `result_dict` contained:
   - `chosen_input_ids`: Python `list` of **100,000 individual `torch.Tensor` objects** (variable-length, one per pair)
   - `rejected_input_ids`: same
3. `ProcessPoolExecutor` serialized this via **Python pickle** to send back through a Unix pipe
4. Pickling 200K individual small PyTorch tensors is extremely slow — each tensor requires its own `__reduce_ex__` call + storage/dtype/shape metadata
5. The `_result_handler` background thread in the main process reads results **one shard at a time** (serially), so 32 workers each queuing ~1.2GB of pickle data serialized through one reader
6. At ~10–20 MB/s effective throughput (pipe buffer contention + per-tensor Python overhead), transferring 32 × 1.2GB ≈ 38GB takes **30–90 minutes**

### Why `futex_wait_queue_me`

This is not a programming deadlock (no circular wait). It is the OS state of Python threads blocked on the `multiprocessing.Queue` internal `Condition` variable:

- **Workers:** main Python thread put result on queue and went back to waiting for the next task (which never comes). `futex_wait_queue_me` = waiting on `queue.get()` for next task.
- **Worker feeder threads:** background thread slowly writing pickled bytes to the pipe.
- **Main process:** main thread blocked in `as_completed()` waiting for the first future to be signalled. `futex_wait_queue_me` = waiting on a `threading.Event`.
- **Main process `_result_handler` thread:** slowly reading pickled bytes from pipes.

### Key signals that identify this

| Signal | Meaning |
|--------|---------|
| All workers in `futex_wait_queue_me` | Waiting on Python queue/event, not OS I/O |
| CPU at 0.1–0.4% across all workers | No compute — blocked on pipe I/O |
| Log stopped exactly when tokenization finished | Workers done tokenizing, now stuck on return |
| No output file in `processed_dpo_dataset/` | `torch.save()` never reached |
| Main process also in `futex_wait_queue_me` | Stuck in `as_completed()` waiting for first result |

---

## Why it's slow to pickle individual tensors

Python's pickle for `list[torch.Tensor]` where each tensor is small:

```
pickle(list of 100K tensors) ≠ pickle(one 100K-row stacked tensor)
```

- **List of tensors:** creates a separate pickle record for each tensor's storage, dtype, shape, stride, offset. For 100K tensors, that is ~100K object-level operations. Even at 5ms/tensor = **500 seconds per shard**.
- **Stacked tensor:** one contiguous memory block, one pickle record, serialized as raw bytes. Essentially instant.

The pipe itself is not the bottleneck — the per-object Python overhead is.

---

## Fix Comparison

### Option A — Temp files (implemented) ✅

**Idea:** Workers write their shard result to a temp `.pt` file and return only the file path string through the pipe.

```python
# In _tokenize_shard (worker side):
tmp_path = os.path.join(tempfile.gettempdir(), f"dpo_shard_{shard_idx}_{os.getpid()}.pt")
torch.save(shard_data, tmp_path)
return {"shard_idx": shard_idx, "tmp_path": tmp_path}

# In tokenize_dpo_pairs_parallel (main process side):
raw = future.result()           # receives ~50-byte path string
result = torch.load(raw["tmp_path"], map_location="cpu", weights_only=False)
os.unlink(raw["tmp_path"])      # clean up immediately
```

**Why this works:**

- `torch.save()` writes variable-length tensors in its own optimized binary format — no per-object Python overhead. A shard of 100K tensors writes in seconds.
- The IPC payload shrinks from ~1.2GB of Python pickle to ~50 bytes (a file path string). The `_result_handler` thread has nothing to do.
- `torch.load()` in the main process is fast for the same reason torch.save() is fast.
- Data format is completely unchanged — variable-length tensors stay variable-length on disk.

**Temp space requirements:**

- Peak: `num_workers` × shard_size concurrent temp files
- With 32 workers and 100K pairs/shard at ~1.2GB each ≈ **38GB peak** in `/tmp`
- `/tmp` is a 444GB overlay FS in this environment — fine
- Files are deleted immediately after each shard is merged, so in practice peak is lower

**Tradeoffs:**

| | Pros | Cons |
|---|---|---|
| Option A | Zero format change; variable-length preserved; trivial to implement | Needs ~38GB free in `/tmp` during run; extra disk write per shard |

---

### Option B — Pad and stack within the shard

**Idea:** Instead of returning a list of variable-length tensors, pad them within the shard to the shard's max length and return a stacked tensor.

```python
# In _tokenize_shard:
stacked_chosen = torch.nn.utils.rnn.pad_sequence(
    chosen_input_ids_all, batch_first=True, padding_value=pad_token_id
)
# stacked_chosen.shape = (100_000, max_len_in_this_shard)
return {
    "shard_idx": shard_idx,
    "chosen_input_ids": stacked_chosen,   # one contiguous tensor, fast to pickle
    ...
}
```

**Why this is fast:** One stacked tensor is one pickle record — serialized as raw bytes at memory bandwidth speed regardless of element count.

**The padding question:**

The tokenizer already uses `padding=False` to produce variable-length sequences. Option B adds a second round of padding *after* tokenization. The padding level depends on the shard's actual max length, not the global `max_length=2048`:

```
shard max = max(len(seq) for seq in all sequences in this shard)
```

If your sequence lengths are tightly clustered (e.g. all 1800–2048), the shard max ≈ global max and padding overhead is minimal. If you have many short sequences (e.g. 200 tokens) alongside occasional long ones (2048 tokens), shard padding wastes the short ones — **this can 4–10× the storage size**.

In the OpenMathInstruct dataset, token lengths vary widely (see `reports/figures/token_lengths.png`), so Option B would significantly inflate the final `tokens.pt` size. That's the main reason it was not chosen.

**If you want Option B for faster training data loading** (stacked tensors allow direct indexing without collation):

1. Change the final `tokens.pt` format from `list[Tensor]` to `Tensor` throughout
2. Accept larger file size on disk (~4–10× depending on length distribution)
3. Update `DPODataset.__getitem__` to slice instead of index
4. Attention masks already tell the model to ignore padding — training is unaffected

**Tradeoffs:**

| | Pros | Cons |
|---|---|---|
| Option B | No temp disk space needed; fast IPC | Inflates `tokens.pt` by 4–10× due to padding short sequences to shard max; requires changing final storage format |

---

## Option B variant — pad for IPC only, unpad before storing

You could pad inside the worker (for fast IPC), then strip the padding before storing:

```python
# In _tokenize_shard: stack (padded, fast IPC)
# In main process _merge_shard: trim each row to its actual length (via chosen_length)
#   chosen_ids = [row[:length] for row, length in zip(stacked, lengths)]
```

This preserves variable-length storage but adds O(N) Python-level slicing on the main process. The slicing itself is fast; the main benefit is keeping the on-disk format unchanged while still fixing IPC. The temp-file approach (Option A) is simpler and achieves the same result.

---

## Bug #2 — Rayon Thread Pool Explosion (discovered 2026-04-30, ~22:50)

### What happened

After Option A was deployed, a second stall appeared in step [3/4]. Symptoms were identical (`futex_wait_queue_me`, frozen context switches, CPU declining) but the diagnostic fingerprint was different:

- CPU declined **gradually** (334% → 84% over 35 min) rather than immediately
- Workers had **97 threads each** (1 Python + 96 Rayon) — visible in `/proc/PID/status`
- Total system threads: 32 workers × 97 = **3,104 threads**
- No shard files ever appeared → `torch.save()` never ran

### Root cause

The HuggingFace `tokenizers` library (Rust backend) spawns a **Rayon thread pool sized to the number of CPU cores** inside each worker process. With 32 workers each creating a 96-thread Rayon pool simultaneously, the system has 3,072 Rayon threads competing for the same 96 cores. Each worker gets 1/32 of normal CPU share:

- Tokenization that takes 3 min with 1 worker → ~100 min with 32 workers (massively over-subscribed)
- Workers appeared stuck because Python main threads got almost no CPU time
- Context switches frozen because threads were rarely scheduled

### Fix

Set `TOKENIZERS_PARALLELISM=false` before launching. This is the official HuggingFace environment variable that disables the Rust tokenizer's internal thread pool:

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=/storage/arik/nlp_final_project:$PYTHONPATH \
  nohup python -u scripts/preprocess_dpo_data.py > /tmp/preprocess_run.log 2>&1 &
```

With this set, each worker uses 1 thread for tokenization. Workers then compete normally for CPU instead of spawning 3,104 threads. Expected per-shard tokenization time drops back to ~3–5 min.

Alternative: reduce `num_workers` from 32 to 4–8. With fewer workers the Rayon pools don't over-subscribe as badly. But `TOKENIZERS_PARALLELISM=false` is the cleaner fix — it lets all 32 workers run in parallel without the thread explosion.

### Diagnostic signature

| | IPC deadlock (Bug #1) | Rayon explosion (Bug #2) |
|---|---|---|
| CPU at freeze | Near-zero immediately | Gradual decline over 30+ min |
| Context switches | Frozen immediately | Frozen, but CPU still non-zero |
| Shard files | Never appear | Never appear |
| Worker threads | ~5 (main + Python bg) | 97 (main + 96 Rayon) |
| `pipe_wait` on result_handler | Yes | Yes |
| Fix | Option A (temp files) | `TOKENIZERS_PARALLELISM=false` |

---

## Lesson

When using `ProcessPoolExecutor` with large result objects, **always profile the IPC cost** before assuming the bottleneck is compute. The rule of thumb:

- **Small results (< 1MB):** return through pipe normally  
- **Large results (> 10MB):** write to temp file, return path  
- **Lists of many small Python objects:** always more expensive to pickle than one large array — consolidate (stack/concatenate) before returning, or use temp files

When workers create their own thread pools (Rayon, OpenMP, MKL), **always cap their thread count** when running many workers in parallel. The pattern is:

```bash
# Before any ProcessPoolExecutor run with HuggingFace tokenizers:
export TOKENIZERS_PARALLELISM=false
# Or set OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 for other libraries
```

The `futex_wait_queue_me` kernel wait channel combined with near-zero CPU is the diagnostic fingerprint of the IPC deadlock. Gradually declining CPU with `futex_wait_queue_me` is the fingerprint of thread-pool over-subscription.
