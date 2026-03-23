#!/usr/bin/env python3
"""
Keep the GPU alive by running a scheduled workload every INTERVAL (seconds).
"""
import gc
from time import time, sleep

import torch
from tqdm import tqdm

RUN_DURATION = 600        # 10 minutes
INTERVAL = 3600           # 1 hour
SIZE = 8192
NUM_STREAMS = 4


def gpu_stress():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nUsing GPU: {gpu_name}")

    # Allocate matrices
    matrices = []
    for _ in range(NUM_STREAMS):
        a = torch.randn(SIZE, SIZE, device=device)
        b = torch.randn(SIZE, SIZE, device=device)
        matrices.append((a, b))

    streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]

    # Warmup
    for i in range(NUM_STREAMS):
        with torch.cuda.stream(streams[i]):
            a, b = matrices[i]
            _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    start_time = time()
    iterations = 0
    last_update = start_time

    with tqdm(total=RUN_DURATION, desc="GPU stress test", unit="s") as pbar:
        while time() - start_time < RUN_DURATION:
            for i in range(NUM_STREAMS):
                with torch.cuda.stream(streams[i]):
                    a, b = matrices[i]
                    _ = torch.matmul(a, b)

            torch.cuda.synchronize()
            iterations += 1

            now = time()
            delta = now - last_update
            pbar.update(delta)
            last_update = now

    # cleanup
    del matrices
    del streams
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"Completed iterations: {iterations}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    print("Hourly GPU scheduler started.")

    while True:
        cycle_start = time()

        print("\nStarting scheduled GPU workload...")
        gpu_stress()

        elapsed = time() - cycle_start
        sleep_time = max(0, INTERVAL - elapsed)

        print(f"Sleeping {sleep_time/60:.1f} minutes until next run...\n")
        sleep(sleep_time)


if __name__ == "__main__":
    main()
