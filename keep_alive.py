import torch
import time
import gc

SIZE = 8192
NUM_STREAMS = 4
DURATION = 600  # seconds (10 minutes)

device = torch.device("cuda")

def main():
    matrices = []

    for _ in range(NUM_STREAMS):
        a = torch.randn(SIZE, SIZE, device=device)
        b = torch.randn(SIZE, SIZE, device=device)
        matrices.append((a, b))

    streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]

    start = time.time()

    while time.time() - start < DURATION:
        for i in range(NUM_STREAMS):
            with torch.cuda.stream(streams[i]):
                a, b = matrices[i]
                torch.matmul(a, b)

        torch.cuda.synchronize()

    # cleanup
    del matrices
    del streams
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
