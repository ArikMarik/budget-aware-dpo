import torch
import time
import os
import gc
from torch.nn.utils.rnn import pad_sequence

def main():
    chosen_encodings_path = "data/processed_dpo_dataset/chosen_encodings.pt"
    rejected_encodings_path = "data/processed_dpo_dataset/rejected_encodings.pt"

    # Load chosen list of tensors
    print("Loading chosen data (list of tensors)...")
    start = time.time()
    chosen_data = torch.load(chosen_encodings_path, weights_only=False)
    chosen_load_time = time.time() - start
    print(f"Chosen load time: {chosen_load_time:.4f} seconds")
    print(f"Number of tensors: {len(chosen_data['input_ids'])}")
    print(f"keys: {list(chosen_data.keys())}")
    print(f"Inputs tensor shape: {chosen_data['input_ids'].shape}, dtype: {chosen_data['input_ids'].dtype}")
    print(f"Length tensor shape: {chosen_data['true_lengths'].shape}, dtype: {chosen_data['true_lengths'].dtype}")


    # Clear memory before loading padded tensor
    print("\nClearing memory...")
    del chosen_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleared.")

    # Load rejected list of tensors
    print("Loading rejected data (list of tensors)...")
    start = time.time()
    rejected_data = torch.load(rejected_encodings_path, weights_only=False)
    rejected_load_time = time.time() - start
    print(f"Rejected load time: {rejected_load_time:.4f} seconds")
    print(f"Number of tensors: {len(rejected_data['input_ids'])}")
    print(f"keys: {list(rejected_data.keys())}")
    print(f"Inputs tensor shape: {rejected_data['input_ids'].shape}, dtype: {rejected_data['input_ids'].dtype}")
    print(f"Length tensor shape: {rejected_data['true_lengths'].shape}, dtype: {rejected_data['true_lengths'].dtype}")

    # Compare results
    print("\n" + "="*50)
    print("LOADING SPEED COMPARISON RESULTS")
    print("="*50)
    print(f"Chosen load time: {chosen_load_time:.4f} s")
    print(f"Rejected load time:    {rejected_load_time:.4f} s")
    if rejected_load_time < chosen_load_time:
        print(f"Speedup: {chosen_load_time / rejected_load_time:.2f}x faster")
    else:
        print(f"Slowdown: {rejected_load_time / chosen_load_time:.2f}x slower")
    print("="*50)

if __name__ == "__main__":
    main()
