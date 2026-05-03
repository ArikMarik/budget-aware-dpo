import torch
import time
import os
import gc
from torch.nn.utils.rnn import pad_sequence

def main():
    original_path = "data/processed_dpo_dataset/rejected_encodings.pt"
    padded_path = "data/processed_dpo_dataset/rejected_encodings_padded.pt"

    # Load original list of tensors
    print("Loading original data (list of tensors)...")
    start = time.time()
    original_data = torch.load(original_path)
    original_load_time = time.time() - start
    print(f"Original load time: {original_load_time:.4f} seconds")
    print(f"Number of tensors: {len(original_data['input_ids'])}")
    print(f"keys: {list(original_data.keys())}")
    print(f"First tensor shape: {original_data['input_ids'][0].shape}, dtype: {original_data['input_ids'][0].dtype}")

    # Check if padded file already exists
    if os.path.exists(padded_path):
        print(f"\nPadded file already exists at {padded_path}, skipping generation.")
    else:
        # Combine into single padded tensor
        print("\nCombining tensors into padded tensor...")
        start = time.time()
        padded_tensor = pad_sequence(original_data["input_ids"], batch_first=True, padding_value=0)
        padding_time = time.time() - start
        print(f"Padding time: {padding_time:.4f} seconds")
        print(f"Padded tensor shape: {padded_tensor.shape}, dtype: {padded_tensor.dtype}")

        # Save padded tensor
        print(f"\nSaving padded tensor to {padded_path}...")
        start = time.time()
        torch.save(padded_tensor, padded_path)
        save_time = time.time() - start
        file_size = os.path.getsize(padded_path) / (1024 ** 3)  # GB
        print(f"Save time: {save_time:.4f} seconds")
        print(f"Padded file size: {file_size:.2f} GB")

        # Clear padded tensor from memory
        del padded_tensor

    # Clear memory before loading padded tensor
    print("\nClearing memory...")
    del original_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleared.")

    # Load padded tensor
    print("\nLoading padded tensor...")
    start = time.time()
    loaded_padded = torch.load(padded_path, map_location="cpu")
    padded_load_time = time.time() - start
    print(f"Padded load time: {padded_load_time:.4f} seconds")

    # Compare results
    print("\n" + "="*50)
    print("LOADING SPEED COMPARISON RESULTS")
    print("="*50)
    print(f"Original (list) load time: {original_load_time:.4f} s")
    print(f"Padded tensor load time:    {padded_load_time:.4f} s")
    if padded_load_time < original_load_time:
        print(f"Speedup: {original_load_time / padded_load_time:.2f}x faster")
    else:
        print(f"Slowdown: {padded_load_time / original_load_time:.2f}x slower")
    print("="*50)

if __name__ == "__main__":
    main()
