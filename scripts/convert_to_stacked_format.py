import torch
import torch.nn.utils.rnn as rnn_utils
from pathlib import Path
from tqdm import tqdm
import sys

def convert_to_stacked(input_path: Path, output_path: Path, max_length_cap: int = 2048):
    """Convert list of variable-length token tensors to stacked padded tensor with true lengths."""
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
    
    print(f"Loading {input_path}...")
    data = torch.load(input_path, weights_only=False)
    input_ids_list = data["input_ids"]
    if not isinstance(input_ids_list, list):
        raise ValueError(f"Expected 'input_ids' to be a list, got {type(input_ids_list)}")
    
    N = len(input_ids_list)
    print(f"Processing {N} sequences...")
    
    # Step 1: Truncate sequences exceeding cap, record true tokenized lengths
    truncated_tensors = []
    true_lengths = []
    for ids in tqdm(input_ids_list, desc="Truncating & recording lengths"):
        orig_len = len(ids)
        if orig_len > max_length_cap:
            true_len = max_length_cap
            truncated = ids[:max_length_cap]
        else:
            true_len = orig_len
            truncated = ids
        true_lengths.append(true_len)
        truncated_tensors.append(truncated)
    
    # Step 2: Pad all sequences to max true length (capped at max_length_cap)
    padded_input_ids = rnn_utils.pad_sequence(truncated_tensors, batch_first=True, padding_value=0)
    
    # Step 3: Memory-efficient dtypes (token IDs fit in int32, lengths ≤2048 fit in int16)
    input_ids_int32 = padded_input_ids.to(torch.int32)
    true_lengths_tensor = torch.tensor(true_lengths, dtype=torch.int16)
    
    # Step 4: Save stacked tensor + true lengths
    output_data = {
        "input_ids": input_ids_int32,
        "true_lengths": true_lengths_tensor
    }
    torch.save(output_data, output_path)
    print(f"Saved {output_path}")
    print(f"  input_ids: shape={input_ids_int32.shape}, dtype={input_ids_int32.dtype}")
    print(f"  true_lengths: shape={true_lengths_tensor.shape}, dtype={true_lengths_tensor.dtype}")
    return output_data

if __name__ == "__main__":
    BASE_DIR = Path("/storage/arik/nlp_final_project/data/processed_dpo_dataset")
    MAX_CAP = 2048
    
    # Convert chosen encodings
    convert_to_stacked(
        BASE_DIR / "chosen_encodings.pt",
        BASE_DIR / "chosen_stacked.pt",
        MAX_CAP
    )
    
    # Convert rejected encodings
    convert_to_stacked(
        BASE_DIR / "rejected_encodings.pt",
        BASE_DIR / "rejected_stacked.pt",
        MAX_CAP
    )
