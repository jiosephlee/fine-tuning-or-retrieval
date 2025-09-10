import sys
import os
import torch
from transformers import AutoTokenizer

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.data_preparation import PretrainingDataReplay, get_pretraining_batches, fill_up_batch_with_pretraining_chunks

def test_pretraining_module(output_file):
    """
    Tests the pretraining data module and writes the output to a file.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", trust_remote_code=True)
    
    # --- Test Configuration ---
    replay_file_path = "data/Olmo/dclm_10M_tokens.npy"
    chunk_size = 64  # Small chunk size for easy inspection
    batch_size = 4
    
    # --- Initialization ---
    data_replay = PretrainingDataReplay(replay_file_path)
    
    with open(output_file, "w") as f:
        f.write("--- Pretraining Module Test --- \n\n")

        # --- Test 1: Basic functionality of get_pretraining_batches ---
        f.write("--- Test 1: get_pretraining_batches ---\n")
        num_batches_to_get = 2
        pretraining_chunks = get_pretraining_batches(data_replay, num_batches_to_get, batch_size, chunk_size)
        
        assert len(pretraining_chunks) == num_batches_to_get * batch_size
        for chunk in pretraining_chunks:
            assert chunk.shape == (chunk_size,)
            
        f.write(f"Requested {num_batches_to_get} batches of size {batch_size}. Total chunks received: {len(pretraining_chunks)}\n\n")
        
        for i, chunk in enumerate(pretraining_chunks[:2]):  # Display first 2 chunks
            f.write(f"## Chunk {i+1} Tokens ##\n{chunk.tolist()}\n")
            f.write(f"## Chunk {i+1} Decoded ##\n'{tokenizer.decode(chunk)}'\n\n")

        # --- Test 2: fill_up_batch_with_pretraining_chunks ---
        f.write("--- Test 2: fill_up_batch_with_pretraining_chunks ---\n")
        ideal_batch_size = 8
        initial_batch = [torch.randint(0, 1000, (chunk_size,)) for _ in range(3)]
        f.write(f"Initial batch size: {len(initial_batch)}, Ideal batch size: {ideal_batch_size}\n")
        
        filled_batch = fill_up_batch_with_pretraining_chunks(initial_batch, data_replay, ideal_batch_size, chunk_size)
        
        assert len(filled_batch) == ideal_batch_size
        for chunk in filled_batch:
            assert chunk.shape == (chunk_size,)
            
        f.write(f"Filled batch size: {len(filled_batch)}\n\n")
        f.write(f"Displaying the {len(filled_batch) - 3} new chunks added from pretraining data:\n")
        for i, chunk in enumerate(filled_batch[3:]):
            f.write(f"## New Chunk {i+1} Tokens ##\n{chunk.tolist()}\n")
            f.write(f"## New Chunk {i+1} Decoded ##\n'{tokenizer.decode(chunk)}'\n\n")

        # --- Test 3: Sequential calls to verify data replay ---
        f.write("--- Test 3: Sequential calls to verify data replay ---\n")
        f.write("Fetching 1 more chunk to see if it continues from where Test 2 left off.\n")
        next_chunk_batch = get_pretraining_batches(data_replay, 1, 1, chunk_size)
        
        assert len(next_chunk_batch) == 1
        
        next_chunk = next_chunk_batch[0]
        
        assert next_chunk.shape == (chunk_size,)
        
        f.write(f"## Next Chunk Tokens ##\n{next_chunk.tolist()}\n")
        f.write(f"## Next Chunk Decoded ##\n'{tokenizer.decode(next_chunk)}'\n\n")
        
        f.write("Test complete. Check the decoded text to ensure the chunks are coherent and sequential.")

if __name__ == "__main__":
    # Create the testing directory if it doesn't exist
    import os
    os.makedirs("scripts/testing", exist_ok=True)
    
    output_filename = "scripts/testing/pretraining_module_test_output.txt"
    test_pretraining_module(output_filename)
    print(f"Test output written to {output_filename}")
