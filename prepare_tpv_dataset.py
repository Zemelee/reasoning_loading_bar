# This software is for non-commercial use only.
# Commercial use requires a separate license.

import os
import json
import numpy as np
import random
import argparse
import pickle
import glob

def create_regression_dataset(input_base_dir, output_dir, seed=42, train_split_ratio=0.8, end_of_thinking_token="</think>"):
    """
    Loads data saved by extract_llm_hidden_states.py, filters generations,
    and creates a regression dataset.

    Args:
        input_base_dir (str): The directory where results from extract_llm_hidden_states.py are stored.
        output_dir (str): Directory to save the processed train/test datasets.
        seed (int): Random seed for train/test split.
        train_split_ratio (float): Ratio of problems for the training set.
        end_of_thinking_token (str): The token to look for to mark the end of relevant output.
    """
    print(f"Starting dataset creation with input: {input_base_dir}, output: {output_dir}, seed: {seed}")

    if not os.path.isdir(input_base_dir):
        print(f"Error: Input directory '{input_base_dir}' not found.")
        return

    problem_data_map = {} # Stores {problem_id: [list_of_records_for_this_problem]}
    problem_dirs = sorted(glob.glob(os.path.join(input_base_dir, "problem_*")))

    if not problem_dirs:
        print(f"No 'problem_*' directories found in '{input_base_dir}'.")
        return

    for problem_dir_path in problem_dirs:
        if not os.path.isdir(problem_dir_path):
            continue
        
        try:
            problem_id_str = os.path.basename(problem_dir_path).split('_')[-1]
            problem_id = int(problem_id_str)
        except (IndexError, ValueError):
            print(f"Warning: Could not parse problem ID from directory '{problem_dir_path}'. Skipping.")
            continue

        print(f"\nProcessing Problem ID: {problem_id} (from {problem_dir_path})")
        current_problem_records = []
        generation_dirs = sorted(glob.glob(os.path.join(problem_dir_path, "generation_*")))

        if not generation_dirs:
            print(f"  No 'generation_*' directories found in '{problem_dir_path}'.")
            continue

        for gen_dir_path in generation_dirs:
            if not os.path.isdir(gen_dir_path):
                continue

            # Define paths for the necessary files
            tokens_path = os.path.join(gen_dir_path, "tokens.json")
            metadata_path = os.path.join(gen_dir_path, "metadata.json")
            hs_input_path = os.path.join(gen_dir_path, "token_hidden_states_input.npy") # Added
            hs_output_path = os.path.join(gen_dir_path, "token_hidden_states_output.npy")
            error_path = os.path.join(gen_dir_path, "error.txt")

            if os.path.exists(error_path):
                print(f"  Skipping generation {os.path.basename(gen_dir_path)} due to error file.")
                continue

            # Updated check to include hs_input_path
            required_files = [tokens_path, metadata_path, hs_input_path, hs_output_path]
            if not all(os.path.exists(p) for p in required_files):
                print(f"  Warning: Missing one or more required files in '{gen_dir_path}'. Skipping generation.")
                # for req_f in required_files:
                #     if not os.path.exists(req_f):
                #         print(f"    Missing: {req_f}")
                continue

            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                with open(tokens_path, 'r', encoding='utf-8') as f:
                    full_tokens_list = json.load(f)
                
                input_hs_array = np.load(hs_input_path) # Load input hidden states
                output_hs_array = np.load(hs_output_path)

                prompt_len = metadata.get("input_prompt_length_tokens")
                actual_gen_len = metadata.get("output_generated_length_tokens")

                if prompt_len is None or actual_gen_len is None:
                    print(f"  Warning: Missing token length information in metadata for '{gen_dir_path}'. Skipping.")
                    continue
                
                # Assertions for data integrity
                assert input_hs_array.shape[0] == prompt_len, \
                    f"Input HS length mismatch in {gen_dir_path}: Got {input_hs_array.shape[0]}, expected {prompt_len} (prompt_len from metadata)."
                
                assert output_hs_array.shape[0] == actual_gen_len, \
                    f"Output HS length mismatch in {gen_dir_path}: Got {output_hs_array.shape[0]}, expected {actual_gen_len} (actual_gen_len from metadata)."

                # Extract the generated tokens that correspond to the output_hs_array
                # This slice correctly represents the tokens for which output hidden states were saved.
                generated_tokens_actual = full_tokens_list[prompt_len : prompt_len + actual_gen_len]
                
                if not generated_tokens_actual: # No generated tokens (actual_gen_len is 0)
                    # print(f"  No generated tokens for {os.path.basename(gen_dir_path)} (actual_gen_len is 0). Skipping.")
                    continue

                try:
                    # Search for end_of_thinking_token within the tokens that have corresponding hidden states
                    think_idx_in_gen = generated_tokens_actual.index(end_of_thinking_token)
                except ValueError:
                    # print(f"  '{end_of_thinking_token}' not found in generated output of '{gen_dir_path}'. Skipping generation.")
                    continue # </think> token not found in this generation's output

                # If found, N is the number of tokens up to and including </think>
                # within the generated_tokens_actual sequence.
                N = think_idx_in_gen + 1
                
                # Collect records (hidden_state, i/N) for this generation
                # The hidden states are from output_hs_array, which corresponds to generated_tokens_actual
                for j in range(N): # Iterate from 0 up to think_idx_in_gen (inclusive)
                    hidden_state = output_hs_array[j]
                    relative_position = (j + 1) / N # 1-based index for 'i'
                    current_problem_records.append((hidden_state, relative_position))
                
                # print(f"  Successfully processed generation {os.path.basename(gen_dir_path)}, found '{end_of_thinking_token}' at index {think_idx_in_gen}, N={N}. Added {N} records.")

            except AssertionError as e_assert: # Catch assertion errors specifically
                print(f"  AssertionError in {gen_dir_path}: {e_assert}. Skipping generation.")
                continue
            except Exception as e:
                print(f"  Error processing generation '{gen_dir_path}': {e}. Skipping.")
                continue
        
        if current_problem_records:
            problem_data_map[problem_id] = current_problem_records
            print(f"  Problem ID {problem_id}: Collected {len(current_problem_records)} records from its valid generations.")
        else:
            print(f"  Problem ID {problem_id}: No valid records collected (no generations contained '{end_of_thinking_token}' or other issues).")


    if not problem_data_map:
        print("No problems with valid generations found. Exiting.")
        return

    # Train/Test Split based on problem IDs
    all_problem_ids = list(problem_data_map.keys())
    random.seed(seed)
    random.shuffle(all_problem_ids)

    num_train = int(len(all_problem_ids) * train_split_ratio)
    train_problem_ids = all_problem_ids[:num_train]
    test_problem_ids = all_problem_ids[num_train:]

    print(f"\nTotal problems with valid data: {len(all_problem_ids)}")
    print(f"Splitting into {len(train_problem_ids)} train problems and {len(test_problem_ids)} test problems.")

    train_dataset = []
    for pid in train_problem_ids:
        train_dataset.extend(problem_data_map[pid])

    test_dataset = []
    for pid in test_problem_ids:
        test_dataset.extend(problem_data_map[pid])

    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_regression_dataset.pkl")
    test_path = os.path.join(output_dir, "test_regression_dataset.pkl")

    try:
        with open(train_path, "wb") as f:
            pickle.dump(train_dataset, f)
        print(f"\nTraining dataset saved to '{train_path}' with {len(train_dataset)} records.")

        with open(test_path, "wb") as f:
            pickle.dump(test_dataset, f)
        print(f"Test dataset saved to '{test_path}' with {len(test_dataset)} records.")
    except Exception as e:
        print(f"Error saving datasets: {e}")

    print("\nDataset creation process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a regression dataset from saved LLM hidden states and tokens. "
                    "Filters for generations containing a specific token (e.g., '</think>') "
                    "and creates (hidden_state, i/N) records."
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="llama_model_results_batched", 
        help="Base directory of the saved model results (the 'output_dir' from the extraction script)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="llama_math_tpv_dataset", 
        help="Directory to save the processed train and test dataset .pkl files."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for the train/test split of problems."
    )
    parser.add_argument(
        "--train_split_ratio", 
        type=float, 
        default=0.8, 
        help="Ratio of problems to include in the training set (0.0 to 1.0)."
    )
    parser.add_argument(
        "--end_of_thinking_token",
        type=str,
        default="</think>",
        help="The special token that marks the end of the relevant sequence portion in the generated output."
    )
    
    args = parser.parse_args()

    if not (0.0 <= args.train_split_ratio <= 1.0):
        print("Error: train_split_ratio must be between 0.0 and 1.0.")
    else:
        create_regression_dataset(
            args.input_dir, 
            args.output_dir, 
            args.seed, 
            args.train_split_ratio,
            args.end_of_thinking_token
        )
