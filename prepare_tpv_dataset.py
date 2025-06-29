import os
import json
import numpy as np
import random
import pickle
import glob

seed = 21
input_base_dir = "llama_model_results_batched"
output_dir = "llama_math_tpv_dataset"
train_split_ratio = 0.8
end_of_thinking_token = "</think>"
print(f"开始创建数据集 input: {input_base_dir}, output: {output_dir}, seed: {seed}")

if not os.path.isdir(input_base_dir):
    print(f"Error: Input directory '{input_base_dir}' not found.")
    quit()

problem_data_map = {}  # Stores {problem_id: [list_of_records_for_this_problem]}
# ['llama_model_results_batched/problem_xxx']
problem_dirs = sorted(glob.glob(os.path.join(input_base_dir, "problem_*")))
if not problem_dirs:
    print(f"No 'problem_*' directories found in '{input_base_dir}'.")
    quit()
# 'llama_model_results_batched/problem_0'
for problem_dir_path in problem_dirs:
    if not os.path.isdir(problem_dir_path):
        continue
    try:
        problem_id_str = os.path.basename(problem_dir_path).split("_")[-1] # 提取问题数字 xxx
        problem_id = int(problem_id_str)
    except (IndexError, ValueError):
        print(
            f"Warning: Could not parse problem ID from directory '{problem_dir_path}'. Skipping."
        )
        continue
    print(f"\nProcessing Problem ID: {problem_id} (from {problem_dir_path})")
    current_problem_records = []
    # llama_model_results_batched/problem_0/generation_yyy
    generation_dirs = sorted(glob.glob(os.path.join(problem_dir_path, "generation_*")))
    if not generation_dirs:
        print(f"  No 'generation_*' directories found in '{problem_dir_path}'.")
        continue
    for gen_dir_path in generation_dirs:
        if not os.path.isdir(gen_dir_path):
            continue
        # tokens_path = 'llama_model_results_batched/problem_0/generation_0/tokens.json'
        tokens_path = os.path.join(gen_dir_path, "tokens.json")
        metadata_path = os.path.join(gen_dir_path, "metadata.json")
        hs_input_path = os.path.join(
            gen_dir_path, "token_hidden_states_input.npy"
        )  # 'llama_model_results_batched/problem_0/generation_0/token_hidden_states_input.npy'
        hs_output_path = os.path.join(gen_dir_path, "token_hidden_states_output.npy")
        error_path = os.path.join(gen_dir_path, "error.txt")
        if os.path.exists(error_path):
            print(
                f"Skipping generation {os.path.basename(gen_dir_path)} due to error file."
            )
            continue
        # Updated check to include hs_input_path
        required_files = [tokens_path, metadata_path, hs_input_path, hs_output_path]
        if not all(os.path.exists(p) for p in required_files):
            print(
                f"Warning: Missing one or more required files in '{gen_dir_path}'. Skipping generation."
            )
            continue
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            with open(tokens_path, "r", encoding="utf-8") as f:
                full_tokens_list = json.load(f)
            input_hs_array = np.load(hs_input_path)  # Load input hidden states
            output_hs_array = np.load(hs_output_path)
            prompt_len = metadata.get("input_prompt_length_tokens") # 67
            actual_gen_len = metadata.get("output_generated_length_tokens") # 755
            if prompt_len is None or actual_gen_len is None:
                print(
                    f"Warning: Missing token length information in metadata for '{gen_dir_path}'. Skipping."
                )
                continue
            # Assertions for data integrity
            assert (
                input_hs_array.shape[0] == prompt_len
            ), f"Input HS length mismatch in {gen_dir_path}: Got {input_hs_array.shape[0]}, expected {prompt_len} (prompt_len from metadata)."
            assert (
                output_hs_array.shape[0] == actual_gen_len
            ), f"Output HS length mismatch in {gen_dir_path}: Got {output_hs_array.shape[0]}, expected {actual_gen_len} (actual_gen_len from metadata)."
            # Extract the generated tokens that correspond to the output_hs_array
            # This slice correctly represents the tokens for which output hidden states were saved.
            generated_tokens_actual = full_tokens_list[
                prompt_len : prompt_len + actual_gen_len
            ] # 获取实际生成的 tokens
            if not generated_tokens_actual:  # No generated tokens (actual_gen_len is 0)
                # print(f"  No generated tokens for {os.path.basename(gen_dir_path)} (actual_gen_len is 0). Skipping.")
                continue
            try:
                # Search for end_of_thinking_token within the tokens that have corresponding hidden states
                think_idx_in_gen = generated_tokens_actual.index(end_of_thinking_token) # '</think>'的位置=573
            except ValueError:
                # print(f"  '{end_of_thinking_token}' not found in generated output of '{gen_dir_path}'. Skipping generation.")
                continue  # </think> token not found in this generation's output
            # If found, N is the number of tokens up to and including </think>
            # within the generated_tokens_actual sequence.
            N = think_idx_in_gen + 1
            # Collect records (hidden_state, i/N) for this generation
            # The hidden states are from output_hs_array, which corresponds to generated_tokens_actual
            for j in range(N):  # Iterate from 0 up to think_idx_in_gen (inclusive)
                hidden_state = output_hs_array[j]
                relative_position = (j + 1) / N  # 1-based index for 'i'
                current_problem_records.append((hidden_state, relative_position))
            # print(f"  Successfully processed generation {os.path.basename(gen_dir_path)}, found '{end_of_thinking_token}' at index {think_idx_in_gen}, N={N}. Added {N} records.")
        except AssertionError as e_assert:  # Catch assertion errors specifically
            print(
                f"  AssertionError in {gen_dir_path}: {e_assert}. Skipping generation."
            )
            continue
        except Exception as e:
            print(f"  Error processing generation '{gen_dir_path}': {e}. Skipping.")
            continue
    if current_problem_records:
        problem_data_map[problem_id] = current_problem_records
        print(
            f"  Problem ID {problem_id}: Collected {len(current_problem_records)} records from its valid generations."
        )
    else:
        print(
            f"  Problem ID {problem_id}: No valid records collected (no generations contained '{end_of_thinking_token}' or other issues)."
        )
if not problem_data_map:
    print("No problems with valid generations found. Exiting.")
    quit()
# Train/Test Split based on problem IDs
all_problem_ids = list(problem_data_map.keys()) # 存在有效数据的Problem ID
random.seed(seed)
random.shuffle(all_problem_ids)
num_train = int(len(all_problem_ids) * train_split_ratio)
train_problem_ids = all_problem_ids[:num_train]
test_problem_ids = all_problem_ids[num_train:]
print(f"\nTotal problems with valid data: {len(all_problem_ids)}")
print(
    f"Splitting into {len(train_problem_ids)} train problems and {len(test_problem_ids)} test problems."
)
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
    print(
        f"\nTraining dataset saved to '{train_path}' with {len(train_dataset)} records."
    )
    with open(test_path, "wb") as f:
        pickle.dump(test_dataset, f)
    print(f"Test dataset saved to '{test_path}' with {len(test_dataset)} records.")
except Exception as e:
    print(f"Error saving datasets: {e}")
print("\nDataset creation process complete.")
