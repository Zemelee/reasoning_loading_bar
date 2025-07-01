import os
import json
import numpy as np
import random
import pickle
import glob

seed = 11
input_base_dir = "llama_model_results_batched"
output_dir = "llama_math_tpv_dataset"
train_split_ratio = 0.8
end_of_thinking_token = "</think>"
print(f"开始创建数据集 input: {input_base_dir}, output: {output_dir}, seed: {seed}")

problem_data_map = {}
# ['llama_model_results_batched/problem_xxx']
problem_dirs = sorted(glob.glob(os.path.join(input_base_dir, "problem_*")))

# 'llama_model_results_batched/problem_0'
for p_dir_path in problem_dirs:
    if not os.path.isdir(p_dir_path):
        continue
    try:
        # 提取问题数字 xxx
        problem_id_str = os.path.basename(p_dir_path).split("_")[-1]
        problem_id = int(problem_id_str) # 0
    except (IndexError, ValueError):
        print(f"Warning: 无法从 {p_dir_path} 解析PID.")
        continue
    print(f"正在处理PID: {problem_id} (from {p_dir_path})")
    current_problem_records = []
    # llama_model_results_batched/problem_0/generation_yyy
    generation_dirs = sorted(glob.glob(os.path.join(p_dir_path, "generation_*")))
    if not generation_dirs:
        print(f"{p_dir_path}中未找到'generation_*'目录.")
        continue
    # llama_model_results_batched/problem_0/generation_0
    for gen_dir_path in generation_dirs:
        if not os.path.isdir(gen_dir_path):
            continue
        # tokens_path = 'llama_model_results_batched/problem_0/generation_0/tokens.json'
        tokens_path = os.path.join(gen_dir_path, "tokens.json")
        metadata_path = os.path.join(gen_dir_path, "metadata.json")
        # 'llama_model_results_batched/problem_0/generation_0/token_hidden_states_input.npy'
        hs_input_path = os.path.join(gen_dir_path, "token_hidden_states_input.npy")
        hs_output_path = os.path.join(gen_dir_path, "token_hidden_states_output.npy")
        error_path = os.path.join(gen_dir_path, "error.txt")
        if os.path.exists(error_path):
            print(f"文件错误，跳过生成: {os.path.basename(gen_dir_path)}.")
            continue
        # Updated check to include hs_input_path
        required_files = [tokens_path, metadata_path, hs_input_path, hs_output_path]
        if not all(os.path.exists(p) for p in required_files):
            print(f"警告：'{gen_dir_path}' 中缺少必需文件!")
            continue
        try:
            # 加载数据并验证一致性
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            with open(tokens_path, "r", encoding="utf-8") as f:
                full_tokens_list = json.load(f) # token序列
            input_hs_array = np.load(hs_input_path)  # 输入隐藏状态
            output_hs_array = np.load(hs_output_path) # 输出隐藏状态
            prompt_len = metadata.get("input_prompt_length_tokens")  # 67
            actual_gen_len = metadata.get("output_generated_length_tokens")  # 755
            if prompt_len is None or actual_gen_len is None:
                print(f"警告：'{gen_dir_path}' 的metadata中缺少token长度信息!")
                continue
            # Assertions for data integrity
            assert (
                input_hs_array.shape[0] == prompt_len
            ), f"Input HS length mismatch in {gen_dir_path}: Got {input_hs_array.shape[0]}, expected {prompt_len} (prompt_len from metadata)."
            assert (
                output_hs_array.shape[0] == actual_gen_len
            ), f"Output HS length mismatch in {gen_dir_path}: Got {output_hs_array.shape[0]}, expected {actual_gen_len} (actual_gen_len from metadata)."
            # 定位 end_of_thinking_token 并提取有效 token 序列， 表示获取实际生成的 tokens
            generated_tokens_actual = full_tokens_list[
                prompt_len : prompt_len + actual_gen_len
            ]
            if not generated_tokens_actual:  # No generated tokens (actual_gen_len is 0)
                print(f"{os.path.basename(gen_dir_path)}无token: actual_gen_len=!")
                continue
            try:
                # '</think>'的位置=573
                think_idx_in_gen = generated_tokens_actual.index(end_of_thinking_token)
            except ValueError:
                print(f"未在输出中找到'{end_of_thinking_token}'-'{gen_dir_path}'.")
                continue  # 未在生成输出中找到</think>
            # 如果找到，则 N 是 generated_tokens_actual 序列中从开始到包含 end_of_thinking_token 的 token 总数。
            N = think_idx_in_gen + 1
            # 构造训练样本
            for j in range(N):
                # 每个 token 的输出隐藏状态对应一个
                hidden_state = output_hs_array[j]
                relative_position = (j + 1) / N  # 计算相对位置0~1
                current_problem_records.append((hidden_state, relative_position))
        except AssertionError as e_assert:  # Catch assertion errors specifically
            print(f"AssertionError {gen_dir_path}: {e_assert}.")
            continue
        except Exception as e:
            print(f"处理生成时发生错误：'{gen_dir_path}': {e}. 跳过.")
            continue
    if current_problem_records:
        problem_data_map[problem_id] = current_problem_records
        print(f"问题ID {problem_id}：收集 {len(current_problem_records)} 条记录。")
    else:
        print(f"问题ID {problem_id}：未收集到任何有效记录")
if not problem_data_map:
    print("未找到具有有效生成结果的问题，程序退出.")
    quit()

# 按照问题 ID 划分训练集和测试集
all_problem_ids = list(problem_data_map.keys())
random.seed(seed)
random.shuffle(all_problem_ids)
num_train = int(len(all_problem_ids) * train_split_ratio)
train_problem_ids = all_problem_ids[:num_train]
test_problem_ids = all_problem_ids[num_train:]
print(f"\n 有效总问题数: {len(all_problem_ids)}")
print(f"训练{len(train_problem_ids)}\测试{len(test_problem_ids)}")

train_dataset = []
for pid in train_problem_ids:
    train_dataset.extend(problem_data_map[pid])
test_dataset = []
for pid in test_problem_ids:
    test_dataset.extend(problem_data_map[pid])
# 合并所有记录并保存为 pickle 文件(隐藏状态+相对位置)
os.makedirs(output_dir, exist_ok=True)
train_path = os.path.join(output_dir, "train_regression_dataset.pkl")
test_path = os.path.join(output_dir, "test_regression_dataset.pkl")
try:
    with open(train_path, "wb") as f:
        pickle.dump(train_dataset, f)
    print(f"训练集已保存至{train_path}, {len(train_dataset)} 条记录")
    with open(test_path, "wb") as f:
        pickle.dump(test_dataset, f)
    print(f"测试数据集已保存至{test_path}, {len(test_dataset)} 条记录。")
except Exception as e:
    print(f"保存数据集时发生错误：{e}")
