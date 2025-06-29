import gc
import json
import os
import random
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_problems

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"


# 一批序列的隐状态提取优化生成
def generate_batch_with_last_hidden_states(
    model,
    inputs,
    tokenizer,
    num_return_sequences,
    max_new_tokens,
    temperature,
    do_sample,
    top_p,
):
    input_ids = inputs["input_ids"]  # [batch_size, seq_len] 提取输入张量
    attention_mask = inputs["attention_mask"]  # # [batch_size, seq_len]
    start_time = time.time()
    # 生成文本
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
        top_p=top_p if do_sample else None,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )
    generation_time = time.time() - start_time

    #  [(layer_0_hs_prompt, layer_1_hs_prompt, ..., layer_n_hs_prompt),   # 第 0 步（prompt）
    #  (layer_0_hs_step1, layer_1_hs_step1, ..., layer_n_hs_step1),     # 第 1 步（生成第一个 token）
    #  (layer_0_hs_step2, layer_1_hs_step2, ..., layer_n_hs_step2),     # 第 2 步（生成第二个 token）
    # 每一个元素对应生成过程中的一个 token 步骤（包括 prompt）
    # 每个步骤的 hidden states 是一个 tuple
    # 其中 layer_n_hs.shape: (batch_size, seq_len, hidden_size)
    # 提取输入提示（prompt）中最后一个 token 的隐藏状态hidden_states，作为该 prompt 的语义表示
    # hidden_states.shape = [num_return_seq(4), seq_len(prompt_token长度), hidden_size]
    # 最后一层的隐藏状态通常用于预测下一个 token

    prompt_last_layer_hs_batch = outputs.hidden_states[0][-1]
    last_layer_hidden_states_input_list = [
        # detach将张量从计算图分离(不参与微分)
        prompt_last_layer_hs_batch[k].detach().cpu().numpy()
        for k in range(num_return_sequences)
    ]

    last_layer_hidden_states_output_list = []
    prompt_len = input_ids.shape[1]  # (batch_size, seq_len)

    model_hidden_size = getattr(model.config, "hidden_size", None)
    if model_hidden_size is None and outputs.hidden_states and outputs.hidden_states[0]:
        model_hidden_size = outputs.hidden_states[0][-1].shape[-1]
    if model_hidden_size is None:
        raise ValueError("无法确定空隐藏状态数组的模型隐藏大小.")

    all_responses = []
    all_tokens_lists = []
    actual_lengths_generated_list = []
    for k in range(num_return_sequences):  # 单问题生成的多个答案
        # 拿到第k个答案=提示词+生成
        current_sequence_ids_k = outputs.sequences[k]  # token_ids
        # 解码第k个答案
        response_k = tokenizer.decode(current_sequence_ids_k, skip_special_tokens=True)
        all_responses.append(response_k)

        # 答案 转 token词
        tokens_k_full_sequence = tokenizer.convert_ids_to_tokens(current_sequence_ids_k)
        all_tokens_lists.append(tokens_k_full_sequence)

        # 确定为序列k生成的令牌的实际数量（不包括提示和填充）
        sequence_k_generated_ids_only = current_sequence_ids_k[prompt_len:]
        # 计算每个生成序列中实际有效的 token 数量
        actual_gen_len_k = 0  # 实际生成的 token 数量
        for token_id in sequence_k_generated_ids_only:
            if token_id == tokenizer.pad_token_id:
                break
            actual_gen_len_k += 1
            if token_id == tokenizer.eos_token_id:
                break
        actual_lengths_generated_list.append(actual_gen_len_k)

        # 存储k序列中 tokens 的最后一层隐藏状态
        gen_k_output_hs_for_actual_tokens = []
        # 可用的隐藏状态步骤数量
        num_hs_steps_available_for_gen_tokens = len(outputs.hidden_states)
        # 提取每个生成 token 的最后一层隐藏状态
        for step_offset in range(
            min(actual_gen_len_k, num_hs_steps_available_for_gen_tokens)
        ):
            hs_tensor_for_step = outputs.hidden_states[step_offset][-1]
            # 选择第 k 个序列的第 0 个位置的隐藏状态
            token_hs = hs_tensor_for_step[k, 0, :].detach().cpu().numpy()
            gen_k_output_hs_for_actual_tokens.append(token_hs)

        if gen_k_output_hs_for_actual_tokens:
            last_layer_hidden_states_output_list.append(
                np.array(gen_k_output_hs_for_actual_tokens)
            )
        else:  # Handle cases where actual_gen_len_k is 0
            last_layer_hidden_states_output_list.append(
                np.empty((0, model_hidden_size), dtype=np.float32)
            )

    return (
        all_responses,  # 文本答案 * k
        all_tokens_lists,  # tokens词 * k
        last_layer_hidden_states_input_list,  # outputs.hidden_states[0][-1] * k
        last_layer_hidden_states_output_list,  # 每个生成 token 的最后一层隐藏状态 * k
        actual_lengths_generated_list,  # 实际生成的 token 数量 * k
        generation_time,  # 耗时 * k
    )






# Example usage:
if __name__ == "__main__":
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_new_tokens=1024,
    # 数据集
    dataset="math500",  # math500 or gsm8k
    start_problem_index=0,
    end_problem_index=4,
    gens_per_p=3,
    output_dir="llama_model_results_batched",
    prompt_template="{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n<think>\n",
    temperature=0.6,
    do_sample=True,  #
    top_p=0.95,
    seed=42,
    results = {
        "config": {
            "model_name": model_name,
            "max_new_tokens": max_new_tokens,
            "dataset_name": dataset,
            "gens_per_p": gens_per_p,
            "prompt_template": prompt_template,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": top_p,
            "seed": seed,
        },
        "problems": {},
    }

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Sets seed for all GPUs
        print(f"[INFO] Seeds set to {seed} for random, numpy, and torch.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set memory-efficient configuration
    if device == "cuda":
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Model loading
    print(f"Loading model {model_name} and tokenizer...")
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("tokenizer.pad_token:", tokenizer.pad_token)  # <｜end▁of▁sentence｜>
    print("tokenizer.eos_token:", tokenizer.eos_token)  # <｜end▁of▁sentence｜>
    print("tokenizer.bos_token:", tokenizer.bos_token)  # <｜begin▁of▁sentence｜>
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad_token was None, set to eos_token: {tokenizer.eos_token}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Distributes model across available GPUs/CPU
        torch_dtype=torch.float16,
    )

    # 确保模型配置中包含 hidden_size 这对构造空的隐藏状态数组非常关键
    if not hasattr(model, "config") or not hasattr(model.config, "hidden_size"):
        # 如果没有直接可用的 hidden_size（通常在 HF 模型中是有的），尝试回退处理
        print("[Warning] 未找到 model.config.hidden_size。尝试推断或使用默认值。")
        # 作为最后的手段，可以尝试从某一层中获取，或者抛出错误。
        # 此处暂时假设之后可以通过 outputs.hidden_states[0][-1].shape[-1] 获取

    # Load dataset
    print(f"Loading dataset {dataset}")
    problems = load_problems(
        dataset, starting_index=start_problem_index, end_index=end_problem_index
    )
    assert len(problems) > 0
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(results["config"], f, indent=2)
    results["config_path"] = config_path

    # 对于数据集的每个问题
    for problem_idx, problem_text_content in enumerate(problems):
        actual_idx = problem_idx + start_problem_index
        print(f"\nProcessing problem {actual_idx}...")
        problem_dir = os.path.join(output_dir, f"problem_{actual_idx}")
        os.makedirs(problem_dir, exist_ok=True)
        p_res_entry = {
            # 'llama_model_results_batched/problem_0/original_problem.txt'
            "original_problem_path": os.path.join(problem_dir, "original_problem.txt"),
            "formatted_prompt_path": os.path.join(problem_dir, "formatted_prompt.txt"),
            "generations": {},
        }

        # 写入原始和格式化后的问题
        with open(p_res_entry["original_problem_path"], "w", encoding="utf-8") as f:
            f.write(problem_text_content)
        # problem + COT + {{}} + <think>
        formatted_problem = prompt_template.format(problem=problem_text_content)
        with open(p_res_entry["formatted_prompt_path"], "w", encoding="utf-8") as f:
            f.write(formatted_problem)

        inputs = tokenizer(formatted_problem, return_tensors="pt").to(device)
        try:
            # Generate all sequences for this problem in one batch
            print(
                f"Generating {gens_per_p} sequences for problem {actual_idx} in a batch..."
            )
            (
                all_responses,  # 文本答案 * k
                all_tokens_lists,  # tokens词 * k
                all_hs_inputs,  # outputs.hidden_states[0][-1] * k
                all_hs_outputs,  # 每个生成 token 的最后一层隐藏状态 * k
                all_actual_lengths,  # 实际生成的 token 数量 * k
                batch_generation_time,  # 耗时 * k
            ) = generate_batch_with_last_hidden_states(
                model,
                inputs,
                tokenizer,
                num_return_sequences=gens_per_p, # k
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
            )

            print(
                f"Batch generation for problem {actual_idx} completed in {batch_generation_time:.2f} seconds."
            )

            # 处理batch中的每个生成
            for gen_idx in range(gens_per_p):
                gen_dir = os.path.join(problem_dir, f"generation_{gen_idx}")
                os.makedirs(gen_dir, exist_ok=True)

                # Ensure lists have enough elements if generation was partial (shouldn't happen with num_return_sequences)
                if gen_idx >= len(all_responses):
                    print(
                        f"[Warning] Problem {actual_idx}, gen {gen_idx}: Expected generation not found in batch output. Skipping."
                    )
                    error_info = {
                        "error_path": os.path.join(gen_dir, "error.txt"),
                        "error": "Generation missing from batch output.",
                    }
                    with open(error_info["error_path"], "w") as f_err:
                        f_err.write(error_info["error"])
                    p_res_entry["generations"][gen_idx] = error_info
                    continue

                current_response = all_responses[gen_idx] # 第k个文本答案
                current_tokens_list = all_tokens_lists[gen_idx] # 第k个tokens
                current_hs_input = all_hs_inputs[gen_idx] # 第k个最后一层的提示词隐藏状态
                current_hs_output = all_hs_outputs[gen_idx] # 第k个最后一层的生成
                current_actual_generated_len = all_actual_lengths[gen_idx] # 第k个实际生成的token长度

                generation_files = {
                    "response_path": os.path.join(gen_dir, "response.txt"),
                    "metadata_path": os.path.join(gen_dir, "metadata.json"),
                    "tokens_path": os.path.join(gen_dir, "tokens.json"),
                    "hidden_states_input_path": os.path.join(
                        gen_dir, "token_hidden_states_input.npy"
                    ),
                    "hidden_states_output_path": os.path.join(
                        gen_dir, "token_hidden_states_output.npy"
                    ),
                }

                try:
                    with open(
                        generation_files["response_path"], "w", encoding="utf-8"
                    ) as f:
                        f.write(current_response)

                    with open(
                        generation_files["tokens_path"], "w", encoding="utf-8"
                    ) as f:
                        json.dump(current_tokens_list, f, indent=2)

                    np.save(
                        generation_files["hidden_states_input_path"], current_hs_input
                    )
                    np.save(
                        generation_files["hidden_states_output_path"], current_hs_output
                    )

                    metadata = {
                        "batch_generation_time_seconds_for_problem": batch_generation_time,
                        "estimated_time_per_generation_in_batch": (
                            batch_generation_time / gens_per_p
                            if gens_per_p > 0
                            else batch_generation_time
                        ),
                        "input_prompt_length_tokens": inputs["input_ids"].shape[1],
                        "output_generated_length_tokens": current_actual_generated_len,
                        "total_tokens_in_sequence_file": len(
                            current_tokens_list
                        ),  # Includes prompt, generated, EOS, padding
                        "prompt_plus_output_tokens": inputs["input_ids"].shape[1]
                        + current_actual_generated_len,
                    }
                    with open(generation_files["metadata_path"], "w") as f:
                        json.dump(metadata, f, indent=2)

                    print(
                        f"  Saved results for problem {actual_idx}, generation {gen_idx} to {gen_dir}"
                    )
                    # Add metadata dict to files dict
                    generation_files["metadata"] = metadata
                    p_res_entry["generations"][gen_idx] = generation_files

                except Exception as e_gen_save:
                    print(f"保存{actual_idx}结果出现错误, 生成 {gen_idx}: {e_gen_save}")
                    error_path = os.path.join(gen_dir, "error.txt")
                    with open(error_path, "w") as f_err:
                        f_err.write(f"Error during saving: {str(e_gen_save)}")
                    p_res_entry["generations"][gen_idx] = {
                        "error_path": error_path,
                        "error": f"Error during saving: {str(e_gen_save)}",
                    }

        except Exception as e_problem_batch:
            print(
                f"Error during batch generation or processing for problem {actual_idx}: {e_problem_batch}"
            )
            # Log error for all generations of this problem if batch failed
            for gen_idx_err in range(gens_per_p):
                gen_dir = os.path.join(problem_dir, f"generation_{gen_idx_err}")
                os.makedirs(gen_dir, exist_ok=True)
                error_path = os.path.join(gen_dir, "error.txt")
                error_msg = f"Batch generation/processing failed for problem: {str(e_problem_batch)}"
                with open(error_path, "w") as f_err:
                    f_err.write(error_msg)
                p_res_entry["generations"][gen_idx_err] = {
                    "error_path": error_path,
                    "error": error_msg,
                }

        results["problems"][actual_idx] = p_res_entry

        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        print(f"问题 {actual_idx} 处理完成")

    print("\nGeneration process complete.")
    final_summary_path = os.path.join(output_dir, "summary_results.json")
    with open(final_summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results summary saved to {final_summary_path}")

