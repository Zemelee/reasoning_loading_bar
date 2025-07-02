import os
from types import SimpleNamespace
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from utils import load_problems

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# 对模型隐藏状态进行一次定向干预
def generate_with_intervention(
    model,
    tokenizer,
    prompt,
    max_new_tokens=1,
    intervention_vector=None,
    intervention_scale=1.0,
):
    """通过在生成过程中注入干预来生成 token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    hook_handles = []

    # 在模型中拦截某一层的输出，并注入干预向量
    def forward_hook(module, input_args, output):
        original_output = output
        hidden_states_to_modify = None
        if isinstance(output, tuple):
            hidden_states_to_modify = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states_to_modify = output
        else:
            logging.warning(f"钩子捕获层输出中未包含预期的隐藏状态张量: {type(output)}")
            return output
        if hidden_states_to_modify is None or not isinstance(
            hidden_states_to_modify, torch.Tensor
        ):
            logging.warning("钩子层输出不包含预期的隐藏状态张量.")
            return
        # clone避免修改原始状态
        modified_states = hidden_states_to_modify.clone()
        # Reshape intervention vector to [1, 1, hidden_dim] to broadcast correctly
        reshaped_vector = intervention_vector.view(1, 1, -1).to(
            hidden_states_to_modify.device
        )
        # 干预最后一个 token 的隐藏状态
        modified_states[:, -1:, :] = modified_states[:, -1:, :] + (
            reshaped_vector * intervention_scale
        )
        # 返回干预后的输出
        if isinstance(original_output, tuple):
            return (modified_states,) + original_output[1:]
        else:
            return modified_states

    # 在模型的 model.model.norm 这一层（final norm）注册钩子
    # .norm 是 output logits 之前的关键位置
    target_intervention_module = model.model.norm
    if not hasattr(target_intervention_module, "register_forward_hook"):
        raise ValueError(f"model.model.norm 没有 'register_forward_hook' 方法.")
    hook_handles.append(target_intervention_module.register_forward_hook(forward_hook))
    logging.debug(f"干预钩子已注册，类型：{type(target_intervention_module)}")
    logging.debug(f"干预向量形状：{intervention_vector.shape}")
    logging.debug(f"干预强度系数: {intervention_scale}")
    try:
        # Generate with the hook in place
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for reproducibility
                num_beams=1,
                use_cache=True,  # Use KV cache for efficiency
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        # 清除钩子，确保不影响之后的推理
        for handle in hook_handles:
            handle.remove()
        if hook_handles:
            logging.debug("清除干预hooks.")
    return result


args = SimpleNamespace(
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    tokenizer_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    vector_path="tpv_model/tpv_linear_weights.npy",
    output_gens_dir="llama_intervention_responses",
    dataset="math500",
    task_name="math500",
    problem_start_idx=30,
    problem_end_idx=31,
    max_new_tokens=2048,
    alpha=100.0,
    device="auto",
    torch_dtype="float16",
)
if args.tokenizer_name_or_path is None:
    args.tokenizer_name_or_path = args.model_name_or_path
logging.info(f"Loading tokenizer from: {args.tokenizer_name_or_path}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logging.info("Set pad_token to eos_token.")
logging.info(f"Loading model from: {args.model_name_or_path}")
model_dtype = getattr(torch, args.torch_dtype)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    device_map=args.device if args.device != "auto" else "auto",
    torch_dtype=model_dtype,
    low_cpu_mem_usage=True,
)
device = model.device
logging.info(f"模型已加载，{device}")
logging.info("正在准备干预向量...")
hidden_dim = model.config.hidden_size
logging.info(f"模型隐藏层维度: {hidden_dim}")
# 加载干预向量，线性回归权重
intervention_vector = torch.Tensor(np.load(args.vector_path)).to(device)
# Reshape为1D向量
intervention_vector = intervention_vector.squeeze()
if intervention_vector.ndim != 1:
    raise ValueError(f"干预向量压缩后须是1D,但实际为 {intervention_vector.shape}")
if intervention_vector.shape[0] != hidden_dim:
    raise ValueError(
        f"干预向量维度{intervention_vector.shape[0]} != {hidden_dim}模型隐藏层维度"
    )
logging.info(f"干预向量已重塑为形状：{intervention_vector.shape}")
intervention_scale = args.alpha
logging.info(f"使用的干预缩放系数 (alpha):{intervention_scale}")
# Construct the full path for output generations
model_folder_name = args.model_name_or_path.split("/")[-1]
# llama_intervention_responses/DeepSeek-R1-Distill-Qwen-1.5B/math500/intervention
generations_base_dir = os.path.join(
    args.output_gens_dir, model_folder_name, args.task_name, "intervention"
)  
# 遍历题目集
problems = load_problems(args.dataset, args.problem_start_idx, args.problem_end_idx)
for i, problem in enumerate(problems):
    problem_num = i
    logging.info(f"正在处理问题 {problem_num}...")
    problem_output_dir = os.path.join(generations_base_dir, f"problem_{problem_num}")
    os.makedirs(problem_output_dir, exist_ok=True)
    prompt = (
        problem
        + "\nPlease reason step by step, and put your final answer within \\boxed{}.\n<think>\n"
    )
    # Generate with single intervention scale
    output_with_intervention = generate_with_intervention(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        intervention_vector=intervention_vector,
        intervention_scale=intervention_scale,
    )
    output_file_path = os.path.join(
        problem_output_dir, f"response_alpha_{args.alpha}.txt"
    )
    try:
        with open(output_file_path, "w") as f:
            f.write(output_with_intervention)
        logging.info(f"{problem_num}-{args.alpha},保存至 {output_file_path}")
    except IOError as e:
        logging.error(f"写入输出文件 {output_file_path} 失败：{e}")

logging.info("处理结束")
