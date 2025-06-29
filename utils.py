from datasets import load_dataset

# 从指定数据集加载一定数量的问题，可指定起始索引
def load_problems(dataset, starting_index=None, end_index=None):
    if dataset == "math500":
        if starting_index is None:
            starting_index = 30
        if end_index is None:
            end_index = 500
        problems = load_dataset("HuggingFaceH4/MATH-500")["test"]["problem"][starting_index:end_index]
    
    elif dataset == "gsm8k":
        if starting_index is None:
            starting_index = 30
        if end_index is None:
            end_index = 330
        problems = load_dataset("openai/gsm8k", "main")["train"]["question"][starting_index:end_index]

    else:
        raise ValueError("不支持的数据集。请使用'math500'或'gsm8k'。")
    
    return problems