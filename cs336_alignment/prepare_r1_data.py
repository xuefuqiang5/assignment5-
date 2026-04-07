import json
import random

# 配置路径 (请根据你实际存放文件的路径进行修改)
INPUT_TRAIN_FILE = "../data/gsm8k/train.jsonl"   # 原始的 GSM8K 训练集
INPUT_TEST_FILE = "../data/gsm8k/test.jsonl"     # 原始的 GSM8K 测试集
OUTPUT_TRAIN_FILE = "../data/gsm8k/gsm8k_rl_train.jsonl" # 处理后用于 RL 训练的集合
OUTPUT_VALID_FILE = "../data/gsm8k/gsm8k_rl_valid.jsonl" # 处理后用于 RL 验证的集合
OUTPUT_TEST_FILE = "../data/gsm8k/gsm8k_rl_test.jsonl"   # 处理后用于最终评估的集合

# 验证集比例 (例如 0.05 代表抽取 5% 作为验证集)
VALID_RATIO = 0.05

# 强迫模型产生思维链的 Prompt 模板
PROMPT_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant
solves it. The Assistant first thinks about the reasoning process in the mind and
then provides the User with the answer. The reasoning process is enclosed within
<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,
i.e., <think> reasoning process here </think> <answer> answer here </answer>.

User: {question}
Assistant: <think>\n"""

def process_gsm8k_line(line_data):
    """处理单行数据，提取 prompt 和 ground_truth"""
    data = json.loads(line_data)
    
    raw_question = data.get("question", "")
    raw_answer = data.get("answer", "")
    
    # 1. 组装 Prompt
    formatted_prompt = PROMPT_TEMPLATE.format(question=raw_question)
    
    # 2. 提取干净的标准答案 (Ground Truth)
    # GSM8K 的答案规范是最终答案在 "#### " 之后
    if "####" in raw_answer:
        clean_gt = raw_answer.split("####")[-1].strip()
    else:
        # 兜底机制：如果遇到个别脏数据没有 ####，直接跳过或者设为空
        clean_gt = ""
        
    return {
        "prompt": formatted_prompt,
        "ground_truth": clean_gt
    }

def main():
    print("🚀 开始处理 GSM8K 数据集...")
    
    # ==========================================
    # 第一部分：处理 Train 文件并切分为 Train 和 Valid
    # ==========================================
    processed_train_dataset = []
    
    with open(INPUT_TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            processed_item = process_gsm8k_line(line)
            if processed_item["ground_truth"]:
                processed_train_dataset.append(processed_item)

    # 打乱训练数据集，确保拆分时的随机性
    random.seed(42) 
    random.shuffle(processed_train_dataset)
    
    # 计算拆分点
    total_train_samples = len(processed_train_dataset)
    valid_size = int(total_train_samples * VALID_RATIO)
    train_size = total_train_samples - valid_size
    
    valid_dataset = processed_train_dataset[:valid_size]
    train_dataset = processed_train_dataset[valid_size:]

    # ==========================================
    # 第二部分：独立处理 Test 文件
    # ==========================================
    test_dataset = []
    with open(INPUT_TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            processed_item = process_gsm8k_line(line)
            if processed_item["ground_truth"]:
                test_dataset.append(processed_item)
                
    test_size = len(test_dataset)

    # ==========================================
    # 第三部分：将所有处理好的数据写入新的 JSONL 文件
    # ==========================================
    def write_jsonl(data_list, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    write_jsonl(train_dataset, OUTPUT_TRAIN_FILE)
    write_jsonl(valid_dataset, OUTPUT_VALID_FILE)
    write_jsonl(test_dataset, OUTPUT_TEST_FILE) # 新增：写入 test 数据集
    
    print(f"✅ 处理完成！")
    print(f"总计有效训练/验证数据: {total_train_samples} 条")
    print(f"-> 训练集 ({OUTPUT_TRAIN_FILE}): {train_size} 条")
    print(f"-> 验证集 ({OUTPUT_VALID_FILE}): {valid_size} 条")
    print(f"-> 测试集 ({OUTPUT_TEST_FILE}): {test_size} 条")

if __name__ == "__main__":
    main()