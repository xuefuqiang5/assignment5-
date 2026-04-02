import json
import os
from datasets import load_dataset

def prepare_math_dataset():
    # 1. 创建存放数据的目录
    data_dir = "../data/MATH"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    save_path = os.path.join(data_dir, "validation.jsonl")
    
    print("🚀 正在从 Hugging Face 下载 MATH 数据集 (hendrycks/competition_math)...")
    
    # 2. 加载数据集
    # 我们通常使用 'test' 划分作为评测的验证集，因为它包含了 5000 道标准数学竞赛题
    try:
        dataset = load_dataset("hendrycks/competition_math", trust_remote_code=True)
        test_data = dataset['test']
    except Exception as e:
        print(f"❌ 下载失败，请检查网络或是否设置了 HF 镜像源。错误信息: {e}")
        return

    print(f"✅ 下载成功！共计 {len(test_data)} 条题目。")
    print(f"📦 正在转换为 JSONL 格式并保存至 {save_path}...")

    # 3. 写入 jsonl 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        for entry in test_data:
            # 结构转换：确保包含题目 (problem) 和标准答案 (solution)
            # 这里的字段名要和你的评测脚本对齐
            json_record = {
                "question": entry['problem'],
                "answer": entry['solution'],
                "subject": entry['subject'],
                "level": entry['level']
            }
            f.write(json.dumps(json_record, ensure_ascii=False) + '\n')

    print(f"✨ 数据准备完成！你可以开始第一个实验了。")

if __name__ == "__main__":
    prepare_math_dataset()