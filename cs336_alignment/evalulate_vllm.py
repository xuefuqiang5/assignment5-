import json
import statistics
from typing import Callable, List
from vllm import LLM, SamplingParams

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str], # [新增参数] 必须传入标准答案才能计算 reward
    eval_sampling_params: SamplingParams,
    output_file: str = "evaluation_results.json"
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    print(f"🚀 开始评估 {len(prompts)} 条数据...")
    
    # 1. 使用 vLLM 进行批量高效推理
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    format_rewards = []
    answer_rewards = []
    total_rewards = []
    
    # 2. 遍历每一个输出并使用 reward_fn 进行打分
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]
        prompt = prompts[i]
        
        # 调用我们之前定义的奖励函数
        # 注意：这里假设 r1_zero_reward_fn 已经在上下文中定义好了
        reward_dict = reward_fn(generated_text, ground_truth)
        
        # 收集分数用于计算平均值
        format_rewards.append(reward_dict.get("format_reward", 0.0))
        answer_rewards.append(reward_dict.get("answer_reward", 0.0))
        total_rewards.append(reward_dict.get("reward", 0.0))
        
        # 保存详细结果以便后续 Debug
        results.append({
            "id": i,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "rewards": reward_dict
        })

    # 3. 计算评估指标 (Metrics)
    metrics = {
        "total_samples": len(prompts),
        "avg_format_reward": statistics.mean(format_rewards),
        "avg_answer_reward": statistics.mean(answer_rewards),
        "avg_total_reward": statistics.mean(total_rewards),
    }
    
    print("\n" + "="*40)
    print("📊 评估指标汇总 (Evaluation Metrics):")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"- {key}: {value}")
    print("="*40)

    # 4. 将结果序列化到磁盘 (Serialize results to disk)
    final_output = {
        "metrics": metrics,
        "detailed_results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 详细评估结果已保存至: {output_file}")

# ==========================================
# 🚀 实际使用示例
# ==========================================
if __name__ == "__main__":
    # 假设你已经定义了 r1_zero_reward_fn
    from drgrpo_grader import r1_zero_reward_fn 
    
    # 1. 加载我们在上一步清洗好的测试集 (Test Dataset)
    TEST_FILE = "../data/gsm8k/gsm8k_rl_test.jsonl"
    prompts = []
    ground_truths = []
    
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
            ground_truths.append(data["ground_truth"])
            
    # 为了快速测试，我们可以先只切片前 100 条数据
    # prompts = prompts[:100]
    # ground_truths = ground_truths[:100]
            
    # 2. 初始化模型和参数
    print("🧠 正在加载 vLLM 模型...")
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B", dtype="float16")
    
    sampling_params = SamplingParams(
        temperature=0.0,  # 评估阶段建议设为 0.0 (Greedy Decoding) 保证确定性
        max_tokens=8192,  # 给足长思维链空间
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )
    
    # 3. 运行评估
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_file="gsm8k_base_model_eval.json"
    )