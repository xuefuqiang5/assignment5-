import os
from vllm import LLM, SamplingParams

def main():
    print("🚀 准备加载模型 (由于是第一次运行，系统会自动从镜像站下载约 3GB 的权重，请耐心等待)...")
    
    # 初始化 vLLM 模型。
    # 这里的 "Qwen/Qwen2.5-Math-1.5B" 就是 HuggingFace 上的模型名称。
    # 代码执行到这里时，如果本地没有，就会自动触发下载。
    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B", 
        dtype="float16",
        # 如果你后续要在 5090 单卡上同时跑 PyTorch 训练，可以解除下面这行的注释限制显存
        # gpu_memory_utilization=0.35 
    )

    print("✅ 模型下载/加载成功！正在设置生成参数...")
    
    # 按照作业文档 PDF 中要求的参数进行设置
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"], # 生成到闭合标签就立刻停止
        include_stop_str_in_output=True
    )

    # 这里我们用一道简单的测试题，先帮你把下载和运行流程跑通
    test_question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
    # 使用作业要求的 r1_zero prompt 模板拼接
    prompt = f"""A conversation between User and Assistant. The User asks a question, and the Assistant
solves it. The Assistant first thinks about the reasoning process in the mind and
then provides the User with the answer. The reasoning process is enclosed within
<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,
i.e., <think> reasoning process here </think> <answer> answer here </answer>.

User: {test_question}
Assistant: <think>
"""

    prompts = [prompt]

    print("🧠 模型正在思考中...")
    # vLLM 开始批量生成回答
    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "="*60)
    print("🎯 模型输出结果：")
    for output in outputs:
        generated_text = output.outputs[0].text
        print(generated_text)
    print("="*60)
    print("🎉 测试完毕！如果你看到了上面的推理过程，说明模型已经完全部署成功了！")

if __name__ == "__main__":
    main()