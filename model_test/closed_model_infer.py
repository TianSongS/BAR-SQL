import pandas as pd
import re
import asyncio
from tqdm.asyncio import tqdm_asyncio
from utils import call_deepseekv32

# 假设这是你引入的外部函数，或者在这里定义它
# from your_module import chat_gpt4o 

# ========== 配置 ==========
input_path = "test.csv"
output_path = "output.csv"

MODEL = "gpt-4o" 
CONCURRENCY = 3  

# ========== 核心函数 ==========

async def call_api(prompt):
    """适配新的 chat_gpt4o 调用方式"""
    try:
        content = await asyncio.to_thread(
            call_deepseekv32, 
            prompt=prompt+'(注意，有些问题可能不应输出SQL，另外无论是输出新SQL还是更改SQL或任何其他结论，将任何需要输出的SQL完整内容或任何结论放在```res ... ```中，其他思考内容放在此之前)', 
            temperature=0.6, 
        )
        
        if not content:
            return "", ""

        # 提取 <think> 与 <answer> 段
        if 'base' in MODEL.lower():
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)

            think_text = think_match.group(1).strip() if think_match else ""
            answer_text = answer_match.group(1).strip() if answer_match else ""
            
            if not answer_text: 
                    answer_text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            return think_text, answer_text
        else:
            # 1. 优先尝试标准标签解析 (<think>, <answer>)
            think_match = re.search(r"(?s)(?<=<think>)(.*?)(?=</think>)", content)
            answer_match = re.search(r"(?s)(?<=<answer>)(.*?)(?=</answer>)", content)

            if think_match or answer_match:
                think_text = think_match.group(0).strip() if think_match else ""
                answer_text = answer_match.group(0).strip() if answer_match else ""
            else:
                # 2. 如果没有标签，适配 GPT-4o 等输出的 Markdown 代码块格式
                # 逻辑：提取 ```sql ... ``` 内部为 answer，代码块前面的文本为 think
                code_match = re.search(r"```(?:res)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
                
                if code_match:
                    # 提取代码块内容作为 answer
                    answer_text = code_match.group(1).strip()
                    # 提取代码块起始位置之前的所有文本作为 think
                    think_text = content[:code_match.start()].strip()
                else:
                    # 3. 既无标签也无代码块，完全兜底
                    think_text = ""
                    answer_text = content

            return think_text, answer_text

    except Exception as e:
        print(f"⚠️ 调用异常：{e}")
        return "", ""


async def process_all(prompts):
    """并发执行所有调用，保持原始顺序"""
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded_call(prompt):
        async with sem:
            return await call_api(prompt)

    tasks = [bounded_call(p) for p in prompts]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing")
    return results


# ========== 主程序入口 ==========

async def main():
    # 1. 读取CSV
    df = pd.read_csv(input_path)
    
    if "query" not in df.columns:
        raise ValueError("CSV中没有 'query' 列，请检查文件。")

    prompts = df["query"].astype(str).tolist()

    # 2. 异步调用
    results = await process_all(prompts)

    # 3. 保存结果
    think_list, answer_list = zip(*results)
    df["think"] = think_list
    df["answer"] = answer_list

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 处理完成！结果已保存至：{output_path}")


if __name__ == "__main__":
    asyncio.run(main())