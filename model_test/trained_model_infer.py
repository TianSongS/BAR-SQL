# CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server --served-model-name sft --model /lpai/nl2sql/DC_tss/data/models/sft_1024/models/best  --tensor-parallel-size 1 --port 6001 --max_model_len 25000
import pandas as pd
import re
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio

input_path = 'input.csv'
output_path = "output.csv"

BASE_URL = "http://0.0.0.0:6001/v1/chat/completions"
API_KEY = "EMPTY"
MODEL = 'sft'
CONCURRENCY = 4   # 并发请求数，可根据本地服务器性能调整



async def call_api(session, prompt):
    """调用本地 OpenAI API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    try:
        async with session.post(BASE_URL, json=payload, headers=headers, timeout=120) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"❌ API 调用失败 [{resp.status}]：{text[:200]}")
                return "", ""
            result = await resp.json()
            content = result["choices"][0]["message"]["content"].strip()

            # 提取 <think> 与 <answer> 段
            if 'base' in MODEL.lower():
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)

                think_text = think_match.group(1).strip() if think_match else ""
                answer_text = answer_match.group(1).strip() if answer_match else ""
                answer_text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                return think_text, answer_text
            else:
                think_match = re.search(r"(?s)(?<=<think>)(.*?)(?=</think>)", content)
                answer_match = re.search(r"(?s)(?<=<answer>)(.*?)(?=</answer>)", content)

                think_text = think_match.group(0).strip() if think_match else ""
                answer_text = answer_match.group(0).strip() if answer_match else ""

                return think_text, answer_text


    except Exception as e:
        print(f"⚠️ 调用异常：{e}")
        return "", ""


async def process_all(prompts):
    """并发执行所有 API 调用，保持原始顺序"""
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def bounded_call(prompt):
            async with sem:
                return await call_api(session, prompt)

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


# ========== 运行 ==========
if __name__ == "__main__":
    asyncio.run(main())