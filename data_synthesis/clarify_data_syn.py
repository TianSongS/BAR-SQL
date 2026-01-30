import json
import pandas as pd
import json
import re
from tqdm import tqdm
import os
from typing import Optional, Dict
import concurrent.futures

from prompt_txt import CLARIFICATION_CATEGORIES,CLARIFY_QUESTION_REWRITE_PROMPT,CLARIFY_COT_SYN_PROMPT
from utils import extract_schema_from_prompt,extract_question_from_prompt,replace_user_query_with_new_question,chat_gpt4o,get_llm_res,save_checkpoint,read_data


def generate_clarification_scenario(original_question: str, schema_context: str, categories_str: str) -> dict:
    """
    调用LLM，让其自行选择类别并生成一个包含模糊问题和对应澄清话术的场景。
    """

    # 使用更新后的Prompt模板
    prompt = CLARIFY_QUESTION_REWRITE_PROMPT.format(
        schema_info=schema_context,
        clarification_categories=categories_str,
        original_question=original_question
    )
    
    try:
        response = get_llm_res(prompt)
        response_content = response.choices[0].message.content
        
        try:
            # 直接解析JSON字符串
            parsed_result = json.loads(response_content)
            
            # 验证返回结果是否包含我们需要的2个键
            required_keys = ["rewrite_logic", "ambiguous_question","difficulty_analysis"]
            if all(key in parsed_result for key in required_keys):
                return parsed_result
            else:
                return {"error": "LLM response JSON missing required keys.", "raw_response": response_content}

        except (json.JSONDecodeError, IndexError) as e:
            return {"error": "Failed to parse JSON response.", "details": str(e), "raw_response": response_content}
            
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


def generate_cot_synthesis_prompt(base_prompt: str,ambiguous_question: str, ambiguous_reason: str) -> str:
    """
    将基础的NL2SQL Prompt与CoT合成指令结合，生成最终的Prompt。

    Args:
        base_prompt (str): 您的原始NL2SQL基础Prompt，包含所有背景知识。
        ambiguous_question (str): 经过模糊化处理的用户问题。
        ambiguous_reason (str): 对问题模糊点的解释说明。

    Returns:
        str: 组合后，可直接用于调用LLM的完整Prompt。
    """
    
    # 确保不修改原始的few-shot示例，因为它是为生成SQL设计的
    # 我们只保留上下文，移除掉生成SQL的指令和示例
    prompt_parts = base_prompt.split("--- 6.few shot")
    context_prompt = prompt_parts[0] # 只取第一部分作为上下文

    # 组装新的Prompt
    final_prompt = CLARIFY_COT_SYN_PROMPT.format(
        nl2sql_base_prompt=context_prompt.strip(),
        ambiguous_question=ambiguous_question,
        ambiguous_reason=ambiguous_reason
    )
    
    return final_prompt

def parse_llm_clarification_output(llm_output: str) -> Optional[Dict[str, str]]:
    """
    解析遵循特定格式的LLM输出，提取思考过程和澄清问题。

    Args:
        llm_output (str): 从LLM获取的原始文本输出。
                          预期格式为：
                          [思考过程]
                          ...
                          [澄清问题]
                          ...

    Returns:
        Optional[Dict[str, str]]: 如果解析成功，返回一个包含 'thought_process' 和 
                                     'clarification_question' 的字典。
                                     如果输入格式不正确或缺少任一标签，则返回 None。
    """
    try:
        # 使用正则表达式来匹配和提取两个部分的内容
        # re.DOTALL (或 re.S) 标志让 '.' 可以匹配包括换行符在内的任意字符
        pattern = r"\[思考过程\](.*?)\[澄清问题\](.*)"
        match = re.search(pattern, llm_output, re.DOTALL)

        if not match:
            # 如果正则表达式没有匹配到，说明格式不符合预期
            print("解析失败：在LLM输出中未找到 '[思考过程]' 和 '[澄清问题]' 标签。")
            return None

        # group(1) 对应第一个括号内的内容 (思考过程)
        # group(2) 对应第二个括号内的内容 (澄清问题)
        thought_process = match.group(1).strip()
        clarification_question = match.group(2).strip()
        
        # 进一步校验，确保提取的内容不是空的
        if not thought_process or not clarification_question:
            print("解析警告：提取到的思考过程或澄清问题为空。")
            # 根据您的需求，这里也可以返回None或保留空字符串
        
        return {
            "thought_process": thought_process,
            "clarification_question": clarification_question
        }
    except Exception as e:
        print(f"解析时发生未知错误: {e}")
        return None


def get_new_out_cot(base_prompt,ambiguous_question,ambiguous_reason):
    promot = generate_cot_synthesis_prompt(base_prompt,ambiguous_question,ambiguous_reason)
    res = chat_gpt4o(promot)
    res_dict = parse_llm_clarification_output(res)
    return res_dict['thought_process'],res_dict['clarification_question']


def process_row(row_data):
    """
    处理单行数据的函数，封装了原始for循环中的核心逻辑。
    """
    USER_INITIAL_PROMPT, response, index_name = row_data
    try:
        schema_info = extract_schema_from_prompt(USER_INITIAL_PROMPT)
        sample_original_question = extract_question_from_prompt(USER_INITIAL_PROMPT)

        if schema_info and sample_original_question:
            scenario_result = generate_clarification_scenario(
                original_question=sample_original_question,
                schema_context=schema_info,
                categories_str=CLARIFICATION_CATEGORIES
            )
            if "error" not in scenario_result:
                USER_NEW_PROMPT = replace_user_query_with_new_question(USER_INITIAL_PROMPT, scenario_result['ambiguous_question'].replace('\\', '\\\\'))
                cot_process, clarification_question = get_new_out_cot(USER_NEW_PROMPT, scenario_result['ambiguous_question'], scenario_result['rewrite_logic'])

                return {
                    'original_prompt': USER_INITIAL_PROMPT,
                    'rewrite_logic': scenario_result['rewrite_logic'],
                    'original_question':sample_original_question,
                    'ambiguous_question': scenario_result['ambiguous_question'],
                    'difficulty_analysis':scenario_result['difficulty_analysis'],
                    'cot_process': cot_process,
                    'clarification_question': clarification_question
                }
    except Exception as e:
        # 在并发环境中，打印错误比使用pass更好，以便于调试
        print(f"处理时发生错误: {e} for prompt: {USER_INITIAL_PROMPT[:50]}...")
    return None



def main():
    # --- 配置参数 ---
    WORKERS = 4
    BATCH_SIZE = 12
    CHECKPOINT_FILE = 'clarification_data_checkpoint.csv'

    data_to_process = read_data()

    # --- 断点续传逻辑 ---
    processed_prompts = set()
    if os.path.exists(CHECKPOINT_FILE):
        try:
            processed_df = pd.read_csv(CHECKPOINT_FILE)
            processed_prompts = set(processed_df['original_prompt'])
            print(f"从检查点文件加载了 {len(processed_prompts)} 条已处理的数据。")
        except Exception as e:
            print(f"读取检查点文件失败: {e}。将从头开始。")

    # 过滤掉已处理的数据
    if processed_prompts:
        initial_count = len(data_to_process)
        data_to_process = data_to_process[~data_to_process['query'].isin(processed_prompts)]
        print(f"已过滤 {initial_count - len(data_to_process)} 条已处理的数据。剩余 {len(data_to_process)} 条待处理。")

    if data_to_process.empty:
        print("所有数据都已处理完毕。")
        return

    # --- 并发处理 ---
    batch_results = []
    
    # 将DataFrame转换为元组列表，以便传递给处理函数
    tasks = [tuple(row) for row in data_to_process[['query', 'response', 'index_name']].itertuples(index=False)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        # 使用tqdm来显示进度条
        future_to_prompt = {executor.submit(process_row, task): task[0] for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_prompt), total=len(tasks), desc="Processing data"):
            result = future.result()
            if result:
                batch_results.append(result)
            
            # 当累积到一定数量时，保存一次
            if len(batch_results) >= BATCH_SIZE:
                save_checkpoint(batch_results, CHECKPOINT_FILE)
                batch_results = [] # 清空批次列表

    if batch_results:
        save_checkpoint(batch_results, CHECKPOINT_FILE)

    print("\n所有任务处理完成。")
    if os.path.exists(CHECKPOINT_FILE):
        final_df = pd.read_csv(CHECKPOINT_FILE)
        print(f"总共成功生成并保存了 {len(final_df)} 条数据到 {CHECKPOINT_FILE}")
    else:
        print("\n没有生成任何有效数据，未创建文件。")


if __name__ == "__main__":
    main()