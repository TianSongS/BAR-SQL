import json
import pandas as pd
import re
from tqdm import tqdm
import os
from typing import Dict, Optional
import concurrent.futures
from functools import partial
from utils import (
    extract_schema_from_prompt,
    extract_data_kg_from_prompt,
    extract_question_from_prompt,
    extract_indicator_knowledge_from_prompt,
    extract_domain_rules_from_prompt,
    chat_gpt4o,
    save_checkpoint,
)

from prompt_txt import CONSTRAINT_VALIDATION_PROMPT,CONSTRAINT_COT_SYNTHESIS_PROMPT,CONSTRAINT_EXTRACTION_PROMPT


# --- 核心功能函数 ---

def generate_new_constraint(
    original_question: str,
    schema_info: str,
    indicator_knowledge: str,
    domain_rules: str,
    data_kg: str
) -> Optional[Dict]:
    """步骤1: 调用LLM，提取一个问题中未满足的业务约束。"""
    prompt = CONSTRAINT_EXTRACTION_PROMPT.format(
        schema_kg=schema_info,
        knowledge_graph=indicator_knowledge,
        domain_kg=domain_rules,
        question=original_question,
        data_kg=data_kg)
    try:
        response_content = chat_gpt4o(prompt)
        clean_json_str = response_content.strip().replace('```json', '').replace('```', '')
        data = json.loads(clean_json_str)
        if "think" in data and "constraint_description" in data:
            return data
        return None
    except Exception as e:
        print(f"约束提取时发生错误: {e}")
        return None
    
def validate_constraint_conflict(
    original_question: str,
    new_constraint: str
) -> bool:
    """步骤2: 调用LLM，验证新约束是否与原问题冲突。(无变化)"""
    prompt = CONSTRAINT_VALIDATION_PROMPT.format(
        question=original_question,
        new_constraint=new_constraint
    )
    try:
        response_content = chat_gpt4o(prompt)
        clean_json_str = response_content.strip().replace('```json', '').replace('```', '')
        data = json.loads(clean_json_str)
        return data.get("constraint_satisfied", True)
    except Exception as e:
        print(f"冲突校验时发生错误: {e}")
        return True

def add_constraint_to_prompt(original_prompt: str, new_constraint_text: str) -> Optional[str]:
    """
    将新的约束条件智能地插入到原始Prompt的'指标计算逻辑'部分，
    作为一个独立的'必备约束'行。
    """
    # 构造要插入的行，可以增加缩进以保持格式美观
    constraint_line = f"必备约束：{new_constraint_text}"
    
    lines = original_prompt.split('\n')
    
    # 寻找 "计算逻辑：" 这一行作为插入点的锚点
    insert_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("计算逻辑："):
            insert_index = i
            break
            
    if insert_index != -1:
        # 在 "计算逻辑：" 这一行的正上方插入新的约束行
        lines.insert(insert_index, constraint_line)
        return '\n'.join(lines)
    else:
        # 如果找不到锚点，则返回None表示失败
        print(f"错误：在Prompt中未找到锚点 '计算逻辑：'")
        return None

# 种子文件
df_sql = pd.read_csv('SQL_data.csv')

def extract_think_content(response: str) -> str:
    """
    从response中提取<think>和</think>之间的内容
    """
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def generate_cot_for_new_constraint(
    modified_user_prompt: str,
    df_sql
) -> Optional[Dict]:
    """
    步骤3: 调用LLM，基于修改后的完整Prompt生成思考过程(CoT)和结论。
    函数签名已简化，现在接收完整的 modified_user_prompt。
    随机从df中选择一行数据作为few-shot示例。
    """
    # 随机从df中选择一行
    random_row = df_sql.sample(n=1).iloc[0]
    example_query = random_row['original_question']
    example_think = extract_think_content(random_row['response'])
    
    # 使用选中的示例数据格式化prompt
    prompt = CONSTRAINT_COT_SYNTHESIS_PROMPT.format(
        modified_user_prompt=modified_user_prompt,
        example_query=example_query,
        example_think=example_think
    )
    
    try:
        response_content = chat_gpt4o(prompt)
        if response_content:
            return response_content
        print(f"CoT合成解析失败: {response_content}")
        return None
    except Exception as e:
        print(f"CoT合成时发生未知错误: {e}")
        return None

def process_row_for_constraint_based_ask(row_data,df_sql):
    """
    处理单行数据，执行新思路下的追问数据合成完整流程。
    """
    USER_INITIAL_PROMPT, response = row_data
    try:
        # 步骤1: 从原始prompt中提取所有必要信息 (为生成约束所需)
        original_question = extract_question_from_prompt(USER_INITIAL_PROMPT)
        schema_info = extract_schema_from_prompt(USER_INITIAL_PROMPT)
        indicator = extract_indicator_knowledge_from_prompt(USER_INITIAL_PROMPT)
        domain_rules = extract_domain_rules_from_prompt(USER_INITIAL_PROMPT)
        data_kg = extract_data_kg_from_prompt(USER_INITIAL_PROMPT)

        if not all([original_question, schema_info, indicator, domain_rules]):
            print("信息提取不完整，跳过此行。")
            return None

        # 步骤2: 生成新的业务约束
        constraint_data = generate_new_constraint(original_question, schema_info, indicator, domain_rules,data_kg)
        if not constraint_data:
            return None
        
        new_constraint = constraint_data["constraint_description"]
        # ask_for_user = constraint_data["ask_for_user"]
        if '无法提取' in new_constraint:
            return  {'status': '无法提取',
                      'reason': constraint_data["think"],
                      'original_prompt': USER_INITIAL_PROMPT,
                    'original_question': original_question,
                      }
        # 步骤3: 验证约束与问题是否冲突
        constraint_satisfied = validate_constraint_conflict(original_question, new_constraint)
        if constraint_satisfied:
            return {'status': 'skip', 
                    'reason': '约束冲突',
                      'original_prompt': USER_INITIAL_PROMPT,
                    'original_question': original_question,}
        
        # 步骤4: 将新约束添加到原始Prompt中，生成修改后的版本
        modified_user_prompt = add_constraint_to_prompt(USER_INITIAL_PROMPT, new_constraint)
        if not modified_user_prompt:
            # 如果添加失败（例如找不到标记），则跳过此行
            return None

        # 步骤5: 使用修改后的完整Prompt生成追问CoT
        cot_result = generate_cot_for_new_constraint(
            modified_user_prompt,df_sql
        )

        if cot_result:
            return {
                'status': 'success', 
                'reason': constraint_data["think"],
                'original_prompt': USER_INITIAL_PROMPT,
                'original_question': original_question,
                'added_constraint': new_constraint,
                'modified_prompt': modified_user_prompt, # 保存修改后的Prompt，便于检查
                'cot_process': cot_result,
            }
        
        return None

    except Exception as e:
        print(f"处理行时发生严重错误: {e} for prompt: {USER_INITIAL_PROMPT[:50]}...")
        return None

# --- 主函数 (与上一版相同) ---
def main_constraint_based_synthesis():
    """
    用于合成基于“新增约束”的“追问”类别数据的主函数。
    """
    WORKERS = 2
    BATCH_SIZE = 30
    CHECKPOINT_FILE = 'constraint.csv'
    
    data_to_process = pd.read_csv('sql数据.csv')
    print(f"准备开始处理 {len(data_to_process)} 条数据，用于生成基于约束的追问样本。")
    
    processed_prompts = set()
    if os.path.exists(CHECKPOINT_FILE):
        try:
            processed_df = pd.read_csv(CHECKPOINT_FILE)
            if 'original_prompt' in processed_df.columns:
                processed_prompts = set(processed_df['original_prompt'])
                print(f"从检查点文件加载了 {len(processed_prompts)} 条已处理的数据。")
        except Exception as e:
            print(f"读取检查点文件失败: {e}。将从头开始。")

    if processed_prompts:
        initial_count = len(data_to_process)
        data_to_process = data_to_process[~data_to_process['query'].isin(processed_prompts)]
        print(f"已过滤 {initial_count - len(data_to_process)} 条已处理的数据。剩余 {len(data_to_process)} 条待处理。")

    if data_to_process.empty:
        print("所有数据都已处理完毕。")
        return

    batch_results = []
    tasks = [tuple(row) for row in data_to_process[['query', 'response']].itertuples(index=False)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        process_func = partial(process_row_for_constraint_based_ask, df_sql=df_sql)
        future_to_prompt = {executor.submit(process_func, task): task[0] for task in tasks}

        for future in tqdm(concurrent.futures.as_completed(future_to_prompt), total=len(tasks), desc="合成基于约束的追问数据"):
            result = future.result()
            if result:
                batch_results.append(result)
            
            if len(batch_results) >= BATCH_SIZE:
                save_checkpoint(batch_results, CHECKPOINT_FILE)
                batch_results = []

    if batch_results:
        save_checkpoint(batch_results, CHECKPOINT_FILE)

    print("\n所有追问数据合成任务处理完成。")
    if os.path.exists(CHECKPOINT_FILE):
        final_df = pd.read_csv(CHECKPOINT_FILE)
        print(f"总共成功生成并保存了 {len(final_df)} 条追问数据到 {CHECKPOINT_FILE}")


if __name__ == "__main__":
    print("开始合成【基于新增约束的追问数据】...")
    main_constraint_based_synthesis()