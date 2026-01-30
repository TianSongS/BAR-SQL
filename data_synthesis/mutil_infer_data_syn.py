# 导入所有必要的库
import functools
print = functools.partial(print, flush=True)
import json
import pandas as pd
import re
from tqdm import tqdm
import os
from typing import Dict, List, Optional
import concurrent.futures
import time
import sys
import random
import threading
from utils import (
    extract_schema_from_prompt,
    extract_indicator_knowledge_from_prompt,
    extract_domain_rules_from_prompt,
    call_gpt5,
    call_claude4_5,
    call_claude,
    chat_gpt4o,
    save_checkpoint,
    read_data,
    execute_sql_with_validation,
    extract_data_kg_from_prompt
)

from prompt_txt import MUTIL_STEP_INFER_Q_GENERATE_PROMPT,SQL_VALIDATION_PROMPT,MUTIL_STEP_INFER_COT_GENERATION_PROMPT,NL2SQL_FINAL_PROMPT


def refine_sql(generated_data,schema_info,knowledge_info,domain_info,data_kg,model):
    question = generated_data['question']
    question_analysis = generated_data['question_analysis']
    prompt = NL2SQL_FINAL_PROMPT.format(
        schema_info=schema_info,
        knowledge_info=knowledge_info,
        domain_info=domain_info,
        data_kg = data_kg,
        question=question,
        question_analysis=question_analysis
    )
    if model == 'claude':
        response_content = call_claude4_5(prompt)
    else:
        response_content = call_gpt5(prompt=prompt)

    clean_json_str = response_content.strip().replace('```json', '').replace('```', '')
    temp = json.loads(clean_json_str)
    sql = temp['sql']
    generated_data['sql'] = sql
    return generated_data

def generate_q_and_sql(
    schema_info: str,
    knowledge_info: str,
    domain_info: str,
    data_kg: str,
) -> Optional[Dict]:
    """
    步骤1: 根据上下文和指定难度，生成问题、解题步骤和SQL。
    """
    prompt = MUTIL_STEP_INFER_Q_GENERATE_PROMPT.format(
        schema_info=schema_info,
        knowledge_info=knowledge_info,
        domain_info=domain_info,
        data_kg = data_kg,
    )
    try:
        response_content = call_claude4_5(prompt)
        # response_content = call_gpt5(prompt=prompt)
        clean_json_str = response_content.strip().replace('```json', '').replace('```', '')
        data = json.loads(clean_json_str)
        
        # 验证所有必需的键是否存在
        if all(k in data for k in ['thought_process', 'question', 'sql']):
            data = refine_sql(generated_data=data,        
                                schema_info=schema_info,
                                knowledge_info=knowledge_info,
                                domain_info=domain_info,
                                data_kg = data_kg,model='claude')
            return data
        else:
            return None
            
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"错误: 在生成Q&SQL时发生未知错误: {e}")
        return None

def validate_sql_with_voting(
    schema_info: str,
    knowledge_info: str,
    domain_info: str,
    question: str,
    data_kg:str,
    sql_to_validate: str,
    question_analysis:str
) -> bool:
    """
    使用多线程并发调用3个不同的大模型进行SQL正确性验证，并根据投票结果返回最终结论。
    """
    result, is_error, error_msg = execute_sql_with_validation(sql_to_validate)
    if is_error:
        print(f'执行出错{error_msg}')
        return False

    prompt = SQL_VALIDATION_PROMPT.format(
        schema_info=schema_info,
        knowledge_info=knowledge_info,
        domain_info=domain_info,
        question=question,
        data_kg= data_kg,
        sql_to_validate=sql_to_validate,
    )

    # 三个模型的调用函数
    model_functions = [call_claude, call_gpt5]

    correct_votes = 0

    def call_model(func):
        """单个模型调用及结果解析"""
        model_name = func.__name__
        try:
            response_content = func(prompt=prompt)
            # 尝试提取JSON部分
            match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if not match:
                # 如果找不到 ```json ... ``` 块，就从字符串末尾开始找 {
                last_brace_index = response_content.rfind('{')
                if last_brace_index != -1:
                    json_str = response_content[last_brace_index:]
                    data = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON found", response_content, 0)
            else:
                data = json.loads(match.group(1))
            
            is_correct = data.get("is_correct") is True
            print(f"模型 {model_name} 投票: {'通过 ✓' if is_correct else '未通过 ✗'}")
            return is_correct
        except json.JSONDecodeError:
            print(f"模型 {model_name} 投票: 未通过 ✗ (无法解析JSON响应)")
            return False
        except Exception as e:
            print(f"模型 {model_name} 投票: 未通过 ✗ (调用失败: {e})")
            return False

    # 并发执行三个模型
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_model, func) for func in model_functions]
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                correct_votes += 1

    # 需要三个模型都认为正确才算通过
    print(f"投票正确数为{correct_votes}")
    return correct_votes == len(model_functions)


def generate_cot(
    schema_info: str,
    knowledge_info: str,
    domain_info: str,
    question: str,
    data_kg:str,
    sql: str
) -> Optional[str]:
    """
    步骤3: 为已通过校验的数据生成CoT。
    """
    prompt = MUTIL_STEP_INFER_COT_GENERATION_PROMPT.format(
        schema_info=schema_info,
        knowledge_info=knowledge_info,
        domain_info=domain_info,
        data_kg=data_kg,
        question=question,
        sql=sql
    )
    try:
        response_content = chat_gpt4o(prompt)
        return response_content.strip()
    except Exception as e:
        print(f"错误: 在生成CoT时发生未知错误: {e}")
        return None

# --- 数据处理流程 ---

def process_row_for_multistep(row_data) -> List[Dict]:
    """
    处理单行种子数据，执行生成、校验、CoT生成的完整流程。
    为 "L1","L2","L3","L4" 难度分别生成一条数据。
    """
    USER_INITIAL_PROMPT, _, _ = row_data # response 和 index_name 在此场景下可能不需要
    successful_results = []

    try:
        # 从原始prompt中提取所有必要信息
        schema_info = extract_schema_from_prompt(USER_INITIAL_PROMPT)
        # 注意：您的prompt中使用的是 knowledge_info 和 domain_info
        knowledge_info = extract_indicator_knowledge_from_prompt(USER_INITIAL_PROMPT)
        domain_info = extract_domain_rules_from_prompt(USER_INITIAL_PROMPT)
        data_kg = extract_data_kg_from_prompt(USER_INITIAL_PROMPT)

        if not all([schema_info, knowledge_info, domain_info]):
            print(f"警告: 无法从Prompt中提取完整上下文信息，跳过。")
            return []
        # difficulties = ["L3","L4"]
        difficulties = ["L1"]
        for difficulty in difficulties:
            # print(f"\n处理难度: {difficulty} for prompt: {USER_INITIAL_PROMPT[:60]}...")
            
            # 1. 生成问题和SQL
            generated_data = generate_q_and_sql(schema_info, knowledge_info, domain_info, data_kg)
            if not generated_data:
                print(f"-> 生成阶段失败。")
                continue # 继续处理下一种难度

            question = generated_data['question']
            sql = generated_data['sql']
            thought_process = generated_data['thought_process']
            steps = generated_data['solution_steps']
            question_analysis = generated_data['question_analysis']

            # 2. 校验SQL
            is_valid = validate_sql_with_voting(schema_info, knowledge_info, domain_info, question, data_kg, sql,question_analysis)
            if not is_valid:
                print(f"-> 校验阶段失败")
                continue

            # 3. 生成CoT
            cot_process = generate_cot(schema_info, knowledge_info, domain_info, question, data_kg, sql)
            if not cot_process:
                print(f"-> CoT生成阶段失败。")
                continue
            
            # 如果所有步骤都成功
            final_record = {
                'original_prompt': USER_INITIAL_PROMPT,
                'thought_process':thought_process,
                'steps':steps,
                'question': question,
                'sql': sql,
                'cot_process': cot_process
            }
            successful_results.append(final_record)
            
            # 稍微延时，避免过于频繁的API请求
            print('成功合成一条数据')
            time.sleep(1) 

    except Exception as e:
        print(f"处理行时发生严重错误: {e} for prompt: {USER_INITIAL_PROMPT[:60]}...")
    
    return successful_results


# --- 主函数 ---

def main_multistep_synthesis(target_count=100):
    """
    用于合成"多步推理"类别数据的主函数。
    
    Args:
        target_count: 目标生成的数据总量，达到或超过此数量后停止合成
    """
    WORKERS = 4  # 并发数设置为2
    BATCH_SIZE = 1
    CHECKPOINT_FILE = 'multistep_inference_data_checkpoint.csv'
    
    # 读取原始种子数据
    seed_data = read_data()
    print(f"加载了 {len(seed_data)} 条种子数据用于生成多步推理样本。")
    print(f"目标生成数据量: {target_count} 条")
    print(f"并发数: {WORKERS}")
    
    # 统计已生成的数据量
    current_count = 0
    if os.path.exists(CHECKPOINT_FILE):
        try:
            processed_df = pd.read_csv(CHECKPOINT_FILE)
            current_count = len(processed_df)
            print(f"从检查点文件加载了 {current_count} 条已生成的数据。")
        except Exception as e:
            print(f"读取检查点文件失败: {e}。将从头开始。")

    if current_count >= target_count:
        print(f"已达到目标数量 ({current_count}/{target_count})，无需继续生成。")
        return

    print(f"还需生成至少 {target_count - current_count} 条数据。")
    
    # 将种子数据转换为列表，便于随机采样
    seed_list = [tuple(row) for row in seed_data[['query', 'response', 'index_name']].itertuples(index=False)]
    
    batch_results = []
    lock = threading.Lock()  # 用于线程安全的结果收集
    
    # 使用进度条显示
    with tqdm(total=target_count, initial=current_count, desc="合成多步推理数据", file=sys.stderr) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = []
            
            while current_count < target_count:
                # 提交任务到线程池，直到达到目标数量
                if len(futures) < WORKERS * 4:  # 保持一定数量的待处理任务
                    sampled_row = random.choice(seed_list)
                    future = executor.submit(process_row_for_multistep, sampled_row)
                    futures.append(future)
                
                # 处理已完成的任务
                done_futures = []
                for future in futures:
                    if future.done():
                        done_futures.append(future)
                
                for future in done_futures:
                    futures.remove(future)
                    try:
                        results_list = future.result()
                        if results_list:
                            success_count = len(results_list)
                            
                            with lock:
                                batch_results.extend(results_list)
                                current_count += success_count
                                pbar.update(success_count)
                                
                                # 达到批次大小时保存
                                if len(batch_results) >= BATCH_SIZE:
                                    save_checkpoint(batch_results, CHECKPOINT_FILE)
                                    batch_results = []
                    except Exception as e:
                        print(f"处理任务时发生错误: {e}")
                
                # 如果已达到目标，取消剩余任务
                if current_count >= target_count:
                    for future in futures:
                        future.cancel()
                    break
                
                # 短暂休眠，避免过于频繁的循环
                time.sleep(0.5)
            
            # 等待所有剩余任务完成
            for future in concurrent.futures.as_completed(futures):
                if current_count >= target_count:
                    break
                try:
                    results_list = future.result()
                    if results_list:
                        success_count = len(results_list)
                        with lock:
                            batch_results.extend(results_list)
                            current_count += success_count
                            pbar.update(success_count)
                except Exception as e:
                    print(f"处理任务时发生错误: {e}")
    
    # 保存最后一批不足BATCH_SIZE的结果
    if batch_results:
        save_checkpoint(batch_results, CHECKPOINT_FILE)

    print(f"\n所有多步推理数据合成任务处理完成。")
    if os.path.exists(CHECKPOINT_FILE):
        final_df = pd.read_csv(CHECKPOINT_FILE)
        print(f"总共成功生成并保存了 {len(final_df)} 条多步推理数据到 {CHECKPOINT_FILE}")


if __name__ == "__main__":
    print("开始合成【多步推理数据】...")
    main_multistep_synthesis(target_count=10000)