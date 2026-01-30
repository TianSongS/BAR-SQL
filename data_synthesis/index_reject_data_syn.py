import pandas as pd
import re
from tqdm import tqdm
import requests
import os
import concurrent.futures
from utils import extract_schema_from_prompt,extract_question_from_prompt,replace_user_query_with_new_question,chat_gpt4o,get_llm_res,save_checkpoint,read_data



# 导入新的Prompt模板
from prompt_txt import REJECTION_COT_SYN_PROMPT 

def get_sim_query(query, replace_type='index'):
    """
    调用外部API，对查询语句中的实体进行随机替换。
    我们将专门用它来替换'index'（指标）。
    """
    # 注意：确保这个URL是您可以访问的内网地址
    api_url = "http://0.0.0.0:9002/random_replace"
    json_dict = {
        "query": query,
        "random_type": replace_type,
        "random_num": 1  # 每次只替换1个指标，确保意图清晰
    }
    try:
        response = requests.post(api_url, json=json_dict, timeout=10)
        response.raise_for_status() # 如果请求失败则抛出异常
        
        data = response.json()
        new_query = data.get('new_query')
        
        # 解析被替换的指标名称
        replace_massage = data.get('replace_massage', '')
        replaced_indicator = ''
        if '->' in replace_massage:
            try:
                # 示例: 'index:上险量 -> 试驾到定单转化率\n'
                replaced_indicator = replace_massage.split('->')[1].strip()
            except IndexError:
                pass # 解析失败则返回空

        # 确保返回的查询确实发生了改变
        if new_query and new_query != query:
            return new_query, replaced_indicator
        else:
            return None, None

    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        return None, None


def generate_rejection_cot(unanswerable_question: str, schema_info: str):
    """
    调用LLM，生成拒识的思考过程和固定的结论。
    """
    prompt = REJECTION_COT_SYN_PROMPT.format(
        schema_info=schema_info,
        unanswerable_question=unanswerable_question
    )
    
    try:
        # 您可以使用 chat_gpt4o 或其他 get_llm_res 函数
        response_content = chat_gpt4o(prompt) 
        
        # 解析输出
        pattern = r"\[思考过程\](.*?)\[结论\](.*)"
        match = re.search(pattern, response_content, re.DOTALL)
        if match:
            thought_process = match.group(1).strip()
            conclusion = match.group(2).strip()
            # 增加一个校验，确保结论是正确的拒识话术
            if "由于缺少对应的指标定义" in conclusion:
                 return {"thought_process": thought_process, "conclusion": conclusion}
        
        # 如果解析失败或结论不正确，返回错误信息
        return {"error": "Failed to parse rejection CoT response.", "raw_response": response_content}

    except Exception as e:
        return {"error": f"An unexpected error occurred during CoT generation: {e}"}


def process_row_for_rejection(row_data):
    """
    处理单行数据，执行指标替换和拒识COT生成的完整流程。
    """
    USER_INITIAL_PROMPT, response, index_name = row_data
    try:
        original_question = extract_question_from_prompt(USER_INITIAL_PROMPT)
        schema_info = extract_schema_from_prompt(USER_INITIAL_PROMPT)

        if not original_question or not schema_info:
            return None

        # 1. 指标替换
        swapped_question, new_indicator = get_sim_query(original_question, replace_type='index')

        # 如果替换成功
        if swapped_question:
            # 2. 生成拒识COT
            rejection_result = generate_rejection_cot(swapped_question, schema_info)
            
            if "error" not in rejection_result:
                return {
                    'original_prompt': USER_INITIAL_PROMPT,
                    'original_question': original_question,
                    'swapped_question': swapped_question, # 替换了指标的问题
                    'new_indicator': new_indicator, # 被换上去的新指标
                    'cot_process': rejection_result['thought_process'],
                    'conclusion': rejection_result['conclusion']
                }
        return None # 如果替换失败或后续步骤出错，则返回None
        
    except Exception as e:
        print(f"处理行时发生错误: {e} for prompt: {USER_INITIAL_PROMPT[:50]}...")
        return None

def main_rejection_synthesis():
    """
    用于合成指标拒识数据的的主函数。
    """
    # --- 配置参数 ---
    WORKERS = 4
    BATCH_SIZE = 30
    CHECKPOINT_FILE = 'indicator_rejection_data_1230.csv'

    data_to_process = read_data()
    
    print(f"准备开始处理 {len(data_to_process)} 条数据用于生成拒识样本。")
    
    # --- 断点续传逻辑 (与您原有的main函数相同) ---
    processed_prompts = set()
    if os.path.exists(CHECKPOINT_FILE):
        try:
            processed_df = pd.read_csv(CHECKPOINT_FILE)
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

    # --- 并发处理 ---
    batch_results = []
    tasks = [tuple(row) for row in data_to_process[['query', 'response', 'index_name']].itertuples(index=False)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_prompt = {executor.submit(process_row_for_rejection, task): task[0] for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_prompt), total=len(tasks), desc="合成拒识数据"):
            result = future.result()
            if result:
                batch_results.append(result)
            
            if len(batch_results) >= BATCH_SIZE:
                save_checkpoint(batch_results, CHECKPOINT_FILE)
                batch_results = []

    # --- 最终保存 ---
    if batch_results:
        save_checkpoint(batch_results, CHECKPOINT_FILE)

    print("\n所有拒识数据合成任务处理完成。")
    if os.path.exists(CHECKPOINT_FILE):
        final_df = pd.read_csv(CHECKPOINT_FILE)
        print(f"总共成功生成并保存了 {len(final_df)} 条拒识数据到 {CHECKPOINT_FILE}")


if __name__ == "__main__":

    print("开始合成【指标拒识数据】...")
    main_rejection_synthesis()
