import functools
print = functools.partial(print, flush=True)
import json
import pandas as pd
import re
from tqdm import tqdm
import os
from typing import Dict, List, Optional, Tuple
import concurrent.futures
import time
import sys
import random
import pymysql
import traceback
from utils import (
    extract_schema_from_prompt,
    extract_question_from_prompt,
    extract_indicator_knowledge_from_prompt,
    extract_domain_rules_from_prompt,
    chat_gpt4o,
    save_checkpoint,
    get_deepseekV3,
    read_data
)

ERROR_TYPES = [
    {
        "category": "语法错误",
        "type_name": "关键字拼写错误 (Keyword Misspelling)",
        "instruction": "将SQL中的一个关键字（如 SELECT, FROM, WHERE）进行错误的拼写。"
    },
    {
        "category": "语法错误",
        "type_name": "函数名拼写错误 (Function Misspelling)",
        "instruction": "将SQL中的一个函数（如 COUNT, CONCAT）进行错误的拼写。"
    },
    {
        "category": "语法错误",
        "type_name": "非法字符注入 (Illegal Character Injection)",
        "instruction": "在SQL语句的非法位置（例如，列名前）插入一个不合法的特殊字符，如 '@', '#', '&'。"
    },
    {
        "category": "语法错误",
        "type_name": "删除关键运算符 (Critical Operator Deletion)",
        "instruction": "从SQL中删除一个关键的运算符或符号，例如 WHERE 子句中的 '=' 或括号。"
    },
    {
        "category": "语法错误",
        "type_name": "UNION列数不匹配 (UNION Column Mismatch)",
        "instruction": "在原始查询后添加一个 'UNION ALL' 子句，其 SELECT 列表的列数与原始查询不匹配。"
    },
    {
        "category": "语法错误",
        "type_name": "窗口函数缺少OVER子句 (Window Function without OVER)",
        "instruction": "在 SELECT 列表中添加一个窗口函数（如 ROW_NUMBER()），但省略必需的 OVER () 子句。"
    },
    {
        "category": "标识符错误",
        "type_name": "列名歧义 (Ambiguous Column Name)",
        "instruction": "修改查询，使其引用一个在多个连接的表中都存在的、未指定表别名的列（例如 'day_id'），从而导致列名歧义错误。"
    },
    {
        "category": "运行时错误",
        "type_name": "除零错误 (Division by Zero)",
        "instruction": "修改查询，使其包含一个会导致除以零错误的算术表达式。"
    },
    {
        "category": "运行时错误",
        "type_name": "数据类型转换错误 (Data Type Conversion Error)",
        "instruction": "对一个明确为字符串的列或值尝试进行非法的算术运算，例如 'AND t4.series_name + 1 = 1'。"
    },
    {
        "category": "运行时错误",
        "type_name": "非法GROUP BY用法 (Invalid GROUP BY Usage)",
        "instruction": "添加一个 GROUP BY 子句，但在 SELECT 列表中包含一个既未被聚合也未在 GROUP BY 子句中声明的列。"
    }
]

# 数据库连接
def get_db_connection():
    """获取数据库连接"""
    return pymysql.connect(
        host='dip...',
        port=9030,
        user='user',
        passwd='pwd',
        db='db',
        charset='utf8'
    )

def execute_sql_with_error(sql: str) -> Tuple[bool, str, Optional[List[Dict]]]:
    """
    执行SQL并捕获错误信息
    返回: (是否成功, 错误信息或成功消息, 查询结果)
    """
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        connection.commit()
        return True, "SQL executed successfully", result
    except pymysql.err.ProgrammingError as e:
        # 提取最后一行的错误信息
        error_msg = str(e.args[1]) if len(e.args) > 1 else str(e)
        return False, error_msg, None
    except Exception as e:
        # 获取完整的错误追踪信息的最后一行
        tb_lines = traceback.format_exc().strip().split('\n')
        error_msg = tb_lines[-1] if tb_lines else str(e)
        return False, error_msg, None
    finally:
        if connection:
            connection.close()

def extract_sql_from_response(response: str) -> Optional[str]:
    """从response中提取SQL语句"""
    # 尝试多种模式匹配SQL
    patterns = [
        r'SQL\s*->\s*(SELECT[\s\S]+?)(?:\n\n|$)',  # SQL -> 开头的模式
        r'```sql\s*([\s\S]+?)\s*```',  # markdown代码块
        r'(SELECT[\s\S]+?)(?:\n\n|$)',  # 直接以SELECT开头
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def generate_error_sql_prompt(sql: str, error_type: dict) -> str:
    """生成错误SQL的prompt"""
    prompt_template = f"""
你是一位专业的SQL数据生成专家。你的任务是根据给定的[原始SQL]生成两个错误的、不可执行的SQL查询。

**[建议策略]**
- 错误类型: {error_type['type_name']}
- 指令: {error_type['instruction']}

**[主要任务]**
生成 **两个** 错误的SQL查询：
1.  **第一个查询**: 必须严格遵循上述[建议策略]来生成。
2.  **第二个查询**: 请自行选择一种 **与第一个不同** 的、任意类型的错误来生成。

**[备用指令]**
如果你判断[建议策略]难以应用或不适用于给定的[原始SQL]，你必须忽略它。转而，你的任务是生成 **两个** 包含不同类型错误的SQL查询（例如，一个语法错误和一个运行时错误）。

**[原始SQL]**
```sql
{sql.strip()}
```

**[输出格式]**
无论你遵循[主要任务]还是[备用指令]，都 **必须** 返回一个包含两个JSON对象的JSON数组。每个对象代表一个错误的SQL查询。格式如下：

```json
[
  {{
    "error_type_applied": "...",
    "modified_sql": "..."
  }},
  {{
    "error_type_applied": "...",
    "modified_sql": "..."
  }}
]
```

# error_type_applied中使用中文解释,另外在错误的SQL语句中不要进行任何的注释！
"""
    return prompt_template

def generate_reflection_prompt(
    schema_info: str,
    knowledge_info: str,
    domain_info: str,
    question: str,
    error_sql: str,
    error_message: str,
    correct_sql: str,
    error_type: str
) -> str:
    """生成模型反思的prompt"""
    prompt = f"""你是一位经验丰富的数据分析师，现在需要分析一个SQL执行错误并进行反思改正。

**[背景信息]**
{schema_info}

{knowledge_info}

{domain_info}

**[用户问题]**
{question}

**[错误的SQL查询]**
```sql
{error_sql}
```

**[执行错误信息]**
```
{error_message}
```

**[正确的SQL查询]**
```sql
{correct_sql}
```

**[任务要求]**
请你模拟一个逐步反思和修正的过程，包括：
1. 错误分析：详细分析错误信息，识别具体是什么类型的错误（如语法错误、标识符错误、运行时错误等）
2. 错误定位：准确定位SQL中导致错误的具体部分
3. 修正思路：解释应该如何修正这个错误，为什么要这样修正
4. 逐步改进：展示从错误SQL到正确SQL的思考
5. 经验总结：总结这类错误的特征和避免方法

请以第一人称的方式，模拟真实的思考过程，包括可能的试错和调整。输出应该自然流畅，体现出逐步分析和解决问题的过程。

**[输出格式]**
请直接输出反思过程的文本，不需要任何格式标记。反思过程应该详细、自然，体现出真实的问题解决思路。

**[参考信息]**
正确的SQL应该能够回答用户的问题，你的反思过程应该最终得出一个正确的解决方案。
"""
    return prompt

def get_error_sql_generation(prompt: str) -> Optional[List[Dict]]:
    """调用LLM生成错误SQL"""
    
    try:
        content = get_deepseekV3(prompt)
            
        # 提取JSON
        match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = content.strip()
        
        data = json.loads(json_str)
        if isinstance(data, list) and len(data) == 2:
            return data
        else:
            print("生成的错误SQL格式不正确")
            return None
                
            
    except Exception as e:
        print(f"生成错误SQL时发生异常: {e}")
        return None

def generate_reflection(prompt: str) -> Optional[str]:
    """调用LLM生成反思内容"""
    try:
        response_content = get_deepseekV3(prompt=prompt)
        return response_content.strip()
    except Exception as e:
        print(f"生成反思内容时发生异常: {e}")
        return None

def process_row_for_reflection(row_data) -> List[Dict]:
    """处理单行数据，生成反思训练数据"""
    USER_INITIAL_PROMPT, response, index_name = row_data
    successful_results = []
    
    try:
        # 1. 提取必要信息
        schema_info = extract_schema_from_prompt(USER_INITIAL_PROMPT)
        knowledge_info = extract_indicator_knowledge_from_prompt(USER_INITIAL_PROMPT)
        domain_info = extract_domain_rules_from_prompt(USER_INITIAL_PROMPT)
        question = extract_question_from_prompt(USER_INITIAL_PROMPT)
        
        # 提取正确的SQL
        correct_sql = extract_sql_from_response(response)
        
        if not all([schema_info, knowledge_info, domain_info, question, correct_sql]):
            print(f"警告: 无法从数据中提取完整信息，跳过。")
            return []
        
        # 2. 随机选择错误类型并生成错误SQL
        error_type = random.choice(ERROR_TYPES)
        error_prompt = generate_error_sql_prompt(correct_sql, error_type)
        error_sqls = get_error_sql_generation(error_prompt)
        
        if not error_sqls:
            print("生成错误SQL失败，跳过。")
            return []
        
        # 3. 对每个错误SQL进行处理
        for error_sql_data in error_sqls:
            error_sql = error_sql_data.get('modified_sql', '')
            error_type_applied = error_sql_data.get('error_type_applied', '')
            
            if not error_sql:
                continue
            
            # 4. 执行错误SQL获取错误信息
            success, error_message, _ = execute_sql_with_error(error_sql)
            
            if success:
                # 如果SQL意外成功执行，跳过
                print(f"警告: 错误SQL意外成功执行，跳过。")
                continue
            
            # 5. 生成反思内容
            reflection_prompt = generate_reflection_prompt(
                schema_info=schema_info,
                knowledge_info=knowledge_info,
                domain_info=domain_info,
                question=question,
                error_sql=error_sql,
                error_message=error_message,
                correct_sql=correct_sql,
                error_type=error_type_applied
            )
            
            reflection_content = generate_reflection(reflection_prompt)
            
            if not reflection_content:
                print("生成反思内容失败，跳过。")
                continue
            
            # 6. 构建最终记录
            final_record = {
                'original_prompt': USER_INITIAL_PROMPT,
                'question': question,
                'correct_sql': correct_sql,
                'error_sql': error_sql,
                'error_type': error_type_applied,
                'error_message': error_message,
                'reflection_process': reflection_content,
                'schema_info': schema_info,
                'knowledge_info': knowledge_info,
                'domain_info': domain_info
            }
            
            successful_results.append(final_record)
            print(f"成功生成一条反思数据: {error_type_applied}")
            
            # 稍微延时，避免过于频繁的API请求
            time.sleep(1)
            
    except Exception as e:
        print(f"处理行时发生错误: {e}")
        
    return successful_results

def main_reflection_synthesis():
    """主函数：合成模型反思数据"""
    WORKERS = 4
    BATCH_SIZE = 5
    CHECKPOINT_FILE = 'reflection_data_checkpoint.csv'
    
    # 读取原始数据
    data_to_process = read_data()
    print(f"准备开始处理 {len(data_to_process)} 条种子数据用于生成反思样本。")
    
    # 检查已处理的数据
    processed_prompts = set()
    if os.path.exists(CHECKPOINT_FILE):
        try:
            processed_df = pd.read_csv(CHECKPOINT_FILE)
            processed_prompts = set(processed_df['original_prompt'])
            print(f"从检查点文件加载了 {len(processed_prompts)} 条已处理的种子数据。")
        except Exception as e:
            print(f"读取检查点文件失败: {e}。将从头开始。")
    
    # 过滤已处理的数据
    if processed_prompts:
        initial_count = len(data_to_process)
        data_to_process = data_to_process[~data_to_process['query'].isin(processed_prompts)]
        print(f"已过滤 {initial_count - len(data_to_process)} 条已处理的数据。剩余 {len(data_to_process)} 条待处理。")
    
    if data_to_process.empty:
        print("所有数据都已处理完毕。")
        return
    
    # 准备任务
    batch_results = []
    tasks = [tuple(row) for row in data_to_process[['query', 'response', 'index_name']].itertuples(index=False)]
    
    # 并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_prompt = {executor.submit(process_row_for_reflection, task): task[0] for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_prompt), 
                          total=len(tasks), 
                          desc="合成反思数据",
                          file=sys.stderr):
            try:
                results_list = future.result()
                if results_list:
                    batch_results.extend(results_list)
                
                # 定期保存
                if len(batch_results) >= BATCH_SIZE:
                    save_checkpoint(batch_results, CHECKPOINT_FILE)
                    batch_results = []
                    
            except Exception as e:
                print(f"处理任务时发生错误: {e}")
    
    # 保存最后一批结果
    if batch_results:
        save_checkpoint(batch_results, CHECKPOINT_FILE)
    
    print("\n所有反思数据合成任务处理完成。")
    if os.path.exists(CHECKPOINT_FILE):
        final_df = pd.read_csv(CHECKPOINT_FILE)
        print(f"总共成功生成并保存了 {len(final_df)} 条反思数据到 {CHECKPOINT_FILE}")

if __name__ == "__main__":
    print("开始合成【模型反思数据】...")
    main_reflection_synthesis()
