import pandas as pd
import re
import os
import json # 导入 json
from utils import verify_sql_equivalence
from taming_sql import SQLEquivalenceValidator
from utils import * # 导入 chat_gpt4o

# --- 用于提取的辅助函数 (来自原始代码) ---

def extract_think_answer_from_response(text):
    """从 response 列中提取 <answer>...</answer>"""
    if pd.isna(text):
        return None
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None
    return answer

def extract_schema_from_prompt(full_prompt: str) -> str:
    """
    从完整的初始Prompt中提取"--- 2.schema kg"和"--- 3.knowledge graph"之间的所有内容。
    """
    start_marker = "--- 2.schema kg"
    end_marker = "--- 3.knowledge graph"
    
    start_idx = full_prompt.find(start_marker)
    end_idx = full_prompt.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        return ""
    
    # 从start_marker之后开始提取，到end_marker之前结束
    start_idx += len(start_marker)
    schema_content = full_prompt[start_idx:end_idx].strip()
    
    return schema_content

def extract_question_from_prompt(input_text):
    regex_pattern = r".*问题：(.*?)，写出对应的SQL语句"
    match = re.search(regex_pattern, input_text, re.DOTALL)

    if match:
        # 捕获组从1开始索引，group(1) 提取第一个括号内的内容
        user_question = match.group(1).strip()
        return user_question
    else:
        return ''

def extract_indicator_knowledge_from_prompt(full_prompt: str) -> str:
    """
    使用正则表达式从完整Prompt中提取“指标知识”部分（--- 3.knowledge graph 段落）
    """
    pattern = re.compile(r"---\s*3\.knowledge graph\s*(.*?)---\s*4\.domain kg", re.DOTALL)
    match = pattern.search(full_prompt)
    if match:
        return match.group(1).strip()
    return ""


def extract_domain_rules_from_prompt(full_prompt: str) -> str:
    """
    使用正则表达式从完整Prompt中提取“领域业务规则”部分（--- 4.domain kg 段落）
    """
    pattern = re.compile(r"---\s*4\.domain kg\s*(.*?)---\s*5\.data kg", re.DOTALL)
    match = pattern.search(full_prompt)
    if match:
        return match.group(1).strip()
    return ""

def extract_data_kg_from_prompt(full_prompt: str) -> str:
    pattern = re.compile(r"---\s*5\.data kg\s*(.*?)---\s*6\.few shot", re.DOTALL)
    match = pattern.search(full_prompt)
    if match:
        return match.group(1).strip().replace("-", "")
    return ""

SEMANTIC_PROMPT_TEMPLATE = """你是一个语义分析专家。你的任务是判断两个句子是否**语义完全一致**。

"语义完全一致" 的定义是：
1. 两个句子传达的核心含义或问题必须完全相同。
2. 任何一个句子都不能比另一个句子包含**额外**的语义信息（例如：额外的约束、不同的侧重点或主题）。

句子 A: {sentence_a}
句子 B: {sentence_b}

请严格按照以下 JSON 格式返回你的分析结果，不要包含任何额外的解释或 ```json ... ``` 标记：
{{
"reason": "简要解释你的判断理由。"
"is_identical": true, // 如果语义完全一致，则为 true，否则为 false
}}
"""

def main_verification(input_path, output_path):
    
    print(f"开始处理文件：{input_path}")
    
    try:
        df = pd.read_csv(input_path)

    except FileNotFoundError:
        print(f"错误：文件未找到 {input_path}")
        return
    except Exception as e:
        print(f"读取 CSV 时出错：{e}")
        return

    # 初始化 LLM 验证器
    try:
        llm_validator = SQLEquivalenceValidator()
    except Exception as e:
        print(f"实例化 SQLEquivalenceValidator 时出错：{e}")
        print("将退出脚本。")
        return

    # 初始化新列
    df["equivalent_exec"] = None
    df["msg_exec"] = None
    df["equivalent_llm"] = None
    df["msg_llm"] = None
    df["equivalent_semantic"] = None # (新) 用于歧义澄清
    df["msg_semantic"] = None      # (新) 用于歧义澄清
    df["error_processing"] = None # 用于记录处理过程中的错误

    print(f"总行数：{len(df)}")

    # 定义需要执行验证的类型
    exec_validation_types = ['sql', '多步推理', '反思数据']
    llm_validation_type = '维度退化'
    semantic_validation_type = '歧义澄清' # (新)

    for idx, row in df.iterrows():
        
        if (idx + 1) % 50 == 0:
            print(f"正在处理第 {idx + 1}行...")
            
        row_type = row.get('type')
        
        try:
            golden_output = extract_think_answer_from_response(row['response'])
            model_output = row.get('answer')
            if pd.isna(model_output):
                 model_output = "" # 确保是字符串

            # --- 3. 根据类型进行验证 ---
            
            if row_type in exec_validation_types:
                # --- 3a. 执行验证 (SQL, 多步推理, 反思数据) ---
                
                # (重构) SQL 检查移到这里
                is_model_sql_valid = isinstance(model_output, str) and re.search(r"select", model_output, re.IGNORECASE)
                is_gold_sql_valid = isinstance(golden_output, str) and re.search(r"select", golden_output, re.IGNORECASE)

                if not is_gold_sql_valid:
                    df.at[idx, "error_processing"] = "Exec Val: Golden SQL (from response) 无效或缺失"
                    continue
                    
                if not is_model_sql_valid:
                    df.at[idx, "equivalent_exec"] = None # (注意：原代码这里是 None，也许该设为 False?)
                    df.at[idx, "msg_exec"] = "Model SQL missing SELECT"
                    continue
                
                try:
                    context_dic = {
                        'schema' : extract_schema_from_prompt(row['query']),
                        'question' : extract_question_from_prompt(row['query']),
                        'knowledge_graph' : extract_indicator_knowledge_from_prompt(row['query']),
                        'domain_rules' : extract_domain_rules_from_prompt(row['query']),
                        'data_kg' : extract_data_kg_from_prompt(row['query'])
                    }
                    equivalent, msg = verify_sql_equivalence(golden_output, model_output,row_type=='多步推理',context_dic)
                    df.at[idx, "equivalent_exec"] = equivalent
                    df.at[idx, "msg_exec"] = msg
                except Exception as e:
                    df.at[idx, "equivalent_exec"] = None
                    df.at[idx, "msg_exec"] = f"Execution Error: {e}"

            elif row_type == llm_validation_type:
                # --- 3b. LLM 验证 (维度退化) ---
                
                # (重构) SQL 检查移到这里
                is_model_sql_valid = isinstance(model_output, str) and re.search(r"select", model_output, re.IGNORECASE)
                is_gold_sql_valid = isinstance(golden_output, str) and re.search(r"select", golden_output, re.IGNORECASE)

                if not is_gold_sql_valid:
                    df.at[idx, "error_processing"] = "LLM Val: Golden SQL (from response) 无效或缺失"
                    continue
                    
                if not is_model_sql_valid:
                    df.at[idx, "equivalent_llm"] = False # 如果模型没给SQL，那它就是错的
                    df.at[idx, "msg_llm"] = "Model SQL missing SELECT"
                    continue

                query_text = row.get('query')
                
                # 提取 Schema 和 NL_Query
                try:
                    schema = extract_schema_from_prompt(query_text)
                    nl_query = extract_question_from_prompt(query_text)
                    
                    if not schema:
                         df.at[idx, "error_processing"] = "LLM Val: Schema extract failed"
                         continue
                    if not nl_query:
                         df.at[idx, "error_processing"] = "LLM Val: Question extract failed"
                         continue
                         
                except Exception as e:
                    df.at[idx, "error_processing"] = f"LLM Val: schema/question extract failed: {e}"
                    continue
                
                # LLM 校验 SQL 等价
                try:
                    is_eq, reasons = llm_validator.is_equivalent(
                        sql1=model_output,     # Model SQL
                        sql2=golden_output,    # Golden SQL
                        schema=schema,
                        nl_query=nl_query,
                        runs=3
                    )
                    df.at[idx, "equivalent_llm"] = is_eq
                    df.at[idx, "msg_llm"] = str(reasons)
                    
                except ValueError as e: # 捕获原始代码中提到的特定错误
                    df.at[idx, "error_processing"] = f"LLM Val: Validation error: {e}"
                except Exception as e:
                    df.at[idx, "error_processing"] = f"LLM Val: Unexpected error: {e}"

            elif row_type == semantic_validation_type:
                # --- (新) 3c. 语义一致性验证 (歧义澄清) ---
                
                # 此时，golden_output 和 model_output 是澄清句
                golden_clarification = golden_output
                model_clarification = model_output

                # (新) 检查澄清句是否有效 (非空字符串)
                if not isinstance(golden_clarification, str) or not golden_clarification.strip():
                    df.at[idx, "error_processing"] = "Semantic Val: Golden clarification (from response) 无效或缺失"
                    continue
                
                if not isinstance(model_clarification, str) or not model_clarification.strip():
                    df.at[idx, "equivalent_semantic"] = False # 模型没有提供澄清
                    df.at[idx, "msg_semantic"] = "Semantic Val: Model clarification (from answer) 无效或缺失"
                    continue

                # 调用 LLM 进行语义判断
                try:
                    prompt = SEMANTIC_PROMPT_TEMPLATE.format(
                        sentence_a=golden_clarification,
                        sentence_b=model_clarification
                    )
                    
                    # 使用较低的 temperature (0.1) 以获得更稳定的判断
                    llm_response_str = chat_gpt4o(prompt, temperature=0.1)
                    
                    try:
                        llm_response_json = json.loads(llm_response_str)
                    except json.JSONDecodeError:
                        # 如果直接解析失败，尝试从 ```json ... ``` 中提取
                        json_match = re.search(r"\{.*\}", llm_response_str, re.DOTALL)
                        if not json_match:
                            raise ValueError(f"LLM response was not valid JSON: {llm_response_str}")
                        llm_response_json = json.loads(json_match.group(0))

                    df.at[idx, "equivalent_semantic"] = llm_response_json.get("is_identical")
                    df.at[idx, "msg_semantic"] = llm_response_json.get("reason", "No reason provided.")
                    
                except json.JSONDecodeError:
                    df.at[idx, "error_processing"] = f"Semantic Val: Failed to parse JSON response: {llm_response_str}"
                except Exception as e:
                    df.at[idx, "error_processing"] = f"Semantic Val: Unexpected error: {e}"

            else:
                df.at[idx, "error_processing"] = f"未知的 'type': {row_type}"

        except Exception as e:
            df.at[idx, "error_processing"] = f"处理行时发生意外错误: {e}"


    # --- 4. 保存结果 ---
    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n🎉 处理完成！结果已保存至：{output_path}")
    except Exception as e:
        print(f"保存文件时出错：{e}")

# --- 运行主函数 ---
if __name__ == "__main__":
    # 使用你提供的文件路径
    input_file = "input.csv"
    
    # 定义输出路径
    output_file = "output.csv"
    
    # 检查输入文件是否存在 (在你的真实环境中)
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在于：{input_file}")
        print("请确保路径正确或文件存在。")
    else:
        main_verification(input_file, output_file)