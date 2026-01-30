from openai import OpenAI
import re
import requests
import json
import uuid
import pandas as pd
import os
import pymysql
import pymysql.err


client_deep = OpenAI(
    # Token
    api_key="ak-...",
    base_url="https://...",
)

client_claude = OpenAI(
    api_key="EMPTY",
    base_url="http://.../v1",
    default_headers={
        "BCS-APIHub-RequestId": str(uuid.uuid4()),
        "X-CHJ-GWToken": "eyJ...",
    },
    timeout=150,
    max_retries=2,
)

def get_deepseekV3(prompt):
    chat_completion = client_deep.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="deepseek-ai__deepseek-v3",
        temperature=0.6,
        stream=False,
    )

    return chat_completion.choices[0].message.content

def call_claude(prompt: str, temperature: float = 0.6) -> str:
    """
    调用模型进行对话
    
    Args:
        user_question: 用户的问题
        temperature: 温度参数，控制输出的随机性（0-2之间，默认0.7）
        
    Returns:
        模型的输出内容
    """
    response = client_claude.chat.completions.create(
        model="google-claude-opus-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    return response.choices[0].message.content


def call_claude4_5(prompt: str, temperature: float = 0.6) -> str:
    """
    调用模型进行对话
    
    Args:
        user_question: 用户的问题
        temperature: 温度参数，控制输出的随机性（0-2之间，默认0.7）
        
    Returns:
        模型的输出内容
    """
    response = client_claude.chat.completions.create(
        model="aws-claude-sonnet-4-5",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    return response.choices[0].message.content

def call_gpt5(prompt, 
                    system_prompt="You are a helpful assistant.",
                    model="azure-gpt-5", 
                    stream=False, 
                    reasoning_effort="medium",
                    temperature=0.7):
    """
    调用4.0/3.0 的 GPT-5 聊天接口

    参数:
        prompt (str): 用户输入的提示词
        system_prompt (str): 系统提示词，默认为 "You are a helpful assistant."
        model (str): 模型ID，默认 "azure-gpt-5"
        stream (bool): 是否启用流式响应，默认 False
        reasoning_effort (str): 推理强度，可选 "low" | "medium" | "high"
        temperature (float): 温度系数，范围0-2，默认0.7
    
    返回:
        str: AI的回复内容，如果出错返回None
    """

    # 接口地址
    url = "http://..."

    # 你的业务 Token
    token = "eyJ..."

    # 请求头
    headers = {
        "Token": token
    }

    # 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # 请求体
    payload = {
        "model": model,
        "messages": messages,
        "reasoning_effort": reasoning_effort,
        "stream": stream,
        "temperature": temperature  # 添加温度参数
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=150)
        response.raise_for_status()
        result = response.json()
        
        # 提取AI的回复内容
        if result and "choices" in result and len(result["choices"]) > 0:
            ai_response = result["choices"][0]["message"]["content"]
            print(f"🤖 AI回复：\n{ai_response}")
            return ai_response
        else:
            print("❌ 响应格式异常")
            return None
            
    except requests.exceptions.RequestException as e:
        print("❌ 请求失败：", e)
        return None
    except json.JSONDecodeError:
        print("❌ 响应内容无法解析为JSON：", response.text)
        return None

def chat_gpt4o(prompt,history=[],temperature=0.8,maxTokens=-1):
    payload = json.dumps({
              "model": "chatgpt-4o-latest-20250326",
            "temperature": temperature,
              "messages": [
                {
                  "role": "user",
                  "content": [
                    {
                      "type": "text",
                      "text": prompt
                    }
                  ]
                }
              ]
        })

    if maxTokens != -1:
        payload["maxTokens"]=maxTokens

    headers = {
        "Token": 'token'
    }
    while True:
        try:
            response = requests.request("POST", f'http://...', headers=headers, data=payload)
            response = json.loads(response.text)
            return response['choices'][0]['message']['content']

        except Exception as e:
            pass


def read_data():
    # --- 数据加载 ---
    df = pd.read_csv('/....csv')
    df = df[['query', 'response']].copy()
    return df


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
    
    start_idx += len(start_marker)
    schema_content = full_prompt[start_idx:end_idx].strip()
    
    return schema_content

def get_llm_res(prompt):
    client = OpenAI(
        # LPAI Token
        api_key="ak-...",
        # api url，注意截止到/v1即可，后续的chat/completions client会自动拼上
        base_url="https://.../deepseek-v3/v1",
    )

    response = client.chat.completions.create(
        model="deepseek-ai__deepseek-v3",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.2,
        max_tokens=500,
        response_format={"type": "json_object"}
    )
    return response

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

def replace_user_query_with_new_question(text: str, new_question: str) -> str:
    """
    将 USER_INITIAL_PROMPT 中 '问题：' 和 '答案：' 之间的内容替换为新的问题。

    参数:
        text (str): 原始文本，包含 USER_INITIAL_PROMPT。
        new_question (str): 要替换的新问题内容。

    返回:
        str: 替换后的新文本。
    """
    pattern = r'(?s)(问题：)(.*?)(，写出对应的SQL语句)'
    replacement = r'\1' + new_question + r'\3'
    return re.sub(pattern, replacement, text, count=1)

def save_checkpoint(data_list, filename):
    """
    将一批数据追加保存到CSV文件中。
    """
    if not data_list:
        return
    
    df_to_save = pd.DataFrame(data_list)
    # 检查文件是否存在以决定是否写入header
    header = not os.path.exists(filename)
    df_to_save.to_csv(filename, mode='a', index=False, header=header, encoding='utf-8-sig')
    print(f"已保存 {len(data_list)} 条数据到 {filename}")

# --- 1. 数据库连接配置 ---
# 使用您提供的连接信息
connection_config = {
    'host': 'dip...',
    'port': 1000,
    'user': 'user',
    'passwd': 'pwd',
    'db': 'db',
    'charset': 'utf8',
    'cursorclass': pymysql.cursors.DictCursor  # 在连接时指定DictCursor更简洁
}

# --- 2. SQL 执行函数 ---
def fetchAll(sql, conn):
    """
    执行SQL查询并返回所有结果。
    注意：已移除 conn.commit()，因为它对于 SELECT 查询不是必需的。
    """
    cursor = conn.cursor()  # Cursorclass 已在连接时指定
    try:
        cursor.execute(sql)
        r = cursor.fetchall()
        return r
    finally:
        cursor.close()


def execute_sql_with_validation(sql_query):
    """
    封装的SQL执行函数，包含结果验证逻辑。
    
    Args:
        sql_query: 要执行的SQL语句
        connection: 数据库连接对象
    
    Returns:
        tuple: (result, is_error, error_msg)
            - result: 查询结果或错误信息
            - is_error: 是否发生错误 (True/False)
            - error_msg: 错误消息（如果有）
    """
    # 检查SQL是否为空
    connection = pymysql.connect(**connection_config)
    if pd.isna(sql_query) or not sql_query.strip():
        return "SQL query is empty", True, "SQL query is empty"
    
    try:
        # 执行查询
        data = fetchAll(sql_query, connection)
        
        # 将结果转换为字符串并检查是否以"["开头
        result_str = str(data)
        
        if not result_str.startswith('['):
            # 结果不以"["开头，返回False
            error_msg = f"Result does not start with '[': {result_str[:100]}"
            return data, True, error_msg
        
        # 执行成功且结果格式正确
        return data, False, None
    
    # 捕获特定的 pymysql 错误 (如语法错误)
    except (pymysql.err.ProgrammingError, pymysql.err.OperationalError) as db_e:
        error_msg = str(db_e.args[1]) if len(db_e.args) > 1 else str(db_e)
        return error_msg, True, error_msg
    
    # 捕获所有其他常规错误
    except Exception as e:
        error_msg = str(e)
        return error_msg, True, error_msg
