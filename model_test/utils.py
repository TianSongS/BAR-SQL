import pymysql
import traceback
from typing import Tuple, List, Dict, Optional, Any
from collections import Counter  # 导入 Counter
import requests
import uuid
import traceback
import datetime
from decimal import Decimal
from collections import Counter
from openai import OpenAI
import json
import math

def is_number_approx_equal(v1: Any, v2: Any, rel_tol=1e-3, abs_tol=1e-6) -> bool:
    """
    判断两个值是否近似相等。
    支持 int, float, Decimal 以及可以转为数字的字符串。
    """
    # 1. 如果直接相等（包含 None==None, 字符串完全相等, 整数相等），直接返回 True
    if v1 == v2:
        return True

    # 2. 尝试转换为浮点数进行比较
    try:
        f1 = float(v1)
        f2 = float(v2)
        # math.isclose 处理浮点数精度问题
        # rel_tol: 相对误差 (默认 1e-5, 即 0.001%)
        # abs_tol: 绝对误差 (处理接近 0 的数值)
        return math.isclose(f1, f2, rel_tol=rel_tol, abs_tol=abs_tol)
    except (ValueError, TypeError):
        # 无法转换为数字，说明是真正的字符串差异或其他类型差异
        return False

def row_approx_match(row1: Dict, row2: Dict) -> bool:
    """判断两行数据是否近似相等 (忽略列顺序，允许数值误差)"""
    if len(row1) != len(row2):
        return False
    # 严格校验列名必须一致（如果允许列名不同但含义相同，这一步交给LLM）
    if set(row1.keys()) != set(row2.keys()):
        return False
        
    for key in row1:
        if not is_number_approx_equal(row1[key], row2[key]):
            return False
    return True

def check_fuzzy_equivalence(result1: List[Dict], result2: List[Dict]) -> bool:
    """
    在无序且允许数值误差的情况下，比较两个结果集是否一致。
    复杂度 O(N^2)，适用于结果集较小的情况。
    """
    # 浅拷贝一份 result2 的索引，用于标记匹配状态，避免修改原数据
    unmatched_indices = set(range(len(result2)))
    
    for row1 in result1:
        match_found = False
        # 在剩余未匹配的行中寻找
        for idx in list(unmatched_indices):
            row2 = result2[idx]
            if row_approx_match(row1, row2):
                unmatched_indices.remove(idx)
                match_found = True
                break
        
        # 如果 row1 在 result2 中找不到对应的近似行，则模糊匹配失败
        if not match_found:
            return False
            
    return True

def call_deepseek_reasoner(
        prompt,
        system_prompt="You are a helpful assistant.",
        model="deepseek-reasoner",
        stream=False,
        temperature=0.7
    ):
    """
    调用 DeepSeek Reasoner v3.2 的聊天接口（基于你提供的 curl 示例封装）

    参数:
        prompt (str): 用户输入的提示词
        system_prompt (str): 系统提示词
        model (str): 模型名称，默认 deepseek-reasoner
        stream (bool): 是否使用流式响应
        temperature (float): 生成温度
        
    返回:
        str: AI 的回复内容
    """

    # DeepSeek API URL（根据你给的 curl 完整复制）
    url = "https://api..../chat/completions"

    # 你给出的 Token（可换成环境变量）
    token = "sk-"

    headers = {
        "Token": token
    }

    # 构建消息体
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=150)
        response.raise_for_status()
        result = response.json()

        # 提取输出内容
        if "choices" in result and len(result["choices"]) > 0:
            try:
                return result["choices"][0]["message"]["content"]
            except:
                return str(result["choices"][0]["message"])
        else:
            print("❌ 响应格式异常")
            return None

    except requests.exceptions.RequestException as e:
        print("❌ 请求失败：", e)
        return None
    except json.JSONDecodeError:
        print("❌ JSON 解析失败，原始响应：", response.text)
        return None
    


def call_deepseekv32(prompt, 
                    system_prompt="You are a helpful assistant.",
                    model="bailian-deepseek-v3_2", 
                    stream=False, 
                    temperature=0.6):
    """
    调用融合云4.0/3.0 的 GPT-5 聊天接口

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
    url = "http://..deepseek-v3_2"

    # 你的业务 Token
    token = "eyJ"

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
        "stream": stream,
        "max_tokens":18192,
        "temperature": temperature  # 添加温度参数
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=150)
        response.raise_for_status()
        result = response.json()
        
        # 提取AI的回复内容
        if result and "choices" in result and len(result["choices"]) > 0:
            try:
                ai_response = result["choices"][0]["message"]["content"]
            except:
                ai_response = result["choices"][0]["message"]
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



def call_gemini3(prompt, 
                    system_prompt="You are a helpful assistant.",
                    model="gemini-3-pro-preview", 
                    stream=False, 
                    temperature=0.7):
    """
    调用融合云4.0/3.0 的 GPT-5 聊天接口

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
    url = "http://"

    # 你的业务 Token
    token = "eyJ"

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
        "stream": stream,
        "max_tokens":18192,
        "temperature": temperature  # 添加温度参数
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=150)
        response.raise_for_status()
        result = response.json()
        
        # 提取AI的回复内容
        if result and "choices" in result and len(result["choices"]) > 0:
            try:
                ai_response = result["choices"][0]["message"]["content"]
            except:
                ai_response = result["choices"][0]["message"]
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


from openai import OpenAI

client_deep = OpenAI(
    # LPAI Token
    api_key="ak-infer",
    # api url，注意截止到/v1即可，后续的chat/completions client会自动拼上
    base_url="https://.../v1",
)

def get_deepseekV3(prompt):
    chat_completion = client_deep.chat.completions.create(
        messages=[
            {
                # role: user为固定值
                "role": "user",
                # content中填写自定义的内容
                "content": prompt,
            }
        ],
        model="deepseek-ai__deepseek-v3",
        temperature=0.6,
        stream=False,
    )

    return chat_completion.choices[0].message.content


client_claude = OpenAI(
    api_key="EMPTY",
    base_url="http://.../v1",
    default_headers={
        "BCS-APIHub-RequestId": str(uuid.uuid4()),
        "X-CHJ-GWToken": "eyJ",
    },
    timeout=150,
    max_retries=2,
)

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
    调用融合云4.0/3.0 的 GPT-5 聊天接口

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
    token = "eyJ"

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
              "model": "chatgpt-4o",
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
            # print(response)
            return response['choices'][0]['message']['content']

        except Exception as e:
            pass


def get_db_connection():
    """获取数据库连接"""
    return pymysql.connect(
        host='dip',
        port=1000,
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
        # 将结果转换为 List[Dict[str, Any]]，确保类型一致
        # fetchall() 已经返回 List[Dict]
        return True, "SQL executed successfully", result
    except pymysql.err.ProgrammingError as e:
        error_msg = str(e.args[1]) if len(e.args) > 1 else str(e)
        return False, error_msg, None
    except Exception as e:
        tb_lines = traceback.format_exc().strip().split('\n')
        error_msg = tb_lines[-1] if tb_lines else str(e)
        return False, error_msg, None
    finally:
        if connection:
            connection.close()

def safe_serializer(obj):
    """JSON序列化辅助函数，处理数据库特有的类型"""
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj) # 或者 str(obj) 以保持绝对精度
    raise TypeError(f"Type {type(obj)} not serializable")

def format_result_for_llm(result: List[Dict], max_rows: int = 10) -> str:
    """格式化SQL结果供LLM阅读，防止token爆炸"""
    if not result:
        return "[] (Empty Result)"
    
    # 截取前N行
    preview = result[:max_rows]
    
    # 转换为字符串，处理特殊类型
    try:
        json_str = json.dumps(preview, default=safe_serializer, ensure_ascii=False, indent=2)
    except Exception:
        json_str = str(preview) # 降级处理

    info = f"Total Rows: {len(result)}\nData Preview (Top {max_rows}):\n{json_str}"
    if len(result) > max_rows:
        info += "\n... (more rows omitted)"
    return info

# ==========================================
# 2. LLM 调用与核心逻辑
# ==========================================

def extract_json_from_llm_response(content: str) -> Dict:
    """从LLM返回中提取JSON，处理Markdown代码块"""
    try:
        # 尝试直接解析
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试去除 ```json ... ``` 包裹
    cleaned = content.strip()
    if "```" in cleaned:
        # 提取第一个代码块
        parts = cleaned.split("```")
        for part in parts:
            if part.strip().startswith("json"):
                cleaned = part.strip()[4:] # 去掉 json
                break
            elif part.strip().startswith("{"):
                cleaned = part.strip()
                break
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 兜底：如果解析失败，构造一个默认失败结果
        print(f"LLM output JSON parse failed. Raw content: {content}")
        return {"is_equivalent": False, "reason": "LLM response format error"}

def _evaluate_with_llm(
    context: Dict, 
    result1: List[Dict], 
    result2: List[Dict]
) -> Tuple[bool, str]:
    """构造Prompt并调用LLM进行语义比对"""
    
    question = context.get("question", "未知用户问题")
    schema_info = context.get("schema", "无Schema信息")
    knowledge_graph = context['knowledge_graph']
    domain_rules = context['domain_rules']
    data_kg = context['data_kg']

    # 格式化数据
    res1_str = format_result_for_llm(result1)
    res2_str = format_result_for_llm(result2)

    prompt = f"""
你是一位资深的业务数据验收专家。你的任务是判断【待测结果 (Result B)】是否正确回答了用户的业务问题。
请使用【基准结果 (Result A)】作为“标准答案”来进行校验。

### 1. 上下文信息
- **用户问题**: 
{question}

- **数据库 Schema**: 
{schema_info}

- **业务逻辑与知识**: 
{knowledge_graph}
{domain_rules}
{data_kg}

### 2. 待比对数据
**结果集 A (基准/标准答案)**:
{res1_str}

**结果集 B (待测/模型生成)**:
{res2_str}

### 3. 判定标准 (Evaluation Criteria)
你的核心判断逻辑是：**结果集 B 是否能够满足用户的问题需求？**
请遵循以下【宽松匹配原则】：

#### ✅ 允许的差异 (视为正确)：
1. **多余的列**：如果 B 包含了 A 所有的核心数据列，但额外多出了其他列（如 ID、辅助字段等），视为正确。
2. **列名不同**：只要 B 中列的数据值和业务含义与 A 对应（例如 A叫`leads_cnt`，B叫`num`），视为正确。
3. **数据格式差异**：
   - 数字精度：`0.33` vs `33.0%` vs `0.3333` 视为一致。
   - 文本格式：`2024-01` vs `202401` 视为一致。
   - 类型差异：字符串 `"100"` 与 数字 `100` 视为一致。
4. **行序差异**：除非问题明确要求“排名”、“前N名”，否则行顺序不同视为正确。

#### ❌ 拒绝的情况 (视为错误)：
1. **核心数值错误**：B 中的关键指标数值与 A 偏差较大（超过容差范围）。
2. **缺失核心列**：用户问“销量和占比”，B 中只返回了“销量”，缺少“占比”。
3. **行数/粒度严重不符**：A 有 5 行（按大区分组），B 只有 1 行（总计）或 100 行（按城市分组），且未回答原问题粒度。
4. **数据为空**：A 有数据，但 B 为空。

### 4. 输出要求
请仅输出标准的 JSON 格式，不要包含Markdown标记（如 ```json）：
{{
    "reason": "请简要说明理由。如果判定为 True，说明 B 是如何满足需求的；如果为 False，指出 B 缺失了什么或错在哪里。",
    "is_equivalent": true 或 false
}}
"""
    
    try:
        # 调用用户提供的 chat_gpt4o 函数
        response_content = chat_gpt4o(prompt, temperature=0.1) # 调低温度以获得稳定输出
        
        # 解析结果
        result_json = extract_json_from_llm_response(response_content)
        
        return result_json.get("is_equivalent", False), f"[LLM Judge]: {result_json.get('reason', 'No reason provided')}"
        
    except Exception as e:
        return False, f"[LLM Judge Error]: {str(e)}"


def verify_sql_equivalence(
    sql1: str, 
    sql2: str, 
    use_llm: bool = False, 
    context: Dict = None
) -> Tuple[bool, str]:
    """
    验证两个SQL查询的结果是否一致。
    
    参数:
    - sql1: 基准SQL
    - sql2: 待验证SQL
    - use_llm: 是否在严格规则失败后启用LLM语义判断 (默认False)
    - context: 上下文信息，包含 {'question': '...', 'schema': '...'}
    """

    if sql1 == sql2:
        return True, "Two SQLs are char Match"

    if context is None:
        context = {}

    # 1. 执行第一个SQL
    success1, msg1, result1 = execute_sql_with_error(sql1)
    if not success1:
        return False, f"SQL1 execution failed: {msg1}"

    # 2. 执行第二个SQL
    success2, msg2, result2 = execute_sql_with_error(sql2)
    if not success2:
        return False, f"SQL2 execution failed: {msg2}"

    if result1 is None or result2 is None:
        return False, "One of the results is None."

# =========================================
    # Stage 1: 严格规则检查 (Strict Hash Match)
    # =========================================
    
    # 预检查：如果都是空集，直接一致
    if len(result1) == 0 and len(result2) == 0:
        return True, "Both results are empty."

    # 尝试 Counter 严格匹配 (速度最快)
    try:
        hashable_rows1 = [tuple(sorted((k, str(v)) for k, v in row.items())) for row in result1]
        hashable_rows2 = [tuple(sorted((k, str(v)) for k, v in row.items())) for row in result2]
        
        if Counter(hashable_rows1) == Counter(hashable_rows2):
            return True, "Results are strictly equivalent (SQL RUN Match)."
    except Exception:
        pass # 转换失败则忽略，继续往下走

    # =========================================
    # Stage 2: 数值模糊匹配 (Fuzzy Numeric Match)
    # =========================================
    # 如果严格匹配失败，且行数一致，很可能是浮点数精度问题
    # 这里增加判定，避免精度问题直接导致校验失败

    if len(result1) == len(result2):
        if check_fuzzy_equivalence(result1, result2):
            return True, "Results are equivalent (Fuzzy Numeric Match)."
        else:
            pass

    # =========================================
    # Stage 3: LLM 语义判断 (Fallback)
    # =========================================
    # 只有当 use_llm 为 True 时才进入 LLM 流程
    # 否则，因为 Stage 1 和 Stage 2 都没过，直接返回 False
        
    if use_llm and len(result1)<31 and len(result2)<31:
        return _evaluate_with_llm(context, result1, result2)
    else:
        if len(result1) != len(result2):
            diff1 = Counter(hashable_rows1) - Counter(hashable_rows2)
            return False, f"Result content mismatch (Strict Rule). Unique to SQL1: {list(dict(diff1).keys())[:1]}"
        return False, "Values mismatch strictly and fuzzy match failed."
