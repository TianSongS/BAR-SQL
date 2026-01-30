import os
import json
import sqlparse
from collections import Counter
import requests
from typing import List, Tuple
from openai import OpenAI

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
            response = requests.request("POST", f'http://...gpt-4o', headers=headers, data=payload)
            response = json.loads(response.text)
            # print(response)
            return response['choices'][0]['message']['content']

        except Exception as e:
            pass


class SQLEquivalenceValidator:
    """
    基于论文 "Taming SQL Complexity: LLM-Based Equivalence Evaluation for Text-to-SQL"
    实现的SQL等价性评估工具。 (V2 - 强化版)
    
    V2版本根据论文中常见的不等价模式，在Prompt中加入了“关键检查点”，
    旨在提升LLM对细微但关键的逻辑错误的识别能力，例如字段归属错误。
    """
    
    def __init__(self):
        """
        初始化验证器。
        
        :param api_key: OpenAI API密钥。如果为None，则会尝试从环境变量 'OPENAI_API_KEY' 读取。
        :param model: 要使用的GPT模型名称。
        """
        api_key = 'ak-infer-'
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Please provide it as an argument or set the 'OPENAI_API_KEY' environment variable.")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://.../deepseek-v3/v1",
    )
        self.model = 'deepseek-v3-0324'

    def _preprocess_sql(self, sql: str) -> str:
        """
        对SQL查询进行标准化预处理。
        - 格式化（去除多余空格，统一关键字大小写等）
        - 移除注释
        - 正确处理以 'WITH' 或 'SELECT' 开头的查询，移除可能的前导文本
        
        :param sql: 原始SQL字符串。
        :return: 标准化后的SQL字符串。
        """
        # 使用sqlparse进行基础格式化
        formatted_sql = sqlparse.format(
            sql, 
            reindent=True, 
            keyword_case='upper',  
            identifier_case='lower',
            strip_comments=True
        ).strip()

        # 为了不区分大小写地查找关键字，转换为大写
        upper_sql = formatted_sql.upper()

        # 查找'SELECT'和'WITH'的起始位置
        select_pos = upper_sql.find('SELECT')
        with_pos = upper_sql.find('WITH')

        start_pos = -1

        # 根据找到的位置决定真正的SQL起始点
        if select_pos != -1 and with_pos != -1:
            # 如果两者都存在，SQL语句从更早出现的那一个开始
            start_pos = min(select_pos, with_pos)
        elif select_pos != -1:
            # 只找到了 SELECT
            start_pos = select_pos
        elif with_pos != -1:
            # 只找到了 WITH
            start_pos = with_pos
        
        # 如果找到了有效的起始关键字，则从该位置截取
        if start_pos != -1:
            formatted_sql = formatted_sql[start_pos:]
            
        # 返回清理首尾空白和末尾分号的最终结果
        return formatted_sql.strip().rstrip(';')

    def _construct_prompt(self, sql1: str, sql2: str, schema: str, nl_query: str) -> str:
        """
        构建用于LLM评估的 "Miniature & Mull" (微缩与推敲) Prompt。
        (V2版本: 增加了关键检查点)
        
        :param sql1: 第一个SQL查询。
        :param sql2: 第二个SQL查询。
        :param schema: 相关的数据库Schema。
        :param nl_query: 原始的自然语言问题。
        :return: 构建好的Prompt字符串。
        """
        prompt = f"""
        ### 任务定义
        你是一位资深的数据库专家和SQL审核员。你的任务是严格判断下面提供的两个SQL查询（Query 1 和 Query 2）在语义上是否等价。

        ### 等价性标准
        如果两个查询在任何可能的、符合schema的数据库状态下，都返回**完全相同**的结果集（包括相同的列、相同的行、以及相同的行顺序），那么它们就是语义等价的。请注意，别名的不同、格式的差异或注释不影响语义等价性。

        ### 业务场景上下文
        - **自然语言问题**: "{nl_query}"
        - **数据库Schema**:
        ```sql
        {schema}
        ```

        ### 待评估的SQL查询对
        **Query 1 (Q1):**
        ```sql
        {sql1}
        ```

        **Query 2 (Q2):**
        ```sql
        {sql2}
        ```

        ### 思考与分析方法 ("微缩与推敲"策略)
        为了得出最准确的结论，请严格遵循以下思考步骤：

        #### 1. 关键检查点 (Key Checkpoints)
        在进行推演前，请首先针对以下常见错误模式，对两个SQL进行系统性地审查。这将帮助你构思更有针对性的测试数据。
        - **`WHERE`子句审查**: 仔细核对`WHERE`子句中的每个条件，特别是过滤条件的**字段归属**（例如，`t1.day_id` vs `t2.day_id`）和比较操作符（例如 `>` vs `>=`）是否正确反映了业务逻辑。
        - **`JOIN`条件审查**: 检查`JOIN`的类型（`INNER`, `LEFT`等）和关联字段（`ON`条件）是否准确无误。
        - **聚合与去重审查**: 确认聚合函数（`SUM`, `COUNT`, `AVG`等）、`GROUP BY`的字段以及`DISTINCT`的使用是否一致且符合查询意图。
        - **子查询审查**: 如果存在子查询，需要独立分析其逻辑是否正确（例如，关联条件、过滤），以及它与主查询的交互是否符合预期（例如，标量子查询是否可能返回多行）。
        - **逻辑操作符审查**: 留意`AND`和`OR`的组合逻辑，确保其优先级和组合方式正确无误。
        - **排序审查**: 检查`ORDER BY`子句的排序字段和排序方向（`ASC` vs `DESC`）是否一致。如果查询的业务意图要求特定排序，这一点至关重要。
        - **函数使用审查**: 检查两个查询中所使用的函数（例如日期函数、字符串函数等）是否功能相同，参数是否一致。不同数据库方言中的同名函数可能存在细微差异。
        
        #### 2. 构思微型数据库
        基于上述检查点的发现，在你的脑海中构思一个或多个包含少量数据（3-5行即可）的微型数据库实例。**你的实例应特意设计来测试你在检查点发现的潜在差异**。

        #### 3. 首次推演
        在你构思的第一个数据库实例上， mentally "执行" Q1 和 Q2，并对比它们的输出结果。详细记录这个过程。

        #### 4. 推敲与寻找反例
        - 如果首次推演结果相同，请不要立即下结论。你需要“推敲”一下，调整你的数据库实例来测试其他边界情况（例如：加入NULL值、重复值等）。
        - 你的目标是尝试找到一个**反例**——即一个能让Q1和Q2产生不同结果的数据库实例。

        #### 5. 得出结论
        - 如果你经过多次不同实例的推敲，始终无法找到任何反例，那么你可以得出结论：“等价”。
        - 如果你成功找到了一个反例，那么结论就是：“不等价”，并清晰地说明这个反例是什么，以及为什么它导致了不同的结果。

        ### 输出格式
        请将你的完整分析过程和最终结论以一个JSON对象的形式返回，不要包含任何额外的解释性文字。JSON格式如下：
        {{
          "reasoning": "这里是你的详细思考过程，包括你对关键检查点的分析、你构思的微型数据库实例、在每个实例上的推演步骤、以及最终如何得出结论的逻辑。",
          "is_equivalent": true  // 或者 false
        }}
        """
        return prompt

    def _call_llm(self, prompt: str) -> dict:
        try:
            content = chat_gpt4o(prompt)
            if content.startswith('```json'):
                content = content[7:]  # 移除 ```json
            elif content.startswith('```'):
                content = content[3:]  # 移除 ```
            
            if content.endswith('```'):
                content = content[:-3]  # 移除结尾的 ```
            
            # 去除前后空白字符
            content = content.strip()
            return json.loads(content)
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return {"reasoning": f"API call failed with error: {e}", "is_equivalent": None}

    def is_equivalent(self, sql1: str, sql2: str, schema: str, nl_query: str, runs: int = 3) -> Tuple[bool, List[str]]:
        print("--- Stage 1: Preprocessing SQL queries ---")
        preprocessed_sql1 = self._preprocess_sql(sql1)
        preprocessed_sql2 = self._preprocess_sql(sql2)
        
        print("Preprocessed SQL 1:\n", preprocessed_sql1)
        print("\nPreprocessed SQL 2:\n", preprocessed_sql2)

        print("\n--- Stage 2: Exact Match Check ---")
        if preprocessed_sql1 == preprocessed_sql2:
            print("Result: Equivalent (Exact match after preprocessing)")
            return True, ["Exact match after preprocessing."]

        print("Result: Not an exact match. Proceeding to LLM evaluation.")

        print(f"\n--- Stage 3 & 4: LLM Evaluation (runs={runs}) ---")
        prompt = self._construct_prompt(preprocessed_sql1, preprocessed_sql2, schema, nl_query)
        
        judgements = []
        reasonings = []
        
        for i in range(runs):
            print(f"Running LLM evaluation: run {i+1}/{runs}...")
            result = self._call_llm(prompt)
            
            if result.get("is_equivalent") is not None:
                judgements.append(result["is_equivalent"])
                reasonings.append(f"Run {i+1} Reasoning:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
            else:
                reasonings.append(f"Run {i+1} Failed:\n{result.get('reasoning', 'Unknown error.')}")

        if not judgements:
            print("LLM evaluation failed for all runs.")
            return None, reasonings
            
        print("\n--- Final Decision (Majority Vote) ---")
        vote_count = Counter(judgements)
        winner, count = vote_count.most_common(1)[0]

        if count > len(judgements) / 2:
            print(f"Stable result found. Majority vote is: {winner} ({count}/{len(judgements)})")
            return winner, reasonings
        else:
            print(f"Unstable result. Votes: {dict(vote_count)}. No clear majority.")
            return None, reasonings