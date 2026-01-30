import re
from typing import List
import numpy as np
import pandas as pd
import pymysql
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Dict, Optional, Any
from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()
import re
from typing import List

# 新增的导入
import numpy as np
try:
    from openai import OpenAI
except ImportError:
    print("Warning: 'openai' package not installed. Embedding-based rewards will fail.")
    print("Please install it using 'pip install openai'")
    OpenAI = None


class StillFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        # 修改后的正则表达式：不再要求字符串必须以 </answer> 结尾
        pattern = r'<think>.*?</think>\s*<answer>.*?</answer>'
        matches = [re.search(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]



_PARSE_ERROR_TAG = "ANSWER_PARSE_ERROR"

def _parse_answer_content(content: str) -> str:
    """
    一个统一的解析器，只提取 <answer>...</answer> 标签内的内容。
    """
    # 检查标签数量是否正确
    if content.count('<answer>') != 1 or content.count('</answer>') != 1:
        return _PARSE_ERROR_TAG
    
    # 使用正则表达式匹配 <answer> 标签内容（启用DOTALL模式匹配换行符）
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    
    if not answer_match:
        return _PARSE_ERROR_TAG
    
    return answer_match.group(1).strip()

class SqlGrammarCheak(ORM):
    """
    奖励函数：只检查 <answer> 内容的 SQL 语法。
    1. 如果 answer 是 SQL，则 EXPLAIN 检查语法。
    2. 如果 answer 是 NL，且 solution 也是 NL，奖励 1.0。
    3. 如果 answer 是 NL，但 solution 是 SQL，奖励 0.0。
    """
    def __init__(self):
        self.pymysql = __import__('pymysql')
        self.re = __import__('re')
        self.db_config = {'host': "dip",
             'port': 1000,
             'user': 'user',
             'passwd': 'pwd',
             'db': 'db',
             'charset': 'utf8'}
        
        self.logging = __import__('logging')
        self.logger = self.logging.getLogger('SqlGrammarCheak')
        self.logger.setLevel(self.logging.ERROR)
        # ... (日志配置) ...
        self.conn = None # 不在 init 时连接

    def _get_connection(self):
        """获取或重连数据库"""
        if self.conn is None or not hasattr(self.conn, 'open') or not self.conn.open:
            try:
                self.conn = self.pymysql.connect(**self.db_config)
            except self.pymysql.Error as e:
                self.logger.error(f"数据库连接失败: {str(e)}")
                self.conn = None
        return self.conn

    def validate_sql_syntax(self, sql):
        """使用 EXPLAIN 验证 SQL 语法"""
        conn = self._get_connection()
        if conn is None:
            return False # 无法连接数据库，语法检查失败

        try:
            with conn.cursor() as cursor:
                cursor.execute("EXPLAIN " + sql)
            return True
        except self.pymysql.Error as e:
            # self.logger.error(f"SQL验证错误: {str(e)}, SQL: {sql}")
            if e.args[0] in (2006, 2013, 2003):  # 连接错误
                self.conn = None # 标记连接已断开
            return False
        
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        
        # --- [修改点 1: 获取 type 列表] ---
        types_list = kwargs.get('type', [None] * len(completions))
        if len(types_list) != len(completions):
            types_list = [None] * len(completions)

        # --- [修改点 2: 修改循环以包含 item_type] ---
        for content, sol, item_type in zip(completions, solution, types_list):
            reward = 0.0
            answer = _parse_answer_content(content) # 假设 _parse_answer_content 在外部定义

            if answer == _PARSE_ERROR_TAG: # 假设 _PARSE_ERROR_TAG 在外部定义
                rewards.append(0.0)
                continue

            # --- [修改点 3: 插入 "维度退化" 的特定逻辑] ---
            if item_type == '维度退化':
                is_valid_syntax = self.validate_sql_syntax(answer)
                
                # 需求：语法验证失败 (False) 则奖励 1，成功 (True) 则奖励 0
                reward = 1.0 if not is_valid_syntax else 0.0
                
                rewards.append(reward)
                continue # 处理完毕，跳过后续的通用逻辑
            # --- [结束修改点 3] ---

            # [以下为原始逻辑，保持不变]
            
            # 启发式判断 answer 和 solution 是 SQL 还是 NL
            is_answer_sql = 'SELECT' in answer.upper()
            is_solution_sql = 'SELECT' in sol.upper() and '不支持查询' not in sol

            if is_answer_sql:
                # 场景1：模型输出了 SQL。无论标准答案是什么，我们都检查其语法。
                reward = float(self.validate_sql_syntax(answer))
            elif not is_answer_sql and not is_solution_sql:
                # 场景2：模型输出 NL，标准答案也是 NL。
                # 它正确地识别了这是一个 NL 场景，语法上“正确”。
                reward = 1.0
            elif not is_answer_sql and is_solution_sql:
                # 场景3：模型输出 NL，但标准答案是 SQL。
                # 它没有输出 SQL，语法检查失败。
                reward = 0.0
            
            rewards.append(reward)
            
        # 在所有处理完成后关闭连接
        try:
            if self.conn and hasattr(self.conn, 'close'):
                self.conn.close()
                self.conn = None
        except Exception as e:
            self.logger.error(f"关闭数据库连接错误: {str(e)}")
        
        return rewards


class SoftOverlong(ORM):

    def __init__(self, tokenizer, soft_max_length, soft_cache_length):
        self.tokenizer = tokenizer
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


class SqlAccuracy(ORM):
    
    def __init__(self):
        self.requests = __import__('requests')
        self.json = __import__('json')
        self.re = __import__('re')
        
        # --- 新增：用于余弦相似度的模块 ---
        self.np = __import__('numpy')
        import difflib
        import sqlglot
        from sqlglot import exp
        
        self.sqlglot = sqlglot
        self.exp = exp
        self.SequenceMatcher = difflib.SequenceMatcher

        # --- 新增：VLLM 嵌入模型客户端初始化 ---
        if OpenAI:
            self.VLLM_BASE_URL = "https://..."
            self.VLLM_API_KEY = "dummy"
            self.EMBEDDING_MODEL = "Qwen__Qwen3-Embedding-0.6B-main"
            try:
                self.client = OpenAI(base_url=self.VLLM_BASE_URL, api_key=self.VLLM_API_KEY)
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                self.client = None
        else:
            self.client = None


    def safe_parse(self,sql):
        """尝试解析SQL，解析失败返回None"""
        try:
            return self.sqlglot.parse_one(sql)
        except:
            return None

    def extract_elements(self, parsed_sql):
        """从AST中提取关键要素"""
        if not parsed_sql:
            return set(), set(), set(), set()
        
        # 修复点 3: 使用 self.exp 访问表达式类型
        
        # 1. 表名 (Tables)
        tables = set(t.name.lower() for t in parsed_sql.find_all(self.exp.Table))
        
        # 2. 列名 (Columns) - 忽略表别名，只看字段名
        columns = set(c.name.lower() for c in parsed_sql.find_all(self.exp.Column))
        
        # 3. 关键词 (Keywords)
        keywords = set()
        if parsed_sql.find(self.exp.Group): keywords.add('GROUP BY')
        if parsed_sql.find(self.exp.Order): keywords.add('ORDER BY')
        if parsed_sql.find(self.exp.Limit): keywords.add('LIMIT')
        if parsed_sql.find(self.exp.Where): keywords.add('WHERE')
        if parsed_sql.find(self.exp.Join): keywords.add('JOIN')
        if parsed_sql.find(self.exp.Distinct): keywords.add('DISTINCT')
        if parsed_sql.find(self.exp.CTE): keywords.add('CTE')

        # 4. 字面量 (Literals)
        literals = set()
        for node in parsed_sql.find_all(self.exp.Literal):
            literals.add(node.this.replace("'", "").replace('"', '')) 
        
        return tables, columns, keywords, literals

    def calculate_set_score(self,pred_set, gold_set, mode='recall'):
        """计算集合相似度"""
        if not gold_set: return 1.0 # Gold没有此类要素，Model也没有则满分
        if not pred_set: return 0.0
        
        intersection = len(pred_set & gold_set)
        
        if mode == 'recall':
            # 召回率：Gold有的，Model写出来多少？
            return intersection / len(gold_set)
        elif mode == 'jaccard':
            # Jaccard：交集/并集，惩罚写多了错的情况
            return intersection / len(pred_set | gold_set)

    def calculate_table_score_with_fuzzy(self, pred_tables, gold_tables):
        """表名评分（增加模糊匹配）"""
        if not gold_tables: return 1.0
        if not pred_tables: return 0.0
        
        matches = 0
        for g_table in gold_tables:
            if g_table in pred_tables:
                matches += 1
                continue
            
            best_ratio = 0
            for p_table in pred_tables:
                # 修复点 4: 使用 self.SequenceMatcher
                ratio = self.SequenceMatcher(None, g_table, p_table).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
            
            if best_ratio > 0.85:
                matches += 0.8
                
        return matches / len(gold_tables)

    def compute_rule_based_reward(self,pred_sql, gold_sql):
        """
        主评分函数
        """
        # 0. 解析
        pred_ast = self.safe_parse(pred_sql)
        gold_ast = self.safe_parse(gold_sql)
        
        # 如果模型生成的SQL连语法都不对，直接给极低分
        if not pred_ast:
            return 0.0
        
        # 1. 提取要素
        p_tables, p_cols, p_kws, p_lits = self.extract_elements(pred_ast)
        g_tables, g_cols, g_kws, g_lits = self.extract_elements(gold_ast)
        
        # 2. 分项计算
        
        # A. 表名分数 (30%) - 使用模糊匹配容忍 _hf
        score_table = self.calculate_table_score_with_fuzzy(p_tables, g_tables)
        
        # B. 列名分数 (30%) - 使用 Recall，只要把Gold里的核心列都写出来就行，多写不扣分
        score_col = self.calculate_set_score(p_cols, g_cols, mode='recall')
        
        # C. 关键词分数 (20%) - 使用 Jaccard，惩罚结构性差异
        score_kw = self.calculate_set_score(p_kws, g_kws, mode='jaccard')
        
        # D. 字面量分数 (20%) - 检查 '2024', '北京' 是否存在
        score_lit = self.calculate_set_score(p_lits, g_lits, mode='recall')
        
        # 3. 加权总分
        total_score = (
            0.3 * score_table +
            0.3 * score_col +
            0.2 * score_kw +
            0.2 * score_lit
        )
        
        return round(total_score, 4)


    def parse_think_answer(self, content):
        """
        解析包含<think>和<answer>标签的内容。
        返回: <answer> 标签内的字符串，或 "格式错误"
        """
        # (此函数保持不变)
        if content.count('<think>') != 1 or content.count('<answer>') != 1 or content.count('</think>') != 1 or content.count('</answer>') != 1:
            return "格式错误"
        
        think_match = self.re.search(r'<think>(.*?)</think>', content, self.re.DOTALL)
        answer_match = self.re.search(r'<answer>(.*?)</answer>', content, self.re.DOTALL)
        
        if not think_match or not answer_match:
            return "格式错误"
        
        return answer_match.group(1).strip()

    
    def compare_sql(self, gold_sql, pred_sql):
        """
        [阶段一] SQL执行验证API。
        (此函数保持不变)
        """
        headers = {
            'Content-Type': 'application/json',
        }
        data = {"caseSql": gold_sql, "inferSql": pred_sql}
        try:
            response = self.requests.post('http://...', headers=headers, data=self.json.dumps(data))
            if response.json()['status'] == 0:
                return response.json()['result']['match'] # True or False
            else:
                return response.json()['status'] # 错误状态
        except:
            return "调用错误"


    def score_total(self, gold_sql: str, pred_sql: str, dialect: str = "mysql", base_url: str = "http://..."):
        """
        [阶段二] SQL等价性评估API (相似性)。
        (此函数保持不变)
        """
        payload = {"gold_sql": gold_sql, "pred_sql": pred_sql, "dialect": dialect}
        try:
            resp = self.requests.post(f"{base_url}/score", json=payload, timeout=2)
            resp.raise_for_status() 
            return resp.json()['total_score'] # 0.0 ~ 1.0
        except self.requests.RequestException as e:
            return 0.0
        except Exception as e:
            return 0.0

    # --- 新增：嵌入模型相关辅助函数 ---

    def _get_embeddings(self, batch_texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量，并按要求分块（100条/次）。
        """
        if not self.client:
            print("Embedding client not initialized. Returning empty embeddings.")
            return [[]] * len(batch_texts)
        
        all_embeddings = []
        batch_size = 100 # 按要求，单次请求控制在100以下
        
        for i in range(0, len(batch_texts), batch_size):
            batch = batch_texts[i:i + batch_size]
            try:
                batch = [str(t) for t in batch]
                response = self.client.embeddings.create(model=self.EMBEDDING_MODEL, input=batch)
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                print(f"Error getting embeddings for batch: {e}")
                # 如果批次失败，用空列表填充该批次
                all_embeddings.extend([[]] * len(batch))
                
        return all_embeddings

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        计算两个嵌入向量的余弦相似度。
        """
        if not v1 or not v2: # 处理获取嵌入失败的空列表
            return 0.0
        
        v1_np = self.np.array(v1)
        v2_np = self.np.array(v2)
        
        norm_v1 = self.np.linalg.norm(v1_np)
        norm_v2 = self.np.linalg.norm(v2_np)
        
        if norm_v1 == 0 or norm_v2 == 0: # 处理零向量
            return 0.0
            
        similarity = self.np.dot(v1_np, v2_np) / (norm_v1 * norm_v2)
        return similarity

    def get_db_connection(self):
            """<新增> 获取数据库连接"""
            return pymysql.connect(
                host='di',
                port=1000,
                user='user',
                passwd='pwd',
                db='db',
                charset='utf8'
            )

    def execute_sql_with_error(self, sql: str) -> Tuple[bool, str, Optional[List[Dict]]]:
        """
        <新增> 执行SQL并捕获错误信息
        返回: (是否成功, 错误信息或成功消息, 查询结果)
        """
        connection = None
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql)
            result = cursor.fetchall()
            cursor.close()
            connection.commit()
            # fetchall返回已经是Tuple[Dict]或List[Dict]，取决于pymysql版本，这里统一处理
            return True, "SQL executed successfully", list(result)
        except pymysql.err.ProgrammingError as e:
            error_msg = str(e.args[1]) if len(e.args) > 1 else str(e)
            return False, error_msg, None
        except Exception as e:
            # 简单处理traceback
            error_msg = str(e)
            return False, error_msg, None
        finally:
            if connection:
                connection.close()
    # =========================================================================
    # Section 4: <新增> 业务结果评分逻辑 (融合 DenseSQLRewardEvaluator)
    # =========================================================================

    def _compute_column_score(self, df_true, df_pred):
        """<新增> Schema 评分：衡量意图识别的准确性"""
        cols_true = df_true.columns.tolist()
        cols_pred = df_pred.columns.tolist()
        
        if not cols_true: return 0.0, {}
        if not cols_pred: return 0.0, {}

        # 1. 计算相似度矩阵 (复用 self._get_embeddings)
        emb_true = self._get_embeddings(cols_true)
        emb_pred = self._get_embeddings(cols_pred)
        
        # 确保embedding获取成功，否则无法计算
        if not emb_true or not emb_pred or len(emb_true[0]) == 0:
             return 0.0, {}

        sim_matrix = cosine_similarity(emb_true, emb_pred) 
        
        # 2. 匈牙利算法匹配
        cost_matrix = 1.0 - sim_matrix 
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 3. 提取匹配结果
        match_map = {}
        matched_sim_sum = 0.0
        
        for r, c in zip(row_ind, col_ind):
            sim = sim_matrix[r, c]
            # 阈值：只有相似度 > 0.75 才算匹配上
            if sim > 0.75:
                match_map[cols_pred[c]] = cols_true[r]
                matched_sim_sum += sim
        
        # A. 基础分：匹配的总质量 / 业务需求的列数
        base_score = matched_sim_sum / len(cols_true)
        
        # B. 多列惩罚
        if len(cols_pred) > len(cols_true):
            extra_ratio = (len(cols_pred) - len(cols_true)) / len(cols_true)
            penalty = 1.0 / (1.0 + 0.2 * extra_ratio) 
            base_score *= penalty
            
        return max(0.0, min(base_score, 1.0)), match_map

    def _compute_row_score(self, len_true, len_pred):
        """<新增> 行数评分：对数平滑"""
        if len_true == len_pred: return 1.0
        if len_pred == 0: return 0.0
        if len_true == 0: return 0.0 
        
        ratio = max(len_true, len_pred) / min(len_true, len_pred)
        return 1.0 / (1.0 + 0.3 * np.log(ratio))

    def _compute_value_score(self, df_true, df_pred, match_map):
        """<新增> 数值评分"""
        if not match_map: return 0.0
        
        total_score_sum = 0.0
        true_cols = df_true.columns.tolist()
        pred_cols_map = {v: k for k, v in match_map.items()} # True -> Pred
        
        for t_col in true_cols:
            p_col = pred_cols_map.get(t_col)
            
            if p_col is None:
                col_f1 = 0.0
            else:
                vals_true = df_true[t_col].tolist()
                vals_pred = df_pred[p_col].tolist()
                
                hits = 0
                target_set = [str(v).strip() for v in vals_true]
                
                for v_p in vals_pred:
                    v_p_str = str(v_p).strip()
                    matched = False
                    for i, v_t_str in enumerate(target_set):
                        if self._is_match(v_p_str, v_t_str):
                            matched = True
                            del target_set[i] 
                            break
                    if matched:
                        hits += 1
                
                precision = hits / len(vals_pred) if len(vals_pred) > 0 else 0
                recall = hits / len(vals_true) if len(vals_true) > 0 else 0
                
                if precision + recall > 0:
                    col_f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    col_f1 = 0.0
            
            total_score_sum += col_f1
            
        final_value_score = total_score_sum / len(true_cols)
        return final_value_score

    def _is_match(self, v1, v2):
        """<新增> 数值匹配辅助函数"""
        try:
            f1, f2 = float(v1), float(v2)
            if f2 == 0: return abs(f1 - f2) < 1e-6
            return abs(f1 - f2) / abs(f2) < 0.01
        except:
            return str(v1).strip() == str(v2).strip()

    def calculate_dense_score(self, result_true, result_pred):
        """<新增> 计算业务结果总分"""
        if not result_true and not result_pred: return 0.0 # 都是空，暂定为0或根据业务定义
        # 即使 result_true 为空（业务查不出数据），也需要 result_pred 也是空才对，这里由 row_score 处理
        
        df_true = pd.DataFrame(result_true) if result_true else pd.DataFrame()
        df_pred = pd.DataFrame(result_pred) if result_pred else pd.DataFrame()
        
        # 1. Schema (权重 0.4)
        s_schema, match_map = self._compute_column_score(df_true, df_pred)
        
        # 2. Row (权重 0.1)
        s_row = self._compute_row_score(len(df_true), len(df_pred))
        
        # 3. Value (权重 0.5)
        s_value = self._compute_value_score(df_true, df_pred, match_map)
        
        final_score = 0.4 * s_schema + 0.1 * s_row + 0.5 * s_value
        return round(final_score, 4)
    # --- 重构后的 __call__ 方法 ---

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = [0.0] * len(completions)
        
        # 1. 获取 type 列表
        types_list = kwargs.get('type', [None] * len(completions))
        if len(types_list) != len(completions):
            types_list = [None] * len(completions)

        # 2. 准备用于批量嵌入的列表
        # (存储需要计算相似度的 [预测, 答案] 对)
        embedding_texts_batch = []
        # (存储这些对在原始 rewards 列表中的索引)
        embedding_pairs_indices = []

        # 3. 第一次循环：处理非嵌入任务，并收集嵌入任务
        for index, (content, sol, item_type) in enumerate(zip(completions, solution, types_list)):
            
            # 3.1 解析答案
            answer = self.parse_think_answer(content)
            sol_clean = sol.strip()
            
            if answer == '格式错误':
                rewards[index] = 0.0
                continue
            
            pred_answer = answer.strip()
            gold_answer = sol_clean
            
            # 3.2 根据 type 分派逻辑
            if item_type in ('SQL', '反思数据'):
                # 逻辑：两阶段 SQL 评估
                sql_compare_res = self.compare_sql(gold_answer, pred_answer)
                if isinstance(sql_compare_res, bool) and sql_compare_res is True:
                    rewards[index] = 1.0
                else:
                    # 阶段一失败，调用阶段二并乘以 0.5
                    similarity_score = self.score_total(gold_answer, pred_answer)
                    rewards[index] = similarity_score * 0.5
            elif item_type == '多步推理':
                # --- <修改>: 50% 规则分 + 50% 业务执行分 ---
                
                # 1. 原有的 SQL AST 规则评分
                rule_score = self.compute_rule_based_reward(pred_answer, gold_answer)
                
                # 2. 新增的 业务结果执行评分
                exec_score = 0.0
                
                # 执行 Gold SQL
                g_success, g_msg, g_result = self.execute_sql_with_error(gold_answer)
                
                if not g_success:
                    # 如果标准答案都执行不了，说明数据环境或标注有问题
                    # 这里保守处理，如果Gold执行失败，只看 AST 分数，或者判0
                    # 策略：如果Gold错，则无法评估Pred的执行结果，Exec得分为0
                    print(f"[Warning] Gold SQL execution failed: {g_msg}")
                    exec_score = 0.0
                else:
                    # 执行 Pred SQL
                    p_success, p_msg, p_result = self.execute_sql_with_error(pred_answer)
                    
                    if not p_success:
                        # 预测SQL执行失败（语法错或表不存在），Exec得分为0
                        exec_score = 0.0
                    else:
                        # 两者都执行成功，计算结果相似度
                        exec_score = self.calculate_dense_score(g_result, p_result)
                
                # 3. 融合评分 (各占50%)
                rewards[index] = 0.5 * rule_score + 0.5 * exec_score
            
            elif item_type == '维度退化':
                # 逻辑：直接 阶段二 SQL 评估 * 0.5
                similarity_score = self.score_total(gold_answer, pred_answer)
                rewards[index] = similarity_score 
            
            elif item_type in ('指标拒识', '追问_必备约束'):
                # 逻辑：严格字符串匹配
                if pred_answer == gold_answer:
                    rewards[index] = 1.0
                else:
                    rewards[index] = 0.0
            
            elif item_type in ('歧义澄清', '维度拒识'):
                # 逻辑：余弦相似度（稍后批量处理）
                embedding_texts_batch.append(pred_answer)
                embedding_texts_batch.append(gold_answer)
                embedding_pairs_indices.append(index)
            
            else:
                # 未知 type，奖励为 0
                rewards[index] = 0.0

        # 4. 第二阶段：批量处理所有嵌入任务
        if embedding_texts_batch:
            all_embeddings = self._get_embeddings(embedding_texts_batch)
            
            if not all_embeddings or len(all_embeddings) != len(embedding_texts_batch):
                print(f"Embedding batch processing failed. Expected {len(embedding_texts_batch)} embeddings, got {len(all_embeddings)}.")
                # 失败的奖励保持 0.0
            else:
                pair_count = len(embedding_pairs_indices)
                for i in range(pair_count):
                    original_index = embedding_pairs_indices[i]
                    
                    # 从批量结果中取出对应的 [预测, 答案] 嵌入
                    emb_pred = all_embeddings[i * 2]
                    emb_gold = all_embeddings[i * 2 + 1]
                    
                    sim = self._cosine_similarity(emb_pred, emb_gold)
                    
                    # 应用新的相似度奖励逻辑
                    if sim >= 0.96:
                        reward = sim  # 直接赋值相似度
                    elif sim >= 0.9: # 0.9 <= sim < 0.96
                        reward = sim * 0.8
                    else: # sim < 0.9
                        reward = 0.0
                    
                    rewards[original_index] = reward
        
        return rewards

orms['still_format'] = StillFormat
orms['sql_acc'] = SqlAccuracy
orms['sql_grammar_cheak'] = SqlGrammarCheak
orms['soft_overlong'] = SoftOverlong
