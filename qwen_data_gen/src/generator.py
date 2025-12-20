import json
import random
import sys
import uuid
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config

class DataGenerator:
    def __init__(self, retriever=None):
        self.client = genai.Client(api_key=Config.API_KEY)
        self.retriever = retriever
        
        # Persona Rotation: 多样化人设
        self.personas_cn = [
            "资深 DDD 架构师，擅长领域驱动设计和解耦。",
            "Python 性能优化专家，关注代码效率和算法复杂度。",
            "高级后端工程师，注重代码的可读性、健壮性和异常处理。",
            "安全审计专家，关注潜在的代码漏洞和数据安全。"
        ]
        self.personas_en = [
            "A senior DDD architect, skilled in domain-driven design and decoupling.",
            "A Python performance optimization expert, focusing on code efficiency and algorithmic complexity.",
            "A senior backend engineer, emphasizing code readability, robustness, and exception handling.",
            "A security audit expert, focusing on potential code vulnerabilities and data security."
        ]

    def _get_random_persona(self, lang='cn'):
        if lang == 'en':
            return random.choice(self.personas_en)
        return random.choice(self.personas_cn)

    def _get_random_temp(self):
        return random.uniform(Config.MIN_TEMP, Config.MAX_TEMP)

    def _format_assistant_content(self, output_content):
        """将助手的输出（可能是字符串或字典）格式化为统一的字符串"""
        if isinstance(output_content, str):
            return output_content
        elif isinstance(output_content, dict):
            trace = output_content.get("reasoning_trace", "")
            if isinstance(trace, (dict, list)):
                trace = json.dumps(trace, ensure_ascii=False, indent=2)
            
            plan = output_content.get("architectural_modification_plan") or \
                   output_content.get("answer") or \
                   output_content.get("output") or \
                   {k: v for k, v in output_content.items() if k != "reasoning_trace"}
            
            if isinstance(plan, (dict, list)):
                plan = json.dumps(plan, ensure_ascii=False, indent=2)
                
            return f"推理过程 (Reasoning Trace):\n{trace}\n\n最终答案:\n{plan}"
        return str(output_content)

    def _extract_clean_instruction(self, raw_data):
        """清洗 LLM 返回的 instruction，防止包含 Prompt 模板"""
        instruction = raw_data.get("instruction", "")
        output = raw_data.get("output")

        # 如果 instruction 看起来像 Prompt 模板，尝试从 output 恢复
        if "你现在扮演" in instruction or "任务：" in instruction:
            if isinstance(output, dict):
                for key in ["requirement", "new_requirement", "question"]:
                    if key in output and output[key]:
                        return output[key]
        
        return instruction

    def _build_context_str(self, chunk: dict) -> str:
        if not self.retriever or not Config.RAG_ENABLED:
            return ""
        
        related_chunks = self.retriever.retrieve(chunk, k=Config.RAG_TOP_K)
        if not related_chunks:
            return ""
            
        context_parts = ["\n【相关项目代码上下文 (RAG)】:"]
        for i, rc in enumerate(related_chunks, 1):
            context_parts.append(f"片段 {i} (来自 {rc['file_name']}):\n```python\n{rc['code']}\n```")
        return "\n".join(context_parts)

    def _build_qa_prompt(self, chunk: dict, persona: str, context_str: str, lang: str) -> str:
        if lang == 'en':
            return f"""
        You are now acting as: {persona}
        
        Please analyze the following Python code snippet (from file: {chunk['file_name']}):
        
        ```python
        {chunk['code']}
        ```
        {context_str}
        
        Task:
        1. Propose an in-depth technical question regarding the business logic or technical implementation of this code.
        2. Provide a detailed Reasoning Trace, explaining how you analyzed the code to arrive at the answer.
        3. Give the final answer.
        
        Please output strictly in JSON format, without Markdown markers, as follows:
        {{
            "instruction": "Your question...",
            "input": "The code snippet...",
            "output": {{
                "reasoning_trace": "Step-by-step reasoning process...",
                "answer": "The final answer..."
            }}
        }}
        """
        else: # cn
            return f"""
        你现在扮演一名：{persona}
        
        请分析以下 Python 代码片段（来自文件: {chunk['file_name']}）：
        
        ```python
        {chunk['code']}
        ```
        {context_str}
        
        任务：
        1. 针对这段代码的业务逻辑或技术实现，提出了一个有深度的技术问题。
        2. 提供详细的推理过程 (Reasoning Trace)，解释你是如何分析代码并得出答案的。
        3. 给出最终答案。
        
        请严格以 JSON 格式输出，不要包含 Markdown 标记，格式如下：
        {{
            "instruction": "你的问题...",
            "input": "原文代码片段...",
            "output": {{
                "reasoning_trace": "逐步推理过程...",
                "answer": "最终答案..."
            }}
        }}
        """

    def _build_arch_prompt(self, chunk: dict, persona: str, context_str: str, lang: str) -> str:
        if lang == 'en':
            return f"""
        You are now acting as: {persona}
        
        Based on the following existing code logic (from file: {chunk['file_name']}):
        
        ```python
        {chunk['code']}
        ```
        {context_str}
        
        Task:
        1. Assume a new business requirement has been proposed (e.g., add caching, switch to asynchronous processing, add audit logs, etc. Please devise a reasonable requirement yourself).
        2. Design an architectural modification plan based on the current code.
        3. Provide a detailed Reasoning Trace, explaining why this solution was chosen and how it aligns with DDD or high-quality code principles.
        
        Please output strictly in JSON format, without Markdown markers, as follows:
        {{
            "instruction": "Description of the new requirement...",
            "input": "The current code snippet...",
            "output": {{
                "reasoning_trace": "Reasoning process...",
                "architectural_modification_plan": "Detailed plan or code changes..."
            }}
        }}
        """
        else: # cn
            return f"""
        你现在扮演一名：{persona}
        
        基于以下现有的代码逻辑（来自文件: {chunk['file_name']}）：
        
        ```python
        {chunk['code']}
        ```
        {context_str}
        
        任务：
        1. 假设业务方提出了一个新的需求（例如：增加缓存、改为异步处理、增加审计日志等，请自行构思一个合理需求）。
        2. 设计一个基于当前代码的架构修改方案。
        3. 提供详细的推理过程 (Reasoning Trace)，解释为什么选择这个方案，以及它如何符合 DDD 或高质量代码原则。
        
        请严格以 JSON 格式输出，不要包含 Markdown 标记，格式如下：
        {{
            "instruction": "新需求描述...",
            "input": "当前代码片段...",
            "output": {{
                "reasoning_trace": "推理过程...",
                "architectural_modification_plan": "详细的设计方案或代码变更..."
            }}
        }}
        """

    def _review_and_refine(self, chunk: dict, raw_data: dict, lang: str) -> dict:
        """自我修正：让 LLM 充当 Critic 对生成结果进行打分和修正"""
        if not Config.CRITIC_ENABLED:
            return raw_data

        critic_prompt_cn = f"""
        你是一名严格的代码审查专家 (Critic)。请评估以下由 AI 生成的代码问答对的质量。

        【原始代码】:
        ```python
        {chunk['code']}
        ```

        【生成的问答】:
        Instruction: {raw_data.get('instruction')}
        Output: {raw_data.get('output')}

        任务：
        1. 评分 (0-10分)：评估问题的深度、推理的逻辑性以及代码的准确性。
        2. 审查：指出存在的问题（如逻辑错误、幻觉、格式不规范）。
        3. 修正：如果分数低于 10 分，请提供修正后的 Output 内容（保持 JSON 结构）。

        请严格以 JSON 格式输出：
        {{
            "score": 8,
            "critique": "推理过程略显简单...",
            "refined_output": "修正后的完整 Output 内容..."
        }}
        """
        
        critic_prompt_en = f"""
        You are a strict code review expert (Critic). Please evaluate the quality of the following AI-generated code Q&A pair.

        【Original Code】:
        ```python
        {chunk['code']}
        ```

        【Generated Q&A】:
        Instruction: {raw_data.get('instruction')}
        Output: {raw_data.get('output')}

        Task:
        1. Score (0-10): Evaluate the depth of the question, the logic of the reasoning, and the accuracy of the code.
        2. Critique: Point out any issues (e.g., logical errors, hallucinations, formatting issues).
        3. Refine: If the score is below 10, provide the refined 'output' content (maintaining the JSON structure).

        Please output strictly in JSON format:
        {{
            "score": 8,
            "critique": "The reasoning process is a bit too simple...",
            "refined_output": "The complete refined 'output' content..."
        }}
        """
        critic_prompt = critic_prompt_en if lang == 'en' else critic_prompt_cn
        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_NAME,
                contents=critic_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            review_result = json.loads(response.text)
            
            score = review_result.get("score", 0)
            if score < Config.CRITIC_THRESHOLD:
                print(f"    [Critic] 质量未达标 (分: {score})，已丢弃。原因: {review_result.get('critique')}")
                return None
            
            if review_result.get("refined_output"):
                print(f"    [Critic] 质量通过 (分: {score})，已应用修正建议。")
                raw_data["output"] = review_result["refined_output"]
            
            return raw_data
        except Exception as e:
            print(f"    [Critic] 审查过程出错: {e}，保留原始结果。")
            return raw_data

    # 优化指数退避策略：初始等待至少 10秒，每次失败等待时间翻倍 (multiplier=2)，最大等待 120秒
    @retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=2, min=10, max=120))
    def generate_sample(self, chunk: dict, scenario: str = "qa") -> dict:
        # 根据比例决定语言
        lang = 'en' if random.random() < Config.ENGLISH_RATIO else 'cn'

        persona = self._get_random_persona(lang=lang)
        context_str = self._build_context_str(chunk)
        
        if scenario == "qa":
            prompt = self._build_qa_prompt(chunk, persona, context_str, lang)
        else:
            prompt = self._build_arch_prompt(chunk, persona, context_str, lang)

        try:
            # 使用 Google 原生 SDK 生成内容
            # 我们将 System Instruction 拼接到 Prompt 前面，并启用 JSON 模式
            full_prompt = f"你是一个负责生成高质量代码微调数据的助手。请只输出 JSON。\n\n{prompt}"
            
            response = self.client.models.generate_content(
                model=Config.MODEL_NAME,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=self._get_random_temp(),
                    response_mime_type="application/json"
                )
            )
        except Exception as e:
            if "404" not in str(e) and "Not Found" not in str(e):
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"\n[Warning] 触发 Google 限流 (429)，正在等待重试... (详情: {str(e)[:50]}...)")
                raise e
            print(f"\n[Error] API 调用失败 (404): 找不到模型或 API Key 无效。")
            print(f"详细错误信息: {e}")
            print(f"请检查 .env 文件：\n1. GOOGLE_API_KEY 是否是真实的？(不要使用示例 Key)\n2. MODEL_NAME 是否支持？(当前: {Config.MODEL_NAME})")
            sys.exit(1) # 配置错误直接退出，避免空跑

        content = response.text
        try:
            raw_data = json.loads(content)
            
            # 执行自我修正 (Critic Loop)
            raw_data = self._review_and_refine(chunk, raw_data, lang)
            if raw_data is None:
                return None # 质量过低被丢弃
            
            # 转换数据格式为 Qwen 微调标准格式
            scenario_mapped = "business_logic" if scenario == "qa" else "architecture_design"
            user_content = self._extract_clean_instruction(raw_data)
            
            # 将代码原文和推理过程都放在 Assistant 回复中
            # 强制使用 chunk['code'] (原始代码段) 作为上下文，确保数据的准确性和合规性
            assistant_content_str = ""
            code_content = chunk.get('code', "")
            if code_content:
                assistant_content_str += f"相关代码:\n```python\n{code_content}\n```\n\n"
            
            assistant_content_str += self._format_assistant_content(raw_data.get("output"))

            new_data = {
                "id": str(uuid.uuid4()),
                "scenario": scenario_mapped,
                "code_ref": chunk["file_name"],
                "conversation": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content_str}
                ]
            }
            return new_data
        except json.JSONDecodeError:
            print(f"JSON 解析失败，跳过该条目。原始内容: {content[:100]}...")
            return None
